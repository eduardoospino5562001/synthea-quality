"""[VALIDATE] Build the validation report of one module over one dataset.

The dataset is opened once (:func:`synthea_quality.dataset.open_dataset`), ``patients``
and ``conditions`` are read once with the union of the columns the three analyses need,
and the frames are handed to the existing computations:

* the population summary is :func:`synthea_quality.profile.demographics.profile_patients`,
  keeping its population, age and sex sections;
* prevalence is :mod:`synthea_quality.prevalence.compute` — among the patients alive at the
  end, at the reference date;
* incidence is :mod:`synthea_quality.incidence.compute` — every patient until death, over
  the ``window_years`` before the reference date.

Nothing is guessed. Without a reference date, the patients or the condition records,
every condition's prevalence and incidence are ``SKIPPED`` with the reason; without a
``STOP`` column only prevalence is (incidence does not need it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from synthea_quality.dataset import open_dataset
from synthea_quality.discovery import DiscoveryResult
from synthea_quality.incidence import compute as incidence
from synthea_quality.incidence.build import export_history, population_note
from synthea_quality.incidence.models import ConditionIncidence
from synthea_quality.prevalence import compute as prevalence
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.prevalence.models import ConditionResult
from synthea_quality.profile.demographics import PATIENT_COLUMNS as PROFILE_COLUMNS
from synthea_quality.profile.demographics import profile_patients, skipped_patient_sections
from synthea_quality.profile.models import SectionStatus
from synthea_quality.profile.population import DEFAULT_AGE_BANDS, validate_age_bands
from synthea_quality.validate.models import (
    ConditionValidation,
    ModuleInfo,
    ModuleValidationReport,
)

TABLES = ("patients", "encounters", "conditions")
#: The population summary keeps these sections of the profile.
POPULATION_SECTIONS = ("population", "age", "distribution.GENDER")

PATIENT_COLUMNS = tuple(dict.fromkeys(("Id", *PROFILE_COLUMNS, *incidence.PATIENT_COLUMNS)))
CONDITION_COLUMNS = tuple(
    dict.fromkeys((*prevalence.CONDITION_COLUMNS, *incidence.CONDITION_COLUMNS))
)


def build_module_validation(
    data_dir: str | Path,
    *,
    module: ModuleInfo,
    definitions: Sequence[ConditionDefinition],
    module_file: str | Path,
    reference_date: str | None = None,
    metadata: str | Path | None = None,
    window_years: int = incidence.DEFAULT_WINDOW_YEARS,
    age_bands: Sequence[int] = DEFAULT_AGE_BANDS,
    discovery: DiscoveryResult | None = None,
    generated_at: str | None = None,
) -> ModuleValidationReport:
    """Validate ``definitions`` of ``module`` against the dataset in ``data_dir``.

    :raises ValueError: invalid window or age bands, or both ``reference_date`` and
        ``metadata``.
    :raises EmptyDatasetError: the directory holds none of the known tables.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    """
    bands = validate_age_bands(age_bands)
    if window_years < 1:
        raise ValueError("the window must be at least one year")
    context = open_dataset(
        data_dir, tables=TABLES, reference_date=reference_date, metadata=metadata,
        discovery=discovery,
    )
    resolution = context.resolution
    reference = context.reference
    patients, patients_input = context.load(
        "patients", PATIENT_COLUMNS, incidence.REQUIRED_PATIENT_COLUMNS,
        "the population, the alive cohort and the followed population",
    )
    conditions, conditions_input = context.load(
        "conditions", CONDITION_COLUMNS, incidence.REQUIRED_CONDITION_COLUMNS,
        "condition records",
    )
    inputs = [patients_input]
    if resolution.encounters is not None:
        inputs.append(resolution.encounters)
    inputs.append(conditions_input)

    history = export_history(metadata, conditions, reference, window_years)
    notes = [*resolution.notes, population_note(incidence.POPULATION_ALL), *history.pop("notes")]
    common = dict(
        data_dir=str(context.data_dir),
        module_file=str(module_file),
        module=module,
        reference_date=resolution.reference,
        reference_reason=resolution.reason,
        age_bands=bands,
        history=history,
        inputs=tuple(inputs),
        notes=tuple(notes),
        **({"generated_at": generated_at} if generated_at is not None else {}),
    )

    reason = None
    if reference is None:
        reason = resolution.reason or "no reference date"
    elif patients is None:
        reason = f"the population is unknown: {patients_input.reason}"
    elif conditions is None:
        reason = f"no condition records: {conditions_input.reason}"
    if reason is not None:
        return ModuleValidationReport(
            alive=None,
            population=tuple(
                s for s in skipped_patient_sections(reason) if s.section_id in POPULATION_SECTIONS
            ),
            conditions=tuple(_skipped(d, reason) for d in definitions),
            **common,
        )

    assert reference is not None and patients is not None and conditions is not None
    population = tuple(
        section
        for section in profile_patients(
            patients[[c for c in PROFILE_COLUMNS if c in patients.columns]],
            reference=reference,
            reference_reason=resolution.reason,
            age_bands=bands,
        )
        if section.section_id in POPULATION_SECTIONS
    )

    alive_cohort = prevalence.build_cohort(patients, reference, bands)
    no_stop = "STOP" not in conditions.columns
    alive_records = None if no_stop else prevalence.prepare_records(
        conditions, alive_cohort, reference
    )
    followed = incidence.build_cohort(
        patients, reference, window_years=window_years, age_bands=bands
    )
    followed_records = incidence.prepare_records(conditions, followed)

    results = []
    for definition in definitions:
        if alive_records is None:
            prevalence_result: ConditionResult = _skipped_prevalence(
                definition,
                "no condition records usable for prevalence: conditions.csv has no STOP column",
            )
        else:
            prevalence_result = prevalence.condition_prevalence(
                definition, alive_records, alive_cohort
            )
        results.append(
            ConditionValidation(
                name=definition.name,
                codes=definition.codes,
                acute=definition.acute,
                prevalence=prevalence_result,
                incidence=incidence.condition_incidence(definition, followed_records, followed),
            )
        )
    return ModuleValidationReport(
        alive=alive_cohort.size,
        population=population,
        conditions=tuple(results),
        incidence_window=followed.window,
        incidence_population=followed.population,
        cohort={"followed": followed.size, "excluded": dict(followed.excluded)},
        **common,
    )


def _skipped_prevalence(definition: ConditionDefinition, reason: str) -> ConditionResult:
    return ConditionResult(
        definition.name, definition.codes, SectionStatus.SKIPPED, reason=reason,
        acute=definition.acute,
    )


def _skipped(definition: ConditionDefinition, reason: str) -> ConditionValidation:
    return ConditionValidation(
        name=definition.name,
        codes=definition.codes,
        acute=definition.acute,
        prevalence=_skipped_prevalence(definition, reason),
        incidence=ConditionIncidence(
            definition.name, definition.codes, SectionStatus.SKIPPED,
            reason=reason, acute=definition.acute,
        ),
    )
