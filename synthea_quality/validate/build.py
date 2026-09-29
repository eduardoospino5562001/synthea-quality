"""[VALIDATE] Build the validation report of one module over one dataset.

The dataset is opened once (:func:`synthea_quality.dataset.open_dataset`), ``patients``
and ``conditions`` are read once with the union of the columns the three analyses need,
and the frames are handed to the existing computations:

* the population summary is :func:`synthea_quality.profile.demographics.profile_patients`,
  keeping its population, age and sex sections;
* prevalence is :mod:`synthea_quality.prevalence.compute` — among the patients alive at the
  end, at the reference date;
* incidence is :mod:`synthea_quality.incidence.compute` — every patient until death, over
  the ``window_years`` before the reference date;
* observation values, when the module file has ``observations``, are
  :func:`synthea_quality.observations.build.observation_results` — the latest value of
  each alive patient, or of each patient of a condition cohort;
* medications, when the module file has ``medications``, are
  :func:`synthea_quality.medications.compute.medication_results` — the share of the
  alive patients, or of a condition cohort, with each medication, active and ever.

Nothing is guessed. Without a reference date or the patients, everything is ``SKIPPED``
with the reason. Without the condition records, every condition's prevalence and
incidence, and every observation limited to a cohort, are; without a ``STOP`` column only
prevalence and the cohorts are (incidence does not need it). Without ``observations.csv``
only the observations are; without ``medications.csv`` only the medications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from synthea_quality.dataset import open_dataset
from synthea_quality.discovery import DiscoveryResult
from synthea_quality.incidence import compute as incidence
from synthea_quality.incidence.build import export_history
from synthea_quality.incidence.models import ConditionIncidence
from synthea_quality.prevalence import compute as prevalence
from synthea_quality.medications import compute as medications_compute
from synthea_quality.medications.definitions import MedicationDefinition
from synthea_quality.observations.build import (
    load_observations,
    observation_results,
    skipped_observation,
)
from synthea_quality.observations.compute import prepare_values
from synthea_quality.observations.definitions import ObservationDefinition
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
#: This command has no --alive-only; the note points to the one that does.
WHY_ALL_PATIENTS = (
    f"{incidence.SURVIVOR_BIAS} `synthea-incidence --alive-only` measures that cohort instead."
)
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
    observations: Sequence[ObservationDefinition] = (),
    medications: Sequence[MedicationDefinition] = (),
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
    tables = (
        *TABLES,
        *(["observations"] if observations else []),
        *(["medications"] if medications else []),
    )
    context = open_dataset(
        data_dir, tables=tables, reference_date=reference_date, metadata=metadata,
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
    observation_frame = None
    if observations:
        observation_frame, observations_input = load_observations(context)
        inputs.append(observations_input)
    medication_frame = None
    if medications:
        medication_frame, medications_input = context.load(
            "medications", medications_compute.MEDICATION_COLUMNS,
            medications_compute.REQUIRED_MEDICATION_COLUMNS, "medication records",
        )
        inputs.append(medications_input)

    history = export_history(metadata, conditions, reference, window_years)
    notes = [*resolution.notes, WHY_ALL_PATIENTS, *history.pop("notes")]
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
    if reason is not None:
        return ModuleValidationReport(
            alive=None,
            population=tuple(
                s for s in skipped_patient_sections(reason) if s.section_id in POPULATION_SECTIONS
            ),
            conditions=tuple(_skipped(d, reason) for d in definitions),
            observations=tuple(skipped_observation(d, reason) for d in observations),
            medications=tuple(
                medications_compute.skipped_medication(d, reason) for d in medications
            ),
            **common,
        )

    assert reference is not None and patients is not None
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
    followed = incidence.build_cohort(
        patients, reference, window_years=window_years, age_bands=bands
    )
    alive_records = None
    if conditions is None:
        records_reason = f"no condition records: {conditions_input.reason}"
        results = [_skipped(d, records_reason) for d in definitions]
    else:
        records_reason = "conditions.csv has no STOP column"
        if "STOP" in conditions.columns:
            alive_records = prevalence.prepare_records(conditions, alive_cohort, reference)
        followed_records = incidence.prepare_records(conditions, followed)
        results = []
        for definition in definitions:
            if alive_records is None:
                prevalence_result: ConditionResult = _skipped_prevalence(
                    definition,
                    f"no condition records usable for prevalence: {records_reason}",
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
                    incidence=incidence.condition_incidence(
                        definition, followed_records, followed
                    ),
                )
            )

    if not observations:
        observation_part: tuple = ()
    elif observation_frame is None:
        why = f"no observation values: {observations_input.reason}"
        observation_part = tuple(skipped_observation(d, why) for d in observations)
    else:
        observation_part = observation_results(
            observations,
            definitions,
            prepare_values(observation_frame, alive_cohort, reference),
            alive_cohort,
            reference,
            condition_records=alive_records,
            records_reason=records_reason,
        )
    if not medications:
        medication_part: tuple = ()
    elif medication_frame is None:
        why = f"no medication records: {medications_input.reason}"
        medication_part = tuple(
            medications_compute.skipped_medication(d, why) for d in medications
        )
    else:
        medication_part = medications_compute.medication_results(
            medications,
            definitions,
            medications_compute.prepare_medications(medication_frame, alive_cohort, reference),
            alive_cohort,
            condition_records=alive_records,
            records_reason=records_reason,
        )
    return ModuleValidationReport(
        alive=alive_cohort.size,
        population=population,
        conditions=tuple(results),
        observations=observation_part,
        medications=medication_part,
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
