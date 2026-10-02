"""[OBSERVATIONS] Build the observation values report of one dataset directory.

The only module of the package that touches the file system. It opens the dataset with
:func:`synthea_quality.dataset.open_dataset` — discovery, structural validation of the
tables it reads before any is loaded, the reference date — and loads only the columns
needed. ``conditions`` is read only when an observation is limited to a cohort.

Nothing is guessed. Without a reference date or the alive patients every result is
``SKIPPED`` with the reason; without ``observations.csv`` (or one of its required
columns) too. An observation whose cohort cannot be selected (no ``conditions.csv``) is
``SKIPPED`` on its own. A table that is present but unreadable marks the report
incomplete.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from synthea_quality.condition_cohort import select_cohort
from synthea_quality.dataset import DatasetContext, open_dataset
from synthea_quality.discovery import DiscoveryResult
from synthea_quality.export_history import read_export_history
from synthea_quality.observations.compute import (
    DEFAULT_TOP,
    OBSERVATION_COLUMNS,
    REQUIRED_OBSERVATION_COLUMNS,
    Values,
    general_table,
    observation_values,
    prepare_values,
)
from synthea_quality.observations.definitions import ObservationDefinition
from synthea_quality.observations.models import (
    GeneralTable,
    ObservationResult,
    ObservationsReport,
)
from synthea_quality.prevalence import compute as prevalence
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.profile.models import SectionStatus, TableInput
from synthea_quality.profile.population import DEFAULT_AGE_BANDS, validate_age_bands

OBSERVATIONS_USED_FOR = "observation values"
CONDITIONS_USED_FOR = "the condition cohorts"


def build_observations(
    data_dir: str | Path,
    *,
    observations: Sequence[ObservationDefinition] = (),
    conditions: Sequence[ConditionDefinition] = (),
    module_file: str | Path | None = None,
    reference_date: str | None = None,
    metadata: str | Path | None = None,
    age_bands: Sequence[int] = DEFAULT_AGE_BANDS,
    lookback_years: int | None = None,
    top: int = DEFAULT_TOP,
    discovery: DiscoveryResult | None = None,
    generated_at: str | None = None,
) -> ObservationsReport:
    """Describe ``observations`` and every numeric code of the dataset in ``data_dir``.

    :param lookback_years: applies to the general table and to every observation that
        does not set its own.
    :raises ValueError: invalid age bands, ``top`` or ``lookback_years``, or both
        ``reference_date`` and ``metadata``.
    :raises EmptyDatasetError: the directory holds none of the known tables.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    """
    bands = validate_age_bands(age_bands)
    if top < 1:
        raise ValueError("top must be at least 1")
    if lookback_years is not None and lookback_years < 1:
        raise ValueError("the lookback must be at least one year")
    if lookback_years is not None:
        observations = [
            d if d.lookback_years is not None else replace(d, lookback_years=lookback_years)
            for d in observations
        ]
    with_cohort = any(d.cohort is not None for d in observations)
    tables = ("patients", "encounters", "observations", *(["conditions"] if with_cohort else []))
    context = open_dataset(
        data_dir, tables=tables, reference_date=reference_date, metadata=metadata,
        discovery=discovery,
    )
    resolution = context.resolution
    patients, patients_input = context.load(
        "patients", prevalence.PATIENT_COLUMNS, ("Id", "DEATHDATE"),
        "the alive patients, ages and sex",
    )
    frame, observations_input = load_observations(context)
    inputs = [patients_input]
    if resolution.encounters is not None:
        inputs.append(resolution.encounters)
    inputs.append(observations_input)
    condition_frame = condition_records = records_reason = None
    reference = context.reference
    if with_cohort:
        condition_frame, conditions_input = context.load(
            "conditions", prevalence.CONDITION_COLUMNS, prevalence.REQUIRED_CONDITION_COLUMNS,
            CONDITIONS_USED_FOR,
        )
        inputs.append(conditions_input)
        records_reason = conditions_input.reason
    common = dict(
        data_dir=str(context.data_dir),
        module_file=str(module_file) if module_file is not None else None,
        reference_date=resolution.reference,
        reference_reason=resolution.reason,
        age_bands=bands,
        lookback_years=lookback_years,
        inputs=tuple(inputs),
        export_history=read_export_history(metadata, reference),
        **({"generated_at": generated_at} if generated_at is not None else {}),
    )

    reason = None
    if reference is None:
        reason = resolution.reason or "no reference date"
    elif patients is None:
        reason = f"the alive patients are unknown: {patients_input.reason}"
    elif frame is None:
        reason = f"no observation values: {observations_input.reason}"
    if reason is not None:
        return ObservationsReport(
            alive=None,
            observations=tuple(skipped_observation(d, reason) for d in observations),
            general=GeneralTable(SectionStatus.SKIPPED, reason=reason, top=top),
            notes=resolution.notes,
            **common,
        )

    assert reference is not None and patients is not None and frame is not None
    population = prevalence.build_cohort(patients, reference, bands)
    if condition_frame is not None:
        condition_records = prevalence.prepare_records(condition_frame, population, reference)
    values = prepare_values(frame, population, reference)
    return ObservationsReport(
        alive=population.size,
        observations=observation_results(
            observations, conditions, values, population, reference,
            condition_records=condition_records, records_reason=records_reason,
        ),
        general=general_table(
            values, population, reference, lookback_years=lookback_years, top=top
        ),
        notes=(*resolution.notes, *population.notes),
        **common,
    )


def load_observations(context: DatasetContext) -> tuple[pd.DataFrame | None, TableInput]:
    """The columns of ``observations.csv`` this package uses, and what happened to the table."""
    return context.load(
        "observations", OBSERVATION_COLUMNS, REQUIRED_OBSERVATION_COLUMNS, OBSERVATIONS_USED_FOR
    )


def observation_results(
    observations: Sequence[ObservationDefinition],
    conditions: Sequence[ConditionDefinition],
    values: Values,
    population: prevalence.Cohort,
    reference: date,
    *,
    condition_records: prevalence.Records | None = None,
    records_reason: str | None = None,
) -> tuple[ObservationResult, ...]:
    """Every observation asked for, each among the alive or among its cohort.

    Shared with ``synthea-validate-module``, so both describe an observation the same way.
    """
    by_name: Mapping[str, ConditionDefinition] = {c.name: c for c in conditions}
    results = []
    for definition in observations:
        members = None
        if definition.cohort is not None:
            members, reason = select_cohort(
                definition.cohort, by_name, condition_records, records_reason
            )
            if members is None:
                results.append(skipped_observation(definition, reason or "no cohort"))
                continue
        results.append(
            observation_values(definition, values, population, reference, members=members)
        )
    return tuple(results)


def skipped_observation(definition: ObservationDefinition, reason: str) -> ObservationResult:
    return ObservationResult(
        name=definition.key,
        code=definition.code,
        description=None,
        status=SectionStatus.SKIPPED,
        reason=reason,
        cohort=definition.cohort,
        lookback_years=definition.lookback_years,
        reference_range=definition.reference_range,
    )
