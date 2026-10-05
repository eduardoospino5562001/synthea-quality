"""[INCIDENCE] Build the incidence report of one dataset directory.

The only module of the package that reads the dataset. It opens it with
:func:`synthea_quality.dataset.open_dataset`, as the prevalence does: discovery,
structural validation of ``patients``, ``encounters`` and ``conditions`` before loading
them, the reference date, and only the columns needed.

It also states what is known about the **exported history**, because incidence depends on
it. Synthea's CSV export keeps only the last ``exporter.years_of_history`` years (10 by
default; 0 keeps everything): a condition that ended before that horizon is not in the
files, so a prior case can look like a new one, and a window reaching past the horizon
has missing events. The setting is recorded in Synthea's run metadata file; with
``--metadata`` the report uses it, without it the report says the value is unknown and
gives the earliest condition record it saw.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from synthea_quality.dataset import open_dataset
from synthea_quality.discovery import DiscoveryResult
from synthea_quality.export_history import read_export_history, synthea_cutoff
from synthea_quality.incidence.compute import (
    CONDITION_COLUMNS,
    DEFAULT_WINDOW_YEARS,
    PATIENT_COLUMNS,
    POPULATION_ALIVE,
    POPULATION_ALL,
    REQUIRED_CONDITION_COLUMNS,
    REQUIRED_PATIENT_COLUMNS,
    WHY_ALL_PATIENTS,
    build_cohort,
    condition_incidence,
    prepare_records,
    window_start,
)
from synthea_quality.incidence.models import ConditionIncidence, IncidenceReport
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.profile.dates import parse_date_only
from synthea_quality.profile.models import SectionStatus
from synthea_quality.profile.population import DEFAULT_AGE_BANDS, validate_age_bands

INCIDENCE_TABLES = ("patients", "encounters", "conditions")
PATIENTS_USED_FOR = "the followed population: birth, death and sex"
CONDITIONS_USED_FOR = "condition records"
YEARS_OF_HISTORY_KEY = "exporter.years_of_history"
SYNTHEA_DEFAULT_YEARS_OF_HISTORY = 10


def build_incidence(
    data_dir: str | Path,
    *,
    definitions: Sequence[ConditionDefinition] = (),
    reference_date: str | None = None,
    metadata: str | Path | None = None,
    window_years: int = DEFAULT_WINDOW_YEARS,
    alive_only: bool = False,
    age_bands: Sequence[int] = DEFAULT_AGE_BANDS,
    discovery: DiscoveryResult | None = None,
    generated_at: str | None = None,
) -> IncidenceReport:
    """Incidence of ``definitions`` in ``data_dir``.

    :raises ValueError: invalid window, age bands, or both ``reference_date`` and ``metadata``.
    :raises EmptyDatasetError: the directory holds none of the known tables.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    """
    bands = validate_age_bands(age_bands)
    if window_years < 1:
        raise ValueError("the window must be at least one year")
    population = POPULATION_ALIVE if alive_only else POPULATION_ALL
    context = open_dataset(
        data_dir,
        tables=INCIDENCE_TABLES,
        reference_date=reference_date,
        metadata=metadata,
        discovery=discovery,
    )
    resolution = context.resolution
    patients, patients_input = context.load(
        "patients", PATIENT_COLUMNS, REQUIRED_PATIENT_COLUMNS, PATIENTS_USED_FOR
    )
    conditions, conditions_input = context.load(
        "conditions", CONDITION_COLUMNS, REQUIRED_CONDITION_COLUMNS, CONDITIONS_USED_FOR
    )
    inputs = [patients_input]
    if resolution.encounters is not None:
        inputs.append(resolution.encounters)
    inputs.append(conditions_input)

    reason = None
    if resolution.reference is None:
        reason = resolution.reason or "no reference date"
    elif patients is None:
        reason = f"the population is unknown: {patients_input.reason}"
    elif conditions is None:
        reason = f"no condition records: {conditions_input.reason}"

    reference = context.reference
    history = export_history(metadata, conditions, reference, window_years)
    exported = read_export_history(metadata, reference)
    notes = [*resolution.notes, population_note(population), *history.pop("notes")]
    common: dict[str, Any] = dict(
        data_dir=str(context.data_dir),
        reference_date=resolution.reference,
        reference_reason=resolution.reason,
        population=population,
        age_bands=bands,
        history=history,
        inputs=tuple(inputs),
        export_history=exported,
        **({"generated_at": generated_at} if generated_at is not None else {}),
    )
    if reason is not None:
        return IncidenceReport(
            window=None,
            conditions=tuple(
                ConditionIncidence(
                    d.name, d.codes, SectionStatus.SKIPPED, reason=reason, acute=d.acute
                )
                for d in definitions
            ),
            notes=tuple(notes),
            **common,
        )

    assert reference is not None and patients is not None and conditions is not None
    cohort = build_cohort(
        patients, reference, window_years=window_years, population=population, age_bands=bands
    )
    records = prepare_records(conditions, cohort)
    return IncidenceReport(
        window=cohort.window,
        conditions=tuple(condition_incidence(d, records, cohort) for d in definitions),
        cohort={"followed": cohort.size, "excluded": dict(cohort.excluded)},
        notes=tuple(notes),
        **common,
    )


def population_note(population: str) -> str:
    if population == POPULATION_ALL:
        return WHY_ALL_PATIENTS
    return (
        "Population restricted to the patients alive at the end of the simulation "
        "(--alive-only): the time and events of those who died during the window are left "
        "out, which biases the rate down for conditions that shorten life (survivor bias)."
    )


def export_history(
    metadata: str | Path | None,
    conditions: pd.DataFrame | None,
    reference: date | None,
    window_years: int,
) -> dict[str, Any]:
    """What is known about the exported history, and the notes it calls for."""
    exported = read_export_history(metadata, reference)
    years = exported.years
    earliest = None
    if conditions is not None and len(conditions):
        starts = parse_date_only(conditions["START"]).values.dropna()
        earliest = starts.min().date().isoformat() if len(starts) else None
    history: dict[str, Any] = {
        "years_of_history": years,
        "source": "synthea_metadata" if years is not None else None,
        "earliest_condition_start": earliest,
    }
    notes: list[str] = []
    if reference is None:
        history["notes"] = notes
        return history
    start = window_start(reference, window_years)
    if years is None:
        if metadata is None:
            why = f"{YEARS_OF_HISTORY_KEY} is only in Synthea's run metadata; use --metadata"
        else:
            why = (
                f"the metadata file `{exported.metadata_file}` has no usable "
                f"{YEARS_OF_HISTORY_KEY}"
            )
        notes.append(
            f"The exported history is unknown ({why}). Synthea exports "
            f"{SYNTHEA_DEFAULT_YEARS_OF_HISTORY} "
            f"years by default, and a condition that ended before then is not in the files, "
            f"so a prior case could be counted as new. The earliest condition record here "
            f"starts on {earliest or 'no date'}."
        )
    elif years == 0:
        notes.append(f"{YEARS_OF_HISTORY_KEY} = 0: the whole history was exported.")
    else:
        horizon = synthea_cutoff(reference, years)
        if start < horizon:
            notes.append(
                f"The window starts on {start.isoformat()}, before the exported history "
                f"({YEARS_OF_HISTORY_KEY} = {years}, from {horizon.isoformat()}): events before "
                f"that date are not in the files, so the rate is underestimated."
            )
        else:
            notes.append(
                f"{YEARS_OF_HISTORY_KEY} = {years}: prior cases are seen over the "
                f"{years - window_years} exported year(s) before the window; a condition that "
                f"ended before {horizon.isoformat()} is not in the files, so such a prior case "
                f"would be counted as new."
            )
    history["notes"] = notes
    return history

