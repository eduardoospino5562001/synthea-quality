"""[PREVALENCE] Build the prevalence report of one dataset directory.

The only module of the package that touches the file system, following the profile's
builder step by step: discovery, structural validation of the tables it reads
(``patients``, ``encounters``, ``conditions``) before any of them is loaded, the shared
reference-date resolution, and a loader that reads only the columns needed.

Nothing is guessed. Every rate needs the alive patients and the reference date, so
without either every result is ``SKIPPED`` with the reason; without ``conditions.csv``
(or its ``PATIENT``, ``CODE``, ``START`` or ``STOP`` column) every result is ``SKIPPED``
too. A table that is present but unreadable marks the report incomplete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import EmptyDatasetError, TableLoadError
from synthea_quality.loader import DatasetLoader
from synthea_quality.prevalence.compute import (
    CONDITION_COLUMNS,
    DEFAULT_TOP,
    PATIENT_COLUMNS,
    REQUIRED_CONDITION_COLUMNS,
    build_cohort,
    condition_prevalence,
    general_table,
    prepare_records,
)
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.prevalence.models import (
    ConditionResult,
    GeneralTable,
    PrevalenceReport,
)
from synthea_quality.prevalence.social import describe_list
from synthea_quality.profile.models import InputState, SectionStatus, TableInput
from synthea_quality.profile.population import DEFAULT_AGE_BANDS, validate_age_bands
from synthea_quality.profile.reference import parse_reference_date, resolve_reference_date
from synthea_quality.schema.tables import SYNTHEA_TABLES
from synthea_quality.structure import StructureReport, gate_reason, validate_tables

PREVALENCE_TABLES = ("patients", "encounters", "conditions")
PATIENTS_USED_FOR = "the alive cohort, ages and sex"
CONDITIONS_USED_FOR = "condition records"


def build_prevalence(
    data_dir: str | Path,
    *,
    definitions: Sequence[ConditionDefinition] = (),
    reference_date: str | None = None,
    metadata: str | Path | None = None,
    age_bands: Sequence[int] = DEFAULT_AGE_BANDS,
    include_social: bool = False,
    top: int = DEFAULT_TOP,
    discovery: DiscoveryResult | None = None,
    generated_at: str | None = None,
) -> PrevalenceReport:
    """Compute the prevalence of ``definitions`` and the general table for ``data_dir``.

    :raises ValueError: invalid age bands or ``top``, or both ``reference_date`` and
        ``metadata``.
    :raises EmptyDatasetError: the directory holds none of the known tables.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    """
    bands = validate_age_bands(age_bands)
    if top < 1:
        raise ValueError("top must be at least 1")
    data_path = Path(data_dir)
    found = discovery if discovery is not None else discover_dataset(data_path)
    if not found.tables:
        raise EmptyDatasetError(
            f"no Synthea CSV table was found in {data_path}: a dataset directory has to "
            f"hold at least one of the {len(SYNTHEA_TABLES)} tables the schema contract "
            f"describes (for example patients.csv or conditions.csv)"
        )

    loader = DatasetLoader()
    readable: dict[str, Path] = {}
    for table in found.tables:
        if table.name in PREVALENCE_TABLES:
            try:
                loader.read_header(table.path)
            except TableLoadError:
                continue
            readable[table.name] = table.path
    structure = validate_tables(readable)

    resolution = resolve_reference_date(
        found, loader, user_date=reference_date, metadata_path=metadata, structure=structure
    )
    patients, patients_input = _load(
        found, loader, structure, "patients", PATIENT_COLUMNS, ("Id", "DEATHDATE"),
        PATIENTS_USED_FOR,
    )
    conditions, conditions_input = _load(
        found, loader, structure, "conditions", CONDITION_COLUMNS, REQUIRED_CONDITION_COLUMNS,
        CONDITIONS_USED_FOR,
    )
    inputs = [patients_input]
    if resolution.encounters is not None:
        inputs.append(resolution.encounters)
    inputs.append(conditions_input)

    common = dict(
        data_dir=str(data_path),
        reference_date=resolution.reference,
        reference_reason=resolution.reason,
        age_bands=bands,
        social_list=describe_list(),
        inputs=tuple(inputs),
        **({"generated_at": generated_at} if generated_at is not None else {}),
    )

    reason = None
    if resolution.reference is None:
        reason = resolution.reason or "no reference date"
    elif patients is None:
        reason = f"the alive patients are unknown: {patients_input.reason}"
    elif conditions is None:
        reason = f"no condition records: {conditions_input.reason}"
    if reason is not None:
        return PrevalenceReport(
            alive=None,
            conditions=tuple(
                ConditionResult(d.name, d.codes, SectionStatus.SKIPPED, reason=reason)
                for d in definitions
            ),
            general=GeneralTable(
                SectionStatus.SKIPPED, reason=reason, top=top, include_social=include_social
            ),
            notes=resolution.notes,
            **common,
        )

    reference = parse_reference_date(resolution.reference.value)
    cohort = build_cohort(patients, reference, bands)
    records = prepare_records(conditions, cohort, reference)
    return PrevalenceReport(
        alive=cohort.size,
        conditions=tuple(condition_prevalence(d, records, cohort) for d in definitions),
        general=general_table(records, cohort, include_social=include_social, top=top),
        notes=(*resolution.notes, *cohort.notes),
        **common,
    )


def _load(
    found: DiscoveryResult,
    loader: DatasetLoader,
    structure: Mapping[str, StructureReport],
    name: str,
    wanted: Sequence[str],
    required: Sequence[str],
    used_for: str,
) -> tuple[pd.DataFrame | None, TableInput]:
    """The ``wanted`` columns of table ``name`` that exist, or ``None`` and why not."""
    table = next((t for t in found.tables if t.name == name), None)
    if table is None:
        reason = f"{name}.csv is not in the dataset"
        return None, TableInput(name, used_for, InputState.ABSENT, reason=reason)
    gated = gate_reason(structure, name)
    if gated is not None:
        return None, TableInput(name, used_for, InputState.UNREADABLE, reason=gated)
    try:
        header = loader.read_header(table.path)
        missing = [c for c in required if c not in header]
        if missing:
            reason = f"{name}.csv has no {', '.join(missing)} column(s)"
            return None, TableInput(name, used_for, InputState.ABSENT, reason=reason)
        loaded = loader.load(
            table.path, table=name, columns=[c for c in wanted if c in header]
        )
    except TableLoadError as exc:
        reason = f"{name}.csv could not be read: {exc}"
        return None, TableInput(name, used_for, InputState.UNREADABLE, reason=reason)
    return loaded.frame, TableInput(name, used_for, InputState.READ, rows=loaded.rows)
