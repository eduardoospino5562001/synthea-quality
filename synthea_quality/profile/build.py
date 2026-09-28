"""[PROFILE] Build the profile of one dataset directory.

This is the only module of the profile that touches the file system. It reuses the
pieces the quality checks already trust, in the same order:

1. :func:`~synthea_quality.discovery.discover_dataset` finds the tables;
2. :func:`~synthea_quality.structure.validate_tables` checks, for the two tables the
   profile reads (``patients`` and ``encounters``) whose header can be read, that every
   row has as many fields as the header. A table that fails is not read: the loader
   would pad a short row with nulls and the profile would count invented empty values;
3. one :class:`~synthea_quality.loader.DatasetLoader` reads only the columns needed;
4. :mod:`~synthea_quality.profile.reference` resolves the reference date and
   :mod:`~synthea_quality.profile.demographics` computes the sections.

A directory with no known table at all is an input error, as it is for
``synthea-quality``: a profile of nothing would be a page of skipped sections. A
missing ``patients.csv`` is not: the sections are skipped with the reason, and the
profile still records what was found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import EmptyDatasetError, TableLoadError
from synthea_quality.loader import DatasetLoader
from synthea_quality.profile.demographics import (
    PATIENT_COLUMNS,
    profile_patients,
    skipped_patient_sections,
)
from synthea_quality.profile.models import (
    DatasetProfile,
    InputState,
    ProfileSection,
    TableInput,
)
from synthea_quality.profile.population import DEFAULT_AGE_BANDS, validate_age_bands
from synthea_quality.profile.reference import (
    parse_reference_date,
    resolve_reference_date,
)
from synthea_quality.schema.tables import SYNTHEA_TABLES
from synthea_quality.structure import StructureReport, gate_reason, validate_tables

#: Tables the profile reads.
PROFILE_TABLES = ("patients", "encounters")
PATIENTS_USED_FOR = "population and demographic sections"


def build_profile(
    data_dir: str | Path,
    *,
    reference_date: str | None = None,
    metadata: str | Path | None = None,
    age_bands: Sequence[int] = DEFAULT_AGE_BANDS,
    top_counties: int | None = None,
    discovery: DiscoveryResult | None = None,
    generated_at: str | None = None,
) -> DatasetProfile:
    """Profile the Synthea CSV dataset in ``data_dir``.

    :param reference_date: ``YYYY-MM-DD`` to measure ages at, instead of resolving it.
    :param metadata: a Synthea metadata JSON file whose ``endTime`` is the reference date.
    :param age_bands: lower bounds of the age bands.
    :param top_counties: how many counties to list (10 by default).
    :param generated_at: timestamp to record; the current UTC time by default.
    :raises ValueError: invalid age bands, or both ``reference_date`` and ``metadata``.
    :raises EmptyDatasetError: the directory holds none of the known tables.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    """
    bands = validate_age_bands(age_bands)
    if top_counties is not None and top_counties < 1:
        raise ValueError("top_counties must be at least 1")
    data_path = Path(data_dir)
    found = discovery if discovery is not None else discover_dataset(data_path)
    if not found.tables:
        raise EmptyDatasetError(
            f"no Synthea CSV table was found in {data_path}: a dataset directory has to "
            f"hold at least one of the {len(SYNTHEA_TABLES)} tables the schema contract "
            f"describes (for example patients.csv or encounters.csv)"
        )

    loader = DatasetLoader()
    # As in reporting.build: only a table whose header can be read goes through the
    # structural validation. The others fail again, with the loader's message, when they
    # are loaded, and are recorded as unreadable there.
    readable: dict[str, Path] = {}
    for table in found.tables:
        if table.name in PROFILE_TABLES:
            try:
                loader.read_header(table.path)
            except TableLoadError:
                continue
            readable[table.name] = table.path
    structure = validate_tables(readable)

    resolution = resolve_reference_date(
        found,
        loader,
        user_date=reference_date,
        metadata_path=metadata,
        structure=structure,
    )
    reference = (
        parse_reference_date(resolution.reference.value)
        if resolution.reference is not None
        else None
    )

    patients_frame, patients_input = _load_patients(found, loader, structure)
    if patients_frame is None:
        assert patients_input.reason is not None
        sections: tuple[ProfileSection, ...] = skipped_patient_sections(patients_input.reason)
    else:
        sections = profile_patients(
            patients_frame,
            reference=reference,
            reference_reason=resolution.reason,
            age_bands=bands,
            top_counties=top_counties,
        )

    inputs = [patients_input]
    if resolution.encounters is not None:
        inputs.append(resolution.encounters)

    extra: dict[str, str] = {}
    if generated_at is not None:
        extra["generated_at"] = generated_at
    return DatasetProfile(
        data_dir=str(data_path),
        reference_date=resolution.reference,
        reference_reason=resolution.reason,
        age_bands=bands,
        sections=sections,
        inputs=tuple(inputs),
        notes=resolution.notes,
        **extra,
    )


def _load_patients(
    found: DiscoveryResult,
    loader: DatasetLoader,
    structure: Mapping[str, StructureReport],
) -> tuple[pd.DataFrame | None, TableInput]:
    """The profiled columns of ``patients.csv``, or ``None`` and why not."""
    table = next((t for t in found.tables if t.name == "patients"), None)
    if table is None:
        return None, TableInput(
            "patients",
            PATIENTS_USED_FOR,
            InputState.ABSENT,
            reason="patients.csv is not in the dataset",
        )
    gated = gate_reason(structure, "patients")
    if gated is not None:
        return None, TableInput(
            "patients", PATIENTS_USED_FOR, InputState.UNREADABLE, reason=gated
        )
    try:
        header = loader.read_header(table.path)
        columns = [c for c in PATIENT_COLUMNS if c in header]
        # With none of the profiled columns, still read one so the rows are counted.
        loaded = loader.load(table.path, table="patients", columns=columns or [header[0]])
    except TableLoadError as exc:
        return None, TableInput(
            "patients",
            PATIENTS_USED_FOR,
            InputState.UNREADABLE,
            reason=f"patients.csv could not be read: {exc}",
        )
    frame = loaded.frame if columns else loaded.frame[[]]
    return frame, TableInput(
        "patients", PATIENTS_USED_FOR, InputState.READ, rows=loaded.rows
    )
