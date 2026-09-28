"""[PROFILE] The reference date: when the simulation ended, and how we know.

Who is alive and how old they are both depend on a date the CSV export does not
contain. What Synthea's source says about it (read at commit ``44f645c``):

* ``Generator`` simulates every person up to ``stop``, set from
  ``GeneratorOptions.endTime`` (``Generator.java``). ``endTime`` defaults to
  ``System.currentTimeMillis()`` — the moment Synthea was run — and the ``-e YYYYMMDD``
  command-line switch overrides it (``App.java``);
* ``MetadataExporter`` writes that value, formatted ``yyyyMMdd`` in UTC, as ``endTime``
  in ``output/metadata/<run>.json``;
* ``CSVExporter`` never writes it, and the official sample archive ships only the CSV
  files, without the metadata directory.

So there is exactly one confirmed source, and it is usually absent. This module
resolves the date in a fixed order and records where it came from:

1. a date the user gives explicitly (``ReferenceSource.USER``);
2. ``endTime`` from a Synthea metadata file the user points at
   (``ReferenceSource.SYNTHEA_METADATA``) — the confirmed source;
3. otherwise the latest ``START``/``STOP`` timestamp in ``encounters.csv``
   (``ReferenceSource.MAX_ENCOUNTER_DATE``), always marked ``approximate``. Every
   exported encounter happened before the simulation stopped, so this is normally a
   lower bound of the real end date, and on the official sample it falls one day before
   the archive was published. It is an estimate and the report says so.

When none of them is available the reference date is ``None`` with a reason, and the
sections that need it are ``SKIPPED``; nothing else is guessed.

A metadata ``endTime`` earlier than the latest encounter is contradictory (the
metadata may belong to another run). It is not an error — the user chose the source —
but the resolution carries a note so the report shows the inconsistency. A date the
user types is taken as it is: it may legitimately be any date an analysis is anchored
at, and nothing is read to second-guess it.

This module depends on neither the renderers nor the command line, so later analyses
(prevalence, incidence) can reuse it as it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

import pandas as pd

from synthea_quality.discovery import DiscoveryResult
from synthea_quality.errors import SyntheaQualityError, TableLoadError
from synthea_quality.loader import DatasetLoader
from synthea_quality.profile.dates import parse_timestamps
from synthea_quality.profile.models import (
    InputState,
    ReferenceDate,
    ReferenceSource,
    TableInput,
)
from synthea_quality.structure import StructureReport, gate_reason

#: Key and format of the end of the simulation in a Synthea metadata file.
METADATA_END_TIME_KEY = "endTime"
METADATA_END_TIME_FORMAT = "%Y%m%d"

#: Where the end of the simulation is defined and recorded in Synthea's source.
SYNTHEA_END_TIME_PROVENANCE = (
    "Synthea simulates up to Generator.stop = GeneratorOptions.endTime (default: the time "
    "Synthea ran; '-e YYYYMMDD' overrides it), and MetadataExporter writes it as endTime"
)

#: The encounter columns scanned for the approximation.
ENCOUNTER_DATE_COLUMNS = ("START", "STOP")


class ReferenceDateError(SyntheaQualityError):
    """An explicitly requested reference date cannot be used (bad value or file)."""


def parse_reference_date(text: str) -> date:
    """Parse ``YYYY-MM-DD`` strictly, or raise :class:`ValueError`."""
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"not a YYYY-MM-DD date: {text!r}") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"not a YYYY-MM-DD date: {text!r}")
    return parsed


def reference_from_user(text: str) -> ReferenceDate:
    """A reference date given explicitly, e.g. on the command line."""
    try:
        value = parse_reference_date(text)
    except ValueError as exc:
        raise ReferenceDateError(str(exc)) from exc
    return ReferenceDate(
        value=value.isoformat(),
        source=ReferenceSource.USER,
        detail="given explicitly by the user (--reference-date)",
        approximate=False,
    )


def reference_from_metadata(path: str | Path) -> ReferenceDate:
    """The ``endTime`` recorded in a Synthea metadata JSON file.

    :raises ReferenceDateError: the file cannot be read, is not a JSON object, or has no
        valid ``endTime``.
    """
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ReferenceDateError(f"metadata file {file_path} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReferenceDateError(f"metadata file {file_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReferenceDateError(f"metadata file {file_path} is not a JSON object")
    raw = data.get(METADATA_END_TIME_KEY)
    if not isinstance(raw, str):
        raise ReferenceDateError(
            f"metadata file {file_path} has no string '{METADATA_END_TIME_KEY}' "
            f"(expected yyyyMMdd, as MetadataExporter writes it)"
        )
    try:
        value = datetime.strptime(raw, METADATA_END_TIME_FORMAT).date()
    except ValueError as exc:
        raise ReferenceDateError(
            f"metadata file {file_path} has {METADATA_END_TIME_KEY}={raw!r}, not yyyyMMdd"
        ) from exc
    if value.strftime(METADATA_END_TIME_FORMAT) != raw:
        raise ReferenceDateError(
            f"metadata file {file_path} has {METADATA_END_TIME_KEY}={raw!r}, not yyyyMMdd"
        )
    return ReferenceDate(
        value=value.isoformat(),
        source=ReferenceSource.SYNTHEA_METADATA,
        detail=(
            f"{METADATA_END_TIME_KEY}={raw} read from {file_path}. {SYNTHEA_END_TIME_PROVENANCE}"
        ),
        approximate=False,
    )


@dataclass(frozen=True, slots=True)
class LatestEncounter:
    """The latest encounter timestamp found in ``encounters.csv``."""

    #: ``None`` when no value parsed.
    timestamp: datetime | None
    #: Values that parsed, across the scanned columns.
    values: int
    #: Values present but unparseable, ignored and counted.
    unparseable: int
    columns: tuple[str, ...]

    @property
    def date(self) -> date | None:
        return self.timestamp.date() if self.timestamp is not None else None


def latest_encounter(frame: pd.DataFrame) -> LatestEncounter:
    """Scan whichever of ``START``/``STOP`` ``frame`` holds for the latest timestamp."""
    columns = tuple(c for c in ENCOUNTER_DATE_COLUMNS if c in frame.columns)
    latest: pd.Timestamp | None = None
    values = unparseable = 0
    for column in columns:
        parsed = parse_timestamps(frame[column])
        values += parsed.valid
        unparseable += parsed.unparseable
        if parsed.valid:
            candidate = parsed.values.max()
            if latest is None or candidate > latest:
                latest = candidate
    return LatestEncounter(
        timestamp=latest.to_pydatetime() if latest is not None else None,
        values=values,
        unparseable=unparseable,
        columns=columns,
    )


def reference_from_encounters(latest: LatestEncounter) -> ReferenceDate | None:
    """The approximation from the latest encounter, or ``None`` when there is none."""
    if latest.timestamp is None:
        return None
    ignored = (
        f"; {latest.unparseable} unparseable value(s) were ignored" if latest.unparseable else ""
    )
    return ReferenceDate(
        value=latest.timestamp.date().isoformat(),
        source=ReferenceSource.MAX_ENCOUNTER_DATE,
        detail=(
            f"APPROXIMATION: the latest encounters.{'/'.join(latest.columns)} value, "
            f"{latest.timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')} (over {latest.values} "
            f"timestamps{ignored}). The CSV export does not record when the simulation "
            f"ended: Synthea simulates up to Generator.stop, which defaults to the time it "
            f"ran ('-e YYYYMMDD' overrides it) and is written only to the run metadata "
            f"file, as endTime. Every exported encounter precedes that end, so this date is "
            f"normally a lower bound of it. Use --reference-date or --metadata for an "
            f"exact date."
        ),
        approximate=True,
    )


@dataclass(frozen=True, slots=True)
class ReferenceResolution:
    """Outcome of :func:`resolve_reference_date`."""

    reference: ReferenceDate | None
    #: Why there is no reference date; ``None`` when there is one.
    reason: str | None
    #: Observations to show in the report (e.g. a contradictory metadata endTime).
    notes: tuple[str, ...] = ()
    #: The latest encounter, when ``encounters.csv`` was scanned and could be read.
    latest: LatestEncounter | None = None
    #: What happened to ``encounters.csv``; ``None`` when it was not needed.
    encounters: TableInput | None = None


#: What ``encounters.csv`` is read for, depending on the source of the date.
USED_FOR_APPROXIMATION = "reference date approximation (latest START/STOP)"
USED_FOR_METADATA_CHECK = "consistency of the metadata endTime (latest START/STOP)"


def resolve_reference_date(
    discovery: DiscoveryResult,
    loader: DatasetLoader,
    *,
    user_date: str | None = None,
    metadata_path: str | Path | None = None,
    structure: Mapping[str, StructureReport] | None = None,
) -> ReferenceResolution:
    """Resolve the reference date for a dataset, following the order in the module docstring.

    ``encounters.csv`` (only its ``START`` and ``STOP`` columns) is read when it is the
    fallback, and with a metadata file, to compare its ``endTime`` with the latest
    encounter. A date the user gives is taken as it is, and nothing is read.

    :raises ValueError: both ``user_date`` and ``metadata_path`` were given.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    """
    if user_date is not None and metadata_path is not None:
        raise ValueError("give either a reference date or a metadata file, not both")

    if user_date is not None:
        return ReferenceResolution(reference=reference_from_user(user_date), reason=None)

    if metadata_path is not None:
        explicit = reference_from_metadata(metadata_path)
        latest, encounters = scan_encounters(
            discovery, loader, structure, used_for=USED_FOR_METADATA_CHECK
        )
        notes: list[str] = []
        latest_date = latest.date if latest is not None else None
        if latest_date is not None and latest_date.isoformat() > explicit.value:
            notes.append(
                f"The metadata endTime ({explicit.value}) is earlier than the latest encounter "
                f"({latest_date.isoformat()}). A simulation cannot export encounters after it "
                f"ended, so the metadata file may belong to a different run. The metadata date "
                f"is used as requested."
            )
        return ReferenceResolution(
            reference=explicit,
            reason=None,
            notes=tuple(notes),
            latest=latest,
            encounters=encounters,
        )

    latest, encounters = scan_encounters(
        discovery, loader, structure, used_for=USED_FOR_APPROXIMATION
    )
    if latest is None:
        return ReferenceResolution(
            reference=None,
            reason=(
                f"no reference date: none was given and the approximation from encounters "
                f"is unavailable ({encounters.reason})"
            ),
            encounters=encounters,
        )
    approximation = reference_from_encounters(latest)
    if approximation is None:
        return ReferenceResolution(
            reference=None,
            reason=(
                "no reference date: none was given and encounters.csv holds no parseable "
                "START/STOP timestamp"
            ),
            latest=latest,
            encounters=encounters,
        )
    return ReferenceResolution(
        reference=approximation, reason=None, latest=latest, encounters=encounters
    )


def scan_encounters(
    discovery: DiscoveryResult,
    loader: DatasetLoader,
    structure: Mapping[str, StructureReport] | None,
    *,
    used_for: str,
) -> tuple[LatestEncounter | None, TableInput]:
    """Read ``encounters.csv``'s timestamps and find the latest one.

    Returns the latest encounter (``None`` when the table could not be scanned) and what
    happened to the table. A table whose rows do not line up with its header is not
    read: the loader would pad a short row with nulls, and the maximum could come from
    the wrong column.
    """
    table = next((t for t in discovery.tables if t.name == "encounters"), None)
    if table is None:
        return None, TableInput(
            "encounters",
            used_for,
            InputState.ABSENT,
            reason="encounters.csv is not in the dataset",
        )
    gated = gate_reason(structure, "encounters")
    if gated is not None:
        return None, TableInput("encounters", used_for, InputState.UNREADABLE, reason=gated)
    try:
        header = loader.read_header(table.path)
        columns = [c for c in ENCOUNTER_DATE_COLUMNS if c in header]
        if not columns:
            return None, TableInput(
                "encounters",
                used_for,
                InputState.ABSENT,
                reason="encounters.csv has neither a START nor a STOP column",
            )
        loaded = loader.load(table.path, table="encounters", columns=columns)
    except TableLoadError as exc:
        return None, TableInput(
            "encounters",
            used_for,
            InputState.UNREADABLE,
            reason=f"encounters.csv could not be read: {exc}",
        )
    return latest_encounter(loaded.frame), TableInput(
        "encounters", used_for, InputState.READ, rows=loaded.rows
    )
