"""Dataset discovery: which Synthea CSV tables are present in a directory.

Discovery only looks at the file system. It never opens a CSV and never parses
content, so it stays fast and safe on a directory full of large files; reading
and validating the data is the loader's job.

Nothing is required to be present. Synthea decides which CSV files to write from
``exporter.csv.included_files`` / ``exporter.csv.excluded_files``, so a missing
table is a fact to report, never an error.

A name that *looks* like a table but is not a regular file — a directory called
``encounters.csv``, a symbolic link that does not resolve — is recorded in
``anomalous_entries`` with a readable reason instead of being skipped in silence. The
table still counts as missing, because a name that yields no file is still a table that
was not observed; what changes is that the dataset now says why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthea_quality.errors import DiscoveryError
from synthea_quality.models import AnomalousEntry
from synthea_quality.schema.tables import CONTRACT_ID, SYNTHEA_TABLES, TableSpec

CSV_SUFFIX = ".csv"


@dataclass(frozen=True, slots=True)
class DiscoveredTable:
    """A table from the known catalogue that was found in the dataset."""

    spec: TableSpec
    path: Path

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def file_name(self) -> str:
        """Name of the file on disk, which may differ in case from the catalogue."""
        return self.path.name


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Outcome of inspecting one dataset directory."""

    data_dir: Path
    #: Identifier of the schema contract the catalogue came from, or ``None``
    #: when the caller supplied a custom catalogue.
    contract_id: str | None
    tables: tuple[DiscoveredTable, ...]
    missing_tables: tuple[TableSpec, ...]
    unknown_csv_files: tuple[str, ...]
    ignored_files: tuple[str, ...]
    #: Names that could be a table but are not readable files, with the reason.
    anomalous_entries: tuple[AnomalousEntry, ...] = ()

    @property
    def table_names(self) -> tuple[str, ...]:
        """Names of the known tables found, in a deterministic order."""
        return tuple(table.name for table in self.tables)

    @property
    def is_empty(self) -> bool:
        """True when the directory holds no Synthea CSV data at all."""
        return not self.tables and not self.unknown_csv_files


def discover_dataset(
    data_dir: str | Path,
    *,
    known_tables: Sequence[TableSpec] | None = None,
) -> DiscoveryResult:
    """Inspect ``data_dir`` and classify the files found in it.

    Files are classified as known tables (matched case-insensitively against
    ``known_tables``), unknown CSV files, or ignored non-CSV files. Sub-
    directories are skipped.

    :param data_dir: directory holding the Synthea CSV export.
    :param known_tables: catalogue to match against; defaults to the current
        confirmed contract (:data:`synthea_quality.schema.tables.SYNTHEA_TABLES`).
    :raises DiscoveryError: the directory does not exist, is not a directory, or
        contains several files that map to the same table.
    """
    root = Path(data_dir)
    if not root.exists():
        raise DiscoveryError(f"dataset directory does not exist: {root}")
    if not root.is_dir():
        raise DiscoveryError(f"dataset path is not a directory: {root}")

    specs = tuple(known_tables) if known_tables is not None else SYNTHEA_TABLES
    by_file_name = _index_by_file_name(specs)

    tables: list[DiscoveredTable] = []
    unknown_csv_files: list[str] = []
    ignored_files: list[str] = []
    anomalous_entries: list[AnomalousEntry] = []
    seen_file_names: dict[str, str] = {}

    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if not entry.is_file():
            # A sub-directory is a normal thing to find here and is passed over quietly.
            # A name that could be a table but is not a file is worth saying out loud:
            # otherwise it is indistinguishable from a table the generator never wrote.
            anomaly = _anomalous_entry(entry)
            if anomaly is not None:
                anomalous_entries.append(anomaly)
            continue
        if entry.suffix.lower() != CSV_SUFFIX:
            ignored_files.append(entry.name)
            continue

        spec = by_file_name.get(entry.name.lower())
        if spec is None:
            unknown_csv_files.append(entry.name)
            continue

        previous = seen_file_names.get(spec.name)
        if previous is not None:
            raise DiscoveryError(
                "ambiguous dataset: several files map to the same table "
                f"'{spec.name}': {previous}, {entry.name}"
            )

        seen_file_names[spec.name] = entry.name
        tables.append(DiscoveredTable(spec=spec, path=entry))

    found_names = set(seen_file_names)
    return DiscoveryResult(
        data_dir=root,
        contract_id=CONTRACT_ID if known_tables is None else None,
        tables=tuple(sorted(tables, key=lambda table: table.name)),
        missing_tables=tuple(spec for spec in specs if spec.name not in found_names),
        unknown_csv_files=tuple(unknown_csv_files),
        ignored_files=tuple(ignored_files),
        anomalous_entries=tuple(
            sorted(anomalous_entries, key=lambda entry: entry.file_name)
        ),
    )


def _anomalous_entry(entry: Path) -> AnomalousEntry | None:
    """Explain a path that could be a table name but is not a readable file.

    ``None`` for anything that is not even a CSV name, so ordinary sub-directories stay
    as quiet as they were.
    """
    if entry.suffix.lower() != CSV_SUFFIX:
        return None
    if entry.is_dir():
        return AnomalousEntry(
            file_name=entry.name,
            reason=(
                f"'{entry.name}' is a directory, not a file, so no table can be read "
                f"from it; the table it names counts as missing"
            ),
        )
    if entry.is_symlink():
        return AnomalousEntry(
            file_name=entry.name,
            reason=(
                f"'{entry.name}' is a symbolic link that does not resolve to a file, so "
                f"no table can be read from it; the table it names counts as missing"
            ),
        )
    return AnomalousEntry(
        file_name=entry.name,
        reason=(
            f"'{entry.name}' is not a regular file, so no table can be read from it; the "
            f"table it names counts as missing"
        ),
    )


def _index_by_file_name(specs: Sequence[TableSpec]) -> dict[str, TableSpec]:
    """Index a catalogue by lower-cased file name, rejecting duplicates.

    A duplicate means the catalogue itself is wrong, which is a bug in the tool
    and not a problem with the user's dataset, so it raises ``ValueError``.
    """
    index: dict[str, TableSpec] = {}
    for spec in specs:
        key = spec.file_name.lower()
        if key in index:
            raise ValueError(
                f"catalogue is inconsistent: '{spec.file_name}' is declared twice "
                f"({index[key].name}, {spec.name})"
            )
        index[key] = spec
    return index
