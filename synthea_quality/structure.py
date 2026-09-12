"""[STRUCTURE] Row-level structural validation of a Synthea CSV file.

Why this module exists
----------------------
A CSV is a table only if every row has exactly as many fields as the header. pandas
cannot be trusted to say so, and the measurement is unambiguous:

* a row with **more** fields than the header is reported only when every column is
  read. The key checks load a column subset, so on a `patients.csv` whose third data
  row had four fields, ``pk.patients.Id`` reported ``PASS`` while every whole-table
  check of that table was ``SKIPPED`` — and the run ended with exit code ``0``;
* a row with **fewer** fields is padded with nulls and reported by nothing, so
  ``nulls.patients`` counted an invented null and reported ``PASS``.

Both are false negatives about a table whose rows do not line up with its columns, and
no other check in this tool can be believed once that is true. So every table's rows are
validated here, once per file, **before any check loads it**.

How
---
The standard library's :mod:`csv` reader does the parsing, so quoting, embedded
delimiters and embedded newlines are handled by the format's own rules rather than by
splitting lines. A row is a defect when its field count differs from the header's.

What it produces
----------------
A :class:`StructureReport` per table. The caller decides what to do with it; this module
only states the fact. The reporting layer turns a failing report into a ``FAIL`` check
(``structure.<table>``) because corrupted rows are a deterministic property of the
dataset, not a tool failure — and the check runners report ``SKIPPED``, never ``PASS``,
for every rule of a table whose structure is broken.

Severity
--------
``HIGH``. A malformed date or a duplicated row is confined to the row it is in
(``MEDIUM``); a row that does not line up with the header invalidates *every* value of
that row, because each one may be attributed to the wrong column, and it is the one
defect that makes the whole table unverifiable. That is what ``HIGH`` means here: the
contents of the table cannot be trusted.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from synthea_quality.errors import TableLoadError
from synthea_quality.loader import open_readable
from synthea_quality.models import DEFAULT_SAMPLE_LIMIT, CheckResult, Severity, Status

#: Prefix of the check a structural defect is reported as, so it lands in a category.
STRUCTURE_CHECK_PREFIX = "structure."

#: Corrupted rows cannot be attributed to columns, so nothing about the table holds.
STRUCTURE_SEVERITY = Severity.HIGH


@dataclass(frozen=True, slots=True)
class RowDefect:
    """One row whose field count does not match the header."""

    #: Data row number: 1 is the first row after the header, i.e. line 2 of the file.
    row: int
    #: Fields actually found on that row.
    fields: int

    def to_sample(self) -> dict[str, object]:
        """Sample entry a report can show to point at the offending row."""
        return {"row": self.row, "fields": self.fields}


@dataclass(frozen=True, slots=True)
class StructureReport:
    """Result of validating one file's rows against its own header."""

    table: str
    path: Path
    #: Number of fields the header declares.
    header_fields: int
    #: Data rows read from the file, however malformed.
    rows_checked: int
    #: Rows whose field count differs from the header, counted in full.
    defects_total: int
    #: The first ``sample_limit`` defects, so a report stays bounded.
    defects: tuple[RowDefect, ...] = ()
    sample_limit: int = DEFAULT_SAMPLE_LIMIT

    @property
    def ok(self) -> bool:
        """True when every row has as many fields as the header."""
        return self.defects_total == 0

    @property
    def check_id(self) -> str:
        return f"{STRUCTURE_CHECK_PREFIX}{self.table}"

    def as_reason(self) -> str:
        """One-line reason a check of this table must not report a verdict."""
        first = self.defects[0] if self.defects else None
        detail = (
            f" (the first one is row {first.row}, with {first.fields} field(s))"
            if first is not None
            else ""
        )
        return (
            f"table '{self.table}' has {self.defects_total} row(s) whose field count does not "
            f"match its header of {self.header_fields} field(s){detail}, so its rows cannot be "
            f"attributed to its columns"
        )

    def to_check_result(self) -> CheckResult:
        """The finding this report produces: a ``FAIL``, never an internal error."""
        return CheckResult(
            check_id=self.check_id,
            status=Status.FAIL,
            severity=STRUCTURE_SEVERITY,
            message=(
                f"{self.defects_total} row(s) of {self.table} have a field count that does not "
                f"match the header of {self.header_fields} field(s), so those rows cannot be "
                f"read as {self.header_fields} columns (the file is malformed, not the tool)"
            ),
            table=self.table,
            metrics={
                "rows": self.rows_checked,
                "header_fields": self.header_fields,
                "defect_rows": self.defects_total,
                "sample_limit": self.sample_limit,
            },
            samples=tuple(defect.to_sample() for defect in self.defects),
            metadata={
                "rule": "every row of a CSV must have as many fields as the header",
                "validated_with": "csv (standard library), so quoting and embedded newlines hold",
                "consequence": "every check of this table is SKIPPED: no verdict is invented",
            },
        )


def validate_structure(
    path: str | Path,
    *,
    table: str | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> StructureReport:
    """Read ``path`` and compare every row's field count with the header's.

    :param path: CSV file to read. It is opened read-only and never modified.
    :param table: logical table name; defaults to the file name without suffix.
    :param sample_limit: how many offending rows to keep as samples. The total is
        always counted in full.
    :raises TableLoadError: the file cannot be opened, is empty, or is not valid CSV.
    """
    file_path = Path(path)
    table_name = table or file_path.stem

    with open_readable(file_path) as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader, None)
        except csv.Error as exc:
            raise TableLoadError(f"{file_path} has an unreadable header: {exc}") from exc
        if header is None:
            raise TableLoadError(f"{file_path} is empty (no header line)")

        expected = len(header)
        rows_checked = 0
        defects_total = 0
        defects: list[RowDefect] = []
        try:
            for row in reader:
                rows_checked += 1
                if len(row) != expected:
                    defects_total += 1
                    if len(defects) < sample_limit:
                        defects.append(RowDefect(row=rows_checked, fields=len(row)))
        except csv.Error as exc:
            raise TableLoadError(f"{file_path} is not well-formed CSV: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise TableLoadError(f"{file_path} is not valid UTF-8: {exc}") from exc
        except OSError as exc:
            raise TableLoadError(f"{file_path} could not be read: {exc}") from exc

    return StructureReport(
        table=table_name,
        path=file_path,
        header_fields=expected,
        rows_checked=rows_checked,
        defects_total=defects_total,
        defects=tuple(defects),
        sample_limit=sample_limit,
    )


def validate_tables(
    paths: Mapping[str, Path],
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> dict[str, StructureReport]:
    """Validate every table in ``paths``, one file at a time.

    A table that cannot be read at all is left out of the result: the loader's own error
    contract already reports it, and repeating it here would invent a second verdict
    about the same failure.
    """
    reports: dict[str, StructureReport] = {}
    for table, path in paths.items():
        try:
            reports[table] = validate_structure(path, table=table, sample_limit=sample_limit)
        except TableLoadError:
            continue
    return reports


def gate_reason(
    structure: Mapping[str, StructureReport] | None, table: str
) -> str | None:
    """Reason no check of ``table`` may report a verdict, or ``None`` when it may.

    ``None`` for both "the table is well formed" and "nothing is known about its
    structure": the second case is a table that could not be read, which the callers
    already report as an unreadable table.
    """
    if not structure:
        return None
    report = structure.get(table)
    if report is None or report.ok:
        return None
    return report.as_reason()
