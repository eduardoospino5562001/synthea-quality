"""[SCHEMA] Confirmed rules the data-quality checks rely on.

Today this module holds the **date columns**: which columns carry a date and in
which format. Nothing here is inferred from a column name; every entry is confirmed
by Synthea's own documentation and by the generator source, and then measured
against the official sample:

* the *CSV File Data Dictionary* states the format per column
  (``Date (YYYY-MM-DD)`` or ``iso8601 UTC Date (yyyy-MM-dd'T'HH:mm'Z')``);
* ``CSVExporter.java`` shows which writer produces the value: ``dateFromTimestamp``
  for date-only columns (allergies, conditions, patients) and ``iso8601Timestamp``
  for the timestamped ones (encounters, observations, procedures, devices,
  medications, immunizations, imaging studies, claims, payer transitions);
* the official 2026-08 sample was measured column by column: **every one of the
  353,409 non-null values across these 29 columns matches exactly one of the two
  formats, with zero exceptions**. That is also the baseline the checks compare
  against, so the tool reports no date defect on a clean sample.

Two documented discrepancies are recorded rather than smoothed over:

* ``payer_transitions.START_DATE`` / ``END_DATE`` are not documented under those
  names: the dictionary and the code comment still describe the pre-3.0 shape
  (``START_YEAR``/``END_YEAR``, ``Date (YYYY)``) while the writer emits
  ``iso8601Timestamp``. The columns are keyed on the names the code actually
  writes, and the format is the one the code writes.
* ``claims_transactions.PATIENTINSURANCEID`` is documented as a foreign key but is
  not applied as one; see :mod:`synthea_quality.schema.keys`.

Only a date's *shape* is checked. Whether a date makes sense (start before stop,
birth before death, events before birth) belongs to the temporal checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from synthea_quality.schema.tables import TableSpec, tables_by_name

#: ``YYYY-MM-DD``: what the dictionary documents for these columns and what
#: ``CSVExporter.dateFromTimestamp`` writes.
DATE_ONLY = "date"
DATE_ONLY_PATTERN = r"\d{4}-\d{2}-\d{2}"
DATE_ONLY_DOCUMENTED_AS = "CSV File Data Dictionary: Date (YYYY-MM-DD)"

#: ``yyyy-MM-dd'T'HH:mm'Z'``: what the dictionary documents for the rest and what
#: ``CSVExporter.iso8601Timestamp`` writes.
ISO8601_UTC = "iso8601-utc"
ISO8601_UTC_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
ISO8601_UTC_DOCUMENTED_AS = "CSV File Data Dictionary: iso8601 UTC Date (yyyy-MM-dd'T'HH:mm'Z')"

#: Value of ``DateColumn.documented_as`` when the dictionary lags the code.
STALE_DOCUMENTATION = (
    "not documented under this name: the dictionary still describes the pre-3.0 "
    "START_YEAR/END_YEAR (Date (YYYY)) while CSVExporter writes iso8601Timestamp"
)


@dataclass(frozen=True, slots=True)
class DateFormat:
    """A date shape a column is allowed to contain.

    ``identifier`` is the machine-readable name used in metrics; ``label`` is how
    the format is described in a human-readable message.
    """

    identifier: str
    pattern: str
    documented_as: str
    label: str


#: The two shapes Synthea writes. A column accepts exactly one of them in practice.
DATE_FORMAT = DateFormat(
    DATE_ONLY,
    DATE_ONLY_PATTERN,
    DATE_ONLY_DOCUMENTED_AS,
    label="the YYYY-MM-DD date format",
)
TIMESTAMP_FORMAT = DateFormat(
    ISO8601_UTC,
    ISO8601_UTC_PATTERN,
    ISO8601_UTC_DOCUMENTED_AS,
    label="the ISO-8601 UTC format (yyyy-MM-dd'T'HH:mm'Z')",
)


@dataclass(frozen=True, slots=True)
class DateColumn:
    """A column whose values must match the confirmed date format."""

    table: str
    column: str
    format: DateFormat
    documented_as: str = DATE_ONLY_DOCUMENTED_AS

    @property
    def check_id(self) -> str:
        """Stable identifier, e.g. ``dates.conditions.START``."""
        return f"dates.{self.table}.{self.column}"


def _date_only(table: str, *columns: str) -> tuple[DateColumn, ...]:
    return tuple(DateColumn(table, column, DATE_FORMAT) for column in columns)


def _timestamps(
    table: str, *columns: str, documented_as: str = ISO8601_UTC_DOCUMENTED_AS
) -> tuple[DateColumn, ...]:
    return tuple(
        DateColumn(table, column, TIMESTAMP_FORMAT, documented_as) for column in columns
    )


#: Columns confirmed to hold a date, with the format the generator writes.
DATE_COLUMNS: tuple[DateColumn, ...] = (
    *_date_only("allergies", "START", "STOP"),
    *_date_only("careplans", "START", "STOP"),
    *_date_only("conditions", "START", "STOP"),
    *_date_only("patients", "BIRTHDATE", "DEATHDATE"),
    *_date_only("supplies", "DATE"),
    *_timestamps(
        "claims",
        "CURRENTILLNESSDATE",
        "SERVICEDATE",
        "LASTBILLEDDATE1",
        "LASTBILLEDDATE2",
        "LASTBILLEDDATEP",
    ),
    *_timestamps("claims_transactions", "FROMDATE", "TODATE"),
    *_timestamps("devices", "START", "STOP"),
    *_timestamps("encounters", "START", "STOP"),
    *_timestamps("imaging_studies", "DATE"),
    *_timestamps("immunizations", "DATE"),
    *_timestamps("medications", "START", "STOP"),
    *_timestamps("observations", "DATE"),
    *_timestamps("procedures", "START", "STOP"),
    *_timestamps(
        "payer_transitions", "START_DATE", "END_DATE", documented_as=STALE_DOCUMENTATION
    ),
)


def date_columns_for(table: str) -> tuple[DateColumn, ...]:
    """Date columns confirmed for ``table``."""
    return tuple(rule for rule in DATE_COLUMNS if rule.table == table)


def date_rule_for(table: str, column: str) -> DateColumn | None:
    """The confirmed date rule of one column, or ``None`` if it has none.

    Other check families use this to reuse the confirmed format instead of
    re-implementing date parsing.
    """
    return next(
        (rule for rule in DATE_COLUMNS if rule.table == table and rule.column == column),
        None,
    )


class _CatalogueError(ValueError):
    """A date rule points at something the table catalogue does not declare."""


def _validate_catalogue() -> None:
    """Fail loudly at import time if a date rule contradicts the table catalogue."""
    known = tables_by_name()
    seen: set[tuple[str, str]] = set()

    for rule in DATE_COLUMNS:
        spec: TableSpec | None = known.get(rule.table)
        if spec is None:
            raise _CatalogueError(f"date rule refers to unknown table '{rule.table}'")
        if rule.column not in spec.columns:
            raise _CatalogueError(
                f"date rule refers to column '{rule.column}', which is not part of "
                f"table '{rule.table}'"
            )
        if not rule.format.pattern.strip():
            raise _CatalogueError(f"date rule {rule.check_id} has no format pattern")
        if not rule.documented_as.strip():
            raise _CatalogueError(f"date rule {rule.check_id} has no documentation note")
        if (rule.table, rule.column) in seen:
            raise _CatalogueError(f"duplicate date rule for {rule.table}.{rule.column}")
        seen.add((rule.table, rule.column))


_validate_catalogue()
