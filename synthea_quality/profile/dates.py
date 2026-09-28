"""[PROFILE] Strict parsing of the two date shapes Synthea writes.

The loader keeps every value as text on purpose; turning text into a date is the job
of the code that needs a date, so it happens here and only here.

Parsing is strict and never silent:

* the shape must match exactly the pattern confirmed in
  :mod:`synthea_quality.schema.quality` (``YYYY-MM-DD`` for date-only columns,
  ``yyyy-MM-dd'T'HH:mm:ss'Z'`` for timestamps), so ``2020-1-5`` or ``2020-01-05 10:00``
  is not quietly accepted;
* the calendar must be valid too: ``2021-02-30`` has the right shape and is still
  unparseable;
* every value that is not empty and does not parse is **counted**, never dropped
  without a trace. Callers report ``empty`` and ``unparseable`` next to the numbers
  computed from the values that did parse.

A date pandas cannot represent (outside the years 1677–2262 at nanosecond resolution)
counts as unparseable as well. Synthea generates no such date, and reporting one is
better than failing the whole profile over it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from synthea_quality.schema.quality import DATE_ONLY_PATTERN, ISO8601_UTC_PATTERN

#: ``strptime`` equivalents of the confirmed patterns.
DATE_ONLY_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True, slots=True, eq=False)
class ParsedDates:
    """A text column converted to datetimes, with the losses counted."""

    #: Same index as the input; ``NaT`` where the value was empty or unparseable.
    values: pd.Series
    #: Values that were an empty field (the only null the loader produces).
    empty: int
    #: Values that were present but did not parse.
    unparseable: int

    @property
    def valid(self) -> int:
        """How many values parsed."""
        return int(self.values.notna().sum())


def parse_date_only(values: pd.Series) -> ParsedDates:
    """Parse a ``YYYY-MM-DD`` column (e.g. ``patients.BIRTHDATE``)."""
    return _parse(values, DATE_ONLY_PATTERN, DATE_ONLY_FORMAT)


def parse_timestamps(values: pd.Series) -> ParsedDates:
    """Parse an ISO-8601 UTC column (e.g. ``encounters.START``) as naive UTC datetimes."""
    return _parse(values, ISO8601_UTC_PATTERN, TIMESTAMP_FORMAT)


def _parse(values: pd.Series, pattern: str, fmt: str) -> ParsedDates:
    empty = values.isna()
    well_shaped = values.str.fullmatch(pattern, na=False).astype(bool)
    parsed = pd.to_datetime(values.where(well_shaped), format=fmt, errors="coerce")
    unparseable = int((~empty & parsed.isna()).sum())
    return ParsedDates(values=parsed, empty=int(empty.sum()), unparseable=unparseable)
