"""[EXPORT] How much history a Synthea CSV export holds, read in one place.

``synthea-profile``, ``synthea-prevalence``, ``synthea-incidence``,
``synthea-observations`` and ``synthea-validate-module`` all accept ``--metadata``,
a Synthea run metadata file. Besides ``endTime`` (the reference date), that file may
hold ``exporter.years_of_history``: how many years back the CSV export reaches (10 by
default; 0 keeps everything). Every analysis report shows the same notice from it, so
the reading lives here.

What Synthea's source says (read at commit ``44f645c``):

* ``MetadataExporter.java:93-97`` writes ``"exporter.years_of_history"`` with
  ``Config.get(...)``: the value is a **string** (for example ``"10"``);
* ``Exporter.java:729-779`` (``filterForExport``) keeps what is after
  ``cutoffDate = endTime - Utilities.convertTime("years", N)``: conditions, allergies,
  medications and care plans still active after the cut-off, procedures,
  immunizations and encounters that start after it, observations per
  ``keepObservation``;
* ``Utilities.java:75`` converts with ``TimeUnit.DAYS.toMillis((long) 365.25 * value)``:
  the cast applies to 365.25 first, so the cut-off is ``endTime`` minus
  **365 × N days**, not calendar years.

So :func:`synthea_cutoff` subtracts ``365 * years`` days from the reference date, and
the notice cites these lines instead of restating them loosely.

A value of ``0`` means the whole history was exported (no filter). Without metadata
there is nothing to read. Only the standard library is imported here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

#: Key of the setting in a Synthea metadata file.
YEARS_OF_HISTORY_KEY = "exporter.years_of_history"

#: Largest usable value of the setting: more than any possible simulation. Above it
#: the value is reported as invalid instead of overflowing date arithmetic.
MAX_YEARS_OF_HISTORY = 1000

#: What the notice cites as the source of the cut-off rule.
CUTOFF_PROVENANCE = (
    "Synthea keeps what is after endTime minus 365 * N days "
    "(Exporter.filterForExport with Utilities.convertTime, which truncates 365.25 to 365)"
)


@dataclass(frozen=True, slots=True)
class ExportHistory:
    """What a Synthea metadata file says about the exported history."""

    #: The setting's value, or ``None`` when there is no usable one.
    years: int | None
    #: ``"no_metadata"`` (no file given), ``"read"``, ``"missing_key"`` or ``"invalid"``.
    status: str
    #: The metadata path as given, or ``None`` without a file.
    metadata_file: str | None
    #: The cut-off day: only set when ``years > 0`` and the reference date is known.
    cutoff: date | None
    #: Why there is no usable value; ``None`` when there is one or no file was given.
    reason: str | None = None

    @classmethod
    def no_metadata(cls) -> "ExportHistory":
        """No metadata file was given, so nothing is known from it."""
        return cls(years=None, status="no_metadata", metadata_file=None, cutoff=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "years_of_history": self.years,
            "status": self.status,
            "metadata_file": self.metadata_file,
            "cutoff": self.cutoff.isoformat() if self.cutoff is not None else None,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportHistory":
        raw = data.get("cutoff")
        return cls(
            years=data.get("years_of_history"),
            status=data["status"],
            metadata_file=data.get("metadata_file"),
            cutoff=date.fromisoformat(raw) if raw is not None else None,
            reason=data.get("reason"),
        )


def read_export_history(
    metadata: str | Path | None, reference: date | None
) -> ExportHistory:
    """The exported history of ``metadata`` at ``reference``.

    ``"10"`` and ``10`` are valid; a negative, decimal or textual value is invalid, as
    is a value above ``MAX_YEARS_OF_HISTORY``. A file that cannot be read is invalid
    too, without raising: the reference-date resolution already reported it.
    """
    if metadata is None:
        return ExportHistory.no_metadata()
    file_path = Path(metadata)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        # ValueError is explicit although it is the base of JSONDecodeError: a huge
        # integer literal fails json.loads with a plain ValueError.
        return ExportHistory(
            years=None,
            status="invalid",
            metadata_file=str(metadata),
            cutoff=None,
            reason="the file could not be read",
        )
    if not isinstance(data, dict):
        return ExportHistory(
            years=None,
            status="invalid",
            metadata_file=str(metadata),
            cutoff=None,
            reason="the file is not a JSON object",
        )
    if YEARS_OF_HISTORY_KEY not in data or data[YEARS_OF_HISTORY_KEY] is None:
        return ExportHistory(
            years=None,
            status="missing_key",
            metadata_file=str(metadata),
            cutoff=None,
            reason="the key is absent",
        )
    years = _parse_years(data[YEARS_OF_HISTORY_KEY])
    if years is None:
        return ExportHistory(
            years=None,
            status="invalid",
            metadata_file=str(metadata),
            cutoff=None,
            reason=f"value {data[YEARS_OF_HISTORY_KEY]!r} is not a non-negative integer",
        )
    if years > MAX_YEARS_OF_HISTORY:
        return ExportHistory(
            years=None,
            status="invalid",
            metadata_file=str(metadata),
            cutoff=None,
            reason=f"value {data[YEARS_OF_HISTORY_KEY]!r} is out of range",
        )
    try:
        cutoff = (
            synthea_cutoff(reference, years) if years > 0 and reference is not None else None
        )
    except (ValueError, OverflowError):
        return ExportHistory(
            years=None,
            status="invalid",
            metadata_file=str(metadata),
            cutoff=None,
            reason=f"value {data[YEARS_OF_HISTORY_KEY]!r} is out of range",
        )
    return ExportHistory(
        years=years,
        status="read",
        metadata_file=str(metadata),
        cutoff=cutoff,
    )


def _parse_years(value: Any) -> int | None:
    """A non-negative integer number of years, or ``None`` when unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        # [0-9], not \d: Unicode digits (for example "١٠") convert with int() but are
        # not a value Synthea writes.
        if re.fullmatch(r"[0-9]+", value.strip()):
            try:
                return int(value.strip())
            except (ValueError, OverflowError):
                # A digit string too long for int() (thousands of digits).
                return None
        return None
    return None


def synthea_cutoff(reference: date, years: int) -> date:
    """The export cut-off for ``years`` of history at ``reference``.

    ``reference`` minus 365 × ``years`` days, as Synthea's ``filterForExport`` computes
    it (see the module docstring): a calendar-year subtraction would land up to a few
    days off because of leap years.

    :raises ValueError: a negative or out-of-range number of years.
    """
    if years < 0:
        raise ValueError("years of history must be non-negative")
    try:
        return reference - timedelta(days=365 * years)
    except OverflowError as exc:
        raise ValueError(f"years of history {years!r} is out of range") from exc


def notice_lines(history: ExportHistory) -> list[str]:
    """The time-filtered dataset notice as Markdown lines, empty when none applies.

    A whole-history export (``0``) and no metadata give no notice. The wording states
    no verdict: it describes the filter and what it may rest on.
    """
    if history.status == "read" and history.years is not None and history.years > 0:
        cutoff = history.cutoff.isoformat() if history.cutoff is not None else "an unknown date"
        return [
            f"> **Time-filtered dataset.** This dataset was exported with "
            f"`{YEARS_OF_HISTORY_KEY} = {history.years}` (read from `{history.metadata_file}`).",
            f"> Synthea leaves out most records from before the cut-off, {cutoff} "
            f"(365 × {history.years} days before the end of the simulation); conditions, "
            f"allergies, medications and care plans still active after it are kept. Statistics",
            f"> that look back in time — lifetime prevalence, \"ever\" medication use, prior "
            f"cases in incidence, the earliest records — may rest on incomplete data.",
        ]
    if history.status in ("missing_key", "invalid"):
        return [
            f"> **Exported history unknown.** The metadata file `{history.metadata_file}` "
            f"has no usable `{YEARS_OF_HISTORY_KEY}` ({history.reason}).",
        ]
    return []
