"""[OBSERVATIONS] Structured result of a description of observation values.

The same principles as the other reports: a result is ``COMPUTED`` or ``SKIPPED`` with its
reason, provenance travels with it, and :meth:`ObservationsReport.to_dict` is
deterministic (fixed key order, rounded floats). The JSON layout is versioned by
``OBSERVATIONS_SCHEMA_VERSION``.

A distribution is described, never judged: n, minimum, the 5th, 25th, 50th, 75th and
95th percentiles and the maximum, with the units the values were written in. Below
:data:`MIN_PATIENTS_FOR_PERCENTILES` patients only n, minimum, median and maximum are
given: with a handful of values the outer percentiles are interpolations between two or
three numbers and suggest a precision the data do not have. A reference
range is shown next to it with how many patients fall below, within and above it.

Only the standard library is imported here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synthea_quality import __version__
from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.export_history import ExportHistory
from synthea_quality.models import utc_now_iso
from synthea_quality.observations.definitions import ReferenceRange
from synthea_quality.profile.models import ReferenceDate, SectionStatus, TableInput

#: Version of the JSON layout produced by :meth:`ObservationsReport.to_dict`.
OBSERVATIONS_SCHEMA_VERSION = 1

#: Decimal places kept for values in the structured result.
VALUE_DECIMALS = 6

#: The percentiles of every summary.
PERCENTILES = (5, 25, 50, 75, 95)

#: Fewest values for which the 5th, 25th, 75th and 95th percentiles are given; below it a
#: summary has only n, minimum, median and maximum.
MIN_PATIENTS_FOR_PERCENTILES = 10

#: Said wherever a summary below that threshold is shown.
PERCENTILES_NOTE = (
    f"Percentiles are not shown below {MIN_PATIENTS_FOR_PERCENTILES} patients: only n, "
    f"minimum, median and maximum."
)

#: How percentiles are computed, as the report states it.
PERCENTILE_METHOD = (
    "linear interpolation between the closest ranks (Hyndman and Fan type 7, the default of "
    "R, NumPy and pandas)"
)

#: What each patient contributes, as the report states it.
VALUE_RULE = (
    "each patient contributes one value per code and units: the latest with DATE on or before "
    "the reference date (the median of the values sharing that latest DATE, if several)"
)


@dataclass(frozen=True, slots=True)
class ValueSummary:
    """A distribution of one value per patient.

    Every statistic is ``None`` when ``n`` is 0, and the 5th, 25th, 75th and 95th
    percentiles are ``None`` when ``n`` is below :data:`MIN_PATIENTS_FOR_PERCENTILES`.
    """

    n: int
    minimum: float | None = None
    p5: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    p95: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        core = (self.minimum, self.median, self.maximum)
        outer = (self.p5, self.p25, self.p75, self.p95)
        if self.n < 0:
            raise ValueError("n must be >= 0")
        if (self.n == 0) != all(value is None for value in core):
            raise ValueError("a summary has minimum, median and maximum exactly when n > 0")
        if (self.n >= MIN_PATIENTS_FOR_PERCENTILES) != all(v is not None for v in outer):
            raise ValueError(
                f"a summary has its outer percentiles exactly when n >= "
                f"{MIN_PATIENTS_FOR_PERCENTILES}"
            )

    @property
    def percentiles_withheld(self) -> bool:
        """True when there are values but too few for the outer percentiles."""
        return 0 < self.n < MIN_PATIENTS_FOR_PERCENTILES

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "min": _round(self.minimum),
            "p5": _round(self.p5),
            "p25": _round(self.p25),
            "median": _round(self.median),
            "p75": _round(self.p75),
            "p95": _round(self.p95),
            "max": _round(self.maximum),
        }


@dataclass(frozen=True, slots=True)
class ValueStratum:
    """The distribution within one value of a dimension (age band, ``GENDER``)."""

    dimension: str
    #: ``None`` for patients whose value is empty (``GENDER``) — never merged into another.
    value: str | None
    #: Patients of the population in the stratum, with a value or not.
    patients: int
    summary: ValueSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "patients": self.patients,
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RangeComparison:
    """A reference range next to the observed values: how many fall below, within, above."""

    range: ReferenceRange
    status: SectionStatus
    reason: str | None = None
    below: int = 0
    within: int = 0
    above: int = 0

    def __post_init__(self) -> None:
        if (self.status is SectionStatus.SKIPPED) != (self.reason is not None):
            raise ValueError("a range comparison has a reason exactly when it is skipped")

    def to_dict(self) -> dict[str, Any]:
        computed = self.status is SectionStatus.COMPUTED
        return {
            **self.range.to_dict(),
            "note": self.range.note,
            "status": self.status.value,
            "reason": self.reason,
            "below": self.below if computed else None,
            "within": self.within if computed else None,
            "above": self.above if computed else None,
        }


@dataclass(frozen=True, slots=True)
class UnitGroup:
    """The values of one code written in one unit; units are never converted."""

    #: ``None`` when ``UNITS`` is empty.
    units: str | None
    summary: ValueSummary
    strata: tuple[ValueStratum, ...] = ()
    reference_range: RangeComparison | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "summary": self.summary.to_dict(),
            "strata": [stratum.to_dict() for stratum in self.strata],
            "reference_range": self.reference_range.to_dict() if self.reference_range else None,
            "metrics": _sorted(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """The values of one observation asked for, one group per unit."""

    name: str
    code: str
    description: str | None
    status: SectionStatus
    reason: str | None = None
    cohort: CohortSpec | None = None
    #: Patients the values describe: the alive patients, or those of the cohort.
    population: int | None = None
    lookback_years: int | None = None
    groups: tuple[UnitGroup, ...] = ()
    #: Set when the definition has a range: a range in units no group has is not compared.
    reference_range: ReferenceRange | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is SectionStatus.SKIPPED:
            if not self.reason or self.groups:
                raise ValueError(f"skipped observation {self.name!r} needs a reason and no values")
        elif self.reason is not None or not self.groups:
            raise ValueError(f"computed observation {self.name!r} needs values and no reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "status": self.status.value,
            "reason": self.reason,
            "cohort": self.cohort.to_dict() if self.cohort else None,
            "population": self.population,
            "lookback_years": self.lookback_years,
            "reference_range": self.reference_range.to_dict() if self.reference_range else None,
            "groups": [group.to_dict() for group in self.groups],
            "metrics": _sorted(self.metrics),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class GeneralRow:
    """One (code, units) of the general table."""

    code: str
    units: str | None
    description: str | None
    summary: ValueSummary
    #: The code is written in more than one unit in the table.
    mixed_units: bool = False
    #: The code also has rows whose ``TYPE`` is not ``numeric``.
    other_types: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "units": self.units,
            "description": self.description,
            "mixed_units": self.mixed_units,
            "other_types": self.other_types,
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GeneralTable:
    """Every numeric (code, units) among the alive, and the codes that need care."""

    status: SectionStatus
    reason: str | None = None
    rows: tuple[GeneralRow, ...] = ()
    #: How many rows the Markdown lists (the JSON always carries every row).
    top: int = 30
    #: Codes whose numeric rows use more than one unit: ``{code, description, units}``.
    mixed_units: tuple[dict[str, Any], ...] = ()
    #: Codes written with more than one ``TYPE``: ``{code, description, types}``.
    mixed_types: tuple[dict[str, Any], ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "top": self.top,
            "metrics": _sorted(self.metrics),
            "notes": list(self.notes),
            "mixed_units": [_sorted(item) for item in self.mixed_units],
            "mixed_types": [_sorted(item) for item in self.mixed_types],
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class ObservationsReport:
    """The complete result of one ``synthea-observations`` run."""

    data_dir: str
    reference_date: ReferenceDate | None
    reference_reason: str | None
    age_bands: tuple[int, ...]
    alive: int | None
    observations: tuple[ObservationResult, ...]
    general: GeneralTable
    module_file: str | None = None
    lookback_years: int | None = None
    inputs: tuple[TableInput, ...] = ()
    notes: tuple[str, ...] = ()
    #: What the metadata file says about the exported history (no notice without it).
    export_history: ExportHistory = field(default_factory=ExportHistory.no_metadata)
    generated_at: str = field(default_factory=utc_now_iso)
    tool_version: str = __version__

    @property
    def incomplete(self) -> bool:
        return any(item.state.value == "unreadable" for item in self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OBSERVATIONS_SCHEMA_VERSION,
            "tool_version": self.tool_version,
            "generated_at": self.generated_at,
            "data_dir": self.data_dir,
            "module_file": self.module_file,
            "reference_date": self.reference_date.to_dict() if self.reference_date else None,
            "reference_reason": self.reference_reason,
            "age_bands": list(self.age_bands),
            "alive": self.alive,
            "lookback_years": self.lookback_years,
            "definitions": definitions(),
            "inputs": [item.to_dict() for item in self.inputs],
            "notes": list(self.notes),
            "observations": [item.to_dict() for item in self.observations],
            "export_history": self.export_history.to_dict(),
            "general": self.general.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def definitions() -> dict[str, str]:
    """The rules behind every number, for the JSON of any report that has observations."""
    return {
        "population": "patients alive at the end of the simulation (DEATHDATE empty), or those "
        "of the cohort when one is given",
        "value": VALUE_RULE,
        "values_used": "rows with TYPE numeric whose VALUE is a finite number; every other row "
        "is counted, never dropped silently",
        "units": "each UNITS is described separately; values are never converted",
        "percentiles": PERCENTILE_METHOD,
        "percentiles_minimum_patients": f"the 5th, 25th, 75th and 95th percentiles are given "
        f"only with at least {MIN_PATIENTS_FOR_PERCENTILES} patients",
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), VALUE_DECIMALS)


def _sorted(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sorted(item) for item in value]
    return value
