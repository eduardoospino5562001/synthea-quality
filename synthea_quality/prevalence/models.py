"""[PREVALENCE] Structured result of a prevalence run, and the Wilson interval.

The same principles as the profile models: a result is ``COMPUTED`` or ``SKIPPED`` with
its reason, provenance travels with it, and :meth:`PrevalenceReport.to_dict` is
deterministic (fixed key order, rounded floats), so two runs over the same input differ
only in ``generated_at``. The JSON layout is versioned by ``PREVALENCE_SCHEMA_VERSION``,
independently of the quality report and of the profile.

Why Wilson
----------
The normal-approximation interval ``p ± z·sqrt(p(1-p)/n)`` collapses to a single point
at 0 or at n patients and can leave ``[0, 1]`` — exactly the cases that matter for a
rare condition in a small synthetic population. The Wilson score interval stays inside
``[0, 1]``, is never empty and behaves well for small ``n``. With a denominator of 0
there is no rate and no interval: both are ``None``, never a made-up 0.

Only the standard library is imported here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from synthea_quality import __version__
from synthea_quality.models import utc_now_iso
from synthea_quality.profile.models import ReferenceDate, SectionStatus, TableInput

#: Version of the JSON layout produced by :meth:`PrevalenceReport.to_dict`.
PREVALENCE_SCHEMA_VERSION = 1

#: Two-sided 95% normal quantile.
Z_95 = 1.959963984540054

#: Decimal places kept for rates and interval bounds in the structured result.
RATE_DECIMALS = 6

MEASURES = ("point", "lifetime")


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for ``successes`` out of ``total``, or ``None`` when ``total`` is 0."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(f"invalid proportion {successes}/{total}")
    if total == 0:
        return None
    p = successes / total
    z2 = z * z
    centre = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = (z / (1 + z2 / total)) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    # At 0 or at ``total`` successes one bound is exactly 0 or 1; say so rather than let
    # floating point leave it at 0.9999999999999999.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == total else min(1.0, centre + half)
    return low, high


@dataclass(frozen=True, slots=True)
class Rate:
    """``numerator`` patients out of ``denominator``, with its 95% Wilson interval."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator < 0 or not 0 <= self.numerator <= max(self.denominator, 0):
            raise ValueError(f"invalid rate {self.numerator}/{self.denominator}")

    @property
    def value(self) -> float | None:
        if self.denominator == 0:
            return None
        return round(self.numerator / self.denominator, RATE_DECIMALS)

    @property
    def interval(self) -> tuple[float, float] | None:
        bounds = wilson_interval(self.numerator, self.denominator)
        if bounds is None:
            return None
        return round(bounds[0], RATE_DECIMALS), round(bounds[1], RATE_DECIMALS)

    def to_dict(self) -> dict[str, Any]:
        interval = self.interval
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.value,
            "ci95_low": interval[0] if interval else None,
            "ci95_high": interval[1] if interval else None,
        }


@dataclass(frozen=True, slots=True)
class CodeRef:
    """A condition code; ``system`` ``None`` matches the code in any system."""

    code: str
    system: str | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("a code must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"system": self.system, "code": self.code}


@dataclass(frozen=True, slots=True)
class Expected:
    """A reference value to show next to an observed rate. It is never a verdict."""

    measure: str
    value: float
    source: str | None = None

    def __post_init__(self) -> None:
        if self.measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}, not {self.measure!r}")
        if not (isinstance(self.value, (int, float)) and 0.0 <= float(self.value) <= 1.0):
            raise ValueError(f"an expected {self.measure} prevalence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"measure": self.measure, "value": float(self.value), "source": self.source}


@dataclass(frozen=True, slots=True)
class ExpectedComparison:
    """An expected value next to the observed rate: the difference and its position."""

    expected: Expected
    observed: Rate

    @property
    def difference(self) -> float | None:
        observed = self.observed.value
        if observed is None:
            return None
        return round(observed - float(self.expected.value), RATE_DECIMALS)

    @property
    def position(self) -> str | None:
        """``"inside"`` or ``"outside"`` the observed rate's 95% CI; ``None`` without one."""
        interval = self.observed.interval
        if interval is None:
            return None
        low, high = interval
        return "inside" if low <= float(self.expected.value) <= high else "outside"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.expected.to_dict(),
            "observed": self.observed.value,
            "difference": self.difference,
            "position_in_ci95": self.position,
        }


@dataclass(frozen=True, slots=True)
class Stratum:
    """Point and lifetime prevalence within one value of a dimension."""

    dimension: str
    #: ``None`` for patients whose value is empty (``GENDER``) — never merged into another.
    value: str | None
    point: Rate
    lifetime: Rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "value": self.value,
            "point": self.point.to_dict(),
            "lifetime": self.lifetime.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ConditionResult:
    """Prevalence of one condition asked for (one or several codes under a name)."""

    name: str
    codes: tuple[CodeRef, ...]
    status: SectionStatus
    reason: str | None = None
    point: Rate | None = None
    lifetime: Rate | None = None
    strata: tuple[Stratum, ...] = ()
    expected: tuple[ExpectedComparison, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is SectionStatus.SKIPPED:
            if not self.reason or self.point or self.lifetime or self.strata:
                raise ValueError(f"skipped condition {self.name!r} needs a reason and no numbers")
        elif self.reason is not None or self.point is None or self.lifetime is None:
            raise ValueError(f"computed condition {self.name!r} needs both rates and no reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "codes": [code.to_dict() for code in self.codes],
            "status": self.status.value,
            "reason": self.reason,
            "point": self.point.to_dict() if self.point else None,
            "lifetime": self.lifetime.to_dict() if self.lifetime else None,
            "strata": [stratum.to_dict() for stratum in self.strata],
            "expected": [item.to_dict() for item in self.expected],
            "metrics": _sorted(self.metrics),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class GeneralRow:
    """One code of the general table."""

    system: str | None
    code: str
    description: str | None
    point: Rate
    lifetime: Rate
    #: True when the code is on the social and administrative list.
    social: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "code": self.code,
            "description": self.description,
            "social": self.social,
            "point": self.point.to_dict(),
            "lifetime": self.lifetime.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GeneralTable:
    """Every condition code among the alive, ordered by point prevalence."""

    status: SectionStatus
    reason: str | None = None
    rows: tuple[GeneralRow, ...] = ()
    #: How many rows the Markdown lists (the JSON always carries every row).
    top: int = 30
    include_social: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "top": self.top,
            "include_social": self.include_social,
            "metrics": _sorted(self.metrics),
            "notes": list(self.notes),
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class PrevalenceReport:
    """The complete result of one prevalence run."""

    data_dir: str
    reference_date: ReferenceDate | None
    reference_reason: str | None
    age_bands: tuple[int, ...]
    alive: int | None
    conditions: tuple[ConditionResult, ...]
    general: GeneralTable
    social_list: dict[str, Any]
    inputs: tuple[TableInput, ...] = ()
    notes: tuple[str, ...] = ()
    generated_at: str = field(default_factory=utc_now_iso)
    tool_version: str = __version__

    @property
    def incomplete(self) -> bool:
        return any(item.state.value == "unreadable" for item in self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREVALENCE_SCHEMA_VERSION,
            "tool_version": self.tool_version,
            "generated_at": self.generated_at,
            "data_dir": self.data_dir,
            "reference_date": self.reference_date.to_dict() if self.reference_date else None,
            "reference_reason": self.reference_reason,
            "age_bands": list(self.age_bands),
            "alive": self.alive,
            "definitions": {
                "point": "alive patients with a record whose START <= reference date and whose "
                "STOP is empty or after the reference date, over the alive patients",
                "lifetime": "alive patients with a record whose START <= reference date, over "
                "the alive patients",
                "interval": "95% Wilson score interval",
            },
            "inputs": [item.to_dict() for item in self.inputs],
            "notes": list(self.notes),
            "social_list": _sorted(self.social_list),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "general": self.general.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _sorted(value: Any) -> Any:
    """A copy of ``value`` with every mapping's keys sorted, for deterministic JSON."""
    if isinstance(value, dict):
        return {key: _sorted(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sorted(item) for item in value]
    return value
