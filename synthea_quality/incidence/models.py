"""[INCIDENCE] Structured result of an incidence run.

Same principles as the profile and the prevalence: ``COMPUTED`` or ``SKIPPED`` with the
reason, provenance with the result, and a deterministic, versioned JSON layout
(``INCIDENCE_SCHEMA_VERSION``, independent of the other reports). Rates are expressed per
1,000 person-years; person-years are rounded to six decimals, so two runs over the same
input produce the same text apart from ``generated_at``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synthea_quality import __version__
from synthea_quality.incidence.poisson import exact_interval
from synthea_quality.models import utc_now_iso
from synthea_quality.prevalence.models import CodeRef, ExpectedComparison
from synthea_quality.profile.models import ReferenceDate, SectionStatus, TableInput

#: Version of the JSON layout produced by :meth:`IncidenceReport.to_dict`.
INCIDENCE_SCHEMA_VERSION = 1

#: Rates are per this many person-years.
PER_PERSON_YEARS = 1000
#: Person-time is counted in whole days and converted once, so strata add up exactly.
DAYS_PER_YEAR = 365.25
DECIMALS = 6


@dataclass(frozen=True, slots=True)
class IncidenceRate:
    """``events`` first events over ``person_days`` of time at risk."""

    events: int
    person_days: int

    def __post_init__(self) -> None:
        if self.events < 0 or self.person_days < 0:
            raise ValueError("events and person-time must be non-negative")

    @property
    def person_years(self) -> float:
        return round(self.person_days / DAYS_PER_YEAR, DECIMALS)

    @property
    def value(self) -> float | None:
        """Events per 1,000 person-years, or ``None`` without person-time."""
        if self.person_days == 0:
            return None
        return round(self.events / self.person_days * DAYS_PER_YEAR * PER_PERSON_YEARS, DECIMALS)

    @property
    def interval(self) -> tuple[float, float] | None:
        """Exact 95% Poisson interval per 1,000 person-years, or ``None`` without person-time."""
        if self.person_days == 0:
            return None
        low, high = exact_interval(self.events)
        scale = DAYS_PER_YEAR * PER_PERSON_YEARS / self.person_days
        return round(low * scale, DECIMALS), round(high * scale, DECIMALS)

    def to_dict(self) -> dict[str, Any]:
        interval = self.interval
        return {
            "events": self.events,
            "person_days": self.person_days,
            "person_years": self.person_years,
            "per_1000_person_years": self.value,
            "ci95_low": interval[0] if interval else None,
            "ci95_high": interval[1] if interval else None,
        }


@dataclass(frozen=True, slots=True)
class IncidenceStratum:
    dimension: str
    value: str | None
    rate: IncidenceRate

    def to_dict(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "value": self.value, **self.rate.to_dict()}


@dataclass(frozen=True, slots=True)
class ConditionIncidence:
    """Incidence of one condition asked for."""

    name: str
    codes: tuple[CodeRef, ...]
    status: SectionStatus
    reason: str | None = None
    acute: bool = False
    rate: IncidenceRate | None = None
    strata: tuple[IncidenceStratum, ...] = ()
    expected: tuple[ExpectedComparison, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is SectionStatus.SKIPPED:
            if not self.reason or self.rate is not None or self.strata:
                raise ValueError(f"skipped condition {self.name!r} needs a reason and no numbers")
        elif self.reason is not None or self.rate is None:
            raise ValueError(f"computed condition {self.name!r} needs a rate and no reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "codes": [code.to_dict() for code in self.codes],
            "acute": self.acute,
            "status": self.status.value,
            "reason": self.reason,
            "rate": self.rate.to_dict() if self.rate else None,
            "strata": [stratum.to_dict() for stratum in self.strata],
            "expected": [item.to_dict() for item in self.expected],
            "metrics": _sorted(self.metrics),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class Window:
    """The observation window ``[start, end]`` (ISO dates) and its length in years."""

    years: int
    start: str
    end: str

    def to_dict(self) -> dict[str, Any]:
        return {"years": self.years, "start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class IncidenceReport:
    """The complete result of one incidence run."""

    data_dir: str
    reference_date: ReferenceDate | None
    reference_reason: str | None
    window: Window | None
    #: ``"all"`` (deceased included until death) or ``"alive"`` (``--alive-only``).
    population: str
    age_bands: tuple[int, ...]
    conditions: tuple[ConditionIncidence, ...]
    #: What is known about the exported history (``exporter.years_of_history``, earliest date).
    history: dict[str, Any] = field(default_factory=dict)
    cohort: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[TableInput, ...] = ()
    notes: tuple[str, ...] = ()
    generated_at: str = field(default_factory=utc_now_iso)
    tool_version: str = __version__

    @property
    def incomplete(self) -> bool:
        return any(item.state.value == "unreadable" for item in self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INCIDENCE_SCHEMA_VERSION,
            "tool_version": self.tool_version,
            "generated_at": self.generated_at,
            "data_dir": self.data_dir,
            "reference_date": self.reference_date.to_dict() if self.reference_date else None,
            "reference_reason": self.reference_reason,
            "window": self.window.to_dict() if self.window else None,
            "population": self.population,
            "age_bands": list(self.age_bands),
            "definitions": {
                "rate": "first events in the window over person-years at risk, per 1,000",
                "at_risk": "from the later of the window start and birth, to the earliest of "
                "the reference date, death and the first event; patients with a record of the "
                "condition before the window are prior cases and are left out",
                "interval": "exact (Garwood) 95% Poisson interval",
            },
            "history": _sorted(self.history),
            "cohort": _sorted(self.cohort),
            "inputs": [item.to_dict() for item in self.inputs],
            "notes": list(self.notes),
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _sorted(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sorted(item) for item in value]
    return value
