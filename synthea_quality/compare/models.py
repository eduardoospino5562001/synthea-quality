"""[COMPARE] The structured result of comparing two module validations.

A :class:`Figure` is one measured value on one side: its value, its 95%
interval when it has one, its numerator and denominator (events and
person-time for incidence), its unit and whether it was measured or skipped.
A skipped figure keeps its reason and no numbers: an absent measurement is
never read as zero.

A :class:`Comparison` pairs the figure of the baseline with the figure of the
candidate under one stable key (block, element, measure, stratum). The
difference is ``candidate - baseline`` in the same unit as the figure — a
proportion stays a proportion; the render turns it into percentage points
(S2). The relative difference is ``(candidate - baseline) / baseline`` and is
``None`` when the baseline is missing or zero, never an infinity.

Only the standard library is imported here, so importing this package stays
cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A figure whose source could not be measured; it carries a reason and no numbers.
SKIPPED = "skipped"
#: A figure that was measured.
COMPUTED = "computed"

#: Two figures on the two sides with the same value.
UNCHANGED = "unchanged"
#: Two figures on the two sides with different values.
CHANGED = "changed"
#: A figure found on one side only.
ONLY_IN_BASELINE = "only_in_baseline"
ONLY_IN_CANDIDATE = "only_in_candidate"

#: Units in which a figure can be expressed.
PROPORTION = "proportion"
PER_1000_PERSON_YEARS = "per_1000_person_years"
COUNT = "count"
#: A percentage of a population (for example ``alive_percent`` of the
#: population section); read in percentage points in the report (S2).
PERCENT = "percent"
#: Days from the index date to an outcome (median of the days per patient
#: of the indexed cohort).
DAYS = "days"


@dataclass(frozen=True, slots=True, order=True)
class ComparisonKey:
    """The stable key of one figure: block, element, measure and stratum.

    Two figures that differ in any of these four parts are two different
    figures: for example the point and the lifetime prevalence of the same
    condition, or the same patients cut by sex and by age, never share a key.
    The order is the stable order of the "largest changes" table.
    """

    block: str
    #: Condition, medication, observation or outcome name, metric or section id.
    element: str
    #: ``point``, ``lifetime``, ``incidence``, ``active``, ``ever``, a statistic
    #: of an observation summary, ``patients``, ``count``, ``cross`` ...
    measure: str
    #: ``sex:F``, ``age:18-64``, ``units:mm[Hg]|age:65-74`` ... empty for totals.
    stratum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "element": self.element,
            "measure": self.measure,
            "stratum": self.stratum,
        }


@dataclass(frozen=True, slots=True)
class Figure:
    """One measured value on one side of the comparison.

    ``unit`` says how to read ``value``: a proportion, a rate per 1,000
    person-years or a plain count. ``ci95_low``/``ci95_high`` are the 95%
    interval of a rate; ``numerator`` and ``denominator`` are the counts behind
    it (events and person-days for incidence). ``state`` is ``"computed"`` when
    a number exists or ``"skipped"`` when the source could not be measured; a
    skipped figure keeps ``reason`` and no numbers, never a made-up zero.
    """

    unit: str
    value: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    numerator: int | None = None
    denominator: int | None = None
    state: str = COMPUTED
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state == SKIPPED:
            if not self.reason or not self.reason.strip():
                raise ValueError(f"a skipped figure needs a reason: {self!r}")
            if (
                self.value is not None
                or self.ci95_low is not None
                or self.ci95_high is not None
                or self.numerator is not None
                or self.denominator is not None
            ):
                raise ValueError(f"a skipped figure carries no numbers: {self!r}")
        elif self.reason is not None:
            raise ValueError(f"a computed figure carries no reason: {self!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "value": self.value,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "state": self.state,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Comparison:
    """One figure measured on both sides (or only one side) of the comparison.

    ``difference`` is ``candidate - baseline`` in the same unit as the figure;
    it is ``None`` when either figure has no value. ``relative_difference`` is
    ``(candidate - baseline) / baseline`` and is ``None`` when the baseline is
    missing or zero, never an infinity. ``state`` is ``"changed"``,
    ``"unchanged"``, ``"only_in_baseline"``, ``"only_in_candidate"`` or
    ``"skipped"`` with its reason.
    """

    key: ComparisonKey
    unit: str
    baseline: Figure | None
    candidate: Figure | None
    difference: float | None
    relative_difference: float | None
    state: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "unit": self.unit,
            "baseline": (self.baseline.to_dict() if self.baseline is not None else None),
            "candidate": (self.candidate.to_dict() if self.candidate is not None else None),
            "difference": self.difference,
            "relative_difference": self.relative_difference,
            "state": self.state,
            "reason": self.reason,
        }
