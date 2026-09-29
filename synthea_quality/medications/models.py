"""[MEDICATIONS] Structured result of the medications of a cohort.

The same principles as the other reports: a result is ``COMPUTED`` or ``SKIPPED`` with
its reason, provenance travels with it, and ``to_dict`` is deterministic. The shares are
:class:`~synthea_quality.prevalence.models.Rate` values, so they carry the same 95% Wilson
interval as prevalence, and an expected share is an
:class:`~synthea_quality.prevalence.models.ExpectedComparison`: inside or outside the CI,
never a verdict.

Only the standard library is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.prevalence.models import ExpectedComparison, Rate
from synthea_quality.profile.models import SectionStatus

#: The rules behind every share, for the JSON of a report with medications.
DEFINITIONS = {
    "active": "patients of the cohort with a record of any of the codes whose START is on or "
    "before the reference date and whose STOP is empty or after it (compared by calendar "
    "day), over the patients of the cohort",
    "ever": "patients of the cohort with a record of any of the codes whose START is on or "
    "before the reference date, over the patients of the cohort",
    "with_reason": "those of them with such a record whose REASONCODE is one of the cohort "
    "condition's codes",
    "cohort": "alive patients with the cohort's condition at the reference date, or every "
    "alive patient when the medication has no cohort",
    "interval": "95% Wilson score interval",
}


@dataclass(frozen=True, slots=True)
class MedicationResult:
    """The share of a cohort with one medication (one or several codes under a name)."""

    name: str
    codes: tuple[str, ...]
    status: SectionStatus
    reason: str | None = None
    cohort: CohortSpec | None = None
    #: Most frequent description of each code found in the table.
    descriptions: dict[str, str | None] = field(default_factory=dict)
    active: Rate | None = None
    ever: Rate | None = None
    #: Patients counted in ``active`` / ``ever`` with a record whose REASONCODE is one of
    #: the cohort condition's codes; ``None`` without a cohort or a REASONCODE column.
    active_with_reason: int | None = None
    ever_with_reason: int | None = None
    expected: tuple[ExpectedComparison, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is SectionStatus.SKIPPED:
            if not self.reason or self.active or self.ever:
                raise ValueError(f"skipped medication {self.name!r} needs a reason and no shares")
        elif self.reason is not None or self.active is None or self.ever is None:
            raise ValueError(f"computed medication {self.name!r} needs both shares and no reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "codes": list(self.codes),
            "descriptions": {code: self.descriptions[code] for code in sorted(self.descriptions)},
            "cohort": self.cohort.to_dict() if self.cohort else None,
            "status": self.status.value,
            "reason": self.reason,
            "active": self.active.to_dict() if self.active else None,
            "ever": self.ever.to_dict() if self.ever else None,
            "active_with_reason": self.active_with_reason,
            "ever_with_reason": self.ever_with_reason,
            "expected": [item.to_dict() for item in self.expected],
            "metrics": {key: self.metrics[key] for key in sorted(self.metrics)},
            "notes": list(self.notes),
        }
