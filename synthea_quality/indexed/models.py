"""[INDEXED] Results of the indexed cohort: counts of distinct patients, never rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synthea_quality.indexed.definitions import Window
from synthea_quality.observations.models import ValueSummary
from synthea_quality.prevalence.models import Rate
from synthea_quality.profile.models import SectionStatus

#: Label of the sex stratum of a patient whose ``GENDER`` is empty.
UNKNOWN_SEX = "unknown sex"


def _window_dict(window: Window) -> dict[str, int | None]:
    """The window as JSON; :class:`Window` lives in definitions and has no ``to_dict``."""
    return {"from_days": window.from_days, "to_days": window.to_days}



@dataclass(frozen=True, slots=True)
class Stratum:
    """One numerator/denominator cell; both counts are distinct patients."""

    #: "all", a sex value, :data:`UNKNOWN_SEX`, an age band or ``"unknown age"``.
    label: str
    #: Distinct patients of the stratum with the outcome.
    patients: int
    #: Distinct patients of the cohort in the stratum.
    cohort: int

    def to_dict(self) -> dict[str, Any]:
        """The cell as JSON: its label and the rate with its Wilson interval."""
        return {"label": self.label, **Rate(self.patients, self.cohort).to_dict()}


@dataclass(frozen=True, slots=True)
class Breakdown:
    """The outcome in total, by sex and by age band at the index date."""

    total: Stratum
    by_sex: tuple[Stratum, ...]
    by_age: tuple[Stratum, ...]

    def to_dict(self) -> dict[str, Any]:
        """The total and every stratum as JSON lists, in their stable order."""
        return {
            "total": self.total.to_dict(),
            "by_sex": [stratum.to_dict() for stratum in self.by_sex],
            "by_age": [stratum.to_dict() for stratum in self.by_age],
        }


@dataclass(frozen=True, slots=True)
class EventResult:
    """Patients of the cohort with one record of the codes within the window.

    The ``rows_*`` counts are records, not patients: they say what was left out on the
    way to the distinct patients of :attr:`breakdown`. ``days`` describes one value per
    patient, the days from the index date to their first record of the window.
    """

    name: str
    table: str
    codes: tuple[str, ...]
    window: Window
    breakdown: Breakdown
    days: ValueSummary
    #: Records of the codes of patients of the cohort, any date.
    rows_matched: int
    #: Of those, records whose date is empty or unparseable.
    rows_date_unusable: int
    #: Of those, records dated after the reference date.
    rows_after_reference: int
    #: Of those, records with a usable date, not after the reference date and outside
    #: the window.
    rows_outside_window: int

    def to_dict(self) -> dict[str, Any]:
        """The event outcome as JSON: patients by stratum and the records left out."""
        return {
            "type": "event",
            "status": SectionStatus.COMPUTED.value,
            "name": self.name,
            "table": self.table,
            "codes": list(self.codes),
            "window": _window_dict(self.window),
            "breakdown": self.breakdown.to_dict(),
            "days_to_first_record": self.days.to_dict(),
            "rows_matched": self.rows_matched,
            "rows_date_unusable": self.rows_date_unusable,
            "rows_after_reference": self.rows_after_reference,
            "rows_outside_window": self.rows_outside_window,
        }


@dataclass(frozen=True, slots=True)
class ValueShare:
    """The patients of one VALUE text on the date the outcome picks."""

    #: The ``VALUE`` text of the observation, as recorded.
    value: str
    #: Distinct patients with that VALUE on the picked date, never rows.
    patients: int

    def to_dict(self) -> dict[str, Any]:
        """The share as JSON: the VALUE text and its distinct patients."""
        return {"value": self.value, "patients": self.patients}


@dataclass(frozen=True, slots=True)
class ValueResult:
    """The values of a code on the first or last date of the window.

    A patient picks one date; every row of that patient on that date contributes its
    VALUE. Two distinct values on the same date (the sample's two stage rows) put the
    patient in both shares, so :attr:`values` may add up to more than
    :attr:`breakdown`; :attr:`patients_with_several_values` says how many do. The
    ``rows_*`` counts are observations, not patients: they say what was left out on the
    way to the distinct patients.
    """

    name: str
    code: str
    pick: str
    window: Window
    breakdown: Breakdown
    #: Most patients first; ties by the VALUE text.
    values: tuple[ValueShare, ...]
    #: Patients with more than one distinct VALUE on the picked date.
    patients_with_several_values: int
    #: Observations of the code of patients of the cohort, any date.
    rows_matched: int
    #: Of those, observations whose date is empty or unparseable.
    rows_date_unusable: int
    #: Of those, observations dated after the reference date.
    rows_after_reference: int
    #: Of those, observations with a usable date, not after the reference date and
    #: outside the window.
    rows_outside_window: int
    #: Of those in the window, observations with an empty VALUE; they do not count.
    rows_value_empty: int

    def to_dict(self) -> dict[str, Any]:
        """The value outcome as JSON: the shares, the strata and the observations left out."""
        return {
            "type": "value",
            "status": SectionStatus.COMPUTED.value,
            "name": self.name,
            "code": self.code,
            "pick": self.pick,
            "window": _window_dict(self.window),
            "breakdown": self.breakdown.to_dict(),
            "values": [share.to_dict() for share in self.values],
            "patients_with_several_values": self.patients_with_several_values,
            "rows_matched": self.rows_matched,
            "rows_date_unusable": self.rows_date_unusable,
            "rows_after_reference": self.rows_after_reference,
            "rows_outside_window": self.rows_outside_window,
            "rows_value_empty": self.rows_value_empty,
        }


@dataclass(frozen=True, slots=True)
class DeathResult:
    """Deaths of the cohort within the window and, optionally, their recorded cause.

    The cause is the ``VALUE`` text of observation ``69453-9``, a description rather
    than a code. When the outcome names ``cause_values``, the three cause counts always
    come together and add up to :attr:`breakdown`'s patients: a death with the cause, a
    death with another cause and a death with no cause record. All counts are patients,
    never rows; ``days`` is one value per deceased patient.
    """

    name: str
    cause_values: tuple[str, ...]
    window: Window
    breakdown: Breakdown
    days: ValueSummary
    #: Patients whose ``DEATHDATE`` is after the reference date; they do not count.
    deaths_after_reference: int
    #: Patients with a usable ``DEATHDATE`` outside the window, not after the reference.
    deaths_outside_window: int
    #: Patients whose ``DEATHDATE`` is present but unparseable; they do not count.
    deathdate_unusable: int
    #: Deaths with a ``69453-9`` VALUE in ``cause_values``; ``None`` without a cause.
    with_cause: int | None
    #: Deaths with a ``69453-9`` VALUE, none of them in ``cause_values``.
    other_cause: int | None
    #: Deaths without any ``69453-9`` record.
    no_cause_record: int | None

    def to_dict(self) -> dict[str, Any]:
        """The death outcome as JSON: deaths by stratum, days and cause counts."""
        return {
            "type": "death",
            "status": SectionStatus.COMPUTED.value,
            "name": self.name,
            "cause_values": list(self.cause_values),
            "window": _window_dict(self.window),
            "breakdown": self.breakdown.to_dict(),
            "days_to_death": self.days.to_dict(),
            "deaths_after_reference": self.deaths_after_reference,
            "deaths_outside_window": self.deaths_outside_window,
            "deathdate_unusable": self.deathdate_unusable,
            "with_cause": self.with_cause,
            "other_cause": self.other_cause,
            "no_cause_record": self.no_cause_record,
        }


@dataclass(frozen=True, slots=True)
class SkippedOutcome:
    """An outcome that could not be measured, such as one whose table is missing.

    ``reason`` is never empty: a section left out without its reason would hide what
    the dataset does not say.
    """

    name: str
    #: "event", "value" or "death": the kind the module file asked for.
    type: str
    #: Why the outcome could not be measured; never empty.
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("a skipped outcome needs a reason")

    def to_dict(self) -> dict[str, Any]:
        """The skipped outcome as JSON: its kind, name and reason."""
        return {
            "type": self.type,
            "status": SectionStatus.SKIPPED.value,
            "name": self.name,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class IndexedCohortResult:
    """The indexed cohort: its size, the records left out and every outcome.

    All patient counts are distinct patients, alive or deceased. The ``rows_*`` counts
    are records of the index condition, not patients: they say what was left out on the
    way to the cohort.
    """

    #: Name of the condition whose earliest ``START`` is a patient's index date.
    index_condition: str
    #: The reference date, ``YYYY-MM-DD``, or None without a reference date.
    reference_date: str | None
    #: Distinct patients of the cohort, alive and deceased, each counted once.
    patients: int
    #: Records of the index condition, any date.
    rows_matched: int
    #: Of those, records of a ``PATIENT`` absent from ``patients``.
    rows_unknown_patient: int
    #: Of those, records with an empty or unparseable ``START``.
    rows_start_unusable: int
    #: Of those, records with a ``START`` after the reference date.
    rows_after_reference: int
    outcomes: tuple[EventResult | ValueResult | DeathResult | SkippedOutcome, ...]
    #: Why the whole cohort could not be built; ``None`` when it was.
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The cohort and every outcome as JSON, in the order the module file gives."""
        status = SectionStatus.SKIPPED if self.reason is not None else SectionStatus.COMPUTED
        return {
            "index_condition": self.index_condition,
            "reference_date": self.reference_date,
            "patients": self.patients,
            "rows_matched": self.rows_matched,
            "rows_unknown_patient": self.rows_unknown_patient,
            "rows_start_unusable": self.rows_start_unusable,
            "rows_after_reference": self.rows_after_reference,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "status": status.value,
            "reason": self.reason,
        }
