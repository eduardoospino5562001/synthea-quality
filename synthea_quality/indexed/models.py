"""[INDEXED] Results of the indexed cohort: counts of distinct patients, never rows.

An event of several parts (:class:`EventMatchResult`) keeps the rule of every part and
what each of them found, so the report says which rule selected which patients.

A cross of two outcomes (:class:`CrossResult`) is an intersection of patients, so its
numerator can never be larger than its denominator: the denominator is the cohort of
the stratum and the numerator the patients of the crossed outcome inside it. A cross
that could not be computed is ``SKIPPED`` with its reason, like every other result of
the tool; it never shows a made-up number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synthea_quality.indexed.definitions import DEFAULT_OP, Window
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
class EventMatchResult:
    """One part of an event outcome: the patients it selects and the records left out.

    The part is the rule itself (its table, its codes and its filters) plus what it
    found: its distinct patients and the records that did not count. Every record left
    out is counted in exactly one of the ``rows_*`` counters, so the counters add up to
    the records of the part.
    """

    #: The label of the part in the report.
    name: str
    #: The table the part reads.
    table: str
    #: The codes that select a record.
    codes: tuple[str, ...]
    #: The exact ``VALUE`` texts a record must have.
    values: tuple[str, ...]
    #: The exact ``REASONCODE`` codes a record must have.
    reasons: tuple[str, ...]
    #: When the record counts: on its ``START`` or on its ``STOP`` date.
    at: str
    #: Distinct patients the part selects, never rows.
    patients: int
    #: Records of the codes of the part, of the patients of the cohort, any date.
    rows_matched: int
    #: Of those, records left out by the filters of the part.
    rows_filtered_out: int
    #: Of those, records without a ``STOP``: the part counts on their end date.
    rows_not_ended: int
    #: Of those, records whose date is empty or unparseable.
    rows_date_unusable: int
    #: Of those, records dated after the reference date.
    rows_after_reference: int
    #: Of those, records with a usable date, not after the reference date and outside
    #: the window.
    rows_outside_window: int

    def to_dict(self) -> dict[str, Any]:
        """The part as JSON: its rule, its patients and the records left out."""
        return {
            "name": self.name,
            "table": self.table,
            "codes": list(self.codes),
            "values": list(self.values),
            "reasons": list(self.reasons),
            "at": self.at,
            "patients": self.patients,
            "rows_matched": self.rows_matched,
            "rows_filtered_out": self.rows_filtered_out,
            "rows_not_ended": self.rows_not_ended,
            "rows_date_unusable": self.rows_date_unusable,
            "rows_after_reference": self.rows_after_reference,
            "rows_outside_window": self.rows_outside_window,
        }


@dataclass(frozen=True, slots=True)
class EventResult:
    """Patients of the cohort with one record of the codes within the window.

    The ``rows_*`` counts are records, not patients: they say what was left out on the
    way to the distinct patients of :attr:`breakdown`. They are the sum of the counters
    of the parts of :attr:`matches`. ``days`` describes one value per patient, the days
    from the index date to the outcome: with ``any`` the earliest of its parts, with
    ``all`` the latest, the day the last part is met.

    An outcome of one table has no ``table`` of its own in the report of a composite
    one: a composite outcome has no table and no codes, and its parts say everything.
    """

    name: str
    table: str | None
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
    #: Of those, records left out by the filters of the parts.
    rows_filtered_out: int = 0
    #: Of those, records without a ``STOP``: a part counts on its end date.
    rows_not_ended: int = 0
    #: How the parts combine: ``any`` or ``all``.
    op: str = DEFAULT_OP
    #: The parts of the outcome, with what each of them found.
    matches: tuple[EventMatchResult, ...] = ()

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
            "rows_filtered_out": self.rows_filtered_out,
            "rows_not_ended": self.rows_not_ended,
            "op": self.op,
            "matches": [match.to_dict() for match in self.matches],
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
class CrossStratum:
    """One cell of a cross: the patients of the outcome inside a stratum of the other.

    Both counts are distinct patients: the denominator is the cohort of the stratum and
    the numerator the patients of the crossed outcome inside it. ``with_cause`` counts
    the deaths with the recorded cause of the crossed outcome, and is ``None`` when the
    crossed outcome is not a death with a cause.
    """

    #: The label of the stratum: a value of the crossed outcome, ``"no value"`` or
    #: ``"with <outcome>"`` and ``"without <outcome>"`` for an event.
    label: str
    #: Distinct patients of the stratum with the crossed outcome.
    numerator: int
    #: Distinct patients of the cohort in the stratum.
    denominator: int
    #: Deaths with the cause, when the crossed outcome is a death with a cause.
    with_cause: int | None

    def to_dict(self) -> dict[str, Any]:
        """The cell as JSON: its label, the rate with its interval and the cause."""
        return {
            "label": self.label,
            **Rate(self.numerator, self.denominator).to_dict(),
            "with_cause": self.with_cause,
        }


@dataclass(frozen=True, slots=True)
class CrossResult:
    """One crossing of the ``by`` list: an outcome measured in the strata of another.

    ``by_type`` is the kind of the outcome that gives the strata: ``"value"`` gives one
    stratum per value (and a last one for the patients without a value) and ``"event"``
    gives two, with and without the event. The strata are in the order of the values of
    the crossed outcome, and a patient with more than one value counts in each of them:
    ``patients_in_several_strata`` says how many do.
    """

    #: Name of the outcome that is measured.
    outcome: str
    #: Name of the outcome that gives the strata.
    by: str
    #: "value" or "event": the kind of the outcome that gives the strata.
    by_type: str
    #: The strata of the cross, in their stable order.
    strata: tuple[CrossStratum, ...]
    #: Patients under more than one stratum of the crossed outcome.
    patients_in_several_strata: int
    #: Why the cross could not be computed; ``None`` when it could.
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The cross as JSON: its strata and, when it was skipped, its reason."""
        status = SectionStatus.SKIPPED if self.reason is not None else SectionStatus.COMPUTED
        return {
            "outcome": self.outcome,
            "by": self.by,
            "by_type": self.by_type,
            "status": status.value,
            "strata": [stratum.to_dict() for stratum in self.strata],
            "patients_in_several_strata": self.patients_in_several_strata,
            "reason": self.reason,
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
    way to the cohort. The ``crosses`` measure an outcome inside the strata of another
    one, as the module file's ``by`` list asks.
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
    #: The crossings of the module file's ``by`` list, in the order of that list.
    crosses: tuple[CrossResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """The cohort, every outcome and every crossing as JSON, in the file's order."""
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
            "crosses": [cross.to_dict() for cross in self.crosses],
            "status": status.value,
            "reason": self.reason,
        }
