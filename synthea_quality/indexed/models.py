"""[INDEXED] Results of the indexed cohort: counts of distinct patients, never rows."""

from __future__ import annotations

from dataclasses import dataclass

from synthea_quality.indexed.definitions import Window
from synthea_quality.observations.models import ValueSummary

#: Label of the sex stratum of a patient whose ``GENDER`` is empty.
UNKNOWN_SEX = "unknown sex"


@dataclass(frozen=True, slots=True)
class Stratum:
    """One numerator/denominator cell; both counts are distinct patients."""

    #: "all", a sex value, :data:`UNKNOWN_SEX`, an age band or ``"unknown age"``.
    label: str
    #: Distinct patients of the stratum with the outcome.
    patients: int
    #: Distinct patients of the cohort in the stratum.
    cohort: int


@dataclass(frozen=True, slots=True)
class Breakdown:
    """The outcome in total, by sex and by age band at the index date."""

    total: Stratum
    by_sex: tuple[Stratum, ...]
    by_age: tuple[Stratum, ...]


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
