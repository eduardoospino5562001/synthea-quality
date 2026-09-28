"""[INCIDENCE] Time at risk, first events and rates, from frames that are already loaded.

The window is ``[ref − N years, ref]`` (a window start that would fall on a 29 February
in a common year falls on the 28th). For each patient of the population:

* **entry** is the later of the window start and the birth date;
* **exit** is the earliest of the reference date and the death date;
* a patient is **left out, and counted**, when the birth date is empty or unparseable,
  when they were born after the reference date, when they died before the window started
  (they contribute no time), when a present death date cannot be parsed (the exit is
  unknown), and — with ``--alive-only`` — when they died at all.

For each condition, among those patients:

* a **prior case** has a record of the condition whose ``START`` is before their entry;
  they are not at risk and are counted;
* the **first event** is the earliest ``START`` of any of the condition's codes between
  entry and exit; the time at risk ends there;
* **person-time** is the whole days between entry and exit (or the first event); it is
  converted to years (365.25 days) only when a rate is reported, so strata add up exactly;
* later records of an at-risk patient are not events (first events only) and are
  counted, since for an acute condition they may be new episodes.

Strata
------
By ``GENDER`` (an empty value is its own stratum) and by age band. Ages change during a
five-year window, so a patient's person-time is **split between the bands they pass
through**: the interval is cut at the birthdays that start each band (a 29 February
birthday falls on 1 March in a common year, as for the profile's ages), and each piece
counts in its band. An event counts in the band of the patient's age on the day it
happens. The simpler alternative, the age at the start of the window, would put five
years of a three-year-old in the ``0-4`` band.

A record whose ``START`` cannot be parsed cannot be placed in time: it is counted and
ignored. This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from synthea_quality.incidence.models import (
    ConditionIncidence,
    IncidenceRate,
    IncidenceStratum,
    Window,
)
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.prevalence.models import ExpectedComparison, normalise_system
from synthea_quality.profile.dates import parse_date_only
from synthea_quality.profile.models import SectionStatus
from synthea_quality.profile.population import (
    DEFAULT_AGE_BANDS,
    age_band_labels,
    validate_age_bands,
)

PATIENT_COLUMNS = ("Id", "BIRTHDATE", "DEATHDATE", "GENDER")
CONDITION_COLUMNS = ("PATIENT", "CODE", "START", "SYSTEM")
REQUIRED_PATIENT_COLUMNS = ("Id", "BIRTHDATE", "DEATHDATE")
REQUIRED_CONDITION_COLUMNS = ("PATIENT", "CODE", "START")

DEFAULT_WINDOW_YEARS = 5

POPULATION_ALL = "all"
POPULATION_ALIVE = "alive"

#: One line the report shows so that nobody coming from the profile or the prevalence
#: (which describe the patients alive at the end) is surprised by the deceased here.
WHY_ALL_PATIENTS = (
    "Incidence counts every patient, deceased included, until their death: restricting it to "
    "the patients alive at the end would drop the time and events of those who died during the "
    "window (survivor bias). Use --alive-only to measure that cohort instead."
)


def window_start(reference: date, years: int) -> date:
    """``reference`` minus ``years`` calendar years (29 February → 28 February)."""
    if years < 1:
        raise ValueError("the window must be at least one year")
    try:
        return reference.replace(year=reference.year - years)
    except ValueError:
        return reference.replace(year=reference.year - years, day=28)


@dataclass(frozen=True, slots=True)
class Person:
    birth: date
    entry: date
    exit: date
    gender: str | None


@dataclass(frozen=True, slots=True, eq=False)
class Cohort:
    """The patients whose time can be followed through the window."""

    window: Window
    population: str
    people: dict[str, Person]
    excluded: dict[str, int] = field(default_factory=dict)
    age_bands: tuple[int, ...] = DEFAULT_AGE_BANDS

    @property
    def size(self) -> int:
        return len(self.people)


def build_cohort(
    patients: pd.DataFrame,
    reference: date,
    *,
    window_years: int = DEFAULT_WINDOW_YEARS,
    population: str = POPULATION_ALL,
    age_bands: tuple[int, ...] = DEFAULT_AGE_BANDS,
) -> Cohort:
    """The followed population (text columns of :data:`PATIENT_COLUMNS`)."""
    if population not in (POPULATION_ALL, POPULATION_ALIVE):
        raise ValueError(f"population must be {POPULATION_ALL!r} or {POPULATION_ALIVE!r}")
    start = window_start(reference, window_years)
    window = Window(window_years, start.isoformat(), reference.isoformat())
    frame = patients.dropna(subset=["Id"]).drop_duplicates("Id")
    births = parse_date_only(frame["BIRTHDATE"]).values
    deaths = parse_date_only(frame["DEATHDATE"]).values
    death_missing = frame["DEATHDATE"].isna()
    genders = frame["GENDER"] if "GENDER" in frame.columns else pd.Series(None, index=frame.index)

    excluded = {
        "birthdate_unusable": 0,
        "born_after_reference": 0,
        "deathdate_unparseable": 0,
        "died_before_window": 0,
        "deceased_not_in_alive_only": 0,
    }
    people: dict[str, Person] = {}
    for index, patient_id in frame["Id"].items():
        birth = births[index]
        if pd.isna(birth):
            excluded["birthdate_unusable"] += 1
            continue
        birth = birth.date()
        if not death_missing[index] and population == POPULATION_ALIVE:
            excluded["deceased_not_in_alive_only"] += 1
            continue
        death = deaths[index]
        if not death_missing[index] and pd.isna(death):
            excluded["deathdate_unparseable"] += 1
            continue
        death = None if pd.isna(death) else death.date()
        if birth > reference:
            excluded["born_after_reference"] += 1
            continue
        if death is not None and death < start:
            excluded["died_before_window"] += 1
            continue
        gender = genders[index]
        people[str(patient_id)] = Person(
            birth=birth,
            entry=max(start, birth),
            exit=min(reference, death) if death is not None else reference,
            gender=None if pd.isna(gender) else str(gender),
        )
    return Cohort(
        window=window,
        population=population,
        people=people,
        excluded=excluded,
        age_bands=validate_age_bands(age_bands),
    )


@dataclass(frozen=True, slots=True, eq=False)
class Records:
    """Condition records placed in time: PATIENT, SYSTEM ("" when absent), CODE, start."""

    frame: pd.DataFrame
    has_system: bool
    start_unusable: int


def prepare_records(conditions: pd.DataFrame, cohort: Cohort) -> Records:
    """Keep the records of the followed patients that have a code, and parse their start."""
    has_system = "SYSTEM" in conditions.columns
    followed = conditions["PATIENT"].isin(list(cohort.people)) & conditions["CODE"].notna()
    rows = conditions[followed]
    starts = parse_date_only(rows["START"]).values
    frame = pd.DataFrame(
        {
            "PATIENT": rows["PATIENT"],
            "SYSTEM": (
                rows["SYSTEM"].map(normalise_system, na_action="ignore").fillna("")
                if has_system
                else ""
            ),
            "CODE": rows["CODE"],
            "start": starts,
        }
    )
    usable = frame["start"].notna()
    return Records(frame[usable], has_system, int((~usable).sum()))


def condition_incidence(
    definition: ConditionDefinition, records: Records, cohort: Cohort
) -> ConditionIncidence:
    """First events per 1,000 person-years of one condition, in total and by sex."""
    frame = records.frame
    match = pd.Series(False, index=frame.index)
    for ref in definition.codes:
        this = frame["CODE"] == ref.code
        if ref.system is not None and records.has_system:
            this &= frame["SYSTEM"] == ref.system
        match |= this
    starts_by_patient: dict[str, list[date]] = {}
    for patient, start in zip(frame.loc[match, "PATIENT"], frame.loc[match, "start"]):
        starts_by_patient.setdefault(str(patient), []).append(start.date())

    followed: list[tuple[Person, date | None]] = []
    prior_cases = later_records = records_after_exit = 0
    for patient_id, person in cohort.people.items():
        starts = sorted(starts_by_patient.get(patient_id, ()))
        if starts and starts[0] < person.entry:
            prior_cases += 1
            continue
        in_window = [s for s in starts if s <= person.exit]
        records_after_exit += len(starts) - len(in_window)
        event = in_window[0] if in_window else None
        later_records += max(0, len(in_window) - 1)
        followed.append((person, event))

    total = _rate(followed)
    strata = [*_age_strata(followed, cohort.age_bands), *_sex_strata(followed)]
    metrics: dict[str, Any] = {
        "followed": cohort.size,
        "prior_cases": prior_cases,
        "at_risk": len(followed),
        "events": total.events,
        "person_days": total.person_days,
        "later_records_not_counted": later_records,
        "records_after_exit": records_after_exit,
        "records_start_unusable": records.start_unusable,
    }
    notes = [
        f"{prior_cases} patient(s) with a record of the condition before entering the window "
        f"are prior cases and are not at risk."
    ]
    if definition.acute and later_records:
        notes.append(
            f"{later_records} later record(s) of at-risk patients within the window were not "
            f"counted: only first events are, so repeated episodes of this acute condition are "
            f"not included."
        )
    by_measure = {"incidence": total}
    return ConditionIncidence(
        name=definition.name,
        codes=definition.codes,
        status=SectionStatus.COMPUTED,
        acute=definition.acute,
        rate=total,
        strata=tuple(strata),
        expected=tuple(
            ExpectedComparison(expected, by_measure[expected.measure])
            for expected in definition.expected
            if expected.measure in by_measure
        ),
        metrics=metrics,
        notes=tuple(notes),
    )


def _days(start: date, end: date) -> int:
    return max(0, (end - start).days)


def _rate(followed: list[tuple[Person, date | None]]) -> IncidenceRate:
    events = sum(1 for _, event in followed if event is not None)
    days = sum(_days(p.entry, event or p.exit) for p, event in followed)
    return IncidenceRate(events, days)


def _sex_strata(followed: list[tuple[Person, date | None]]) -> list[IncidenceStratum]:
    values = sorted({p.gender for p, _ in followed if p.gender is not None})
    if any(p.gender is None for p, _ in followed):
        values.append(None)
    return [
        IncidenceStratum("GENDER", value, _rate([f for f in followed if f[0].gender == value]))
        for value in values
    ]


def anniversary(birth: date, years: int) -> date:
    """The day a person born on ``birth`` turns ``years`` (29 February → 1 March)."""
    try:
        return birth.replace(year=birth.year + years)
    except ValueError:
        return date(birth.year + years, 3, 1)


def age_on(birth: date, day: date) -> int:
    """Completed years on ``day``, by calendar birthday (consistent with :func:`anniversary`)."""
    return day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))


def band_days(person: Person, end: date, bounds: tuple[int, ...]) -> list[int]:
    """Days of ``[person.entry, end)`` spent in each age band."""
    starts = [anniversary(person.birth, b) for b in bounds]
    split = []
    for index, low in enumerate(starts):
        high = starts[index + 1] if index + 1 < len(starts) else None
        piece_start = max(person.entry, low)
        piece_end = end if high is None else min(end, high)
        split.append(max(0, (piece_end - piece_start).days))
    return split


def band_of(age: int, bounds: tuple[int, ...]) -> int:
    """Index of the band an age belongs to."""
    return max(i for i, low in enumerate(bounds) if age >= low)


def _age_strata(
    followed: list[tuple[Person, date | None]], bounds: tuple[int, ...]
) -> list[IncidenceStratum]:
    days = [0] * len(bounds)
    events = [0] * len(bounds)
    for person, event in followed:
        for index, piece in enumerate(band_days(person, event or person.exit, bounds)):
            days[index] += piece
        if event is not None:
            events[band_of(age_on(person.birth, event), bounds)] += 1
    return [
        IncidenceStratum("age_band", label, IncidenceRate(events[i], days[i]))
        for i, label in enumerate(age_band_labels(bounds))
    ]
