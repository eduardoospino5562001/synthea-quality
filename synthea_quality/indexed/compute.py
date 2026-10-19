"""[INDEXED] The indexed cohort: one row per patient with an index date.

The cohort is built from the condition of the module file's ``index`` block. It takes
the decisions the outcomes of :mod:`synthea_quality.indexed.definitions` rely on:

1. A patient is in the cohort when they have a record of the index condition whose
   ``START`` is on or before the reference date, **alive or deceased**. Including the
   deceased is what makes a ``death`` outcome meaningful; the other cohorts of the tool
   only count alive patients.
2. The index date is the earliest ``START`` of those records.
3. The age band is the age on the **index date**, not on the reference date: an outcome
   window starts at the index date, so a patient must not move bands in between.
4. Every patient counts once, however many records of the condition they have: the
   missing unit of the outcome schema is a patient, never a row.
5. Nothing is dropped silently. The result counts the condition records of a patient
   unknown to ``patients``, the records whose ``START`` is empty or unparseable, and
   the records after the reference date.

This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from synthea_quality.incidence.compute import age_on, band_of
from synthea_quality.indexed.definitions import (
    DeathOutcome,
    EventOutcome,
    ValueOutcome,
    Window,
)
from synthea_quality.indexed.models import (
    UNKNOWN_SEX,
    Breakdown,
    DeathResult,
    EventResult,
    Stratum,
    ValueResult,
    ValueShare,
)
from synthea_quality.observations.compute import summarise
from synthea_quality.prevalence.compute import condition_rows
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.profile.dates import parse_date_only, parse_timestamps
from synthea_quality.profile.population import age_band_labels

#: Age band of a patient whose ``BIRTHDATE`` is empty or unparseable.
UNKNOWN_AGE = "unknown age"
#: Column holding the record date of each table an event outcome may read.
DATE_COLUMNS = {"observations": "DATE", "immunizations": "DATE"}
#: Tables whose record date is a day (``YYYY-MM-DD``); the rest are UTC timestamps.
DATE_ONLY_TABLES = ("conditions", "careplans")
#: The observation of the cause of death; its VALUE is the cause's description, not a code.
CAUSE_OF_DEATH_CODE = "69453-9"


@dataclass(frozen=True, slots=True, eq=False)
class IndexedCohort:
    """The cohort's patients, one row each, and the records left out on the way."""

    #: Indexed by ``PATIENT``; columns ``INDEX_DATE``, ``SEX``, ``AGE_BAND``, ``DEATH_DATE``.
    frame: pd.DataFrame
    #: Condition records of the index condition, any date.
    rows_matched: int
    #: Of those, records of a ``PATIENT`` absent from ``patients``.
    rows_unknown_patient: int
    #: Of those, records with an empty or unparseable ``START``.
    rows_start_unusable: int
    #: Of those, records with a ``START`` after the reference date.
    rows_after_reference: int

    @property
    def size(self) -> int:
        """How many patients are in the cohort."""
        return len(self.frame)


def build_indexed_cohort(
    patients: pd.DataFrame,
    conditions: pd.DataFrame,
    definition: ConditionDefinition,
    reference: date,
    age_bands: Sequence[int],
) -> IndexedCohort:
    """The patients of ``definition`` with their index date, sex, age band and death date.

    A birth missing or after the index date gives the unknown age band.

    ``patients`` has the text columns ``Id``, ``BIRTHDATE``, ``DEATHDATE`` and ``GENDER``
    (see :data:`synthea_quality.prevalence.compute.PATIENT_COLUMNS`); an empty
    ``GENDER`` becomes ``None``. The condition rows are selected with
    :func:`synthea_quality.prevalence.compute.condition_rows`, the same match as the
    prevalence numerators.
    """
    matched = condition_rows(conditions, definition)
    start = parse_date_only(conditions["START"])
    known = conditions["PATIENT"].isin(patients["Id"])
    ref = pd.Timestamp(reference)

    usable = matched & known & start.values.notna()
    in_time = usable & (start.values <= ref)
    first_start = start.values[in_time].groupby(conditions.loc[in_time, "PATIENT"]).min()

    labels = age_band_labels(age_bands)
    bounds = tuple(age_bands)
    people = patients.set_index("Id")
    births = parse_date_only(people["BIRTHDATE"]).values
    deaths = parse_date_only(people["DEATHDATE"]).values
    sexes = people["GENDER"]

    rows = []
    for patient in sorted(first_start.index):
        index_date = first_start.loc[patient]
        birth = births.loc[patient]
        if pd.isna(birth):
            band = UNKNOWN_AGE
        else:
            age = age_on(birth.date(), index_date.date())
            band = UNKNOWN_AGE if age < 0 else labels[band_of(age, bounds)]
        sex = sexes.loc[patient]
        rows.append(
            {
                "PATIENT": patient,
                "INDEX_DATE": index_date,
                "SEX": None if pd.isna(sex) else str(sex),
                "AGE_BAND": band,
                "DEATH_DATE": deaths.loc[patient],
            }
        )
    frame = pd.DataFrame(
        rows, columns=["PATIENT", "INDEX_DATE", "SEX", "AGE_BAND", "DEATH_DATE"]
    ).set_index("PATIENT")
    frame["INDEX_DATE"] = pd.to_datetime(frame["INDEX_DATE"])
    frame["DEATH_DATE"] = pd.to_datetime(frame["DEATH_DATE"])

    return IndexedCohort(
        frame=frame,
        rows_matched=int(matched.sum()),
        rows_unknown_patient=int((matched & ~known).sum()),
        rows_start_unusable=int((matched & known & start.values.isna()).sum()),
        rows_after_reference=int((usable & ~(start.values <= ref)).sum()),
    )


def record_days(cohort: IndexedCohort, table: pd.DataFrame, table_name: str) -> pd.Series:
    """Days from each patient's index date to every record of ``table``, as floats.

    The record date comes from ``DATE`` in ``observations`` and ``immunizations`` and
    from ``START`` elsewhere (see :data:`DATE_COLUMNS`); ``conditions`` and ``careplans``
    write a day and the rest a UTC timestamp (see :data:`DATE_ONLY_TABLES`). A record of
    a patient outside the cohort, or with an empty or unparseable date, is ``NaN``.
    """
    column = DATE_COLUMNS.get(table_name, "START")
    if table.empty:
        return pd.Series(float("nan"), index=table.index, dtype="float")
    if table_name in DATE_ONLY_TABLES:
        dates = parse_date_only(table[column]).values
    else:
        dates = parse_timestamps(table[column]).values.dt.normalize()
    index_dates = table["PATIENT"].map(cohort.frame["INDEX_DATE"])
    days = (dates - index_dates).dt.days
    return days.astype("float")


def breakdown(cohort: IndexedCohort, hits: pd.Index, age_bands: Sequence[int]) -> Breakdown:
    """The outcome of ``hits`` (patients of ``cohort.frame``) in total and by stratum.

    Every count is a distinct patient. ``by_age`` has a stratum for every band of
    :func:`~synthea_quality.profile.population.age_band_labels`, even an empty one, and
    :data:`UNKNOWN_AGE` last only when some patient has no usable birth date.
    """
    hit_index = pd.Index(hits).unique()
    frame = cohort.frame
    total = Stratum("all", len(hit_index), cohort.size)
    sexes = frame["SEX"].where(frame["SEX"].notna(), UNKNOWN_SEX)
    by_sex = tuple(
        _stratum(label, frame.index[sexes == label], hit_index)
        for label in sorted(sexes.unique())
    )
    by_age = [
        _stratum(label, _band_members(frame, label), hit_index)
        for label in age_band_labels(age_bands)
    ]
    if (frame["AGE_BAND"] == UNKNOWN_AGE).any():
        by_age.append(_stratum(UNKNOWN_AGE, _band_members(frame, UNKNOWN_AGE), hit_index))
    return Breakdown(total, by_sex, tuple(by_age))


def _band_members(frame: pd.DataFrame, label: str) -> pd.Index:
    """Patients of ``frame`` whose age band is ``label``."""
    return frame.index[frame["AGE_BAND"] == label]


def _stratum(label: str, members: pd.Index, hits: pd.Index) -> Stratum:
    """The cell of ``label``: its patients with the outcome out of its cohort."""
    return Stratum(label, int(members.isin(hits).sum()), len(members))


def _window_masks(
    cohort: IndexedCohort,
    matched: pd.DataFrame,
    days: pd.Series,
    window: Window,
    reference: date,
) -> tuple[pd.Series, pd.Series]:
    """(in_window, after_reference) for the records ``matched``, one flag per row."""
    if matched.empty:
        after = pd.Series(False, index=matched.index)
    else:
        index_dates = matched["PATIENT"].map(cohort.frame["INDEX_DATE"])
        after = days > (pd.Timestamp(reference) - index_dates).dt.days
    in_window = days.notna() & ~after & (days >= window.from_days)
    if window.to_days is not None:
        in_window &= days <= window.to_days
    return in_window, after


def measure_event(
    cohort: IndexedCohort,
    table: pd.DataFrame,
    outcome: EventOutcome,
    reference: date,
    age_bands: Sequence[int],
) -> EventResult:
    """Patients of the cohort with a record of ``outcome.codes`` in its window.

    Only ``CODE`` selects a record: ``SYSTEM`` is not consulted, because the outcome
    definition has no system. The window is counted in days from each patient's index
    date, and :attr:`~synthea_quality.indexed.models.EventResult.days` describes the days
    to the first record of the window, one value per patient.
    """
    selected = table["PATIENT"].isin(cohort.frame.index) & table["CODE"].isin(outcome.codes)
    matched = table[selected]
    days = record_days(cohort, matched, outcome.table)
    in_window, after = _window_masks(cohort, matched, days, outcome.window, reference)
    first = days[in_window].groupby(matched.loc[in_window, "PATIENT"]).min()
    return EventResult(
        name=outcome.name,
        table=outcome.table,
        codes=outcome.codes,
        window=outcome.window,
        breakdown=breakdown(cohort, first.index, age_bands),
        days=summarise(first),
        rows_matched=len(matched),
        rows_date_unusable=int(days.isna().sum()),
        rows_after_reference=int(after.sum()),
        rows_outside_window=int((days.notna() & ~after & ~in_window).sum()),
    )


def measure_value(
    cohort: IndexedCohort,
    observations: pd.DataFrame,
    outcome: ValueOutcome,
    reference: date,
    age_bands: Sequence[int],
) -> ValueResult:
    """The values of ``outcome.code`` on the first or last date of its window.

    A patient picks one date, so ``pick`` decides which of their in-window values is
    kept; every row of that patient on that date contributes its VALUE. Distinct values
    on the same date (the sample has two stage rows on one date) put the patient in
    every one of them: the shares may add up to more than the patients of the breakdown,
    and ``patients_with_several_values`` says how many patients do. An empty VALUE is
    not a value, so it is counted and left out; a patient without a value stays in the
    denominator.
    """
    matched = observations[
        observations["PATIENT"].isin(cohort.frame.index)
        & (observations["CODE"] == outcome.code)
    ]
    days = record_days(cohort, matched, "observations")
    in_window, after = _window_masks(cohort, matched, days, outcome.window, reference)
    valued = in_window & matched["VALUE"].notna()
    on_picked_date = pd.Series(False, index=matched.index)
    if valued.any():
        pick = "min" if outcome.pick == "first" else "max"
        picked_day = days.where(valued).groupby(matched["PATIENT"]).transform(pick)
        on_picked_date = valued & (days == picked_day)
    values_by_patient: dict[str, set[str]] = {
        str(patient): {str(value) for value in values}
        for patient, values in matched[on_picked_date].groupby("PATIENT")["VALUE"]
    }
    counts: dict[str, int] = {}
    for values in values_by_patient.values():
        for value in values:
            counts[value] = counts.get(value, 0) + 1
    shares = tuple(
        ValueShare(value, patients)
        for value, patients in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return ValueResult(
        name=outcome.name,
        code=outcome.code,
        pick=outcome.pick,
        window=outcome.window,
        breakdown=breakdown(cohort, pd.Index(list(values_by_patient)), age_bands),
        values=shares,
        patients_with_several_values=sum(
            1 for values in values_by_patient.values() if len(values) > 1
        ),
        rows_matched=len(matched),
        rows_date_unusable=int(days.isna().sum()),
        rows_after_reference=int(after.sum()),
        rows_outside_window=int((days.notna() & ~after & ~in_window).sum()),
        rows_value_empty=int((in_window & matched["VALUE"].isna()).sum()),
    )


def measure_death(
    cohort: IndexedCohort,
    patients: pd.DataFrame,
    observations: pd.DataFrame | None,
    outcome: DeathOutcome,
    reference: date,
    age_bands: Sequence[int],
) -> DeathResult:
    """Deaths of the cohort within the window and, optionally, their recorded cause.

    ``DEATHDATE`` is a day, not a timestamp. A present but unparseable date leaves the
    patient without a death date and is counted: it is never silently read as alive. The
    cause of death is the VALUE text of observation :data:`CAUSE_OF_DEATH_CODE`, a
    description rather than a code, so ``cause_values`` match the exact texts of the
    module file. Every count is a patient, never a row.
    """
    frame = cohort.frame
    death_dates = frame["DEATH_DATE"]
    days = (death_dates - frame["INDEX_DATE"]).dt.days.astype("float")
    after = death_dates > pd.Timestamp(reference)
    in_window = days.notna() & ~after & (days >= outcome.window.from_days)
    if outcome.window.to_days is not None:
        in_window &= days <= outcome.window.to_days
    dead = frame.index[in_window]
    unusable = parse_date_only(patients.set_index("Id")["DEATHDATE"].reindex(frame.index))
    with_cause, other_cause, no_cause_record = _cause_counts(
        observations, dead, outcome.cause_values
    )
    return DeathResult(
        name=outcome.name,
        cause_values=outcome.cause_values,
        window=outcome.window,
        breakdown=breakdown(cohort, dead, age_bands),
        days=summarise(days[in_window]),
        deaths_after_reference=int(after.sum()),
        deaths_outside_window=int((death_dates.notna() & ~after & ~in_window).sum()),
        deathdate_unusable=unusable.unparseable,
        with_cause=with_cause,
        other_cause=other_cause,
        no_cause_record=no_cause_record,
    )


def _cause_counts(
    observations: pd.DataFrame | None,
    dead: pd.Index,
    cause_values: tuple[str, ...],
) -> tuple[int | None, int | None, int | None]:
    """(with_cause, other_cause, no_cause_record) of the deceased ``dead``.

    A patient has a cause only with a ``69453-9`` row whose ``VALUE`` is not empty, so
    ``no_cause_record`` counts the deceased without such a row. All three are ``None``
    without ``cause_values`` or without an observations table: the outcome then measures
    deaths only.
    """
    if not cause_values or observations is None:
        return None, None, None
    recorded = observations[
        observations["PATIENT"].isin(dead)
        & (observations["CODE"] == CAUSE_OF_DEATH_CODE)
    ]
    by_patient = {
        patient: causes
        for patient, values in recorded.groupby("PATIENT")["VALUE"]
        if (causes := {str(value) for value in values if pd.notna(value) and str(value).strip()})
    }
    wanted = set(cause_values)
    with_cause = sum(1 for values in by_patient.values() if values & wanted)
    return with_cause, len(by_patient) - with_cause, len(dead) - len(by_patient)
