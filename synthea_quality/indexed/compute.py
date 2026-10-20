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
6. The ``groups`` of a ``value`` outcome are applied **before** anything is counted: a
   ``VALUE`` text that belongs to a group counts under the label of the group, so a
   patient with two rows of the same group on the same day counts once. A text outside
   every group keeps its own text.
7. A cross of two outcomes is an intersection of patients, never of counts. The members
   of an outcome come from the same masks its measure uses, so the total of a cross is
   the total of the outcome it measures, and the strata may add up to more than the
   cohort when a patient has more than one value. A ``value`` outcome whose values hold
   the very text of ``NO_VALUE_LABEL`` gives no cross: that label already names the
   patients without a value, so the cross is left out with its reason.
8. An event outcome can have several parts and filters. Every part is measured on its
   own, and its records go through six steps in this order: the code, the filters of
   the part, the end of the record, its date, the reference date and the window. A
   record left out is counted in exactly one of them, and the date of a part is the
   first record of the window. The parts then combine: with ``any`` the patient counts
   with one part and reaches the outcome on the earliest of them; with ``all`` the
   patient needs every part and reaches the outcome on the latest of them, the day the
   last part is met.

This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType

import pandas as pd

from synthea_quality.incidence.compute import age_on, band_of
from synthea_quality.indexed.definitions import (
    NO_VALUE_LABEL,
    CrossSpec,
    DeathOutcome,
    EventMatch,
    EventOutcome,
    ValueOutcome,
    Window,
)
from synthea_quality.indexed.models import (
    UNKNOWN_SEX,
    Breakdown,
    CrossResult,
    CrossStratum,
    DeathResult,
    EventMatchResult,
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
#: The column that holds the end of a record; a part can count on it.
STOP_COLUMN = "STOP"
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


@dataclass(frozen=True, slots=True, eq=False)
class OutcomeMembers:
    """The patients of an outcome, as sets, so that two outcomes can be crossed.

    A cross needs the patients of each outcome and not only their counts: the patients
    of the crossed outcome are intersected with the strata of the other one. The sets
    come from the same masks the measures use, so the numerator of a cross is never
    larger than the total of the outcome it measures.

    ``values`` holds the labels of a ``value`` outcome and the patients under them,
    already grouped; it is empty for the other kinds of outcome. ``with_cause`` holds
    the deaths with the recorded cause of a ``death`` outcome, and is ``None`` when the
    outcome has no cause to count. Two members are equal only when they are the same
    object: nothing compares or hashes them, and the read-only mapping of the values
    would make a field-by-field equality misleading.
    """

    #: The patients that count in the outcome: the same set as its total numerator.
    patients: frozenset[str]
    #: Each label of a ``value`` outcome and the patients under it; empty otherwise.
    values: Mapping[str, frozenset[str]] = field(default_factory=dict)
    #: The deaths with the recorded cause of a ``death`` outcome; ``None`` without one.
    with_cause: frozenset[str] | None = None

    def __post_init__(self) -> None:
        # The mapping is read-only: the members of an outcome never change once measured.
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


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
    ``GENDER`` becomes ``None``, and so does every patient when the table has no ``GENDER``
    column. The condition rows are selected with
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
    sexes = people["GENDER"] if "GENDER" in people.columns else pd.Series(None, index=people.index)

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


def record_days(
    cohort: IndexedCohort,
    table: pd.DataFrame,
    table_name: str,
    column: str | None = None,
) -> pd.Series:
    """Days from each patient's index date to every record of ``table``, as floats.

    The record date comes from ``DATE`` in ``observations`` and ``immunizations`` and
    from ``START`` elsewhere (see :data:`DATE_COLUMNS`); ``conditions`` and ``careplans``
    write a day and the rest a UTC timestamp (see :data:`DATE_ONLY_TABLES`). A record of
    a patient outside the cohort, or with an empty or unparseable date, is ``NaN``.

    ``column`` reads another date column of the same table, with the same rule of the
    table: a part that counts on the end of a record asks for its ``STOP`` date.
    """
    column = column or DATE_COLUMNS.get(table_name, "START")
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


@dataclass(frozen=True, slots=True, eq=False)
class _PartSelection:
    """The records of one part of an event outcome and what it found.

    The ``rows_*`` counts are records, not patients: they say what was left out on the
    way to the distinct patients of the part. Every record left out is counted in
    exactly one of them.
    """

    #: The records of the codes of the part, of the patients of the cohort.
    matched: pd.DataFrame
    #: The days to the first in-window record, one value per patient.
    first: pd.Series
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


def _part_selection(
    cohort: IndexedCohort,
    table: pd.DataFrame,
    part: EventMatch,
    window: Window,
    reference: date,
) -> _PartSelection:
    """The records of one part of an event outcome and the patients they select.

    The records of the part go through six steps, in this order, and every record left
    out is counted in exactly one of them: the code (of a patient of the cohort), the
    filters of the part, the end of the record, its date, the reference date and the
    window. The date of the part is the first record of the window.
    """
    selected = table["PATIENT"].isin(cohort.frame.index) & table["CODE"].isin(part.codes)
    matched = table[selected]
    filtered = _keep_filtered(matched, part)
    ended = _keep_ended(filtered, part)
    days = record_days(cohort, ended, part.table, column=_date_column(part))
    in_window, after = _window_masks(cohort, ended, days, window, reference)
    first = days[in_window].groupby(ended.loc[in_window, "PATIENT"]).min()
    return _PartSelection(
        matched=matched,
        first=first,
        rows_matched=len(matched),
        rows_filtered_out=len(matched) - len(filtered),
        rows_not_ended=len(filtered) - len(ended),
        rows_date_unusable=int(days.isna().sum()),
        rows_after_reference=int(after.sum()),
        rows_outside_window=int(
            (days.notna() & ~after & ~in_window).sum()
        ),
    )


def _keep_filtered(matched: pd.DataFrame, part: EventMatch) -> pd.DataFrame:
    """The records of ``matched`` that pass the filters of ``part``.

    ``values`` and ``reasons`` are exact texts, compared as they are written: the
    loader keeps every value of a CSV as text, so nothing is converted here.
    """
    kept = matched
    if part.values:
        kept = kept[kept["VALUE"].isin(part.values)]
    if part.reasons:
        kept = kept[kept["REASONCODE"].isin(part.reasons)]
    return kept


def _keep_ended(matched: pd.DataFrame, part: EventMatch) -> pd.DataFrame:
    """The records of ``matched`` that ended, when the part counts on their end.

    A record without a ``STOP`` is still open: it has no end date to count on, so it
    is left out here. A record with a ``STOP`` that does not parse is left out later,
    as a date that cannot be used.
    """
    if part.at != "stop":
        return matched
    return matched[matched[STOP_COLUMN].notna()]


def _date_column(part: EventMatch) -> str | None:
    """The date column a part counts on: its ``STOP``, or the one of its table."""
    return STOP_COLUMN if part.at == "stop" else None


@dataclass(frozen=True, slots=True, eq=False)
class _EventSelection:
    """The parts of an event outcome and the day each patient reaches it."""

    #: The selection of every part of the outcome, in the order of the parts.
    parts: tuple[_PartSelection, ...]
    #: The days from the index date to the outcome, one value per patient that counts.
    days: pd.Series


def _event_selection(
    cohort: IndexedCohort,
    tables: pd.DataFrame | Mapping[str, pd.DataFrame],
    outcome: EventOutcome,
    reference: date,
) -> _EventSelection:
    """The parts of an event outcome and the patients they select.

    The masks are computed once and used by both the measure and the members of the
    outcome, so the two can never disagree on which patients count.
    """
    frames = _tables_of(tables, outcome)
    parts = outcome.parts
    selections = tuple(
        _part_selection(cohort, frame, part, outcome.window, reference)
        for frame, part in zip(frames, parts)
    )
    return _EventSelection(selections, _outcome_days(selections, outcome.op))


def _tables_of(
    tables: pd.DataFrame | Mapping[str, pd.DataFrame], outcome: EventOutcome
) -> tuple[pd.DataFrame, ...]:
    """The table of every part of ``outcome``.

    A single frame is the table of an outcome of one part. An outcome of several parts
    needs one frame per part, given by the name of its table.

    :raises ValueError: the outcome has several parts and only one frame was given, or
        the frames do not hold the table of one of its parts.
    """
    parts = outcome.parts
    if isinstance(tables, pd.DataFrame):
        if len(parts) > 1:
            raise ValueError(
                f"the outcome {outcome.name!r} has {len(parts)} parts: it needs a table "
                "for each of them, not a single one"
            )
        return (tables,)
    frames = []
    for part in parts:
        if part.table not in tables:
            raise ValueError(
                f"the outcome {outcome.name!r} needs the table {part.table!r} of its "
                f"part {part.name!r}: it was not given"
            )
        frames.append(tables[part.table])
    return tuple(frames)


def _outcome_days(selections: Sequence[_PartSelection], op: str) -> pd.Series:
    """The day each patient reaches the outcome, from the days of its parts.

    With ``any`` a patient counts with one part and reaches the outcome on the
    earliest of them. With ``all`` the patient needs every part and reaches it on the
    latest of them: the day the last part is met.
    """
    days = pd.concat([selection.first for selection in selections])
    if op == "any":
        return days.groupby(level=0).min()
    # With "all" the patient needs every part: the outcome is on the latest of them.
    counted = days.groupby(level=0).count()
    complete = counted[counted == len(selections)].index
    return days[days.index.isin(complete)].groupby(level=0).max()


def measure_event(
    cohort: IndexedCohort,
    tables: pd.DataFrame | Mapping[str, pd.DataFrame],
    outcome: EventOutcome,
    reference: date,
    age_bands: Sequence[int],
) -> EventResult:
    """Patients of the cohort with a record of the codes of its parts in its window.

    Only ``CODE`` selects a record: ``SYSTEM`` is not consulted, because the outcome
    definition has no system. The window is counted in days from each patient's index
    date, and :attr:`~synthea_quality.indexed.models.EventResult.days` describes the
    days to the outcome, one value per patient: with ``any`` the earliest of its parts,
    with ``all`` the latest, the day the last part is met.

    ``tables`` is the table of the outcome when it has one part, and one table per part
    by name when it has several.

    :raises ValueError: the outcome has several parts and only one table was given, or
        the tables do not hold the table of one of its parts.
    """
    selection = _event_selection(cohort, tables, outcome, reference)
    matches = tuple(
        _match_result(part, part_selection)
        for part, part_selection in zip(outcome.parts, selection.parts)
    )
    return EventResult(
        name=outcome.name,
        table=outcome.table,
        codes=outcome.codes,
        window=outcome.window,
        breakdown=breakdown(cohort, selection.days.index, age_bands),
        days=summarise(selection.days),
        rows_matched=sum(match.rows_matched for match in matches),
        rows_date_unusable=sum(match.rows_date_unusable for match in matches),
        rows_after_reference=sum(match.rows_after_reference for match in matches),
        rows_outside_window=sum(match.rows_outside_window for match in matches),
        rows_filtered_out=sum(match.rows_filtered_out for match in matches),
        rows_not_ended=sum(match.rows_not_ended for match in matches),
        op=outcome.op,
        matches=matches,
    )


def _match_result(part: EventMatch, selection: _PartSelection) -> EventMatchResult:
    """What one part found: its rule, its patients and the records left out."""
    return EventMatchResult(
        name=part.name,
        table=part.table,
        codes=part.codes,
        values=part.values,
        reasons=part.reasons,
        at=part.at,
        patients=len(selection.first),
        rows_matched=selection.rows_matched,
        rows_filtered_out=selection.rows_filtered_out,
        rows_not_ended=selection.rows_not_ended,
        rows_date_unusable=selection.rows_date_unusable,
        rows_after_reference=selection.rows_after_reference,
        rows_outside_window=selection.rows_outside_window,
    )


def event_members(
    cohort: IndexedCohort,
    tables: pd.DataFrame | Mapping[str, pd.DataFrame],
    outcome: EventOutcome,
    reference: date,
) -> OutcomeMembers:
    """The patients of an event outcome: those with a record of its parts in its window.

    The set is the one of the total numerator of :func:`measure_event`: a cross of this
    outcome counts exactly the patients the outcome counts.

    :raises ValueError: the outcome has several parts and only one table was given, or
        the tables do not hold the table of one of its parts.
    """
    selection = _event_selection(cohort, tables, outcome, reference)
    return OutcomeMembers(patients=frozenset(selection.days.index))


@dataclass(frozen=True, slots=True, eq=False)
class _ValueSelection:
    """The records of a value outcome and the labels they give to each patient."""

    #: The observations of the code of the outcome, of the patients of the cohort.
    matched: pd.DataFrame
    #: The days from the index date to each of those observations.
    days: pd.Series
    #: Which of those observations fall in the window.
    in_window: pd.Series
    #: Which of those observations are dated after the reference date.
    after_reference: pd.Series
    #: The labels of each patient on the picked date, already grouped.
    values_by_patient: dict[str, frozenset[str]]


def _value_selection(
    cohort: IndexedCohort,
    observations: pd.DataFrame,
    outcome: ValueOutcome,
    reference: date,
) -> _ValueSelection:
    """The records of a value outcome and the labels they give to each patient.

    The groups are applied here, before anything is counted, so the shares, the
    breakdown and the patients with more than one value all speak of the same labels.
    The masks are computed once and used by both the measure and the members of the
    outcome, so the two can never disagree on which patients count.
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
    groups = _group_labels(outcome)
    values_by_patient = {
        str(patient): frozenset(_label_of(str(value), groups) for value in values)
        for patient, values in matched[on_picked_date].groupby("PATIENT")["VALUE"]
    }
    return _ValueSelection(matched, days, in_window, after, values_by_patient)


def _group_labels(outcome: ValueOutcome) -> dict[str, str]:
    """Each grouped ``VALUE`` text under the label of its group."""
    return {text: label for label, texts in outcome.groups for text in texts}


def _label_of(value: str, groups: Mapping[str, str]) -> str:
    """The label of ``value``: the label of its group, or the text itself.

    A group label that is also the text of a value outside every group counts them
    together: they are the same label.
    """
    return groups.get(value, value)


def _patients_by_label(
    values_by_patient: Mapping[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """Each label of a value outcome and the patients under it."""
    by_label: dict[str, set[str]] = {}
    for patient, labels in values_by_patient.items():
        for label in labels:
            by_label.setdefault(label, set()).add(patient)
    return {label: frozenset(patients) for label, patients in by_label.items()}


def _value_order(patients_by_label: Mapping[str, frozenset[str]]) -> tuple[str, ...]:
    """The labels of a value outcome, most patients first and by label on a tie.

    The order is the one of the table of values; the strata of a cross follow it, so
    both tables of the report speak of the values in the same order.
    """
    return tuple(
        sorted(patients_by_label, key=lambda label: (-len(patients_by_label[label]), label))
    )


def _patients_with_several_labels(
    patients_by_label: Mapping[str, frozenset[str]],
) -> int:
    """How many patients are under more than one label."""
    counts: dict[str, int] = {}
    for patients in patients_by_label.values():
        for patient in patients:
            counts[patient] = counts.get(patient, 0) + 1
    return sum(1 for count in counts.values() if count > 1)


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

    The ``groups`` of the outcome are applied first: a VALUE text that belongs to a
    group counts under the label of the group, so a patient with two rows of the same
    group on the same date counts once. A text outside every group keeps its own text,
    and a label that is also a text outside every group counts both together.
    """
    selection = _value_selection(cohort, observations, outcome, reference)
    patients_by_label = _patients_by_label(selection.values_by_patient)
    shares = tuple(
        ValueShare(label, len(patients_by_label[label]))
        for label in _value_order(patients_by_label)
    )
    return ValueResult(
        name=outcome.name,
        code=outcome.code,
        pick=outcome.pick,
        window=outcome.window,
        breakdown=breakdown(cohort, pd.Index(list(selection.values_by_patient)), age_bands),
        values=shares,
        patients_with_several_values=_patients_with_several_labels(patients_by_label),
        rows_matched=len(selection.matched),
        rows_date_unusable=int(selection.days.isna().sum()),
        rows_after_reference=int(selection.after_reference.sum()),
        rows_outside_window=int(
            (
                selection.days.notna()
                & ~selection.after_reference
                & ~selection.in_window
            ).sum()
        ),
        rows_value_empty=int(
            (selection.in_window & selection.matched["VALUE"].isna()).sum()
        ),
    )


def value_members(
    cohort: IndexedCohort,
    observations: pd.DataFrame,
    outcome: ValueOutcome,
    reference: date,
) -> OutcomeMembers:
    """The patients of a value outcome and the labels they have.

    The set of patients is the one of the total numerator of :func:`measure_value`, and
    the labels are already grouped: a cross of this outcome counts exactly the patients
    the outcome counts, under the same labels.
    """
    selection = _value_selection(cohort, observations, outcome, reference)
    return OutcomeMembers(
        patients=frozenset(selection.values_by_patient),
        values=_patients_by_label(selection.values_by_patient),
    )


@dataclass(frozen=True, slots=True, eq=False)
class _DeathSelection:
    """The deaths of the cohort in the window and their recorded cause."""

    #: The ``DEATHDATE`` of each patient of the cohort.
    death_dates: pd.Series
    #: The days from the index date to each of those dates.
    days: pd.Series
    #: Which of those deaths are dated after the reference date.
    after_reference: pd.Series
    #: Which of those deaths fall in the window.
    in_window: pd.Series
    #: The patients who died in the window.
    dead: pd.Index
    #: Patients whose ``DEATHDATE`` is present but unparseable.
    deathdate_unusable: int
    #: (deaths with the cause, deaths with any cause record); ``None`` without a cause.
    causes: tuple[frozenset[str], frozenset[str]] | None


def _death_selection(
    cohort: IndexedCohort,
    patients: pd.DataFrame,
    observations: pd.DataFrame | None,
    outcome: DeathOutcome,
    reference: date,
) -> _DeathSelection:
    """The deaths of the cohort in the window and the cause they recorded.

    The masks are computed once and used by both the measure and the members of the
    outcome, so the two can never disagree on which patients count.
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
    causes = _cause_members(observations, dead, outcome.cause_values)
    return _DeathSelection(
        death_dates, days, after, in_window, dead, unusable.unparseable, causes
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
    selection = _death_selection(cohort, patients, observations, outcome, reference)
    with_cause, other_cause, no_cause_record = _cause_counts(selection)
    return DeathResult(
        name=outcome.name,
        cause_values=outcome.cause_values,
        window=outcome.window,
        breakdown=breakdown(cohort, selection.dead, age_bands),
        days=summarise(selection.days[selection.in_window]),
        deaths_after_reference=int(selection.after_reference.sum()),
        deaths_outside_window=int(
            (
                selection.death_dates.notna()
                & ~selection.after_reference
                & ~selection.in_window
            ).sum()
        ),
        deathdate_unusable=selection.deathdate_unusable,
        with_cause=with_cause,
        other_cause=other_cause,
        no_cause_record=no_cause_record,
    )


def death_members(
    cohort: IndexedCohort,
    patients: pd.DataFrame,
    observations: pd.DataFrame | None,
    outcome: DeathOutcome,
    reference: date,
) -> OutcomeMembers:
    """The patients of a death outcome and the deaths with its recorded cause.

    The set of patients is the one of the total numerator of :func:`measure_death`. The
    cause is ``None`` without ``cause_values`` or without an observations table: there
    is then no cause to count.
    """
    selection = _death_selection(cohort, patients, observations, outcome, reference)
    return OutcomeMembers(
        patients=frozenset(selection.dead),
        with_cause=selection.causes[0] if selection.causes is not None else None,
    )


def _cause_members(
    observations: pd.DataFrame | None,
    dead: pd.Index,
    cause_values: tuple[str, ...],
) -> tuple[frozenset[str], frozenset[str]] | None:
    """(deaths with the cause, deaths with any cause record) of the deceased ``dead``.

    A patient has a cause only with a ``69453-9`` row whose ``VALUE`` is not empty, so
    the second set leaves out the deceased without such a row. The result is ``None``
    without ``cause_values`` or without an observations table: the outcome then measures
    deaths only.
    """
    if not cause_values or observations is None:
        return None
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
    with_cause = frozenset(
        patient for patient, causes in by_patient.items() if causes & wanted
    )
    return with_cause, frozenset(by_patient)


def _cause_counts(
    selection: _DeathSelection,
) -> tuple[int | None, int | None, int | None]:
    """(with_cause, other_cause, no_cause_record) of the deaths of ``selection``.

    The three counts add up to the deaths of the window. All three are ``None`` without
    a cause to count: the outcome then measures deaths only.
    """
    if selection.causes is None:
        return None, None, None
    with_cause, recorded = selection.causes
    return (
        len(with_cause),
        len(recorded) - len(with_cause),
        len(selection.dead) - len(recorded),
    )


def measure_cross(
    cohort: IndexedCohort,
    cross: CrossSpec,
    outcome: OutcomeMembers,
    by: OutcomeMembers,
    by_type: str,
) -> CrossResult:
    """The patients of ``outcome`` in each stratum of ``by``.

    ``outcome`` and ``by`` are the members of the two outcomes of ``cross``: the one
    that is measured and the one that gives the strata. ``by_type`` is the kind of the
    outcome that gives the strata: ``"value"`` gives one stratum per value, plus a last
    one for the patients without a value, and ``"event"`` gives two, with and without
    the event.

    The denominator of a stratum is the cohort of the stratum and the numerator the
    patients of ``outcome`` inside it; the two outcomes keep their own windows, and no
    order between them is required: the cross does not say that the value was observed
    before the death. A patient with more than one value counts in every stratum it
    belongs to, so the denominators may add up to more than the cohort.

    A ``value`` outcome with a value whose label is exactly ``NO_VALUE_LABEL`` gives no
    cross: the last stratum of the patients without a value would then have the same
    label as that value, and the label would no longer say which stratum it names. The
    text is not renamed either, so that the label always means the same thing; the
    module file puts it in a group with another label instead.

    :raises ValueError: ``by_type`` is neither ``"value"`` nor ``"event"``.
    """
    if by_type == "value" and NO_VALUE_LABEL in by.values:
        return CrossResult(
            outcome=cross.outcome,
            by=cross.by,
            by_type=by_type,
            strata=(),
            patients_in_several_strata=0,
            reason=(
                f"a value of {cross.by!r} is the text {NO_VALUE_LABEL!r}, the label of "
                "the patients without a value; put it in a group with another label"
            ),
        )
    if by_type == "value":
        strata, several = _value_strata(cohort, outcome, by)
    elif by_type == "event":
        strata, several = _event_strata(cohort, cross, outcome, by)
    else:
        raise ValueError(f"by_type must be 'value' or 'event', not {by_type!r}")
    return CrossResult(
        outcome=cross.outcome,
        by=cross.by,
        by_type=by_type,
        strata=tuple(strata),
        patients_in_several_strata=several,
    )


def skipped_cross(cross: CrossSpec, by_type: str, name: str, reason: str) -> CrossResult:
    """A cross that could not be computed: no strata and the reason why.

    ``name`` is the outcome that could not be measured, ``X`` or ``Y`` of the crossing.
    """
    return CrossResult(
        outcome=cross.outcome,
        by=cross.by,
        by_type=by_type,
        strata=(),
        patients_in_several_strata=0,
        reason=f"outcome '{name}' was not computed: {reason}",
    )


def _value_strata(
    cohort: IndexedCohort,
    outcome: OutcomeMembers,
    by: OutcomeMembers,
) -> tuple[list[CrossStratum], int]:
    """One stratum per value of ``by``, and a last one for the patients without a value."""
    strata = [
        _cross_stratum(label, by.values[label], outcome) for label in _value_order(by.values)
    ]
    valued: set[str] = set()
    for patients in by.values.values():
        valued |= patients
    strata.append(_cross_stratum(NO_VALUE_LABEL, set(cohort.frame.index) - valued, outcome))
    return strata, _patients_with_several_labels(by.values)


def _event_strata(
    cohort: IndexedCohort,
    cross: CrossSpec,
    outcome: OutcomeMembers,
    by: OutcomeMembers,
) -> tuple[list[CrossStratum], int]:
    """Two strata of ``by``: the patients with the event and the ones without it."""
    with_event = set(by.patients)
    without = set(cohort.frame.index) - with_event
    return [
        _cross_stratum(f"with {cross.by}", with_event, outcome),
        _cross_stratum(f"without {cross.by}", without, outcome),
    ], 0


def _cross_stratum(
    label: str,
    members: set[str] | frozenset[str],
    outcome: OutcomeMembers,
) -> CrossStratum:
    """The cell of ``label``: the patients of ``outcome`` among ``members``."""
    with_cause = None if outcome.with_cause is None else len(members & outcome.with_cause)
    return CrossStratum(
        label=label,
        numerator=len(members & outcome.patients),
        denominator=len(members),
        with_cause=with_cause,
    )
