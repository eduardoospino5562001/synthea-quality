"""[PREVALENCE] Point and lifetime prevalence, from frames that are already loaded.

Definitions (``ref`` is the reference date, a calendar day):

* a record counts for **lifetime** prevalence when ``START <= ref``;
* it counts for **point** prevalence when, in addition, ``STOP`` is empty or
  ``STOP > ref``. ``START == ref`` is active on that day; ``STOP == ref`` is not
  (``conditions.csv`` writes both as ``YYYY-MM-DD``, and a condition whose stop day is the
  reference day has ended by the end of it);
* the numerator is the number of **distinct alive patients** with at least one such record
  of any of the condition's codes; the denominator is the number of alive patients, or
  the alive patients of the stratum.

What is counted, then left out (every count is in the result):

* rows of deceased patients and rows whose ``PATIENT`` is not in ``patients.csv``;
* rows without a ``CODE``;
* rows whose ``START`` is empty or unparseable: they cannot be placed in time;
* rows that start after the reference date;
* rows whose ``STOP`` is present but unparseable are kept for lifetime prevalence, which
  does not need a stop, and left out of point prevalence, which does.

Every condition reports how many of its records have no ``STOP`` out of the records
used: such a record counts as active at the reference date. For a condition its
definition declares acute, a note says that point prevalence then counts every past
event as still active and that lifetime prevalence is the meaningful measure. Whether a
condition is acute is never inferred here.

Strata: the age band of each alive patient at the reference date (the profile's bands and
ages, by calendar birthday) and ``GENDER``. A patient without a known age is left out of
the age strata and counted; an empty ``GENDER`` is its own stratum. An empty stratum
keeps its row, with no rate.

This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.prevalence.exclusion import ExclusionList, codes_absent_from_data
from synthea_quality.prevalence.models import (
    MEASURES,
    ConditionResult,
    ExpectedComparison,
    GeneralRow,
    GeneralTable,
    Rate,
    Stratum,
    normalise_system,
)
from synthea_quality.prevalence.social import (
    SOCIAL_CODES,
    SOCIAL_LIST_ID,
    SOURCE_MODULES,
    is_social,
)
from synthea_quality.profile.dates import parse_date_only
from synthea_quality.profile.models import SectionStatus
from synthea_quality.profile.population import (
    age_band_labels,
    age_band_of,
    ages_at,
    alive_patient_ids,
)

#: Columns of ``patients.csv`` and ``conditions.csv`` this module uses.
PATIENT_COLUMNS = ("Id", "BIRTHDATE", "DEATHDATE", "GENDER")
CONDITION_COLUMNS = ("PATIENT", "CODE", "START", "STOP", "SYSTEM", "DESCRIPTION")
REQUIRED_CONDITION_COLUMNS = ("PATIENT", "CODE", "START", "STOP")

#: Point prevalence at most this share of lifetime prevalence triggers the acute note.
ACUTE_NOTE_RATIO = 0.1

DEFAULT_TOP = 30


# --------------------------------------------------------------------------- #
# the alive cohort
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, eq=False)
class Cohort:
    """The alive patients, with what the strata need to know about each."""

    alive_ids: pd.Index
    patient_ids: pd.Index
    #: Age band label per alive patient (index: Id); missing when the age is unknown.
    age_band: pd.Series | None
    #: ``GENDER`` per alive patient (index: Id); missing when empty.
    gender: pd.Series | None
    band_labels: tuple[str, ...]
    #: Alive patients left out of the age strata, and why.
    without_age: int
    notes: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.alive_ids)


def build_cohort(patients: pd.DataFrame, reference: date, age_bands: Sequence[int]) -> Cohort:
    """The alive cohort of ``patients`` (text columns of :data:`PATIENT_COLUMNS`).

    :raises KeyError: ``patients`` has no ``Id`` or no ``DEATHDATE`` column.
    """
    alive_ids = alive_patient_ids(patients)
    alive = patients[patients["Id"].isin(alive_ids)].drop_duplicates("Id").set_index("Id")
    notes: list[str] = []

    age_band = None
    without_age = 0
    if "BIRTHDATE" in alive.columns:
        ages = ages_at(parse_date_only(alive["BIRTHDATE"]).values, reference)
        age_band = age_band_of(ages, age_bands)
        without_age = int(age_band.isna().sum())
    else:
        notes.append("patients.csv has no BIRTHDATE column: no age strata.")

    gender = alive["GENDER"] if "GENDER" in alive.columns else None
    if gender is None:
        notes.append("patients.csv has no GENDER column: no sex strata.")

    return Cohort(
        alive_ids=alive_ids,
        patient_ids=pd.Index(patients["Id"].dropna().unique()),
        age_band=age_band,
        gender=gender,
        band_labels=age_band_labels(age_bands),
        without_age=without_age,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------- #
# condition records placed in time
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, eq=False)
class Records:
    """The alive patients' condition records that can be placed in time."""

    #: PATIENT, SYSTEM ("" when absent), CODE, lifetime (bool), point (bool), has_stop (bool)
    frame: pd.DataFrame
    #: Most frequent description per (SYSTEM, CODE) over the whole table.
    descriptions: dict[tuple[str, str], str | None]
    #: Every (SYSTEM, CODE) of the whole table, alive or not.
    table_codes: frozenset[tuple[str, str]]
    has_system: bool
    metrics: dict[str, int]


def prepare_records(conditions: pd.DataFrame, cohort: Cohort, reference: date) -> Records:
    """Place every condition row in time relative to ``reference`` and count what is left out."""
    has_system = "SYSTEM" in conditions.columns
    system = (
        conditions["SYSTEM"].map(normalise_system, na_action="ignore").fillna("")
        if has_system
        else pd.Series("", index=conditions.index)
    )
    coded = conditions["CODE"].notna()
    descriptions = most_frequent_descriptions(conditions, system, coded)
    table_codes = frozenset(zip(system[coded], conditions.loc[coded, "CODE"]))

    alive = conditions["PATIENT"].isin(cohort.alive_ids)
    known = conditions["PATIENT"].isin(cohort.patient_ids)
    start = parse_date_only(conditions["START"])
    stop = parse_date_only(conditions["STOP"])
    stop_missing = conditions["STOP"].isna()
    stop_bad = ~stop_missing & stop.values.isna()
    ref = pd.Timestamp(reference)

    usable = alive & coded & start.values.notna()
    lifetime = usable & (start.values <= ref)
    point = lifetime & ~stop_bad & (stop_missing | (stop.values > ref))

    frame = pd.DataFrame(
        {
            "PATIENT": conditions["PATIENT"],
            "SYSTEM": system,
            "CODE": conditions["CODE"],
            "lifetime": lifetime,
            "point": point,
            "has_stop": ~stop_missing,
        }
    )[lifetime]

    metrics = {
        "rows": len(conditions),
        "rows_alive": int(alive.sum()),
        "rows_deceased": int((known & ~alive).sum()),
        "rows_unknown_patient": int((~known).sum()),
        "rows_without_code": int((alive & ~coded).sum()),
        "rows_start_unusable": int((alive & coded & start.values.isna()).sum()),
        "rows_after_reference": int((usable & ~(start.values <= ref)).sum()),
        "rows_stop_unparseable": int((lifetime & stop_bad).sum()),
        "rows_used": int(lifetime.sum()),
    }
    return Records(frame, descriptions, table_codes, has_system, metrics)


def most_frequent_descriptions(
    table: pd.DataFrame, system: pd.Series, coded: pd.Series
) -> dict[tuple[str, str], str | None]:
    """The most frequent ``DESCRIPTION`` of every (SYSTEM, CODE) of ``table``'s ``coded`` rows.

    Ties go to a non-empty description, then to the first in alphabetical order.
    """
    if "DESCRIPTION" not in table.columns:
        return {}
    work = pd.DataFrame(
        {"SYSTEM": system, "CODE": table["CODE"], "DESCRIPTION": table["DESCRIPTION"]}
    )[coded]
    counted = work.groupby(["SYSTEM", "CODE", "DESCRIPTION"], dropna=False).size().reset_index()
    counted.columns = ["SYSTEM", "CODE", "DESCRIPTION", "records"]
    counted["_empty"] = counted["DESCRIPTION"].isna()
    counted["_text"] = counted["DESCRIPTION"].fillna("")
    counted = counted.sort_values(
        ["SYSTEM", "CODE", "records", "_empty", "_text"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    ).drop_duplicates(["SYSTEM", "CODE"])
    return {
        (row.SYSTEM, row.CODE): (None if pd.isna(row.DESCRIPTION) else str(row.DESCRIPTION))
        for row in counted.itertuples(index=False)
    }


# --------------------------------------------------------------------------- #
# conditions asked for
# --------------------------------------------------------------------------- #


def condition_prevalence(
    definition: ConditionDefinition, records: Records, cohort: Cohort
) -> ConditionResult:
    """Point and lifetime prevalence of one condition, in total and by stratum."""
    frame = records.frame
    match, per_code, missing_codes = _match(definition, records)
    rows = frame[match]
    point_ids = pd.Index(rows.loc[rows["point"], "PATIENT"].unique())
    lifetime_ids = pd.Index(rows["PATIENT"].unique())
    point = Rate(len(point_ids), cohort.size)
    lifetime = Rate(len(lifetime_ids), cohort.size)

    strata: list[Stratum] = []
    if cohort.age_band is not None:
        for label in cohort.band_labels:
            members = cohort.age_band[cohort.age_band == label].index
            strata.append(_stratum("age_band", label, members, point_ids, lifetime_ids))
    if cohort.gender is not None:
        values = sorted(cohort.gender.dropna().unique()) + (
            [None] if cohort.gender.isna().any() else []
        )
        for value in values:
            members = (
                cohort.gender[cohort.gender.isna()].index
                if value is None
                else cohort.gender[cohort.gender == value].index
            )
            strata.append(_stratum("GENDER", value, members, point_ids, lifetime_ids))

    records_without_stop = int((~rows["has_stop"]).sum())
    metrics: dict[str, Any] = {
        "records": len(rows),
        "records_by_code": per_code,
        "records_without_stop": records_without_stop,
        "records_with_stop": len(rows) - records_without_stop,
        "alive_without_age": cohort.without_age,
    }
    notes: list[str] = []
    if missing_codes:
        notes.append(
            f"No record of {', '.join(missing_codes)} in conditions.csv (any patient, any date)."
        )
    if definition.acute and records_without_stop:
        notes.append(
            f"{records_without_stop} of {len(rows)} records have no STOP date, so point "
            f"prevalence counts every past event as still active; for an acute condition, "
            f"lifetime prevalence is the meaningful measure."
        )
    if lifetime.numerator and point.numerator <= ACUTE_NOTE_RATIO * lifetime.numerator:
        notes.append(
            "Point prevalence is far below lifetime prevalence: the condition's records mostly "
            "end before the reference date. Point prevalence describes chronic or still-active "
            "conditions; for acute events (for example a myocardial infarction) the relevant "
            "measure is lifetime prevalence."
        )
    if cohort.without_age:
        notes.append(
            f"{cohort.without_age} alive patient(s) without a known age are left out of the age "
            f"strata (and counted in the totals)."
        )

    by_measure = {"point": point, "lifetime": lifetime}
    return ConditionResult(
        name=definition.name,
        codes=definition.codes,
        status=SectionStatus.COMPUTED,
        acute=definition.acute,
        point=point,
        lifetime=lifetime,
        strata=tuple(strata),
        expected=tuple(
            ExpectedComparison(expected, by_measure[expected.measure])
            for expected in definition.expected
            # a definition shared with the incidence report may carry its measure too
            if expected.measure in by_measure
        ),
        metrics=metrics,
        notes=tuple(notes),
    )


def patients_with(
    definition: ConditionDefinition, records: Records, rule: str = "point"
) -> pd.Index:
    """The alive patients who have ``definition`` at the reference date under ``rule``.

    ``rule`` is ``"point"`` (a record active at the reference date) or ``"lifetime"`` (a
    record started on or before it): the numerators of :func:`condition_prevalence`, so a
    cohort defined by a condition is exactly the patients its prevalence counts.
    """
    if rule not in MEASURES:
        raise ValueError(f"rule must be one of {list(MEASURES)}, not {rule!r}")
    match, _, _ = _match(definition, records)
    rows = records.frame[match]
    if rule == "point":
        rows = rows[rows["point"]]
    return pd.Index(rows["PATIENT"].unique())


def _match(
    definition: ConditionDefinition, records: Records
) -> tuple[pd.Series, dict[str, int], list[str]]:
    """Rows of any of the definition's codes, records per code, and codes absent from the table."""
    frame = records.frame
    match = pd.Series(False, index=frame.index)
    per_code: dict[str, int] = {}
    missing_codes: list[str] = []
    for ref in definition.codes:
        this = frame["CODE"] == ref.code
        if ref.system is not None and records.has_system:
            this &= frame["SYSTEM"] == ref.system
        match |= this
        per_code[_label(ref.system, ref.code)] = int(this.sum())
        in_table = any(
            code == ref.code
            and (ref.system is None or not records.has_system or system == ref.system)
            for system, code in records.table_codes
        )
        if not in_table:
            missing_codes.append(_label(ref.system, ref.code))
    return match, per_code, missing_codes


def _stratum(
    dimension: str,
    value: str | None,
    members: pd.Index,
    point_ids: pd.Index,
    lifetime_ids: pd.Index,
) -> Stratum:
    return Stratum(
        dimension=dimension,
        value=value,
        point=Rate(int(point_ids.isin(members).sum()), len(members)),
        lifetime=Rate(int(lifetime_ids.isin(members).sum()), len(members)),
    )


def _label(system: str | None, code: str) -> str:
    return code if system is None else f"{system}|{code}"


# --------------------------------------------------------------------------- #
# general table
# --------------------------------------------------------------------------- #


def general_table(
    records: Records,
    cohort: Cohort,
    *,
    include_social: bool = False,
    top: int = DEFAULT_TOP,
    exclude: ExclusionList | None = None,
) -> GeneralTable:
    """Every code with at least one alive patient, by point then lifetime prevalence.

    With ``exclude`` the codes of that file are left out instead of the built-in
    social and administrative list; the two options are mutually exclusive.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    if exclude is not None and include_social:
        raise ValueError("an exclusion file and --include-social cannot be combined")
    frame = records.frame
    keys = ["SYSTEM", "CODE"]
    grouped = (
        frame.groupby(keys)
        .agg(lifetime=("PATIENT", "nunique"), records=("PATIENT", "size"))
        .join(frame[frame["point"]].groupby(keys)["PATIENT"].nunique().rename("point"))
        .fillna({"point": 0})
        .reset_index()
    )
    if exclude is None:
        grouped["social"] = [
            is_social(system or None, code) for system, code in zip(grouped["SYSTEM"], grouped["CODE"])
        ]
    else:
        excluded = set(exclude.codes)
        grouped["social"] = [str(code) in excluded for code in grouped["CODE"]]
    social_records = int(grouped.loc[grouped["social"], "records"].sum())
    social_codes = int(grouped["social"].sum())
    if exclude is not None or not include_social:
        grouped = grouped[~grouped["social"]]
    grouped = grouped.sort_values(
        ["point", "lifetime", "SYSTEM", "CODE"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    rows = tuple(
        GeneralRow(
            system=row.SYSTEM or None,
            code=str(row.CODE),
            description=records.descriptions.get((row.SYSTEM, row.CODE)),
            point=Rate(int(row.point), cohort.size),
            lifetime=Rate(int(row.lifetime), cohort.size),
            social=bool(row.social),
        )
        for row in grouped.itertuples(index=False)
    )
    metrics: dict[str, Any] = {
        "alive": cohort.size,
        "codes": len(rows),
        "social_codes_in_data": social_codes,
        "social_records_in_data": social_records,
        **{f"records_{k}": v for k, v in records.metrics.items()},
    }
    notes = [
        "Sorted by point prevalence, then lifetime prevalence, then system and code; the JSON "
        "lists every code.",
    ]
    if exclude is not None:
        short = exclude.sha256[:12]
        notes.append(
            f"Codes listed in {exclude.path} are excluded: {social_codes} code(s) and "
            f"{social_records} record(s) of alive patients (file sha256 {short}, "
            f"{len(exclude.codes)} code(s) listed, {exclude.duplicates} duplicate(s)). "
            f"Conditions asked for by code are never filtered."
        )
        absent = codes_absent_from_data(
            exclude, {code for _, code in records.table_codes}
        )
        if absent:
            listed = ", ".join(f"`{code}`" for code in absent)
            notes.append(
                f"{len(absent)} listed code(s) are not in conditions.csv "
                f"(any patient, any date): {listed}."
            )
        return GeneralTable(
            status=SectionStatus.COMPUTED,
            rows=rows,
            top=top,
            include_social=False,
            metrics=metrics,
            notes=tuple(notes),
        )
    list_note = (
        f"list {SOCIAL_LIST_ID}: {len(SOCIAL_CODES)} codes written by "
        f"{' and '.join(SOURCE_MODULES)}"
    )
    if include_social:
        notes.append(
            f"Social and administrative codes are included ({social_codes} code(s), "
            f"{social_records} record(s); {list_note}); they are marked in the table."
        )
    else:
        notes.append(
            f"Social and administrative codes are excluded: {social_codes} code(s) and "
            f"{social_records} record(s) of alive patients ({list_note}). Run with "
            f"--include-social to include them. Conditions asked for by code are never "
            f"filtered."
        )
    return GeneralTable(
        status=SectionStatus.COMPUTED,
        rows=rows,
        top=top,
        include_social=include_social,
        metrics=metrics,
        notes=tuple(notes),
    )
