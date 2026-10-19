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
from synthea_quality.prevalence.compute import condition_rows
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.profile.dates import parse_date_only
from synthea_quality.profile.population import age_band_labels

#: Age band of a patient whose ``BIRTHDATE`` is empty or unparseable.
UNKNOWN_AGE = "unknown age"


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
            band = labels[band_of(age, bounds)]
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
