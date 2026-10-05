"""[PROFILE] Who is alive, and how old they are at the reference date.

These are the building blocks every population analysis needs, kept free of any
report or command-line concern so prevalence and incidence can reuse them unchanged.

Alive
-----
Following the Synthea maintainer's guidance, analyses focus on the patients **alive at
the end of the simulation**; the deceased are extra records. The CSV export states it
directly: ``CSVExporter`` writes ``DEATHDATE`` only for a person who died, so a patient is
alive exactly when ``DEATHDATE`` is empty. A ``DEATHDATE`` that is present but not a
valid date still says the patient died; it is counted as unparseable by the callers
that parse it, never reinterpreted as "alive".

Age
---
Completed years at the reference date, by calendar birthday rather than by dividing a
number of days by 365: someone born on 2000-06-15 is 25 on 2025-06-15 and 24 the day
before. A person born on 29 February has their birthday on 1 March in a common year.
A birth date after the reference date has no age; callers count it instead of turning
it into a negative number.

Age bands
---------
Given as increasing lower bounds starting at 0. ``(0, 5, 18, 45, 65)`` produces
``0-4``, ``5-17``, ``18-44``, ``45-64`` and ``65+``, so every age falls in exactly one
band and a patient who turns 5 on the reference date is in ``5-17``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

#: The rule, as the report states it.
ALIVE_RULE = (
    "a patient is alive at the end of the simulation when patients.DEATHDATE is empty "
    "(CSVExporter writes DEATHDATE only for a patient who died)"
)

#: Default lower bounds of the age bands.
DEFAULT_AGE_BANDS: tuple[int, ...] = (0, 5, 18, 45, 65)


def alive_mask(patients: pd.DataFrame) -> pd.Series:
    """Boolean mask of the patients alive at the end of the simulation.

    :raises KeyError: ``patients`` has no ``DEATHDATE`` column, so aliveness is unknown.
    """
    if "DEATHDATE" not in patients.columns:
        raise KeyError("DEATHDATE")
    return patients["DEATHDATE"].isna()


def select_alive(patients: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``patients`` alive at the end of the simulation."""
    return patients[alive_mask(patients)]


def alive_patient_ids(patients: pd.DataFrame) -> pd.Index:
    """Identifiers (``patients.Id``) of the patients alive at the end of the simulation.

    This is what a clinical table's ``PATIENT`` column is matched against: a record
    belongs to the alive cohort exactly when its ``PATIENT`` is in this index. Empty
    identifiers are left out, since no record can reference them.

    :raises KeyError: ``patients`` has no ``Id`` or no ``DEATHDATE`` column.
    """
    if "Id" not in patients.columns:
        raise KeyError("Id")
    ids = patients.loc[alive_mask(patients), "Id"].dropna()
    return pd.Index(ids.unique(), name="Id")


def ages_at(birthdates: pd.Series, reference: date) -> pd.Series:
    """Completed years at ``reference`` for parsed ``birthdates``.

    :param birthdates: datetimes, ``NaT`` where unknown (see
        :func:`synthea_quality.profile.dates.parse_date_only`).
    :returns: a nullable integer series on the same index; missing where the birth date
        is unknown or after ``reference``.
    """
    known = birthdates.notna()
    years = birthdates.dt.year
    before_birthday = (birthdates.dt.month > reference.month) | (
        (birthdates.dt.month == reference.month) & (birthdates.dt.day > reference.day)
    )
    ages = (reference.year - years - before_birthday.astype(int)).astype("Int64")
    after_reference = known & (birthdates > pd.Timestamp(reference))
    return ages.where(known & ~after_reference)


def validate_age_bands(bounds: Sequence[int]) -> tuple[int, ...]:
    """Check that ``bounds`` are strictly increasing integers starting at 0.

    :raises ValueError: the bounds cannot describe a partition of ages.
    """
    values = tuple(bounds)
    if not values:
        raise ValueError("at least one age band is needed")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError(f"age band bounds must be integers: {list(values)}")
    if values[0] != 0:
        raise ValueError(f"the first age band must start at 0: {list(values)}")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"age band bounds must be strictly increasing: {list(values)}")
    return values


def age_band_labels(bounds: Sequence[int]) -> tuple[str, ...]:
    """Labels of the bands, e.g. ``("0-4", "5-17", ..., "65+")``."""
    values = validate_age_bands(bounds)
    labels = [f"{low}-{high - 1}" for low, high in zip(values, values[1:])]
    labels.append(f"{values[-1]}+")
    return tuple(labels)


def age_band_of(ages: pd.Series, bounds: Sequence[int]) -> pd.Series:
    """The band label of every age (missing where the age is missing)."""
    values = validate_age_bands(bounds)
    labels = age_band_labels(values)
    edges = [*values, float("inf")]
    banded = pd.cut(ages.astype("float"), bins=edges, right=False, labels=labels)
    return banded.astype("object").where(ages.notna(), None)
