"""[PROFILE] Population and demographic sections, computed from ``patients.csv`` alone.

Sections, in report order:

``population``        total, alive and deceased patients, over every row;
``age``               the alive patients' age at the reference date: bands, median,
                      minimum and maximum;
``distribution.<C>``  the alive patients by ``GENDER``, ``RACE``, ``ETHNICITY`` and
                      ``STATE``, and the top counties by ``COUNTY``;
``date_range``        earliest and latest ``BIRTHDATE`` and ``DEATHDATE``, over every row;
``completeness``      empty and unparseable values of every column the profile used.

Choices a reader should know about:

* **Denominators are explicit.** Distributions are shares of the alive patients, and an
  empty value is its own row (``value: null``), so the rows of a distribution add up to
  the denominator. The age bands are shares of the alive patients that *have* an age;
  those without one are counted under ``excluded``.
* **Nothing is dropped silently.** A value that is empty or does not parse is counted
  where the section uses it and again, per column, in ``completeness``.
* **Order is deterministic.** Distribution rows are ordered by count, highest first,
  then by value; the empty row comes last. Age bands keep their natural order.
* **A missing column skips only what needs it.** Without ``STATE`` only the state
  distribution is skipped; without ``DEATHDATE`` every section that depends on who is
  alive is skipped, because guessing aliveness would misstate every number.

This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import pandas as pd

from synthea_quality.profile.dates import ParsedDates, parse_date_only
from synthea_quality.profile.models import (
    CategoryCount,
    ProfileSection,
    SectionStatus,
    percent,
)
from synthea_quality.profile.population import (
    ALIVE_RULE,
    age_band_labels,
    age_band_of,
    ages_at,
    alive_mask,
)

#: Every ``patients`` column the profile reads.
PATIENT_COLUMNS = ("BIRTHDATE", "DEATHDATE", "GENDER", "RACE", "ETHNICITY", "STATE", "COUNTY")
#: Columns holding a ``YYYY-MM-DD`` date.
DATE_COLUMNS = ("BIRTHDATE", "DEATHDATE")
#: Categorical distributions: column, title, and how many values to list (``None``: all).
DISTRIBUTIONS: tuple[tuple[str, str, int | None], ...] = (
    ("GENDER", "Gender", None),
    ("RACE", "Race", None),
    ("ETHNICITY", "Ethnicity", None),
    ("STATE", "State", None),
    ("COUNTY", "County", 10),
)

POPULATION = ("population", "Population")
AGE = ("age", "Age at the reference date")
DATE_RANGE = ("date_range", "Date range")
COMPLETENESS = ("completeness", "Empty and unparseable values")


def section_ids() -> tuple[tuple[str, str], ...]:
    """Identifier and title of every section, in report order."""
    return (
        POPULATION,
        AGE,
        *((f"distribution.{column}", title) for column, title, _ in DISTRIBUTIONS),
        DATE_RANGE,
        COMPLETENESS,
    )


def skipped_patient_sections(reason: str) -> tuple[ProfileSection, ...]:
    """Every section, skipped for the same reason (e.g. no ``patients.csv``)."""
    return tuple(ProfileSection.skipped(sid, title, reason) for sid, title in section_ids())


def profile_patients(
    patients: pd.DataFrame,
    *,
    reference: date | None,
    reference_reason: str | None,
    age_bands: Sequence[int],
    top_counties: int | None = None,
) -> tuple[ProfileSection, ...]:
    """Compute every section from ``patients`` (the columns of :data:`PATIENT_COLUMNS`
    that the file has, loaded as text).

    :param reference: date to measure ages at; ``None`` skips the age section.
    :param reference_reason: why there is no reference date, shown when it is ``None``.
    :param top_counties: override how many counties to list.
    """
    parsed = {c: parse_date_only(patients[c]) for c in DATE_COLUMNS if c in patients.columns}
    has_deathdate = "DEATHDATE" in patients.columns
    alive = alive_mask(patients) if has_deathdate else None
    no_deathdate = (
        "patients.csv has no DEATHDATE column, so it cannot be told who is alive "
        f"({ALIVE_RULE})"
    )

    sections: list[ProfileSection] = []
    sections.append(
        _population(patients, alive, parsed, reference)
        if alive is not None
        else ProfileSection.skipped(*POPULATION, no_deathdate)
    )
    sections.append(
        _age(patients, alive, parsed, reference, reference_reason, age_bands)
        if alive is not None
        else ProfileSection.skipped(*AGE, no_deathdate)
    )
    for column, title, limit in DISTRIBUTIONS:
        if column == "COUNTY" and top_counties is not None:
            limit = top_counties
        section_id = f"distribution.{column}"
        if alive is None:
            sections.append(ProfileSection.skipped(section_id, title, no_deathdate))
        elif column not in patients.columns:
            sections.append(
                ProfileSection.skipped(section_id, title, f"patients.csv has no {column} column")
            )
        else:
            sections.append(
                _distribution(section_id, title, patients.loc[alive, column], column, limit)
            )
    sections.append(_date_range(parsed))
    sections.append(_completeness(patients, parsed))
    return tuple(sections)


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def _population(
    patients: pd.DataFrame,
    alive: pd.Series,
    parsed: dict[str, ParsedDates],
    reference: date | None,
) -> ProfileSection:
    total = len(patients)
    alive_count = int(alive.sum())
    deceased = total - alive_count
    metrics: dict[str, Any] = {
        "total": total,
        "alive": alive_count,
        "alive_percent": percent(alive_count, total),
        "deceased": deceased,
        "deceased_percent": percent(deceased, total),
        "deceased_with_unparseable_deathdate": parsed["DEATHDATE"].unparseable,
    }
    notes = [
        f"Alive: {ALIVE_RULE}.",
        "The rest of the profile describes the alive patients; deceased patients are "
        "counted here and in the date range only.",
    ]
    if reference is not None:
        after = int((parsed["DEATHDATE"].values > pd.Timestamp(reference)).sum())
        metrics["deaths_after_reference_date"] = after
        if after:
            notes.append(
                f"{after} death date(s) fall after the reference date {reference.isoformat()}."
            )
    return ProfileSection(
        section_id=POPULATION[0],
        title=POPULATION[1],
        status=SectionStatus.COMPUTED,
        metrics=metrics,
        notes=tuple(notes),
    )


def _age(
    patients: pd.DataFrame,
    alive: pd.Series,
    parsed: dict[str, ParsedDates],
    reference: date | None,
    reference_reason: str | None,
    age_bands: Sequence[int],
) -> ProfileSection:
    if reference is None:
        return ProfileSection.skipped(*AGE, reference_reason or "no reference date")
    if "BIRTHDATE" not in parsed:
        return ProfileSection.skipped(*AGE, "patients.csv has no BIRTHDATE column")

    birthdates = parsed["BIRTHDATE"].values[alive]
    # An unknown birth date is either an empty field or a value that did not parse.
    empty = int(patients.loc[alive, "BIRTHDATE"].isna().sum())
    unparseable = int(birthdates.isna().sum()) - empty
    ages = ages_at(birthdates, reference)
    born_after = int((birthdates > pd.Timestamp(reference)).sum())
    known = ages.dropna().astype(int)

    bands = age_band_of(ages, age_bands)
    band_counts = bands.value_counts()
    rows = tuple(
        CategoryCount(label, count, percent(count, len(known)))
        for label in age_band_labels(age_bands)
        for count in (int(band_counts.get(label, 0)),)
    )
    metrics = {
        "alive": int(alive.sum()),
        "denominator": len(known),
        "reference_date": reference.isoformat(),
        "median": float(known.median()) if len(known) else None,
        "min": int(known.min()) if len(known) else None,
        "max": int(known.max()) if len(known) else None,
        "excluded": {
            "birthdate_empty": empty,
            "birthdate_unparseable": unparseable,
            "born_after_reference_date": born_after,
        },
    }
    return ProfileSection(
        section_id=AGE[0],
        title=AGE[1],
        status=SectionStatus.COMPUTED,
        metrics=metrics,
        distributions={"age_band": rows},
        notes=(
            "Age is completed years at the reference date, by calendar birthday.",
            "Band percentages are shares of the alive patients with a known age "
            "(the denominator); the others are counted under excluded.",
        ),
    )


def _distribution(
    section_id: str, title: str, values: pd.Series, column: str, limit: int | None
) -> ProfileSection:
    denominator = len(values)
    empty = int(values.isna().sum())
    counts = values.dropna().value_counts()
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    shown = ordered if limit is None else ordered[:limit]
    rest = ordered[len(shown):]

    rows = [
        CategoryCount(str(value), int(count), percent(int(count), denominator))
        for value, count in shown
    ]
    if empty:
        rows.append(CategoryCount(None, empty, percent(empty, denominator)))
    metrics: dict[str, Any] = {
        "denominator": denominator,
        "distinct_values": len(ordered),
        "empty": empty,
    }
    if limit is not None:
        other_count = sum(int(count) for _, count in rest)
        metrics.update(
            {
                "shown": len(shown),
                "other_values": len(rest),
                "other_count": other_count,
                "other_percent": percent(other_count, denominator),
            }
        )
    notes = ["Shares of the alive patients; an empty value is listed as its own row."]
    if limit is not None:
        notes.append(
            f"Only the {limit} most frequent values are listed; the rest are summed in "
            f"other_count."
        )
    return ProfileSection(
        section_id=section_id,
        title=title,
        status=SectionStatus.COMPUTED,
        metrics=metrics,
        distributions={column: tuple(rows)},
        notes=tuple(notes),
    )


def _date_range(parsed: dict[str, ParsedDates]) -> ProfileSection:
    if not parsed:
        return ProfileSection.skipped(
            *DATE_RANGE, "patients.csv has neither a BIRTHDATE nor a DEATHDATE column"
        )
    metrics: dict[str, Any] = {}
    for column, dates in parsed.items():
        valid = dates.values.dropna()
        metrics[column] = {
            "min": valid.min().date().isoformat() if len(valid) else None,
            "max": valid.max().date().isoformat() if len(valid) else None,
            "valid": len(valid),
            "empty": dates.empty,
            "unparseable": dates.unparseable,
        }
    notes = ["Over every patient, alive or deceased; unparseable values are not included."]
    if "DEATHDATE" in parsed:
        notes.append("An empty DEATHDATE is expected: it marks a patient who is alive.")
    missing = [c for c in DATE_COLUMNS if c not in parsed]
    if missing:
        notes.append(f"Not in patients.csv: {', '.join(missing)}.")
    return ProfileSection(
        section_id=DATE_RANGE[0],
        title=DATE_RANGE[1],
        status=SectionStatus.COMPUTED,
        metrics=metrics,
        notes=tuple(notes),
    )


def _completeness(patients: pd.DataFrame, parsed: dict[str, ParsedDates]) -> ProfileSection:
    present = [c for c in PATIENT_COLUMNS if c in patients.columns]
    if not present:
        return ProfileSection.skipped(
            *COMPLETENESS,
            f"patients.csv has none of the columns the profile uses: {list(PATIENT_COLUMNS)}",
        )
    metrics: dict[str, Any] = {}
    for column in present:
        metrics[column] = {
            "rows": len(patients),
            "empty": int(patients[column].isna().sum()),
            "unparseable": parsed[column].unparseable if column in parsed else None,
        }
    notes = [
        "Over every patient. unparseable applies to date columns only (null elsewhere).",
    ]
    missing = [c for c in PATIENT_COLUMNS if c not in patients.columns]
    if missing:
        notes.append(f"Not in patients.csv, so not profiled: {', '.join(missing)}.")
    return ProfileSection(
        section_id=COMPLETENESS[0],
        title=COMPLETENESS[1],
        status=SectionStatus.COMPUTED,
        metrics=metrics,
        notes=tuple(notes),
    )

