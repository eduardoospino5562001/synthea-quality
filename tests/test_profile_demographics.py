"""Tests for alive selection, ages and the demographic sections of a profile."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from synthea_quality.profile.dates import parse_date_only
from synthea_quality.profile.demographics import (
    PATIENT_COLUMNS,
    profile_patients,
    section_ids,
    skipped_patient_sections,
)
from synthea_quality.profile.models import SectionStatus
from synthea_quality.profile.population import (
    DEFAULT_AGE_BANDS,
    age_band_labels,
    age_band_of,
    ages_at,
    alive_mask,
    alive_patient_ids,
    select_alive,
    validate_age_bands,
)

REFERENCE = date(2026, 8, 17)


def patients(rows: list[dict[str, str]], columns=PATIENT_COLUMNS) -> pd.DataFrame:
    """A patients frame as the loader produces it: text, an empty field as the only null."""
    data = {c: [row.get(c, "") for row in rows] for c in columns}
    frame = pd.DataFrame(data, dtype="str")
    return frame.replace("", None).astype("str")


def by_value(section, name):
    return {row.value: (row.count, row.percent) for row in section.distributions[name]}


def compute(frame, reference=REFERENCE, **kwargs):
    sections = profile_patients(
        frame,
        reference=reference,
        reference_reason=None if reference else "no reference date: test",
        age_bands=kwargs.pop("age_bands", DEFAULT_AGE_BANDS),
        **kwargs,
    )
    return {section.section_id: section for section in sections}


# --------------------------------------------------------------------------- #
# population helpers
# --------------------------------------------------------------------------- #


def test_alive_means_an_empty_deathdate_even_when_the_deathdate_is_garbage():
    frame = patients(
        [{"DEATHDATE": ""}, {"DEATHDATE": "2020-01-01"}, {"DEATHDATE": "not a date"}]
    )
    assert alive_mask(frame).tolist() == [True, False, False]
    assert len(select_alive(frame)) == 1


def test_alive_patient_ids_are_the_ids_of_the_alive_rows():
    frame = patients(
        [
            {"Id": "p1", "DEATHDATE": ""},
            {"Id": "p2", "DEATHDATE": "2020-01-01"},
            {"Id": "p3", "DEATHDATE": ""},
            {"Id": "", "DEATHDATE": ""},
        ],
        columns=("Id", "DEATHDATE"),
    )
    assert list(alive_patient_ids(frame)) == ["p1", "p3"]


def test_alive_patient_ids_need_id_and_deathdate():
    with pytest.raises(KeyError):
        alive_patient_ids(patients([{"DEATHDATE": ""}], columns=("DEATHDATE",)))
    with pytest.raises(KeyError):
        alive_patient_ids(patients([{"Id": "p1"}], columns=("Id",)))


def test_alive_mask_needs_a_deathdate_column():
    with pytest.raises(KeyError):
        alive_mask(patients([{"GENDER": "F"}], columns=("GENDER",)))


@pytest.mark.parametrize(
    "birthdate, expected",
    [
        ("2026-08-17", 0),  # born on the reference date
        ("2021-08-17", 5),  # fifth birthday on the reference date
        ("2021-08-18", 4),  # fifth birthday the day after
        ("2000-02-29", 26),
        ("1926-08-18", 99),
    ],
)
def test_age_is_completed_years_by_calendar_birthday(birthdate, expected):
    parsed = parse_date_only(pd.Series([birthdate], dtype="str"))
    assert ages_at(parsed.values, REFERENCE).tolist() == [expected]


def test_a_leap_day_birthday_is_reached_on_the_first_of_march():
    parsed = parse_date_only(pd.Series(["2004-02-29"], dtype="str")).values
    assert ages_at(parsed, date(2025, 2, 28)).tolist() == [20]
    assert ages_at(parsed, date(2025, 3, 1)).tolist() == [21]


def test_unknown_and_future_birthdates_have_no_age():
    parsed = parse_date_only(pd.Series(["2030-01-01", None, "bad"], dtype="str")).values
    assert ages_at(parsed, REFERENCE).isna().tolist() == [True, True, True]


def test_age_band_labels_and_validation():
    assert age_band_labels(DEFAULT_AGE_BANDS) == ("0-4", "5-17", "18-44", "45-64", "65+")
    assert age_band_labels((0, 18)) == ("0-17", "18+")
    assert age_band_labels((0,)) == ("0+",)
    for bad in ((), (5, 18), (0, 18, 18), (0, 18, 5), (0, 1.5)):
        with pytest.raises(ValueError):
            validate_age_bands(bad)


def test_every_band_boundary_falls_in_the_upper_band():
    ages = pd.Series([0, 4, 5, 17, 18, 44, 45, 64, 65, 110, None], dtype="Int64")
    assert age_band_of(ages, DEFAULT_AGE_BANDS).tolist() == [
        "0-4", "0-4", "5-17", "5-17", "18-44", "18-44", "45-64", "45-64", "65+", "65+", None,
    ]


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def mixed_population() -> pd.DataFrame:
    """Nine alive patients on every band boundary, two deceased, and bad values."""
    alive = [
        ("2026-08-17", "F", "white", "nonhispanic", "Massachusetts", "Suffolk County"),
        ("2021-08-18", "M", "white", "nonhispanic", "Massachusetts", "Suffolk County"),  # 4
        ("2021-08-17", "F", "black", "hispanic", "Massachusetts", "Middlesex County"),  # 5
        ("2008-08-18", "M", "asian", "nonhispanic", "Massachusetts", "Middlesex County"),  # 17
        ("2008-08-17", "F", "white", "nonhispanic", "Massachusetts", ""),  # 18
        ("1961-08-18", "M", "white", "", "Massachusetts", "Essex County"),  # 64
        ("1961-08-17", "F", "white", "nonhispanic", "Massachusetts", "Essex County"),  # 65
        ("", "F", "white", "nonhispanic", "Massachusetts", "Essex County"),  # empty
        ("1990-13-01", "M", "white", "nonhispanic", "Massachusetts", "Essex County"),  # bad
        ("2027-01-01", "F", "white", "nonhispanic", "Massachusetts", "Essex County"),  # future
    ]
    rows = [
        dict(zip(("BIRTHDATE", "GENDER", "RACE", "ETHNICITY", "STATE", "COUNTY"), values))
        for values in alive
    ]
    rows.append({"BIRTHDATE": "1930-11-06", "DEATHDATE": "2018-09-23", "GENDER": "M"})
    rows.append({"BIRTHDATE": "1940-01-01", "DEATHDATE": "someday", "GENDER": "F"})
    return patients(rows)


def test_sections_come_in_report_order_and_all_compute():
    sections = profile_patients(
        mixed_population(), reference=REFERENCE, reference_reason=None,
        age_bands=DEFAULT_AGE_BANDS,
    )
    assert [s.section_id for s in sections] == [sid for sid, _ in section_ids()]
    assert all(s.status is SectionStatus.COMPUTED for s in sections)


def test_population_counts_alive_and_deceased():
    metrics = compute(mixed_population())["population"].metrics
    assert metrics["total"] == 12
    assert metrics["alive"] == 10
    assert metrics["deceased"] == 2
    assert metrics["alive"] + metrics["deceased"] == metrics["total"]
    assert metrics["alive_percent"] == 83.33
    assert metrics["deceased_with_unparseable_deathdate"] == 1
    assert metrics["deaths_after_reference_date"] == 0


def test_age_section_bands_boundaries_and_counts_exclusions():
    age = compute(mixed_population())["age"]
    metrics = age.metrics
    assert metrics["alive"] == 10
    assert metrics["denominator"] == 7
    assert metrics["excluded"] == {
        "birthdate_empty": 1,
        "birthdate_unparseable": 1,
        "born_after_reference_date": 1,
    }
    assert metrics["min"] == 0
    assert metrics["max"] == 65
    assert metrics["median"] == 17.0  # ages 0, 4, 5, 17, 18, 64, 65
    assert by_value(age, "age_band") == {
        "0-4": (2, 28.57),
        "5-17": (2, 28.57),
        "18-44": (1, 14.29),
        "45-64": (1, 14.29),
        "65+": (1, 14.29),
    }
    assert [row.value for row in age.distributions["age_band"]] == [
        "0-4", "5-17", "18-44", "45-64", "65+",
    ]


def test_age_bands_are_configurable():
    age = compute(mixed_population(), age_bands=(0, 18))["age"]
    assert by_value(age, "age_band") == {"0-17": (4, 57.14), "18+": (3, 42.86)}


def test_age_section_is_skipped_without_a_reference_date():
    age = compute(mixed_population(), reference=None)["age"]
    assert age.status is SectionStatus.SKIPPED
    assert age.reason == "no reference date: test"


def test_distributions_describe_only_the_alive_and_list_empty_values():
    sections = compute(mixed_population())
    gender = sections["distribution.GENDER"]
    assert gender.metrics["denominator"] == 10
    assert by_value(gender, "GENDER") == {"F": (6, 60.0), "M": (4, 40.0)}
    ethnicity = sections["distribution.ETHNICITY"]
    assert by_value(ethnicity, "ETHNICITY") == {
        "nonhispanic": (8, 80.0), "hispanic": (1, 10.0), None: (1, 10.0),
    }
    assert ethnicity.distributions["ETHNICITY"][-1].value is None
    assert sum(row.count for row in ethnicity.distributions["ETHNICITY"]) == 10


def test_distribution_order_is_count_then_value():
    race = compute(mixed_population())["distribution.RACE"]
    assert [row.value for row in race.distributions["RACE"]] == ["white", "asian", "black"]


def test_county_lists_the_top_values_and_sums_the_rest():
    county = compute(mixed_population(), top_counties=2)["distribution.COUNTY"]
    assert [row.value for row in county.distributions["COUNTY"]] == [
        "Essex County", "Middlesex County", None,
    ]
    assert county.metrics["distinct_values"] == 3
    assert county.metrics["shown"] == 2
    assert county.metrics["other_values"] == 1
    assert county.metrics["other_count"] == 2
    assert county.metrics["empty"] == 1


def test_county_defaults_to_the_top_ten():
    county = compute(mixed_population())["distribution.COUNTY"]
    assert county.metrics["shown"] == 3
    assert county.metrics["other_count"] == 0


def test_date_range_covers_every_patient_and_counts_bad_values():
    metrics = compute(mixed_population())["date_range"].metrics
    assert metrics["BIRTHDATE"] == {
        "min": "1930-11-06", "max": "2027-01-01", "valid": 10, "empty": 1, "unparseable": 1,
    }
    assert metrics["DEATHDATE"] == {
        "min": "2018-09-23", "max": "2018-09-23", "valid": 1, "empty": 10, "unparseable": 1,
    }


def test_completeness_reports_every_used_column():
    metrics = compute(mixed_population())["completeness"].metrics
    assert list(metrics) == sorted(PATIENT_COLUMNS)
    assert metrics["BIRTHDATE"] == {"rows": 12, "empty": 1, "unparseable": 1}
    assert metrics["COUNTY"] == {"rows": 12, "empty": 3, "unparseable": None}


def test_a_missing_categorical_column_skips_only_its_distribution():
    columns = tuple(c for c in PATIENT_COLUMNS if c != "STATE")
    frame = patients([{"BIRTHDATE": "2000-01-01", "GENDER": "F"}], columns=columns)
    sections = compute(frame)
    assert sections["distribution.STATE"].status is SectionStatus.SKIPPED
    assert "no STATE column" in sections["distribution.STATE"].reason
    assert sections["distribution.GENDER"].status is SectionStatus.COMPUTED
    assert any("STATE" in note for note in sections["completeness"].notes)


def test_without_deathdate_everything_about_the_alive_is_skipped():
    columns = tuple(c for c in PATIENT_COLUMNS if c != "DEATHDATE")
    sections = compute(patients([{"BIRTHDATE": "2000-01-01", "GENDER": "F"}], columns=columns))
    for section_id in ("population", "age", "distribution.GENDER"):
        assert sections[section_id].status is SectionStatus.SKIPPED
        assert "DEATHDATE" in sections[section_id].reason
    assert sections["date_range"].status is SectionStatus.COMPUTED
    assert sections["completeness"].status is SectionStatus.COMPUTED


def test_empty_table_computes_zeros_without_inventing_statistics():
    sections = compute(patients([]))
    assert sections["population"].metrics["total"] == 0
    assert sections["age"].metrics["median"] is None
    assert sections["date_range"].metrics["BIRTHDATE"]["min"] is None


def test_skipped_patient_sections_cover_every_section():
    sections = skipped_patient_sections("patients.csv is not in the dataset")
    assert [s.section_id for s in sections] == [sid for sid, _ in section_ids()]
    assert {s.reason for s in sections} == {"patients.csv is not in the dataset"}


def test_a_cut_inside_a_tie_is_reported():
    # Essex 5, Middlesex 2, Norfolk 2, Suffolk 1: a top-2 cut splits the tie at 2.
    rows = [{"COUNTY": "Essex County"}] * 5 + [
        {"COUNTY": "Middlesex County"},
        {"COUNTY": "Middlesex County"},
        {"COUNTY": "Norfolk County"},
        {"COUNTY": "Norfolk County"},
        {"COUNTY": "Suffolk County"},
    ]
    frame = patients([dict(row, BIRTHDATE="2000-01-01") for row in rows])
    county = compute(frame, top_counties=2)["distribution.COUNTY"]
    assert [row.value for row in county.distributions["COUNTY"]] == [
        "Essex County", "Middlesex County",
    ]
    assert county.metrics["tie_at_cut"] == {"count": 2, "values": 2, "listed": 1}
    assert (
        "2 values tie at count 2; ties are broken alphabetically "
        "(1 listed, 1 summed in other_count)."
    ) in county.notes


def test_a_cut_between_different_counts_has_no_tie_note():
    rows = [{"COUNTY": "A"}] * 3 + [{"COUNTY": "B"}] * 2 + [{"COUNTY": "C"}]
    frame = patients([dict(row, BIRTHDATE="2000-01-01") for row in rows])
    for limit in (1, 2, 5):
        county = compute(frame, top_counties=limit)["distribution.COUNTY"]
        assert "tie_at_cut" not in county.metrics
        assert not any("tie at count" in note for note in county.notes)
