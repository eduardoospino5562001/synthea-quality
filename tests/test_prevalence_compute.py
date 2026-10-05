"""Tests for point and lifetime prevalence, strata and the general table."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from synthea_quality.prevalence.compute import (
    CONDITION_COLUMNS,
    PATIENT_COLUMNS,
    build_cohort,
    condition_prevalence,
    general_table,
    prepare_records,
)
from synthea_quality.prevalence.definitions import ConditionDefinition, parse_condition_option
from synthea_quality.prevalence.models import SNOMED_CT, CodeRef, Expected
from synthea_quality.profile.models import SectionStatus

REF = date(2026, 8, 17)
BANDS = (0, 18, 65)
SNOMED = "http://snomed.info/sct"


def text_frame(rows: list[dict[str, str]], columns) -> pd.DataFrame:
    """Rows as the loader produces them: text, an empty field as the only null."""
    data = pd.DataFrame([[row.get(c, "") for c in columns] for row in rows], columns=list(columns))
    return data.replace("", None).astype("str") if len(data) else data.astype("str")


def patients(*rows) -> pd.DataFrame:
    return text_frame(list(rows), PATIENT_COLUMNS)


def conditions(*rows, columns=CONDITION_COLUMNS) -> pd.DataFrame:
    full = [{"SYSTEM": SNOMED, "DESCRIPTION": "Thing", **row} for row in rows]
    return text_frame(full, columns)


def population():
    return patients(
        {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},  # 76
        {"Id": "a2", "BIRTHDATE": "1990-01-01", "GENDER": "F"},  # 36
        {"Id": "a3", "BIRTHDATE": "2010-01-01", "GENDER": "F"},  # 16
        {"Id": "a4", "BIRTHDATE": "1980-01-01", "GENDER": "M"},  # 46
        {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2020-01-01", "GENDER": "M"},
    )


def run(condition_rows, definition="C=100", *, people=None, bands=BANDS):
    cohort = build_cohort(people if people is not None else population(), REF, bands)
    records = prepare_records(conditions(*condition_rows), cohort, REF)
    return condition_prevalence(parse_condition_option(definition), records, cohort), records


# --------------------------------------------------------------------------- #
# exact boundaries against the reference date
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "start, stop, point, lifetime",
    [
        ("2026-08-17", "", 1, 1),            # starts on the reference day: active
        ("2026-08-18", "", 0, 0),            # starts the day after: not yet
        ("2026-08-16", "2026-08-17", 0, 1),  # stops on the reference day: ended
        ("2026-08-16", "2026-08-18", 1, 1),  # stops the day after: active
        ("2020-01-01", "2020-06-01", 0, 1),  # resolved long ago
        ("2020-01-01", "", 1, 1),            # never stopped
    ],
)
def test_start_and_stop_against_the_reference_date(start, stop, point, lifetime):
    result, _ = run([{"PATIENT": "a1", "CODE": "100", "START": start, "STOP": stop}])
    assert (result.point.numerator, result.lifetime.numerator) == (point, lifetime)
    assert result.point.denominator == result.lifetime.denominator == 4


def test_unusable_dates_are_counted_and_left_out():
    result, records = run(
        [
            {"PATIENT": "a1", "CODE": "100", "START": "", "STOP": ""},
            {"PATIENT": "a2", "CODE": "100", "START": "someday", "STOP": ""},
            {"PATIENT": "a3", "CODE": "100", "START": "2020-01-01", "STOP": "garbage"},
        ]
    )
    assert records.metrics["rows_start_unusable"] == 2
    assert records.metrics["rows_stop_unparseable"] == 1
    # an unparseable STOP keeps the record for lifetime, not for point
    assert (result.point.numerator, result.lifetime.numerator) == (0, 1)


def test_deceased_and_unknown_patients_are_counted_and_left_out():
    result, records = run(
        [
            {"PATIENT": "d1", "CODE": "100", "START": "2010-01-01"},
            {"PATIENT": "zz", "CODE": "100", "START": "2010-01-01"},
            {"PATIENT": "a1", "CODE": "", "START": "2010-01-01"},
        ]
    )
    assert result.lifetime.numerator == 0
    m = records.metrics
    assert (m["rows_deceased"], m["rows_unknown_patient"], m["rows_without_code"]) == (1, 1, 1)


# --------------------------------------------------------------------------- #
# grouped codes
# --------------------------------------------------------------------------- #


def test_a_patient_with_several_codes_of_a_condition_counts_once():
    result, _ = run(
        [
            {"PATIENT": "a1", "CODE": "100", "START": "2020-01-01", "STOP": "2020-01-05"},
            {"PATIENT": "a1", "CODE": "200", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "200", "START": "2021-01-01", "STOP": "2021-02-01"},
            {"PATIENT": "a3", "CODE": "300", "START": "2021-01-01"},  # not in the group
        ],
        "MI=100,200",
    )
    assert (result.point.numerator, result.lifetime.numerator) == (1, 2)
    assert result.metrics["records"] == 3
    assert result.metrics["records_by_code"] == {f"{SNOMED}|100": 1, f"{SNOMED}|200": 2}
    assert result.metrics["records_without_stop"] == 1


def test_a_code_is_matched_with_its_system():
    result, _ = run(
        [
            {"PATIENT": "a1", "CODE": "100", "START": "2020-01-01", "SYSTEM": "http://loinc.org"},
            {"PATIENT": "a2", "CODE": "100", "START": "2020-01-01"},
        ]
    )
    assert result.lifetime.numerator == 1


def test_a_code_absent_from_the_table_is_reported():
    result, _ = run([{"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"}], "C=100,999")
    assert any("No record of" in note and "999" in note for note in result.notes)


def test_a_table_without_system_matches_by_code():
    cohort = build_cohort(population(), REF, BANDS)
    frame = conditions(
        {"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"},
        columns=("PATIENT", "CODE", "START", "STOP", "DESCRIPTION"),
    )
    records = prepare_records(frame, cohort, REF)
    definition = ConditionDefinition("C", (CodeRef("100", SNOMED_CT),))
    assert condition_prevalence(definition, records, cohort).lifetime.numerator == 1


# --------------------------------------------------------------------------- #
# strata
# --------------------------------------------------------------------------- #


def test_strata_use_the_alive_of_each_stratum_as_denominator():
    result, _ = run(
        [
            {"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"},  # 65+, M
            {"PATIENT": "a3", "CODE": "100", "START": "2020-01-01", "STOP": "2021-01-01"},  # 0-17 F
        ]
    )
    by = {(s.dimension, s.value): s for s in result.strata}
    assert [(s.dimension, s.value) for s in result.strata] == [
        ("age_band", "0-17"), ("age_band", "18-64"), ("age_band", "65+"),
        ("GENDER", "F"), ("GENDER", "M"),
    ]
    child, adult, male = by["age_band", "0-17"], by["age_band", "18-64"], by["GENDER", "M"]
    assert (child.point.numerator, child.point.denominator) == (0, 1)
    assert child.lifetime.numerator == 1
    assert (adult.lifetime.numerator, adult.lifetime.denominator) == (0, 2)
    assert (male.point.numerator, male.point.denominator) == (1, 2)


def test_an_empty_stratum_keeps_its_row_without_a_rate():
    result, _ = run(
        [{"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"}], bands=(0, 18, 65, 100)
    )
    empty = next(s for s in result.strata if s.value == "100+")
    assert (empty.point.denominator, empty.point.value, empty.point.interval) == (0, None, None)


def test_patients_without_age_or_gender_are_handled_explicitly():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "", "GENDER": ""},
    )
    result, _ = run([{"PATIENT": "a2", "CODE": "100", "START": "2020-01-01"}], people=people)
    assert result.metrics["alive_without_age"] == 1
    assert sum(s.lifetime.denominator for s in result.strata if s.dimension == "age_band") == 1
    empty_gender = next(s for s in result.strata if s.dimension == "GENDER" and s.value is None)
    assert (empty_gender.lifetime.numerator, empty_gender.lifetime.denominator) == (1, 1)
    assert any("without a known age" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# notes and expected values
# --------------------------------------------------------------------------- #


def test_the_acute_note_appears_when_point_is_far_below_lifetime():
    rows = [
        {"PATIENT": p, "CODE": "100", "START": "2020-01-01", "STOP": "2020-01-02"}
        for p in ("a1", "a2", "a3", "a4")
    ]
    result, _ = run(rows)
    assert (result.point.numerator, result.lifetime.numerator) == (0, 4)
    assert any("acute events" in note for note in result.notes)
    chronic, _ = run([{"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"}])
    assert not any("acute events" in note for note in chronic.notes)


def test_expected_values_are_placed_inside_or_outside_the_interval():
    cohort = build_cohort(population(), REF, BANDS)
    records = prepare_records(
        conditions({"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"}), cohort, REF
    )
    definition = ConditionDefinition(
        "C", (CodeRef("100", SNOMED_CT),), (Expected("lifetime", 0.25), Expected("point", 0.99))
    )
    result = condition_prevalence(definition, records, cohort)
    positions = {e.expected.measure: e.position for e in result.expected}
    assert positions == {"lifetime": "inside", "point": "outside"}


# --------------------------------------------------------------------------- #
# general table and the social list
# --------------------------------------------------------------------------- #


def general(rows, **kwargs):
    cohort = build_cohort(population(), REF, BANDS)
    return general_table(prepare_records(conditions(*rows), cohort, REF), cohort, **kwargs)


GENERAL_ROWS = [
    {"PATIENT": "a1", "CODE": "73595000", "START": "2020-01-01", "DESCRIPTION": "Stress (finding)"},
    {"PATIENT": "a2", "CODE": "73595000", "START": "2020-01-01", "DESCRIPTION": "Stress (finding)"},
    {"PATIENT": "a1", "CODE": "714628002", "START": "2020-01-01",
     "DESCRIPTION": "Prediabetes (finding)"},
    {"PATIENT": "a1", "CODE": "200", "START": "2020-01-01", "STOP": "2020-02-01"},
    {"PATIENT": "a2", "CODE": "200", "START": "2020-01-01", "STOP": "2020-02-01"},
    {"PATIENT": "a3", "CODE": "300", "START": "2020-01-01"},
]


def test_general_table_excludes_the_social_list_by_default_and_says_so():
    table = general(GENERAL_ROWS)
    codes = [row.code for row in table.rows]
    assert "73595000" not in codes
    assert "714628002" in codes  # a clinical (finding) stays
    assert table.metrics["social_codes_in_data"] == 1
    assert table.metrics["social_records_in_data"] == 2
    assert any("--include-social" in note and "excluded" in note for note in table.notes)


def test_general_table_can_include_the_social_list():
    table = general(GENERAL_ROWS, include_social=True)
    stress = next(row for row in table.rows if row.code == "73595000")
    assert stress.social and table.include_social
    assert (stress.point.numerator, stress.lifetime.numerator) == (2, 2)


def test_general_table_order_is_point_then_lifetime_then_code():
    table = general(GENERAL_ROWS)
    assert [(r.code, r.point.numerator, r.lifetime.numerator) for r in table.rows] == [
        ("300", 1, 1),
        ("714628002", 1, 1),
        ("200", 0, 2),
    ]
    assert table.rows[1].description == "Prediabetes (finding)"


ACUTE_NOTE = (
    "records have no STOP date, so point prevalence counts every past event as still "
    "active; for an acute condition, lifetime prevalence is the meaningful measure."
)
UNSTOPPED = [
    {"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"},
    {"PATIENT": "a2", "CODE": "100", "START": "2021-01-01", "STOP": "2021-01-05"},
    {"PATIENT": "a3", "CODE": "100", "START": "2022-01-01"},
]


def test_every_condition_reports_its_records_without_stop():
    result, _ = run(UNSTOPPED)
    assert (result.metrics["records_without_stop"], result.metrics["records"]) == (2, 3)
    assert result.acute is False
    assert not any(ACUTE_NOTE in note for note in result.notes)  # never inferred


def test_a_declared_acute_condition_with_unstopped_records_gets_the_note():
    result, _ = run(UNSTOPPED, "C=100;acute")
    assert result.acute is True
    assert f"2 of 3 {ACUTE_NOTE}" in result.notes


def test_a_declared_acute_condition_whose_records_all_stop_gets_no_note():
    rows = [{"PATIENT": "a1", "CODE": "100", "START": "2020-01-01", "STOP": "2020-01-03"}]
    result, _ = run(rows, "C=100;acute")
    assert not any(ACUTE_NOTE in note for note in result.notes)


def test_expected_values_of_other_reports_are_left_to_them():
    cohort = build_cohort(population(), REF, BANDS)
    records = prepare_records(
        conditions({"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"}), cohort, REF
    )
    definition = ConditionDefinition(
        "C", (CodeRef("100", SNOMED_CT),), (Expected("incidence", 5.0), Expected("point", 0.2))
    )
    result = condition_prevalence(definition, records, cohort)
    assert [e.expected.measure for e in result.expected] == ["point"]


# --------------------------------------------------------------------------- #
# the patients a condition selects (cohorts)
# --------------------------------------------------------------------------- #


def test_patients_with_is_the_numerator_of_each_measure():
    from synthea_quality.prevalence.compute import patients_with

    result, records = run(
        [
            {"PATIENT": "a1", "CODE": "100", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "100", "START": "2020-01-01", "STOP": "2021-01-01"},
            {"PATIENT": "d1", "CODE": "100", "START": "2010-01-01"},
        ]
    )
    definition = parse_condition_option("C=100")
    point = patients_with(definition, records, "point")
    lifetime = patients_with(definition, records, "lifetime")
    assert sorted(point) == ["a1"]
    assert sorted(lifetime) == ["a1", "a2"]
    assert (len(point), len(lifetime)) == (result.point.numerator, result.lifetime.numerator)
    assert sorted(patients_with(definition, records)) == ["a1"]  # point by default
    with pytest.raises(ValueError):
        patients_with(definition, records, "incidence")


# --------------------------------------------------------------------------- #
# no condition records in range (header-only, after the reference, deceased)
# --------------------------------------------------------------------------- #


def test_general_table_with_no_condition_rows_is_computed_and_empty():
    table = general([])
    assert table.status is SectionStatus.COMPUTED
    assert table.rows == ()
    assert table.metrics["codes"] == 0


def test_general_table_with_all_records_after_the_reference_is_empty():
    table = general([{"PATIENT": "a1", "CODE": "100", "START": "2027-01-01"}])
    assert table.status is SectionStatus.COMPUTED
    assert table.rows == ()
    assert table.metrics["codes"] == 0


def test_general_table_with_only_deceased_records_is_empty():
    table = general([{"PATIENT": "d1", "CODE": "100", "START": "2010-01-01"}])
    assert table.status is SectionStatus.COMPUTED
    assert table.rows == ()
    assert table.metrics["codes"] == 0


def test_a_requested_condition_with_no_records_in_range_reports_zero_with_absent_note():
    result, _ = run([], "C=999")
    assert (result.point.numerator, result.point.denominator) == (0, 4)
    assert (result.lifetime.numerator, result.lifetime.denominator) == (0, 4)
    assert any("No record of" in note and "999" in note for note in result.notes)
