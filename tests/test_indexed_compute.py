"""Tests for the indexed cohort: one row per patient with an index date."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from synthea_quality.indexed.compute import (
    UNKNOWN_AGE,
    build_indexed_cohort,
    death_members,
    event_members,
    measure_cross,
    measure_death,
    measure_event,
    measure_value,
    skipped_cross,
    value_members,
)
from synthea_quality.indexed.definitions import (
    CrossSpec,
    DeathOutcome,
    EventMatch,
    EventOutcome,
    ValueOutcome,
    Window,
)
from synthea_quality.indexed.models import ValueShare
from synthea_quality.prevalence.compute import CONDITION_COLUMNS, PATIENT_COLUMNS
from synthea_quality.prevalence.definitions import parse_condition_option

REF = date(2026, 1, 1)
BANDS = (0, 18, 65)
SNOMED = "http://snomed.info/sct"
COVID = parse_condition_option("COVID-19=840539006")
HOSPITAL = "1505002"
EVENT_COLUMNS = ("PATIENT", "CODE", "START")


def text_frame(rows: list[dict[str, str]], columns) -> pd.DataFrame:
    """Rows as the loader produces them: text, an empty field as the only null."""
    data = pd.DataFrame([[row.get(c, "") for c in columns] for row in rows], columns=list(columns))
    return data.replace("", None).astype("str") if len(data) else data.astype("str")


def patients(*rows) -> pd.DataFrame:
    return text_frame(list(rows), PATIENT_COLUMNS)


def conditions(*rows, columns=CONDITION_COLUMNS) -> pd.DataFrame:
    full = [{"SYSTEM": SNOMED, "DESCRIPTION": "COVID-19", **row} for row in rows]
    return text_frame(full, columns)


def run(people, records, definition=COVID):
    return build_indexed_cohort(people, records, definition, REF, BANDS)


def test_t1_a_patient_with_three_records_is_one_patient():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-02-01"},
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-03-01"},
        ),
    )
    assert cohort.size == 1
    assert cohort.rows_matched == 3


def test_t2_the_index_date_is_the_earliest_start():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-05-01"},
            {"PATIENT": "a1", "CODE": "840539006", "START": "2019-03-02"},
        ),
    )
    assert cohort.frame.loc["a1", "INDEX_DATE"] == pd.Timestamp("2019-03-02")


def test_t3_a_deceased_patient_is_in_the_cohort_with_their_death_date():
    cohort = run(
        patients(
            {"Id": "d1", "BIRTHDATE": "1950-01-01", "DEATHDATE": "2021-07-08", "GENDER": "F"}
        ),
        conditions({"PATIENT": "d1", "CODE": "840539006", "START": "2020-01-01"}),
    )
    assert cohort.size == 1
    assert cohort.frame.loc["d1", "DEATH_DATE"] == pd.Timestamp("2021-07-08")


def test_t4_a_record_after_the_reference_date_is_left_out_and_counted():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a1", "CODE": "840539006", "START": "2026-06-01"},
        ),
    )
    assert cohort.size == 1
    assert cohort.frame.loc["a1", "INDEX_DATE"] == pd.Timestamp("2020-01-01")
    assert cohort.rows_after_reference == 1


def test_t5_a_record_of_a_patient_outside_patients_is_left_out_and_counted():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "z9", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    assert cohort.size == 1
    assert cohort.rows_unknown_patient == 1
    assert cohort.rows_matched == 2


def test_t6_an_unparseable_start_is_left_out_and_counted():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-13-45"}),
    )
    assert cohort.size == 0
    assert cohort.rows_start_unusable == 1


def test_t7_a_table_without_system_is_matched_by_code():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            columns=("PATIENT", "CODE", "START", "STOP", "DESCRIPTION"),
        ),
    )
    assert cohort.size == 1


def test_t8_a_different_system_is_not_the_condition():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01",
             "SYSTEM": "http://loinc.org"}
        ),
    )
    assert cohort.size == 0


def test_t9_the_age_band_is_the_age_on_the_index_date():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "2000-06-01", "GENDER": "M"}),
        conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2017-06-01"}),
    )
    assert cohort.frame.loc["a1", "AGE_BAND"] == "0-17"


def test_t10_an_empty_birthdate_has_an_unknown_age_band():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "", "GENDER": "M"}),
        conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}),
    )
    assert cohort.frame.loc["a1", "AGE_BAND"] == UNKNOWN_AGE


def test_t13_a_birth_after_the_index_date_has_an_unknown_age_band():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "2021-01-01", "GENDER": "M"}),
        conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}),
    )
    assert cohort.frame.loc["a1", "AGE_BAND"] == UNKNOWN_AGE


def test_t11_no_record_of_the_condition_gives_an_empty_cohort():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions({"PATIENT": "a1", "CODE": "999", "START": "2020-01-01"}),
    )
    assert cohort.size == 0
    assert cohort.rows_matched == 0
    assert cohort.rows_unknown_patient == 0
    assert cohort.rows_start_unusable == 0
    assert cohort.rows_after_reference == 0


def test_t12_either_code_of_the_definition_selects_its_patient():
    definition = parse_condition_option("COVID-19=840539006,111")
    cohort = run(
        patients(
            {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        ),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "111", "START": "2020-01-01"},
        ),
        definition,
    )
    assert cohort.size == 2


def encounters(*rows, columns=EVENT_COLUMNS) -> pd.DataFrame:
    return text_frame(list(rows), columns)


def event(window=None, codes=(HOSPITAL,), table="encounters") -> EventOutcome:
    return EventOutcome(
        "Hospital admission", table, tuple(codes), Window() if window is None else window
    )


def single_patient_cohort(index_start: str):
    return run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions({"PATIENT": "a1", "CODE": "840539006", "START": index_start}),
    )


def test_e1_three_records_of_one_patient_are_one_patient():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters(
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"},
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-03-01T10:00:00Z"},
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-04-01T10:00:00Z"},
        ),
        event(),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.rows_matched == 3


def test_e2_a_record_before_the_index_is_outside_the_default_window():
    result = measure_event(
        single_patient_cohort("2020-01-11"),
        encounters({"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-01T10:00:00Z"}),
        event(),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 0
    assert result.rows_outside_window == 1


def test_e3_the_same_record_counts_with_a_negative_from_days():
    result = measure_event(
        single_patient_cohort("2020-01-11"),
        encounters({"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-01T10:00:00Z"}),
        event(window=Window(-30, None)),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.rows_outside_window == 0


def test_e4_to_days_includes_day_thirty_and_excludes_day_thirty_one():
    cohort = run(
        patients(
            {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        ),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    result = measure_event(
        cohort,
        encounters(
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-31T10:00:00Z"},
            {"PATIENT": "a2", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"},
        ),
        event(window=Window(0, 30)),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.rows_outside_window == 1
    assert result.days.n == 1
    assert result.days.minimum == 30


def test_e5_a_record_after_the_reference_date_does_not_count():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters({"PATIENT": "a1", "CODE": HOSPITAL, "START": "2026-06-01T10:00:00Z"}),
        event(),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 0
    assert result.rows_after_reference == 1


def test_e6_an_unparseable_record_date_does_not_count():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters({"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-13-45T10:00:00Z"}),
        event(),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 0
    assert result.rows_date_unusable == 1


def test_e7_a_record_of_a_patient_outside_the_cohort_is_ignored():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters({"PATIENT": "z9", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"}),
        event(),
        REF,
        BANDS,
    )
    assert result.rows_matched == 0
    assert result.breakdown.total.patients == 0


def test_e8_days_are_measured_to_the_first_record_of_the_window():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters(
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-21T10:00:00Z"},
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-06T10:00:00Z"},
        ),
        event(),
        REF,
        BANDS,
    )
    assert result.days.n == 1
    assert result.days.minimum == 5.0


def test_e9_a_timestamp_and_a_date_only_table_both_count_day_zero():
    cohort = single_patient_cohort("2020-01-01")
    timestamp = measure_event(
        cohort,
        encounters({"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-01T10:00:00Z"}),
        event(),
        REF,
        BANDS,
    )
    date_only = measure_event(
        cohort,
        encounters(
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-01"},
            columns=("PATIENT", "CODE", "START"),
        ),
        event(table="careplans"),
        REF,
        BANDS,
    )
    assert timestamp.breakdown.total.patients == 1
    assert timestamp.days.minimum == 0.0
    assert date_only.breakdown.total.patients == 1
    assert date_only.days.minimum == 0.0


def test_e10_the_breakdown_is_by_sex_and_age_at_the_index_date():
    cohort = run(
        patients(
            {"Id": "f10", "BIRTHDATE": "2007-06-01", "GENDER": "F"},
            {"Id": "f40", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
            {"Id": "m40", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        ),
        conditions(
            {"PATIENT": "f10", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "f40", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "m40", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    result = measure_event(
        cohort,
        encounters({"PATIENT": "f40", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"}),
        event(),
        REF,
        BANDS,
    )
    total = result.breakdown.total
    assert (total.patients, total.cohort) == (1, 3)
    assert [(s.label, s.patients, s.cohort) for s in result.breakdown.by_sex] == [
        ("F", 1, 2),
        ("M", 0, 1),
    ]
    assert [(s.label, s.patients, s.cohort) for s in result.breakdown.by_age] == [
        ("0-17", 0, 1),
        ("18-64", 1, 2),
        ("65+", 0, 0),
    ]


def test_e11_an_empty_cohort_has_no_patients_and_no_days():
    cohort = run(
        patients({"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"}),
        conditions({"PATIENT": "a1", "CODE": "999", "START": "2020-01-01"}),
    )
    result = measure_event(cohort, encounters(), event(), REF, BANDS)
    assert (result.breakdown.total.patients, result.breakdown.total.cohort) == (0, 0)
    assert result.days.n == 0


def test_e12_either_code_of_the_outcome_counts_its_patient():
    cohort = run(
        patients(
            {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        ),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    result = measure_event(
        cohort,
        encounters(
            {"PATIENT": "a1", "CODE": "111", "START": "2020-02-01T10:00:00Z"},
            {"PATIENT": "a2", "CODE": "222", "START": "2020-02-01T10:00:00Z"},
        ),
        event(codes=("111", "222")),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 2


STAGE = "21908-9"
BREAST = "Malignant neoplasm of breast (disorder)"
HEART = "Chronic congestive heart failure (disorder)"
OBSERVATION_COLUMNS = ("PATIENT", "CODE", "DATE", "VALUE")


def observations(*rows) -> pd.DataFrame:
    return text_frame(list(rows), OBSERVATION_COLUMNS)


def value_outcome(pick="first", window=None) -> ValueOutcome:
    return ValueOutcome(
        "Stage group", STAGE, pick, Window() if window is None else window
    )


def death_outcome(cause_values=(), window=None) -> DeathOutcome:
    return DeathOutcome(
        "Death", tuple(cause_values), Window() if window is None else window
    )


def test_v1_pick_first_takes_the_value_of_the_earliest_date():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": "A"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-10T10:00:00Z", "VALUE": "B"},
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert result.values == (ValueShare("A", 1),)


def test_v2_pick_last_takes_the_value_of_the_latest_date():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": "A"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-10T10:00:00Z", "VALUE": "B"},
        ),
        value_outcome("last"),
        REF,
        BANDS,
    )
    assert result.values == (ValueShare("B", 1),)


def test_v3_two_values_on_the_picked_day_count_the_patient_in_both():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 2"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "stage IIA"},
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert result.values == (
        ValueShare("Stage 2", 1),
        ValueShare("stage IIA", 1),
    )
    assert result.patients_with_several_values == 1
    assert result.breakdown.total.patients == 1


def test_v4_the_same_value_in_three_rows_is_one_patient():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 2"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T11:00:00Z",
             "VALUE": "Stage 2"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T12:00:00Z",
             "VALUE": "Stage 2"},
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert result.values == (ValueShare("Stage 2", 1),)


def test_v5_an_empty_value_in_the_window_does_not_count():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": ""}
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert result.values == ()
    assert result.breakdown.total.patients == 0
    assert result.rows_value_empty == 1


def test_v6_a_value_before_the_index_is_outside_the_default_window():
    result = measure_value(
        single_patient_cohort("2020-01-11"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-01T10:00:00Z", "VALUE": "A"}
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert result.values == ()
    assert result.rows_outside_window == 1


def test_v7_values_are_ordered_by_patients_then_text():
    cohort = run(
        patients(
            {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
            {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a4", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        ),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a3", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a4", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    result = measure_value(
        cohort,
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": "X"},
            {"PATIENT": "a2", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": "X"},
            {"PATIENT": "a3", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": "A"},
            {"PATIENT": "a4", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": "B"},
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert result.values == (
        ValueShare("X", 2),
        ValueShare("A", 1),
        ValueShare("B", 1),
    )


def test_v8_a_patient_without_a_value_stays_in_the_denominator():
    cohort = run(
        patients(
            {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
            {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        ),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a3", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    result = measure_value(
        cohort,
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z", "VALUE": "A"}
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert (result.breakdown.total.patients, result.breakdown.total.cohort) == (1, 3)


def test_d1_one_death_in_a_cohort_of_three():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    cohort = run(
        people,
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a3", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    result = measure_death(cohort, people, None, death_outcome(), REF, BANDS)
    assert (result.breakdown.total.patients, result.breakdown.total.cohort) == (1, 3)
    assert result.days.n == 1


def test_d2_a_death_on_the_index_date_counts_as_day_zero():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-03-01", "GENDER": "M"}
    )
    cohort = run(people, conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-03-01"}))
    result = measure_death(cohort, people, None, death_outcome(), REF, BANDS)
    assert result.breakdown.total.patients == 1
    assert result.days.minimum == 0.0


def test_d3_a_death_after_the_reference_date_does_not_count():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2026-06-01", "GENDER": "M"}
    )
    cohort = run(people, conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}))
    result = measure_death(cohort, people, None, death_outcome(), REF, BANDS)
    assert result.breakdown.total.patients == 0
    assert result.deaths_after_reference == 1


def test_d4_an_unparseable_deathdate_does_not_count():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-02-30", "GENDER": "M"}
    )
    cohort = run(people, conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}))
    result = measure_death(cohort, people, None, death_outcome(), REF, BANDS)
    assert result.breakdown.total.patients == 0
    assert result.deathdate_unusable == 1


def test_d5_a_death_after_to_days_is_outside_the_window():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-02-15", "GENDER": "M"}
    )
    cohort = run(people, conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}))
    result = measure_death(
        cohort, people, None, death_outcome(window=Window(0, 30)), REF, BANDS
    )
    assert result.breakdown.total.patients == 0
    assert result.deaths_outside_window == 1


def test_d6_the_cause_of_death_is_the_value_of_69453_9():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-02", "GENDER": "F"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-03", "GENDER": "M"},
    )
    cohort = run(
        people,
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a3", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    result = measure_death(
        cohort,
        people,
        observations(
            {"PATIENT": "a1", "CODE": "69453-9", "DATE": "2020-06-05T10:00:00Z",
             "VALUE": BREAST},
            {"PATIENT": "a2", "CODE": "69453-9", "DATE": "2020-06-05T10:00:00Z",
             "VALUE": HEART},
        ),
        death_outcome((BREAST,)),
        REF,
        BANDS,
    )
    assert (result.with_cause, result.other_cause, result.no_cause_record) == (1, 1, 1)
    assert (
        result.with_cause + result.other_cause + result.no_cause_record
        == result.breakdown.total.patients
    )


def test_d7_without_cause_values_the_three_cause_counts_are_none():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"}
    )
    cohort = run(people, conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}))
    result = measure_death(
        cohort,
        people,
        observations(
            {"PATIENT": "a1", "CODE": "69453-9", "DATE": "2020-06-05T10:00:00Z",
             "VALUE": BREAST}
        ),
        death_outcome(),
        REF,
        BANDS,
    )
    assert (result.with_cause, result.other_cause, result.no_cause_record) == (None, None, None)


def test_d8_two_cause_rows_of_one_patient_count_one_patient():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"}
    )
    cohort = run(people, conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}))
    result = measure_death(
        cohort,
        people,
        observations(
            {"PATIENT": "a1", "CODE": "69453-9", "DATE": "2020-06-05T10:00:00Z",
             "VALUE": BREAST},
            {"PATIENT": "a1", "CODE": "69453-9", "DATE": "2020-06-05T11:00:00Z",
             "VALUE": BREAST},
        ),
        death_outcome((BREAST,)),
        REF,
        BANDS,
    )
    assert result.with_cause == 1


def test_d9_the_death_breakdown_is_by_sex_and_age_at_the_index_date():
    people = patients(
        {"Id": "f70", "BIRTHDATE": "1956-06-01", "DEATHDATE": "2020-06-01", "GENDER": "F"}
    )
    cohort = run(
        people, conditions({"PATIENT": "f70", "CODE": "840539006", "START": "2020-01-01"})
    )
    result = measure_death(cohort, people, None, death_outcome(), REF, BANDS)
    assert [(s.label, s.patients, s.cohort) for s in result.breakdown.by_sex] == [("F", 1, 1)]
    assert [(s.label, s.patients, s.cohort) for s in result.breakdown.by_age] == [
        ("0-17", 0, 0),
        ("18-64", 1, 1),
        ("65+", 0, 0),
    ]


def test_d10_a_death_whose_cause_rows_are_all_empty_has_no_cause_record():
    people = patients(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"}
    )
    cohort = run(people, conditions({"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"}))
    result = measure_death(
        cohort,
        people,
        observations(
            {"PATIENT": "a1", "CODE": "69453-9", "DATE": "2020-06-05T10:00:00Z", "VALUE": ""}
        ),
        death_outcome((BREAST,)),
        REF,
        BANDS,
    )
    assert (result.with_cause, result.other_cause, result.no_cause_record) == (0, 0, 1)


STAGE_GROUPS = (
    ("Stage 1", ("Stage 1 (qualifier value)", "Stage 1A (qualifier value)")),
    ("Stage 2", ("Stage 2 (qualifier value)", "Stage 2A (qualifier value)")),
)


def grouped_outcome(groups=STAGE_GROUPS, pick="first", window=None) -> ValueOutcome:
    """A value outcome whose groups join the texts of a stage."""
    return ValueOutcome(
        "Stage group", STAGE, pick, Window() if window is None else window, groups
    )


def test_g1_two_texts_of_the_same_group_count_as_one_value():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 1 (qualifier value)"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 1A (qualifier value)"},
        ),
        grouped_outcome(),
        REF,
        BANDS,
    )
    assert result.values == (ValueShare("Stage 1", 1),)
    assert result.patients_with_several_values == 0
    assert result.breakdown.total.patients == 1


def test_g2_two_texts_of_two_groups_count_in_both():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 1 (qualifier value)"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 2 (qualifier value)"},
        ),
        grouped_outcome(),
        REF,
        BANDS,
    )
    assert result.values == (ValueShare("Stage 1", 1), ValueShare("Stage 2", 1))
    assert result.patients_with_several_values == 1
    assert result.breakdown.total.patients == 1


def test_g3_a_text_outside_the_groups_keeps_its_text():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Unstaged"}
        ),
        grouped_outcome(),
        REF,
        BANDS,
    )
    assert result.values == (ValueShare("Unstaged", 1),)


def test_g4_without_groups_the_values_keep_their_text():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 1 (qualifier value)"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 1A (qualifier value)"},
        ),
        value_outcome("first"),
        REF,
        BANDS,
    )
    assert result.values == (
        ValueShare("Stage 1 (qualifier value)", 1),
        ValueShare("Stage 1A (qualifier value)", 1),
    )
    assert result.patients_with_several_values == 1


def test_g5_a_group_label_that_is_also_a_text_counts_them_together():
    result = measure_value(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 2A (qualifier value)"},
            {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": "Stage 2"},
        ),
        grouped_outcome((("Stage 2", ("Stage 2A (qualifier value)",)),)),
        REF,
        BANDS,
    )
    assert result.values == (ValueShare("Stage 2", 1),)
    assert result.patients_with_several_values == 0


def cohort_with(*rows):
    """A cohort of the given patients, each with the index condition on 2020-01-01."""
    people = patients(*rows)
    records = conditions(
        *[
            {"PATIENT": row["Id"], "CODE": "840539006", "START": "2020-01-01"}
            for row in rows
        ]
    )
    return people, run(people, records)


def test_m1_the_event_members_are_the_patients_of_the_measure():
    cohort = run(
        patients(
            {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
            {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        ),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a3", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    table = encounters(
        {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"},
        {"PATIENT": "a2", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"},
    )
    result = measure_event(cohort, table, event(), REF, BANDS)
    members = event_members(cohort, table, event(), REF)
    assert members.patients == frozenset({"a1", "a2"})
    assert len(members.patients) == result.breakdown.total.patients


def test_m2_the_value_members_are_the_patients_and_the_labels_of_the_measure():
    cohort = run(
        patients(
            {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
            {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        ),
        conditions(
            {"PATIENT": "a1", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a2", "CODE": "840539006", "START": "2020-01-01"},
            {"PATIENT": "a3", "CODE": "840539006", "START": "2020-01-01"},
        ),
    )
    table = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a2", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 2"},
    )
    result = measure_value(cohort, table, value_outcome(), REF, BANDS)
    members = value_members(cohort, table, value_outcome(), REF)
    assert members.patients == frozenset({"a1", "a2"})
    assert len(members.patients) == result.breakdown.total.patients
    assert members.values == {
        "Stage 1": frozenset({"a1"}),
        "Stage 2": frozenset({"a2"}),
    }


def test_m3_the_death_members_are_the_patients_and_the_cause_of_the_measure():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-02", "GENDER": "M"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
    )
    causes = observations(
        {"PATIENT": "a2", "CODE": "69453-9", "DATE": "2020-06-05T10:00:00Z", "VALUE": BREAST}
    )
    result = measure_death(cohort, people, causes, death_outcome((BREAST,)), REF, BANDS)
    members = death_members(cohort, people, causes, death_outcome((BREAST,)), REF)
    assert members.patients == frozenset({"a1", "a2"})
    assert len(members.patients) == result.breakdown.total.patients
    assert members.with_cause == frozenset({"a2"})
    assert len(members.with_cause) == result.with_cause


def test_m4_a_death_without_a_cause_has_no_cause_members():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"}
    )
    members = death_members(cohort, people, observations(), death_outcome(), REF)
    assert members.patients == frozenset({"a1"})
    assert members.with_cause is None


def test_x1_the_strata_of_a_value_cross_are_in_the_order_of_the_values():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-02", "GENDER": "F"},
        {"Id": "a4", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    stage = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a2", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a3", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 2"},
    )
    cross = measure_cross(
        cohort,
        CrossSpec("Death", "Stage group"),
        death_members(cohort, people, observations(), death_outcome(), REF),
        value_members(cohort, stage, value_outcome(), REF),
        "value",
    )
    assert (cross.outcome, cross.by, cross.by_type) == ("Death", "Stage group", "value")
    assert [stratum.label for stratum in cross.strata] == ["Stage 1", "Stage 2", "no value"]
    assert [(s.numerator, s.denominator) for s in cross.strata] == [(1, 2), (1, 1), (0, 1)]
    assert cross.patients_in_several_strata == 0


def test_x2_the_strata_of_an_event_cross_are_with_and_without_the_event():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-02", "GENDER": "F"},
        {"Id": "a4", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    table = encounters({"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"})
    cross = measure_cross(
        cohort,
        CrossSpec("Death", "Hospital admission"),
        death_members(cohort, people, observations(), death_outcome(), REF),
        event_members(cohort, table, event(), REF),
        "event",
    )
    assert [stratum.label for stratum in cross.strata] == [
        "with Hospital admission",
        "without Hospital admission",
    ]
    assert [(s.numerator, s.denominator) for s in cross.strata] == [(1, 1), (1, 3)]
    assert cross.patients_in_several_strata == 0


def test_x3_a_death_cross_with_a_cause_counts_the_cause_in_every_stratum():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-02", "GENDER": "F"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a4", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    stage = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a2", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a3", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 2"},
    )
    causes = observations(
        {"PATIENT": "a2", "CODE": "69453-9", "DATE": "2020-06-05T10:00:00Z", "VALUE": BREAST}
    )
    cross = measure_cross(
        cohort,
        CrossSpec("Death", "Stage group"),
        death_members(cohort, people, causes, death_outcome((BREAST,)), REF),
        value_members(cohort, stage, value_outcome(), REF),
        "value",
    )
    assert [
        (s.label, s.numerator, s.denominator, s.with_cause) for s in cross.strata
    ] == [
        ("Stage 1", 2, 2, 1),
        ("Stage 2", 0, 1, 0),
        ("no value", 0, 1, 0),
    ]
    assert sum(s.with_cause for s in cross.strata) == 1


def test_x4_a_value_outcome_counts_the_patients_with_a_value():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        {"Id": "a4", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    stage = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a2", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
    )
    table = encounters({"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"})
    cross = measure_cross(
        cohort,
        CrossSpec("Stage group", "Hospital admission"),
        value_members(cohort, stage, value_outcome(), REF),
        event_members(cohort, table, event(), REF),
        "event",
    )
    assert [(s.label, s.numerator, s.denominator) for s in cross.strata] == [
        ("with Hospital admission", 1, 1),
        ("without Hospital admission", 1, 3),
    ]
    assert all(s.with_cause is None for s in cross.strata)


def test_x5_a_patient_with_two_values_counts_in_every_stratum():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    stage = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 2"},
        {"PATIENT": "a2", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
    )
    cross = measure_cross(
        cohort,
        CrossSpec("Death", "Stage group"),
        death_members(cohort, people, observations(), death_outcome(), REF),
        value_members(cohort, stage, value_outcome(), REF),
        "value",
    )
    assert [(s.label, s.numerator, s.denominator) for s in cross.strata] == [
        ("Stage 1", 1, 2),
        ("Stage 2", 1, 1),
        ("no value", 0, 1),
    ]
    assert cross.patients_in_several_strata == 1


def test_x6_each_outcome_keeps_its_own_window():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-01-15", "GENDER": "M"},
    )
    stage = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
        {"PATIENT": "a2", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "Stage 1"},
    )
    cross = measure_cross(
        cohort,
        CrossSpec("Death", "Stage group"),
        death_members(cohort, people, observations(), death_outcome(window=Window(0, 30)), REF),
        value_members(cohort, stage, value_outcome(window=Window(0, 365)), REF),
        "value",
    )
    assert [(s.label, s.numerator, s.denominator) for s in cross.strata] == [
        ("Stage 1", 1, 2),
        ("no value", 0, 0),
    ]


def test_x7_a_skipped_cross_has_no_strata_and_its_reason():
    cross = skipped_cross(
        CrossSpec("Death", "Stage group"), "value", "Stage group", "observations.csv is missing"
    )
    assert cross.strata == ()
    assert cross.patients_in_several_strata == 0
    assert cross.reason == (
        "outcome 'Stage group' was not computed: observations.csv is missing"
    )


def test_x8_an_unknown_kind_of_strata_is_an_error():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "F"}
    )
    with pytest.raises(ValueError):
        measure_cross(
            cohort,
            CrossSpec("Death", "Stage group"),
            death_members(cohort, people, observations(), death_outcome(), REF),
            value_members(cohort, observations(), value_outcome(), REF),
            "death",
        )


def test_x9_a_value_with_the_no_value_text_skips_the_cross():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    stage = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "no value"},
    )
    cross = measure_cross(
        cohort,
        CrossSpec("Death", "Stage group"),
        death_members(cohort, people, observations(), death_outcome(), REF),
        value_members(cohort, stage, value_outcome(), REF),
        "value",
    )
    assert cross.strata == ()
    assert cross.patients_in_several_strata == 0
    assert cross.reason == (
        "a value of 'Stage group' is the text 'no value', the label of the patients "
        "without a value; put it in a group with another label"
    )


def test_x10_the_no_value_text_in_a_group_computes_the_cross():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "DEATHDATE": "2020-06-01", "GENDER": "F"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    stage = observations(
        {"PATIENT": "a1", "CODE": STAGE, "DATE": "2020-02-01T10:00:00Z", "VALUE": "no value"},
    )
    cross = measure_cross(
        cohort,
        CrossSpec("Death", "Stage group"),
        death_members(cohort, people, observations(), death_outcome(), REF),
        value_members(
            cohort, stage, grouped_outcome((("Undetermined", ("no value",)),)), REF
        ),
        "value",
    )
    assert cross.reason is None
    assert [(s.label, s.numerator, s.denominator) for s in cross.strata] == [
        ("Undetermined", 1, 1),
        ("no value", 0, 1),
    ]


CAREPLAN = "736376001"
COVID_CODE = "840539006"
SUSPECTED_CODE = "840544004"
NEGATIVE_TEST = "94531-1"
NEGATIVE = "Not detected (qualifier value)"
CAREPLAN_COLUMNS = ("PATIENT", "CODE", "START", "STOP", "REASONCODE")
ENCOUNTER_COLUMNS = ("PATIENT", "CODE", "START", "STOP", "REASONCODE")


def careplans(*rows) -> pd.DataFrame:
    return text_frame(list(rows), CAREPLAN_COLUMNS)


def encounters_with_stop(*rows) -> pd.DataFrame:
    return text_frame(list(rows), ENCOUNTER_COLUMNS)


def part(
    name: str,
    table: str,
    codes: tuple[str, ...],
    values: tuple[str, ...] = (),
    reasons: tuple[str, ...] = (),
    at: str = "start",
) -> EventMatch:
    """One part of an event outcome: its table, its codes and its filters."""
    return EventMatch(name, table, tuple(codes), tuple(values), tuple(reasons), at)


def composite(
    name: str, parts, op: str = "any", window: Window | None = None
) -> EventOutcome:
    """An event outcome of several parts."""
    return EventOutcome(
        name,
        None,
        (),
        Window() if window is None else window,
        matches=tuple(parts),
        op=op,
    )


def filtered_event(
    table: str,
    codes: tuple[str, ...],
    values: tuple[str, ...] = (),
    reasons: tuple[str, ...] = (),
    at: str = "start",
    window: Window | None = None,
) -> EventOutcome:
    """An event of one table with the filters of a part."""
    return EventOutcome(
        "Filtered",
        table,
        tuple(codes),
        Window() if window is None else window,
        values=tuple(values),
        reasons=tuple(reasons),
        at=at,
    )


def test_p1_only_the_records_of_the_codes_of_the_cohort_are_matched():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters(
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"},
            {"PATIENT": "a1", "CODE": "999", "START": "2020-02-01T10:00:00Z"},
            {"PATIENT": "z9", "CODE": HOSPITAL, "START": "2020-02-01T10:00:00Z"},
        ),
        event(),
        REF,
        BANDS,
    )
    assert result.rows_matched == 1
    assert result.matches[0].rows_matched == 1


def test_p2_a_record_whose_value_is_not_wanted_is_left_out_and_counted():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        observations(
            {"PATIENT": "a1", "CODE": NEGATIVE_TEST, "DATE": "2020-01-05T10:00:00Z",
             "VALUE": "Detected (qualifier value)"},
            {"PATIENT": "a1", "CODE": NEGATIVE_TEST, "DATE": "2020-01-06T10:00:00Z",
             "VALUE": NEGATIVE},
        ),
        filtered_event("observations", (NEGATIVE_TEST,), values=(NEGATIVE,)),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.rows_filtered_out == 1
    assert result.matches[0].rows_filtered_out == 1
    assert result.days.minimum == 5.0


def test_p3_a_record_whose_reason_is_not_wanted_is_left_out_and_counted():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-02",
             "STOP": "2020-01-10", "REASONCODE": SUSPECTED_CODE},
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-03",
             "STOP": "2020-01-20", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), reasons=(COVID_CODE,)),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.rows_filtered_out == 1
    assert result.matches[0].rows_filtered_out == 1
    assert result.days.minimum == 2.0


def test_p4_a_record_that_never_ended_is_left_out_and_counted():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-02",
             "STOP": "", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), at="stop"),
        REF,
        BANDS,
    )
    assert result.rows_not_ended == 1
    assert result.matches[0].rows_not_ended == 1
    assert result.rows_date_unusable == 0
    assert result.breakdown.total.patients == 0


def test_p5_a_record_with_an_unusable_date_is_left_out_and_counted():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-02",
             "STOP": "2020-13-45", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), at="stop"),
        REF,
        BANDS,
    )
    assert result.rows_date_unusable == 1
    assert result.rows_not_ended == 0
    assert result.breakdown.total.patients == 0


def test_p6_a_record_after_the_reference_date_is_left_out_and_counted():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-02",
             "STOP": "2026-06-01", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), at="stop"),
        REF,
        BANDS,
    )
    assert result.rows_after_reference == 1
    assert result.breakdown.total.patients == 0


def test_p7_a_record_outside_the_window_is_left_out_and_counted():
    result = measure_event(
        single_patient_cohort("2020-01-11"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-05", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), at="stop"),
        REF,
        BANDS,
    )
    assert result.rows_outside_window == 1
    assert result.breakdown.total.patients == 0


def test_p8_every_record_left_out_is_counted_in_one_counter():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-05", "REASONCODE": COVID_CODE},
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-05", "REASONCODE": SUSPECTED_CODE},
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "", "REASONCODE": COVID_CODE},
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-13-45", "REASONCODE": COVID_CODE},
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2026-06-01", "REASONCODE": COVID_CODE},
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2019-12-01",
             "STOP": "2019-12-20", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), reasons=(COVID_CODE,), at="stop"),
        REF,
        BANDS,
    )
    assert result.rows_matched == 6
    assert result.rows_filtered_out == 1
    assert result.rows_not_ended == 1
    assert result.rows_date_unusable == 1
    assert result.rows_after_reference == 1
    assert result.rows_outside_window == 1
    assert result.rows_matched == (
        result.rows_filtered_out
        + result.rows_not_ended
        + result.rows_date_unusable
        + result.rows_after_reference
        + result.rows_outside_window
        + 1
    )
    assert result.breakdown.total.patients == 1
    assert result.days.minimum == 4.0


def test_p9_a_record_left_out_by_two_steps_counts_in_the_first_one():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-02",
             "STOP": "", "REASONCODE": SUSPECTED_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), reasons=(COVID_CODE,), at="stop"),
        REF,
        BANDS,
    )
    assert result.rows_filtered_out == 1
    assert result.rows_not_ended == 0
    assert result.breakdown.total.patients == 0


def test_p10_a_careplan_counts_on_the_day_of_its_stop():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-02",
             "STOP": "2020-01-20", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), at="stop"),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.days.minimum == 19.0


def test_p11_an_encounter_counts_on_the_day_of_its_stop():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters_with_stop(
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-02T10:00:00Z",
             "STOP": "2020-02-01T10:00:00Z"},
        ),
        filtered_event("encounters", (HOSPITAL,), at="stop"),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.days.minimum == 31.0


def test_p12_a_record_that_started_before_the_index_and_stopped_in_the_window_counts():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2019-12-20",
             "STOP": "2020-01-05", "REASONCODE": COVID_CODE},
        ),
        filtered_event("careplans", (CAREPLAN,), at="stop"),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.days.minimum == 4.0
    assert result.rows_outside_window == 0


def recovered_outcome(op: str = "any") -> EventOutcome:
    """The event of several parts of the plan: an ended isolation or a negative test."""
    return composite(
        "Recovered",
        (
            part(
                "Isolation ended",
                "careplans",
                (CAREPLAN,),
                reasons=(COVID_CODE,),
                at="stop",
            ),
            part("Negative test", "observations", (NEGATIVE_TEST,), values=(NEGATIVE,)),
        ),
        op=op,
    )


def recovered_tables() -> dict[str, pd.DataFrame]:
    """The tables of the event of several parts: the careplans and the observations."""
    return {
        "careplans": careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-20", "REASONCODE": COVID_CODE},
            {"PATIENT": "a2", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-22", "REASONCODE": COVID_CODE},
        ),
        "observations": observations(
            {"PATIENT": "a1", "CODE": NEGATIVE_TEST, "DATE": "2020-01-05T10:00:00Z",
             "VALUE": NEGATIVE},
            {"PATIENT": "a2", "CODE": NEGATIVE_TEST, "DATE": "2020-01-10T10:00:00Z",
             "VALUE": NEGATIVE},
        ),
    }


def test_p13_the_members_of_an_event_of_several_parts_are_the_patients_of_its_measure():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
    )
    tables = recovered_tables()
    result = measure_event(cohort, tables, recovered_outcome(), REF, BANDS)
    members = event_members(cohort, tables, recovered_outcome(), REF)
    assert members.patients == frozenset({"a1", "a2"})
    assert len(members.patients) == result.breakdown.total.patients


def test_p14_a_patient_of_two_parts_counts_once_on_the_earliest_day():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
    )
    result = measure_event(cohort, recovered_tables(), recovered_outcome(), REF, BANDS)
    assert result.breakdown.total.patients == 2
    assert result.days.n == 2
    assert result.days.minimum == 4.0
    assert result.days.maximum == 9.0
    assert result.op == "any"
    assert [match.patients for match in result.matches] == [2, 2]


def test_p15_an_all_outcome_needs_every_part():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
    )
    tables = recovered_tables()
    result = measure_event(cohort, tables, recovered_outcome("all"), REF, BANDS)
    assert result.breakdown.total.patients == 2
    assert result.op == "all"


def test_p16_an_all_outcome_needs_a_patient_of_every_part():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
    )
    tables = {
        "careplans": careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-20", "REASONCODE": COVID_CODE},
            {"PATIENT": "a2", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-22", "REASONCODE": COVID_CODE},
        ),
        "observations": observations(
            {"PATIENT": "a1", "CODE": NEGATIVE_TEST, "DATE": "2020-01-05T10:00:00Z",
             "VALUE": NEGATIVE},
        ),
    }
    result = measure_event(cohort, tables, recovered_outcome("all"), REF, BANDS)
    assert result.breakdown.total.patients == 1
    assert result.days.minimum == 19.0
    assert [match.patients for match in result.matches] == [2, 1]


def test_p17_an_all_outcome_counts_on_the_latest_of_the_first_days_of_its_parts():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    tables = {
        "careplans": careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-03",
             "STOP": "2020-01-03", "REASONCODE": COVID_CODE},
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-11",
             "STOP": "2020-01-11", "REASONCODE": COVID_CODE},
        ),
        "observations": observations(
            {"PATIENT": "a1", "CODE": NEGATIVE_TEST, "DATE": "2020-01-06T10:00:00Z",
             "VALUE": NEGATIVE},
        ),
    }
    result = measure_event(cohort, tables, recovered_outcome("all"), REF, BANDS)
    assert result.breakdown.total.patients == 1
    assert result.days.minimum == 5.0


def test_p18_a_cross_counts_the_patients_of_an_event_of_several_parts():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "1980-01-01", "GENDER": "F"},
        {"Id": "a3", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    tables = {
        "careplans": careplans(
            {"PATIENT": "a1", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-05", "REASONCODE": COVID_CODE},
            {"PATIENT": "a2", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-07", "REASONCODE": COVID_CODE},
            {"PATIENT": "a3", "CODE": CAREPLAN, "START": "2020-01-01",
             "STOP": "2020-01-09", "REASONCODE": COVID_CODE},
        ),
        "observations": observations(
            {"PATIENT": "a1", "CODE": NEGATIVE_TEST, "DATE": "2020-01-02T10:00:00Z",
             "VALUE": NEGATIVE},
            {"PATIENT": "a3", "CODE": NEGATIVE_TEST, "DATE": "2020-01-04T10:00:00Z",
             "VALUE": NEGATIVE},
        ),
    }
    recovered = recovered_outcome("any")
    isolation = composite(
        "Isolation ended",
        (
            part(
                "Isolation ended",
                "careplans",
                (CAREPLAN,),
                reasons=(COVID_CODE,),
                at="stop",
            ),
            part("Negative test", "observations", (NEGATIVE_TEST,), values=(NEGATIVE,)),
        ),
        op="all",
    )
    cross = measure_cross(
        cohort,
        CrossSpec("Recovered", "Isolation ended"),
        event_members(cohort, tables, recovered, REF),
        event_members(cohort, tables, isolation, REF),
        "event",
    )
    assert [(s.label, s.numerator, s.denominator) for s in cross.strata] == [
        ("with Isolation ended", 2, 2),
        ("without Isolation ended", 1, 1),
    ]


def test_p19_a_single_table_for_an_event_of_several_parts_is_an_error():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    with pytest.raises(ValueError):
        measure_event(
            cohort,
            recovered_tables()["careplans"],
            recovered_outcome(),
            REF,
            BANDS,
        )


def test_p20_a_mapping_without_the_table_of_a_part_is_an_error():
    people, cohort = cohort_with(
        {"Id": "a1", "BIRTHDATE": "1980-01-01", "GENDER": "M"},
    )
    with pytest.raises(ValueError):
        measure_event(
            cohort,
            {"careplans": recovered_tables()["careplans"]},
            recovered_outcome(),
            REF,
            BANDS,
        )


def test_p21_an_event_of_before_gives_the_same_figures_as_in_0_11_0():
    result = measure_event(
        single_patient_cohort("2020-01-01"),
        encounters(
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-11T10:00:00Z"},
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-01-21T10:00:00Z"},
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2020-13-45T10:00:00Z"},
            {"PATIENT": "a1", "CODE": HOSPITAL, "START": "2026-06-01T10:00:00Z"},
        ),
        event(),
        REF,
        BANDS,
    )
    assert result.breakdown.total.patients == 1
    assert result.days.n == 1
    assert result.days.minimum == 10.0
    assert result.rows_matched == 4
    assert result.rows_date_unusable == 1
    assert result.rows_after_reference == 1
    assert result.rows_outside_window == 0
    assert result.rows_filtered_out == 0
    assert result.rows_not_ended == 0
    assert result.op == "any"
    assert len(result.matches) == 1
    assert result.matches[0].name == "encounters: 1505002"
    assert result.matches[0].patients == 1
