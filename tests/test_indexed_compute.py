"""Tests for the indexed cohort: one row per patient with an index date."""

from __future__ import annotations

from datetime import date

import pandas as pd

from synthea_quality.indexed.compute import UNKNOWN_AGE, build_indexed_cohort, measure_event
from synthea_quality.indexed.definitions import EventOutcome, Window
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
            {"Id": "f10", "BIRTHDATE": "2010-01-01", "GENDER": "F"},
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
