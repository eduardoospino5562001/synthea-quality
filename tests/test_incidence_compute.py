"""Tests for time at risk, first events and incidence rates."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from synthea_quality.incidence.compute import (
    CONDITION_COLUMNS,
    PATIENT_COLUMNS,
    build_cohort,
    condition_incidence,
    prepare_records,
    window_start,
)
from synthea_quality.prevalence.definitions import parse_condition_option

REF = date(2026, 1, 1)
SNOMED = "http://snomed.info/sct"


def text_frame(rows, columns) -> pd.DataFrame:
    data = pd.DataFrame([[row.get(c, "") for c in columns] for row in rows], columns=list(columns))
    return data.replace("", None).astype("str") if len(data) else data.astype("str")


def patients(*rows):
    return text_frame(list(rows), PATIENT_COLUMNS)


def conditions(*rows):
    return text_frame([{"SYSTEM": SNOMED, **r} for r in rows], CONDITION_COLUMNS)


def run(people, condition_rows, definition="C=100", **kwargs):
    cohort = build_cohort(people, REF, **kwargs)
    records = prepare_records(conditions(*condition_rows), cohort)
    return condition_incidence(parse_condition_option(definition), records, cohort), cohort


def days(start: str, end: str) -> int:
    """Whole days between two ISO dates (person-time is counted in days)."""
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def test_window_start_and_its_29_february_rule():
    assert window_start(date(2026, 8, 17), 5) == date(2021, 8, 17)
    assert window_start(date(2024, 2, 29), 1) == date(2023, 2, 28)
    with pytest.raises(ValueError):
        window_start(REF, 0)


def test_person_time_runs_from_window_start_to_reference_without_events():
    result, cohort = run(patients({"Id": "p1", "BIRTHDATE": "1950-01-01"}), [])
    assert cohort.window.start == "2021-01-01"
    assert result.rate.events == 0
    assert result.rate.person_days == days("2021-01-01", "2026-01-01")


def test_time_at_risk_stops_at_the_first_event_and_later_records_are_counted():
    result, _ = run(
        patients({"Id": "p1", "BIRTHDATE": "1950-01-01"}),
        [
            {"PATIENT": "p1", "CODE": "100", "START": "2023-01-01"},
            {"PATIENT": "p1", "CODE": "100", "START": "2024-06-01"},
        ],
        "C=100;acute",
    )
    assert (result.rate.events, result.rate.person_days) == (1, days("2021-01-01", "2023-01-01"))
    assert result.metrics["later_records_not_counted"] == 1
    assert any("repeated episodes" in note for note in result.notes)


def test_later_records_are_not_described_as_episodes_unless_declared_acute():
    result, _ = run(
        patients({"Id": "p1", "BIRTHDATE": "1950-01-01"}),
        [
            {"PATIENT": "p1", "CODE": "100", "START": "2023-01-01"},
            {"PATIENT": "p1", "CODE": "100", "START": "2024-06-01"},
        ],
    )
    assert result.metrics["later_records_not_counted"] == 1
    assert not any("repeated episodes" in note for note in result.notes)


def test_a_record_before_the_window_makes_a_prior_case_not_at_risk():
    result, _ = run(
        patients({"Id": "p1", "BIRTHDATE": "1950-01-01"}, {"Id": "p2", "BIRTHDATE": "1950-01-01"}),
        [
            {"PATIENT": "p1", "CODE": "100", "START": "2020-12-31"},  # the day before the window
            {"PATIENT": "p1", "CODE": "100", "START": "2023-01-01"},
            {"PATIENT": "p2", "CODE": "100", "START": "2021-01-01"},  # the first day: an event
        ],
    )
    assert result.metrics["prior_cases"] == 1
    assert result.metrics["at_risk"] == 1
    assert (result.rate.events, result.rate.person_days) == (1, 0.0)


def test_an_event_on_the_reference_date_counts_and_a_later_one_does_not():
    result, _ = run(
        patients({"Id": "p1", "BIRTHDATE": "1950-01-01"}, {"Id": "p2", "BIRTHDATE": "1950-01-01"}),
        [
            {"PATIENT": "p1", "CODE": "100", "START": "2026-01-01"},
            {"PATIENT": "p2", "CODE": "100", "START": "2026-01-02"},
        ],
    )
    assert result.rate.events == 1
    assert result.metrics["records_after_exit"] == 1


def test_the_deceased_contribute_time_until_death():
    result, cohort = run(
        patients({"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2023-01-01"}), []
    )
    assert cohort.size == 1
    assert result.rate.person_days == days("2021-01-01", "2023-01-01")


def test_alive_only_leaves_out_the_deceased_and_counts_them():
    result, cohort = run(
        patients(
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2023-01-01"},
            {"Id": "a1", "BIRTHDATE": "1940-01-01"},
        ),
        [],
        population="alive",
    )
    assert cohort.size == 1 and cohort.excluded["deceased_not_in_alive_only"] == 1
    assert result.rate.person_days == days("2021-01-01", "2026-01-01")


def test_births_during_the_window_enter_at_birth():
    result, _ = run(patients({"Id": "b1", "BIRTHDATE": "2024-01-01"}), [])
    assert result.rate.person_days == days("2024-01-01", "2026-01-01")


def test_exclusions_are_counted():
    _, cohort = run(
        patients(
            {"Id": "x1", "BIRTHDATE": ""},
            {"Id": "x2", "BIRTHDATE": "not a date"},
            {"Id": "x3", "BIRTHDATE": "2027-01-01"},
            {"Id": "x4", "BIRTHDATE": "1930-01-01", "DEATHDATE": "2019-01-01"},
            {"Id": "x5", "BIRTHDATE": "1930-01-01", "DEATHDATE": "someday"},
            {"Id": "ok", "BIRTHDATE": "1990-01-01"},
        ),
        [],
    )
    assert cohort.size == 1
    assert cohort.excluded == {
        "birthdate_unusable": 2,
        "born_after_reference": 1,
        "deathdate_unparseable": 1,
        "died_before_window": 1,
        "deceased_not_in_alive_only": 0,
    }


def test_grouped_codes_take_the_earliest_start_and_unusable_starts_are_counted():
    result, _ = run(
        patients({"Id": "p1", "BIRTHDATE": "1950-01-01"}),
        [
            {"PATIENT": "p1", "CODE": "200", "START": "2024-01-01"},
            {"PATIENT": "p1", "CODE": "100", "START": "2022-01-01"},
            {"PATIENT": "p1", "CODE": "100", "START": "garbage"},
        ],
        "MI=100,200",
    )
    assert (result.rate.events, result.rate.person_days) == (1, days("2021-01-01", "2022-01-01"))
    assert result.metrics["records_start_unusable"] == 1


def test_sex_strata_sum_to_the_total():
    result, _ = run(
        patients(
            {"Id": "m1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "f1", "BIRTHDATE": "1950-01-01", "GENDER": "F"},
            {"Id": "u1", "BIRTHDATE": "1950-01-01", "GENDER": ""},
        ),
        [{"PATIENT": "f1", "CODE": "100", "START": "2022-01-01"}],
    )
    sex = {s.value: s.rate for s in result.strata if s.dimension == "GENDER"}
    assert list(sex) == ["F", "M", None]
    assert sex["F"].events == 1 and sex["M"].events == 0
    assert sum(r.person_days for r in sex.values()) == result.rate.person_days


def test_expected_incidence_is_placed_inside_or_outside_the_interval():
    from synthea_quality.prevalence.definitions import assemble

    cohort = build_cohort(patients({"Id": "p1", "BIRTHDATE": "1950-01-01"}), REF)
    records = prepare_records(conditions({"PATIENT": "p1", "CODE": "100", "START": "2022-01-01"}),
                              cohort)
    (definition,) = assemble(["C=100"], None, ["C:incidence=100"], measures=("incidence",))
    result = condition_incidence(definition, records, cohort)
    assert result.expected[0].position == "inside"  # 1 event in 1 year: 1000 per 1,000 PY
