"""Behaviour of the temporal checks, plus the runner that drives them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from synthea_quality.checks.temporal import (
    check_event_after_birth,
    check_interval,
    check_life_span,
    run_temporal_checks,
)
from synthea_quality.models import Severity, Status
from synthea_quality.schema.temporal import EventDateRule, IntervalRule, LifeSpanRule

INTERVAL = IntervalRule("encounters", "START", "STOP", "documented for this test")
LIFE_SPAN = LifeSpanRule("patients", "BIRTHDATE", "DEATHDATE", "documented for this test")
EVENT = EventDateRule("conditions", "START", "PATIENT", "documented for this test")

BIRTH_DATES = {"p1": "1980-01-01", "p2": "1990-06-15"}


def interval_frame(starts: list, stops: list, patients: list | None = None) -> pd.DataFrame:
    data = {"START": starts, "STOP": stops}
    data["PATIENT"] = patients if patients is not None else ["p1"] * len(starts)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# start <= stop
# --------------------------------------------------------------------------- #


def test_start_before_stop_passes() -> None:
    frame = interval_frame(["2020-01-01T08:00:00Z"], ["2020-01-01T09:00:00Z"])

    result = check_interval(INTERVAL, frame)

    assert result.status is Status.PASS
    assert result.check_id == "temporal.start_le_stop.encounters"
    assert result.metrics["evaluated"] == 1
    assert result.metrics["violations"] == 0
    assert result.samples == ()


def test_start_equal_to_stop_passes() -> None:
    """Zero-length events are legitimate: 995 medication rows and 489 condition rows do this."""
    frame = interval_frame(["2020-01-01T08:00:00Z"], ["2020-01-01T08:00:00Z"])

    result = check_interval(INTERVAL, frame)

    assert result.status is Status.PASS
    assert result.metrics["violations"] == 0


def test_start_after_stop_fails_with_a_traceable_sample() -> None:
    frame = interval_frame(
        ["2020-01-01T10:00:00Z"], ["2020-01-01T09:00:00Z"], patients=["p9"]
    )
    frame["Id"] = ["e1"]

    result = check_interval(INTERVAL, frame)

    assert result.status is Status.FAIL
    assert result.severity is Severity.MEDIUM
    assert result.metrics["violations"] == 1
    assert result.metrics["violation_pct"] == 100.0
    assert result.samples == (
        {
            "row": 1,
            "values": {
                "START": "2020-01-01T10:00:00Z",
                "STOP": "2020-01-01T09:00:00Z",
                "PATIENT": "p9",
                "Id": "e1",
            },
        },
    )


def test_null_stop_means_an_open_event_and_is_allowed() -> None:
    frame = interval_frame(["2020-01-01T08:00:00Z", "2030-01-01T08:00:00Z"], [None, None])

    result = check_interval(INTERVAL, frame)

    assert result.status is Status.NOT_APPLICABLE
    assert result.metrics["null_stop"] == 2
    assert result.metrics["evaluated"] == 0
    assert result.metrics["violations"] == 0
    assert "still open" in result.message


def test_null_start_is_counted_and_not_compared() -> None:
    frame = interval_frame([None, "2020-01-01T08:00:00Z"], ["2020-01-01T09:00:00Z", None])

    result = check_interval(INTERVAL, frame)

    assert result.status is Status.NOT_APPLICABLE
    assert result.metrics["null_start"] == 1
    assert result.metrics["null_stop"] == 1


def test_mixed_rows_only_compare_the_complete_ones() -> None:
    frame = interval_frame(
        ["2020-01-01T08:00:00Z", "2021-01-01T10:00:00Z", None],
        ["2020-01-01T09:00:00Z", "2021-01-01T09:00:00Z", None],
    )

    result = check_interval(INTERVAL, frame)

    assert result.status is Status.FAIL
    assert result.metrics["evaluated"] == 2
    assert result.metrics["violations"] == 1
    assert result.metrics["violation_pct"] == 50.0


def test_unreadable_dates_are_excluded_rather_than_compared() -> None:
    """Comparing a value that matches no confirmed format would judge nothing."""
    frame = interval_frame(["31/12/2020", "2020-01-01T08:00:00Z"], ["2020-01-01T09:00:00Z", "2020-01-01T10:00:00Z"])

    result = check_interval(INTERVAL, frame)

    assert result.metrics["unusable_start"] == 1
    assert result.metrics["evaluated"] == 1
    assert result.metrics["violations"] == 0
    assert result.status is Status.PASS


def test_multiple_violations_are_counted_in_full_with_bounded_samples() -> None:
    frame = interval_frame(
        ["2020-01-01T10:00:00Z"] * 40,
        ["2020-01-01T09:00:00Z"] * 40,
    )

    result = check_interval(INTERVAL, frame, sample_limit=3)

    assert result.metrics["violations"] == 40
    assert len(result.samples) == 3
    assert result.metadata["relation"] == "START <= STOP"
    assert result.metadata["sample_limit"] == 3


# --------------------------------------------------------------------------- #
# birth <= death
# --------------------------------------------------------------------------- #


def life_frame(births: list, deaths: list) -> pd.DataFrame:
    return pd.DataFrame({"Id": [f"p{i}" for i in range(len(births))], "BIRTHDATE": births, "DEATHDATE": deaths})


def test_birth_before_death_passes() -> None:
    result = check_life_span(LIFE_SPAN, life_frame(["1980-01-01"], ["2020-05-05"]))

    assert result.status is Status.PASS
    assert result.check_id == "temporal.birth_le_death.patients"
    assert result.metrics["evaluated"] == 1


def test_birth_equal_to_death_is_allowed() -> None:
    """Technically possible (the dates have day resolution), so equality must pass."""
    result = check_life_span(LIFE_SPAN, life_frame(["1980-01-01"], ["1980-01-01"]))

    assert result.status is Status.PASS
    assert result.metrics["violations"] == 0


def test_death_before_birth_fails() -> None:
    frame = life_frame(["1980-01-01"], ["1979-12-31"])

    result = check_life_span(LIFE_SPAN, frame)

    assert result.status is Status.FAIL
    assert result.metrics["violations"] == 1
    assert result.samples[0]["values"] == {
        "BIRTHDATE": "1980-01-01",
        "DEATHDATE": "1979-12-31",
        "Id": "p0",
    }


def test_null_death_date_is_a_patient_alive_at_the_end() -> None:
    frame = life_frame(["1980-01-01", "1990-02-02"], [None, None])

    result = check_life_span(LIFE_SPAN, frame)

    assert result.status is Status.NOT_APPLICABLE
    assert result.metrics["null_death"] == 2
    assert "alive at the end of the simulation" in result.message


def test_life_span_skips_patients_without_a_death_date() -> None:
    frame = life_frame(["1980-01-01", "1990-02-02"], [None, "2020-01-01"])

    result = check_life_span(LIFE_SPAN, frame)

    assert result.status is Status.PASS
    assert result.metrics["evaluated"] == 1
    assert result.metrics["null_death"] == 1


# --------------------------------------------------------------------------- #
# event >= birth
# --------------------------------------------------------------------------- #


def event_frame(dates: list, patients: list) -> pd.DataFrame:
    return pd.DataFrame({"START": dates, "PATIENT": patients})


def test_event_after_birth_passes() -> None:
    result = check_event_after_birth(EVENT, event_frame(["2000-01-01", "2010-01-01"], ["p1", "p2"]), BIRTH_DATES)

    assert result.status is Status.PASS
    assert result.check_id == "temporal.event_after_birth.conditions.START"
    assert result.metrics["evaluated"] == 2


def test_event_before_birth_is_a_warning_not_a_failure() -> None:
    result = check_event_after_birth(EVENT, event_frame(["1979-01-01"], ["p1"]), BIRTH_DATES)

    assert result.status is Status.WARNING
    assert result.severity is Severity.MEDIUM
    assert result.metrics["violations"] == 1
    assert result.samples[0]["values"] == {
        "START": "1979-01-01",
        "PATIENT": "p1",
        "patient_birth_date": "1980-01-01",
    }


def test_event_on_the_birth_date_is_not_early() -> None:
    result = check_event_after_birth(EVENT, event_frame(["1980-01-01"], ["p1"]), BIRTH_DATES)

    assert result.status is Status.PASS


def test_event_of_an_unknown_patient_is_counted_and_not_judged() -> None:
    result = check_event_after_birth(
        EVENT, event_frame(["2000-01-01", "1900-01-01"], ["p1", "ghost"]), BIRTH_DATES
    )

    assert result.status is Status.PASS
    assert result.metrics["evaluated"] == 1
    assert result.metrics["event_without_birth_date"] == 1
    assert result.metrics["violations"] == 0


def test_event_check_without_birth_dates_is_skipped() -> None:
    result = check_event_after_birth(EVENT, event_frame(["2000-01-01"], ["p1"]), None)

    assert result.status is Status.SKIPPED
    assert "could not be built" in result.message


def test_event_check_with_no_comparable_row_is_not_applicable() -> None:
    result = check_event_after_birth(EVENT, event_frame([None], ["p1"]), BIRTH_DATES)

    assert result.status is Status.NOT_APPLICABLE
    assert result.metrics["null_event"] == 1


def test_event_check_bounds_its_samples() -> None:
    result = check_event_after_birth(
        EVENT, event_frame(["1900-01-01"] * 30, ["p1"] * 30), BIRTH_DATES, sample_limit=2
    )

    assert result.metrics["violations"] == 30
    assert len(result.samples) == 2


# --------------------------------------------------------------------------- #
# the runner over a dataset directory
# --------------------------------------------------------------------------- #

PATIENTS = "Id,BIRTHDATE,DEATHDATE\np1,1980-01-01,\np2,1990-02-02,2020-03-03\n"
ENCOUNTERS = "Id,START,STOP,PATIENT\ne1,2020-01-01T08:00:00Z,2020-01-01T09:00:00Z,p1\n"
CONDITIONS = "START,STOP,PATIENT\n2000-01-01,,p1\n"


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_runner_passes_on_a_consistent_dataset(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", PATIENTS)
    write(tmp_path, "encounters.csv", ENCOUNTERS)
    write(tmp_path, "conditions.csv", CONDITIONS)

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}

    assert by_id["temporal.start_le_stop.encounters"].status is Status.PASS
    assert by_id["temporal.birth_le_death.patients"].status is Status.PASS
    assert by_id["temporal.event_after_birth.conditions.START"].status is Status.PASS
    assert by_id["temporal.event_after_birth.encounters.START"].status is Status.PASS
    assert not [result for result in by_id.values() if result.status is Status.FAIL]


def test_runner_reports_an_inverted_interval(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", PATIENTS)
    write(
        tmp_path,
        "encounters.csv",
        "Id,START,STOP,PATIENT\ne1,2020-01-01T10:00:00Z,2020-01-01T09:00:00Z,p1\n",
    )

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}
    result = by_id["temporal.start_le_stop.encounters"]

    assert result.status is Status.FAIL
    assert result.samples[0]["row"] == 1
    assert result.samples[0]["values"]["PATIENT"] == "p1"


def test_runner_reports_a_death_before_birth(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,BIRTHDATE,DEATHDATE\np1,1990-01-01,1980-01-01\n")

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}
    result = by_id["temporal.birth_le_death.patients"]

    assert result.status is Status.FAIL
    assert result.samples[0]["values"]["Id"] == "p1"


def test_runner_warns_about_an_event_before_birth(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", PATIENTS)
    write(tmp_path, "conditions.csv", "START,STOP,PATIENT\n1970-01-01,,p1\n")

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}
    result = by_id["temporal.event_after_birth.conditions.START"]

    assert result.status is Status.WARNING
    assert result.metrics["violations"] == 1


def test_runner_skips_a_missing_table(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", PATIENTS)

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}

    skipped = by_id["temporal.start_le_stop.encounters"]
    assert skipped.status is Status.SKIPPED
    assert "not present in this dataset" in skipped.message
    assert by_id["temporal.event_after_birth.observations.DATE"].status is Status.SKIPPED


def test_runner_skips_a_rule_whose_column_is_absent(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", PATIENTS)
    write(tmp_path, "encounters.csv", "Id,START,PATIENT\ne1,2020-01-01T08:00:00Z,p1\n")

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}

    interval = by_id["temporal.start_le_stop.encounters"]
    assert interval.status is Status.SKIPPED
    assert "no column(s) ['STOP']" in interval.message
    # the event rule of the same table still runs: availability is per rule
    assert by_id["temporal.event_after_birth.encounters.START"].status is Status.PASS


def test_runner_skips_event_checks_when_the_patients_table_is_missing(tmp_path: Path) -> None:
    write(tmp_path, "conditions.csv", CONDITIONS)

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}
    result = by_id["temporal.event_after_birth.conditions.START"]

    assert result.status is Status.SKIPPED
    assert "patients" in result.message


def test_runner_skips_event_checks_when_patients_has_no_birthdate(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,FIRST\np1,Ana\n")
    write(tmp_path, "conditions.csv", CONDITIONS)

    by_id = {result.check_id: result for result in run_temporal_checks(tmp_path)}
    result = by_id["temporal.event_after_birth.conditions.START"]

    assert result.status is Status.SKIPPED
    assert "patients' has no column(s) ['BIRTHDATE']" in result.message


def test_runner_covers_every_rule_once(tmp_path: Path) -> None:
    from synthea_quality.schema.temporal import (
        EVENT_DATE_RULES,
        INTERVAL_RULES,
        LIFE_SPAN_RULES,
    )

    write(tmp_path, "patients.csv", PATIENTS)
    results = run_temporal_checks(tmp_path)
    expected = len(INTERVAL_RULES) + len(LIFE_SPAN_RULES) + len(EVENT_DATE_RULES)

    assert len(results) == expected
    assert len({result.check_id for result in results}) == expected


def test_runner_results_are_deterministic_and_sorted(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", PATIENTS)
    write(tmp_path, "encounters.csv", ENCOUNTERS)

    first = [result.check_id for result in run_temporal_checks(tmp_path)]
    second = [result.check_id for result in run_temporal_checks(tmp_path)]

    assert first == second == sorted(first)


def test_runner_loads_one_table_at_a_time_with_only_the_columns_it_needs(
    tmp_path: Path, monkeypatch
) -> None:
    import synthea_quality.checks.temporal as module

    write(tmp_path, "patients.csv", PATIENTS)
    write(tmp_path, "encounters.csv", ENCOUNTERS)
    loaded: list[tuple[str, ...]] = []
    original = module.load_table

    def tracking(file_path, *, columns=None, **kwargs):
        loaded.append(tuple(columns) if columns else ())
        return original(file_path, columns=columns, **kwargs)

    monkeypatch.setattr(module, "load_table", tracking)
    results = run_temporal_checks(tmp_path)

    assert results
    assert ("Id", "BIRTHDATE") in loaded
    encounters_loads = [columns for columns in loaded if set(columns) >= {"START", "STOP", "PATIENT"}]
    assert len(encounters_loads) == 1
    assert len(encounters_loads[0]) <= 5  # rule columns plus a bounded set of identifiers
