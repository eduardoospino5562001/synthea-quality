"""Tests for building the observation values report from a dataset directory."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.observations.build import build_observations
from synthea_quality.observations.definitions import ObservationDefinition
from synthea_quality.prevalence.definitions import parse_condition_option
from synthea_quality.profile.models import InputState, SectionStatus
from synthea_quality.schema.tables import tables_by_name

SNOMED = "http://snomed.info/sct"
SBP = "8480-6"


def write_table(directory: Path, name: str, rows, columns=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def reading(patient, value, day="2026-01-01T00:00:00Z"):
    return {"DATE": day, "PATIENT": patient, "CODE": SBP, "DESCRIPTION": "Systolic",
            "VALUE": value, "UNITS": "mm[Hg]", "TYPE": "numeric"}


def dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-06-01", "GENDER": "F"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "M"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [{"PATIENT": "a1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2015-01-01"}],
    )
    write_table(
        directory,
        "observations",
        [reading("a1", "150"), reading("a2", "120", "2020-01-01T00:00:00Z"), reading("d1", "170")],
    )
    return directory


HTN = parse_condition_option("Hypertension=59621000")
COHORT = ObservationDefinition(SBP, name="SBP, hypertension", cohort=CohortSpec("Hypertension"))


def build(directory, **kwargs):
    return build_observations(directory, generated_at="2026-09-29T00:00:00+00:00", **kwargs)


def test_the_observations_and_the_general_table(tmp_path):
    report = build(
        dataset(tmp_path), observations=[ObservationDefinition(SBP), COHORT], conditions=[HTN]
    )
    everyone, cohort = report.observations
    assert report.alive == 2
    assert everyone.groups[0].summary.n == 2
    assert (cohort.population, cohort.groups[0].summary.median) == (1, 150)
    assert [r.code for r in report.general.rows] == [SBP]
    assert [i.table for i in report.inputs] == [
        "patients", "encounters", "observations", "conditions",
    ]


def test_conditions_are_read_only_for_a_cohort(tmp_path):
    report = build(dataset(tmp_path), observations=[ObservationDefinition(SBP)])
    assert "conditions" not in [i.table for i in report.inputs]


def test_the_command_lookback_applies_unless_an_observation_sets_its_own(tmp_path):
    report = build(
        dataset(tmp_path),
        observations=[ObservationDefinition(SBP), ObservationDefinition(SBP, name="own",
                                                                        lookback_years=10)],
        lookback_years=2,
    )
    short, own = report.observations
    assert (short.lookback_years, short.groups[0].summary.n) == (2, 1)
    assert (own.lookback_years, own.groups[0].summary.n) == (10, 2)
    assert report.general.rows[0].summary.n == 1


def test_a_cohort_without_conditions_csv_is_skipped(tmp_path):
    directory = dataset(tmp_path)
    (directory / "conditions.csv").unlink()
    report = build(directory, observations=[ObservationDefinition(SBP), COHORT], conditions=[HTN])
    everyone, cohort = report.observations
    assert everyone.status is SectionStatus.COMPUTED
    assert cohort.status is SectionStatus.SKIPPED
    assert "conditions.csv is not in the dataset" in cohort.reason


@pytest.mark.parametrize("remove", ["observations.csv", "patients.csv", "encounters.csv"])
def test_a_missing_input_skips_everything_with_the_reason(tmp_path, remove):
    directory = dataset(tmp_path)
    (directory / remove).unlink()
    report = build(directory, observations=[ObservationDefinition(SBP)])
    assert report.observations[0].status is SectionStatus.SKIPPED
    assert report.general.status is SectionStatus.SKIPPED
    assert report.general.reason == report.observations[0].reason


def test_an_unreadable_table_marks_the_report_incomplete(tmp_path):
    directory = dataset(tmp_path)
    (directory / "observations.csv").write_text("DATE,PATIENT,CODE,VALUE\n1,2\n", encoding="utf-8")
    report = build(directory, observations=[ObservationDefinition(SBP)])
    assert report.incomplete
    assert [i.state for i in report.inputs if i.table == "observations"] == [InputState.UNREADABLE]


def test_invalid_arguments(tmp_path):
    for kwargs in ({"top": 0}, {"lookback_years": 0}, {"age_bands": (3,)}):
        with pytest.raises(ValueError):
            build(dataset(tmp_path), **kwargs)
