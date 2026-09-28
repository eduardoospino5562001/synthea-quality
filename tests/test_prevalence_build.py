"""Tests for building a prevalence report from a dataset directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from synthea_quality.errors import EmptyDatasetError
from synthea_quality.prevalence.build import build_prevalence
from synthea_quality.prevalence.definitions import assemble
from synthea_quality.profile.models import InputState, ReferenceSource, SectionStatus
from synthea_quality.schema.tables import tables_by_name

GENERATED_AT = "2026-09-28T00:00:00+00:00"
SNOMED = "http://snomed.info/sct"


def write_table(directory: Path, name: str, rows: list[dict[str, str]], columns=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-01-01", "GENDER": "F"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2020-01-01", "GENDER": "M"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T10:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "a1", "CODE": "22298006", "SYSTEM": SNOMED, "START": "2020-01-01",
             "STOP": "2020-01-08", "DESCRIPTION": "Myocardial infarction (disorder)"},
            {"PATIENT": "d1", "CODE": "22298006", "SYSTEM": SNOMED, "START": "2015-01-01",
             "DESCRIPTION": "Myocardial infarction (disorder)"},
            {"PATIENT": "a2", "CODE": "73595000", "SYSTEM": SNOMED, "START": "2021-01-01",
             "DESCRIPTION": "Stress (finding)"},
        ],
    )
    return directory


MI = assemble(["Myocardial infarction=22298006"], None, ["Myocardial infarction:lifetime=0.3"])


def test_prevalence_of_a_small_dataset(tmp_path):
    report = build_prevalence(dataset(tmp_path), definitions=MI, generated_at=GENERATED_AT)
    assert report.reference_date.value == "2026-01-01"
    assert report.reference_date.source is ReferenceSource.MAX_ENCOUNTER_DATE
    assert report.alive == 2
    (mi,) = report.conditions
    assert (mi.point.numerator, mi.lifetime.numerator, mi.lifetime.denominator) == (0, 1, 2)
    assert mi.expected[0].position == "inside"
    assert [row.code for row in report.general.rows] == ["22298006"]
    assert report.general.metrics["social_codes_in_data"] == 1
    assert [(i.table, i.state) for i in report.inputs] == [
        ("patients", InputState.READ),
        ("encounters", InputState.READ),
        ("conditions", InputState.READ),
    ]
    assert report.social_list["id"] == "synthea-d9d07a6e-social-1"


def test_same_input_same_json(tmp_path):
    directory = dataset(tmp_path)
    first = build_prevalence(directory, definitions=MI, generated_at=GENERATED_AT).to_json()
    second = build_prevalence(directory, definitions=MI, generated_at=GENERATED_AT).to_json()
    assert first == second
    json.loads(first)


def test_include_social_and_an_explicit_reference_date(tmp_path):
    report = build_prevalence(dataset(tmp_path), include_social=True, reference_date="2020-01-05")
    assert report.reference_date.source is ReferenceSource.USER
    codes = {row.code: row for row in report.general.rows}
    assert "73595000" not in codes  # starts after 2020-01-05
    assert codes["22298006"].point.numerator == 1  # active on 2020-01-05


@pytest.mark.parametrize(
    "remove, reason",
    [
        ("conditions.csv", "conditions.csv is not in the dataset"),
        ("patients.csv", "patients.csv is not in the dataset"),
        ("encounters.csv", "no reference date"),
    ],
)
def test_a_missing_input_skips_every_result_with_the_reason(tmp_path, remove, reason):
    directory = dataset(tmp_path)
    (directory / remove).unlink()
    report = build_prevalence(directory, definitions=MI)
    assert report.general.status is SectionStatus.SKIPPED
    assert reason in report.general.reason
    assert report.conditions[0].status is SectionStatus.SKIPPED
    assert reason in report.conditions[0].reason
    assert not report.incomplete


def test_conditions_without_stop_column_is_skipped(tmp_path):
    directory = dataset(tmp_path)
    write_table(directory, "conditions", [{"PATIENT": "a1", "CODE": "1", "START": "2020-01-01"}],
                columns=("PATIENT", "CODE", "START"))
    report = build_prevalence(directory)
    assert "no STOP column" in report.general.reason


def test_a_malformed_conditions_table_marks_the_report_incomplete(tmp_path):
    directory = dataset(tmp_path)
    with (directory / "conditions.csv").open("a", encoding="utf-8") as handle:
        handle.write("2020-01-01,,a1\n")
    report = build_prevalence(directory, definitions=MI)
    assert report.incomplete
    assert report.general.status is SectionStatus.SKIPPED
    assert "field count" in report.general.reason


def test_invalid_arguments(tmp_path):
    with pytest.raises(ValueError):
        build_prevalence(dataset(tmp_path), top=0)
    with pytest.raises(ValueError):
        build_prevalence(dataset(tmp_path), age_bands=(5,))
    (tmp_path / "empty").mkdir()
    with pytest.raises(EmptyDatasetError):
        build_prevalence(tmp_path / "empty")
