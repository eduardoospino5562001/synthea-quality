"""Tests for building an incidence report from a dataset directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from synthea_quality.errors import EmptyDatasetError
from synthea_quality.incidence.build import build_incidence
from synthea_quality.prevalence.definitions import assemble
from synthea_quality.profile.models import InputState, SectionStatus
from synthea_quality.schema.tables import tables_by_name

GENERATED_AT = "2026-09-28T00:00:00+00:00"
SNOMED = "http://snomed.info/sct"
MI = assemble(["MI=22298006;acute"], None, ["MI:incidence=5"], measures=("incidence",))


def write_table(directory: Path, name: str, rows, columns=None) -> Path:
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
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "M"},
            {"Id": "d0", "BIRTHDATE": "1930-01-01", "DEATHDATE": "2010-01-01", "GENDER": "F"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T10:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "d1", "CODE": "22298006", "SYSTEM": SNOMED, "START": "2023-01-01",
             "STOP": "2023-01-08"},
        ],
    )
    return directory


def test_incidence_of_a_small_dataset_includes_the_deceased_until_death(tmp_path):
    report = build_incidence(dataset(tmp_path), definitions=MI, generated_at=GENERATED_AT)
    assert report.window.to_dict() == {"years": 5, "start": "2021-01-01", "end": "2026-01-01"}
    assert report.population == "all"
    assert report.cohort == {
        "followed": 2,
        "excluded": {
            "birthdate_unusable": 0,
            "born_after_reference": 0,
            "deathdate_unparseable": 0,
            "died_before_window": 1,
            "deceased_not_in_alive_only": 0,
        },
    }
    (mi,) = report.conditions
    assert mi.rate.events == 1
    assert mi.expected[0].position in ("inside", "outside")
    assert any("survivor bias" in note for note in report.notes)
    assert [(i.table, i.state) for i in report.inputs] == [
        ("patients", InputState.READ),
        ("encounters", InputState.READ),
        ("conditions", InputState.READ),
    ]


def test_alive_only_changes_the_population_and_says_why_it_is_biased(tmp_path):
    report = build_incidence(dataset(tmp_path), definitions=MI, alive_only=True)
    assert report.population == "alive"
    assert report.conditions[0].rate.events == 0
    assert any("--alive-only" in note and "survivor bias" in note for note in report.notes)


def test_same_input_same_json(tmp_path):
    directory = dataset(tmp_path)
    first = build_incidence(directory, definitions=MI, generated_at=GENERATED_AT).to_json()
    assert first == build_incidence(directory, definitions=MI, generated_at=GENERATED_AT).to_json()


def test_history_unknown_without_metadata(tmp_path):
    report = build_incidence(dataset(tmp_path), definitions=MI)
    assert report.history["years_of_history"] is None
    assert report.history["earliest_condition_start"] == "2023-01-01"
    assert any("exported history is unknown" in note for note in report.notes)


@pytest.mark.parametrize(
    "years, window, expected",
    [
        (0, 5, "the whole history was exported"),
        (10, 5, "prior cases are seen over the 5 exported year(s) before the window"),
        (3, 5, "before the exported history"),
    ],
)
def test_history_from_the_metadata_file(tmp_path, years, window, expected):
    metadata = tmp_path / "run.json"
    metadata.write_text(
        json.dumps({"endTime": "20260101", "exporter.years_of_history": years}), "utf-8"
    )
    report = build_incidence(
        dataset(tmp_path / "csv"), definitions=MI, metadata=metadata, window_years=window
    )
    assert report.history["years_of_history"] == years
    assert report.history["source"] == "synthea_metadata"
    assert any(expected in note for note in report.notes)


@pytest.mark.parametrize(
    "remove, reason",
    [
        ("conditions.csv", "conditions.csv is not in the dataset"),
        ("patients.csv", "patients.csv is not in the dataset"),
        ("encounters.csv", "no reference date"),
    ],
)
def test_a_missing_input_skips_every_condition(tmp_path, remove, reason):
    directory = dataset(tmp_path)
    (directory / remove).unlink()
    report = build_incidence(directory, definitions=MI)
    assert report.window is None
    assert report.conditions[0].status is SectionStatus.SKIPPED
    assert reason in report.conditions[0].reason
    assert report.conditions[0].acute


def test_a_malformed_conditions_table_marks_the_report_incomplete(tmp_path):
    directory = dataset(tmp_path)
    with (directory / "conditions.csv").open("a", encoding="utf-8") as handle:
        handle.write("2020-01-01,,a1\n")
    report = build_incidence(directory, definitions=MI)
    assert report.incomplete
    assert "field count" in report.conditions[0].reason


def test_invalid_arguments(tmp_path):
    with pytest.raises(ValueError):
        build_incidence(dataset(tmp_path), window_years=0)
    with pytest.raises(ValueError):
        build_incidence(dataset(tmp_path), age_bands=(3,))
    (tmp_path / "empty").mkdir()
    with pytest.raises(EmptyDatasetError):
        build_incidence(tmp_path / "empty")
