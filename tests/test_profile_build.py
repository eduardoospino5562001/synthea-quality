"""Tests for building a profile from a dataset directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from synthea_quality.errors import EmptyDatasetError
from synthea_quality.profile.build import build_profile
from synthea_quality.profile.models import InputState, ReferenceSource, SectionStatus
from synthea_quality.profile.reference import ReferenceDateError
from synthea_quality.schema.tables import tables_by_name

GENERATED_AT = "2026-09-28T00:00:00+00:00"


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
            {"Id": "p1", "BIRTHDATE": "2000-01-01", "GENDER": "F", "STATE": "Massachusetts"},
            {"Id": "p2", "BIRTHDATE": "1950-06-01", "GENDER": "M", "STATE": "Massachusetts"},
            {"Id": "p3", "BIRTHDATE": "1930-01-01", "DEATHDATE": "2010-01-01", "GENDER": "M"},
        ],
    )
    write_table(
        directory,
        "encounters",
        [
            {"Id": "e1", "START": "2025-12-31T10:00:00Z", "STOP": "2026-01-01T09:00:00Z"},
            {"Id": "e2", "START": "2024-01-01T00:00:00Z", "STOP": "2024-01-01T01:00:00Z"},
        ],
    )
    return directory


def test_profile_of_a_small_dataset(tmp_path):
    profile = build_profile(dataset(tmp_path), generated_at=GENERATED_AT)
    assert profile.reference_date.value == "2026-01-01"
    assert profile.reference_date.source is ReferenceSource.MAX_ENCOUNTER_DATE
    assert profile.reference_date.approximate
    population = profile.section("population").metrics
    assert (population["total"], population["alive"], population["deceased"]) == (3, 2, 1)
    age = profile.section("age").metrics
    assert (age["min"], age["max"], age["median"]) == (26, 75, 50.5)
    assert [(i.table, i.state, i.rows) for i in profile.inputs] == [
        ("patients", InputState.READ, 3),
        ("encounters", InputState.READ, 2),
    ]
    assert not profile.incomplete


def test_same_input_gives_the_same_profile(tmp_path):
    directory = dataset(tmp_path)
    first = build_profile(directory, generated_at=GENERATED_AT).to_json()
    second = build_profile(directory, generated_at=GENERATED_AT).to_json()
    assert first == second


def test_explicit_reference_date_is_used_and_encounters_are_not_read(tmp_path):
    profile = build_profile(dataset(tmp_path), reference_date="2020-01-01")
    assert profile.reference_date.source is ReferenceSource.USER
    assert profile.section("age").metrics["max"] == 69
    assert [i.table for i in profile.inputs] == ["patients"]


def test_metadata_end_time_is_used_with_a_note_when_inconsistent(tmp_path):
    directory = dataset(tmp_path / "csv")
    metadata = tmp_path / "run.json"
    metadata.write_text(json.dumps({"endTime": "20251201"}), encoding="utf-8")
    profile = build_profile(directory, metadata=metadata)
    assert profile.reference_date.value == "2025-12-01"
    assert profile.reference_date.source is ReferenceSource.SYNTHEA_METADATA
    assert len(profile.notes) == 1 and "earlier than the latest encounter" in profile.notes[0]


def test_both_explicit_sources_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        build_profile(dataset(tmp_path), reference_date="2020-01-01", metadata=tmp_path / "m")


def test_a_bad_explicit_source_is_an_input_error(tmp_path):
    with pytest.raises(ReferenceDateError):
        build_profile(dataset(tmp_path), reference_date="yesterday")
    with pytest.raises(ReferenceDateError):
        build_profile(dataset(tmp_path), metadata=tmp_path / "absent.json")


def test_invalid_age_bands_are_rejected_before_reading(tmp_path):
    with pytest.raises(ValueError, match="start at 0"):
        build_profile(dataset(tmp_path), age_bands=(1, 18))


def test_missing_patients_skips_every_section_with_the_reason(tmp_path):
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2020-01-01T00:00:00Z"}])
    profile = build_profile(tmp_path)
    assert all(s.status is SectionStatus.SKIPPED for s in profile.sections)
    assert {s.reason for s in profile.sections} == {"patients.csv is not in the dataset"}
    assert profile.inputs[0].state is InputState.ABSENT
    assert not profile.incomplete


def test_missing_encounters_skips_only_the_age_section(tmp_path):
    directory = dataset(tmp_path)
    (directory / "encounters.csv").unlink()
    profile = build_profile(directory)
    assert profile.reference_date is None
    assert "encounters.csv is not in the dataset" in profile.reference_reason
    assert profile.section("age").status is SectionStatus.SKIPPED
    assert profile.section("age").reason == profile.reference_reason
    assert profile.section("population").status is SectionStatus.COMPUTED


def test_structurally_broken_patients_is_skipped_and_marks_the_profile_incomplete(tmp_path):
    directory = dataset(tmp_path)
    with (directory / "patients.csv").open("a", encoding="utf-8") as handle:
        handle.write("p4,1990-01-01\n")
    profile = build_profile(directory)
    assert profile.inputs[0].state is InputState.UNREADABLE
    assert "field count" in profile.inputs[0].reason
    assert all(s.status is SectionStatus.SKIPPED for s in profile.sections)
    assert profile.incomplete


def test_unreadable_patients_marks_the_profile_incomplete(tmp_path):
    directory = dataset(tmp_path)
    (directory / "patients.csv").write_bytes(b"Id,BIRTHDATE\n\xff\xfe,2000-01-01\n")
    profile = build_profile(directory)
    assert profile.inputs[0].state is InputState.UNREADABLE
    assert profile.incomplete


def test_patients_without_any_profiled_column_still_counts_nothing_it_cannot(tmp_path):
    write_table(tmp_path, "patients", [{"Id": "p1"}], columns=("Id", "FIRST"))
    profile = build_profile(tmp_path, reference_date="2020-01-01")
    assert profile.inputs[0].rows == 1
    assert profile.section("population").status is SectionStatus.SKIPPED
    assert profile.section("completeness").status is SectionStatus.SKIPPED


def test_a_directory_without_known_tables_is_an_input_error(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(EmptyDatasetError):
        build_profile(tmp_path)


def test_the_dataset_is_never_modified(tmp_path):
    directory = dataset(tmp_path)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    build_profile(directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
