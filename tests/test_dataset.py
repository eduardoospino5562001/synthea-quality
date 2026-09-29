"""Tests for the shared way analysis reports open a dataset."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from synthea_quality.dataset import open_dataset
from synthea_quality.errors import EmptyDatasetError
from synthea_quality.profile.models import InputState
from synthea_quality.schema.tables import tables_by_name


def write_table(directory: Path, name: str, rows, columns=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dataset(directory: Path) -> Path:
    write_table(directory, "patients", [{"Id": "p1", "BIRTHDATE": "1950-01-01"}])
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    return directory


def test_opens_resolves_the_reference_and_loads_with_provenance(tmp_path):
    context = open_dataset(dataset(tmp_path), tables=("patients", "encounters", "conditions"))
    assert context.reference == date(2026, 1, 1)
    frame, info = context.load("patients", ("Id", "BIRTHDATE", "GENDER"), ("Id",), "cohort")
    assert list(frame.columns) == ["Id", "BIRTHDATE", "GENDER"]
    assert (info.state, info.rows) == (InputState.READ, 1)
    missing, info = context.load("conditions", ("PATIENT",), ("PATIENT",), "records")
    assert missing is None and info.state is InputState.ABSENT


def test_a_missing_required_column_is_reported_as_absent(tmp_path):
    directory = dataset(tmp_path)
    write_table(directory, "conditions", [{"PATIENT": "p1"}], columns=("PATIENT", "CODE"))
    context = open_dataset(directory, tables=("conditions",))
    frame, info = context.load("conditions", ("PATIENT", "START"), ("START",), "records")
    assert frame is None and "no START column" in info.reason


def test_a_malformed_table_is_unreadable(tmp_path):
    directory = dataset(tmp_path)
    with (directory / "patients.csv").open("a", encoding="utf-8") as handle:
        handle.write("p2\n")
    context = open_dataset(directory, tables=("patients",))
    frame, info = context.load("patients", ("Id",), ("Id",), "cohort")
    assert frame is None and info.state is InputState.UNREADABLE


def test_no_known_table_and_no_reference(tmp_path):
    with pytest.raises(EmptyDatasetError):
        open_dataset(tmp_path, tables=("patients",))
    write_table(tmp_path, "patients", [{"Id": "p1"}])
    assert open_dataset(tmp_path, tables=("patients",)).reference is None
