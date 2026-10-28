"""Tests for building the module validation of each dataset (build.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from csv_tables import write_table

from synthea_quality.compare.build import BuildResult, Side, build
from synthea_quality.prevalence.definitions import DefinitionError
from synthea_quality.profile.models import SectionStatus
from synthea_quality.validate.models import ModuleValidationReport

SNOMED = "http://snomed.info/sct"
GENERATED_AT = "2026-09-29T00:00:00+00:00"


def write_dataset(directory: Path, *, conditions: bool = True) -> Path:
    """A tiny Synthea export: patients, encounters, conditions, observations, medications."""

    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-06-01", "GENDER": "F"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "M"},
        ],
    )
    write_table(
        directory,
        "encounters",
        [{"Id": "e1", "PATIENT": "a1", "CODE": "1505002", "START": "2026-01-01T00:00:00Z"}],
    )
    if conditions:
        write_table(
            directory,
            "conditions",
            [
                {
                    "PATIENT": "a1",
                    "CODE": "22298006",
                    "SYSTEM": SNOMED,
                    "START": "2023-01-01",
                    "STOP": "",
                    "DESCRIPTION": "Myocardial infarction (disorder)",
                },
                {
                    "PATIENT": "a2",
                    "CODE": "59621000",
                    "SYSTEM": SNOMED,
                    "START": "2015-01-01",
                    "STOP": "",
                    "DESCRIPTION": "Essential hypertension (disorder)",
                },
            ],
        )
    write_table(
        directory,
        "observations",
        [
            {
                "DATE": "2026-01-01T00:00:00Z",
                "PATIENT": "a1",
                "CODE": "8480-6",
                "VALUE": "150",
                "UNITS": "mm[Hg]",
                "TYPE": "numeric",
            },
            {
                "DATE": "2026-01-01T00:00:00Z",
                "PATIENT": "a2",
                "CODE": "8480-6",
                "VALUE": "120",
                "UNITS": "mm[Hg]",
                "TYPE": "numeric",
            },
        ],
    )
    write_table(
        directory,
        "medications",
        [
            {
                "START": "2025-01-01T00:00:00Z",
                "STOP": "",
                "PATIENT": "a1",
                "PAYER": "",
                "ENCOUNTER": "",
                "CODE": "314076",
                "DESCRIPTION": "lisinopril",
                "BASE_COST": "",
                "PAYER_COVERAGE": "",
                "DISPENSES": "",
                "TOTALCOST": "",
                "REASONCODE": "",
                "REASONDESCRIPTION": "",
            },
        ],
    )
    return directory


def write_module(directory: Path, *, indexed: bool = True) -> Path:
    """A module JSON file applied to both datasets."""

    data = {
        "module": {"name": "Test module"},
        "conditions": [
            {"name": "Myocardial infarction", "codes": ["22298006"], "acute": True},
            {"name": "Hypertension", "codes": ["59621000"], "acute": False},
        ],
        "observations": [
            {
                "name": "Systolic blood pressure",
                "code": "8480-6",
                "reference_range": {"low": 100, "high": 139, "units": "mm[Hg]"},
            }
        ],
        "medications": [{"name": "Lisinopril", "codes": ["314076"]}],
    }
    if indexed:
        data["index"] = {"condition": "Myocardial infarction"}
        data["outcomes"] = [
            {"name": "ICU admission", "type": "event", "table": "encounters", "codes": ["1505002"]},
            {"name": "Death", "type": "death"},
        ]
        data["by"] = [{"outcome": "Death", "by": "ICU admission"}]
    path = directory / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def build_both(directory: Path, module_path: Path) -> BuildResult:
    return build(
        directory,
        directory,
        str(module_path),
        reference_date="2026-01-01",
        generated_at=GENERATED_AT,
    )


def test_build_returns_both_sides_with_the_same_module(tmp_path: Path) -> None:
    directory = write_dataset(tmp_path / "data")
    module_path = write_module(tmp_path / "data")
    result = build_both(directory, module_path)
    assert isinstance(result, BuildResult)
    assert result.module_file == str(module_path)
    assert result.module.name == "Test module"
    assert isinstance(result.baseline, Side)
    assert isinstance(result.candidate, Side)
    assert result.baseline.data_dir == str(directory)
    assert result.candidate.data_dir == str(directory)
    assert isinstance(result.baseline.report, ModuleValidationReport)
    assert isinstance(result.candidate.report, ModuleValidationReport)


def test_same_directory_twice_gives_two_independent_reports(tmp_path: Path) -> None:
    directory = write_dataset(tmp_path / "data")
    module_path = write_module(tmp_path / "data")
    result = build_both(directory, module_path)
    # Same input, same parameters: same figures, but two distinct report objects.
    assert result.baseline.report == result.candidate.report
    assert result.baseline.report is not result.candidate.report


def test_build_propagates_module_errors(tmp_path: Path) -> None:
    directory = write_dataset(tmp_path / "data")
    bad_module = tmp_path / "nonexistent.json"
    with pytest.raises(DefinitionError):
        build(directory, directory, str(bad_module), reference_date="2026-01-01")


def test_build_without_encounters_skips_population_with_reason(tmp_path: Path) -> None:
    directory = write_dataset(tmp_path / "data")
    (directory / "encounters.csv").unlink()
    module_path = write_module(tmp_path / "data")
    result = build(
        directory,
        directory,
        str(module_path),
        reference_date=None,
        generated_at=GENERATED_AT,
    )
    # Without encounters there is no reference date: every population figure is skipped.
    figures = result.baseline.report.population
    assert figures
    for section in figures:
        assert section.status is SectionStatus.SKIPPED
        assert section.reason and "reference date" in section.reason.lower()


def test_reference_date_and_metadata_together_raise(tmp_path: Path) -> None:
    directory = write_dataset(tmp_path / "data")
    module_path = write_module(tmp_path / "data")
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"endTime": "2026-01-01T00:00:00Z"}', encoding="utf-8")
    with pytest.raises(ValueError, match="give either a reference date or a metadata file"):
        build(
            directory,
            directory,
            str(module_path),
            reference_date="2026-01-01",
            metadata_baseline=str(metadata),
            generated_at=GENERATED_AT,
        )
