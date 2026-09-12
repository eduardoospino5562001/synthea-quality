"""Tests for dataset discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import DiscoveryError
from synthea_quality.schema.tables import CURRENT_CONTRACT_ID, SYNTHEA_TABLES, TableSpec

ALL_FILE_NAMES = [spec.file_name for spec in SYNTHEA_TABLES]
SAMPLE_FILE_NAMES = [spec.file_name for spec in SYNTHEA_TABLES if spec.included_by_default]


def write_files(directory: Path, names: Iterable[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("Id\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# happy paths
# --------------------------------------------------------------------------- #


def test_discovers_every_known_table(tmp_path: Path) -> None:
    write_files(tmp_path, ALL_FILE_NAMES)

    result = discover_dataset(tmp_path)

    assert len(result.tables) == len(SYNTHEA_TABLES)
    assert result.missing_tables == ()
    assert result.unknown_csv_files == ()
    assert result.ignored_files == ()
    assert result.is_empty is False


def test_discovers_the_official_sample_shape(tmp_path: Path) -> None:
    """The 18 files of the official CSV sample: no expenses file, no unknowns."""
    write_files(tmp_path, SAMPLE_FILE_NAMES)

    result = discover_dataset(tmp_path)

    assert len(result.tables) == 18
    assert [spec.name for spec in result.missing_tables] == ["patient_expenses"]
    assert result.unknown_csv_files == ()


def test_partial_dataset_reports_missing_tables_without_failing(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv", "conditions.csv"])

    result = discover_dataset(tmp_path)

    assert result.table_names == ("conditions", "patients")
    assert len(result.missing_tables) == len(SYNTHEA_TABLES) - 2
    assert result.is_empty is False


def test_empty_directory_is_a_valid_result(tmp_path: Path) -> None:
    result = discover_dataset(tmp_path)

    assert result.tables == ()
    assert result.unknown_csv_files == ()
    assert len(result.missing_tables) == len(SYNTHEA_TABLES)
    assert result.is_empty is True


def test_table_names_are_returned_in_a_stable_order(tmp_path: Path) -> None:
    write_files(tmp_path, ["procedures.csv", "allergies.csv", "encounters.csv"])

    assert discover_dataset(tmp_path).table_names == ("allergies", "encounters", "procedures")


def test_discovery_is_deterministic(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv", "conditions.csv", "notes.csv"])

    assert discover_dataset(tmp_path) == discover_dataset(tmp_path)


# --------------------------------------------------------------------------- #
# classification of files
# --------------------------------------------------------------------------- #


def test_unknown_csv_files_are_reported(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv", "patients_backup.csv", "supplyx.csv"])

    result = discover_dataset(tmp_path)

    assert result.unknown_csv_files == ("patients_backup.csv", "supplyx.csv")
    assert result.table_names == ("patients",)


def test_non_csv_files_are_ignored_and_directories_are_skipped(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv"])
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")
    (tmp_path / "data-set.zip").write_bytes(b"PK\x03\x04")
    (tmp_path / "nested").mkdir()

    result = discover_dataset(tmp_path)

    assert result.ignored_files == (".DS_Store", "README.md", "data-set.zip")
    assert result.unknown_csv_files == ()
    assert "nested" not in result.ignored_files


def test_excluded_by_default_table_is_still_a_known_table(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv", "patient_expenses.csv"])

    result = discover_dataset(tmp_path)

    assert result.table_names == ("patient_expenses", "patients")
    assert result.unknown_csv_files == ()
    expenses = next(table for table in result.tables if table.name == "patient_expenses")
    assert expenses.spec.is_optional is True


def test_file_name_match_is_case_insensitive_and_keeps_the_real_name(tmp_path: Path) -> None:
    write_files(tmp_path, ["Patients.csv"])

    result = discover_dataset(tmp_path)

    assert result.table_names == ("patients",)
    assert result.unknown_csv_files == ()
    assert result.tables[0].file_name == "Patients.csv"


def test_two_files_for_the_same_table_are_rejected(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv", "Patients.csv"])

    with pytest.raises(DiscoveryError, match="ambiguous dataset"):
        discover_dataset(tmp_path)


def test_discovery_does_not_read_file_contents(tmp_path: Path) -> None:
    """A file is discovered by name; its content is the loader's problem."""
    (tmp_path / "patients.csv").write_bytes(b"\x00not,a,csv\n\xff")

    result = discover_dataset(tmp_path)

    assert result.table_names == ("patients",)


# --------------------------------------------------------------------------- #
# invalid inputs
# --------------------------------------------------------------------------- #


def test_missing_directory_raises_a_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(DiscoveryError, match="does not exist"):
        discover_dataset(missing)


def test_a_file_instead_of_a_directory_is_a_clear_error(tmp_path: Path) -> None:
    file_path = tmp_path / "patients.csv"
    file_path.write_text("Id\n", encoding="utf-8")

    with pytest.raises(DiscoveryError, match="not a directory"):
        discover_dataset(file_path)


def test_string_paths_are_accepted(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv"])

    result = discover_dataset(str(tmp_path))

    assert isinstance(result, DiscoveryResult)
    assert result.table_names == ("patients",)
    assert result.data_dir == tmp_path


# --------------------------------------------------------------------------- #
# catalogue handling
# --------------------------------------------------------------------------- #


def test_custom_catalogue_limits_what_is_known(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv", "conditions.csv"])
    catalogue = (TableSpec("patients", "patients.csv"),)

    result = discover_dataset(tmp_path, known_tables=catalogue)

    assert result.table_names == ("patients",)
    assert result.unknown_csv_files == ("conditions.csv",)
    assert result.missing_tables == ()
    assert result.contract_id is None


def test_default_contract_id_is_recorded(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv"])

    assert discover_dataset(tmp_path).contract_id == CURRENT_CONTRACT_ID


def test_duplicate_file_names_in_a_catalogue_are_a_tool_bug(tmp_path: Path) -> None:
    write_files(tmp_path, ["patients.csv"])
    catalogue = (
        TableSpec("patients", "patients.csv"),
        TableSpec("patients_v2", "patients.csv"),
    )

    with pytest.raises(ValueError, match="declared twice"):
        discover_dataset(tmp_path, known_tables=catalogue)
