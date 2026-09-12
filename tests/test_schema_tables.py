"""Tests for the table catalogue of the current schema contract.

These tests protect the catalogue's internal consistency and its provenance. The
cross-check against a real Synthea dataset happens in the integration test.
"""

from __future__ import annotations

import re

from synthea_quality.schema.tables import (
    CURRENT_CONTRACT_ID,
    SOURCE_COMMIT,
    SOURCE_FILE,
    SOURCE_REPOSITORY,
    SYNTHEA_TABLES,
    tables_by_name,
)


def test_catalogue_has_the_nineteen_declared_tables() -> None:
    # CSVConstants declares 19 *_KEY constants; patient_expenses is excluded from
    # a default export but it is still part of the schema.
    assert len(SYNTHEA_TABLES) == 19


def test_names_and_file_names_are_unique() -> None:
    names = [spec.name for spec in SYNTHEA_TABLES]
    file_names = [spec.file_name for spec in SYNTHEA_TABLES]

    assert len(set(names)) == len(names)
    assert len(set(file_names)) == len(file_names)


def test_file_name_is_the_table_name_plus_csv() -> None:
    for spec in SYNTHEA_TABLES:
        assert spec.file_name == f"{spec.name}.csv", spec


def test_only_patient_expenses_is_excluded_by_default() -> None:
    optional = [spec.name for spec in SYNTHEA_TABLES if spec.is_optional]

    assert optional == ["patient_expenses"]
    assert all(spec.included_by_default for spec in SYNTHEA_TABLES if spec.name != "patient_expenses")


def test_tables_by_name_indexes_the_catalogue() -> None:
    index = tables_by_name()

    assert set(index) == {spec.name for spec in SYNTHEA_TABLES}
    assert index["patients"].file_name == "patients.csv"


def test_provenance_is_recorded_and_looks_like_a_commit() -> None:
    assert CURRENT_CONTRACT_ID == "synthea-csv-2026-08"
    assert SOURCE_REPOSITORY == "synthetichealth/synthea"
    assert SOURCE_FILE.endswith("CSVConstants.java")
    assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT), SOURCE_COMMIT
