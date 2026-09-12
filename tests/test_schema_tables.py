"""Tests for the table catalogue of the current schema contract.

These tests protect the catalogue's internal consistency and its provenance, and
pin a few columns against the Synthea source so that an accidental edit is caught.
The cross-check against a real Synthea dataset happens in the integration test.
"""

from __future__ import annotations

import re

import pytest

from synthea_quality.schema.tables import (
    CONTRACT_ID,
    SOURCE_COMMIT,
    SOURCE_FILE,
    SOURCE_REPOSITORY,
    SYNTHEA_TABLES,
    TableSpec,
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


def test_every_table_declares_unique_columns() -> None:
    for spec in SYNTHEA_TABLES:
        assert spec.columns, spec
        assert len(set(spec.columns)) == len(spec.columns), spec


def test_only_patient_expenses_is_excluded_by_default() -> None:
    optional = [spec.name for spec in SYNTHEA_TABLES if spec.is_optional]

    assert optional == ["patient_expenses"]


def test_tables_by_name_indexes_the_catalogue() -> None:
    index = tables_by_name()

    assert set(index) == {spec.name for spec in SYNTHEA_TABLES}
    assert index["patients"].file_name == "patients.csv"


# --------------------------------------------------------------------------- #
# columns pinned against the Synthea source
# --------------------------------------------------------------------------- #


def test_patients_columns_match_the_current_exporter() -> None:
    columns = tables_by_name()["patients"].columns

    # 2026 schema: MIDDLE, FIPS and INCOME are present (the 2021 dataset lacks them).
    assert len(columns) == 28
    assert columns[:4] == ("Id", "BIRTHDATE", "DEATHDATE", "SSN")
    assert {"MIDDLE", "FIPS", "INCOME"} <= set(columns)


def test_conditions_and_observations_columns_are_exact() -> None:
    index = tables_by_name()

    assert index["conditions"].columns == (
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "SYSTEM",
        "CODE",
        "DESCRIPTION",
    )
    assert index["observations"].columns == (
        "DATE",
        "PATIENT",
        "ENCOUNTER",
        "CATEGORY",
        "CODE",
        "DESCRIPTION",
        "VALUE",
        "UNITS",
        "TYPE",
    )


def test_claims_tables_keep_their_wide_headers() -> None:
    index = tables_by_name()

    assert len(index["claims"].columns) == 31
    assert len(index["claims_transactions"].columns) == 33
    assert index["claims_transactions"].columns[0] == "ID"


# --------------------------------------------------------------------------- #
# provenance and validation
# --------------------------------------------------------------------------- #


def test_provenance_is_recorded_and_looks_like_a_commit() -> None:
    assert CONTRACT_ID == "synthea-csv-2026-08"
    assert SOURCE_REPOSITORY == "synthetichealth/synthea"
    assert SOURCE_FILE.endswith("CSVConstants.java")
    assert re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT), SOURCE_COMMIT


def test_table_spec_rejects_incomplete_definitions() -> None:
    with pytest.raises(ValueError, match="columns"):
        TableSpec("patients", "patients.csv", columns=())
    with pytest.raises(ValueError, match="duplicate columns"):
        TableSpec("patients", "patients.csv", columns=("Id", "Id"))
    with pytest.raises(ValueError, match="non-empty"):
        TableSpec("", "patients.csv", columns=("Id",))
