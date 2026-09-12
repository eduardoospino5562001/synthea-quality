"""Tests for versioned schema contracts and header matching."""

from __future__ import annotations

import pytest

from synthea_quality.schema.contract import (
    CURRENT_CONTRACT,
    ContractStatus,
    MatchKind,
    SchemaContract,
    assess_contract,
)
from synthea_quality.schema.tables import SYNTHEA_TABLES, TableSpec, tables_by_name

CONDITIONS = tables_by_name()["conditions"].columns


# --------------------------------------------------------------------------- #
# contract object
# --------------------------------------------------------------------------- #


def test_current_contract_is_the_2026_08_version() -> None:
    assert CURRENT_CONTRACT.contract_id == "synthea-csv-2026-08"
    assert CURRENT_CONTRACT.tables == SYNTHEA_TABLES
    assert len(CURRENT_CONTRACT.table_names) == 19
    assert CURRENT_CONTRACT.source_commit


def test_contract_lookup_rejects_unknown_tables() -> None:
    with pytest.raises(KeyError, match="has no table"):
        CURRENT_CONTRACT.table("not_a_synthea_table")


def test_a_second_contract_can_be_defined_without_touching_the_first() -> None:
    """Future versions plug in as another SchemaContract instance."""
    legacy = SchemaContract(
        contract_id="synthea-csv-example",
        tables=(TableSpec("conditions", "conditions.csv", columns=("START", "STOP", "PATIENT")),),
        source_repository="synthetichealth/synthea",
        source_commit="0" * 40,
        source_file="example",
    )

    assessment = legacy.assess({"conditions": ("START", "STOP", "PATIENT")})

    assert assessment.status is ContractStatus.COMPATIBLE
    assert assessment.contract_id == "synthea-csv-example"
    # the current contract still describes the real schema
    assert CURRENT_CONTRACT.table("conditions").columns == CONDITIONS


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #


def test_exact_header_matches() -> None:
    match = CURRENT_CONTRACT.match_header("conditions", CONDITIONS)

    assert match.kind is MatchKind.EXACT
    assert match.is_exact is True
    assert match.missing == ()
    assert match.extra == ()
    assert match.misplaced == ()
    assert "matches the contract" in match.describe()


def test_missing_column_is_detected() -> None:
    actual = tuple(column for column in CONDITIONS if column != "SYSTEM")

    match = CURRENT_CONTRACT.match_header("conditions", actual)

    assert match.kind is MatchKind.COLUMNS_DIFFERENT
    assert match.missing == ("SYSTEM",)
    assert match.extra == ()
    assert "missing ['SYSTEM']" in match.describe()


def test_extra_column_is_detected() -> None:
    actual = CONDITIONS + ("EXTRA_COLUMN",)

    match = CURRENT_CONTRACT.match_header("conditions", actual)

    assert match.kind is MatchKind.COLUMNS_DIFFERENT
    assert match.extra == ("EXTRA_COLUMN",)
    assert match.missing == ()


def test_missing_and_extra_columns_are_reported_together() -> None:
    # The 2021 shape of this table: no SYSTEM column, but with a legacy NAME column.
    actual = ("START", "STOP", "PATIENT", "ENCOUNTER", "NAME", "CODE", "DESCRIPTION")

    match = CURRENT_CONTRACT.match_header("conditions", actual)

    assert match.kind is MatchKind.COLUMNS_DIFFERENT
    assert match.missing == ("SYSTEM",)
    assert match.extra == ("NAME",)


def test_duplicated_column_is_detected() -> None:
    actual = ("START", "STOP", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "CODE")

    match = CURRENT_CONTRACT.match_header("conditions", actual)

    assert match.kind is MatchKind.COLUMNS_DIFFERENT
    assert match.duplicated == ("CODE",)
    assert "duplicated ['CODE']" in match.describe()


def test_reordered_columns_are_reported_as_order_difference() -> None:
    actual = ("START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "SYSTEM", "DESCRIPTION")

    match = CURRENT_CONTRACT.match_header("conditions", actual)

    assert match.kind is MatchKind.ORDER_DIFFERS
    assert match.missing == ()
    assert match.extra == ()
    assert match.misplaced == ("CODE", "SYSTEM")
    assert "out of order" in match.describe()


def test_order_difference_does_not_hide_a_missing_column() -> None:
    actual = ("STOP", "START", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE")  # no DESCRIPTION

    match = CURRENT_CONTRACT.match_header("conditions", actual)

    assert match.kind is MatchKind.COLUMNS_DIFFERENT
    assert match.missing == ("DESCRIPTION",)


def test_empty_header_is_columns_different() -> None:
    match = CURRENT_CONTRACT.match_header("conditions", ())

    assert match.kind is MatchKind.COLUMNS_DIFFERENT
    assert match.missing == CONDITIONS
    assert "missing" in match.describe()


# --------------------------------------------------------------------------- #
# assessment
# --------------------------------------------------------------------------- #


def test_assessment_of_a_matching_dataset_is_compatible_without_over_claiming() -> None:
    headers = {name: tables_by_name()[name].columns for name in ("patients", "conditions", "encounters")}

    assessment = assess_contract(headers)

    assert assessment.status is ContractStatus.COMPATIBLE
    assert assessment.tables_assessed == ("conditions", "encounters", "patients")
    assert assessment.non_matching == ()
    assert assessment.reasons == ()
    # The wording states what was checked, not that the dataset version was proven.
    assert "3 of the 19 contract tables were observed" in assessment.summary
    assert "every observed table matches" in assessment.summary


def test_one_non_matching_table_makes_the_assessment_incompatible() -> None:
    headers = {
        "patients": tables_by_name()["patients"].columns,
        "conditions": ("START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"),  # 2021 shape
    }

    assessment = assess_contract(headers)

    assert assessment.status is ContractStatus.INCOMPATIBLE
    assert [match.table for match in assessment.non_matching] == ["conditions"]
    assert assessment.reasons and "conditions" in assessment.reasons[0]
    assert "1 do not match" in assessment.summary


def test_absent_tables_do_not_make_a_dataset_incompatible() -> None:
    """Synthea can legitimately omit tables, so absence is not evidence against a contract.

    It is not evidence *for* it either: the assessment stays a statement about the
    tables that were observed.
    """
    headers = {"patients": tables_by_name()["patients"].columns}

    assessment = assess_contract(headers)

    assert assessment.status is ContractStatus.COMPATIBLE
    assert "1 of the 19 contract tables were observed" in assessment.summary


def test_assessment_without_any_header_is_unknown() -> None:
    assessment = assess_contract({})

    assert assessment.status is ContractStatus.UNKNOWN
    assert assessment.tables_assessed == ()
    assert "cannot be assessed" in assessment.summary
    assert "no known tables were found" in assessment.reasons[0]


def test_full_observation_of_the_contract_is_reported_as_such() -> None:
    headers = {spec.name: spec.columns for spec in SYNTHEA_TABLES}

    assessment = assess_contract(headers)

    assert assessment.status is ContractStatus.COMPATIBLE
    assert "all 19 known tables" in assessment.summary


def test_assessment_rejects_headers_for_unknown_tables() -> None:
    with pytest.raises(ValueError, match="not part of contract"):
        assess_contract({"patients_backup": ("Id",)})


def test_assessment_is_deterministic_in_table_order() -> None:
    headers = {
        "observations": tables_by_name()["observations"].columns,
        "allergies": tables_by_name()["allergies"].columns,
        "patients": tables_by_name()["patients"].columns,
    }

    assert assess_contract(headers).tables_assessed == (
        "allergies",
        "observations",
        "patients",
    )
