"""Invariants of the confirmed key-rule catalogue.

These tests protect the catalogue itself: a rule that points at a table or column
Synthea does not have would produce confident but wrong verdicts, so it must fail
at import time instead.
"""

from __future__ import annotations

import pytest

from synthea_quality.schema import keys
from synthea_quality.schema.keys import (
    FOREIGN_KEYS,
    PRIMARY_KEYS,
    UNRESOLVED_FOREIGN_KEYS,
    ForeignKeyRule,
    PrimaryKeyRule,
    foreign_keys_for,
    parent_columns,
    primary_key_for,
)
from synthea_quality.schema.tables import tables_by_name


def test_every_rule_points_at_a_real_table_and_column() -> None:
    known = tables_by_name()

    for rule in PRIMARY_KEYS:
        assert rule.column in known[rule.table].columns
    for rule in FOREIGN_KEYS:
        assert rule.column in known[rule.table].columns
        assert rule.parent_column in known[rule.parent_table].columns


def test_primary_key_columns_are_the_documented_unique_identifiers() -> None:
    assert {rule.table for rule in PRIMARY_KEYS} == {
        "careplans",
        "claims",
        "claims_transactions",
        "encounters",
        "organizations",
        "patients",
        "payers",
        "providers",
    }
    assert len(PRIMARY_KEYS) == 8
    claims_transactions = primary_key_for("claims_transactions")
    assert claims_transactions is not None
    assert claims_transactions.column == "ID"


def test_imaging_studies_has_no_primary_key_rule() -> None:
    """The dictionary documents its Id as non-unique: repeated values are expected."""
    assert primary_key_for("imaging_studies") is None
    assert "imaging_studies" not in {rule.table for rule in PRIMARY_KEYS}


def test_tables_without_an_identifier_column_have_no_primary_key_rule() -> None:
    for table in ("allergies", "conditions", "devices", "immunizations", "medications",
                  "observations", "payer_transitions", "procedures", "supplies"):
        assert primary_key_for(table) is None


def test_foreign_key_catalogue_is_complete_for_the_child_tables() -> None:
    assert len(FOREIGN_KEYS) == 42
    assert [rule.column for rule in foreign_keys_for("observations")] == ["PATIENT", "ENCOUNTER"]
    assert len(foreign_keys_for("claims")) == 7
    assert foreign_keys_for("organizations") == ()
    assert foreign_keys_for("patients") == ()


def test_parent_columns_lists_only_enforced_targets() -> None:
    parents = parent_columns()

    assert ("patients", "Id") in parents
    assert ("encounters", "Id") in parents
    # the rejected relationship must not create a parent key index
    assert ("payer_transitions", "MEMBERID") not in parents


def test_unresolved_relationship_is_documented_and_not_applied() -> None:
    unresolved = UNRESOLVED_FOREIGN_KEYS[0]

    assert unresolved.rule.check_id == (
        "fk.claims_transactions.PATIENTINSURANCEID->payer_transitions.MEMBERID"
    )
    # documented, but neither applied as a rule nor declared wrong
    assert unresolved.rule not in FOREIGN_KEYS
    assert "Foreign key to the Payer Transitions table member ID" in unresolved.documented_as
    assert "170 of 79,453" in unresolved.observed
    assert "CSVExporter.java:1566" in unresolved.implemented_as
    assert "false alarm" in unresolved.why_not_enforced
    assert "maintainer's confirmation" in unresolved.pending


def test_unresolved_relationship_produces_no_verdict() -> None:
    """It must not appear among the checks that are run, so it can never FAIL."""
    from synthea_quality.schema.keys import FOREIGN_KEYS as applied

    assert all(
        rule.check_id != UNRESOLVED_FOREIGN_KEYS[0].rule.check_id for rule in applied
    )


def test_unresolved_rule_without_its_documentation_is_rejected(monkeypatch) -> None:
    incomplete = keys.UnresolvedForeignKey(
        rule=ForeignKeyRule("conditions", "PATIENT", "patients"),
        documented_as="something",
        observed="",
        implemented_as="x",
        why_not_enforced="y",
        pending="z",
    )
    monkeypatch.setattr(keys, "UNRESOLVED_FOREIGN_KEYS", (incomplete,))

    with pytest.raises(keys._CatalogueError, match="needs a 'observed' note"):
        keys._validate_catalogue()


def test_every_rule_records_its_provenance() -> None:
    for rule in (*PRIMARY_KEYS, *FOREIGN_KEYS):
        assert rule.confirmed_by == keys.DATA_DICTIONARY
        assert rule.confirmed_by.strip()


def test_check_ids_read_as_the_rule_they_apply() -> None:
    assert PrimaryKeyRule("patients", "Id").check_id == "pk.patients.Id"
    assert (
        ForeignKeyRule("conditions", "PATIENT", "patients").check_id
        == "fk.conditions.PATIENT->patients.Id"
    )
    assert (
        ForeignKeyRule("claims_transactions", "CLAIMID", "claims").check_id
        == "fk.claims_transactions.CLAIMID->claims.Id"
    )


# --------------------------------------------------------------------------- #
# the catalogue guard itself
# --------------------------------------------------------------------------- #


def test_rule_for_an_unknown_table_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(keys, "PRIMARY_KEYS", (PrimaryKeyRule("ghosts", "Id"),))

    with pytest.raises(keys._CatalogueError, match="unknown table 'ghosts'"):
        keys._validate_catalogue()


def test_rule_for_a_column_that_does_not_exist_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(keys, "FOREIGN_KEYS", (ForeignKeyRule("patients", "PATIENT", "patients"),))

    with pytest.raises(keys._CatalogueError, match="'PATIENT', which is not part of table 'patients'"):
        keys._validate_catalogue()


def test_rule_with_a_bad_target_column_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        keys,
        "FOREIGN_KEYS",
        (ForeignKeyRule("conditions", "PATIENT", "patients", "NOPE"),),
    )

    with pytest.raises(keys._CatalogueError, match="unknown column|not part of table 'patients'"):
        keys._validate_catalogue()


def test_duplicate_rules_are_rejected(monkeypatch) -> None:
    rule = PrimaryKeyRule("patients", "Id")
    monkeypatch.setattr(keys, "PRIMARY_KEYS", (rule, rule))

    with pytest.raises(keys._CatalogueError, match="duplicate pk rule for patients.Id"):
        keys._validate_catalogue()


def test_local_and_unknown_rule_failures_stay_distinguishable(monkeypatch) -> None:
    """A rule pointing at an unknown table must not be confused with a dataset problem."""
    monkeypatch.setattr(
        keys, "FOREIGN_KEYS", (ForeignKeyRule("conditions", "PATIENT", "ghosts"),)
    )

    with pytest.raises(keys._CatalogueError, match="unknown table 'ghosts'"):
        keys._validate_catalogue()
