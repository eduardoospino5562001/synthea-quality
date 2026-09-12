"""Invariants of the confirmed temporal rule catalogue."""

from __future__ import annotations

import pytest

from synthea_quality.schema import temporal
from synthea_quality.schema.quality import date_rule_for
from synthea_quality.schema.tables import REFERENCE_DATASET, tables_by_name
from synthea_quality.schema.temporal import (
    EVENT_DATE_RULES,
    INTERVAL_RULES,
    LIFE_SPAN_RULES,
    UNRESOLVED_TEMPORAL_RELATIONS,
    EventDateRule,
    IntervalRule,
    LifeSpanRule,
    date_columns_used,
    event_date_rules_for,
    interval_rules_for,
    life_span_rules_for,
)


def test_every_rule_names_columns_that_exist() -> None:
    known = tables_by_name()

    for rule in INTERVAL_RULES:
        assert rule.start_column in known[rule.table].columns
        assert rule.stop_column in known[rule.table].columns
    for rule in LIFE_SPAN_RULES:
        assert rule.birth_column in known[rule.table].columns
        assert rule.death_column in known[rule.table].columns
    for rule in EVENT_DATE_RULES:
        assert rule.column in known[rule.table].columns
        assert rule.patient_column in known[rule.table].columns


def test_every_compared_column_has_a_confirmed_date_format() -> None:
    """The guarantee that keeps date parsing in a single place."""
    for table, column in date_columns_used():
        assert date_rule_for(table, column) is not None


def test_catalogue_contains_exactly_the_confirmed_rule_families() -> None:
    assert len(INTERVAL_RULES) == 7
    assert len(LIFE_SPAN_RULES) == 1
    assert len(EVENT_DATE_RULES) == 11
    assert {rule.table for rule in INTERVAL_RULES} == {
        "allergies",
        "careplans",
        "conditions",
        "devices",
        "encounters",
        "medications",
        "procedures",
    }


def test_check_ids_are_unique_and_descriptive() -> None:
    ids = [
        rule.check_id
        for rule in (*INTERVAL_RULES, *LIFE_SPAN_RULES, *EVENT_DATE_RULES)
    ]

    assert len(ids) == len(set(ids))
    assert "temporal.start_le_stop.conditions" in ids
    assert "temporal.birth_le_death.patients" in ids
    assert "temporal.event_after_birth.observations.DATE" in ids


def test_every_rule_records_the_documentation_it_comes_from() -> None:
    for rule in (*INTERVAL_RULES, *LIFE_SPAN_RULES, *EVENT_DATE_RULES):
        assert "dictionary" in rule.documented_as


def test_only_one_event_column_per_table_to_avoid_double_reporting() -> None:
    by_table: dict[str, list[str]] = {}
    for rule in EVENT_DATE_RULES:
        by_table.setdefault(rule.table, []).append(rule.column)

    assert all(len(columns) == 1 for columns in by_table.values())


def test_no_rule_is_built_on_a_billing_or_coverage_date() -> None:
    """Administrative dates are excluded on purpose; see the module docstring."""
    used = {column for _, column in date_columns_used()}
    excluded = {
        "SERVICEDATE",
        "CURRENTILLNESSDATE",
        "LASTBILLEDDATE1",
        "LASTBILLEDDATE2",
        "LASTBILLEDDATEP",
        "FROMDATE",
        "TODATE",
        "START_DATE",
        "END_DATE",
    }

    assert used.isdisjoint(excluded)


def test_no_clinical_rule_is_smuggled_in() -> None:
    """Events after death, durations, sequences and age rules are explicitly out of scope."""
    used = {column for _, column in date_columns_used()}

    assert "DEATHDATE" in used  # only for birth <= death
    for rule in EVENT_DATE_RULES:
        assert rule.column not in {"DEATHDATE", "STOP"}
    assert all(isinstance(rule, EventDateRule) for rule in EVENT_DATE_RULES)
    assert not hasattr(temporal, "DURATION_RULES")
    assert not hasattr(temporal, "SEQUENCE_RULES")


def test_unresolved_relations_are_documented_and_not_applied() -> None:
    relations = {entry.relation for entry in UNRESOLVED_TEMPORAL_RELATIONS}

    assert relations == {
        "claims_transactions.FROMDATE <= TODATE",
        "payer_transitions.START_DATE <= END_DATE",
    }
    claims, coverage = UNRESOLVED_TEMPORAL_RELATIONS
    # the epoch sentinel is the reason the claim transaction pair is not enforced
    assert claims.relation.startswith("claims_transactions")
    assert "1970-01-01T00:00:00Z" in claims.reference_evidence
    assert "6,555 of 85,047" in claims.reference_evidence
    assert "false failures" in claims.why_not_enforced
    # the evidence is attributed to the reference dataset, never to a reported one
    assert claims.reference_dataset == REFERENCE_DATASET
    assert "2026-08" in claims.reference_dataset
    assert "claimEntry.entry.stop" in claims.implemented_as
    # the coverage pair is clean today, but its writer is the risky one
    assert coverage.relation.startswith("payer_transitions")
    assert "3,815 of 3,815" in coverage.reference_evidence
    assert coverage.reference_dataset == REFERENCE_DATASET
    assert "iso8601Timestamp" in coverage.implemented_as
    for entry in UNRESOLVED_TEMPORAL_RELATIONS:
        assert entry.documented_as.strip() and entry.pending.strip()

    applied = {rule.check_id for rule in INTERVAL_RULES}
    assert "temporal.start_le_stop.claims_transactions" not in applied
    assert "temporal.start_le_stop.payer_transitions" not in applied


def test_lookup_helpers_agree_with_the_catalogue() -> None:
    assert interval_rules_for("conditions") == tuple(
        rule for rule in INTERVAL_RULES if rule.table == "conditions"
    )
    assert life_span_rules_for("patients") == LIFE_SPAN_RULES
    assert life_span_rules_for("conditions") == ()
    assert len(event_date_rules_for("observations")) == 1


# --------------------------------------------------------------------------- #
# the catalogue guard itself
# --------------------------------------------------------------------------- #


def test_rule_for_an_unknown_table_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        temporal, "INTERVAL_RULES", (IntervalRule("ghosts", "START", "STOP", "documented"),)
    )

    with pytest.raises(temporal._CatalogueError, match="unknown table 'ghosts'"):
        temporal._validate_catalogue()


def test_rule_for_a_column_that_does_not_exist_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        temporal, "INTERVAL_RULES", (IntervalRule("patients", "START", "STOP", "documented"),)
    )

    with pytest.raises(temporal._CatalogueError, match="not part of table 'patients'"):
        temporal._validate_catalogue()


def test_rule_without_documentation_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        temporal, "LIFE_SPAN_RULES", (LifeSpanRule("patients", "BIRTHDATE", "DEATHDATE", "  "),)
    )

    with pytest.raises(temporal._CatalogueError, match="has no documentation note"):
        temporal._validate_catalogue()


def test_duplicate_rule_is_rejected(monkeypatch) -> None:
    rule = IntervalRule("conditions", "START", "STOP", "documented")
    monkeypatch.setattr(temporal, "INTERVAL_RULES", (rule, rule))

    with pytest.raises(temporal._CatalogueError, match="duplicate temporal rule"):
        temporal._validate_catalogue()


def test_comparing_a_column_without_a_confirmed_format_is_rejected(monkeypatch) -> None:
    """A rule about a column with no confirmed date format cannot exist.

    The claim transaction pair is a good example of the opposite case: its format
    *is* confirmed, which is why the catalogue rejects it for its semantics, not here.
    """
    unconfirmed = IntervalRule("claims_transactions", "CHARGEID", "ID", "documented")
    monkeypatch.setattr(temporal, "INTERVAL_RULES", (unconfirmed,))

    with pytest.raises(temporal._CatalogueError, match="no confirmed date format"):
        temporal._validate_formats()


def test_unresolved_relation_without_its_notes_is_rejected(monkeypatch) -> None:
    incomplete = temporal.UnresolvedTemporalRelation(
        relation="a <= b",
        documented_as="x",
        implemented_as="y",
        reference_dataset="",
        reference_evidence="e",
        why_not_enforced="w",
        pending="p",
    )
    monkeypatch.setattr(temporal, "UNRESOLVED_TEMPORAL_RELATIONS", (incomplete,))

    with pytest.raises(temporal._CatalogueError, match="needs a 'reference_dataset' note"):
        temporal._validate_catalogue()
