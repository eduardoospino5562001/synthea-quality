"""Invariants of the confirmed date rules.

A date rule that names a column Synthea does not have, or that documents no format,
would produce a confident verdict about nothing: the catalogue must fail at import
time instead.
"""

from __future__ import annotations

import pytest

from synthea_quality.schema import quality
from synthea_quality.schema.quality import (
    DATE_COLUMNS,
    DATE_FORMAT,
    DATE_ONLY,
    ISO8601_UTC,
    STALE_DOCUMENTATION,
    TIMESTAMP_FORMAT,
    DateColumn,
    date_columns_for,
)
from synthea_quality.schema.tables import tables_by_name


def test_every_date_rule_points_at_a_real_column() -> None:
    known = tables_by_name()

    for rule in DATE_COLUMNS:
        assert rule.column in known[rule.table].columns


def test_no_column_has_two_date_rules() -> None:
    pairs = [(rule.table, rule.column) for rule in DATE_COLUMNS]

    assert len(pairs) == len(set(pairs))


def test_catalogue_covers_the_columns_the_dictionary_documents_as_dates() -> None:
    assert len(DATE_COLUMNS) == 29
    assert {rule.table for rule in DATE_COLUMNS} == {
        "allergies",
        "careplans",
        "claims",
        "claims_transactions",
        "conditions",
        "devices",
        "encounters",
        "imaging_studies",
        "immunizations",
        "medications",
        "observations",
        "patients",
        "payer_transitions",
        "procedures",
        "supplies",
    }


def test_the_two_formats_are_the_ones_synthea_writes() -> None:
    assert DATE_FORMAT.identifier == DATE_ONLY
    assert DATE_FORMAT.pattern == r"\d{4}-\d{2}-\d{2}"
    assert TIMESTAMP_FORMAT.identifier == ISO8601_UTC
    assert TIMESTAMP_FORMAT.pattern == r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"


def test_date_only_columns_match_what_the_exporter_writes_with_date_from_timestamp() -> None:
    """CSVExporter uses dateFromTimestamp for allergies, conditions and patients."""
    for table in ("allergies", "careplans", "conditions", "patients", "supplies"):
        for rule in date_columns_for(table):
            assert rule.format is DATE_FORMAT


def test_timestamp_columns_match_what_the_exporter_writes_with_iso8601_timestamp() -> None:
    for table in ("encounters", "observations", "procedures", "medications", "payer_transitions"):
        for rule in date_columns_for(table):
            assert rule.format is TIMESTAMP_FORMAT


def test_payer_transitions_columns_record_the_stale_documentation() -> None:
    """The dictionary still shows START_YEAR/END_YEAR while the code writes these."""
    rules = date_columns_for("payer_transitions")

    assert [rule.column for rule in rules] == ["START_DATE", "END_DATE"]
    for rule in rules:
        assert rule.documented_as == STALE_DOCUMENTATION
        assert "pre-3.0" in rule.documented_as


def test_every_rule_carries_its_documentation() -> None:
    for rule in DATE_COLUMNS:
        assert rule.documented_as.strip()
        assert rule.format.pattern.strip()
        assert rule.check_id == f"dates.{rule.table}.{rule.column}"


def test_tables_without_dates_have_no_rule() -> None:
    for table in ("organizations", "payers", "providers"):
        assert date_columns_for(table) == ()


# --------------------------------------------------------------------------- #
# the catalogue guard itself
# --------------------------------------------------------------------------- #


def test_rule_for_an_unknown_table_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(quality, "DATE_COLUMNS", (DateColumn("ghosts", "START", DATE_FORMAT),))

    with pytest.raises(quality._CatalogueError, match="unknown table 'ghosts'"):
        quality._validate_catalogue()


def test_rule_for_a_column_that_does_not_exist_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        quality, "DATE_COLUMNS", (DateColumn("patients", "NOPE", DATE_FORMAT),)
    )

    with pytest.raises(quality._CatalogueError, match="not part of table 'patients'"):
        quality._validate_catalogue()


def test_rule_without_a_format_pattern_is_rejected(monkeypatch) -> None:
    broken = DateColumn("patients", "BIRTHDATE", quality.DateFormat("x", "  ", "doc", "label"))
    monkeypatch.setattr(quality, "DATE_COLUMNS", (broken,))

    with pytest.raises(quality._CatalogueError, match="has no format pattern"):
        quality._validate_catalogue()


def test_rule_without_a_documentation_note_is_rejected(monkeypatch) -> None:
    broken = DateColumn("patients", "BIRTHDATE", DATE_FORMAT, documented_as="   ")
    monkeypatch.setattr(quality, "DATE_COLUMNS", (broken,))

    with pytest.raises(quality._CatalogueError, match="has no documentation note"):
        quality._validate_catalogue()


def test_duplicate_rule_for_the_same_column_is_rejected(monkeypatch) -> None:
    rule = DateColumn("patients", "BIRTHDATE", DATE_FORMAT)
    monkeypatch.setattr(quality, "DATE_COLUMNS", (rule, rule))

    with pytest.raises(quality._CatalogueError, match="duplicate date rule for patients.BIRTHDATE"):
        quality._validate_catalogue()
