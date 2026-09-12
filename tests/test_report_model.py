"""Tests for the report-level model: categories, findings and the newer fields."""

from __future__ import annotations

import pytest

from synthea_quality.models import (
    CHECK_CATEGORIES,
    CheckResult,
    DatasetReport,
    LoadError,
    Severity,
    Status,
    UnresolvedRelation,
    category_of,
)
from synthea_quality.schema.contract import ContractStatus

FIXED_TIME = "2026-01-01T00:00:00+00:00"


def make_check(
    check_id: str, status: Status = Status.PASS, severity: Severity = Severity.MEDIUM
) -> CheckResult:
    return CheckResult(
        check_id=check_id, status=status, severity=severity, message=f"{check_id} outcome"
    )


def test_category_of_maps_every_family() -> None:
    assert category_of("pk.patients.Id") == "primary_keys"
    assert category_of("fk.conditions.PATIENT->patients.Id") == "foreign_keys"
    assert category_of("dates.conditions.START") == "data_quality"
    assert category_of("duplicates.supplies") == "data_quality"
    assert category_of("empty_columns.payers") == "data_quality"
    assert category_of("nulls.observations") == "data_quality"
    assert category_of("temporal.start_le_stop.conditions") == "temporal"
    assert category_of("schema.contract") == "schema"
    assert category_of("something.unknown") == "other"


def test_every_category_a_check_can_land_in_is_declared() -> None:
    for check_id in ("pk.a", "fk.a->b", "dates.a.B", "temporal.a.b", "schema.x", "mystery"):
        assert category_of(check_id) in CHECK_CATEGORIES


def test_checks_are_grouped_and_counted_by_category() -> None:
    report = DatasetReport(
        data_dir="/tmp/dataset",
        generated_at=FIXED_TIME,
        checks=(
            make_check("pk.patients.Id"),
            make_check("nulls.patients"),
            make_check("temporal.start_le_stop.conditions"),
        ),
    )

    grouped = report.checks_by_category

    assert list(grouped) == list(CHECK_CATEGORIES)
    assert [check.check_id for check in grouped["primary_keys"]] == ["pk.patients.Id"]
    assert [check.check_id for check in grouped["data_quality"]] == ["nulls.patients"]
    assert grouped["foreign_keys"] == ()
    assert report.counts_by_category == {
        "schema": 0,
        "primary_keys": 1,
        "foreign_keys": 0,
        "data_quality": 1,
        "temporal": 1,
        "other": 0,
    }


def test_findings_are_ordered_by_urgency_then_severity() -> None:
    report = DatasetReport(
        data_dir="/tmp/dataset",
        generated_at=FIXED_TIME,
        checks=(
            make_check("nulls.patients", Status.WARNING, Severity.LOW),
            make_check("fk.a->b", Status.WARNING, Severity.HIGH),
            make_check("pk.patients.Id", Status.FAIL, Severity.HIGH),
            make_check("dates.patients.BIRTHDATE", Status.ERROR, Severity.MEDIUM),
            make_check("dates.patients.DEATHDATE", Status.PASS, Severity.MEDIUM),
        ),
    )

    assert [check.check_id for check in report.findings] == [
        "pk.patients.Id",
        "dates.patients.BIRTHDATE",
        "fk.a->b",
        "nulls.patients",
    ]
    assert report.highest_severity_finding is Severity.HIGH


def test_findings_by_severity_counts_only_findings() -> None:
    report = DatasetReport(
        data_dir="/tmp/dataset",
        generated_at=FIXED_TIME,
        checks=(
            make_check("pk.patients.Id", Status.PASS, Severity.HIGH),
            make_check("fk.a->b", Status.WARNING, Severity.HIGH),
            make_check("nulls.patients", Status.WARNING, Severity.LOW),
        ),
    )

    assert report.findings_by_severity == {"HIGH": 1, "MEDIUM": 0, "LOW": 1}


def test_report_without_findings_has_no_highest_severity() -> None:
    report = DatasetReport(
        data_dir="/tmp/dataset",
        generated_at=FIXED_TIME,
        checks=(make_check("pk.patients.Id"), make_check("nulls.patients")),
    )

    assert report.findings == ()
    assert report.highest_severity_finding is None
    assert report.findings_by_severity == {"HIGH": 0, "MEDIUM": 0, "LOW": 0}


def test_new_report_fields_survive_a_json_round_trip() -> None:
    report = DatasetReport(
        data_dir="/tmp/dataset",
        generated_at=FIXED_TIME,
        contract_tables=19,
        contract_status=ContractStatus.INCOMPATIBLE,
        contract_summary="17 of the 19 contract tables were observed; 7 do not match",
        contract_findings=("conditions: missing ['SYSTEM']",),
        load_errors=(LoadError(table="conditions", reason="conditions.csv is empty"),),
        unresolved_relations=(
            UnresolvedRelation(
                kind="foreign_key",
                relation="fk.claims_transactions.PATIENTINSURANCEID->payer_transitions.MEMBERID",
                documented_as="documented",
                implemented_as="implemented",
                reference_dataset="official 2026-08 Synthea sample",
                reference_evidence="170 of 79,453 references",
                why_not_enforced="why",
                pending="pending",
            ),
        ),
    )

    rebuilt = DatasetReport.from_json(report.to_json())

    assert rebuilt.contract_tables == 19
    assert rebuilt.contract_status is ContractStatus.INCOMPATIBLE
    assert rebuilt.contract_summary == report.contract_summary
    assert rebuilt.contract_findings == report.contract_findings
    assert rebuilt.load_errors == report.load_errors
    assert rebuilt.unresolved_relations == report.unresolved_relations
    assert rebuilt.to_dict() == report.to_dict()


def test_contract_status_is_serialised_as_its_plain_name() -> None:
    report = DatasetReport(
        data_dir="/tmp/dataset",
        generated_at=FIXED_TIME,
        contract_status=ContractStatus.COMPATIBLE,
    )

    payload = report.to_dict()

    assert payload["dataset"]["contract_status"] == "COMPATIBLE"
    assert payload["summary"]["by_category"] == {
        "schema": 0,
        "primary_keys": 0,
        "foreign_keys": 0,
        "data_quality": 0,
        "temporal": 0,
        "other": 0,
    }


def test_load_error_requires_its_reason() -> None:
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        LoadError(table="conditions", reason="   ")


def test_unresolved_relation_requires_every_note() -> None:
    with pytest.raises(ValueError, match="pending must be a non-empty string"):
        UnresolvedRelation(
            kind="temporal",
            relation="a <= b",
            documented_as="d",
            implemented_as="i",
            reference_dataset="official 2026-08 Synthea sample",
            reference_evidence="o",
            why_not_enforced="w",
            pending="",
        )


def test_unresolved_relation_requires_a_reference_dataset() -> None:
    """Evidence with no dataset attached could be read as measured on the report's dataset."""
    with pytest.raises(ValueError, match="reference_dataset must be a non-empty string"):
        UnresolvedRelation(
            kind="temporal",
            relation="a <= b",
            documented_as="d",
            implemented_as="i",
            reference_dataset="  ",
            reference_evidence="6,555 of 85,047 rows",
            why_not_enforced="w",
            pending="p",
        )
