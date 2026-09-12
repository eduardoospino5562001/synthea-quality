"""Tests for the structured result models."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from synthea_quality.models import (
    DEFAULT_SAMPLE_LIMIT,
    REPORT_SCHEMA_VERSION,
    CheckResult,
    DatasetReport,
    Severity,
    Status,
    TableSummary,
    utc_now_iso,
)


def make_check(
    check_id: str = "pk.patients.Id",
    status: Status = Status.PASS,
    severity: Severity = Severity.HIGH,
    **kwargs: object,
) -> CheckResult:
    kwargs.setdefault("message", "patients.Id is unique")
    return CheckResult(
        check_id=check_id,
        status=status,
        severity=severity,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# status / severity
# --------------------------------------------------------------------------- #


def test_status_and_severity_serialise_to_plain_strings() -> None:
    assert Status.FAIL.value == "FAIL"
    assert Severity.HIGH.value == "HIGH"
    payload = json.dumps({"status": Status.FAIL, "severity": Severity.LOW})
    assert json.loads(payload) == {"status": "FAIL", "severity": "LOW"}


def test_every_status_is_covered_by_counts() -> None:
    report = DatasetReport(data_dir="/tmp/dataset")
    assert set(report.counts_by_status) == {status.value for status in Status}
    assert all(count == 0 for count in report.counts_by_status.values())


def test_severity_ranks_are_ordered() -> None:
    assert Severity.LOW.rank < Severity.MEDIUM.rank < Severity.HIGH.rank


# --------------------------------------------------------------------------- #
# CheckResult
# --------------------------------------------------------------------------- #


def test_check_result_to_dict_is_json_serialisable() -> None:
    check = make_check(
        metrics={"rows_checked": 3517, "duplicate_keys": 0},
        samples=({"Id": "abc"},),
        metadata={"columns": ["Id"], "rule": "primary key must be unique"},
    )
    payload = json.dumps(check.to_dict(), ensure_ascii=False)

    assert json.loads(payload)["check_id"] == "pk.patients.Id"
    assert json.loads(payload)["metrics"]["rows_checked"] == 3517


def test_check_result_round_trip() -> None:
    check = make_check(
        table="patients",
        status=Status.FAIL,
        severity=Severity.MEDIUM,
        metrics={"invalid": 2},
        samples=({"Id": "dup-1"}, {"Id": "dup-2"}),
        metadata={"rule": "no duplicate ids"},
    )

    assert CheckResult.from_dict(check.to_dict()) == check


def test_check_result_rejects_empty_check_id_and_message() -> None:
    with pytest.raises(ValueError, match="check_id"):
        CheckResult(check_id="   ", status=Status.PASS, severity=Severity.LOW, message="ok")
    with pytest.raises(ValueError, match="message"):
        CheckResult(check_id="pk.patients.Id", status=Status.PASS, severity=Severity.LOW, message="")


def test_check_result_rejects_plain_string_status_or_severity() -> None:
    plain_status: Any = "PASS"
    plain_severity: Any = "LOW"

    with pytest.raises(TypeError, match="status"):
        CheckResult(
            check_id="pk.patients.Id",
            status=plain_status,
            severity=Severity.LOW,
            message="ok",
        )
    with pytest.raises(TypeError, match="severity"):
        CheckResult(
            check_id="pk.patients.Id",
            status=Status.PASS,
            severity=plain_severity,
            message="ok",
        )


def test_check_result_rejects_wrong_container_types() -> None:
    with pytest.raises(TypeError, match="metrics"):
        make_check(metrics=[("rows", 1)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="samples"):
        make_check(samples=[{"Id": "abc"}])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sample"):
        make_check(samples=("not-a-mapping",))  # type: ignore[arg-type]


def test_check_result_defaults_are_not_shared_between_instances() -> None:
    first = make_check()
    second = make_check()

    first.metrics["rows_checked"] = 10  # type: ignore[index]
    first.metadata["rule"] = "changed"  # type: ignore[index]

    assert second.metrics == {}
    assert second.metadata == {}
    assert second.samples == ()


def test_from_dict_rejects_unknown_status() -> None:
    payload = make_check().to_dict()
    payload["status"] = "MAYBE"

    with pytest.raises(ValueError, match="MAYBE"):
        CheckResult.from_dict(payload)


def test_from_dict_ignores_unknown_keys() -> None:
    payload = make_check().to_dict()
    payload["future_optional_key"] = "ignored"

    assert CheckResult.from_dict(payload) == make_check()


def test_bounded_samples_default_is_declared() -> None:
    assert DEFAULT_SAMPLE_LIMIT > 0


# --------------------------------------------------------------------------- #
# TableSummary
# --------------------------------------------------------------------------- #


def test_table_summary_before_loading() -> None:
    summary = TableSummary(name="patients", file_name="patients.csv")

    assert summary.rows is None
    assert summary.columns == ()
    assert summary.schema_match is None
    assert TableSummary.from_dict(summary.to_dict()) == summary


def test_table_summary_round_trip_after_loading() -> None:
    summary = TableSummary(
        name="observations",
        file_name="observations.csv",
        rows=68648,
        columns=("DATE", "PATIENT", "ENCOUNTER"),
        schema_match=True,
    )

    assert TableSummary.from_dict(summary.to_dict()) == summary


def test_table_summary_rejects_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="rows"):
        TableSummary(name="patients", file_name="patients.csv", rows=-1)
    with pytest.raises(ValueError, match="file_name"):
        TableSummary(name="patients", file_name="")


# --------------------------------------------------------------------------- #
# DatasetReport
# --------------------------------------------------------------------------- #


def make_report(**kwargs: object) -> DatasetReport:
    table = TableSummary(
        name="patients",
        file_name="patients.csv",
        rows=108,
        columns=("Id", "BIRTHDATE"),
        schema_match=True,
    )
    checks = (
        make_check(),
        make_check(
            check_id="fk.conditions.PATIENT->patients.Id",
            status=Status.FAIL,
            severity=Severity.HIGH,
            message="3 orphan references",
            table="conditions",
            metrics={"references_total": 3517, "references_invalid": 3, "invalid_pct": 0.09},
            samples=({"PATIENT": "missing-1"},),
        ),
    )
    defaults: dict[str, object] = {
        "data_dir": "/data/sample",
        "tables": (table,),
        "checks": checks,
        "generated_at": "2026-09-12T13:00:00+00:00",
    }
    defaults.update(kwargs)
    return DatasetReport(**defaults)  # type: ignore[arg-type]


def test_report_counts_by_status_and_failure_flag() -> None:
    report = make_report()

    assert report.has_failures is True
    assert report.counts_by_status["PASS"] == 1
    assert report.counts_by_status["FAIL"] == 1
    assert report.counts_by_status["WARNING"] == 0


def test_report_without_findings_is_clean() -> None:
    report = DatasetReport(
        data_dir="/data/sample",
        checks=(make_check(),),
        generated_at="2026-09-12T13:00:00+00:00",
    )

    assert report.has_failures is False
    assert report.highest_severity_finding is None
    assert report.to_dict()["summary"]["highest_severity_finding"] is None


def test_highest_severity_finding_ignores_passing_checks() -> None:
    report = make_report(
        checks=(
            make_check(severity=Severity.HIGH),  # PASS: high severity, not a finding
            make_check(
                check_id="dq.duplicate_rows.conditions",
                status=Status.WARNING,
                severity=Severity.MEDIUM,
                message="duplicate rows",
            ),
        )
    )

    assert report.highest_severity_finding is Severity.MEDIUM


def test_report_round_trip_through_json() -> None:
    report = make_report(
        unknown_files=("notes.txt",),
        missing_known_tables=("supplies",),
        schema_contract="synthea-csv-2026-08",
    )

    text = report.to_json()
    restored = DatasetReport.from_json(text)

    assert restored == report
    assert restored.schema_contract == "synthea-csv-2026-08"
    assert json.loads(text)["schema_version"] == REPORT_SCHEMA_VERSION


def test_report_json_keeps_non_ascii_readable() -> None:
    report = make_report(
        checks=(make_check(message="columna «Id» duplicada"),),
    )

    assert "columna «Id» duplicada" in report.to_json()
    assert "\\u00ab" not in report.to_json()


def test_report_rejects_unknown_schema_version() -> None:
    payload = make_report().to_dict()
    payload["schema_version"] = REPORT_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema_version"):
        DatasetReport.from_dict(payload)


def test_report_schema_version_is_stamped_in_output() -> None:
    assert make_report().to_dict()["schema_version"] == REPORT_SCHEMA_VERSION


def test_generated_at_default_is_iso8601_utc() -> None:
    generated_at = utc_now_iso()
    parsed = datetime.fromisoformat(generated_at)
    offset = parsed.utcoffset()

    assert parsed.tzinfo is not None
    assert offset is not None
    assert offset.total_seconds() == 0

    report = DatasetReport(data_dir="/data/sample")
    assert datetime.fromisoformat(report.generated_at)
