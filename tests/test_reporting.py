"""Tests for the reporting layer: aggregation, JSON and Markdown.

The scenarios cover a clean dataset, failures, warnings, skips, tool errors, an
unresolvable contract and hostile sample content, plus the two properties that make
a report usable: deterministic output and a lossless JSON round-trip.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from synthea_quality.errors import EmptyDatasetError
from synthea_quality.models import (
    CheckResult,
    DatasetReport,
    Severity,
    Status,
    TableSummary,
    UnresolvedRelation,
)
from synthea_quality.reporting import json_report
from synthea_quality.reporting.build import build_report, unresolved_relations
from synthea_quality.reporting.markdown import render_markdown
from synthea_quality.schema.contract import ContractStatus
from synthea_quality.schema.tables import tables_by_name

FIXED_TIME = "2026-01-01T00:00:00+00:00"


def write_table(directory: Path, name: str, rows: list[dict[str, str]]) -> Path:
    """Write a table with the header the contract expects and only the given values.

    Using the real header keeps the fixture contract-compatible, which is what the
    scenarios about the contract need; the columns left out end up empty, as they
    would in a dataset where a module does not populate them.
    """
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_raw(directory: Path, name: str, text: str) -> Path:
    """Write a file exactly as given, for the malformed and broken-dataset cases."""
    path = directory / f"{name}.csv"
    path.write_text(text, encoding="utf-8", newline="")
    return path


def clean_dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "p1", "BIRTHDATE": "1980-01-01"},
            {"Id": "p2", "BIRTHDATE": "1990-02-02", "DEATHDATE": "2020-03-03"},
        ],
    )
    write_table(
        directory,
        "encounters",
        [
            {
                "Id": "e1",
                "START": "2020-01-01T08:00:00Z",
                "STOP": "2020-01-01T09:00:00Z",
                "PATIENT": "p1",
            }
        ],
    )
    write_table(directory, "conditions", [{"START": "2000-01-01", "PATIENT": "p1"}])
    return directory


def make_check(
    check_id: str,
    status: Status = Status.PASS,
    severity: Severity = Severity.MEDIUM,
    **kwargs: Any,
) -> CheckResult:
    kwargs.setdefault("message", f"{check_id} outcome")
    return CheckResult(check_id=check_id, status=status, severity=severity, **kwargs)


def make_report(*checks: CheckResult, **kwargs: Any) -> DatasetReport:
    kwargs.setdefault("data_dir", "/tmp/example")
    kwargs.setdefault("generated_at", FIXED_TIME)
    return DatasetReport(checks=tuple(checks), **kwargs)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def test_build_report_aggregates_every_check_family(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    assert report.schema_contract == "synthea-csv-2026-08"
    assert report.contract_status is ContractStatus.COMPATIBLE
    assert report.contract_tables == 19
    assert report.generated_at == FIXED_TIME
    assert report.tool_version
    assert report.schema_version == 1
    identifiers = {check.check_id for check in report.checks}
    assert any(name.startswith("pk.") for name in identifiers)
    assert any(name.startswith("fk.") for name in identifiers)
    assert any(name.startswith("nulls.") for name in identifiers)
    assert any(name.startswith("temporal.") for name in identifiers)


def test_build_report_fills_the_table_inventory_without_reloading(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)
    by_name = {table.name: table for table in report.tables}

    assert by_name["patients"].rows == 2
    assert by_name["encounters"].rows == 1
    assert by_name["patients"].columns[:2] == ("Id", "BIRTHDATE")
    assert by_name["conditions"].schema_match is True
    assert report.unknown_files == ()
    assert "patient_expenses" in report.missing_known_tables


def test_build_report_records_unreadable_tables_as_load_errors(tmp_path: Path) -> None:
    clean_dataset(tmp_path)
    write_raw(tmp_path, "conditions", "")

    report = build_report(tmp_path, generated_at=FIXED_TIME)

    assert [error.table for error in report.load_errors] == ["conditions"]
    assert "empty" in report.load_errors[0].reason


def test_build_report_of_an_empty_directory_is_an_input_error(tmp_path: Path) -> None:
    """A directory with no known table is refused, not reported as entirely skipped.

    Changed when the audit finding was fixed: this used to return a report whose
    contract was ``UNKNOWN`` and whose 155 checks were all ``SKIPPED``, which the CLI
    turned into exit code 0 — that is, a clean verdict about nothing at all.
    """
    with pytest.raises(EmptyDatasetError) as error:
        build_report(tmp_path, generated_at=FIXED_TIME)

    assert "patients.csv" in str(error.value)
    assert str(tmp_path) in str(error.value)


def test_build_report_of_an_old_layout_is_incompatible(tmp_path: Path) -> None:
    write_raw(
        tmp_path,
        "conditions",
        "START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION\n2000-01-01,,p1,e1,123,A\n",
    )

    report = build_report(tmp_path, generated_at=FIXED_TIME)

    assert report.contract_status is ContractStatus.INCOMPATIBLE
    assert report.contract_findings
    assert any("conditions" in finding for finding in report.contract_findings)


def test_build_report_carries_the_unresolved_relations() -> None:
    relations = unresolved_relations()

    assert {relation.kind for relation in relations} == {"foreign_key", "temporal"}
    assert "fk.claims_transactions.PATIENTINSURANCEID->payer_transitions.MEMBERID" in {
        relation.relation for relation in relations
    }
    assert all(relation.pending.strip() for relation in relations)


def test_check_order_is_deterministic(tmp_path: Path) -> None:
    first = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)
    second = build_report(tmp_path, generated_at=FIXED_TIME)

    assert [check.check_id for check in first.checks] == [
        check.check_id for check in second.checks
    ]
    assert [check.check_id for check in first.checks] == sorted(
        check.check_id for check in first.checks
    )


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def test_json_round_trip_is_lossless(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    text = json_report.dumps(report)
    rebuilt = json_report.loads(text)

    assert rebuilt.to_dict() == report.to_dict()
    assert [check.check_id for check in rebuilt.checks] == [
        check.check_id for check in report.checks
    ]


def test_json_serialisation_is_idempotent(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    once = json_report.dumps(report)

    assert json_report.dumps(json_report.loads(once)) == once


def test_json_keeps_metrics_and_samples(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)
    payload = json.loads(json_report.dumps(report))

    with_metrics = [check for check in payload["checks"] if check["metrics"]]
    with_samples = [check for check in payload["checks"] if check["samples"]]

    assert with_metrics and with_samples
    assert all("message" in check and "metadata" in check for check in payload["checks"])


def test_json_carries_no_markdown(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    text = json_report.dumps(report)

    assert "```" not in text
    assert "# " not in text
    assert "| ---" not in text


def test_json_reports_the_summary_that_matches_the_checks(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)
    payload = json.loads(json_report.dumps(report))
    summary = payload["summary"]

    assert summary["checks_total"] == len(report.checks)
    assert sum(summary["by_status"].values()) == len(report.checks)
    assert sum(summary["by_category"].values()) == len(report.checks)
    assert set(summary["by_status"]) == {status.value for status in Status}
    assert summary["findings_by_severity"] == report.findings_by_severity


def test_json_file_helpers_round_trip(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)
    target = json_report.dump(report, tmp_path / "reports" / "report.json")

    assert target.exists()
    assert json_report.load(target).to_dict() == report.to_dict()
    assert target.read_text(encoding="utf-8").endswith("\n")


def test_json_written_by_an_earlier_version_still_loads() -> None:
    """Keys added inside one report schema version are optional."""
    legacy = {
        "schema_version": 1,
        "dataset": {
            "data_dir": "/data",
            "generated_at": FIXED_TIME,
            "tool_version": "0.1.0",
            "schema_contract": "synthea-csv-2026-08",
            "unknown_files": [],
            "missing_known_tables": ["patient_expenses"],
        },
        "tables": [],
        "checks": [],
        "summary": {"checks_total": 0},
    }

    report = json_report.loads(json.dumps(legacy))

    assert report.data_dir == "/data"
    assert report.contract_status is None
    assert report.contract_tables == 0
    assert report.load_errors == ()
    assert report.unresolved_relations == ()


def test_json_refuses_a_future_schema_version() -> None:
    with pytest.raises(ValueError, match="unsupported report schema_version 99"):
        json_report.loads(json.dumps({"schema_version": 99, "dataset": {"data_dir": "/d"}}))


# --------------------------------------------------------------------------- #
# Markdown: scenarios
# --------------------------------------------------------------------------- #


def test_markdown_of_a_clean_report_says_so() -> None:
    """No dataset is perfectly clean (the official sample has warnings too).

    What matters is that the renderer's clean path is unambiguous, so it is checked
    on a report whose checks all passed.
    """
    report = make_report(
        make_check("pk.patients.Id"),
        make_check("nulls.patients", severity=Severity.LOW),
        make_check("dates.patients.BIRTHDATE", severity=Severity.MEDIUM),
    )

    text = render_markdown(report)

    assert "# Synthea dataset quality report" in text
    assert "Verdict: no check failed and no check produced a warning." in text
    assert "No check failed, no check errored and no check produced a warning." in text
    assert f"**total** | **{len(report.checks)}**" in text


def test_build_report_of_a_consistent_dataset_has_no_failure(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    assert report.counts_by_status[Status.FAIL.value] == 0
    text = render_markdown(report)
    assert "check(s) failed" not in text
    assert f"**total** | **{len(report.checks)}**" in text


def test_markdown_shows_failures_before_the_detail(tmp_path: Path) -> None:
    clean_dataset(tmp_path)
    write_table(
        tmp_path,
        "encounters",
        [
            {
                "Id": "e1",
                "START": "2020-01-01T10:00:00Z",
                "STOP": "2020-01-01T09:00:00Z",
                "PATIENT": "p1",
            }
        ],
    )
    report = build_report(tmp_path, generated_at=FIXED_TIME)

    text = render_markdown(report)

    assert "check(s) failed" in text
    assert text.index("## Findings") < text.index("## Detail by category")
    assert "`FAIL` — `temporal.start_le_stop.encounters`" in text
    assert "Samples:" in text


def test_markdown_of_warnings_without_failures(tmp_path: Path) -> None:
    clean_dataset(tmp_path)
    write_table(
        tmp_path,
        "supplies",
        [
            {"DATE": "2021-05-05", "PATIENT": "p1", "CODE": "abc"},
            {"DATE": "2021-05-05", "PATIENT": "p1", "CODE": "abc"},
        ],
    )
    report = build_report(tmp_path, generated_at=FIXED_TIME)

    text = render_markdown(report)

    assert report.counts_by_status[Status.FAIL.value] == 0
    assert report.counts_by_status[Status.WARNING.value] > 0
    assert "no check failed;" in text and "warning(s)" in text


def test_markdown_explains_skipped_checks(tmp_path: Path) -> None:
    write_table(tmp_path, "patients", [{"Id": "p1", "BIRTHDATE": "1980-01-01"}])
    report = build_report(tmp_path, generated_at=FIXED_TIME)

    text = render_markdown(report)

    assert report.counts_by_status[Status.SKIPPED.value] > 0
    assert "| `SKIPPED` |" in text
    assert "not present in this dataset" in text
    assert "| `NOT_APPLICABLE` |" in text


def test_markdown_shows_a_tool_error_as_a_finding() -> None:
    report = make_report(
        make_check(
            "f",
            Status.ERROR,
            Severity.HIGH,
            message="the tool could not complete this check",
        )
    )

    text = render_markdown(report)

    assert "1 check(s) could not be run" in text
    assert "`ERROR` — `f`" in text


def test_markdown_reports_unresolved_relations_as_pending_not_defects() -> None:
    report = make_report(unresolved_relations=unresolved_relations())

    text = render_markdown(report)

    assert "## Documented relations not applied" in text
    assert "not findings about this dataset and not dataset defects" in text
    assert "fk.claims_transactions.PATIENTINSURANCEID->payer_transitions.MEMBERID" in text
    assert "claims_transactions.FROMDATE <= TODATE" in text
    assert "Pending:" in text
    assert "Implemented in Synthea as:" in text
    assert "Reference evidence (" in text
    # and it is not counted as a failure
    assert "Verdict: no check failed and no check produced a warning." in text


def test_unresolved_relations_do_not_depend_on_the_reported_dataset(tmp_path: Path) -> None:
    """They describe the tool, not the dataset, so they are identical for any input."""
    other = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    assert other.unresolved_relations == unresolved_relations()


def test_a_report_for_another_dataset_never_claims_the_reference_numbers(
    tmp_path: Path,
) -> None:
    """Evidence measured on the reference sample must not read as this dataset's."""
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    text = render_markdown(report)
    head, separator, section = text.partition("## Documented relations not applied")

    assert separator, "the unresolved section must be present"
    assert "not on the dataset analysed in this report" in section
    assert f"`{report.data_dir}`" in section
    for entry in report.unresolved_relations:
        # the quantities and their origin appear only inside that section
        assert entry.reference_dataset not in head
        assert entry.reference_evidence not in head
        assert entry.reference_evidence in section
        assert f"Reference evidence (`{entry.reference_dataset}`)" in section
        assert "2026-08" in entry.reference_dataset


def test_json_attributes_the_reference_evidence_to_its_dataset(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    payload = json.loads(json_report.dumps(report))
    entries = payload["dataset"]["unresolved_relations"]

    assert entries
    for entry in entries:
        assert entry["reference_dataset"].strip()
        assert entry["reference_evidence"].strip()
        assert entry["implemented_as"].strip()
        assert "observed" not in entry


def test_markdown_of_an_incompatible_contract_is_explicit(tmp_path: Path) -> None:
    write_raw(
        tmp_path,
        "conditions",
        "START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION\n2000-01-01,,p1,e1,123,A\n",
    )
    report = build_report(tmp_path, generated_at=FIXED_TIME)

    text = render_markdown(report)

    assert "**INCOMPATIBLE**" in text
    assert "Deviations from the contract:" in text


def test_markdown_never_claims_the_dataset_version(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    text = render_markdown(report)

    assert "**COMPATIBLE**" in text
    assert "not proof that the dataset was produced by one exact" in text
    assert "observed" in text
    lowered = text.lower()
    assert "confirmed version" not in lowered
    assert "the dataset is synthea" not in lowered


def test_markdown_of_an_unknown_contract() -> None:
    """A report whose contract could not be assessed still has to render."""
    report = make_report(
        contract_status=ContractStatus.UNKNOWN,
        contract_summary="no known table was observed, so the contract cannot be assessed",
    )

    text = render_markdown(report)

    assert "**UNKNOWN**" in text
    assert "Tables observed: **0**" in text
    assert "No known table was found in this dataset." in text


def test_markdown_of_a_report_without_findings_or_checks() -> None:
    report = make_report()

    text = render_markdown(report)

    assert "No check failed, no check errored and no check produced a warning." in text
    assert "**total** | **0**" in text


def test_markdown_escapes_hostile_sample_content() -> None:
    """Only table cells must escape pipes; bullets must stay on one line.

    A value containing ``|`` would break a Markdown table, a newline would split a
    bullet, and a backtick inside inline code would break the code span.
    """
    report = make_report(
        make_check(
            "temporal.event_after_birth.conditions.START",
            Status.WARNING,
            Severity.MEDIUM,
            message="1 of 1 row(s) have START before the patient's birth date | see `notes`",
            metrics={"rows": 1, "note": "a|b"},
            samples=(
                {"row": 1, "values": {"START": "1900-01-01", "PATIENT": "p|1"}},
                {"row": 2, "values": {"START": "línea\nnueva", "PATIENT": "`p2`"}},
            ),
        )
    )

    text = render_markdown(report)
    lines = text.splitlines()

    # a pipe that lands in a table cell is escaped, and backticks are neutralised
    assert "\\| see 'notes'" in text
    assert "note=a\\|b" in text
    # a newline inside a value cannot split a bullet
    assert "línea nueva" in text
    assert not any(line.strip() == "nueva" for line in lines)
    # backticks inside a value are neutralised
    assert "`p2`" not in text
    assert "'p2'" in text
    # each sample stays a single bullet
    sample_lines = [line for line in lines if line.startswith("- ") and "PATIENT" in line]
    assert len(sample_lines) == 2


def test_markdown_bounds_the_list_of_passing_checks() -> None:
    """A category with many passes must not turn into a wall of identifiers."""
    report = make_report(
        *(
            make_check(f"nulls.table{i}")
            for i in range(30)
        )
    )

    text = render_markdown(report)

    assert "30 check(s) passed:" in text
    assert "`nulls.table11`" in text
    assert "`nulls.table12`" not in text
    assert "and 18 more" in text


def test_markdown_truncates_extremely_long_messages() -> None:
    report = make_report(
        make_check("dates.patients.BIRTHDATE", Status.FAIL, Severity.MEDIUM, message="x" * 5000)
    )

    text = render_markdown(report)

    assert "…" in text
    assert len(text) < 5000


def test_markdown_lists_the_table_inventory() -> None:
    report = make_report(
        tables=(
            TableSummary(
                name="patients",
                file_name="patients.csv",
                rows=108,
                columns=("Id", "BIRTHDATE"),
                schema_match=True,
            ),
            TableSummary(name="claims", file_name="claims.csv"),
        ),
        unknown_files=("extra.csv",),
    )

    text = render_markdown(report)

    assert "| `patients` | `patients.csv` | 108 | 2 | yes |" in text
    assert "| `claims` | `claims.csv` | unknown | 0 | unknown |" in text
    assert "CSV files that are not part of the contract: `extra.csv`." in text


# --------------------------------------------------------------------------- #
# rendering is decoupled from data access
# --------------------------------------------------------------------------- #


def test_renderers_never_read_a_csv_or_run_a_check(tmp_path: Path, monkeypatch) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    import synthea_quality.discovery as discovery
    import synthea_quality.loader as loader

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a renderer must not touch the dataset")

    monkeypatch.setattr(loader, "load_table", forbidden)
    monkeypatch.setattr(loader, "read_header", forbidden)
    monkeypatch.setattr(discovery, "discover_dataset", forbidden)

    assert render_markdown(report)
    assert json_report.dumps(report)


def test_markdown_module_has_no_data_dependency() -> None:
    from synthea_quality.reporting import markdown

    assert not hasattr(markdown, "load_table")
    assert not hasattr(markdown, "pd")
    assert not hasattr(markdown, "pandas")


def test_both_renderers_are_deterministic(tmp_path: Path) -> None:
    report = build_report(clean_dataset(tmp_path), generated_at=FIXED_TIME)

    assert render_markdown(report) == render_markdown(report)
    assert json_report.dumps(report) == json_report.dumps(report)


def test_sample_limit_is_honoured_end_to_end(tmp_path: Path) -> None:
    write_table(
        tmp_path,
        "patients",
        [{"Id": f"p{i}", "BIRTHDATE": "31/12/1980"} for i in range(20)],
    )
    report = build_report(tmp_path, sample_limit=3, generated_at=FIXED_TIME)
    failing = next(check for check in report.checks if check.status is Status.FAIL)

    assert failing.metrics["invalid"] == 20
    assert len(failing.samples) == 3
