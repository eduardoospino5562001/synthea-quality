"""[REPORTING] Markdown rendering of a report, written for a human reader.

The renderer works only with a :class:`~synthea_quality.models.DatasetReport`: it
reads no CSV, runs no check and imports no data library. Everything it prints comes
from the structured result it is given, so the same JSON always renders to the same
Markdown.

The order is chosen for someone opening the file to understand the state of a
dataset in a few seconds: the scope and the contract verdict first, then the summary,
then the findings that need attention (FAILs before WARNINGs, never buried under a
wall of passes), then the detail per category, then what was deliberately not
checked. Passing checks are summarised by identifier, because a reader looking for
problems does not need fifty lines saying nothing happened; every metric and sample
of every check stays available in the JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from synthea_quality.models import (
    CHECK_CATEGORIES,
    DatasetReport,
    Severity,
    Status,
)

#: How long a message or a metric value may be inside a table cell.
CELL_LIMIT = 220
#: How many passing check identifiers are listed before summarising the rest.
PASS_LIST_LIMIT = 12
#: How long one sample bullet may be before it is truncated.
SAMPLE_LIMIT = 500

_CATEGORY_TITLES = {
    "schema": "Schema",
    "primary_keys": "Primary keys",
    "foreign_keys": "Foreign keys",
    "data_quality": "Data quality",
    "temporal": "Temporal",
    "other": "Other",
}

_LEGEND = (
    "| Status | Meaning |",
    "| --- | --- |",
    "| `PASS` | the dataset satisfies the check |",
    "| `WARNING` | worth a human look, but legitimate Synthea data |",
    "| `FAIL` | a deterministic violation of a confirmed rule |",
    "| `NOT_APPLICABLE` | there was nothing to check in this dataset |",
    "| `SKIPPED` | the check could not be run; the reason is in the message |",
    "| `ERROR` | the tool itself could not complete the check, never a data defect |",
)


def render_markdown(report: DatasetReport) -> str:
    """Render ``report`` as a Markdown document."""
    sections: list[str] = [
        _header(report),
        _scope_section(report),
        _summary_section(report),
        _findings_section(report),
        _load_errors_section(report),
        _detail_section(report),
        _unresolved_section(report),
        _inventory_section(report),
        _footer(report),
    ]
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


def write_markdown(report: DatasetReport, path: str | Path) -> Path:
    """Write the Markdown rendering of ``report`` to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(report), encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def _header(report: DatasetReport) -> str:
    fails = report.counts_by_status[Status.FAIL.value]
    errors = report.counts_by_status[Status.ERROR.value]
    warnings = report.counts_by_status[Status.WARNING.value]

    if fails or errors:
        parts = []
        if fails:
            parts.append(f"**{fails} check(s) failed**")
        if errors:
            parts.append(f"**{errors} check(s) could not be run**")
        if warnings:
            parts.append(f"{warnings} warning(s)")
        verdict = "Verdict: " + ", ".join(parts) + "."
    elif warnings:
        verdict = (
            f"Verdict: no check failed; **{warnings} warning(s)** are worth a look."
        )
    else:
        verdict = "Verdict: no check failed and no check produced a warning."

    return "\n".join(
        [
            "# Synthea dataset quality report",
            "",
            verdict,
            "",
            "| | |",
            "| --- | --- |",
            f"| Dataset | `{_cell(report.data_dir)}` |",
            f"| Generated (UTC) | `{_cell(report.generated_at)}` |",
            f"| Tool version | `{_cell(report.tool_version)}` |",
            f"| Report schema version | `{report.schema_version}` |",
            f"| Schema contract | {_contract_name(report)} |",
        ]
    )


def _scope_section(report: DatasetReport) -> str:
    observed = len(report.tables)
    missing = report.missing_known_tables
    unknown = report.unknown_files

    lines = ["## Scope and schema compatibility", ""]
    lines.append(f"- Tables observed: **{observed}** of the {report.contract_tables} the contract describes.")
    lines.append(f"- Contract result: **{_contract_status(report)}**")
    if report.contract_summary:
        lines.append(f"  - {_cell(report.contract_summary, limit=400)}")
    lines.append(
        "- Missing tables: " + (", ".join(f"`{_cell(name)}`" for name in missing) if missing else "none")
    )
    lines.append(
        "- Unknown CSV files: "
        + (", ".join(f"`{_cell(name)}`" for name in unknown) if unknown else "none")
    )
    if report.anomalous_entries:
        lines.append("- Names that could not be read as a table:")
        lines.extend(
            f"  - {_cell(entry.reason, limit=400)}" for entry in report.anomalous_entries
        )
    if report.contract_findings:
        lines.append("- Deviations from the contract:")
        lines.extend(
            f"  - {_cell(finding, limit=400)}" for finding in report.contract_findings
        )
    lines.append("")
    lines.append(
        "A `COMPATIBLE` result means every **observed** table matches this contract. It is "
        "evidence of consistency, not proof that the dataset was produced by one exact "
        "Synthea version: tables the dataset does not contain were never compared."
    )
    return "\n".join(lines)


def _summary_section(report: DatasetReport) -> str:
    counts = report.counts_by_status
    lines = [
        "## Summary",
        "",
        "| Status | Checks |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{status.value}` | {counts[status.value]} |" for status in Status
    )
    lines.append(f"| **total** | **{len(report.checks)}** |")
    lines.append("")

    severity = report.findings_by_severity
    lines.append(
        "Findings by severity (FAIL, ERROR and WARNING only): "
        + ", ".join(f"`{level.value}` {severity[level.value]}" for level in Severity)
        + "."
    )
    lines.append("")

    by_category = report.counts_by_category
    lines.append(
        "Checks by category: "
        + ", ".join(
            f"`{name}` {by_category[name]}" for name in CHECK_CATEGORIES if by_category[name]
        )
        + ("." if any(by_category.values()) else "none.")
    )
    return "\n".join(lines)


def _findings_section(report: DatasetReport) -> str:
    findings = report.findings
    if not findings:
        return "\n".join(
            [
                "## Findings",
                "",
                "No check failed, no check errored and no check produced a warning.",
            ]
        )

    lines = ["## Findings", ""]
    for check in findings:
        lines.append(f"### `{check.status.value}` — `{_cell(check.check_id)}` ({check.severity.value})")
        lines.append("")
        lines.append(_cell(check.message, limit=500))
        lines.append("")
        if check.metrics:
            lines.append(f"Metrics: {_metrics_text(check.metrics)}")
        if check.samples:
            lines.append("")
            lines.append("Samples:")
            lines.extend(f"- {_sample_text(sample)}" for sample in check.samples)
        lines.append("")
    return "\n".join(lines).rstrip()


def _load_errors_section(report: DatasetReport) -> str:
    if not report.load_errors:
        return ""
    lines = [
        "## Load errors",
        "",
        "These tables could not be read at all, so no check could produce a verdict for "
        "them. A table whose header reads but whose content is malformed is reported by "
        "the checks that had to skip it, further down.",
        "",
        "| Table | Reason |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{_cell(error.table)}` | {_cell(error.reason, limit=400)} |"
        for error in report.load_errors
    )
    return "\n".join(lines)


def _detail_section(report: DatasetReport) -> str:
    lines = ["## Detail by category", ""]
    grouped = report.checks_by_category
    for name in CHECK_CATEGORIES:
        checks = grouped[name]
        if not checks:
            if name == "schema":
                lines.append(
                    "### Schema\n\nNo schema rule is expressed as a check: the contract "
                    "comparison is reported under *Scope and schema compatibility* above."
                )
                lines.append("")
            continue

        lines.append(f"### {_CATEGORY_TITLES[name]} ({len(checks)} check(s))")
        lines.append("")
        noteworthy = [check for check in checks if check.status is not Status.PASS]
        passed = [check for check in checks if check.status is Status.PASS]

        if noteworthy:
            lines.append("| Check | Status | Severity | Message |")
            lines.append("| --- | --- | --- | --- |")
            lines.extend(
                f"| `{_cell(check.check_id)}` | `{check.status.value}` | {check.severity.value} | "
                f"{_cell(check.message)} |"
                for check in noteworthy
            )
            lines.append("")

        if passed:
            listed = ", ".join(f"`{check.check_id}`" for check in passed[:PASS_LIST_LIMIT])
            remaining = len(passed) - PASS_LIST_LIMIT
            tail = (
                f" and {remaining} more" if remaining > 0 else ""
            )
            lines.append(
                f"{len(passed)} check(s) passed: {listed}{tail}. Full metrics for every check "
                "are in the JSON report."
            )
            lines.append("")
    return "\n".join(lines).rstrip()


def _unresolved_section(report: DatasetReport) -> str:
    if not report.unresolved_relations:
        return ""
    lines = [
        "## Documented relations not applied",
        "",
        "Synthea's documentation describes these relationships, but the tool does not "
        "apply them as constraints yet. **They are not findings about this dataset and "
        "not dataset defects**: they are pending a maintainer's confirmation of how the "
        "data is meant to be read.",
        "",
        "The reference evidence quoted below was measured on the reference dataset named "
        f"in each entry, **not on the dataset analysed in this report** "
        f"(`{_cell(report.data_dir)}`).",
        "",
    ]
    for relation in report.unresolved_relations:
        lines.append(f"### `{_cell(relation.relation)}` ({relation.kind})")
        lines.append("")
        lines.append(f"- Documented as: {_cell(relation.documented_as, limit=500)}")
        lines.append(f"- Implemented in Synthea as: {_cell(relation.implemented_as, limit=500)}")
        lines.append(
            f"- Reference evidence (`{_cell(relation.reference_dataset, limit=200)}`): "
            f"{_cell(relation.reference_evidence, limit=500)}"
        )
        lines.append(f"- Why it is not enforced: {_cell(relation.why_not_enforced, limit=500)}")
        lines.append(f"- Pending: {_cell(relation.pending, limit=500)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _inventory_section(report: DatasetReport) -> str:
    if not report.tables:
        return "## Tables\n\nNo known table was found in this dataset."
    lines = [
        "## Tables",
        "",
        "| Table | File | Rows | Columns | Matches contract |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for table in report.tables:
        rows = "unknown" if table.rows is None else f"{table.rows:,}"
        match = {True: "yes", False: "**no**", None: "unknown"}[table.schema_match]
        lines.append(
            f"| `{_cell(table.name)}` | `{_cell(table.file_name)}` | {rows} | "
            f"{len(table.columns)} | {match} |"
        )
    if report.unknown_files:
        lines.append("")
        lines.append(
            "CSV files that are not part of the contract: "
            + ", ".join(f"`{_cell(name)}`" for name in report.unknown_files)
            + "."
        )
    return "\n".join(lines)


def _footer(report: DatasetReport) -> str:
    return "\n".join(
        [
            "## How this report was produced",
            "",
            f"- Tool `{_cell(report.tool_version)}`, report schema `{report.schema_version}`, "
            f"contract {_contract_name(report)}.",
            f"- Generated at `{_cell(report.generated_at)}` for `{_cell(report.data_dir)}`.",
            "- The renderer used only the structured results in this report: it read no CSV "
            "and ran no check, so the same JSON always renders to the same Markdown.",
            "",
            "### Reading this report",
            "",
            *_LEGEND,
        ]
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _contract_name(report: DatasetReport) -> str:
    return f"`{_cell(report.schema_contract)}`" if report.schema_contract else "_none_"


def _contract_status(report: DatasetReport) -> str:
    return report.contract_status.value if report.contract_status else "UNKNOWN"


def _cell(text: Any, *, limit: int = CELL_LIMIT) -> str:
    """One safe table cell: no pipes, no newlines, no runaway length."""
    value = "" if text is None else str(text)
    value = value.replace("|", "\\|").replace("`", "'")
    value = " ".join(value.split())
    if len(value) > limit:
        value = value[: limit - 1] + "…"
    return value


def _value(text: Any, *, limit: int = 120) -> str:
    if text is None:
        return "null"
    value = str(text).replace("`", "'")
    value = " ".join(value.split())
    if len(value) > limit:
        value = value[: limit - 1] + "…"
    return f"`{value}`"


def _metrics_text(metrics: Mapping[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, (list, dict, tuple)):
            parts.append(f"{key}={json.dumps(value, ensure_ascii=False, default=str)}")
        else:
            parts.append(f"{key}={value}")
    return _cell(", ".join(parts), limit=600)


def _sample_text(sample: Mapping[str, Any]) -> str:
    """Render one sample as ``key=value`` pairs, flattening a nested mapping."""
    parts: list[str] = []
    for key, value in sample.items():
        if isinstance(value, Mapping):
            inner = ", ".join(f"{name}={_value(item)}" for name, item in value.items())
            parts.append(f"{key}: {inner}")
        elif isinstance(value, Sequence) and not isinstance(value, str):
            parts.append(f"{key}=[{' '.join(_value(item) for item in value)}]")
        else:
            parts.append(f"{key}={_value(value)}")
    return _truncate(" ".join(parts), SAMPLE_LIMIT)


def _truncate(text: str, limit: int) -> str:
    """Shorten a line without touching its formatting."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["render_markdown", "write_markdown"]
