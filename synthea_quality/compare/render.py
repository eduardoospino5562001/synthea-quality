"""[COMPARE] Markdown and JSON renderings of a comparison.

The Markdown is meant to be read top-down: scope and reference date of each side,
a visible note when the reference dates or the years of history differ, a summary
of how many figures were compared and how many changed, a table of the largest
changes, and one section per block with only the changed rows. The JSON carries
every figure (changed and unchanged) with both sides, their intervals, numerators
and denominators, and the provenance of each side. It states no verdict: there is
no pass/fail, no threshold and no significant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthea_quality import __version__
from synthea_quality.compare.compute import Comparison, largest_changes
from synthea_quality.compare.models import (
    CHANGED,
    COUNT,
    DAYS,
    ONLY_IN_BASELINE,
    ONLY_IN_CANDIDATE,
    PER_1000_PERSON_YEARS,
    PERCENT,
    PROPORTION,
    SKIPPED,
    UNCHANGED,
    Figure,
)
from synthea_quality.table_format import fixed, table_cell
from synthea_quality.validate.models import ModuleInfo, ModuleValidationReport

MARKDOWN_NAME = "synthea_compare.md"
JSON_NAME = "synthea_compare.json"
SCHEMA_VERSION = 1

#: The blocks of a comparison, in the order the report lists them.
BLOCKS = ("population", "conditions", "observations", "medications", "indexed")

#: The human-readable title of each block.
BLOCK_TITLES = {
    "population": "Population",
    "conditions": "Conditions",
    "observations": "Observations",
    "medications": "Medications",
    "indexed": "Indexed cohort",
}

#: Legend shown once under the "Largest changes" table. It tells how to read
#: the units of the figures (percentage points for proportions, per 1,000
#: person-years for rates, integers for counts and days); it states no verdict.
UNITS_LEGEND = "pp = percentage points; incidence per 1,000 person-years; counts and days as integers."


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Everything a rendering needs: the two sides and the paired figures.

    ``generated_at`` is set once for the whole comparison so the Markdown and the
    JSON are deterministic: the same two datasets and the same module always
    produce the same text, apart from this timestamp.
    """

    module_file: str
    module: ModuleInfo
    baseline_dir: str
    candidate_dir: str
    baseline_report: ModuleValidationReport
    candidate_report: ModuleValidationReport
    comparisons: list[Comparison]
    generated_at: str


def render_markdown(result: ComparisonResult, *, top: int = 20) -> str:
    """The comparison as Markdown, top to bottom, with no verdict words."""

    name = result.module.name or Path(result.module_file).stem
    blocks = [
        f"# Comparison: {name}",
        "",
        f"Compares two Synthea datasets: `{result.baseline_dir}` vs `{result.candidate_dir}`.",
        f"Module: `{result.module_file}`",
        f"Version: synthea-quality {__version__}",
        "",
        "## Reference date",
        "",
        _reference_line("Baseline", result.baseline_report),
        _reference_line("Candidate", result.candidate_report),
        "",
    ]
    note = _difference_note(result.baseline_report, result.candidate_report)
    if note:
        blocks += [f"> {note}", ""]

    compared = len(result.comparisons)
    changed = sum(1 for c in result.comparisons if c.state == CHANGED)
    unchanged = sum(1 for c in result.comparisons if c.state == UNCHANGED)
    skipped = sum(1 for c in result.comparisons if c.state == SKIPPED)
    only_baseline = sum(1 for c in result.comparisons if c.state == ONLY_IN_BASELINE)
    only_candidate = sum(1 for c in result.comparisons if c.state == ONLY_IN_CANDIDATE)

    blocks += [
        "## Summary",
        "",
        f"Compared {compared} figures: {changed} changed, {unchanged} unchanged.",
    ]
    if skipped:
        blocks.append(
            f"{skipped} could not be measured on at least one side (skipped, with a reason)."
        )
    if only_baseline or only_candidate:
        blocks.append(
            f"{only_baseline} only in the baseline, {only_candidate} only in the candidate."
        )
    blocks += [""]

    ranked = largest_changes(result.comparisons, top)
    blocks += [f"## Largest changes (top {top})", ""]
    if ranked:
        blocks += _largest_changes_table(ranked)
    else:
        blocks.append("No figures changed.")
    blocks += [""]
    blocks.append(UNITS_LEGEND)
    blocks += [""]

    for block in BLOCKS:
        block_comparisons = [c for c in result.comparisons if c.key.block == block]
        if not block_comparisons:
            continue
        blocks += [f"## {BLOCK_TITLES[block]}", ""]
        block_changed = [c for c in block_comparisons if c.state == CHANGED]
        block_unchanged = [c for c in block_comparisons if c.state == UNCHANGED]
        block_skipped = [c for c in block_comparisons if c.state == SKIPPED]
        blocks.append(
            f"{len(block_comparisons)} figures, {len(block_changed)} changed, "
            f"{len(block_unchanged)} unchanged."
        )
        if block_skipped:
            blocks.append(
                f"{len(block_skipped)} could not be measured (skipped, with a reason)."
            )
        blocks += [""]
        if block == "population":
            for c in block_skipped:
                if c.key.measure == c.key.element and c.reason:
                    blocks.append(f"Section {c.key.element} not measured: {c.reason}.")
                    blocks.append("")
        if block_changed:
            blocks += _block_table(block_changed)
        blocks += [""]

    return "\n".join(line.rstrip() for line in blocks) + "\n"


def _reference_line(label: str, report: ModuleValidationReport) -> str:
    reference = report.reference_date
    if reference is None:
        return f"- {label}: none — {report.reference_reason}"
    kind = "an approximation" if reference.approximate else "exact"
    return (
        f"- {label}: `{reference.value}` — {kind}, source "
        f"`{reference.source.value}`. {reference.detail}"
    )


def _difference_note(
    baseline: ModuleValidationReport, candidate: ModuleValidationReport
) -> str | None:
    """A visible note when the reference dates or the years of history differ."""
    parts: list[str] = []
    base_ref = baseline.reference_date
    cand_ref = candidate.reference_date
    base_date = base_ref.value if base_ref is not None else None
    cand_date = cand_ref.value if cand_ref is not None else None
    if base_date != cand_date:
        parts.append(
            f"the reference dates differ: baseline {base_date or 'none'}, "
            f"candidate {cand_date or 'none'}"
        )
    base_years = baseline.export_history.years
    cand_years = candidate.export_history.years
    if base_years != cand_years:
        parts.append(
            f"the years of history differ: baseline {base_years}, candidate {cand_years}"
        )
    if not parts:
        return None
    return "Note: " + "; ".join(parts) + ". This changes the alive patients and their ages."


def _largest_changes_table(ranked: list[Comparison]) -> list[str]:
    rows = []
    for c in ranked:
        rows.append(
            (
                c.key.block,
                c.key.element,
                c.key.measure,
                c.key.stratum,
                _format_figure(c.baseline),
                _format_figure(c.candidate),
                _format_difference(c.difference, c.unit),
                _format_relative(c.relative_difference),
            )
        )
    return _table(
        ("Block", "Element", "Measure", "Stratum", "Baseline", "Candidate", "Difference", "Relative"),
        rows,
        align=("", "", "", "", "r", "r", "r", "r"),
    )


def _block_table(changed: list[Comparison]) -> list[str]:
    rows = []
    for c in changed:
        rows.append(
            (
                c.key.element,
                c.key.measure,
                c.key.stratum,
                _format_figure(c.baseline),
                _format_figure(c.candidate),
                _format_difference(c.difference, c.unit),
                _format_relative(c.relative_difference),
            )
        )
    return _table(
        ("Element", "Measure", "Stratum", "Baseline", "Candidate", "Difference", "Relative"),
        rows,
        align=("", "", "", "r", "r", "r", "r"),
    )


def _table(
    header: tuple[str, ...], rows: list[tuple[Any, ...]], *, align: tuple[str, ...]
) -> list[str]:
    header_list = list(header)
    separators = ["---:" if a == "r" else "---" for a in align]
    lines = [
        "| " + " | ".join(table_cell(str(cell)) for cell in header_list) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    lines += [
        "| " + " | ".join(table_cell(str(cell)) for cell in row) + " |" for row in rows
    ]
    return lines


def _format_value(value: float | None, unit: str) -> str:
    """One figure's value in its unit, or an em dash when there is no value."""
    if value is None:
        return "—"
    if unit in (PROPORTION, PERCENT):
        return f"{fixed(value * 100, 2)}%"
    if unit == PER_1000_PERSON_YEARS:
        return fixed(value, 2)
    if unit == COUNT:
        return str(int(value))
    if unit == DAYS:
        return f"{int(value)} days"
    return str(value)


def _format_figure(figure: Figure | None) -> str:
    """A figure with its 95% interval when it has one."""
    if figure is None:
        return "—"
    text = _format_value(figure.value, figure.unit)
    if figure.ci95_low is not None and figure.ci95_high is not None:
        low = _format_value(figure.ci95_low, figure.unit)
        high = _format_value(figure.ci95_high, figure.unit)
        return f"{text} ({low}–{high})"
    return text


def _format_difference(difference: float | None, unit: str) -> str:
    """The signed difference, in the figure's unit (percentage points for proportions)."""
    if difference is None:
        return "—"
    if unit in (PROPORTION, PERCENT):
        return f"{fixed(difference * 100, 2, sign=True)} pp"
    if unit == PER_1000_PERSON_YEARS:
        return f"{fixed(difference, 2, sign=True)}"
    if unit == COUNT:
        return f"{int(difference):+d}"
    if unit == DAYS:
        return f"{int(difference):+d} days"
    return f"{difference:+g}"


def _format_relative(relative: float | None) -> str:
    """The relative difference as a percentage, or an em dash when it is None."""
    if relative is None:
        return "—"
    return f"{fixed(relative * 100, 2, sign=True)}%"


def render_json(result: ComparisonResult) -> str:
    """The comparison as JSON, schema_version 1, deterministic apart from generated_at."""

    data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": __version__,
        "generated_at": result.generated_at,
        "baseline": _side_provenance(result.baseline_report, result.baseline_dir, result.module_file),
        "candidate": _side_provenance(result.candidate_report, result.candidate_dir, result.module_file),
        "module": result.module.to_dict(),
        "compared": len(result.comparisons),
        "changed": sum(1 for c in result.comparisons if c.state == CHANGED),
        "unchanged": sum(1 for c in result.comparisons if c.state == UNCHANGED),
        "skipped": sum(1 for c in result.comparisons if c.state == SKIPPED),
        "only_in_baseline": sum(1 for c in result.comparisons if c.state == ONLY_IN_BASELINE),
        "only_in_candidate": sum(1 for c in result.comparisons if c.state == ONLY_IN_CANDIDATE),
        "comparisons": [c.to_dict() for c in result.comparisons],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _side_provenance(
    report: ModuleValidationReport, data_dir: str, module_file: str
) -> dict[str, Any]:
    reference = report.reference_date
    return {
        "data_dir": data_dir,
        "module_file": module_file,
        "reference_date": reference.to_dict() if reference is not None else None,
        "reference_reason": report.reference_reason,
        "years_of_history": report.export_history.years,
        "inputs": [item.to_dict() for item in report.inputs],
        "notes": list(report.notes),
    }


def write_markdown(result: ComparisonResult, path: str | Path, *, top: int = 20) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(result, top=top), encoding="utf-8")
    return target


def write_json(result: ComparisonResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_json(result), encoding="utf-8")
    return target
