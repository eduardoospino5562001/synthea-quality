"""[COMPARE] Compact GitHub and console presentations share one prepared view.

The view keeps notices and unmeasured figures visible while GitHub folds the
provenance and block detail. Formatting never recomputes comparisons. Data is
escaped only at the presentation boundary; JSON retains every original value.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from synthea_quality import __version__
from synthea_quality.compare.compute import Comparison, largest_changes
from synthea_quality.compare.models import (
    CHANGED,
    COUNT,
    DAYS,
    NOT_COMPARED,
    ONLY_IN_BASELINE,
    ONLY_IN_CANDIDATE,
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

#: The states of a figure that was not compared: listed with why in each block.
NOT_COMPARED_STATES = (SKIPPED, ONLY_IN_BASELINE, ONLY_IN_CANDIDATE)

#: Legend shown once under the "Largest changes" table. It tells how to read
#: the units of the figures (percentage points for proportions, per 1,000
#: person-years for rates, 2 decimals for ages, observation values and plain
#: numbers, integers for counts and days); it states no verdict.
UNITS_LEGEND = (
    "pp = percentage points; incidence per 1,000 person-years; ages, observation values and plain numbers "
    "with 2 decimals; counts and days as integers."
)

#: Scope is explicit even when every extracted figure is unchanged.
SCOPE_LIMIT = "Only these figures are compared, not every row of the CSV files."
LARGEST_HEADER = ("Block", "Element", "Measure", "Stratum", "Baseline", "Candidate", "Difference", "Relative")
NOT_COMPARED_HEADER = ("Element", "Measure", "Stratum", "Baseline", "Candidate", "Why not compared")


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


def summary_counts(comparisons: Sequence[Comparison]) -> str:
    """How many figures there are and what happened to each, in one line.

    Only the figures measured on both sides are "compared" (split into changed
    and unchanged); the skipped figures and those found on one side only are
    counted apart, never inside "compared", so the line never calls compared a
    figure that was not. The five counts always appear, zeros included, so the
    wording is the same in the summary, in every block and in the console.

    Example: ``"69 figures: 47 compared (0 changed, 47 unchanged), 6 skipped,
    16 only in the baseline, 0 only in the candidate"``; an empty list gives
    ``"0 figures: 0 compared (0 changed, 0 unchanged), 0 skipped, 0 only in
    the baseline, 0 only in the candidate"``. No trailing period: the caller
    adds it. Raises nothing.
    """

    changed = sum(1 for c in comparisons if c.state == CHANGED)
    unchanged = sum(1 for c in comparisons if c.state == UNCHANGED)
    skipped = sum(1 for c in comparisons if c.state == SKIPPED)
    only_baseline = sum(1 for c in comparisons if c.state == ONLY_IN_BASELINE)
    only_candidate = sum(1 for c in comparisons if c.state == ONLY_IN_CANDIDATE)
    return (
        f"{len(comparisons)} figures: {changed + unchanged} compared ({changed} changed, {unchanged} unchanged), "
        f"{skipped} skipped, {only_baseline} only in the baseline, {only_candidate} only in the candidate"
    )


@dataclass(frozen=True, slots=True)
class BlockView:
    """Prepared block detail keeps both presentations on the same counts."""

    title: str
    counts: str
    changed: list[tuple[str, ...]]
    notices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComparisonView:
    """Presentation-ready content, prepared once for both output formats."""

    name: str
    headline: str
    notices: tuple[str, ...]
    references: tuple[str, ...]
    shared_notes: tuple[str, ...]
    not_compared: list[tuple[str, ...]]
    largest: list[tuple[str, ...]]
    blocks: tuple[BlockView, ...]
    counts: str
    top: int
    baseline_dir: str
    candidate_dir: str
    module_file: str


def _headline(comparisons: Sequence[Comparison]) -> str:
    total = len(comparisons)
    changed = sum(c.state == CHANGED for c in comparisons)
    compared = sum(c.state in (CHANGED, UNCHANGED) for c in comparisons)
    missing = total - compared
    if not total:
        return "No figures to compare."
    if not compared:
        return f"No figures could be compared: {total} of {total} figures not compared."
    if changed:
        first = f"{changed} of {total} figures changed."
    elif missing:
        return f"No compared figure changed; {missing} of {total} figures could not be compared. {SCOPE_LIMIT}"
    else:
        first = "No figures changed."
    return f"{first} {compared} of {total} figures compared, {missing} not compared. {SCOPE_LIMIT}"


def build_view(result: ComparisonResult, *, top: int = 20) -> ComparisonView:
    """Prepare all selection and wording once, leaving renderers only layout.

    Shared notes retain baseline order; side-specific notes remain visible and
    labelled. Missing or approximate reference dates are notices even when equal.
    """
    baseline, candidate = result.baseline_report, result.candidate_report
    notices = []
    difference = _difference_note(baseline, candidate)
    if difference:
        notices.append(difference)
    references = tuple(_reference_line(label, report) for label, report in (
        ("Baseline", baseline), ("Candidate", candidate)
    ))
    for reference, report in zip(references, (baseline, candidate), strict=True):
        if report.reference_date is None or report.reference_date.approximate:
            notices.append(reference)
    shared = tuple(note for note in baseline.notes if note in candidate.notes)
    for label, report in (("Baseline", baseline), ("Candidate", candidate)):
        notices.extend(f"{label}: {note}" for note in report.notes if note not in shared)
    blocks = []
    for block in BLOCKS:
        rows = [c for c in result.comparisons if c.key.block == block]
        if not rows:
            continue
        section_notices = tuple(
            f"Section {c.key.element} not measured: {c.reason}." for c in rows
            if block == "population" and c.state == SKIPPED and c.key.measure == c.key.element and c.reason
        )
        blocks.append(BlockView(
            BLOCK_TITLES[block], summary_counts(rows),
            [row[1:] for row in _change_rows([c for c in rows if c.state == CHANGED])],
            section_notices,
        ))
    return ComparisonView(
        result.module.name or Path(result.module_file).stem, _headline(result.comparisons),
        tuple(notices), references, shared,
        _not_compared_rows([c for c in result.comparisons if c.state in NOT_COMPARED_STATES]),
        _change_rows(largest_changes(result.comparisons, top)), tuple(blocks),
        summary_counts(result.comparisons), top, result.baseline_dir, result.candidate_dir, result.module_file,
    )


def _single_line(text: str) -> str:
    return "".join(" " if unicodedata.category(char).startswith("C") or char.isspace() else char for char in text)


def _markdown_text(text: str) -> str:
    return table_cell(escape(_single_line(text), quote=False).replace("\\", "\\\\"))


def _console_text(text: str) -> str:
    return _single_line(text)


def render_markdown(result: ComparisonResult | ComparisonView, *, top: int = 20) -> str:
    """Fold context, keeping notices and all unmeasured figures visible."""
    view = build_view(result, top=top) if isinstance(result, ComparisonResult) else result
    first, separator, rest = view.headline.partition(". ")
    headline = f"**{first}.** {rest}" if separator else f"**{first}**"
    lines = [f"# Comparison: {_markdown_text(view.name)}", "", headline, ""]
    if view.notices:
        lines += ["## Notes", ""]
        lines += [f"> {_markdown_text(note)}" for note in view.notices] + [""]
    if view.not_compared:
        lines += ["## Not compared", ""] + _table(
            NOT_COMPARED_HEADER, view.not_compared, align=("", "", "", "r", "r", "")
        ) + [""]
    if view.largest:
        lines += [f"## Largest changes (top {view.top})", ""] + _table(
            LARGEST_HEADER, view.largest, align=("", "", "", "", "r", "r", "r", "r")
        ) + ["", UNITS_LEGEND, ""]
    lines += ["<details>",
              f"<summary>Details: datasets, reference dates, {len(view.shared_notes)} shared notes, "
              "figures by block and parts not compared</summary>", "",
              f"Compares two Synthea datasets: {_markdown_text(view.baseline_dir)} "
              f"vs {_markdown_text(view.candidate_dir)}.",
              f"Module: {_markdown_text(view.module_file)}", f"Version: synthea-quality {__version__}", "",
              "## Reference date", ""]
    lines += [f"- {_markdown_text(reference)}" for reference in view.references] + [""]
    if view.shared_notes:
        lines += ["## Shared notes", ""] + [f"- {_markdown_text(note)}" for note in view.shared_notes] + [""]
    lines += ["## Summary", "", f"{view.counts}.", ""]
    for block in view.blocks:
        lines += [f"## {block.title}", "", f"{block.counts}.", ""]
        for notice in block.notices:
            lines += [_markdown_text(notice), ""]
        if block.changed:
            lines += _table(LARGEST_HEADER[1:], block.changed, align=("", "", "", "r", "r", "r", "r")) + [""]
    lines += ["## Parts not compared", "",
              "These parts of the module validation report are not extracted; "
              "they stay in the report of each side:", ""]
    lines += [f"- {part}" for part in NOT_COMPARED] + ["", "</details>"]
    return "\n".join(line.rstrip() for line in lines) + "\n"


def _console_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    prepared = [tuple(_console_text(cell) for cell in row) for row in [header, *rows]]
    widths = [max(len(row[index]) for row in prepared) for index in range(len(header))]
    return ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip() for row in prepared]


def render_console(
    result: ComparisonResult | ComparisonView, markdown_path: Path | None = None,
    json_path: Path | None = None, *, top: int = 20,
) -> str:
    """Present the shared view in plain, aligned text for terminal readers."""
    view = build_view(result, top=top) if isinstance(result, ComparisonResult) else result
    lines = [f"synthea-compare (synthea-quality {__version__})",
             f"Comparison: {_console_text(view.name)}", "", view.headline, ""]
    if view.notices:
        lines += ["Notes:"] + [_console_text(note) for note in view.notices] + [""]
    if view.not_compared:
        lines += ["Not compared:"] + _console_table(NOT_COMPARED_HEADER, view.not_compared) + [""]
    if view.largest:
        lines += [f"Largest changes (top {view.top}):"]
        lines += _console_table(LARGEST_HEADER, view.largest) + ["", UNITS_LEGEND, ""]
    if view.largest or view.not_compared:
        lines += [f"{block.title}:  {block.counts}" for block in view.blocks] + [""]
    lines += [f"Baseline:   {_console_text(view.baseline_dir)}", f"Candidate:  {_console_text(view.candidate_dir)}",
              f"Module:     {_console_text(view.module_file)}"]
    if markdown_path is not None or json_path is not None:
        lines += ["", "Reports:"]
        lines += [f"  {_console_text(str(path))}" for path in (markdown_path, json_path) if path is not None]
    return "\n".join(lines) + "\n"


def _reference_line(label: str, report: ModuleValidationReport) -> str:
    reference = report.reference_date
    if reference is None:
        return f"{label}: none — {report.reference_reason}"
    kind = "an approximation" if reference.approximate else "exact"
    return (
        f"{label}: {reference.value} — {kind}, source "
        f"{reference.source.value}. {reference.detail}"
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
            f"the reference dates differ (baseline {base_date or 'none'}, "
            f"candidate {cand_date or 'none'}): this changes who is alive and their ages"
        )
    base_years = baseline.export_history.years
    cand_years = candidate.export_history.years
    if base_years != cand_years:
        # The exporter's years_of_history filters records, not patients (README,
        # "The exported history"): it never changes who is alive.
        parts.append(
            f"the years of history differ (baseline {base_years}, candidate {cand_years}): this changes which "
            "records were exported (lifetime prevalence, ever use and prior cases look back)"
        )
    if not parts:
        return None
    return "Note: " + "; ".join(parts) + "."


def _change_rows(ranked: list[Comparison]) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
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
    return rows


def _not_compared_rows(rows: list[Comparison]) -> list[tuple[str, ...]]:
    """The figures of a block that were not compared, each with why: the reason
    of the side that could not measure it, or the one side it was found on."""
    table_rows: list[tuple[str, ...]] = []
    for c in rows:
        if c.state == SKIPPED:
            why = c.reason or "skipped"
        elif c.state == ONLY_IN_BASELINE:
            why = "only in the baseline"
        else:
            why = "only in the candidate"
        table_rows.append(
            (
                c.key.element,
                c.key.measure,
                c.key.stratum,
                _format_figure(c.baseline),
                _format_figure(c.candidate),
                why,
            )
        )
    return table_rows


def _table(
    header: tuple[str, ...], rows: list[tuple[str, ...]], *, align: tuple[str, ...]
) -> list[str]:
    header_list = list(header)
    separators = ["---:" if a == "r" else "---" for a in align]
    lines = [
        "| " + " | ".join(_markdown_text(cell) for cell in header_list) + " |",
        "| " + " | ".join(separators) + " |",
    ]
    lines += [
        "| " + " | ".join(_markdown_text(cell) for cell in row) + " |" for row in rows
    ]
    return lines


def _format_value(value: float | None, unit: str) -> str:
    """One figure's value in its unit, or an em dash when there is no value.

    A proportion is shown as a percentage; a ``percent`` already is one, so it
    is not multiplied again. Counts and days are whole numbers. Everything else
    (rates per 1,000 person-years, ages in years, plain numbers and the units
    of an observation such as ``mm[Hg]``) has 2 fixed decimals, never a
    truncated integer.
    """
    if value is None:
        return "—"
    if unit == PROPORTION:
        return f"{fixed(value * 100, 2)}%"
    if unit == PERCENT:
        return f"{fixed(value, 2)}%"
    if unit == COUNT:
        return str(int(value))
    if unit == DAYS:
        return f"{int(value)} days"
    return fixed(value, 2)


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
    """The signed difference, in the figure's unit: percentage points for
    proportions and percents, whole numbers for counts and days, 2 fixed
    decimals for everything else (as in :func:`_format_value`)."""
    if difference is None:
        return "—"
    if unit == PROPORTION:
        return f"{fixed(difference * 100, 2, sign=True)} pp"
    if unit == PERCENT:
        return f"{fixed(difference, 2, sign=True)} pp"
    if unit == COUNT:
        return f"{int(difference):+d}"
    if unit == DAYS:
        return f"{int(difference):+d} days"
    return fixed(difference, 2, sign=True)


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
        "not_compared": list(NOT_COMPARED),
        "figures": len(result.comparisons),
        "compared": sum(1 for c in result.comparisons if c.state in (CHANGED, UNCHANGED)),
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


def write_markdown(result: ComparisonResult | ComparisonView, path: str | Path, *, top: int = 20) -> Path:
    """Write GitHub output, accepting an already prepared view for CLI reuse."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(result, top=top), encoding="utf-8")
    return target


def write_json(result: ComparisonResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_json(result), encoding="utf-8")
    return target
