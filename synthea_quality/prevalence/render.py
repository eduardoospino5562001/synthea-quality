"""[PREVALENCE] JSON and Markdown renderings of a prevalence report.

Both read only the :class:`~synthea_quality.prevalence.models.PrevalenceReport`: no CSV,
no computation, no data library. The Markdown puts the definitions right after the
reference date, because a prevalence without its numerator, denominator and date is not
interpretable; then each condition asked for (totals, expected values, strata, notes);
then the general table. A value given with ``--expected`` is described as inside or
outside the observed rate's 95% CI — never as a pass or a match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from synthea_quality.prevalence.models import (
    ConditionResult,
    GeneralTable,
    PrevalenceReport,
    Rate,
    Stratum,
)
from synthea_quality.profile.models import InputState, SectionStatus

MARKDOWN_NAME = "synthea_prevalence.md"
JSON_NAME = "synthea_prevalence.json"

_DIMENSIONS = (
    ("age_band", "By age band at the reference date", "Age band"),
    ("GENDER", "By sex (`GENDER`)", "`GENDER`"),
)


def dumps(report: PrevalenceReport, *, indent: int | None = 2) -> str:
    """Serialise ``report`` as JSON text, ending with a newline."""
    return report.to_json(indent=indent) + "\n"


def write_json(report: PrevalenceReport, path: str | Path) -> Path:
    return _write(path, dumps(report))


def write_markdown(report: PrevalenceReport, path: str | Path) -> Path:
    return _write(path, render_markdown(report))


def render_markdown(report: PrevalenceReport) -> str:
    """Render ``report`` as a Markdown document."""
    blocks = [
        "# Synthea prevalence report",
        "This report **describes** how many of the patients alive at the end of the "
        "simulation have each condition. It states no verdict: a value given as expected is "
        "shown next to the observed one, inside or outside its 95% confidence interval, and "
        "nothing more. Incidence is not computed here.",
        _scope(report),
        _reference(report),
        _definitions(),
        *_conditions(report),
        _general(report.general, report.alive),
    ]
    return "\n\n".join(block.rstrip() for block in blocks) + "\n"


def _scope(report: PrevalenceReport) -> str:
    lines = [
        "## Scope",
        "",
        f"- Dataset: `{report.data_dir}`",
        f"- Generated: {report.generated_at} by synthea-quality {report.tool_version}",
        f"- Alive patients (the denominator): "
        f"**{report.alive if report.alive is not None else '—'}**",
        "- Tables:",
    ]
    for item in report.inputs:
        if item.state is InputState.READ:
            lines.append(f"  - `{item.table}`: {item.rows} rows, for {item.used_for}")
        else:
            lines.append(f"  - `{item.table}`: **{item.state.value}** — {item.reason}")
    if report.incomplete:
        lines += ["", "**This report is incomplete:** a table it needs could not be read."]
    lines += [f"- Notes: {note}" for note in report.notes]
    return "\n".join(lines)


def _reference(report: PrevalenceReport) -> str:
    reference = report.reference_date
    if reference is None:
        return f"## Reference date\n\n`SKIPPED` — {report.reference_reason}"
    kind = "an **approximation**" if reference.approximate else "exact"
    return (
        f"## Reference date\n\n**{reference.value}** — {kind}, source "
        f"`{reference.source.value}`.\n\n> {reference.detail}"
    )


def _definitions() -> str:
    return "\n".join(
        [
            "## Definitions",
            "",
            "- **Point prevalence**: alive patients with a record whose `START` is on or "
            "before the reference date and whose `STOP` is empty or after it, over the alive "
            "patients.",
            "- **Lifetime prevalence**: alive patients with a record whose `START` is on or "
            "before the reference date, over the alive patients.",
            "- A patient counts once per condition, whichever of its codes they have. Strata "
            "use the alive patients of the stratum as denominator.",
            "- **95% CI**: Wilson score interval.",
        ]
    )


def _conditions(report: PrevalenceReport) -> list[str]:
    if not report.conditions:
        return [
            "## Conditions asked for\n\nNone. Use `--condition NAME=CODE[,CODE...]` or "
            "`--conditions FILE.json` to measure specific conditions."
        ]
    return [_condition(condition) for condition in report.conditions]


def _condition(condition: ConditionResult) -> str:
    codes = ", ".join(
        f"`{ref.code}`" if ref.system is None else f"`{ref.system}|{ref.code}`"
        for ref in condition.codes
    )
    title = f"{condition.name} (declared acute)" if condition.acute else condition.name
    head = f"## Condition: {title}\n\nCodes: {codes}"
    if condition.status is SectionStatus.SKIPPED:
        return f"{head}\n\n`SKIPPED` — {condition.reason}"
    assert condition.point is not None and condition.lifetime is not None
    blocks = [
        head,
        "\n".join(
            _table(
                ("Measure", "Patients", "Alive", "Prevalence", "95% CI"),
                [
                    ("Point", *_rate_cells(condition.point)),
                    ("Lifetime", *_rate_cells(condition.lifetime)),
                ],
                align=("", "r", "r", "r", "r"),
            )
        ),
    ]
    if condition.expected:
        rows = []
        for item in condition.expected:
            position = (
                f"{item.position} the 95% CI" if item.position is not None else "no interval"
            )
            difference = (
                "—" if item.difference is None else f"{item.difference * 100:+.2f} pp"
            )
            rows.append(
                (
                    item.expected.measure.capitalize(),
                    _pct(item.expected.value),
                    _pct(item.observed.value),
                    difference,
                    position,
                    item.expected.source or "—",
                )
            )
        blocks.append(
            "Reference values (shown for comparison, not as a verdict):\n\n"
            + "\n".join(
                _table(
                    ("Measure", "Reference", "Observed", "Difference", "Position", "Source"),
                    rows,
                    align=("", "r", "r", "r", "", ""),
                )
            )
        )
    for dimension, title, label in _DIMENSIONS:
        strata = [s for s in condition.strata if s.dimension == dimension]
        if strata:
            blocks.append(f"### {title}\n\n" + "\n".join(_strata_table(strata, label)))
    by_code = condition.metrics.get("records_by_code", {})
    records = condition.metrics.get("records", 0)
    without_stop = condition.metrics.get("records_without_stop", 0)
    blocks.append(
        f"Records of alive patients used: **{records}**, of which **{without_stop}** have no "
        f"`STOP` (they count as active at the reference date). By code: "
        + ", ".join(f"`{code}` {count}" for code, count in by_code.items())
        + "."
    )
    if condition.notes:
        blocks.append("\n".join(f"- {note}" for note in condition.notes))
    return "\n\n".join(blocks)


def _strata_table(strata: list[Stratum], label: str) -> list[str]:
    rows = [
        (
            "*(empty)*" if s.value is None else s.value,
            s.point.denominator,
            _count(s.point),
            _ci(s.point),
            _count(s.lifetime),
            _ci(s.lifetime),
        )
        for s in strata
    ]
    return _table(
        (label, "Alive", "Point", "Point 95% CI", "Lifetime", "Lifetime 95% CI"),
        rows,
        align=("", "r", "r", "r", "r", "r"),
    )


def _general(table: GeneralTable, alive: int | None) -> str:
    title = "## All conditions by point prevalence"
    if table.status is SectionStatus.SKIPPED:
        return f"{title}\n\n`SKIPPED` — {table.reason}"
    shown = table.rows[: table.top]
    systems = {row.system for row in table.rows}
    with_system = len(systems) > 1
    header = ["#", *(["SYSTEM"] if with_system else []), "CODE", "DESCRIPTION"]
    header += ["Point", "Point 95% CI", "Lifetime", "Lifetime 95% CI"]
    align = ["r", *([""] if with_system else []), "", "", "r", "r", "r", "r"]
    rows = []
    for rank, row in enumerate(shown, start=1):
        description = "—" if row.description is None else _cell(row.description)
        if row.social:
            description += " *(social/administrative)*"
        rows.append(
            (
                rank,
                *([f"`{row.system}`" if row.system else "—"] if with_system else []),
                f"`{row.code}`",
                description,
                _count(row.point),
                _ci(row.point),
                _count(row.lifetime),
                _ci(row.lifetime),
            )
        )
    system_note = ""
    if not with_system and systems:
        (only,) = systems
        system_note = f", all in `{only}`" if only else ""
    summary = (
        f"Codes with at least one alive patient: **{len(table.rows)}**{system_note}. "
        f"The top {len(shown)} are listed; the denominator is the {alive} alive patients."
    )
    blocks = [title, summary, "\n".join(f"- {note}" for note in table.notes)]
    if rows:
        blocks.append("\n".join(_table(header, rows, align=align)))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _rate_cells(rate: Rate) -> tuple[Any, ...]:
    return rate.numerator, rate.denominator, _pct(rate.value), _ci(rate)


def _count(rate: Rate) -> str:
    return f"{rate.numerator} ({_pct(rate.value)})"


def _ci(rate: Rate) -> str:
    interval = rate.interval
    if interval is None:
        return "—"
    return f"{interval[0] * 100:.2f}–{interval[1] * 100:.2f}%"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _table(
    header: Iterable[str], rows: Iterable[Iterable[Any]], *, align: Iterable[str]
) -> list[str]:
    header = list(header)
    separators = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separators) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _write(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
