"""[VALIDATE] JSON and Markdown renderings of a module validation.

The Markdown is meant to be read top-down by someone checking a module: scope and
reference date; the definitions and the two populations (prevalence counts the patients
alive at the end, incidence follows everyone until death); the population summary; one
summary table of every condition; one table of every reference value next to its
observed rate, inside or outside the 95% CI; then each condition's prevalence and
incidence exactly as ``synthea-prevalence`` and ``synthea-incidence`` show them (their
reference values are already in the table above, so they are not repeated there).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from synthea_quality.incidence.render import render_condition as render_incidence
from synthea_quality.prevalence.models import INCIDENCE
from synthea_quality.prevalence.render import render_condition as render_prevalence
from synthea_quality.profile.models import InputState, SectionStatus
from synthea_quality.profile.render import render_section
from synthea_quality.validate.models import ConditionValidation, ModuleValidationReport

MARKDOWN_NAME = "synthea_module_validation.md"
JSON_NAME = "synthea_module_validation.json"


def dumps(report: ModuleValidationReport, *, indent: int | None = 2) -> str:
    return report.to_json(indent=indent) + "\n"


def write_json(report: ModuleValidationReport, path: str | Path) -> Path:
    return _write(path, dumps(report))


def write_markdown(report: ModuleValidationReport, path: str | Path) -> Path:
    return _write(path, render_markdown(report))


def render_markdown(report: ModuleValidationReport) -> str:
    name = report.module.name or Path(report.module_file).stem
    blocks = [
        f"# Module validation: {name}",
        "This report **describes** a Synthea population against the conditions of one "
        "module: their prevalence among the patients alive at the end of the simulation and "
        "their incidence per 1,000 person-years. It states no verdict: each reference value "
        "is shown next to the observed one, inside or outside its 95% confidence interval.",
        _scope(report),
        _reference(report),
        _definitions(report),
        _population(report),
        _summary(report),
        _references(report),
        *(_condition(condition) for condition in report.conditions),
    ]
    return "\n\n".join(block.rstrip() for block in blocks) + "\n"


def _scope(report: ModuleValidationReport) -> str:
    lines = [
        "## Scope",
        "",
        f"- Dataset: `{report.data_dir}`",
        f"- Module file: `{report.module_file}`",
    ]
    if report.module.synthea_modules:
        modules = ", ".join(f"`{m}`" for m in report.module.synthea_modules)
        lines.append(f"- Synthea modules: {modules}")
    if report.module.description:
        lines.append(f"- Description: {report.module.description}")
    lines.append(f"- Generated: {report.generated_at} by synthea-quality {report.tool_version}")
    lines.append("- Tables:")
    for item in report.inputs:
        if item.state is InputState.READ:
            lines.append(f"  - `{item.table}`: {item.rows} rows, for {item.used_for}")
        else:
            lines.append(f"  - `{item.table}`: **{item.state.value}** — {item.reason}")
    if report.incomplete:
        lines += ["", "**This report is incomplete:** a table it needs could not be read."]
    return "\n".join(lines)


def _reference(report: ModuleValidationReport) -> str:
    reference = report.reference_date
    if reference is None:
        return f"## Reference date\n\n`SKIPPED` — {report.reference_reason}"
    kind = "an **approximation**" if reference.approximate else "exact"
    return (
        f"## Reference date\n\n**{reference.value}** — {kind}, source "
        f"`{reference.source.value}`.\n\n> {reference.detail}"
    )


def _definitions(report: ModuleValidationReport) -> str:
    window = report.incidence_window
    period = f"from **{window.start}** to **{window.end}**" if window else "(no window)"
    lines = [
        "## Definitions and populations",
        "",
        "- **Prevalence** — among the patients **alive at the end of the simulation**: "
        "*point*, those with a record whose `START` is on or before the reference date and "
        "whose `STOP` is empty or after it; *lifetime*, those with a record whose `START` is "
        "on or before it. 95% Wilson interval.",
        f"- **Incidence** — first events per 1,000 person-years {period}, following **every "
        "patient, deceased included, until their death**; patients with a record before the "
        "window are prior cases, not at risk. Exact 95% Poisson interval.",
        "- A patient counts once per condition, whichever of its codes they have.",
    ]
    lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines)


def _population(report: ModuleValidationReport) -> str:
    sections = [render_section(section, level=3) for section in report.population]
    followed = report.cohort.get("followed")
    lead = (
        f"Alive at the end: **{report.alive if report.alive is not None else '—'}** "
        f"(the prevalence denominator). Followed for incidence: "
        f"**{followed if followed is not None else '—'}**."
    )
    return "\n\n".join(["## Population summary", lead, *sections])


def _summary(report: ModuleValidationReport) -> str:
    rows = []
    for condition in report.conditions:
        name = f"{condition.name} (acute)" if condition.acute else condition.name
        p = condition.prevalence
        i = condition.incidence
        computed_p = p.status is SectionStatus.COMPUTED
        rows.append(
            (
                name,
                _proportion(p.point) if computed_p else "—",
                _proportion(p.lifetime) if computed_p else "—",
                _incidence(i.rate) if i.status is SectionStatus.COMPUTED else "—",
            )
        )
    table = _table(
        ("Condition", "Point prevalence (95% CI)", "Lifetime prevalence (95% CI)",
         "Incidence per 1,000 person-years (95% CI)"),
        rows,
        align=("", "r", "r", "r"),
    )
    return "## Summary\n\n" + "\n".join(table)


def _references(report: ModuleValidationReport) -> str:
    rows = []
    for condition in report.conditions:
        for item in condition.references:
            is_rate = item.expected.measure == INCIDENCE
            show = _rate_number if is_rate else _percent
            interval = item.observed.interval
            ci = "—" if interval is None else f"{show(interval[0])}–{show(interval[1])}"
            position = f"{item.position} the 95% CI" if item.position else "no interval"
            rows.append(
                (
                    condition.name,
                    "incidence per 1,000 PY" if is_rate else f"{item.expected.measure} prevalence",
                    show(item.expected.value),
                    show(item.observed.value),
                    ci,
                    position,
                    item.expected.source or "—",
                )
            )
    if not rows:
        return "## Reference values\n\nThe module file gives no reference values."
    table = _table(
        ("Condition", "Measure", "Reference", "Observed", "95% CI", "Position", "Source"),
        rows,
        align=("", "", "r", "r", "r", "", ""),
    )
    return (
        "## Reference values\n\nShown for comparison, not as a verdict.\n\n" + "\n".join(table)
    )


def _condition(condition: ConditionValidation) -> str:
    title = f"{condition.name} (declared acute)" if condition.acute else condition.name
    return "\n\n".join(
        [
            f"## Condition: {title}",
            render_prevalence(
                replace(condition.prevalence, expected=()), level=3, title_prefix="Prevalence: "
            ),
            render_incidence(
                replace(condition.incidence, expected=()), level=3, title_prefix="Incidence: "
            ),
        ]
    )


def _proportion(rate: Any) -> str:
    interval = rate.interval
    if rate.value is None or interval is None:
        return "—"
    return f"{_percent(rate.value)} ({_percent(interval[0])}–{_percent(interval[1])})"


def _incidence(rate: Any) -> str:
    interval = rate.interval
    if rate.value is None or interval is None:
        return "—"
    return f"{_rate_number(rate.value)} ({_rate_number(interval[0])}–{_rate_number(interval[1])})"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _rate_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


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
