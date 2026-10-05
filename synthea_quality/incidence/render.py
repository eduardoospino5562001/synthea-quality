"""[INCIDENCE] JSON and Markdown renderings of an incidence report.

Both read only the :class:`~synthea_quality.incidence.models.IncidenceReport`. The Markdown
states, before any number, the population followed (and in one line why the deceased are
in it), the window, what is known about the exported history and the definitions; then
each condition with its rate, reference values described as inside or outside the 95% CI,
strata and counts of who was left out.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synthea_quality.export_history import notice_lines
from synthea_quality.incidence.models import (
    ConditionIncidence,
    IncidenceRate,
    IncidenceReport,
    IncidenceStratum,
)
from synthea_quality.profile.models import InputState, SectionStatus

MARKDOWN_NAME = "synthea_incidence.md"
JSON_NAME = "synthea_incidence.json"

_POPULATION = {
    "all": "every patient, deceased included until their death",
    "alive": "the patients alive at the end of the simulation (`--alive-only`)",
}
_EXCLUDED = {
    "birthdate_unusable": "empty or unparseable BIRTHDATE",
    "born_after_reference": "born after the reference date",
    "deathdate_unparseable": "DEATHDATE present but unparseable",
    "died_before_window": "died before the window started",
    "deceased_not_in_alive_only": "deceased, left out by --alive-only",
}


def dumps(report: IncidenceReport, *, indent: int | None = 2) -> str:
    return report.to_json(indent=indent) + "\n"


def write_json(report: IncidenceReport, path: str | Path) -> Path:
    return _write(path, dumps(report))


def write_markdown(report: IncidenceReport, path: str | Path) -> Path:
    return _write(path, render_markdown(report))


def render_markdown(report: IncidenceReport) -> str:
    blocks = [
        "# Synthea incidence report",
        "This report **describes** how often new cases of each condition appear, per 1,000 "
        "person-years. It states no verdict: a reference value is shown next to the observed "
        "rate, inside or outside its 95% confidence interval, and nothing more.",
    ]
    notice = notice_lines(report.export_history)
    if notice:
        blocks.append("\n".join(notice))
    blocks += [
        _scope(report),
        _reference(report),
        _window(report),
        _definitions(),
        *(render_condition(c) for c in report.conditions),
    ]
    if not report.conditions:
        blocks.append("## Conditions\n\nNone were given.")
    return "\n\n".join(block.rstrip() for block in blocks) + "\n"


def _scope(report: IncidenceReport) -> str:
    lines = [
        "## Scope",
        "",
        f"- Dataset: `{report.data_dir}`",
        f"- Generated: {report.generated_at} by synthea-quality {report.tool_version}",
        f"- Population: {_POPULATION.get(report.population, report.population)}",
    ]
    if report.cohort:
        lines.append(f"- Patients followed: **{report.cohort.get('followed')}**")
        excluded = {k: v for k, v in report.cohort.get("excluded", {}).items() if v}
        for key, count in excluded.items():
            lines.append(f"  - left out, {_EXCLUDED.get(key, key)}: {count}")
    lines.append("- Tables:")
    for item in report.inputs:
        if item.state is InputState.READ:
            lines.append(f"  - `{item.table}`: {item.rows} rows, for {item.used_for}")
        else:
            lines.append(f"  - `{item.table}`: **{item.state.value}** — {item.reason}")
    if report.incomplete:
        lines += ["", "**This report is incomplete:** a table it needs could not be read."]
    return "\n".join(lines)


def _reference(report: IncidenceReport) -> str:
    reference = report.reference_date
    if reference is None:
        return f"## Reference date\n\n`SKIPPED` — {report.reference_reason}"
    kind = "an **approximation**" if reference.approximate else "exact"
    return (
        f"## Reference date\n\n**{reference.value}** — {kind}, source "
        f"`{reference.source.value}`.\n\n> {reference.detail}"
    )


def _window(report: IncidenceReport) -> str:
    lines = ["## Window, population and exported history", ""]
    if report.window is None:
        lines.append("No window: there is no reference date or no usable data.")
    else:
        lines.append(
            f"Window: **{report.window.start}** to **{report.window.end}** "
            f"({report.window.years} years)."
        )
    if report.notes:
        lines += ["", *(f"- {note}" for note in report.notes)]
    return "\n".join(lines)


def _definitions() -> str:
    return "\n".join(
        [
            "## Definitions",
            "",
            "- **Incidence**: first events in the window over the person-years at risk, per "
            "1,000.",
            "- **At risk**: from the later of the window start and birth, to the earliest of the "
            "reference date, death and the first event. A patient with a record of the "
            "condition before entering the window is a prior case and is not at risk.",
            "- **Event**: the earliest `START` of any of the condition's codes; a patient counts "
            "once per condition.",
            "- **Age bands**: each patient's person-years are split between the bands they go "
            "through; an event counts in the band of the age on its day.",
            "- **95% CI**: exact (Garwood) Poisson interval.",
        ]
    )


def render_condition(
    condition: ConditionIncidence, *, level: int = 2, title_prefix: str = "Condition: "
) -> str:
    """One condition's section; ``level`` is the Markdown level of its heading.

    Public so that ``synthea-validate-module`` shows a condition exactly as
    ``synthea-incidence`` does, under its own headings.
    """
    codes = ", ".join(
        f"`{ref.code}`" if ref.system is None else f"`{ref.system}|{ref.code}`"
        for ref in condition.codes
    )
    title = f"{condition.name} (declared acute)" if condition.acute else condition.name
    head = f"{'#' * level} {title_prefix}{title}\n\nCodes: {codes}"
    if condition.status is SectionStatus.SKIPPED:
        return f"{head}\n\n`SKIPPED` — {condition.reason}"
    assert condition.rate is not None
    blocks = [
        head,
        "\n".join(
            _table(
                ("Events", "Person-years", "Per 1,000 person-years", "95% CI"),
                [_rate_cells(condition.rate)],
                align=("r", "r", "r", "r"),
            )
        ),
    ]
    if condition.expected:
        rows = []
        for item in condition.expected:
            position = (
                f"{item.position} the 95% CI" if item.position is not None else "no interval"
            )
            difference = "—" if item.difference is None else f"{item.difference:+.2f}"
            rows.append(
                (
                    _num(item.expected.value),
                    _num(item.observed.value),
                    difference,
                    position,
                    item.expected.source or "—",
                )
            )
        blocks.append(
            "Reference values per 1,000 person-years (shown for comparison, not as a "
            "verdict):\n\n"
            + "\n".join(
                _table(
                    ("Reference", "Observed", "Difference", "Position", "Source"),
                    rows,
                    align=("r", "r", "r", "", ""),
                )
            )
        )
    for dimension, title_text, label in (
        ("age_band", "By age band (person-years split between bands)", "Age band"),
        ("GENDER", "By sex (`GENDER`)", "`GENDER`"),
    ):
        strata = [s for s in condition.strata if s.dimension == dimension]
        if strata:
            blocks.append(
                f"{'#' * (level + 1)} {title_text}\n\n" + "\n".join(_strata(strata, label))
            )
    m = condition.metrics
    blocks.append(
        f"Patients followed: {m.get('followed')}; prior cases (not at risk): "
        f"{m.get('prior_cases')}; at risk: {m.get('at_risk')}. Other records of at-risk "
        f"patients: {m.get('records_on_the_event_day', 0)} on the day of a first event, "
        f"{m.get('later_records_not_counted', 0)} on a later day."
    )
    if condition.notes:
        blocks.append("\n".join(f"- {note}" for note in condition.notes))
    return "\n\n".join(blocks)


def _strata(strata: list[IncidenceStratum], label: str) -> list[str]:
    rows = [("*(empty)*" if s.value is None else s.value, *_rate_cells(s.rate)) for s in strata]
    return _table(
        (label, "Events", "Person-years", "Per 1,000 person-years", "95% CI"),
        rows,
        align=("", "r", "r", "r", "r"),
    )


def _rate_cells(rate: IncidenceRate) -> tuple[Any, ...]:
    interval = rate.interval
    ci = "—" if interval is None else f"{_num(interval[0])}–{_num(interval[1])}"
    return rate.events, f"{rate.person_years:.2f}", _num(rate.value), ci


def _num(value: float | None) -> str:
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
