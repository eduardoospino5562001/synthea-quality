"""[OBSERVATIONS] JSON and Markdown renderings of an observation values report.

Both read only the :class:`~synthea_quality.observations.models.ObservationsReport`: no
CSV, no computation, no data library. The Markdown gives the rules right after the
reference date (one value per patient, units never converted, how percentiles are
computed), then each observation asked for — one table per unit, its reference range
with the patients below, within and above it, strata, and what was left out — then the
general table with the codes that use several units or several ``TYPE`` values.
A reference range is shown next to the observed values, never as a verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from synthea_quality.export_history import notice_lines
from synthea_quality.observations.models import (
    MIN_PATIENTS_FOR_PERCENTILES,
    PERCENTILE_METHOD,
    PERCENTILES_NOTE,
    VALUE_RULE,
    GeneralTable,
    ObservationResult,
    ObservationsReport,
    RangeComparison,
    UnitGroup,
    ValueStratum,
    ValueSummary,
)
from synthea_quality.profile.models import InputState, SectionStatus

MARKDOWN_NAME = "synthea_observations.md"
JSON_NAME = "synthea_observations.json"

_DIMENSIONS = (
    ("age_band", "By age band at the reference date", "Age band"),
    ("GENDER", "By sex (`GENDER`)", "`GENDER`"),
)
_STATS = ("Min", "P5", "P25", "Median", "P75", "P95", "Max")


def dumps(report: ObservationsReport, *, indent: int | None = 2) -> str:
    """Serialise ``report`` as JSON text, ending with a newline."""
    return report.to_json(indent=indent) + "\n"


def write_json(report: ObservationsReport, path: str | Path) -> Path:
    return _write(path, dumps(report))


def write_markdown(report: ObservationsReport, path: str | Path) -> Path:
    return _write(path, render_markdown(report))


def render_markdown(report: ObservationsReport) -> str:
    """Render ``report`` as a Markdown document."""
    blocks = [
        "# Synthea observation values report",
        "This report **describes** the numeric values of `observations.csv` among the "
        "patients alive at the end of the simulation. It states no verdict: a reference range "
        "is shown next to the observed values, with how many patients fall below, within and "
        "above it, and nothing more.",
    ]
    notice = notice_lines(report.export_history)
    if notice:
        blocks.append("\n".join(notice))
    blocks += [
        _scope(report),
        _reference(report),
        render_definitions(),
        *_observations(report),
        render_general(report.general, report.alive),
    ]
    return "\n\n".join(block.rstrip() for block in blocks) + "\n"


def _scope(report: ObservationsReport) -> str:
    lines = [
        "## Scope",
        "",
        f"- Dataset: `{report.data_dir}`",
    ]
    if report.module_file:
        lines.append(f"- Module file: `{report.module_file}`")
    lines += [
        f"- Generated: {report.generated_at} by synthea-quality {report.tool_version}",
        f"- Alive patients: **{report.alive if report.alive is not None else '—'}**",
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


def _reference(report: ObservationsReport) -> str:
    reference = report.reference_date
    if reference is None:
        return f"## Reference date\n\n`SKIPPED` — {report.reference_reason}"
    kind = "an **approximation**" if reference.approximate else "exact"
    return (
        f"## Reference date\n\n**{reference.value}** — {kind}, source "
        f"`{reference.source.value}`.\n\n> {reference.detail}"
    )


def render_definitions(*, level: int = 2) -> str:
    """The rules behind every observation value; public for ``synthea-validate-module``."""
    return "\n".join(
        [
            f"{'#' * level} How values are described",
            "",
            f"- **One value per patient:** {VALUE_RULE}. A patient measured often weighs no "
            "more than one measured once.",
            "- **Used values:** rows with `TYPE` `numeric` whose `VALUE` is a finite number. "
            "Every other row is counted and reported, never dropped silently.",
            "- **Units:** each `UNITS` is described on its own; values are never converted.",
            f"- **Percentiles:** {PERCENTILE_METHOD}. With fewer than "
            f"{MIN_PATIENTS_FOR_PERCENTILES} patients only n, minimum, median and maximum are "
            "shown.",
            "- **Reference range:** shown next to the observed values with the patients below, "
            "within and above it, only for values in the same units.",
        ]
    )


def _observations(report: ObservationsReport) -> list[str]:
    if not report.observations:
        return [
            "## Observations asked for\n\nNone. Use `--observation CODE` or `--module FILE` "
            "to describe specific codes."
        ]
    return [render_observation(item) for item in report.observations]


def render_observation(
    result: ObservationResult, *, level: int = 2, title_prefix: str = "Observation: "
) -> str:
    """One observation's section; ``level`` is the Markdown level of its heading.

    Public so that ``synthea-validate-module`` shows an observation exactly as
    ``synthea-observations`` does.
    """
    lead = [f"Code: `{result.code}`" + (f" — {result.description}" if result.description else "")]
    if result.cohort is not None:
        lead.append(
            f"Population: {result.cohort.describe()} (rule `{result.cohort.rule}`): "
            f"**{_value(result.population)}** patients."
        )
    else:
        lead.append(f"Population: the **{_value(result.population)}** alive patients.")
    if result.lookback_years:
        lead.append(f"Lookback: {result.lookback_years} year(s).")
    head = f"{'#' * level} {title_prefix}{result.name}\n\n" + "\n".join(
        f"- {line}" for line in lead
    )
    if result.status is SectionStatus.SKIPPED:
        blocks = [head, f"`SKIPPED` — {result.reason}"]
    else:
        blocks = [head, *(_group(group, level + 1) for group in result.groups)]
        blocks.append(_rows(result))
    if result.notes:
        blocks.append("\n".join(f"- {note}" for note in result.notes))
    return "\n\n".join(blocks)


def _group(group: UnitGroup, level: int) -> str:
    units = f"`{group.units}`" if group.units else "*(no units)*"
    blocks = [
        f"{'#' * level} Values in {units}",
        "\n".join(
            _table(
                ("Patients with a value", *_STATS),
                [(group.summary.n, *_stats(group.summary))],
                align=("r",) * 8,
            )
        )
        + (f"\n\n{PERCENTILES_NOTE}" if group.summary.percentiles_withheld else ""),
    ]
    if group.reference_range is not None:
        blocks.append(_range(group.reference_range))
    metrics = group.metrics
    age = metrics.get("latest_value_age_days_median")
    if age is not None:
        blocks.append(
            f"Age of the latest values: median **{_num(age)}** days before the reference date; "
            f"{metrics.get('patients_latest_older_than_1y', 0)} older than 1 year, "
            f"{metrics.get('patients_latest_older_than_3y', 0)} older than 3 years."
        )
    for dimension, title, label in _DIMENSIONS:
        strata = [s for s in group.strata if s.dimension == dimension]
        if strata:
            table = "\n".join(_strata_table(strata, label))
            if any(s.summary.percentiles_withheld for s in strata):
                table += f"\n\n{PERCENTILES_NOTE}"
            blocks.append(f"{'#' * (level + 1)} {title}\n\n{table}")
    return "\n\n".join(blocks)


def _range(comparison: RangeComparison) -> str:
    rng = comparison.range
    low = "—" if rng.low is None else _num(rng.low)
    high = "—" if rng.high is None else _num(rng.high)
    lines = ["Reference range (shown for comparison, not as a verdict):", ""]
    if comparison.status is SectionStatus.SKIPPED:
        lines.append(
            f"`SKIPPED` — range {low} to {high} `{rng.units}` ({rng.source or 'no source'}): "
            f"{comparison.reason}."
        )
    else:
        total = comparison.below + comparison.within + comparison.above
        lines += _table(
            ("Low", "High", "Units", "Below", "Within", "Above", "Source"),
            [
                (
                    low,
                    high,
                    f"`{rng.units}`",
                    _share(comparison.below, total),
                    _share(comparison.within, total),
                    _share(comparison.above, total),
                    rng.source or "—",
                )
            ],
            align=("r", "r", "", "r", "r", "r", ""),
        )
    if rng.note:
        lines += ["", f"> {rng.note}"]
    return "\n".join(lines)


def _strata_table(strata: list[ValueStratum], label: str) -> list[str]:
    rows = [
        (
            "*(empty)*" if s.value is None else s.value,
            s.patients,
            s.summary.n,
            *_stats(s.summary),
        )
        for s in strata
    ]
    return _table(
        (label, "Patients", "With a value", *_STATS), rows, align=("", *("r",) * 9)
    )


def _rows(result: ObservationResult) -> str:
    m = result.metrics
    left_out = [
        ("of deceased patients", m.get("rows_deceased", 0)),
        ("of patients not in patients.csv", m.get("rows_unknown_patient", 0)),
        ("with an empty or unparseable DATE", m.get("rows_date_unusable", 0)),
        ("after the reference date", m.get("rows_after_reference", 0)),
        ("with a TYPE other than numeric", m.get("rows_not_numeric_type", 0)),
        ("with a VALUE that is not a number", m.get("rows_value_unparseable", 0)),
    ]
    detail = "; ".join(f"{count} {what}" for what, count in left_out if count) or "none"
    return (
        f"Rows of `{result.code}` usable among the alive: **{m.get('rows_used', 0)}**. "
        f"Left out: {detail}."
    )


def render_general(table: GeneralTable, alive: int | None, *, level: int = 2) -> str:
    """The general table and the codes that need care."""
    title = f"{'#' * level} All numeric observations"
    if table.status is SectionStatus.SKIPPED:
        return f"{title}\n\n`SKIPPED` — {table.reason}"
    shown = table.rows[: table.top]
    blocks = [
        title,
        f"Codes with a value for at least one alive patient: **{table.metrics.get('codes', 0)}**, "
        f"in **{len(table.rows)}** code and unit pairs. The top {len(shown)} are listed; "
        f"each row describes the latest value of the patients among the {alive} alive.",
        "\n".join(f"- {note}" for note in table.notes),
    ]
    sub = "#" * (level + 1)
    if table.mixed_units:
        rows = [
            (
                f"`{item['code']}`",
                _cell(item["description"] or "—"),
                ", ".join(
                    f"`{u}` {n}" if u else f"*(empty)* {n}"
                    for u, n in sorted(item["units"].items())
                ),
            )
            for item in table.mixed_units
        ]
        blocks.append(
            f"{sub} Codes written in more than one unit\n\nRows with `TYPE` numeric, all "
            f"patients. The units are described separately, never converted.\n\n"
            + "\n".join(_table(("CODE", "DESCRIPTION", "Rows by UNITS"), rows, align=("", "", "")))
        )
    else:
        blocks.append(f"{sub} Codes written in more than one unit\n\nNone.")
    if table.mixed_types:
        rows = [
            (
                f"`{item['code']}`",
                _cell(item["description"] or "—"),
                ", ".join(f"{t or '*(empty)*'} {n}" for t, n in sorted(item["types"].items())),
            )
            for item in table.mixed_types
        ]
        blocks.append(
            f"{sub} Codes written with more than one `TYPE`\n\nAll rows, all patients. Only the "
            f"rows with `TYPE` numeric are described.\n\n"
            + "\n".join(_table(("CODE", "DESCRIPTION", "Rows by TYPE"), rows, align=("", "", "")))
        )
    else:
        blocks.append(f"{sub} Codes written with more than one `TYPE`\n\nNone.")
    if shown:
        rows = []
        for rank, row in enumerate(shown, start=1):
            flags = [
                *(["several units"] if row.mixed_units else []),
                *(["several TYPE"] if row.other_types else []),
            ]
            description = _cell(row.description or "—")
            if flags:
                description += f" *({', '.join(flags)})*"
            s = row.summary
            rows.append(
                (
                    rank,
                    f"`{row.code}`",
                    description,
                    f"`{row.units}`" if row.units else "—",
                    s.n,
                    _num(s.minimum),
                    _num(s.median),
                    _num(s.maximum),
                )
            )
        blocks.append(
            f"{sub} By patients with a value\n\n"
            + "\n".join(
                _table(
                    ("#", "CODE", "DESCRIPTION", "UNITS", "Patients", "Min", "Median", "Max"),
                    rows,
                    align=("r", "", "", "", "r", "r", "r", "r"),
                )
            )
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _stats(summary: ValueSummary) -> tuple[str, ...]:
    return tuple(
        _num(v)
        for v in (
            summary.minimum,
            summary.p5,
            summary.p25,
            summary.median,
            summary.p75,
            summary.p95,
            summary.maximum,
        )
    )


def _num(value: float | None) -> str:
    """Up to four decimals, without trailing zeros (``120``, ``1.015``, ``93.9``)."""
    if value is None:
        return "—"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _share(count: int, total: int) -> str:
    if total == 0:
        return str(count)
    return f"{count} ({count / total * 100:.1f}%)"


def _value(value: Any) -> str:
    return "—" if value is None else str(value)


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
