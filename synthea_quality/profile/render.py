"""[PROFILE] JSON and Markdown renderings of a dataset profile.

Both work only with a :class:`~synthea_quality.profile.models.DatasetProfile`: they
read no CSV, compute nothing and import no data library, so the Markdown can always be
reproduced from the JSON alone, exactly as for the quality report.

JSON
----
The layout belongs to :meth:`DatasetProfile.to_dict` and is versioned by
``schema_version``. This module adds only the file-facing details: two-space
indentation, UTF-8 without ASCII escaping, a trailing newline. Two runs over the same
input produce the same text except for ``generated_at``.

Markdown
--------
Written for someone who wants to know what the dataset contains in a minute: the
scope and the reference date first — including, in plain words, whether that date is
an approximation and where it came from — then the population, the age profile, the
distributions, the date range, the empty or unparseable values, and the most common
codes of each clinical table, each introduced by the reminder that its count is
historical. A skipped section
stays in its place with its reason, so a gap is never mistaken for a zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from synthea_quality.export_history import notice_lines
from synthea_quality.profile.models import (
    CategoryCount,
    DatasetProfile,
    InputState,
    ProfileSection,
    SectionStatus,
)

#: Predictable report file names inside the output directory.
MARKDOWN_NAME = "synthea_profile.md"
JSON_NAME = "synthea_profile.json"

EMPTY_LABEL = "*(empty)*"


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def dumps(profile: DatasetProfile, *, indent: int | None = 2) -> str:
    """Serialise ``profile`` as JSON text, ending with a newline."""
    return profile.to_json(indent=indent) + "\n"


def loads(text: str) -> DatasetProfile:
    """Rebuild a profile from :func:`dumps` output."""
    return DatasetProfile.from_json(text)


def write_json(profile: DatasetProfile, path: str | Path) -> Path:
    """Write ``profile`` to ``path`` as UTF-8 JSON and return the path written."""
    return _write(path, dumps(profile))


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def write_markdown(profile: DatasetProfile, path: str | Path) -> Path:
    """Write ``profile`` to ``path`` as Markdown and return the path written."""
    return _write(path, render_markdown(profile))


def render_markdown(profile: DatasetProfile) -> str:
    """Render ``profile`` as a Markdown document."""
    blocks: list[str] = [
        "# Synthea dataset profile",
        "This profile **describes** the dataset. It runs no check and states no verdict: "
        "a section is either computed or skipped with the reason. Data quality is "
        "reported separately by `synthea-quality`.",
    ]
    notice = notice_lines(profile.export_history)
    if notice:
        blocks.append("\n".join(notice))
    blocks += [
        _scope(profile),
        _reference(profile),
    ]
    for section in profile.sections:
        blocks.append(render_section(section))
    return "\n\n".join(block.rstrip() for block in blocks) + "\n"


def _scope(profile: DatasetProfile) -> str:
    lines = [
        "## Scope",
        "",
        f"- Dataset: `{profile.data_dir}`",
        f"- Generated: {profile.generated_at} by synthea-quality {profile.tool_version}",
        "- Tables:",
    ]
    for item in profile.inputs:
        if item.state is InputState.READ:
            lines.append(f"  - `{item.table}`: {item.rows} rows, for {item.used_for}")
        else:
            lines.append(f"  - `{item.table}`: **{item.state.value}** — {item.reason}")
    if profile.incomplete:
        lines += [
            "",
            "**This profile is incomplete:** a table it needs is present but could not be "
            "read, and the sections that depend on it are skipped.",
        ]
    return "\n".join(lines)


def _reference(profile: DatasetProfile) -> str:
    lines = ["## Reference date", ""]
    reference = profile.reference_date
    if reference is None:
        lines.append(f"`SKIPPED` — {profile.reference_reason}")
    else:
        kind = "an **approximation**" if reference.approximate else "exact"
        lines += [
            f"**{reference.value}** — {kind}, source `{reference.source.value}`.",
            "",
            f"> {reference.detail}",
            "",
            "Ages are measured at this date. Alive and deceased come from `DEATHDATE` "
            "(the state at the end of the simulation), not from this date.",
        ]
    if profile.notes:
        lines += ["", "Notes:", *(f"- {note}" for note in profile.notes)]
    return "\n".join(lines)


def render_section(section: ProfileSection, *, level: int = 2) -> str:
    """One profile section; ``level`` is the Markdown level of its heading.

    Public so that ``synthea-validate-module`` shows the population exactly as the profile
    does.
    """
    title = f"{'#' * level} {section.title}"
    if section.status is SectionStatus.SKIPPED:
        return f"{title}\n\n`SKIPPED` — {section.reason}"
    if section.section_id.startswith("codes."):
        # The code renderer places the notes itself: the first one (the count is
        # historical) has to be read before the table, not after it.
        return f"{title}\n\n{_codes(section)}"
    renderer = _RENDERERS.get(section.section_id)
    if renderer is None:
        if section.section_id.startswith("distribution."):
            renderer = _distribution
        else:  # pragma: no cover - every section built today has a renderer
            renderer = _generic
    body = renderer(section)
    notes = "\n".join(f"- {note}" for note in section.notes)
    return f"{title}\n\n{body}" + (f"\n\n{notes}" if notes else "")


def _population(section: ProfileSection) -> str:
    m = section.metrics
    lines = _table(
        ("", "Patients", "%"),
        [
            ("Total", m["total"], _pct(100.0 if m["total"] else 0.0)),
            ("Alive", m["alive"], _pct(m["alive_percent"])),
            ("Deceased", m["deceased"], _pct(m["deceased_percent"])),
        ],
        align=("", "r", "r"),
    )
    extra = []
    if m["deceased_with_unparseable_deathdate"]:
        extra.append(
            f"Deceased with an unparseable DEATHDATE: {m['deceased_with_unparseable_deathdate']}"
        )
    if "deaths_after_reference_date" in m and m["deaths_after_reference_date"]:
        extra.append(f"Deaths after the reference date: {m['deaths_after_reference_date']}")
    return "\n".join(lines + ([""] + extra if extra else []))


def _age(section: ProfileSection) -> str:
    m = section.metrics
    excluded = m["excluded"]
    summary = (
        f"Alive patients: **{m['alive']}**, with a known age: **{m['denominator']}**. "
        f"Median **{_num(m['median'])}**, minimum **{_num(m['min'])}**, "
        f"maximum **{_num(m['max'])}** years."
    )
    excluded_line = (
        f"Without an age: {excluded['birthdate_empty']} empty BIRTHDATE, "
        f"{excluded['birthdate_unparseable']} unparseable, "
        f"{excluded['born_after_reference_date']} born after the reference date."
    )
    table = _table(
        ("Age band", "Patients", "%"),
        [(row.value, row.count, _pct(row.percent)) for row in section.distributions["age_band"]],
        align=("", "r", "r"),
    )
    return "\n".join([summary, "", excluded_line, "", *table])


def _distribution(section: ProfileSection) -> str:
    ((column, rows),) = section.distributions.items()
    m = section.metrics
    table_rows: list[tuple[Any, ...]] = [
        (_label(row), row.count, _pct(row.percent)) for row in rows
    ]
    if m.get("other_values"):
        other = f"*({m['other_values']} other value(s))*"
        table_rows.insert(m["shown"], (other, m["other_count"], _pct(m["other_percent"])))
    header = f"Alive patients: **{m['denominator']}**, distinct values: {m['distinct_values']}."
    return "\n".join(
        [header, "", *_table((f"`{column}`", "Patients", "%"), table_rows, align=("", "r", "r"))]
    )


def _date_range(section: ProfileSection) -> str:
    rows = [
        (
            f"`{column}`",
            v["min"] or "—",
            v["max"] or "—",
            v["valid"],
            v["empty"],
            v["unparseable"],
        )
        for column, v in section.metrics.items()
    ]
    return "\n".join(
        _table(
            ("Column", "Earliest", "Latest", "Valid", "Empty", "Unparseable"),
            rows,
            align=("", "", "", "r", "r", "r"),
        )
    )


def _completeness(section: ProfileSection) -> str:
    rows = [
        (
            f"`{column}`",
            v["rows"],
            v["empty"],
            "—" if v["unparseable"] is None else v["unparseable"],
        )
        for column, v in section.metrics.items()
    ]
    return "\n".join(
        _table(("Column", "Rows", "Empty", "Unparseable"), rows, align=("", "r", "r", "r"))
    )


def _codes(section: ProfileSection) -> str:
    m = section.metrics
    with_system = "SYSTEM" in m["code_identity"]
    lead, *notes = section.notes
    summary = (
        f"Alive patients: **{m['denominator']}**, with at least one record: "
        f"**{m['patients_with_records']}**. Rows: {m['rows']} ({m['rows_alive']} of alive "
        f"patients). Distinct codes: {m['distinct_codes']} in the table, "
        f"{m['distinct_codes_alive']} among the alive; the top {m['shown']} are listed."
    )
    header = ["#", *(["SYSTEM"] if with_system else []), "CODE", "DESCRIPTION"]
    header += ["Patients", "% alive", "Records"]
    align = ["r", *([""] if with_system else []), "", "", "r", "r", "r"]
    rows = []
    for rank, row in enumerate(section.codes, start=1):
        description = "—" if row.description is None else _cell(row.description)
        if row.description_variants > 1:
            description += f" *({row.description_variants} descriptions)*"
        rows.append(
            (
                rank,
                *([f"`{row.system}`" if row.system else "—"] if with_system else []),
                f"`{row.code}`",
                description,
                row.patients,
                _pct(row.percent),
                row.records,
            )
        )
    blocks = [f"> {lead}", summary]
    if rows:
        blocks.append("\n".join(_table(header, rows, align=align)))
    if section.multi_description_codes:
        blocks.append(
            f"Codes written with more than one description: "
            f"**{m['codes_with_multiple_descriptions']}**. Every variant, most frequent "
            f"first (the one shown above), with its records over the whole table:"
        )
        variant_rows = []
        for item in section.multi_description_codes:
            code = f"`{item.code}`" if item.system is None else f"`{item.system}` `{item.code}`"
            for index, variant in enumerate(item.descriptions):
                text = "—" if variant.description is None else _cell(variant.description)
                variant_rows.append((code if index == 0 else "", text, variant.records))
        blocks.append(
            "\n".join(_table(("CODE", "DESCRIPTION", "Records"), variant_rows, align=("", "", "r")))
        )
    if notes:
        blocks.append("\n".join(f"- {note}" for note in notes))
    return "\n\n".join(blocks)


def _generic(section: ProfileSection) -> str:  # pragma: no cover - fallback only
    return "\n".join(f"- {key}: {value}" for key, value in section.metrics.items())


_RENDERERS = {
    "population": _population,
    "age": _age,
    "date_range": _date_range,
    "completeness": _completeness,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _label(row: CategoryCount) -> str:
    return EMPTY_LABEL if row.value is None else _cell(row.value)


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _pct(value: float) -> str:
    return f"{value:.2f}"


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _table(
    header: Iterable[str], rows: Iterable[Iterable[Any]], *, align: Iterable[str]
) -> list[str]:
    header = list(header)
    separators = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separators) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _write(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


__all__ = [
    "JSON_NAME",
    "MARKDOWN_NAME",
    "dumps",
    "loads",
    "render_markdown",
    "write_json",
    "write_markdown",
]
