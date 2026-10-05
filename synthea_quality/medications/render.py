"""[MEDICATIONS] Markdown of the medications of a module validation.

Read only from :class:`~synthea_quality.medications.models.MedicationResult`: no CSV, no
computation. One table gives, for every medication, its population and the active and
ever shares with their 95% CI and how many have the cohort's condition as
``REASONCODE``; a second one what records were used; notes follow. Expected shares are
left to the report's table of reference values, next to those of the conditions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from synthea_quality.medications.models import MedicationResult
from synthea_quality.prevalence.models import Rate
from synthea_quality.profile.models import SectionStatus


def render_medications(results: tuple[MedicationResult, ...], *, level: int = 2) -> str:
    """The medications section; ``level`` is the Markdown level of its heading."""
    rows = []
    records = []
    notes = []
    for result in results:
        codes = ", ".join(f"`{code}`" for code in result.codes)
        population = result.cohort.condition if result.cohort else "alive"
        if result.status is SectionStatus.SKIPPED:
            rows.append((result.name, codes, population, *("—",) * 5))
            notes.append(f"**{result.name}**: `SKIPPED` — {result.reason}")
            continue
        assert result.active is not None and result.ever is not None
        rows.append(
            (
                result.name,
                codes,
                population,
                result.active.denominator,
                _share(result.active),
                _share(result.ever),
                _reason(result.active_with_reason),
                _reason(result.ever_with_reason),
            )
        )
        by_code = ", ".join(
            f"`{code}` {count}" for code, count in result.metrics.get("records_by_code", {}).items()
        )
        records.append(
            (
                result.name,
                result.metrics.get("records", 0),
                result.metrics.get("records_without_stop", 0),
                by_code,
            )
        )
        notes += [f"**{result.name}**: {note}" for note in result.notes]

    blocks = [
        f"{'#' * level} Medications",
        "\n".join(
            [
                "- **Active**: patients of the population with a record of any of the codes "
                "whose `START` is on or before the reference date and whose `STOP` is empty or "
                "after it (by calendar day).",
                "- **Ever**: patients of the population with a record whose `START` is on or "
                "before the reference date.",
                "- **Population**: the alive patients with the named condition at the reference "
                "date (the cohort), or every alive patient. A patient counts once per "
                "medication, whichever of its codes they have. **95% CI**: Wilson score "
                "interval.",
                "- **With the condition as reason**: of those patients, how many have such a "
                "record whose `REASONCODE` is one of the cohort condition's codes.",
            ]
        ),
        "\n".join(
            _table(
                ("Medication", "Codes", "Population", "Patients", "Active (95% CI)",
                 "Ever (95% CI)", "Active, condition as reason", "Ever, condition as reason"),
                rows,
                align=("", "", "", "r", "r", "r", "r", "r"),
            )
        ),
    ]
    if records:
        blocks.append(
            "Records of the population used (a record without `STOP` counts as active):\n\n"
            + "\n".join(
                _table(
                    ("Medication", "Records", "Without STOP", "By code"),
                    records,
                    align=("", "r", "r", ""),
                )
            )
        )
    if notes:
        blocks.append("\n".join(f"- {note}" for note in notes))
    return "\n\n".join(blocks)


def _share(rate: Rate) -> str:
    interval = rate.interval
    if rate.value is None or interval is None:
        return f"{rate.numerator} (—)"
    return (
        f"{rate.numerator} ({rate.value * 100:.2f}%; "
        f"{interval[0] * 100:.2f}–{interval[1] * 100:.2f}%)"
    )


def _reason(count: int | None) -> str:
    return "—" if count is None else str(count)


def _table(
    header: Iterable[str], rows: Iterable[Iterable[Any]], *, align: Iterable[str]
) -> list[str]:
    header = list(header)
    separators = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(separators) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines
