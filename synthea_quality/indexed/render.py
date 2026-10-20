"""[INDEXED] Markdown of the indexed cohort.

Read only from :class:`~synthea_quality.indexed.models.IndexedCohortResult`: no CSV,
no computation. A summary table gives every outcome's window and its patients with a
95% CI; then each outcome repeats itself by sex and by age band at the index date, with
the days from the index date and the records left out on the way to the patients. Each
crossing of the module file follows, with its strata and the share of the crossed
outcome in each of them. An outcome that could not be measured says ``SKIPPED`` and its
reason, never a made-up number; so does a crossing. Every text of the module file or of
the CSV that lands in a table goes through
:func:`~synthea_quality.table_format.table_cell`; the shares and their intervals come
from :class:`~synthea_quality.prevalence.models.Rate`, and the numbers from
:func:`~synthea_quality.table_format.trimmed`.

Shares the medications table helpers until the render helpers are unified (B17).
"""

from __future__ import annotations

from synthea_quality.indexed.definitions import Window
from synthea_quality.indexed.models import (
    Breakdown,
    CrossResult,
    CrossStratum,
    DeathResult,
    EventResult,
    IndexedCohortResult,
    SkippedOutcome,
    Stratum,
    ValueResult,
)
from synthea_quality.medications.render import _share, _table
from synthea_quality.observations.models import MIN_PATIENTS_FOR_PERCENTILES, ValueSummary
from synthea_quality.prevalence.models import Rate
from synthea_quality.table_format import table_cell, trimmed

#: Shown where an outcome has no number: a skipped outcome or a statistic without data.
EMPTY = "—"

#: What the section means, stated once for every indexed cohort.
DEFINITIONS = (
    "- **Index date**: the earliest `START` of the index condition on or before the "
    "reference date.",
    "- **Cohort**: every patient with an index date, alive or deceased; each patient "
    "counts once.",
    "- **Window**: days from each patient's index date (day 0 is the index date).",
    "- **Age**: completed years on the index date. **95% CI**: Wilson score interval.",
)


def render_indexed(result: IndexedCohortResult, *, level: int = 2) -> str:
    """The indexed cohort section; ``level`` is the Markdown level of its heading."""
    heading = [
        f"{'#' * level} Indexed cohort",
        "\n".join(DEFINITIONS),
    ]
    if result.reason is not None:
        return "\n\n".join([*heading, f"`SKIPPED` — {result.reason}"])
    blocks = [
        *heading,
        (
            f"Index condition: **{result.index_condition}** · reference date "
            f"{result.reference_date} · {_count(result.patients, 'patient')}."
        ),
        (
            f"Condition records left out: {result.rows_unknown_patient} of a patient not "
            f"in `patients`, {result.rows_start_unusable} with an unusable `START`, "
            f"{result.rows_after_reference} after the reference date (of "
            f"{_count(result.rows_matched, 'record')} of the condition)."
        ),
        "\n".join(
            _table(
                ("Outcome", "Type", "Window", "Patients (95% CI)"),
                [_summary_row(outcome) for outcome in result.outcomes],
                align=("", "", "", "r"),
            )
        ),
    ]
    for outcome in result.outcomes:
        blocks.append(f"{'#' * (level + 1)} {outcome.name}")
        blocks.extend(_outcome_blocks(outcome))
    for cross in result.crosses:
        blocks.append(f"{'#' * (level + 1)} {cross.outcome} by {cross.by}")
        blocks.extend(_cross_blocks(cross))
    return "\n\n".join(blocks)


def _summary_row(
    outcome: EventResult | ValueResult | DeathResult | SkippedOutcome,
) -> tuple[str, str, str, str]:
    """The summary table row of ``outcome``; a skipped one has no window or patients."""
    if isinstance(outcome, SkippedOutcome):
        return table_cell(outcome.name), outcome.type, EMPTY, EMPTY
    return (
        table_cell(outcome.name),
        _type_text(outcome),
        _window_text(outcome.window),
        _share(Rate(outcome.breakdown.total.patients, outcome.breakdown.total.cohort)),
    )


def _type_text(outcome: EventResult | ValueResult | DeathResult) -> str:
    """The type of a computed outcome, as the module file names it."""
    if isinstance(outcome, EventResult):
        return "event"
    if isinstance(outcome, ValueResult):
        return "value"
    return "death"


def _window_text(window: Window) -> str:
    """The window in days from the index date, in words."""
    if window.to_days is None:
        return f"day {window.from_days} onward"
    return f"days {window.from_days} to {window.to_days}"


def _outcome_blocks(
    outcome: EventResult | ValueResult | DeathResult | SkippedOutcome,
) -> list[str]:
    """The blocks of one outcome below its heading."""
    if isinstance(outcome, SkippedOutcome):
        return [f"`SKIPPED` — {outcome.reason}"]
    blocks = _breakdown_tables(outcome.breakdown)
    if isinstance(outcome, EventResult):
        blocks.append(
            _days_line("Days from the index date to the first record", outcome.days)
        )
        blocks.append(
            f"Records left out: {outcome.rows_date_unusable} with an unusable date, "
            f"{outcome.rows_after_reference} after the reference date, "
            f"{outcome.rows_outside_window} outside the window "
            f"(of {_count(outcome.rows_matched, 'record')} of the codes)."
        )
        return blocks
    if isinstance(outcome, ValueResult):
        blocks.append(
            "\n".join(
                _table(
                    ("Value", "Patients"),
                    [
                        (table_cell(share.value), share.patients)
                        for share in outcome.values
                    ],
                    align=("", "r"),
                )
            )
        )
        if outcome.patients_with_several_values > 0:
            several = outcome.patients_with_several_values
            verb = "has" if several == 1 else "have"
            blocks.append(
                f"{_count(several, 'patient')} {verb} more than one value on the "
                f"picked date and count under each."
            )
        blocks.append(
            f"Records left out: {outcome.rows_date_unusable} with an unusable date, "
            f"{outcome.rows_after_reference} after the reference date, "
            f"{outcome.rows_outside_window} outside the window, "
            f"{outcome.rows_value_empty} with an empty `VALUE` "
            f"(of {_count(outcome.rows_matched, 'observation')} of the code)."
        )
        return blocks
    blocks.append(_days_line("Days from the index date to death", outcome.days))
    if outcome.with_cause is not None:
        causes = ", ".join(f'"{cause}"' for cause in outcome.cause_values)
        blocks.append(
            f"Cause of death (`69453-9`): {outcome.with_cause} with {causes}, "
            f"{outcome.other_cause} with another cause, "
            f"{outcome.no_cause_record} without a cause record."
        )
    blocks.append(
        f"Deaths left out: {outcome.deaths_after_reference} after the reference date, "
        f"{outcome.deaths_outside_window} outside the window, "
        f"{outcome.deathdate_unusable} with an unusable `DEATHDATE`."
    )
    return blocks


def _cross_blocks(cross: CrossResult) -> list[str]:
    """The blocks of one crossing below its heading.

    A crossing that could not be computed says so and shows no table: a made-up number
    would be worse than no number.
    """
    if cross.reason is not None:
        return [f"Not computed: {cross.reason}."]
    blocks = [_cross_table(cross)]
    blocks.append(
        f"The crossing does not require {cross.by} to be observed before "
        f"{cross.outcome}: each outcome keeps its own window."
    )
    if cross.patients_in_several_strata > 0:
        several = cross.patients_in_several_strata
        blocks.append(
            f"{_count(several, 'patient')} "
            f"{'is' if several == 1 else 'are'} under more than one stratum and "
            f"{'counts' if several == 1 else 'count'} in each."
        )
    return blocks


def _cross_table(cross: CrossResult) -> str:
    """The table of a crossing: its strata, their cohort and the crossed outcome.

    The column of the cause is only there when the crossed outcome counts one: a
    crossing of an outcome without a cause has no cause to show.
    """
    with_cause = any(stratum.with_cause is not None for stratum in cross.strata)
    header: tuple[str, ...] = (
        table_cell(cross.by),
        "Cohort",
        f"{table_cell(cross.outcome)} (95% CI)",
    )
    align: tuple[str, ...] = ("", "r", "r")
    if with_cause:
        header = (*header, "With the cause")
        align = (*align, "r")
    rows = [_cross_row(stratum, with_cause) for stratum in cross.strata]
    return "\n".join(_table(header, rows, align=align))


def _cross_row(
    stratum: CrossStratum, with_cause: bool
) -> tuple[str, int, str] | tuple[str, int, str, str]:
    """One table row: the stratum, its cohort and the crossed outcome inside it.

    The cause is only in the row when the table has that column.
    """
    label, cohort, share = _share_row(stratum.label, stratum.numerator, stratum.denominator)
    if not with_cause:
        return (label, cohort, share)
    cause = EMPTY if stratum.with_cause is None else str(stratum.with_cause)
    return (label, cohort, share, cause)


def _breakdown_tables(breakdown: Breakdown) -> list[str]:
    """The two tables of every computed outcome: by sex and by age at the index date."""
    return [
        "\n".join(
            _table(
                ("Sex", "Cohort", "Patients (95% CI)"),
                _stratum_rows(breakdown.by_sex),
                align=("", "r", "r"),
            )
        ),
        "\n".join(
            _table(
                ("Age at index", "Cohort", "Patients (95% CI)"),
                _stratum_rows(breakdown.by_age),
                align=("", "r", "r"),
            )
        ),
    ]


def _stratum_rows(strata: tuple[Stratum, ...]) -> list[tuple[str, int, str]]:
    """One table row per stratum: its label, its cohort and its share with the 95% CI."""
    return [
        _share_row(stratum.label, stratum.patients, stratum.cohort) for stratum in strata
    ]


def _share_row(label: str, numerator: int, denominator: int) -> tuple[str, int, str]:
    """One table row: its label, its cohort and the share with the 95% CI."""
    return (table_cell(label), denominator, _share(Rate(numerator, denominator)))


def _count(n: int, noun: str) -> str:
    """``n`` followed by ``noun``, in the plural unless ``n`` is 1."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _days_line(prefix: str, days: ValueSummary) -> str:
    """``prefix`` followed by the distribution of one value per patient."""
    if days.n > 0 and (days.p25 is None or days.p75 is None):
        percentiles = f"(P25–P75 not shown below {MIN_PATIENTS_FOR_PERCENTILES} patients)"
    else:
        percentiles = f"(P25–P75 {_number(days.p25)}–{_number(days.p75)})"
    return (
        f"{prefix}: n {days.n}, median {_number(days.median)} {percentiles}, "
        f"min {_number(days.minimum)}, max {_number(days.maximum)}."
    )


def _number(value: float | None) -> str:
    """A number of a summary, or :data:`EMPTY` when the distribution has none."""
    return EMPTY if value is None else trimmed(value)
