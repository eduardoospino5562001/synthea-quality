"""Tests for the Markdown of the indexed cohort."""

from __future__ import annotations

import re
from dataclasses import replace

from synthea_quality.indexed.definitions import Window
from synthea_quality.indexed.models import (
    Breakdown,
    DeathResult,
    EventResult,
    IndexedCohortResult,
    SkippedOutcome,
    Stratum,
    ValueResult,
    ValueShare,
)
from synthea_quality.indexed.render import render_indexed
from synthea_quality.observations.models import ValueSummary


def breakdown() -> Breakdown:
    """A breakdown whose total, sexes and age band carry counts of four patients."""
    return Breakdown(
        total=Stratum("all", 1, 4),
        by_sex=(Stratum("F", 1, 2), Stratum("M", 1, 30000)),
        by_age=(Stratum("18-64", 1, 4),),
    )


def event() -> EventResult:
    return EventResult(
        name="Hospital | admission",
        table="encounters",
        codes=("1505002",),
        window=Window(),
        breakdown=breakdown(),
        days=ValueSummary(1, 0, None, None, 0, None, None, 0),
        rows_matched=3,
        rows_date_unusable=0,
        rows_after_reference=1,
        rows_outside_window=1,
    )


def value() -> ValueResult:
    return ValueResult(
        name="Stage group",
        code="21908-9",
        pick="first",
        window=Window(),
        breakdown=breakdown(),
        values=(ValueShare("Stage 2", 1), ValueShare("Stage | IIA", 1)),
        patients_with_several_values=1,
        rows_matched=3,
        rows_date_unusable=1,
        rows_after_reference=1,
        rows_outside_window=0,
        rows_value_empty=1,
    )


def death() -> DeathResult:
    return DeathResult(
        name="Death",
        cause_values=("Malignant neoplasm of breast (disorder)",),
        window=Window(-30, 30),
        breakdown=breakdown(),
        days=ValueSummary(1, 0, None, None, 0, None, None, 0),
        deaths_after_reference=0,
        deaths_outside_window=0,
        deathdate_unusable=0,
        with_cause=1,
        other_cause=0,
        no_cause_record=0,
    )


def result() -> IndexedCohortResult:
    return IndexedCohortResult(
        index_condition="COVID-19",
        reference_date="2026-01-01",
        patients=4,
        rows_matched=4,
        rows_unknown_patient=0,
        rows_start_unusable=1,
        rows_after_reference=1,
        outcomes=(event(), value(), death(), SkippedOutcome("ICU admission", "event", "the devices table is missing")),
    )


def test_r1_the_heading_follows_the_requested_level():
    assert render_indexed(result()).startswith("## Indexed cohort")
    assert render_indexed(result(), level=3).startswith("### Indexed cohort")


def test_r2_the_condition_the_date_and_the_size_are_stated():
    assert "Index condition: **COVID-19** · reference date 2026-01-01 · 4 patients." in render_indexed(result())


def test_r3_the_window_is_written_in_days_from_the_index_date():
    text = render_indexed(result())
    assert "day 0 onward" in text
    assert "days -30 to 30" in text


def test_r4_a_pipe_in_an_outcome_name_is_escaped_in_the_summary_table():
    assert "Hospital \\| admission" in render_indexed(result())


def test_r5_a_skipped_outcome_shows_its_reason_and_no_numbers():
    text = render_indexed(result())
    assert "`SKIPPED` — the devices table is missing" in text
    assert "| ICU admission | event | — | — |" in text


def test_r6_patients_with_several_values_are_explained():
    assert "1 patients have more than one value" in render_indexed(result())


def test_r7_the_cause_of_death_is_listed_with_its_three_counts():
    text = render_indexed(result())
    assert "Cause of death (`69453-9`): 1 with" in text
    assert "without a cause record" in text


def test_r8_no_verdict_words():
    assert re.search(r"\b(pass|fail|ok|match)\b", render_indexed(result()), re.I) is None


def test_r9_the_same_result_renders_the_same_text():
    assert render_indexed(result()) == render_indexed(result())


def test_r10_a_small_share_is_not_shown_as_zero():
    assert "(0.00%" not in render_indexed(result())


def test_r11_a_cohort_that_could_not_be_built_shows_its_reason_only():
    text = render_indexed(replace(result(), reason="x"))
    assert "`SKIPPED` — x" in text
    assert "| Outcome |" not in text


def test_r12_few_values_say_the_percentiles_are_not_shown():
    few = replace(event(), days=ValueSummary(3, 0, None, None, 1, None, None, 2))
    text = render_indexed(replace(result(), outcomes=(few,)))
    assert "P25–P75 not shown below 10 patients" in text


def test_r13_each_stratum_shows_its_cohort():
    text = render_indexed(result())
    assert "| Sex | Cohort | Patients (95% CI) |" in text
    assert "| Age at index | Cohort | Patients (95% CI) |" in text
    assert "| F | 2 | 1 (50.00%; 9.45–90.55%) |" in text
