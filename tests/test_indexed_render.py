"""Tests for the Markdown of the indexed cohort."""

from __future__ import annotations

import re
from dataclasses import replace

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


def cross() -> CrossResult:
    """A crossing of the deaths in the strata of a value, with a cause to count."""
    return CrossResult(
        outcome="Death",
        by="Stage group",
        by_type="value",
        strata=(
            CrossStratum("Stage 2", 1, 3, 1),
            CrossStratum("Stage | IIA", 0, 1, 0),
            CrossStratum("no value", 0, 2, 0),
        ),
        patients_in_several_strata=1,
    )


def cross_without_cause() -> CrossResult:
    """A crossing of the deaths in the strata of an event, without a cause to count."""
    return CrossResult(
        outcome="Death",
        by="Hospital | admission",
        by_type="event",
        strata=(
            CrossStratum("with Hospital | admission", 1, 2, None),
            CrossStratum("without Hospital | admission", 0, 3, None),
        ),
        patients_in_several_strata=0,
    )


def skipped_cross() -> CrossResult:
    """A crossing that could not be computed."""
    return CrossResult(
        outcome="Death",
        by="Stage group",
        by_type="value",
        strata=(),
        patients_in_several_strata=0,
        reason="outcome 'Stage group' was not computed: observations.csv is not in the dataset",
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
    assert "1 patient has more than one value" in render_indexed(result())


def test_r7_the_cause_of_death_is_listed_with_its_three_counts():
    text = render_indexed(result())
    assert "Cause of death (`69453-9`): 1 with" in text
    assert "without a cause record" in text
    assert "0 with another cause" in text


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


def test_r14_a_cross_shows_its_strata_under_its_own_heading():
    text = render_indexed(replace(result(), crosses=(cross(),)))
    assert "### Death by Stage group" in text
    assert "| Stage group | Cohort | Death (95% CI) | With the cause |" in text
    assert "| Stage 2 | 3 | 1 (33.33%; 6.15–79.23%) | 1 |" in text
    assert "| Stage \\| IIA | 1 | 0 (0.00%; 0.00–79.35%) | 0 |" in text
    assert "| no value | 2 | 0 (0.00%; 0.00–65.76%) | 0 |" in text


def test_r15_the_column_of_the_cause_is_only_there_when_there_is_a_cause():
    text = render_indexed(replace(result(), crosses=(cross_without_cause(),)))
    assert "| Hospital \\| admission | Cohort | Death (95% CI) |" in text
    assert "With the cause" not in text


def test_r16_a_cross_says_that_no_order_is_required():
    text = render_indexed(replace(result(), crosses=(cross(),)))
    assert (
        "The crossing does not require Stage group to be observed before Death: "
        "each outcome keeps its own window."
    ) in text


def test_r17_a_patient_under_several_strata_is_explained():
    text = render_indexed(replace(result(), crosses=(cross(),)))
    assert "1 patient is under more than one stratum and counts in each." in text
    quiet = render_indexed(replace(result(), crosses=(cross_without_cause(),)))
    assert "more than one stratum" not in quiet


def test_r18_a_cross_that_could_not_be_computed_shows_its_reason_and_no_table():
    text = render_indexed(replace(result(), crosses=(skipped_cross(),)))
    assert (
        "Not computed: outcome 'Stage group' was not computed: observations.csv is not "
        "in the dataset."
    ) in text
    assert "| Stage group | Cohort |" not in text


def test_r19_without_a_cross_the_text_has_no_crossing():
    text = render_indexed(result())
    assert " by " not in text
    assert "Cross" not in text


def test_r20_no_verdict_words_in_a_crossing():
    text = render_indexed(replace(result(), crosses=(cross(), cross_without_cause())))
    assert re.search(r"\b(pass|fail|ok|match)\b", text, re.I) is None
