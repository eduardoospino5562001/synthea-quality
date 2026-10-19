"""Tests for the JSON layout of the indexed cohort results."""

from __future__ import annotations

import json

import pytest

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
from synthea_quality.observations.models import ValueSummary
from synthea_quality.prevalence.models import Rate
from synthea_quality.profile.models import SectionStatus


def breakdown(patients: int = 1, cohort: int = 4) -> Breakdown:
    """A breakdown whose total, one sex and one age band carry the same counts."""
    return Breakdown(
        total=Stratum("all", patients, cohort),
        by_sex=(Stratum("F", patients, cohort),),
        by_age=(Stratum("18-64", patients, cohort),),
    )


def event(name: str = "Hospital admission") -> EventResult:
    return EventResult(
        name=name,
        table="encounters",
        codes=("1505002",),
        window=Window(),
        breakdown=breakdown(),
        days=ValueSummary(0),
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
        values=(ValueShare("A", 2), ValueShare("B", 1)),
        patients_with_several_values=1,
        rows_matched=3,
        rows_date_unusable=0,
        rows_after_reference=0,
        rows_outside_window=0,
        rows_value_empty=1,
    )


def death(with_cause: int | None) -> DeathResult:
    return DeathResult(
        name="Death",
        cause_values=("Malignant neoplasm of breast (disorder)",),
        window=Window(),
        breakdown=breakdown(),
        days=ValueSummary(0),
        deaths_after_reference=0,
        deaths_outside_window=0,
        deathdate_unusable=0,
        with_cause=with_cause,
        other_cause=None,
        no_cause_record=None,
    )


def test_m1_a_stratum_reuses_the_rate_and_its_interval():
    cell = Stratum("all", 1, 4).to_dict()
    rate = Rate(1, 4).to_dict()
    assert cell["numerator"] == 1
    assert cell["denominator"] == 4
    assert cell["rate"] == 0.25
    assert cell["ci95_low"] == rate["ci95_low"]
    assert cell["ci95_high"] == rate["ci95_high"]


def test_m2_a_stratum_of_an_empty_cohort_has_no_rate_or_interval():
    cell = Stratum("all", 0, 0).to_dict()
    assert cell["rate"] is None
    assert cell["ci95_low"] is None
    assert cell["ci95_high"] is None


def test_m3_an_event_lists_its_codes_and_window_with_its_type_and_status():
    data = event().to_dict()
    assert data["type"] == "event"
    assert data["status"] == SectionStatus.COMPUTED.value
    assert data["codes"] == ["1505002"]
    assert data["window"] == {"from_days": 0, "to_days": None}


def test_m4_a_value_keeps_its_shares_in_order():
    assert value().to_dict()["values"] == [
        {"value": "A", "patients": 2},
        {"value": "B", "patients": 1},
    ]


def test_m5_a_death_without_cause_counts_keeps_the_key_with_none():
    data = death(with_cause=None).to_dict()
    assert "with_cause" in data
    assert data["with_cause"] is None


def test_m6_a_skipped_outcome_without_a_reason_is_an_error():
    with pytest.raises(ValueError):
        SkippedOutcome("Death", "death", "")


def test_m7_the_result_lists_every_outcome_with_its_status():
    result = IndexedCohortResult(
        index_condition="COVID-19",
        reference_date="2026-01-01",
        patients=4,
        rows_matched=5,
        rows_unknown_patient=0,
        rows_start_unusable=1,
        rows_after_reference=0,
        outcomes=(event(), SkippedOutcome("ICU admission", "event", "the table is missing")),
    )
    outcomes = result.to_dict()["outcomes"]
    assert len(outcomes) == 2
    assert [outcome["status"] for outcome in outcomes] == ["COMPUTED", "SKIPPED"]


def test_m8_the_whole_result_is_serialisable():
    result = IndexedCohortResult(
        index_condition="COVID-19",
        reference_date="2026-01-01",
        patients=4,
        rows_matched=5,
        rows_unknown_patient=0,
        rows_start_unusable=1,
        rows_after_reference=0,
        outcomes=(event(), value(), death(with_cause=1), SkippedOutcome("ICU", "event", "no table")),
    )
    text = json.dumps(result.to_dict())
    assert json.loads(text)["index_condition"] == "COVID-19"
