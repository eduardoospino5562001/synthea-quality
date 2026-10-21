"""Tests for the JSON layout of the indexed cohort results."""

from __future__ import annotations

import json

import pytest

from synthea_quality.indexed.definitions import Window
from synthea_quality.indexed.models import (
    Breakdown,
    CrossResult,
    CrossStratum,
    DeathResult,
    EventMatchResult,
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


def match() -> EventMatchResult:
    """A part that ended an isolation, with the records it left out."""
    return EventMatchResult(
        name="Isolation ended",
        table="careplans",
        codes=("736376001",),
        values=(),
        reasons=("840539006",),
        at="stop",
        patients=1551,
        rows_matched=1611,
        rows_filtered_out=0,
        rows_not_ended=60,
        rows_date_unusable=0,
        rows_after_reference=0,
        rows_outside_window=0,
    )


def composite() -> EventResult:
    """An event of two parts, with no table and no codes of its own."""
    return EventResult(
        name="Recovered",
        table=None,
        codes=(),
        window=Window(),
        breakdown=breakdown(),
        days=ValueSummary(0),
        rows_matched=3222,
        rows_date_unusable=0,
        rows_after_reference=0,
        rows_outside_window=0,
        rows_filtered_out=0,
        rows_not_ended=120,
        op="all",
        matches=(match(), match()),
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


def test_m9_a_cross_stratum_reuses_the_rate_and_its_interval():
    cell = CrossStratum("Stage 1", 1, 4, 2).to_dict()
    rate = Rate(1, 4).to_dict()
    assert cell["label"] == "Stage 1"
    assert cell["numerator"] == 1
    assert cell["denominator"] == 4
    assert cell["rate"] == 0.25
    assert cell["ci95_low"] == rate["ci95_low"]
    assert cell["ci95_high"] == rate["ci95_high"]
    assert cell["with_cause"] == 2


def test_m10_a_cross_stratum_without_a_cause_keeps_the_key_with_none():
    cell = CrossStratum("no value", 0, 3, None).to_dict()
    assert cell["with_cause"] is None
    assert cell["rate"] == 0.0


def test_m11_a_computed_cross_lists_its_strata_and_its_status():
    result = CrossResult(
        outcome="Death",
        by="Stage group",
        by_type="value",
        strata=(CrossStratum("Stage 1", 1, 4, 2),),
        patients_in_several_strata=0,
    )
    data = result.to_dict()
    assert data["outcome"] == "Death"
    assert data["by"] == "Stage group"
    assert data["by_type"] == "value"
    assert data["status"] == SectionStatus.COMPUTED.value
    assert data["strata"] == [
        {
            "label": "Stage 1",
            "numerator": 1,
            "denominator": 4,
            "rate": 0.25,
            "ci95_low": CrossStratum("Stage 1", 1, 4, 2).to_dict()["ci95_low"],
            "ci95_high": CrossStratum("Stage 1", 1, 4, 2).to_dict()["ci95_high"],
            "with_cause": 2,
        }
    ]
    assert data["patients_in_several_strata"] == 0
    assert data["reason"] is None


def test_m12_a_skipped_cross_has_no_strata_and_its_reason():
    result = CrossResult(
        outcome="Death",
        by="Stage group",
        by_type="value",
        strata=(),
        patients_in_several_strata=0,
        reason="outcome 'Stage group' was not computed: observations.csv is missing",
    )
    data = result.to_dict()
    assert data["status"] == SectionStatus.SKIPPED.value
    assert data["strata"] == []
    assert data["reason"] == "outcome 'Stage group' was not computed: observations.csv is missing"


def test_m13_a_cross_is_serialisable():
    result = CrossResult(
        outcome="Death",
        by="Stage group",
        by_type="value",
        strata=(CrossStratum("Stage 1", 1, 4, 2),),
        patients_in_several_strata=1,
    )
    text = json.dumps(result.to_dict())
    assert json.loads(text)["strata"][0]["label"] == "Stage 1"


def test_m14_a_part_lists_its_rule_and_its_records():
    data = match().to_dict()
    assert list(data) == [
        "name",
        "table",
        "codes",
        "values",
        "reasons",
        "at",
        "patients",
        "rows_matched",
        "rows_filtered_out",
        "rows_not_ended",
        "rows_date_unusable",
        "rows_after_reference",
        "rows_outside_window",
    ]
    assert data["name"] == "Isolation ended"
    assert data["table"] == "careplans"
    assert data["codes"] == ["736376001"]
    assert data["values"] == []
    assert data["reasons"] == ["840539006"]
    assert data["at"] == "stop"
    assert data["patients"] == 1551
    assert data["rows_not_ended"] == 60


def test_m15_an_event_keeps_its_keys_and_adds_the_new_ones():
    data = event().to_dict()
    assert list(data) == [
        "type",
        "status",
        "name",
        "table",
        "codes",
        "window",
        "breakdown",
        "days_to_first_record",
        "rows_matched",
        "rows_date_unusable",
        "rows_after_reference",
        "rows_outside_window",
        "rows_filtered_out",
        "rows_not_ended",
        "op",
        "matches",
    ]
    assert data["table"] == "encounters"
    assert data["codes"] == ["1505002"]
    assert data["rows_filtered_out"] == 0
    assert data["rows_not_ended"] == 0
    assert data["op"] == "any"
    assert data["matches"] == []


def test_m16_an_event_of_several_parts_has_no_table_and_lists_its_parts():
    data = composite().to_dict()
    assert data["table"] is None
    assert data["codes"] == []
    assert data["op"] == "all"
    assert len(data["matches"]) == 2
    assert data["matches"][0] == match().to_dict()


def test_m17_the_rows_of_an_event_are_the_sum_of_the_rows_of_its_parts():
    data = composite().to_dict()
    for key in (
        "rows_matched",
        "rows_filtered_out",
        "rows_not_ended",
        "rows_date_unusable",
        "rows_after_reference",
        "rows_outside_window",
    ):
        assert data[key] == sum(part[key] for part in data["matches"])


def test_m18_the_new_keys_of_an_event_are_serialisable():
    text = json.dumps(composite().to_dict())
    assert json.loads(text)["matches"][0]["at"] == "stop"
