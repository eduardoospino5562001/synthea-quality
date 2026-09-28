"""Tests for the prevalence result models and the Wilson interval."""

from __future__ import annotations

import json

import pytest

from synthea_quality.prevalence.models import (
    CodeRef,
    ConditionResult,
    Expected,
    ExpectedComparison,
    GeneralTable,
    PrevalenceReport,
    Rate,
    wilson_interval,
)
from synthea_quality.profile.models import SectionStatus


def test_wilson_matches_known_values():
    # Classic textbook values of the 95% Wilson interval.
    low, high = wilson_interval(1, 2)
    assert (round(low, 4), round(high, 4)) == (0.0945, 0.9055)
    low, high = wilson_interval(0, 10)
    assert (low, round(high, 4)) == (0.0, 0.2775)
    # symmetric: k/n mirrors (n-k)/n
    low, high = wilson_interval(7, 99)
    mirror_low, mirror_high = wilson_interval(92, 99)
    assert round(low, 12) == round(1 - mirror_high, 12)
    assert round(high, 12) == round(1 - mirror_low, 12)


def test_wilson_at_the_edges_stays_inside_zero_one_and_is_not_empty():
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0.0 < high < 0.35
    low, high = wilson_interval(10, 10)
    assert 0.65 < low < 1.0 and high == 1.0


def test_no_interval_without_a_denominator():
    assert wilson_interval(0, 0) is None
    rate = Rate(0, 0)
    assert rate.value is None and rate.interval is None
    assert rate.to_dict() == {
        "numerator": 0, "denominator": 0, "rate": None, "ci95_low": None, "ci95_high": None,
    }


@pytest.mark.parametrize("numerator, denominator", [(-1, 5), (6, 5), (0, -1)])
def test_invalid_rates_are_rejected(numerator, denominator):
    with pytest.raises(ValueError):
        Rate(numerator, denominator)


def test_rate_values_are_rounded_for_determinism():
    assert Rate(1, 3).value == 0.333333


def test_expected_must_be_a_proportion_of_a_known_measure():
    with pytest.raises(ValueError):
        Expected("mortality", 0.1)
    with pytest.raises(ValueError):
        Expected("point", 1.5)
    with pytest.raises(ValueError):
        Expected("point", float("nan"))


def test_expected_incidence_is_a_non_negative_rate_per_1000_person_years():
    assert Expected("incidence", 12.5).value == 12.5
    with pytest.raises(ValueError):
        Expected("incidence", -1)


def test_expected_comparison_is_inside_or_outside_the_ci_never_a_verdict():
    observed = Rate(7, 99)
    inside = ExpectedComparison(Expected("lifetime", 0.05), observed)
    outside = ExpectedComparison(Expected("lifetime", 0.30, "survey"), observed)
    assert inside.position == "inside"
    assert outside.position == "outside"
    assert inside.difference == round(7 / 99 - 0.05, 6)
    assert set(outside.to_dict()) == {
        "measure", "value", "source", "observed", "difference", "position_in_ci95",
    }
    assert ExpectedComparison(Expected("point", 0.1), Rate(0, 0)).position is None


def test_condition_results_follow_the_computed_or_skipped_contract():
    codes = (CodeRef("22298006"),)
    with pytest.raises(ValueError):
        ConditionResult("MI", codes, SectionStatus.SKIPPED)
    with pytest.raises(ValueError):
        ConditionResult("MI", codes, SectionStatus.COMPUTED, point=Rate(1, 2))
    skipped = ConditionResult("MI", codes, SectionStatus.SKIPPED, reason="no conditions.csv")
    assert skipped.to_dict()["point"] is None


def test_report_json_is_deterministic_and_versioned():
    report = PrevalenceReport(
        data_dir="/d",
        reference_date=None,
        reference_reason="no reference",
        age_bands=(0, 18),
        alive=None,
        conditions=(),
        general=GeneralTable(SectionStatus.SKIPPED, reason="no reference"),
        social_list={"id": "x", "codes": 0},
        generated_at="2026-09-28T00:00:00+00:00",
    )
    data = json.loads(report.to_json())
    assert data["schema_version"] == 1
    assert list(data)[:4] == ["schema_version", "tool_version", "generated_at", "data_dir"]
    assert "point" in data["definitions"] and "Wilson" in data["definitions"]["interval"]
    assert report.to_json() == report.to_json()
