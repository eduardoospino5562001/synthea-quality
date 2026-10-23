"""Tests for the comparison models: figures, keys and comparisons."""

from __future__ import annotations

import pytest

from synthea_quality.compare.models import (
    CHANGED,
    COMPUTED,
    COUNT,
    ONLY_IN_BASELINE,
    ONLY_IN_CANDIDATE,
    PROPORTION,
    SKIPPED,
    UNCHANGED,
    Comparison,
    ComparisonKey,
    Figure,
)


def test_a_computed_figure_carries_its_value_interval_and_counts():
    figure = Figure(
        unit=PROPORTION,
        value=0.05,
        ci95_low=0.02,
        ci95_high=0.11,
        numerator=5,
        denominator=100,
    )
    assert figure.unit == PROPORTION
    assert figure.value == 0.05
    assert (figure.ci95_low, figure.ci95_high) == (0.02, 0.11)
    assert (figure.numerator, figure.denominator) == (5, 100)
    assert figure.state == COMPUTED
    assert figure.reason is None


def test_a_count_figure_needs_no_rate_fields():
    figure = Figure(unit=COUNT, value=3, numerator=3)
    assert figure.value == 3
    assert figure.state == COMPUTED


def test_a_skipped_figure_needs_a_reason_and_no_numbers():
    figure = Figure(unit=PROPORTION, state=SKIPPED, reason="conditions.csv is missing")
    assert figure.state == SKIPPED
    assert figure.reason == "conditions.csv is missing"
    assert figure.value is None
    assert figure.to_dict()["state"] == SKIPPED


def test_a_skipped_figure_without_a_reason_is_an_error():
    with pytest.raises(ValueError, match="a skipped figure needs a reason"):
        Figure(unit=PROPORTION, state=SKIPPED)


def test_a_skipped_figure_with_numbers_is_an_error():
    with pytest.raises(ValueError, match="a skipped figure carries no numbers"):
        Figure(unit=PROPORTION, value=0.05, state=SKIPPED, reason="missing table")


def test_a_computed_figure_cannot_carry_a_reason():
    with pytest.raises(ValueError, match="a computed figure carries no reason"):
        Figure(unit=PROPORTION, value=0.05, reason="something")


def test_a_key_identifies_block_element_measure_and_stratum():
    key = ComparisonKey("conditions", "Myocardial infarction", "point", "sex:F")
    assert (key.block, key.element, key.measure, key.stratum) == (
        "conditions",
        "Myocardial infarction",
        "point",
        "sex:F",
    )
    assert key.to_dict() == {
        "block": "conditions",
        "element": "Myocardial infarction",
        "measure": "point",
        "stratum": "sex:F",
    }
    assert ComparisonKey("conditions", "MI", "lifetime", "") < ComparisonKey("conditions", "MI", "point", "")
    assert ComparisonKey("conditions", "MI", "point", "sex:F") < ComparisonKey("conditions", "MI", "point", "sex:M")


def test_two_equal_figures_are_unchanged():
    base = Figure(unit=PROPORTION, value=0.05, numerator=5, denominator=100)
    cand = Figure(unit=PROPORTION, value=0.05, numerator=5, denominator=100)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=0.0,
        relative_difference=0.0,
        state=UNCHANGED,
    )
    assert comparison.state == UNCHANGED
    assert comparison.difference == 0.0
    assert comparison.relative_difference == 0.0


def test_a_change_is_candidate_minus_baseline_with_its_sign():
    base = Figure(unit=PROPORTION, value=0.06)
    cand = Figure(unit=PROPORTION, value=0.08)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=0.02,
        relative_difference=(0.08 - 0.06) / 0.06,
        state=CHANGED,
    )
    assert comparison.state == CHANGED
    assert comparison.difference == 0.02
    assert abs(comparison.candidate.value - comparison.baseline.value - 0.02) < 1e-9


def test_a_lower_candidate_has_a_negative_difference():
    base = Figure(unit=PROPORTION, value=0.06)
    cand = Figure(unit=PROPORTION, value=0.04)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=-0.02,
        relative_difference=(0.04 - 0.06) / 0.06,
        state=CHANGED,
    )
    assert comparison.difference == -0.02


def test_a_zero_basis_makes_the_relative_difference_none_and_not_an_infinity():
    base = Figure(unit=COUNT, value=0)
    cand = Figure(unit=COUNT, value=3)
    comparison = Comparison(
        key=ComparisonKey("indexed", "cohort", "patients"),
        unit=COUNT,
        baseline=base,
        candidate=cand,
        difference=3,
        relative_difference=None,
        state=CHANGED,
    )
    assert comparison.relative_difference is None


def test_a_missing_baseline_or_candidate_makes_the_relative_difference_none():
    base = Figure(unit=PROPORTION, value=0.05)
    comparison = Comparison(
        key=ComparisonKey("conditions", "HTN", "point"),
        unit=PROPORTION,
        baseline=None,
        candidate=base,
        difference=None,
        relative_difference=None,
        state=ONLY_IN_CANDIDATE,
    )
    assert comparison.state == ONLY_IN_CANDIDATE
    assert comparison.difference is None
    assert comparison.relative_difference is None
    other = Comparison(
        key=ComparisonKey("conditions", "HTN", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=None,
        difference=None,
        relative_difference=None,
        state=ONLY_IN_BASELINE,
    )
    assert other.state == ONLY_IN_BASELINE


def test_a_skipped_comparison_keeps_its_reason():
    base = Figure(unit=PROPORTION, state=SKIPPED, reason="no condition records")
    cand = Figure(unit=PROPORTION, value=0.05)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=None,
        relative_difference=None,
        state=SKIPPED,
        reason="no condition records",
    )
    assert comparison.state == SKIPPED
    assert comparison.reason == "no condition records"
    assert comparison.difference is None
    assert comparison.relative_difference is None


def test_a_comparison_carries_its_unit_and_figures():
    base = Figure(unit=PROPORTION, value=0.05, numerator=5, denominator=100)
    cand = Figure(unit=PROPORTION, value=0.08, numerator=8, denominator=100)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=0.03,
        relative_difference=0.6,
        state=CHANGED,
    )
    data = comparison.to_dict()
    assert data["key"] == {
        "block": "conditions",
        "element": "MI",
        "measure": "point",
        "stratum": "",
    }
    assert data["unit"] == PROPORTION
    assert data["baseline"]["value"] == 0.05
    assert data["baseline"]["numerator"] == 5
    assert data["candidate"]["value"] == 0.08
    assert data["candidate"]["denominator"] == 100
    assert data["difference"] == 0.03
    assert data["relative_difference"] == 0.6
    assert data["state"] == CHANGED
