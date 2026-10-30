"""Differences, relative differences and the order of the largest changes, through compare().

Each test builds the figures of the two sides and lets :func:`compare` (or
:func:`largest_changes`) work out the state, the difference and the relative
difference, so a wrong sign, a made-up relative difference for a zero basis or
an order by the signed relative difference makes a test fail.
"""

from __future__ import annotations

import pytest

from synthea_quality.compare.compute import compare, largest_changes
from synthea_quality.compare.models import (
    CHANGED,
    PER_1000_PERSON_YEARS,
    PROPORTION,
    SKIPPED,
    UNCHANGED,
    ComparisonKey,
    Figure,
)

POINT = ComparisonKey("conditions", "MI", "point")
INCIDENCE = ComparisonKey("conditions", "MI", "incidence")


def test_a_zero_basis_makes_the_relative_difference_none():
    baseline = {POINT: Figure(unit=PROPORTION, value=0.0, numerator=0, denominator=10)}
    candidate = {POINT: Figure(unit=PROPORTION, value=0.03, numerator=3, denominator=100)}
    [comparison] = compare(baseline, candidate)
    assert comparison.state == CHANGED
    assert comparison.difference == 0.03
    assert comparison.relative_difference is None


def test_equal_figures_are_unchanged():
    figure = Figure(unit=PROPORTION, value=0.05, numerator=5, denominator=100)
    [comparison] = compare({POINT: figure}, {POINT: figure})
    assert comparison.state == UNCHANGED
    assert comparison.difference == 0.0
    assert comparison.relative_difference == 0.0


def test_a_change_is_candidate_minus_baseline():
    baseline = {POINT: Figure(unit=PROPORTION, value=0.06)}
    candidate = {POINT: Figure(unit=PROPORTION, value=0.04)}
    [comparison] = compare(baseline, candidate)
    assert comparison.state == CHANGED
    assert comparison.difference == -0.02
    assert comparison.relative_difference == pytest.approx(-1 / 3)


def test_a_rate_without_person_time_is_a_change_without_a_difference():
    baseline = {INCIDENCE: Figure(unit=PER_1000_PERSON_YEARS, value=2.16, numerator=1, denominator=365)}
    candidate = {INCIDENCE: Figure(unit=PER_1000_PERSON_YEARS, value=None, numerator=0, denominator=0)}
    [comparison] = compare(baseline, candidate)
    assert comparison.state == CHANGED
    assert comparison.difference is None
    assert comparison.relative_difference is None


def test_a_skipped_figure_on_one_side_makes_the_comparison_skipped():
    baseline = {POINT: Figure(unit=PROPORTION, state=SKIPPED, reason="no condition records")}
    candidate = {POINT: Figure(unit=PROPORTION, value=0.05)}
    for comparison in (*compare(baseline, candidate), *compare(candidate, baseline)):
        assert comparison.state == SKIPPED
        assert comparison.reason == "no condition records"
        assert comparison.difference is None
        assert comparison.relative_difference is None


def test_largest_changes_rank_by_the_size_of_the_relative_difference_not_its_sign():
    small_rise = ComparisonKey("conditions", "A", "point")
    large_fall = ComparisonKey("conditions", "B", "point")
    baseline = {
        small_rise: Figure(unit=PROPORTION, value=0.2),
        large_fall: Figure(unit=PROPORTION, value=0.2),
    }
    candidate = {
        small_rise: Figure(unit=PROPORTION, value=0.21),
        large_fall: Figure(unit=PROPORTION, value=0.1),
    }
    ranked = largest_changes(compare(baseline, candidate), 2)
    assert [comparison.key for comparison in ranked] == [large_fall, small_rise]
    assert ranked[0].relative_difference == pytest.approx(-0.5)
    assert ranked[1].relative_difference == pytest.approx(0.05)
