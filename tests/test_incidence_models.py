"""Tests for the exact Poisson interval and the incidence result models."""

from __future__ import annotations

import json

import pytest

from synthea_quality.incidence.models import (
    ConditionIncidence,
    IncidenceRate,
    IncidenceReport,
    Window,
)
from synthea_quality.incidence.poisson import exact_interval, poisson_cdf
from synthea_quality.prevalence.models import CodeRef
from synthea_quality.profile.models import SectionStatus


@pytest.mark.parametrize(
    "k, low, high",
    [
        # Garwood exact 95% limits as found in standard tables of Poisson confidence limits.
        (0, 0.0, 3.689),
        (1, 0.0253, 5.572),
        (2, 0.2422, 7.225),
        (5, 1.623, 11.668),
        (10, 4.795, 18.390),
    ],
)
def test_exact_interval_matches_tabulated_values(k, low, high):
    got_low, got_high = exact_interval(k)
    assert round(got_low, 4 if k < 5 else 3) == low
    assert round(got_high, 3) == high


def test_interval_bounds_have_the_defining_tail_probabilities():
    low, high = exact_interval(7)
    assert 1 - poisson_cdf(6, low) == pytest.approx(0.025, abs=1e-9)
    assert poisson_cdf(7, high) == pytest.approx(0.025, abs=1e-9)


def test_large_counts_stay_finite_and_ordered():
    low, high = exact_interval(5000)
    assert 4800 < low < 5000 < high < 5200


def test_invalid_arguments():
    with pytest.raises(ValueError):
        exact_interval(-1)
    with pytest.raises(ValueError):
        exact_interval(1, alpha=1.5)


def test_rate_per_1000_person_years_and_scaled_interval():
    rate = IncidenceRate(2, 146_100)  # 400 years of 365.25 days
    assert rate.person_years == 400.0
    assert rate.value == 5.0
    low, high = rate.interval
    assert (round(low, 4), round(high, 3)) == (round(0.2422 / 0.4, 4), round(7.225 / 0.4, 3))
    assert rate.to_dict()["events"] == 2


def test_no_rate_without_person_time():
    rate = IncidenceRate(0, 0)
    assert rate.value is None and rate.interval is None
    with pytest.raises(ValueError):
        IncidenceRate(-1, 3)
    with pytest.raises(ValueError):
        IncidenceRate(1, -3)


def test_condition_contract_and_report_json():
    with pytest.raises(ValueError):
        ConditionIncidence("MI", (CodeRef("1"),), SectionStatus.SKIPPED)
    with pytest.raises(ValueError):
        ConditionIncidence("MI", (CodeRef("1"),), SectionStatus.COMPUTED)
    report = IncidenceReport(
        data_dir="/d",
        reference_date=None,
        reference_reason="none",
        window=Window(5, "2021-08-17", "2026-08-17"),
        population="all",
        age_bands=(0, 18),
        conditions=(
            ConditionIncidence(
                "MI", (CodeRef("1"),), SectionStatus.COMPUTED, rate=IncidenceRate(1, 91_311)
            ),
        ),
        generated_at="2026-09-28T00:00:00+00:00",
    )
    data = json.loads(report.to_json())
    assert data["schema_version"] == 1
    assert data["window"] == {"years": 5, "start": "2021-08-17", "end": "2026-08-17"}
    rate = data["conditions"][0]["rate"]
    assert rate["person_days"] == 91_311
    assert rate["per_1000_person_years"] == round(1 / 91_311 * 365.25 * 1000, 6)
    assert "Garwood" in data["definitions"]["interval"]
    assert report.to_json() == report.to_json()
