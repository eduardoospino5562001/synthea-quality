"""Tests for the structured result of observation values."""

from __future__ import annotations

import json

import pytest

from synthea_quality.observations.definitions import ReferenceRange
from synthea_quality.observations.models import (
    GeneralTable,
    ObservationResult,
    ObservationsReport,
    RangeComparison,
    UnitGroup,
    ValueSummary,
)
from synthea_quality.profile.models import SectionStatus


def summary(n=3):
    return ValueSummary(n, 1.0, 1.1, 1.5, 2.0, 2.5, 2.9, 3.0) if n else ValueSummary(0)


def test_a_summary_has_statistics_exactly_when_it_has_values():
    assert summary().to_dict()["median"] == 2.0
    assert ValueSummary(0).to_dict()["median"] is None
    with pytest.raises(ValueError):
        ValueSummary(0, 1.0)
    with pytest.raises(ValueError):
        ValueSummary(2)
    with pytest.raises(ValueError):
        ValueSummary(-1)


def test_values_are_rounded():
    s = ValueSummary(1, *([1 / 3] * 7))
    assert s.to_dict()["min"] == 0.333333


def test_status_invariants():
    group = UnitGroup("mg", summary())
    with pytest.raises(ValueError):
        ObservationResult("x", "1", None, SectionStatus.SKIPPED)
    with pytest.raises(ValueError):
        ObservationResult("x", "1", None, SectionStatus.SKIPPED, reason="r", groups=(group,))
    with pytest.raises(ValueError):
        ObservationResult("x", "1", None, SectionStatus.COMPUTED)
    with pytest.raises(ValueError):
        RangeComparison(ReferenceRange("mg", 1), SectionStatus.SKIPPED)
    skipped = RangeComparison(ReferenceRange("mg", 1), SectionStatus.SKIPPED, reason="units")
    assert skipped.to_dict()["below"] is None


def test_the_report_is_deterministic_json():
    result = ObservationResult(
        "x", "1", "X", SectionStatus.COMPUTED, population=3,
        groups=(UnitGroup("mg", summary(), metrics={"b": 1, "a": 2}),),
    )
    report = ObservationsReport(
        data_dir="d", reference_date=None, reference_reason="none", age_bands=(0, 18),
        alive=3, observations=(result,), general=GeneralTable(SectionStatus.SKIPPED, reason="r"),
        generated_at="2026-09-29T00:00:00+00:00",
    )
    data = json.loads(report.to_json())
    assert data["schema_version"] == 1
    assert list(data["observations"][0]["groups"][0]["metrics"]) == ["a", "b"]
    assert "type 7" in data["definitions"]["percentiles"]
    assert report.to_json() == report.to_json()
    assert not report.incomplete
