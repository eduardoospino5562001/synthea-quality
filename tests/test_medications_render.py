"""Tests for the structured result and the Markdown of the medications of a cohort."""

from __future__ import annotations

import json

import pytest

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.medications.models import MedicationResult
from synthea_quality.medications.render import render_medications
from synthea_quality.prevalence.models import Expected, ExpectedComparison, Rate
from synthea_quality.profile.models import SectionStatus

COHORT = CohortSpec("Hypertension")


def computed(name="Lisinopril", codes=("314076",), active=(14, 17), ever=(14, 17), **kwargs):
    return MedicationResult(
        name, codes, SectionStatus.COMPUTED, cohort=COHORT,
        active=Rate(*active), ever=Rate(*ever), active_with_reason=14, ever_with_reason=14,
        metrics={"records": 299, "records_without_stop": 14,
                 "records_by_code": {"314076": 299}},
        **kwargs,
    )


def test_invariants_and_json():
    with pytest.raises(ValueError):
        MedicationResult("x", ("1",), SectionStatus.SKIPPED)
    with pytest.raises(ValueError):
        MedicationResult("x", ("1",), SectionStatus.SKIPPED, reason="r", active=Rate(0, 1))
    with pytest.raises(ValueError):
        MedicationResult("x", ("1",), SectionStatus.COMPUTED, active=Rate(0, 1))
    result = computed(
        descriptions={"314076": "lisinopril"},
        expected=(ExpectedComparison(Expected("active", 0.9, "S"), Rate(14, 17)),),
    )
    data = json.loads(json.dumps(result.to_dict()))
    assert data["active"]["rate"] == round(14 / 17, 6)
    assert data["cohort"] == {"condition": "Hypertension", "rule": "point"}
    assert data["expected"][0]["position_in_ci95"] == "inside"
    assert list(data["metrics"]) == sorted(data["metrics"])


def test_the_section():
    skipped = MedicationResult(
        "Losartan", ("979485",), SectionStatus.SKIPPED, reason="no medication records",
        cohort=COHORT,
    )
    everyone = MedicationResult(
        "Statin", ("1",), SectionStatus.COMPUTED, active=Rate(0, 0), ever=Rate(0, 0),
        notes=("No record of 1 in medications.csv (any patient, any date): x.",),
    )
    text = render_medications((computed(notes=("14 of 299 records have no STOP.",)),
                               skipped, everyone))
    assert text.startswith("## Medications\n")
    assert (
        "| Lisinopril | `314076` | Hypertension | 17 | 14 (82.35%; 58.97–93.81%) | "
        "14 (82.35%; 58.97–93.81%) | 14 | 14 |"
    ) in text
    assert "| Losartan | `979485` | Hypertension | — | — | — | — | — |" in text
    assert "| Statin | `1` | alive | 0 | 0 (—) | 0 (—) | — | — |" in text
    assert "| Lisinopril | 299 | 14 | `314076` 299 |" in text
    assert "- **Losartan**: `SKIPPED` — no medication records" in text
    assert "- **Lisinopril**: 14 of 299 records have no STOP." in text
    for word in ("PASS", "FAIL", "match"):
        assert word not in text
