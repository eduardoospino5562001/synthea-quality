"""Tests for the medications of a module file."""

from __future__ import annotations

import pytest

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.medications.definitions import MedicationDefinition, load_medications
from synthea_quality.prevalence.definitions import DefinitionError
from synthea_quality.prevalence.models import Expected

NAMES = {"Hypertension"}


def load(items):
    return load_medications({"medications": items}, origin="f", condition_names=NAMES)


def test_a_medication_with_cohort_and_expected_shares():
    (lisinopril,) = load(
        [{"name": " Lisinopril ", "codes": ["314076", " 314076 "], "cohort": "Hypertension",
          "expected": {"active": 0.8, "ever": 1, "source": "S"}}]
    )
    assert lisinopril == MedicationDefinition(
        "Lisinopril", ("314076",), CohortSpec("Hypertension"),
        (Expected("active", 0.8, "S"), Expected("ever", 1.0, "S")),
    )


def test_without_medications_or_cohort():
    assert load_medications({}, origin="f", condition_names=NAMES) == ()
    (plain,) = load([{"name": "Statin", "codes": ["1"]}])
    assert plain.cohort is None and plain.expected == ()


@pytest.mark.parametrize(
    "item, message",
    [
        ({"codes": ["1"]}, "needs a name"),
        ({"name": "A"}, "at least one code"),
        ({"name": "A", "codes": []}, "at least one code"),
        ({"name": "A", "codes": [1]}, "non-empty string"),
        ({"name": "A", "codes": ["http://www.nlm.nih.gov/research/umls/rxnorm|1"]}, "SYSTEM"),
        ({"name": "A", "codes": ["1"], "dose": 1}, "unknown key"),
        ({"name": "A", "codes": ["1"], "cohort": "Diabetes"}, "do not define"),
        ({"name": "A", "codes": ["1"], "expected": {"point": 0.5}}, "unknown key"),
        ({"name": "A", "codes": ["1"], "expected": {"active": 1.5}}, r"\[0, 1\]"),
        ({"name": "A", "codes": ["1"], "expected": {"active": "0.5"}}, "not a number"),
        ({"name": "A", "codes": ["1"], "expected": {"active": True}}, "not a number"),
        ({"name": "A", "codes": ["1"], "expected": {"ever": 0.5, "source": 2}}, "source"),
        ({"name": "A", "codes": ["1"], "expected": [0.5]}, "must be an object"),
        ("A", "not an object"),
    ],
)
def test_malformed_medications_are_input_errors(item, message):
    with pytest.raises(DefinitionError, match=message):
        load([item])


def test_list_errors():
    for value in ([], {}, "x"):
        with pytest.raises(DefinitionError, match="non-empty list"):
            load_medications({"medications": value}, origin="f", condition_names=NAMES)
    with pytest.raises(DefinitionError, match="more than once"):
        load([{"name": "A", "codes": ["1"]}, {"name": "A", "codes": ["2"]}])
