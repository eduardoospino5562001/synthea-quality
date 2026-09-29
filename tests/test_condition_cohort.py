"""Tests for cohorts defined by a condition of the module file."""

from __future__ import annotations

import pytest

from synthea_quality.condition_cohort import CohortSpec, parse_cohort, select_cohort
from synthea_quality.prevalence.definitions import DefinitionError, parse_condition_option

NAMES = {"Hypertension"}


def test_a_name_alone_uses_the_point_rule():
    assert parse_cohort("Hypertension", origin="f", condition_names=NAMES) == CohortSpec(
        "Hypertension", "point"
    )


def test_an_object_can_set_the_rule():
    spec = parse_cohort(
        {"condition": " Hypertension ", "rule": "lifetime"}, origin="f", condition_names=NAMES
    )
    assert spec == CohortSpec("Hypertension", "lifetime")
    assert "on or before the reference date" in spec.describe()
    assert spec.to_dict() == {"condition": "Hypertension", "rule": "lifetime"}


def test_no_cohort():
    assert parse_cohort(None, origin="f", condition_names=NAMES) is None


@pytest.mark.parametrize(
    "raw, message",
    [
        ("Diabetes", "do not define"),
        ({"condition": "Hypertension", "rule": "incidence"}, "rule"),
        ({"condition": "Hypertension", "extra": 1}, "unknown key"),
        ({"rule": "point"}, "needs a condition name"),
        (3, "must be a condition name or an object"),
    ],
)
def test_malformed_cohorts_are_input_errors(raw, message):
    with pytest.raises(DefinitionError, match=message):
        parse_cohort(raw, origin="f", condition_names=NAMES)


def test_without_condition_records_the_cohort_cannot_be_selected():
    definition = parse_condition_option("Hypertension=59621000")
    members, reason = select_cohort(
        CohortSpec("Hypertension"), {"Hypertension": definition}, None, "conditions.csv is absent"
    )
    assert members is None
    assert "conditions.csv is absent" in reason and "active at the reference date" in reason


def test_invalid_specs():
    with pytest.raises(ValueError):
        CohortSpec(" ")
    with pytest.raises(ValueError):
        CohortSpec("X", "incidence")
