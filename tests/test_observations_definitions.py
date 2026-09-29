"""Tests for the observations asked for, their cohorts and reference ranges."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.observations.definitions import (
    CONFIGURATION_NOTE,
    ObservationDefinition,
    ReferenceRange,
    load_module_observations,
    load_observations,
    parse_observation_option,
    unique,
)
from synthea_quality.prevalence.definitions import DefinitionError

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples" / "hypertension.json"


def write(tmp_path: Path, data) -> Path:
    path = tmp_path / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_option_with_and_without_a_name():
    assert parse_observation_option("8480-6") == ObservationDefinition("8480-6")
    named = parse_observation_option(" Systolic = 8480-6 ")
    assert (named.name, named.code, named.key) == ("Systolic", "8480-6", "Systolic")
    with pytest.raises(DefinitionError):
        parse_observation_option("=8480-6")
    with pytest.raises(DefinitionError):
        parse_observation_option("")


def test_the_example_file():
    conditions, observations = load_module_observations(EXAMPLE)
    assert [c.name for c in conditions] == ["Hypertension"]
    systolic, diastolic, cohort_systolic, cohort_diastolic = observations
    assert systolic.reference_range == ReferenceRange(
        "mm[Hg]", 100.0, 139.0, "Synthea configuration (biometrics.yml, blood_pressure.normal)",
        "synthea-configuration",
    )
    assert systolic.reference_range.note == CONFIGURATION_NOTE
    assert diastolic.code == "8462-4" and diastolic.cohort is None
    assert cohort_systolic.cohort == CohortSpec("Hypertension", "point")
    assert cohort_diastolic.reference_range is None


def test_a_file_with_observations_only(tmp_path):
    path = write(tmp_path, {"observations": [{"code": "8480-6", "lookback_years": 3}]})
    conditions, (only,) = load_module_observations(path)
    assert conditions == ()
    assert only == ObservationDefinition("8480-6", lookback_years=3)


def test_a_file_without_observations():
    assert load_observations({"conditions": []}, origin="f", condition_names=()) == ()


@pytest.mark.parametrize(
    "item, message",
    [
        ({"name": "X"}, "needs a code"),
        ({"code": "http://loinc.org|8480-6"}, "no SYSTEM column"),
        ({"code": "1", "units": "x"}, "unknown key"),
        ({"code": "1", "name": " "}, "non-empty string"),
        ({"code": "1", "lookback_years": 0}, "lookback_years"),
        ({"code": "1", "lookback_years": True}, "lookback_years"),
        ({"code": "1", "lookback_years": 1.5}, "lookback_years"),
        ({"code": "1", "cohort": "Diabetes"}, "do not define"),
        ({"code": "1", "reference_range": {"low": 1}}, "units"),
        ({"code": "1", "reference_range": {"units": "mg"}}, "low or a high"),
        ({"code": "1", "reference_range": {"units": "mg", "low": 5, "high": 1}}, "above high"),
        ({"code": "1", "reference_range": {"units": "mg", "low": "5"}}, "not a number"),
        ({"code": "1", "reference_range": {"units": "mg", "low": True}}, "not a number"),
        ({"code": "1", "reference_range": {"units": "mg", "low": 1, "basis": "x"}}, "basis"),
        ({"code": "1", "reference_range": {"units": "mg", "low": 1, "max": 2}}, "unknown key"),
        ({"code": "1", "reference_range": {"units": "mg", "low": 1, "source": 3}}, "source"),
        ({"code": "1", "reference_range": [1, 2]}, "must be an object"),
        ("8480-6", "not an object"),
    ],
)
def test_malformed_observations_are_input_errors(tmp_path, item, message):
    path = write(tmp_path, {"observations": [item]})
    with pytest.raises(DefinitionError, match=message):
        load_module_observations(path)


def test_an_empty_or_malformed_list_is_an_input_error(tmp_path):
    for value in ([], {}, "x"):
        with pytest.raises(DefinitionError, match="non-empty list"):
            load_module_observations(write(tmp_path, {"observations": value}))


def test_a_repeated_name_is_an_input_error(tmp_path):
    items = [{"code": "1", "name": "A"}, {"code": "2", "name": "A"}]
    path = write(tmp_path, {"observations": items})
    with pytest.raises(DefinitionError, match="more than once"):
        load_module_observations(path)
    with pytest.raises(DefinitionError, match="more than once"):
        unique((ObservationDefinition("1"), ObservationDefinition("1")))


def test_an_unreadable_module_file(tmp_path):
    with pytest.raises(DefinitionError, match="could not be read"):
        load_module_observations(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(DefinitionError, match="not valid JSON"):
        load_module_observations(bad)
    with pytest.raises(DefinitionError, match="JSON object"):
        load_module_observations(write(tmp_path, [1]))


def test_a_one_sided_range():
    rng = ReferenceRange("mg/dL", high=200)
    assert rng.note is None
    assert rng.to_dict() == {
        "low": None, "high": 200.0, "units": "mg/dL", "source": None, "basis": None,
    }
    with pytest.raises(ValueError):
        ReferenceRange("mg/dL", low=float("inf"))
