"""Tests for the module file and the module validation result."""

from __future__ import annotations

import json

import pytest

from synthea_quality.prevalence.definitions import DefinitionError
from synthea_quality.prevalence.models import Expected
from synthea_quality.validate.models import ModuleInfo, load_module_file


def write(tmp_path, data) -> str:
    path = tmp_path / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_module_file_with_all_three_measures(tmp_path):
    module, (mi,) = load_module_file(
        write(
            tmp_path,
            {
                "module": {"name": "MI", "synthea_modules": ["myocardial_infarction.json"]},
                "conditions": [
                    {
                        "name": "Myocardial infarction",
                        "codes": ["22298006"],
                        "acute": True,
                        "expected": {"point": 0.01, "lifetime": 0.03, "incidence": 2.5,
                                     "source": "illustrative"},
                    }
                ],
            },
        )
    )
    assert module == ModuleInfo("MI", ("myocardial_infarction.json",), None)
    assert mi.acute
    assert [e.measure for e in mi.expected] == ["point", "lifetime", "incidence"]
    assert mi.expected[2] == Expected("incidence", 2.5, "illustrative")


def test_the_module_block_is_optional(tmp_path):
    module, definitions = load_module_file(
        write(tmp_path, {"conditions": [{"name": "HTN", "codes": ["59621000"]}]})
    )
    assert module == ModuleInfo() and len(definitions) == 1


@pytest.mark.parametrize(
    "data, message",
    [
        ({"module": [], "conditions": [{"name": "A", "codes": ["1"]}]}, "must be an object"),
        ({"module": {"owner": "x"}, "conditions": [{"name": "A", "codes": ["1"]}]}, "unknown key"),
        ({"module": {"synthea_modules": "a.json"}, "conditions": [{"name": "A", "codes": ["1"]}]},
         "list of strings"),
        ({"conditions": [{"name": "A", "codes": ["1"]}, {"name": "A", "codes": ["2"]}]},
         "more than once"),
        ({"conditions": [{"name": "A", "codes": ["1"], "expected": {"incidence": -1}}]}, ">= 0"),
    ],
)
def test_malformed_module_files_are_input_errors(tmp_path, data, message):
    with pytest.raises(DefinitionError, match=message):
        load_module_file(write(tmp_path, data))


def test_a_module_file_with_observations(tmp_path):
    from synthea_quality.validate.models import load_module

    loaded = load_module(
        write(
            tmp_path,
            {
                "conditions": [{"name": "Hypertension", "codes": ["59621000"]}],
                "observations": [
                    {"code": "8480-6", "cohort": "Hypertension"},
                    {"code": "8462-4"},
                ],
            },
        )
    )
    assert [c.name for c in loaded.conditions] == ["Hypertension"]
    assert [o.code for o in loaded.observations] == ["8480-6", "8462-4"]
    assert loaded.observations[0].cohort.condition == "Hypertension"


def test_a_module_file_may_have_observations_only(tmp_path):
    from synthea_quality.validate.models import load_module

    loaded = load_module(write(tmp_path, {"observations": [{"code": "8480-6"}]}))
    assert loaded.conditions == () and len(loaded.observations) == 1
    assert loaded.module == ModuleInfo()


def test_a_module_file_needs_conditions_or_observations(tmp_path):
    with pytest.raises(DefinitionError, match="'conditions', 'observations' or 'medications'"):
        load_module_file(write(tmp_path, {"module": {"name": "X"}}))
    with pytest.raises(DefinitionError, match="do not define"):
        load_module_file(write(tmp_path, {"observations": [{"code": "1", "cohort": "X"}]}))


def test_a_module_file_with_medications(tmp_path):
    from synthea_quality.validate.models import load_module

    loaded = load_module(
        write(
            tmp_path,
            {
                "conditions": [{"name": "Hypertension", "codes": ["59621000"]}],
                "medications": [{"name": "Lisinopril", "codes": ["314076"],
                                 "cohort": "Hypertension", "expected": {"active": 0.8}}],
            },
        )
    )
    (lisinopril,) = loaded.medications
    assert lisinopril.cohort.condition == "Hypertension"
    assert [e.measure for e in lisinopril.expected] == ["active"]
    only = load_module(write(tmp_path, {"medications": [{"name": "A", "codes": ["1"]}]}))
    assert only.conditions == () and len(only.medications) == 1
    with pytest.raises(DefinitionError, match="do not define"):
        load_module(write(tmp_path, {"medications": [{"name": "A", "codes": ["1"],
                                                       "cohort": "X"}]}))


def test_a_condition_never_takes_a_medication_measure(tmp_path):
    with pytest.raises(DefinitionError, match="unknown key"):
        load_module_file(write(tmp_path, {"conditions": [
            {"name": "A", "codes": ["1"], "expected": {"active": 0.5}}]}))
