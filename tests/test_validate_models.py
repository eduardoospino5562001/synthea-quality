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
