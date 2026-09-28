"""Tests for how the conditions asked for are given."""

from __future__ import annotations

import json

import pytest

from synthea_quality.prevalence.definitions import (
    DefinitionError,
    assemble,
    load_conditions_file,
    parse_condition_option,
    parse_expected_option,
)
from synthea_quality.prevalence.models import SNOMED_CT, CodeRef, Expected


def test_condition_option_groups_codes_under_a_name_with_snomed_by_default():
    definition = parse_condition_option("Myocardial infarction=22298006, 401303003,401314000")
    assert definition.name == "Myocardial infarction"
    assert definition.codes == (
        CodeRef("22298006", SNOMED_CT),
        CodeRef("401303003", SNOMED_CT),
        CodeRef("401314000", SNOMED_CT),
    )


def test_condition_option_accepts_an_explicit_system_and_drops_repeated_codes():
    definition = parse_condition_option("X=SNOMED-CT|1,http://loinc.org|2,1")
    assert definition.codes == (CodeRef("1", SNOMED_CT), CodeRef("2", "http://loinc.org"))


@pytest.mark.parametrize("text", ["no equals sign", "=123", "Name=", "Name= , "])
def test_malformed_condition_options_are_input_errors(text):
    with pytest.raises(DefinitionError):
        parse_condition_option(text)


def test_conditions_file_with_expected_values_and_code_objects(tmp_path):
    path = tmp_path / "mi.json"
    path.write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "name": "Myocardial infarction",
                        "codes": ["22298006", {"system": "SNOMED-CT", "code": "401303003"}],
                        "expected": {"lifetime": 0.03, "point": 0.01, "source": "CDC 2019"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (definition,) = load_conditions_file(path)
    assert definition.codes == (CodeRef("22298006", SNOMED_CT), CodeRef("401303003", SNOMED_CT))
    assert definition.expected == (
        Expected("point", 0.01, "CDC 2019"),
        Expected("lifetime", 0.03, "CDC 2019"),
    )


@pytest.mark.parametrize(
    "content, message",
    [
        ("{", "not valid JSON"),
        ('{"conditions": []}', "non-empty"),
        ('{"conditions": [{"name": "X"}]}', "at least one code"),
        ('{"conditions": [{"codes": ["1"]}]}', "needs a name"),
        ('{"conditions": [{"name": "X", "codes": ["1"], "expected": {"incidence": 0.1}}]}',
         "unknown key"),
        ('{"conditions": [{"name": "X", "codes": ["1"], "expected": {"point": 2}}]}', "[0, 1]"),
    ],
)
def test_malformed_conditions_files_are_input_errors(tmp_path, content, message):
    path = tmp_path / "c.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DefinitionError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_conditions_file(path)


def test_expected_option():
    assert parse_expected_option("Myocardial infarction:lifetime=0.03") == (
        "Myocardial infarction",
        Expected("lifetime", 0.03),
    )
    for bad in ("MI=0.03", "MI:incidence=0.1", "MI:point=abc", ":point=0.1", "MI:point=1.2"):
        with pytest.raises(DefinitionError):
            parse_expected_option(bad)


def test_assemble_keeps_order_and_attaches_expected_values(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"conditions": [{"name": "A", "codes": ["1"]}]}), "utf-8")
    definitions = assemble(["B=2"], path, ["B:point=0.5", "A:lifetime=0.1"])
    assert [d.name for d in definitions] == ["A", "B"]
    assert definitions[1].expected == (Expected("point", 0.5),)
    assert definitions[0].expected == (Expected("lifetime", 0.1),)


def test_assemble_rejects_ambiguity():
    with pytest.raises(DefinitionError, match="more than once"):
        assemble(["A=1", "A=2"])
    with pytest.raises(DefinitionError, match="no --condition"):
        assemble(["A=1"], None, ["B:point=0.1"])
    with pytest.raises(DefinitionError, match="more than one expected point"):
        assemble(["A=1"], None, ["A:point=0.1", "A:point=0.2"])
    assert assemble() == ()


def test_acute_is_declared_with_a_suffix_or_a_json_field(tmp_path):
    assert parse_condition_option("MI=22298006,401303003;acute").acute is True
    assert parse_condition_option("MI=22298006; ACUTE ").acute is True
    plain = parse_condition_option("MI=22298006")
    assert plain.acute is False and plain.codes == (CodeRef("22298006", SNOMED_CT),)
    with pytest.raises(DefinitionError, match="only the ';acute' suffix"):
        parse_condition_option("MI=22298006;chronic")
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"conditions": [
        {"name": "MI", "codes": ["1"], "acute": True},
        {"name": "HTN", "codes": ["2"]},
    ]}), "utf-8")
    mi, htn = load_conditions_file(path)
    assert (mi.acute, htn.acute) == (True, False)
    path.write_text(json.dumps({"conditions": [{"name": "MI", "codes": ["1"], "acute": "yes"}]}),
                    "utf-8")
    with pytest.raises(DefinitionError, match="'acute' must be true or false"):
        load_conditions_file(path)


def test_expected_values_keep_the_acute_declaration():
    (mi,) = assemble(["MI=1;acute"], None, ["MI:lifetime=0.1"])
    assert mi.acute and mi.expected == (Expected("lifetime", 0.1),)


def test_each_report_accepts_only_its_own_measures(tmp_path):
    with pytest.raises(DefinitionError, match="measure must be one of"):
        assemble(["MI=1"], None, ["MI:incidence=2.5"])
    (mi,) = assemble(["MI=1;acute"], None, ["MI:incidence=2.5"], measures=("incidence",))
    assert mi.expected == (Expected("incidence", 2.5),) and mi.acute
    with pytest.raises(DefinitionError, match="measure must be one of"):
        assemble(["MI=1"], None, ["MI:point=0.1"], measures=("incidence",))
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"conditions": [
        {"name": "MI", "codes": ["1"], "expected": {"incidence": 40, "source": "CDC"}}
    ]}), "utf-8")
    (from_file,) = load_conditions_file(path, measures=("incidence",))
    assert from_file.expected == (Expected("incidence", 40.0, "CDC"),)
    with pytest.raises(DefinitionError, match="unknown key"):
        load_conditions_file(path)
