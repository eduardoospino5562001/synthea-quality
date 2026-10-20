"""Tests for the index and the outcomes of a module file."""

from __future__ import annotations

import json

import pytest

from synthea_quality.indexed.definitions import (
    CrossSpec,
    DeathOutcome,
    IndexSpec,
    ValueOutcome,
    Window,
    load_crosses,
    load_indexed,
)
from synthea_quality.prevalence.definitions import DefinitionError
from synthea_quality.validate.models import load_module

NAMES = {"COVID-19", "Breast cancer"}

DEATH = {"name": "Death", "type": "death"}
STAGE = {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first"}
#: A file with a ``value`` outcome to cross and a ``death`` outcome to cross with.
BY_FILE = {"index": {"condition": "Breast cancer"}, "outcomes": [STAGE, DEATH]}


def load(data):
    return load_indexed(data, origin="f", condition_names=NAMES)


def load_by(data):
    """The crossings of a file, read after its outcomes."""
    _, outcomes = load(data)
    return load_crosses(data, origin="f", outcomes=outcomes)


def groups_file(groups):
    """A file with one ``value`` outcome that has the given groups."""
    return {"index": {"condition": "Breast cancer"},
            "outcomes": [{"name": "Stage group", "type": "value", "code": "21908-9",
                          "groups": groups}]}


def test_a_index_names_a_condition_the_file_does_not_define():
    data = {"index": {"condition": "Measles"}, "outcomes": [DEATH]}
    with pytest.raises(DefinitionError, match="do not define"):
        load(data)


def test_b_index_with_an_unknown_key():
    data = {"index": {"condition": "COVID-19", "rule": "first"}, "outcomes": [DEATH]}
    with pytest.raises(DefinitionError, match="unknown key.*'index'"):
        load(data)


def test_c_outcomes_without_index():
    with pytest.raises(DefinitionError, match="needs an 'index'"):
        load({"outcomes": [DEATH]})


def test_d_index_without_outcomes():
    with pytest.raises(DefinitionError, match="needs a non-empty 'outcomes'"):
        load({"index": {"condition": "COVID-19"}})


def test_e_empty_outcomes():
    with pytest.raises(DefinitionError, match="non-empty list"):
        load({"index": {"condition": "COVID-19"}, "outcomes": []})


def test_f_unknown_outcome_type():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Survival", "type": "survival"}]}
    with pytest.raises(DefinitionError, match="not 'survival'"):
        load(data)


def test_g_table_not_admitted():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Claim", "type": "event", "table": "claims",
                          "codes": ["1"]}]}
    with pytest.raises(DefinitionError, match="not 'claims'"):
        load(data)


def test_h_empty_codes():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Admission", "type": "event", "table": "encounters",
                          "codes": []}]}
    with pytest.raises(DefinitionError, match="non-empty 'codes'"):
        load(data)


def test_i_from_days_after_to_days():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Late", "type": "death",
                          "window": {"from_days": 5, "to_days": 2}}]}
    with pytest.raises(DefinitionError, match="after 'to_days'"):
        load(data)


def test_j_repeated_outcome_name():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [DEATH, {"name": "Death", "type": "death"}]}
    with pytest.raises(DefinitionError, match="more than once"):
        load(data)


def test_k_unknown_key_in_an_outcome():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Death", "type": "death", "cause": "x"}]}
    with pytest.raises(DefinitionError, match="unknown key"):
        load(data)


def test_l_invalid_pick():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Stage", "type": "value", "code": "21908-9",
                          "pick": "middle"}]}
    with pytest.raises(DefinitionError, match="pick must be one of"):
        load(data)


def test_m_from_days_is_not_an_integer():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Late", "type": "death",
                          "window": {"from_days": "0"}}]}
    with pytest.raises(DefinitionError, match="'from_days' must be an integer"):
        load(data)


def test_n_covid_example_has_four_outcomes_with_the_default_window():
    data = {
        "module": {"name": "COVID-19", "synthea_modules": ["covid19/infection.json"]},
        "conditions": [{"name": "COVID-19", "codes": ["840539006"]}],
        "index": {"condition": "COVID-19"},
        "outcomes": [
            {"name": "Hospital admission", "type": "event", "table": "encounters",
             "codes": ["1505002"]},
            {"name": "ICU admission", "type": "event", "table": "encounters",
             "codes": ["305351004"]},
            {"name": "Ventilated", "type": "event", "table": "devices",
             "codes": ["449071006"]},
            {"name": "Death", "type": "death"},
        ],
    }
    index, outcomes = load(data)
    assert index == IndexSpec("COVID-19")
    assert len(outcomes) == 4
    assert all(outcome.window == Window(0, None) for outcome in outcomes)


def test_o_breast_cancer_example_reads_value_and_death():
    data = {
        "conditions": [{"name": "Breast cancer", "codes": ["254837009"]}],
        "index": {"condition": "Breast cancer"},
        "outcomes": [
            {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first"},
            {"name": "Death", "type": "death",
             "cause_values": ["Malignant neoplasm of breast (disorder)"]},
        ],
    }
    index, outcomes = load(data)
    assert index == IndexSpec("Breast cancer")
    assert outcomes[0] == ValueOutcome("Stage group", "21908-9", "first", Window(0, None))
    assert outcomes[1] == DeathOutcome(
        "Death", ("Malignant neoplasm of breast (disorder)",), Window(0, None)
    )


def test_p_file_without_index_or_outcomes():
    assert load({"conditions": [{"name": "COVID-19", "codes": ["840539006"]}]}) == (None, ())


def test_q_negative_from_days_is_accepted():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Before", "type": "death",
                          "window": {"from_days": -30}}]}
    _, (outcome,) = load(data)
    assert outcome.window == Window(-30, None)


def test_r_load_module_reads_index_and_outcomes(tmp_path):
    data = {
        "module": {"name": "COVID-19", "synthea_modules": ["covid19/infection.json"]},
        "conditions": [{"name": "COVID-19", "codes": ["840539006"]}],
        "index": {"condition": "COVID-19"},
        "outcomes": [
            {"name": "Hospital admission", "type": "event", "table": "encounters",
             "codes": ["1505002"]},
            {"name": "ICU admission", "type": "event", "table": "encounters",
             "codes": ["305351004"]},
            {"name": "Ventilated", "type": "event", "table": "devices",
             "codes": ["449071006"]},
            {"name": "Death", "type": "death"},
        ],
    }
    path = tmp_path / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    module = load_module(path)
    assert module.index == IndexSpec("COVID-19")
    assert len(module.outcomes) == 4


def test_s_load_module_rejects_outcomes_without_index(tmp_path):
    data = {"conditions": [{"name": "COVID-19", "codes": ["840539006"]}],
            "outcomes": [{"name": "Death", "type": "death"}]}
    path = tmp_path / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DefinitionError, match="needs an 'index'"):
        load_module(path)


def test_repeated_code_in_an_outcome_is_rejected():
    data = {"index": {"condition": "COVID-19"},
            "outcomes": [{"name": "Admission", "type": "event", "table": "encounters",
                          "codes": ["1505002", "1505002"]}]}
    with pytest.raises(DefinitionError, match="repeats code '1505002'"):
        load(data)


def test_groups_must_be_an_object():
    with pytest.raises(DefinitionError, match="must be a non-empty object"):
        load(groups_file(["Stage 1"]))


def test_empty_groups_are_rejected():
    with pytest.raises(DefinitionError, match="must be a non-empty object"):
        load(groups_file({}))


def test_a_group_label_cannot_be_empty():
    with pytest.raises(DefinitionError, match="needs a non-empty label"):
        load(groups_file({"   ": ["Stage 1 (qualifier value)"]}))


def test_a_group_cannot_use_the_no_value_label():
    with pytest.raises(DefinitionError, match="cannot use the label 'no value'"):
        load(groups_file({"no value": ["Stage 1 (qualifier value)"]}))


def test_a_group_needs_a_non_empty_list_of_texts():
    with pytest.raises(DefinitionError, match="needs a non-empty list of texts"):
        load(groups_file({"Stage 1": []}))


def test_group_texts_must_be_non_empty_strings():
    with pytest.raises(DefinitionError, match="texts must be non-empty strings"):
        load(groups_file({"Stage 1": [""]}))
    with pytest.raises(DefinitionError, match="texts must be non-empty strings"):
        load(groups_file({"Stage 1": ["Stage 1 (qualifier value)", 7]}))


def test_a_group_cannot_repeat_a_text():
    groups = {"Stage 1": ["Stage 1 (qualifier value)", "Stage 1 (qualifier value)"]}
    with pytest.raises(DefinitionError, match="repeats the text 'Stage 1"):
        load(groups_file(groups))


def test_a_text_cannot_be_in_two_groups():
    groups = {"Stage 1": ["Stage 1 (qualifier value)"],
              "Stage 2": ["Stage 1 (qualifier value)"]}
    with pytest.raises(DefinitionError, match="in more than one group"):
        load(groups_file(groups))


def test_a_group_label_cannot_be_repeated():
    groups = {"Stage 1": ["Stage 1 (qualifier value)"],
              " Stage 1 ": ["Stage 1A (qualifier value)"]}
    with pytest.raises(DefinitionError, match="repeats the label 'Stage 1'"):
        load(groups_file(groups))


def test_a_group_label_cannot_be_a_text_of_another_group():
    groups = {"B": ["A"], "A": ["C"]}
    with pytest.raises(DefinitionError, match="uses the label 'A' as a text of group 'B'"):
        load(groups_file(groups))


def test_a_group_label_cannot_be_a_text_of_a_later_group():
    groups = {"A": ["C"], "B": ["A"]}
    with pytest.raises(DefinitionError, match="uses the label 'A' as a text of group 'B'"):
        load(groups_file(groups))


def test_a_group_label_can_be_a_text_of_its_own_group():
    groups = {"Stage 2": ["Stage 2", "Stage 2A (qualifier value)"]}
    _, (outcome,) = load(groups_file(groups))
    assert outcome == ValueOutcome(
        "Stage group",
        "21908-9",
        "first",
        Window(0, None),
        (("Stage 2", ("Stage 2", "Stage 2A (qualifier value)")),),
    )


def test_groups_are_read_in_the_order_of_the_file_and_stripped():
    groups = {
        " Stage 1 ": ["Stage 1 (qualifier value) ",
                      "Stage 1A (qualifier value)"],
        "Stage 2": ["Stage 2 (qualifier value)"],
    }
    _, (outcome,) = load(groups_file(groups))
    assert outcome == ValueOutcome(
        "Stage group",
        "21908-9",
        "first",
        Window(0, None),
        (("Stage 1", ("Stage 1 (qualifier value)", "Stage 1A (qualifier value)")),
         ("Stage 2", ("Stage 2 (qualifier value)",))),
    )


def test_by_needs_an_index():
    with pytest.raises(DefinitionError, match="needs an 'index'"):
        load_crosses(
            {"by": [{"outcome": "Death", "by": "Stage group"}]}, origin="f", outcomes=()
        )


def test_by_must_be_a_list():
    with pytest.raises(DefinitionError, match="must be a non-empty list"):
        load_by({**BY_FILE, "by": {"outcome": "Death", "by": "Stage group"}})


def test_by_must_not_be_empty():
    with pytest.raises(DefinitionError, match="must be a non-empty list"):
        load_by({**BY_FILE, "by": []})


def test_a_by_entry_must_be_an_object():
    with pytest.raises(DefinitionError, match="is not an object"):
        load_by({**BY_FILE, "by": ["Death"]})


def test_a_by_entry_rejects_unknown_keys():
    entry = {"outcome": "Death", "by": "Stage group", "label": "x"}
    with pytest.raises(DefinitionError, match="unknown key"):
        load_by({**BY_FILE, "by": [entry]})


def test_a_by_entry_needs_both_names():
    with pytest.raises(DefinitionError, match="needs the 'outcome' name"):
        load_by({**BY_FILE, "by": [{"by": "Stage group"}]})
    with pytest.raises(DefinitionError, match="needs the 'by' name"):
        load_by({**BY_FILE, "by": [{"outcome": "Death"}]})


def test_by_names_an_outcome_the_file_does_not_define():
    with pytest.raises(DefinitionError, match="'Survival' is not an outcome of the file"):
        load_by({**BY_FILE, "by": [{"outcome": "Survival", "by": "Stage group"}]})


def test_an_outcome_cannot_be_crossed_with_itself():
    with pytest.raises(DefinitionError, match="cannot be crossed with itself"):
        load_by({**BY_FILE, "by": [{"outcome": "Stage group", "by": "Stage group"}]})


def test_a_death_outcome_cannot_give_the_strata():
    with pytest.raises(DefinitionError, match="need a 'value' or an 'event'"):
        load_by({**BY_FILE, "by": [{"outcome": "Stage group", "by": "Death"}]})


def test_the_same_pair_cannot_be_crossed_twice():
    entry = {"outcome": "Death", "by": "Stage group"}
    with pytest.raises(DefinitionError, match="crossed more than once"):
        load_by({**BY_FILE, "by": [entry, entry]})


def test_by_is_read():
    data = {**BY_FILE, "by": [{"outcome": "Death", "by": "Stage group"}]}
    assert load_by(data) == (CrossSpec("Death", "Stage group"),)


def test_load_module_reads_groups_and_crosses(tmp_path):
    data = {
        "conditions": [{"name": "Breast cancer", "codes": ["254837009"]}],
        "index": {"condition": "Breast cancer"},
        "outcomes": [
            {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first",
             "groups": {"Stage 1": ["Stage 1 (qualifier value)"]}},
            {"name": "Death", "type": "death",
             "cause_values": ["Malignant neoplasm of breast (disorder)"]},
        ],
        "by": [{"outcome": "Death", "by": "Stage group"}],
    }
    path = tmp_path / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    module = load_module(path)
    assert module.crosses == (CrossSpec("Death", "Stage group"),)
    assert module.outcomes[0].groups == (("Stage 1", ("Stage 1 (qualifier value)",)),)


def test_a_file_without_by_or_groups_has_neither(tmp_path):
    data = {
        "conditions": [{"name": "Breast cancer", "codes": ["254837009"]}],
        "index": {"condition": "Breast cancer"},
        "outcomes": [
            {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first"},
            {"name": "Death", "type": "death"},
        ],
    }
    path = tmp_path / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    module = load_module(path)
    assert module.crosses == ()
    assert module.outcomes[0].groups == ()
