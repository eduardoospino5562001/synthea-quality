"""Tests for the index and the outcomes of a module file."""

from __future__ import annotations

import pytest

from synthea_quality.indexed.definitions import (
    DeathOutcome,
    IndexSpec,
    ValueOutcome,
    Window,
    load_indexed,
)
from synthea_quality.prevalence.definitions import DefinitionError

NAMES = {"COVID-19", "Breast cancer"}

DEATH = {"name": "Death", "type": "death"}


def load(data):
    return load_indexed(data, origin="f", condition_names=NAMES)


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
