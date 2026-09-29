"""Tests for the Markdown and JSON renderings of a module validation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from synthea_quality.prevalence.definitions import assemble
from synthea_quality.prevalence.models import ALL_MEASURES
from synthea_quality.schema.tables import tables_by_name
from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.models import ModuleInfo
from synthea_quality.validate.render import dumps, render_markdown, write_json, write_markdown

SNOMED = "http://snomed.info/sct"


def write_table(directory: Path, name: str, rows) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def report(tmp_path):
    write_table(
        tmp_path,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-06-01", "GENDER": "F"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "M"},
        ],
    )
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        tmp_path,
        "conditions",
        [
            {"PATIENT": "a1", "CODE": "22298006", "SYSTEM": SNOMED, "START": "2023-01-01",
             "STOP": "2023-01-08"},
            {"PATIENT": "d1", "CODE": "22298006", "SYSTEM": SNOMED, "START": "2022-06-01",
             "STOP": "2022-06-05"},
            {"PATIENT": "a2", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2015-01-01"},
        ],
    )
    definitions = assemble(
        ["Myocardial infarction=22298006,401314000;acute", "Hypertension=59621000"],
        None,
        ["Myocardial infarction:lifetime=0.3", "Myocardial infarction:incidence=100"],
        measures=ALL_MEASURES,
    )
    return build_module_validation(
        tmp_path,
        module=ModuleInfo("Cardio", ("myocardial_infarction.json",), "A test module."),
        definitions=definitions,
        module_file="examples/cardio.json",
        generated_at="2026-09-29T00:00:00+00:00",
    )


def test_sections_come_in_reading_order(report):
    text = render_markdown(report)
    order = [
        "# Module validation: Cardio",
        "## Scope",
        "## Reference date",
        "## Definitions and populations",
        "## Population summary",
        "## Summary",
        "## Reference values",
        "## Condition: Myocardial infarction (declared acute)",
        "### Prevalence: Myocardial infarction (declared acute)",
        "### Incidence: Myocardial infarction (declared acute)",
        "## Condition: Hypertension",
    ]
    positions = [text.index(title) for title in order]
    assert positions == sorted(positions)
    assert "- Synthea modules: `myocardial_infarction.json`" in text
    assert "- Description: A test module." in text


def test_populations_are_stated_for_each_measure(report):
    text = render_markdown(report)
    assert "among the patients **alive at the end of the simulation**" in text
    assert "following **every patient, deceased included, until their death**" in text
    assert "survivor bias" in text
    # this command has no --alive-only: the note points to the one that does
    assert "`synthea-incidence --alive-only` measures that cohort instead" in text
    assert "Use --alive-only" not in text
    assert (
        "Alive at the end: **2** (the prevalence denominator). Followed for incidence: **3**."
    ) in text
    assert "### Population" in text and "### Age at the reference date" in text


def test_reference_values_are_in_one_table_and_not_repeated(report):
    text = render_markdown(report)
    assert "| Myocardial infarction | lifetime prevalence | 30.00% | 50.00% |" in text
    assert "| Myocardial infarction | incidence per 1,000 PY | 100.00 |" in text
    assert text.count("Reference values") == 1
    assert not re.search(r"\b(pass(ed|es)?|fail(ed|s)?|ok|match(ed|es)?)\b", text, re.I)


def test_summary_table_has_every_condition(report):
    text = render_markdown(report)
    assert "| Myocardial infarction (acute) | 0.00% (0.00%–65.76%) | 50.00% (" in text
    assert "| Hypertension | 50.00% (" in text


def test_json_and_writers(report, tmp_path):
    data = json.loads(dumps(report))
    assert data["module"]["name"] == "Cardio"
    assert data["conditions"][0]["incidence"]["rate"]["events"] == 2
    assert write_json(report, tmp_path / "o" / "v.json").read_text("utf-8").endswith("\n")
    assert write_markdown(report, tmp_path / "o" / "v.md").read_text("utf-8").startswith("# ")


def test_observations_follow_the_conditions(tmp_path):
    from synthea_quality.observations.definitions import CONFIGURATION_NOTE
    from synthea_quality.validate.models import load_module

    write_table(tmp_path, "patients", [{"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"}])
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(tmp_path, "conditions", [
        {"PATIENT": "a1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2015-01-01"},
    ])
    write_table(tmp_path, "observations", [
        {"DATE": "2026-01-01T00:00:00Z", "PATIENT": "a1", "CODE": code, "VALUE": value,
         "UNITS": "mm[Hg]", "TYPE": "numeric"}
        for code, value in (("8480-6", "150"), ("8462-4", "95"))
    ])
    example = Path(__file__).resolve().parents[1] / "examples" / "hypertension.json"
    loaded = load_module(example)
    report = build_module_validation(
        tmp_path, module=loaded.module, definitions=loaded.conditions, module_file=example,
        observations=loaded.observations, generated_at="2026-09-29T00:00:00+00:00",
    )
    text = render_markdown(report)
    summary = text.index("## Observation values")
    assert text.index("## Condition: Hypertension") < summary
    assert (
        "| Systolic blood pressure | `8480-6` | alive | `mm[Hg]` | 1 | 150 | — | 0 / 0 / 1 |"
    ) in text
    assert (
        "| Diastolic blood pressure, patients with hypertension | `8462-4` | Hypertension |"
    ) in text
    assert "### How values are described" in text
    assert "Percentiles are not shown below 10 patients" in text
    assert "## Observation: Systolic blood pressure\n" in text
    assert text.count(f"> {CONFIGURATION_NOTE}") == 2
    data = json.loads(dumps(report))
    assert data["observations"][0]["groups"][0]["reference_range"]["above"] == 1
    assert "type 7" in data["observation_definitions"]["percentiles"]


def test_medications_and_their_reference_values(tmp_path):
    from synthea_quality.condition_cohort import CohortSpec
    from synthea_quality.medications.definitions import MedicationDefinition
    from synthea_quality.prevalence.models import Expected

    write_table(tmp_path, "patients", [{"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"}])
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(tmp_path, "conditions", [
        {"PATIENT": "a1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2015-01-01"},
    ])
    write_table(tmp_path, "medications", [
        {"START": "2020-01-01T00:00:00Z", "PATIENT": "a1", "CODE": "314076",
         "REASONCODE": "59621000"},
    ])
    report = build_module_validation(
        tmp_path, module=ModuleInfo("Hypertension"),
        definitions=assemble(("Hypertension=59621000",), None, (), measures=ALL_MEASURES),
        module_file="m.json",
        medications=(MedicationDefinition(
            "Lisinopril", ("314076",), CohortSpec("Hypertension"),
            (Expected("active", 0.9, "Illustrative"),),
        ),),
        generated_at="2026-09-29T00:00:00+00:00",
    )
    text = render_markdown(report)
    assert "It also gives the share of a population that takes each" in text
    assert text.index("## Condition: Hypertension") < text.index("## Medications")
    assert "| Condition or medication | Measure |" in text
    assert "| Lisinopril | active share | 90.00% | 100.00% |" in text
    assert "| Lisinopril | `314076` | Hypertension | 1 | 1 (100.00%; 20.65–100.00%) |" in text
