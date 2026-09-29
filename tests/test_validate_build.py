"""Tests for building a module validation report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from synthea_quality.incidence.build import build_incidence
from synthea_quality.prevalence.build import build_prevalence
from synthea_quality.prevalence.definitions import assemble
from synthea_quality.prevalence.models import ALL_MEASURES
from synthea_quality.profile.models import SectionStatus
from synthea_quality.schema.tables import tables_by_name
from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.models import ModuleInfo

SNOMED = "http://snomed.info/sct"
GENERATED_AT = "2026-09-29T00:00:00+00:00"


def write_table(directory: Path, name: str, rows, columns=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-06-01", "GENDER": "F"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "M"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "a1", "CODE": "22298006", "SYSTEM": SNOMED, "START": "2023-01-01",
             "STOP": "2023-01-08", "DESCRIPTION": "Myocardial infarction (disorder)"},
            {"PATIENT": "a1", "CODE": "401314000", "SYSTEM": SNOMED, "START": "2023-01-01",
             "DESCRIPTION": "NSTEMI (disorder)"},
            {"PATIENT": "d1", "CODE": "22298006", "SYSTEM": SNOMED, "START": "2022-06-01",
             "STOP": "2022-06-05", "DESCRIPTION": "Myocardial infarction (disorder)"},
            {"PATIENT": "a2", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2015-01-01",
             "DESCRIPTION": "Essential hypertension (disorder)"},
        ],
    )
    return directory


DEFINITIONS = assemble(
    ["Myocardial infarction=22298006,401314000;acute", "Hypertension=59621000"],
    None,
    ["Myocardial infarction:lifetime=0.3", "Myocardial infarction:incidence=100",
     "Hypertension:point=0.5"],
    measures=ALL_MEASURES,
)
MODULE = ModuleInfo("Cardio", ("myocardial_infarction.json",))


def validate(directory, **kwargs):
    return build_module_validation(
        directory, module=MODULE, definitions=DEFINITIONS, module_file="module.json",
        generated_at=GENERATED_AT, **kwargs,
    )


def test_numbers_are_identical_to_the_dedicated_commands(tmp_path):
    directory = dataset(tmp_path)
    report = validate(directory)
    prev = build_prevalence(directory, definitions=DEFINITIONS)
    inc = build_incidence(directory, definitions=DEFINITIONS)
    for mine, theirs in zip(report.conditions, prev.conditions):
        assert mine.prevalence.to_dict() == theirs.to_dict()
    for mine, theirs in zip(report.conditions, inc.conditions):
        assert mine.incidence.to_dict() == theirs.to_dict()
    assert report.alive == prev.alive
    assert report.incidence_window == inc.window
    assert report.cohort == inc.cohort


def test_every_reference_value_is_placed_with_its_own_measure(tmp_path):
    report = validate(dataset(tmp_path))
    mi, htn = report.conditions
    assert [r.expected.measure for r in mi.references] == ["lifetime", "incidence"]
    lifetime, incidence = mi.references
    assert lifetime.observed is mi.prevalence.lifetime and lifetime.position == "inside"
    assert incidence.observed is mi.incidence.rate
    assert incidence.position in ("inside", "outside")
    assert [r.expected.measure for r in htn.references] == ["point"]
    assert mi.acute and mi.prevalence.acute and mi.incidence.acute


def test_population_summary_is_the_profile_sections(tmp_path):
    report = validate(dataset(tmp_path))
    assert [s.section_id for s in report.population] == [
        "population", "age", "distribution.GENDER",
    ]
    assert report.population[0].metrics["alive"] == 2


def test_json_is_deterministic_and_embeds_both_results(tmp_path):
    directory = dataset(tmp_path)
    text = validate(directory).to_json()
    assert text == validate(directory).to_json()
    data = json.loads(text)
    assert data["schema_version"] == 1
    assert data["module"]["synthea_modules"] == ["myocardial_infarction.json"]
    assert set(data["conditions"][0]) == {"name", "codes", "acute", "prevalence", "incidence"}


def test_without_stop_only_prevalence_is_skipped(tmp_path):
    directory = dataset(tmp_path)
    write_table(
        directory, "conditions",
        [{"PATIENT": "a1", "CODE": "22298006", "START": "2023-01-01"}],
        columns=("PATIENT", "CODE", "START"),
    )
    mi = validate(directory).conditions[0]
    assert mi.prevalence.status is SectionStatus.SKIPPED and "STOP" in mi.prevalence.reason
    assert mi.incidence.status is SectionStatus.COMPUTED


@pytest.mark.parametrize("remove", ["patients.csv", "conditions.csv", "encounters.csv"])
def test_a_missing_input_skips_everything_with_the_reason(tmp_path, remove):
    directory = dataset(tmp_path)
    (directory / remove).unlink()
    report = validate(directory)
    for condition in report.conditions:
        assert condition.prevalence.status is SectionStatus.SKIPPED
        assert condition.incidence.status is SectionStatus.SKIPPED
    assert all(s.status is SectionStatus.SKIPPED for s in report.population)


def test_invalid_arguments(tmp_path):
    with pytest.raises(ValueError):
        validate(dataset(tmp_path), window_years=0)
    with pytest.raises(ValueError):
        validate(dataset(tmp_path), age_bands=(2,))
