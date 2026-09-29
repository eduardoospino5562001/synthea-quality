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


@pytest.mark.parametrize("remove", ["patients.csv", "encounters.csv"])
def test_a_missing_input_skips_everything_with_the_reason(tmp_path, remove):
    directory = dataset(tmp_path)
    (directory / remove).unlink()
    report = validate(directory)
    for condition in report.conditions:
        assert condition.prevalence.status is SectionStatus.SKIPPED
        assert condition.incidence.status is SectionStatus.SKIPPED
    assert all(s.status is SectionStatus.SKIPPED for s in report.population)


def test_missing_conditions_skip_the_conditions_but_not_the_population(tmp_path):
    directory = dataset(tmp_path)
    (directory / "conditions.csv").unlink()
    report = validate(directory)
    for condition in report.conditions:
        assert condition.prevalence.status is SectionStatus.SKIPPED
        assert "conditions.csv" in condition.prevalence.reason
        assert condition.incidence.status is SectionStatus.SKIPPED
    assert report.alive == 2
    assert all(s.status is SectionStatus.COMPUTED for s in report.population)


def test_invalid_arguments(tmp_path):
    with pytest.raises(ValueError):
        validate(dataset(tmp_path), window_years=0)
    with pytest.raises(ValueError):
        validate(dataset(tmp_path), age_bands=(2,))


# --------------------------------------------------------------------------- #
# observations of the module file
# --------------------------------------------------------------------------- #


def with_observations(directory: Path) -> Path:
    write_table(
        directory,
        "observations",
        [
            {"DATE": "2026-01-01T00:00:00Z", "PATIENT": patient, "CODE": "8480-6",
             "VALUE": value, "UNITS": "mm[Hg]", "TYPE": "numeric"}
            for patient, value in (("a1", "150"), ("a2", "120"), ("d1", "180"))
        ],
    )
    return directory


def test_observations_use_the_same_functions_as_synthea_observations(tmp_path):
    from synthea_quality.condition_cohort import CohortSpec
    from synthea_quality.observations.build import build_observations
    from synthea_quality.observations.definitions import ObservationDefinition

    directory = with_observations(dataset(tmp_path))
    definitions = assemble(("Hypertension=59621000",), None, (), measures=ALL_MEASURES)
    observations = (
        ObservationDefinition("8480-6"),
        ObservationDefinition("8480-6", name="HTN", cohort=CohortSpec("Hypertension")),
    )
    report = build_module_validation(
        directory, module=ModuleInfo(), definitions=definitions, module_file="m.json",
        observations=observations, generated_at=GENERATED_AT,
    )
    alone = build_observations(
        directory, observations=observations, conditions=definitions, generated_at=GENERATED_AT
    )
    assert [o.to_dict() for o in report.observations] == [o.to_dict() for o in alone.observations]
    everyone, cohort = report.observations
    assert everyone.groups[0].summary.n == 2
    assert (cohort.population, cohort.groups[0].summary.median) == (1, 120)
    assert "observations" in [i.table for i in report.inputs]


def test_without_observations_the_table_is_not_read(tmp_path):
    report = validate(with_observations(dataset(tmp_path)))
    assert report.observations == ()
    assert "observations" not in [i.table for i in report.inputs]
    assert report.to_dict()["observation_definitions"] is None


def test_missing_observations_skip_only_the_observations(tmp_path):
    from synthea_quality.observations.definitions import ObservationDefinition

    directory = dataset(tmp_path)
    report = build_module_validation(
        directory, module=ModuleInfo(), definitions=DEFINITIONS, module_file="m.json",
        observations=(ObservationDefinition("8480-6"),), generated_at=GENERATED_AT,
    )
    assert report.observations[0].status is SectionStatus.SKIPPED
    assert "observations.csv is not in the dataset" in report.observations[0].reason
    assert report.conditions[0].prevalence.status is SectionStatus.COMPUTED


def test_an_observation_only_module_without_conditions_csv(tmp_path):
    from synthea_quality.observations.definitions import ObservationDefinition

    directory = with_observations(dataset(tmp_path))
    (directory / "conditions.csv").unlink()
    report = build_module_validation(
        directory, module=ModuleInfo(), definitions=(), module_file="m.json",
        observations=(ObservationDefinition("8480-6"),), generated_at=GENERATED_AT,
    )
    assert report.observations[0].status is SectionStatus.COMPUTED
    assert report.alive == 2


# --------------------------------------------------------------------------- #
# medications of the module file
# --------------------------------------------------------------------------- #


def with_medications(directory: Path) -> Path:
    write_table(
        directory,
        "medications",
        [
            {"START": "2020-01-01T00:00:00Z", "PATIENT": "a2", "CODE": "314076",
             "REASONCODE": "59621000", "DESCRIPTION": "lisinopril 10 MG Oral Tablet"},
            {"START": "2020-01-01T00:00:00Z", "STOP": "2021-01-01T00:00:00Z", "PATIENT": "a1",
             "CODE": "314076"},
        ],
    )
    return directory


def medication_report(directory, medications, definitions=None):
    return build_module_validation(
        directory, module=ModuleInfo(),
        definitions=definitions if definitions is not None else assemble(
            ("Hypertension=59621000",), None, (), measures=ALL_MEASURES
        ),
        module_file="m.json", medications=medications, generated_at=GENERATED_AT,
    )


def test_medications_among_a_cohort_and_among_the_alive(tmp_path):
    from synthea_quality.condition_cohort import CohortSpec
    from synthea_quality.medications.definitions import MedicationDefinition

    report = medication_report(
        with_medications(dataset(tmp_path)),
        (MedicationDefinition("Lisinopril", ("314076",), CohortSpec("Hypertension")),
         MedicationDefinition("Lisinopril, everyone", ("314076",))),
    )
    cohort, everyone = report.medications
    assert (cohort.active.numerator, cohort.active.denominator) == (1, 1)
    assert cohort.active_with_reason == 1
    assert (everyone.active.numerator, everyone.ever.numerator, everyone.ever.denominator) == (
        1, 2, 2,
    )
    assert "medications" in [i.table for i in report.inputs]
    data = report.to_dict()
    assert data["medications"][0]["name"] == "Lisinopril"
    assert "Wilson" in data["medication_definitions"]["interval"]


def test_without_medications_the_table_is_not_read(tmp_path):
    report = validate(with_medications(dataset(tmp_path)))
    assert report.medications == ()
    assert "medications" not in [i.table for i in report.inputs]
    assert report.to_dict()["medication_definitions"] is None


def test_missing_medications_skip_only_the_medications(tmp_path):
    from synthea_quality.medications.definitions import MedicationDefinition

    report = medication_report(dataset(tmp_path), (MedicationDefinition("A", ("1",)),))
    assert report.medications[0].status is SectionStatus.SKIPPED
    assert "medications.csv is not in the dataset" in report.medications[0].reason
    assert report.conditions[0].prevalence.status is SectionStatus.COMPUTED


def test_a_cohort_without_conditions_csv_is_skipped(tmp_path):
    from synthea_quality.condition_cohort import CohortSpec
    from synthea_quality.medications.definitions import MedicationDefinition

    directory = with_medications(dataset(tmp_path))
    (directory / "conditions.csv").unlink()
    report = medication_report(
        directory, (MedicationDefinition("L", ("314076",), CohortSpec("Hypertension")),
                    MedicationDefinition("All", ("314076",))),
    )
    cohort, everyone = report.medications
    assert cohort.status is SectionStatus.SKIPPED and "conditions.csv" in cohort.reason
    assert everyone.status is SectionStatus.COMPUTED


def test_the_medication_cohort_is_the_prevalence_numerator(tmp_path):
    from synthea_quality.condition_cohort import CohortSpec
    from synthea_quality.medications.definitions import MedicationDefinition

    report = medication_report(
        with_medications(dataset(tmp_path)),
        (MedicationDefinition("L", ("314076",), CohortSpec("Hypertension")),
         MedicationDefinition("L2", ("314076",), CohortSpec("Hypertension", "lifetime"))),
    )
    prevalence = report.conditions[0].prevalence
    point, lifetime = report.medications
    assert point.ever.denominator == prevalence.point.numerator
    assert lifetime.ever.denominator == prevalence.lifetime.numerator
