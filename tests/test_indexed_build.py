"""Tests for the indexed cohort inside a module validation report."""

from __future__ import annotations

import json
from pathlib import Path

from csv_tables import write_table

from synthea_quality.indexed.models import DeathResult, EventResult, SkippedOutcome, ValueResult
from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.cli import print_summary
from synthea_quality.validate.models import ModuleValidationReport, load_module
from synthea_quality.validate.render import render_markdown

SNOMED = "http://snomed.info/sct"
GENERATED_AT = "2026-10-19T00:00:00+00:00"
HOSPITAL_ADMISSION = {
    "name": "Hospital admission",
    "type": "event",
    "table": "encounters",
    "codes": ["1505002"],
}
VENTILATED = {
    "name": "Ventilated",
    "type": "event",
    "table": "devices",
    "codes": ["449071006"],
}


def covid_dataset(directory: Path, *, late_encounter: bool = False) -> Path:
    """Three patients with COVID-19 (one with two rows) and one admission after an index."""
    write_table(
        directory,
        "patients",
        [
            {"Id": "p1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "p2", "BIRTHDATE": "1990-06-01", "GENDER": "F"},
            {"Id": "p3", "BIRTHDATE": "2000-01-01", "GENDER": "M"},
        ],
    )
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "p1", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-01-01",
             "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p1", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-05-01",
             "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p2", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-03-01",
             "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p3", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-04-01",
             "DESCRIPTION": "COVID-19"},
        ],
    )
    if late_encounter:
        encounters = [
            {"Id": "e2", "PATIENT": "p2", "CODE": "1505002", "START": "2025-08-01T00:00:00Z"}
        ]
    else:
        encounters = [
            {"Id": "e1", "PATIENT": "p1", "CODE": "1505002", "START": "2023-02-01T00:00:00Z"}
        ]
    write_table(directory, "encounters", encounters)
    return directory


def with_observations(directory: Path) -> Path:
    """The value of one outcome and the cause of a death, both in observations.csv."""
    write_table(
        directory,
        "observations",
        [
            {"DATE": "2023-02-01T00:00:00Z", "PATIENT": "p1", "CODE": "21908-9", "VALUE": "Stage 2",
             "TYPE": "text"},
            {"DATE": "2023-03-01T00:00:00Z", "PATIENT": "p1", "CODE": "69453-9", "VALUE": "Cause",
             "TYPE": "text"},
        ],
    )
    return directory


def write_module(directory: Path, outcomes: list[dict] | None = None) -> Path:
    """A module file of COVID-19 with ``outcomes``, or without an index when None."""
    data: dict = {
        "module": {"name": "COVID-19"},
        "conditions": [{"name": "COVID-19", "codes": ["840539006"]}],
    }
    if outcomes is not None:
        data["index"] = {"condition": "COVID-19"}
        data["outcomes"] = outcomes
    path = directory / "module.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def validate(directory: Path, module_path: Path, **kwargs) -> ModuleValidationReport:
    loaded = load_module(module_path)
    return build_module_validation(
        directory,
        module=loaded.module,
        definitions=loaded.conditions,
        module_file=module_path,
        index=loaded.index,
        outcomes=loaded.outcomes,
        generated_at=GENERATED_AT,
        **kwargs,
    )


def test_b1_a_module_without_index_has_no_indexed_cohort(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    report = validate(directory, write_module(tmp_path), reference_date="2026-01-01")
    assert report.indexed is None
    assert "indexed_cohort" not in report.to_dict()


def test_b2_the_event_counts_distinct_patients_of_the_cohort(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    report = validate(
        directory, write_module(tmp_path, [HOSPITAL_ADMISSION]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    assert report.indexed.patients == 3
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert (outcome.breakdown.total.patients, outcome.breakdown.total.cohort) == (1, 3)


def test_b3_a_missing_table_skips_only_its_outcome(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    report = validate(
        directory,
        write_module(tmp_path, [HOSPITAL_ADMISSION, VENTILATED]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    admission, ventilated = report.indexed.outcomes
    assert isinstance(admission, EventResult)
    assert admission.breakdown.total.patients == 1
    assert isinstance(ventilated, SkippedOutcome)
    assert ventilated.type == "event"
    assert "devices.csv" in ventilated.reason


def test_b4_observations_are_read_once_for_a_value_and_a_death(tmp_path):
    directory = with_observations(covid_dataset(tmp_path / "csv"))
    outcomes = [
        {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first"},
        {"name": "Death", "type": "death", "cause_values": ["Cause"]},
    ]
    report = validate(
        directory, write_module(tmp_path, outcomes), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    value_outcome, death_outcome = report.indexed.outcomes
    assert isinstance(value_outcome, ValueResult)
    assert isinstance(death_outcome, DeathResult)
    assert [item.table for item in report.inputs].count("observations") == 1


def test_b5_without_conditions_the_cohort_says_why(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    (directory / "conditions.csv").unlink()
    report = validate(
        directory, write_module(tmp_path, [HOSPITAL_ADMISSION]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    assert report.indexed.reason is not None
    assert report.indexed.reason.startswith("no condition records")


def test_b6_a_record_after_the_reference_date_is_left_out_and_counted(tmp_path):
    directory = covid_dataset(tmp_path / "csv", late_encounter=True)
    report = validate(
        directory, write_module(tmp_path, [HOSPITAL_ADMISSION]),
        reference_date="2025-06-30",
    )
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert (outcome.breakdown.total.patients, outcome.breakdown.total.cohort) == (0, 3)
    assert outcome.rows_after_reference == 1


def test_b7_markdown_shows_the_section_only_when_the_module_has_an_index(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    with_index = validate(
        directory, write_module(tmp_path, [HOSPITAL_ADMISSION]),
        reference_date="2026-01-01",
    )
    without_index = validate(
        directory, write_module(tmp_path), reference_date="2026-01-01"
    )
    assert "## Indexed cohort" in render_markdown(with_index)
    assert "## Indexed cohort" not in render_markdown(without_index)


def test_b8_the_console_summary_states_the_cohort_and_each_outcome(tmp_path, capsys):
    directory = covid_dataset(tmp_path / "csv")
    report = validate(
        directory, write_module(tmp_path, [HOSPITAL_ADMISSION]),
        reference_date="2026-01-01",
    )
    print_summary(report, tmp_path / "report.md", tmp_path / "report.json")
    printed = capsys.readouterr().out
    assert "Indexed:    COVID-19: 3 patients" in printed
    assert "Outcome:    Hospital admission: 1/3" in printed


def test_b9_without_a_reference_date_the_json_writes_no_date(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    (directory / "encounters.csv").unlink()
    report = validate(directory, write_module(tmp_path, [HOSPITAL_ADMISSION]))
    assert report.indexed is not None
    data = report.to_dict()["indexed_cohort"]
    assert data["reference_date"] is None
    assert data["status"] == "SKIPPED"
