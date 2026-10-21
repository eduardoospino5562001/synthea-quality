"""Tests for the indexed cohort inside a module validation report."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
from csv_tables import write_table

from synthea_quality.dataset import DatasetContext
from synthea_quality.indexed import build as indexed_build
from synthea_quality.indexed.compute import IndexedCohort, OutcomeMembers
from synthea_quality.indexed.definitions import ValueOutcome
from synthea_quality.indexed.models import DeathResult, EventResult, SkippedOutcome, ValueResult
from synthea_quality.profile.models import InputState, TableInput
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


def cross_dataset(directory: Path) -> Path:
    """Three patients with COVID-19: one dead with a stage and a cause, one with a stage.

    The stages are two different values, so the crossing has one stratum per value and
    a last one for the patient without a stage.
    """
    write_table(
        directory,
        "patients",
        [
            {"Id": "p1", "BIRTHDATE": "1950-01-01", "DEATHDATE": "2023-06-01", "GENDER": "F"},
            {"Id": "p2", "BIRTHDATE": "1990-06-01", "GENDER": "M"},
            {"Id": "p3", "BIRTHDATE": "2000-01-01", "GENDER": "F"},
        ],
    )
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "p1", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-01-01"},
            {"PATIENT": "p2", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-03-01"},
            {"PATIENT": "p3", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-04-01"},
        ],
    )
    write_table(
        directory,
        "observations",
        [
            {"DATE": "2023-02-01T00:00:00Z", "PATIENT": "p1", "CODE": "21908-9", "VALUE": "Stage 1",
             "TYPE": "text"},
            {"DATE": "2023-06-05T00:00:00Z", "PATIENT": "p1", "CODE": "69453-9", "VALUE": "Cause",
             "TYPE": "text"},
            {"DATE": "2023-05-01T00:00:00Z", "PATIENT": "p3", "CODE": "21908-9", "VALUE": "Stage 2",
             "TYPE": "text"},
        ],
    )
    write_table(
        directory,
        "encounters",
        [{"Id": "e1", "PATIENT": "p1", "CODE": "1505002", "START": "2023-02-05T00:00:00Z"}],
    )
    return directory


STAGE = {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first"}
DEATH = {"name": "Death", "type": "death", "cause_values": ["Cause"]}
#: The "Recovered" event of the plan: two parts.
RECOVERED = {
    "name": "Recovered",
    "type": "event",
    "op": "any",
    "match": [
        {"name": "Isolation ended", "table": "careplans", "codes": ["736376001"],
         "reasons": ["840539006"], "at": "stop"},
        {"name": "Negative test", "table": "observations", "codes": ["94531-1"],
         "values": ["Not detected (qualifier value)"]},
    ],
}
#: The same two parts, and every one of them needed.
RECOVERED_ALL = {**RECOVERED, "name": "Recovered from both", "op": "all"}
#: An event of one table without filters, of the same table of the first part.
ISOLATION = {"name": "Isolation", "type": "event", "table": "careplans",
             "codes": ["736376001"]}
#: An event of two parts of the same table.
ISOLATION_TWICE = {
    "name": "Isolation twice",
    "type": "event",
    "op": "any",
    "match": [
        {"name": "Isolation ended", "table": "careplans", "codes": ["736376001"],
         "reasons": ["840539006"], "at": "stop"},
        {"name": "Isolation started", "table": "careplans", "codes": ["736376001"]},
    ],
}
#: The careplans of the dataset of the recovered outcome.
CAREPLANS = [
    {"Id": "c1", "PATIENT": "p1", "CODE": "736376001", "START": "2023-01-01",
     "STOP": "2023-01-10", "REASONCODE": "840539006"},
    {"Id": "c2", "PATIENT": "p2", "CODE": "736376001", "START": "2023-03-01",
     "STOP": "2023-03-05", "REASONCODE": "840544004"},
    {"Id": "c3", "PATIENT": "p3", "CODE": "736376001", "START": "2023-04-01",
     "STOP": "", "REASONCODE": "840539006"},
]
#: The header of ``careplans.csv`` without its ``STOP`` column.
CAREPLANS_WITHOUT_STOP = (
    "Id", "START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE",
    "REASONDESCRIPTION",
)
#: The header of ``careplans.csv`` without its ``REASONCODE`` column.
CAREPLANS_WITHOUT_REASON = (
    "Id", "START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION",
    "REASONDESCRIPTION",
)
#: The observations of the dataset of the recovered outcome.
OBSERVATIONS = [
    {"DATE": "2023-01-05T00:00:00Z", "PATIENT": "p1", "CODE": "94531-1",
     "VALUE": "Detected (qualifier value)"},
    {"DATE": "2023-01-08T00:00:00Z", "PATIENT": "p1", "CODE": "94531-1",
     "VALUE": "Not detected (qualifier value)"},
    {"DATE": "2023-04-10T00:00:00Z", "PATIENT": "p3", "CODE": "94531-1",
     "VALUE": "Not detected (qualifier value)"},
]
#: The header of ``observations.csv`` without its ``VALUE`` column.
OBSERVATIONS_WITHOUT_VALUE = (
    "DATE", "PATIENT", "ENCOUNTER", "CATEGORY", "CODE", "DESCRIPTION", "UNITS", "TYPE",
)
#: The header of ``conditions.csv`` without its ``STOP`` column.
CONDITIONS_WITHOUT_STOP = ("START", "PATIENT", "ENCOUNTER", "SYSTEM", "CODE", "DESCRIPTION")
#: The header of ``medications.csv`` without its ``STOP`` column.
MEDICATIONS_WITHOUT_STOP = (
    "START", "PATIENT", "PAYER", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONCODE",
)
#: An event on the condition records: the end of the COVID-19 condition.
CONDITION_ENDED = {
    "name": "COVID-19 ended",
    "type": "event",
    "table": "conditions",
    "codes": ["840539006"],
    "at": "stop",
}
#: An event on the medication records.
MEDICATION_TAKEN = {
    "name": "Lisinopril",
    "type": "event",
    "table": "medications",
    "codes": ["314076"],
}
#: The medications of the module file of the medication outcome.
MEDICATIONS = [{"name": "Lisinopril", "codes": ["314076"]}]


def recovered_dataset(directory: Path) -> Path:
    """Three patients with COVID-19: one recovered, one isolated, one with a test.

    The first patient ends the isolation of COVID-19 and has a negative test; the
    second one has an isolation of another reason and no end; the third one has an
    isolation of COVID-19 that never ends and a negative test.
    """
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
             "STOP": "2023-01-10"},
            {"PATIENT": "p2", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-03-01"},
            {"PATIENT": "p3", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-04-01",
             "STOP": "2023-04-10"},
        ],
    )
    write_table(directory, "careplans", CAREPLANS)
    write_table(directory, "observations", OBSERVATIONS)
    return directory


def without(rows: list[dict[str, str]], column: str) -> list[dict[str, str]]:
    """The rows of a table without the column ``column``."""
    return [{key: value for key, value in row.items() if key != column} for row in rows]


def ended_covid_dataset(directory: Path) -> Path:
    """The COVID-19 dataset: one condition ends, the other two never do."""
    directory = covid_dataset(directory)
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "p1", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-01-01",
             "STOP": "2023-02-01", "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p1", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-05-01",
             "STOP": "", "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p2", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-03-01",
             "STOP": "", "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p3", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-04-01",
             "STOP": "", "DESCRIPTION": "COVID-19"},
        ],
    )
    return directory


def with_medications(directory: Path, *, stop: bool = True) -> Path:
    """The dataset with one medication record, of the patient of the index."""
    rows = [
        {"START": "2023-01-15T00:00:00Z", "PATIENT": "p1", "CODE": "314076",
         "DESCRIPTION": "lisinopril 10 MG Oral Tablet"},
    ]
    if stop:
        rows[0]["STOP"] = "2023-03-01T00:00:00Z"
    write_table(directory, "medications", rows, columns=None if stop else MEDICATIONS_WITHOUT_STOP)
    return directory


def write_module(
    directory: Path,
    outcomes: list[dict] | None = None,
    crosses: list[dict] | None = None,
    medications: list[dict] | None = None,
) -> Path:
    """A module file of COVID-19 with ``outcomes``, or without an index when None."""
    data: dict = {
        "module": {"name": "COVID-19"},
        "conditions": [{"name": "COVID-19", "codes": ["840539006"]}],
    }
    if outcomes is not None:
        data["index"] = {"condition": "COVID-19"}
        data["outcomes"] = outcomes
    if crosses is not None:
        data["by"] = crosses
    if medications is not None:
        data["medications"] = medications
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
        crosses=loaded.crosses,
        medications=loaded.medications,
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


def test_b10_a_patients_table_without_gender_gives_an_unknown_sex(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    write_table(
        directory,
        "patients",
        [
            {"Id": "p1", "BIRTHDATE": "1950-01-01"},
            {"Id": "p2", "BIRTHDATE": "1990-06-01"},
            {"Id": "p3", "BIRTHDATE": "2000-01-01"},
        ],
        columns=("Id", "BIRTHDATE", "DEATHDATE"),
    )
    report = validate(
        directory, write_module(tmp_path, [HOSPITAL_ADMISSION]), reference_date="2026-01-01"
    )
    outcome = report.to_dict()["indexed_cohort"]["outcomes"][0]
    assert [cell["label"] for cell in outcome["breakdown"]["by_sex"]] == ["unknown sex"]
    assert outcome["breakdown"]["total"]["denominator"] == 3


def test_b11_a_missing_table_gives_its_reason_once(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    report = validate(
        directory,
        write_module(tmp_path, [HOSPITAL_ADMISSION, VENTILATED]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    ventilated = next(o for o in report.indexed.outcomes if o.name == "Ventilated")
    assert isinstance(ventilated, SkippedOutcome)
    assert ventilated.reason == "devices.csv is not in the dataset"


def test_b12_an_event_on_observations_needs_no_value_column(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    write_table(
        directory,
        "observations",
        [{"DATE": "2023-02-01T00:00:00Z", "PATIENT": "p1", "CODE": "21908-9"}],
        columns=("DATE", "PATIENT", "CODE"),
    )
    stage = {"name": "Stage recorded", "type": "event", "table": "observations", "codes": ["21908-9"]}
    report = validate(directory, write_module(tmp_path, [stage]), reference_date="2026-01-01")
    outcome = report.to_dict()["indexed_cohort"]["outcomes"][0]
    assert outcome["status"] == "COMPUTED"
    assert outcome["breakdown"]["total"]["numerator"] == 1


def test_b13_a_table_not_read_states_its_use(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    write_table(
        directory,
        "observations",
        [{"DATE": "2023-02-01T00:00:00Z", "PATIENT": "p1", "CODE": "21908-9"}],
        columns=("DATE", "PATIENT", "CODE"),
    )
    stage = {"name": "Stage recorded", "type": "event", "table": "observations", "codes": ["21908-9"]}
    module_path = write_module(tmp_path, [stage])
    data = json.loads(module_path.read_text(encoding="utf-8"))
    data["observations"] = [{"code": "8480-6"}]
    module_path.write_text(json.dumps(data), encoding="utf-8")
    report = validate(
        directory, module_path, reference_date="2026-01-01",
        observations=load_module(module_path).observations,
    )
    text = render_markdown(report)
    assert (
        "- `observations`: **absent** for observation values — "
        "observations.csv has no VALUE column(s)"
    ) in text
    assert "- `observations`: 1 rows, for indexed cohort outcomes" in text


def test_b14_an_event_is_measured_when_another_outcome_lacks_value(tmp_path):
    directory = covid_dataset(tmp_path / "csv")
    write_table(
        directory,
        "observations",
        [{"DATE": "2023-02-01T00:00:00Z", "PATIENT": "p1", "CODE": "21908-9"}],
        columns=("DATE", "PATIENT", "CODE"),
    )
    outcomes = [
        {"name": "Stage recorded", "type": "event", "table": "observations", "codes": ["21908-9"]},
        {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first"},
        {"name": "Death", "type": "death", "cause_values": ["Cause"]},
    ]
    report = validate(directory, write_module(tmp_path, outcomes), reference_date="2026-01-01")
    measured = {o["name"]: o for o in report.to_dict()["indexed_cohort"]["outcomes"]}
    assert measured["Stage recorded"]["status"] == "COMPUTED"
    assert measured["Stage recorded"]["breakdown"]["total"]["numerator"] == 1
    for name in ("Stage group", "Death"):
        assert measured[name]["status"] == "SKIPPED"
        assert measured[name]["reason"] == "observations.csv has no VALUE column(s)"


def test_b15_the_cross_reaches_the_result_with_the_figures_of_the_measure(tmp_path):
    directory = cross_dataset(tmp_path / "csv")
    report = validate(
        directory,
        write_module(tmp_path, [STAGE, DEATH], [{"outcome": "Death", "by": "Stage group"}]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    (cross,) = report.indexed.crosses
    assert (cross.outcome, cross.by, cross.by_type) == ("Death", "Stage group", "value")
    assert [
        (stratum.label, stratum.numerator, stratum.denominator, stratum.with_cause)
        for stratum in cross.strata
    ] == [
        ("Stage 1", 1, 1, 1),
        ("Stage 2", 0, 1, 0),
        ("no value", 0, 1, 0),
    ]
    assert cross.patients_in_several_strata == 0
    death = report.indexed.outcomes[1]
    assert isinstance(death, DeathResult)
    assert sum(stratum.numerator for stratum in cross.strata) == death.breakdown.total.patients


def test_b16_the_members_of_an_outcome_are_computed_once(tmp_path, monkeypatch):
    directory = cross_dataset(tmp_path / "csv")
    calls: list[str] = []
    original = indexed_build.value_members

    def counted(
        cohort: IndexedCohort,
        observations: pd.DataFrame,
        outcome: ValueOutcome,
        reference: date,
    ) -> OutcomeMembers:
        calls.append(outcome.name)
        return original(cohort, observations, outcome, reference)

    monkeypatch.setattr(indexed_build, "value_members", counted)
    report = validate(
        directory,
        write_module(
            tmp_path,
            [STAGE, DEATH, HOSPITAL_ADMISSION],
            [
                {"outcome": "Death", "by": "Stage group"},
                {"outcome": "Stage group", "by": "Hospital admission"},
            ],
        ),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    assert len(report.indexed.crosses) == 2
    assert calls == ["Stage group"]


def test_b17_an_outcome_that_could_not_be_measured_skips_its_cross(tmp_path):
    directory = cross_dataset(tmp_path / "csv")
    report = validate(
        directory,
        write_module(
            tmp_path,
            [STAGE, DEATH, VENTILATED],
            [
                {"outcome": "Ventilated", "by": "Stage group"},
                {"outcome": "Death", "by": "Ventilated"},
            ],
        ),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    first, second = report.indexed.crosses
    assert first.strata == ()
    assert first.reason == (
        "outcome 'Ventilated' was not computed: devices.csv is not in the dataset"
    )
    assert second.reason == first.reason
    assert second.by_type == "event"


def test_b18_a_cross_names_the_measured_outcome_first(tmp_path):
    directory = cross_dataset(tmp_path / "csv")
    (directory / "observations.csv").unlink()
    report = validate(
        directory,
        write_module(
            tmp_path,
            [STAGE, DEATH, VENTILATED],
            [{"outcome": "Ventilated", "by": "Stage group"}],
        ),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    (cross,) = report.indexed.crosses
    assert cross.reason == (
        "outcome 'Ventilated' was not computed: devices.csv is not in the dataset"
    )


def test_b19_a_cohort_that_could_not_be_built_skips_every_cross(tmp_path):
    directory = cross_dataset(tmp_path / "csv")
    (directory / "conditions.csv").unlink()
    report = validate(
        directory,
        write_module(tmp_path, [STAGE, DEATH], [{"outcome": "Death", "by": "Stage group"}]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    assert report.indexed.reason is not None
    (cross,) = report.indexed.crosses
    assert cross.strata == ()
    assert cross.reason == (
        f"outcome 'Death' was not computed: {report.indexed.reason}"
    )


def test_b20_without_a_cross_the_json_has_none_and_the_markdown_has_none(tmp_path):
    directory = cross_dataset(tmp_path / "csv")
    report = validate(
        directory, write_module(tmp_path, [STAGE, DEATH]), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    assert report.indexed.crosses == ()
    assert report.to_dict()["indexed_cohort"]["crosses"] == []
    assert "### Death by Stage group" not in render_markdown(report)


def test_b21_the_two_examples_are_loaded_without_error():
    repository = Path(__file__).resolve().parents[1]
    for name in ("breast_cancer.json", "covid19.json"):
        loaded = load_module(repository / "examples" / name)
        assert loaded.index is not None
        assert loaded.crosses
        assert all(cross.outcome and cross.by for cross in loaded.crosses)


def test_b22_an_event_of_several_parts_is_measured_with_its_parts(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    report = validate(
        directory, write_module(tmp_path, [RECOVERED]), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert outcome.op == "any"
    assert outcome.table is None
    assert outcome.codes == ()
    assert outcome.breakdown.total.patients == 2
    assert outcome.days.median == 8
    isolation, negative = outcome.matches
    assert (isolation.name, isolation.patients) == ("Isolation ended", 1)
    assert (negative.name, negative.patients) == ("Negative test", 2)
    assert (isolation.rows_matched, isolation.rows_filtered_out) == (3, 1)
    assert (isolation.rows_not_ended, isolation.rows_date_unusable) == (1, 0)
    data = report.to_dict()["indexed_cohort"]["outcomes"][0]
    assert data["table"] is None
    assert data["codes"] == []
    assert data["op"] == "any"
    assert [match["name"] for match in data["matches"]] == [
        "Isolation ended",
        "Negative test",
    ]
    assert data["matches"][0]["reasons"] == ["840539006"]
    assert data["matches"][0]["at"] == "stop"


def test_b23_an_event_of_several_parts_reads_each_table_once(tmp_path, monkeypatch):
    directory = recovered_dataset(tmp_path / "csv")
    calls: list[str] = []
    original = DatasetContext.load

    def counted(
        self: DatasetContext,
        name: str,
        wanted: Sequence[str],
        required: Sequence[str],
        used_for: str,
    ) -> tuple[pd.DataFrame | None, TableInput]:
        calls.append(name)
        return original(self, name, wanted, required, used_for)

    monkeypatch.setattr(DatasetContext, "load", counted)
    report = validate(
        directory,
        write_module(tmp_path, [ISOLATION_TWICE]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    assert calls.count("careplans") == 1
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert outcome.breakdown.total.patients == 3


def test_b24_the_optional_columns_are_asked_for_but_not_required(tmp_path, monkeypatch):
    directory = recovered_dataset(tmp_path / "csv")
    asked: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    original = DatasetContext.load

    def counted(
        self: DatasetContext,
        name: str,
        wanted: Sequence[str],
        required: Sequence[str],
        used_for: str,
    ) -> tuple[pd.DataFrame | None, TableInput]:
        asked[name] = (tuple(wanted), tuple(required))
        return original(self, name, wanted, required, used_for)

    monkeypatch.setattr(DatasetContext, "load", counted)
    report = validate(
        directory, write_module(tmp_path, [RECOVERED]), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    wanted, required = asked["careplans"]
    assert "STOP" in wanted
    assert "REASONCODE" in wanted
    assert "STOP" not in required
    assert "REASONCODE" not in required
    wanted, required = asked["observations"]
    assert "VALUE" in wanted
    assert "VALUE" not in required


def test_b25_an_event_without_the_stop_column_is_skipped(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    write_table(
        directory, "careplans", without(CAREPLANS, "STOP"), columns=CAREPLANS_WITHOUT_STOP
    )
    report = validate(
        directory, write_module(tmp_path, [RECOVERED]), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, SkippedOutcome)
    assert outcome.reason == (
        "careplans.csv has no STOP column(s), which part 'Isolation ended' needs"
    )


def test_b26_an_event_without_the_reason_code_column_is_skipped(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    write_table(
        directory,
        "careplans",
        without(CAREPLANS, "REASONCODE"),
        columns=CAREPLANS_WITHOUT_REASON,
    )
    report = validate(
        directory, write_module(tmp_path, [RECOVERED]), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, SkippedOutcome)
    assert outcome.reason == (
        "careplans.csv has no REASONCODE column(s), which part 'Isolation ended' needs"
    )


def test_b27_an_event_without_the_value_column_is_skipped(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    write_table(
        directory,
        "observations",
        without(OBSERVATIONS, "VALUE"),
        columns=OBSERVATIONS_WITHOUT_VALUE,
    )
    report = validate(
        directory, write_module(tmp_path, [RECOVERED]), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, SkippedOutcome)
    assert outcome.reason == (
        "observations.csv has no VALUE column(s), which part 'Negative test' needs"
    )


def test_b28_an_event_without_its_table_is_skipped(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    (directory / "careplans.csv").unlink()
    report = validate(
        directory, write_module(tmp_path, [RECOVERED]), reference_date="2026-01-01"
    )
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, SkippedOutcome)
    assert outcome.reason == "careplans.csv is not in the dataset"


def test_b29_another_outcome_of_the_same_table_is_still_measured(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    write_table(
        directory,
        "careplans",
        without(CAREPLANS, "REASONCODE"),
        columns=CAREPLANS_WITHOUT_REASON,
    )
    report = validate(
        directory,
        write_module(tmp_path, [RECOVERED, ISOLATION]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    recovered, isolation = report.indexed.outcomes
    assert isinstance(recovered, SkippedOutcome)
    assert recovered.reason == (
        "careplans.csv has no REASONCODE column(s), which part 'Isolation ended' needs"
    )
    assert isinstance(isolation, EventResult)
    assert isolation.breakdown.total.patients == 3


def test_b30_a_cross_can_use_an_event_of_several_parts_as_its_strata(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    report = validate(
        directory,
        write_module(
            tmp_path,
            [RECOVERED, {"name": "Death", "type": "death"}],
            [{"outcome": "Death", "by": "Recovered"}],
        ),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    (cross,) = report.indexed.crosses
    assert cross.by_type == "event"
    assert [
        (stratum.label, stratum.numerator, stratum.denominator)
        for stratum in cross.strata
    ] == [
        ("with Recovered", 0, 2),
        ("without Recovered", 0, 1),
    ]


def test_b31_the_all_op_needs_every_part(tmp_path):
    directory = recovered_dataset(tmp_path / "csv")
    report = validate(
        directory,
        write_module(tmp_path, [RECOVERED_ALL]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert outcome.op == "all"
    assert outcome.breakdown.total.patients == 1
    assert outcome.days.median == 9


def _counted_loads(monkeypatch) -> list[str]:
    """Record the name of every table the report reads, and return the list."""
    calls: list[str] = []
    original = DatasetContext.load

    def counted(
        self: DatasetContext,
        name: str,
        wanted: Sequence[str],
        required: Sequence[str],
        used_for: str,
    ) -> tuple[pd.DataFrame | None, TableInput]:
        calls.append(name)
        return original(self, name, wanted, required, used_for)

    monkeypatch.setattr(DatasetContext, "load", counted)
    return calls


def test_b32_the_condition_records_are_read_once_when_an_outcome_needs_them(
    tmp_path, monkeypatch
):
    """The table the report read is not read again, and its entry says both uses."""
    directory = ended_covid_dataset(tmp_path / "csv")
    calls = _counted_loads(monkeypatch)
    report = validate(
        directory, write_module(tmp_path, [CONDITION_ENDED]), reference_date="2026-01-01"
    )
    assert calls.count("conditions") == 1
    listed = [(item.table, item.used_for) for item in report.inputs]
    assert listed.count(("conditions", "condition records and indexed cohort outcomes")) == 1
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert outcome.breakdown.total.patients == 1


def test_b33_the_medication_records_are_read_once_when_an_outcome_needs_them(
    tmp_path, monkeypatch
):
    """The same with the medication records, of a module file with medications."""
    directory = with_medications(covid_dataset(tmp_path / "csv"))
    calls = _counted_loads(monkeypatch)
    report = validate(
        directory,
        write_module(tmp_path, [MEDICATION_TAKEN], medications=MEDICATIONS),
        reference_date="2026-01-01",
    )
    assert calls.count("medications") == 1
    listed = [(item.table, item.used_for) for item in report.inputs]
    assert listed.count(("medications", "medication records and indexed cohort outcomes")) == 1
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert outcome.breakdown.total.patients == 1


def test_b34_a_medication_table_without_stop_is_read_by_the_cohort_itself(
    tmp_path, monkeypatch
):
    """A table the report cannot read is read by the cohort, with fewer columns."""
    directory = with_medications(covid_dataset(tmp_path / "csv"), stop=False)
    calls = _counted_loads(monkeypatch)
    report = validate(
        directory,
        write_module(tmp_path, [MEDICATION_TAKEN], medications=MEDICATIONS),
        reference_date="2026-01-01",
    )
    assert calls.count("medications") == 2
    listed = [(item.table, item.used_for, item.state) for item in report.inputs]
    assert ("medications", "medication records", InputState.ABSENT) in listed
    assert ("medications", "indexed cohort outcomes", InputState.READ) in listed
    assert report.indexed is not None
    outcome = report.indexed.outcomes[0]
    assert isinstance(outcome, EventResult)
    assert outcome.breakdown.total.patients == 1


def test_b35_an_event_on_conditions_without_stop_is_skipped_with_its_reason(tmp_path):
    """A condition table without ``STOP`` leaves out only the outcome that needs it."""
    directory = covid_dataset(tmp_path / "csv")
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "p1", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-01-01",
             "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p2", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-03-01",
             "DESCRIPTION": "COVID-19"},
            {"PATIENT": "p3", "CODE": "840539006", "SYSTEM": SNOMED, "START": "2023-04-01",
             "DESCRIPTION": "COVID-19"},
        ],
        columns=CONDITIONS_WITHOUT_STOP,
    )
    report = validate(
        directory,
        write_module(tmp_path, [CONDITION_ENDED, HOSPITAL_ADMISSION]),
        reference_date="2026-01-01",
    )
    assert report.indexed is not None
    ended, admission = report.indexed.outcomes
    assert isinstance(ended, SkippedOutcome)
    assert ended.reason == (
        "conditions.csv has no STOP column(s), which part 'conditions: 840539006' needs"
    )
    assert isinstance(admission, EventResult)
    assert admission.breakdown.total.patients == 1


def test_b36_a_module_without_an_event_on_those_tables_lists_them_as_before(tmp_path):
    """A module with no event on the condition or medication records lists them once."""
    directory = with_medications(covid_dataset(tmp_path / "csv"))
    report = validate(
        directory,
        write_module(tmp_path, [HOSPITAL_ADMISSION], medications=MEDICATIONS),
        reference_date="2026-01-01",
    )
    listed = [(item.table, item.used_for, item.rows) for item in report.inputs]
    assert listed == [
        ("patients", "the population, the alive cohort and the followed population", 3),
        ("conditions", "condition records", 4),
        ("medications", "medication records", 1),
        ("encounters", "indexed cohort outcomes", 1),
    ]


def test_b37_the_cohort_does_not_change_the_frame_of_the_report(tmp_path, monkeypatch):
    """The frame the report read reaches the cohort as it was read.

    The cohort measures its outcomes from that frame: if any step changed it in place,
    the figures of the outcomes would change without anything saying so.
    """
    directory = ended_covid_dataset(tmp_path / "csv")
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    original = DatasetContext.load

    def counted(
        self: DatasetContext,
        name: str,
        wanted: Sequence[str],
        required: Sequence[str],
        used_for: str,
    ) -> tuple[pd.DataFrame | None, TableInput]:
        frame, table_input = original(self, name, wanted, required, used_for)
        if name == "conditions" and frame is not None:
            frames[name] = (frame, frame.copy(deep=True))
        return frame, table_input

    monkeypatch.setattr(DatasetContext, "load", counted)
    validate(
        directory, write_module(tmp_path, [CONDITION_ENDED]), reference_date="2026-01-01"
    )
    assert "conditions" in frames
    frame, read = frames["conditions"]
    pd.testing.assert_frame_equal(frame, read)
