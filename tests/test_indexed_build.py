"""Tests for the indexed cohort inside a module validation report."""

from __future__ import annotations

import json
from pathlib import Path

from csv_tables import write_table

from synthea_quality.indexed import build as indexed_build
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


def write_module(
    directory: Path, outcomes: list[dict] | None = None, crosses: list[dict] | None = None
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

    def counted(cohort, observations, outcome, reference):
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
