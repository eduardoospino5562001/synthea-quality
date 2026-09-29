"""Tests for the Markdown and JSON renderings of observation values."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.observations.build import build_observations
from synthea_quality.observations.definitions import (
    CONFIGURATION_NOTE,
    ObservationDefinition,
    ReferenceRange,
)
from synthea_quality.observations.render import (
    _num,
    dumps,
    render_markdown,
    write_json,
    write_markdown,
)
from synthea_quality.prevalence.definitions import parse_condition_option
from synthea_quality.schema.tables import tables_by_name

SNOMED = "http://snomed.info/sct"
SBP = "8480-6"


def write_table(directory: Path, name: str, rows, columns=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def reading(patient, value, day="2026-01-01T00:00:00Z"):
    return {"DATE": day, "PATIENT": patient, "CODE": SBP, "DESCRIPTION": "Systolic",
            "VALUE": value, "UNITS": "mm[Hg]", "TYPE": "numeric"}


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
        [{"PATIENT": "a1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2015-01-01"}],
    )
    write_table(
        directory,
        "observations",
        [reading("a1", "150"), reading("a2", "120", "2020-01-01T00:00:00Z"), reading("d1", "170")],
    )
    return directory



RANGE = ReferenceRange("mm[Hg]", 100, 139, "Synthea configuration", "synthea-configuration")


@pytest.fixture
def report(tmp_path):
    directory = dataset(tmp_path)
    write_table(
        directory,
        "observations",
        [
            reading("a1", "150"),
            reading("a2", "120", "2020-01-01T00:00:00Z"),
            {**reading("a2", "Negative"), "CODE": "2514-8", "TYPE": "text", "UNITS": ""},
            {**reading("a1", "1"), "CODE": "2514-8", "UNITS": "{presence}"},
            {**reading("a1", "10"), "CODE": "33914-3", "UNITS": "mL/min"},
            {**reading("a2", "20"), "CODE": "33914-3", "UNITS": "mL/min/{1.73_m2}"},
        ],
    )
    return build_observations(
        directory,
        observations=[
            ObservationDefinition(SBP, name="Systolic", reference_range=RANGE),
            ObservationDefinition(SBP, name="Systolic, hypertension",
                                  cohort=CohortSpec("Hypertension")),
            ObservationDefinition("missing"),
        ],
        conditions=[parse_condition_option("Hypertension=59621000")],
        generated_at="2026-09-29T00:00:00+00:00",
    )


def test_the_markdown(report):
    text = render_markdown(report)
    assert text.startswith("# Synthea observation values report")
    assert "## How values are described" in text
    assert "## Observation: Systolic\n" in text
    assert "### Values in `mm[Hg]`" in text
    assert "| 2 | 120 | — | — | 135 | — | — | 150 |" in text
    assert (
        "Percentiles are not shown below 10 patients: only n, minimum, median and maximum."
    ) in text
    assert (
        "| 100 | 139 | `mm[Hg]` | 0 (0.0%) | 1 (50.0%) | 1 (50.0%) | Synthea configuration |"
    ) in text
    assert f"> {CONFIGURATION_NOTE}" in text
    assert (
        "alive patients with Hypertension active at the reference date (rule `point`): **1**"
    ) in text
    assert "`SKIPPED` — no row of code missing" in text
    assert "### Codes written in more than one unit" in text
    assert "`mL/min` 1, `mL/min/{1.73_m2}` 1" in text
    assert "numeric 1, text 1" in text
    assert "several units" in text and "several TYPE" in text
    for word in ("PASS", "FAIL", "match"):
        assert word not in text


def test_none_asked_for(tmp_path):
    report = build_observations(dataset(tmp_path), generated_at="2026-09-29T00:00:00+00:00")
    assert "None. Use `--observation CODE`" in render_markdown(report)


def test_the_json_and_the_files(report, tmp_path):
    data = json.loads(dumps(report))
    assert data["observations"][0]["groups"][0]["reference_range"]["within"] == 1
    assert write_json(report, tmp_path / "o" / "r.json").read_text("utf-8") == dumps(report)
    assert write_markdown(report, tmp_path / "o" / "r.md").read_text("utf-8").startswith("#")


def test_numbers_are_shown_without_trailing_zeros():
    assert [_num(v) for v in (120.0, 1.015, 93.9, 0.00001, None)] == [
        "120", "1.015", "93.9", "0", "—",
    ]
