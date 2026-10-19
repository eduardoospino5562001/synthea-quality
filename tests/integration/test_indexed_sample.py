"""Recount the indexed cohorts of the example module files over the official sample.

Every figure the tool reports is recounted here with the standard ``csv`` module and
``datetime`` only, from ``patients.csv``, ``conditions.csv``, ``encounters.csv``,
``devices.csv`` and ``observations.csv``, so the expected numbers do not come from the
code under test. The dataset is fetched with ``python scripts/fetch_official_sample.py``
or pointed at with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pytest

from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.models import ModuleValidationReport, load_module

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "examples"
COVID = EXAMPLES / "covid19.json"
BREAST_CANCER = EXAMPLES / "breast_cancer.json"
MYOCARDIAL_INFARCTION = EXAMPLES / "myocardial_infarction.json"
REFERENCE_DATE = "2026-08-17"
COVID_CODE = "840539006"
BREAST_CANCER_CODE = "254837009"
STAGE_CODE = "21908-9"
#: The date column of each table the examples read.
DATE_COLUMNS = {"observations": "DATE"}
#: The site of each outcome of the COVID-19 example.
COVID_EVENTS = {
    "Hospital admission": ("encounters", {"1505002"}),
    "ICU admission": ("encounters", {"305351004"}),
    "Ventilated": ("devices", {"449071006"}),
}


def _rows(path: Path) -> Iterable[dict[str, str]]:
    """The rows of one CSV file as read by the standard ``csv`` module."""
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _day(value: str) -> date | None:
    """The date of a CSV value, or ``None`` when it is empty or not ISO ``YYYY-MM-DD``."""
    text = value[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _index_dates(sample: Path, code: str) -> dict[str, date]:
    """Each patient's index date: the earliest ``START`` of ``code`` on or before the reference."""
    reference = date.fromisoformat(REFERENCE_DATE)
    listed = {row["Id"] for row in _rows(sample / "patients.csv")}
    earliest: dict[str, date] = {}
    for row in _rows(sample / "conditions.csv"):
        if row["CODE"] != code or row["PATIENT"] not in listed:
            continue
        start = _day(row["START"])
        if start is None or start > reference:
            continue
        patient = row["PATIENT"]
        if patient not in earliest or start < earliest[patient]:
            earliest[patient] = start
    return earliest


def _event_patients(
    sample: Path, table: str, codes: set[str], index_dates: dict[str, date]
) -> int:
    """Distinct cohort patients with a row of ``codes`` between their index and the reference."""
    reference = date.fromisoformat(REFERENCE_DATE)
    column = DATE_COLUMNS.get(table, "START")
    patients = set()
    for row in _rows(sample / f"{table}.csv"):
        if row["CODE"] not in codes:
            continue
        index = index_dates.get(row["PATIENT"])
        if index is None:
            continue
        recorded = _day(row[column])
        if recorded is not None and index <= recorded <= reference:
            patients.add(row["PATIENT"])
    return len(patients)


def _death_patients(sample: Path, index_dates: dict[str, date]) -> int:
    """Distinct cohort patients whose ``DEATHDATE`` is between their index and the reference."""
    reference = date.fromisoformat(REFERENCE_DATE)
    patients = set()
    for row in _rows(sample / "patients.csv"):
        index = index_dates.get(row["Id"])
        if index is None:
            continue
        death = _day(row["DEATHDATE"])
        if death is not None and index <= death <= reference:
            patients.add(row["Id"])
    return len(patients)


def _validate(sample: Path, module_path: Path) -> dict:
    """Run the command's builder over ``sample`` with the example, as the command does."""
    loaded = load_module(module_path)
    report: ModuleValidationReport = build_module_validation(
        sample,
        module=loaded.module,
        definitions=loaded.conditions,
        module_file=module_path,
        index=loaded.index,
        outcomes=loaded.outcomes,
        reference_date=REFERENCE_DATE,
    )
    return report.to_dict()


def _outcome(report: dict, name: str) -> dict:
    """The outcome of the indexed cohort named ``name``."""
    outcomes = report["indexed_cohort"]["outcomes"]
    return next(outcome for outcome in outcomes if outcome["name"] == name)


@pytest.fixture(scope="module")
def covid(sample) -> dict:
    """The validation report of the COVID-19 example over the sample."""
    return _validate(sample, COVID)


@pytest.fixture(scope="module")
def breast(sample) -> dict:
    """The validation report of the breast cancer example over the sample."""
    return _validate(sample, BREAST_CANCER)


def test_i1_the_covid_cohort_is_the_independent_recount(sample, covid):
    index_dates = _index_dates(sample, COVID_CODE)
    assert covid["indexed_cohort"]["patients"] == len(index_dates)
    assert covid["indexed_cohort"]["patients"] == 4


def test_i2_every_covid_outcome_is_the_independent_recount(sample, covid):
    index_dates = _index_dates(sample, COVID_CODE)
    recount = {
        name: _event_patients(sample, table, codes, index_dates)
        for name, (table, codes) in COVID_EVENTS.items()
    }
    recount["Death"] = _death_patients(sample, index_dates)
    measured = {
        outcome["name"]: outcome["breakdown"]["total"]["numerator"]
        for outcome in covid["indexed_cohort"]["outcomes"]
    }
    assert measured == recount
    assert recount == {
        "Hospital admission": 1,
        "ICU admission": 0,
        "Ventilated": 0,
        "Death": 0,
    }


def test_i3_every_covid_patient_falls_in_one_stratum(covid):
    breakdown = _outcome(covid, "Hospital admission")["breakdown"]
    assert sum(cell["denominator"] for cell in breakdown["by_sex"]) == 4
    assert sum(cell["denominator"] for cell in breakdown["by_age"]) == 4


def test_i4_the_breast_cancer_cohort_and_deaths_are_the_independent_recount(sample, breast):
    index_dates = _index_dates(sample, BREAST_CANCER_CODE)
    deaths = _outcome(breast, "Death")["breakdown"]["total"]["numerator"]
    assert breast["indexed_cohort"]["patients"] == len(index_dates) == 1
    assert deaths == _death_patients(sample, index_dates) == 0


def test_i5_the_stage_group_patients_are_the_independent_recount(sample, breast):
    index_dates = _index_dates(sample, BREAST_CANCER_CODE)
    stage = _outcome(breast, "Stage group")["breakdown"]["total"]["numerator"]
    assert stage == _event_patients(sample, "observations", {STAGE_CODE}, index_dates)


def test_i6_only_a_module_with_an_index_writes_the_key(sample, covid):
    myocardial = _validate(sample, MYOCARDIAL_INFARCTION)
    assert "indexed_cohort" in covid
    assert "indexed_cohort" not in myocardial
