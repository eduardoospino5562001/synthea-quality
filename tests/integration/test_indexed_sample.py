"""Recount the indexed cohorts of the example module files over the official sample.

Every figure the tool reports is recounted here with the standard ``csv`` module and
``datetime`` only, from ``patients.csv``, ``conditions.csv``, ``encounters.csv``,
``devices.csv`` and ``observations.csv``, so the expected numbers do not come from the
code under test. The dataset is fetched with ``python scripts/fetch_official_sample.py``
or pointed at with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import json
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
#: The observation whose VALUE holds the cause of death.
CAUSE_OF_DEATH_CODE = "69453-9"
#: The cause the breast cancer example counts.
BREAST_CANCER_CAUSE = "Malignant neoplasm of breast (disorder)"


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


def _event_members(
    sample: Path, table: str, codes: set[str], index_dates: dict[str, date]
) -> set[str]:
    """The cohort patients with a row of ``codes`` between their index and the reference."""
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
    return patients


def _event_patients(
    sample: Path, table: str, codes: set[str], index_dates: dict[str, date]
) -> int:
    """Distinct cohort patients with a row of ``codes`` between their index and the reference."""
    return len(_event_members(sample, table, codes, index_dates))


def _death_members(sample: Path, index_dates: dict[str, date]) -> set[str]:
    """The cohort patients whose ``DEATHDATE`` is between their index and the reference."""
    reference = date.fromisoformat(REFERENCE_DATE)
    patients = set()
    for row in _rows(sample / "patients.csv"):
        index = index_dates.get(row["Id"])
        if index is None:
            continue
        death = _day(row["DEATHDATE"])
        if death is not None and index <= death <= reference:
            patients.add(row["Id"])
    return patients


def _death_patients(sample: Path, index_dates: dict[str, date]) -> int:
    """Distinct cohort patients whose ``DEATHDATE`` is between their index and the reference."""
    return len(_death_members(sample, index_dates))


def _value_labels(
    sample: Path, code: str, index_dates: dict[str, date], groups: dict[str, str]
) -> dict[str, frozenset[str]]:
    """Each patient's labels of ``code`` on their first date with a value.

    The groups are applied first, so a patient with two rows of the same group on that
    date has one label; a text outside every group keeps its own text.
    """
    reference = date.fromisoformat(REFERENCE_DATE)
    by_patient: dict[str, dict[date, set[str]]] = {}
    for row in _rows(sample / "observations.csv"):
        if row["CODE"] != code:
            continue
        index = index_dates.get(row["PATIENT"])
        if index is None:
            continue
        recorded = _day(row["DATE"])
        if recorded is None or not index <= recorded <= reference:
            continue
        value = row["VALUE"]
        if not value:
            continue
        by_patient.setdefault(row["PATIENT"], {}).setdefault(recorded, set()).add(value)
    return {
        patient: frozenset(groups.get(text, text) for text in by_day[min(by_day)])
        for patient, by_day in by_patient.items()
    }


def _value_strata(
    labels: dict[str, frozenset[str]], cohort: set[str]
) -> dict[str, set[str]]:
    """The strata of a value outcome: each label and the patients under it.

    The last stratum, ``no value``, holds the patients of the cohort without a value.
    """
    strata: dict[str, set[str]] = {}
    for patient, patient_labels in labels.items():
        for label in patient_labels:
            strata.setdefault(label, set()).add(patient)
    valued: set[str] = set()
    for members in strata.values():
        valued |= members
    strata["no value"] = set(cohort) - valued
    return strata


def _cause_members(
    sample: Path, deaths: set[str], cause_values: set[str]
) -> set[str]:
    """The deceased patients whose cause of death observation names one of the causes."""
    patients = set()
    for row in _rows(sample / "observations.csv"):
        if row["CODE"] != CAUSE_OF_DEATH_CODE or row["PATIENT"] not in deaths:
            continue
        if row["VALUE"] in cause_values:
            patients.add(row["PATIENT"])
    return patients


def _stage_groups(module_path: Path) -> dict[str, str]:
    """Each ``VALUE`` text of the example's stage groups under its label.

    The groups are input data of the example, read from the file and not from the code
    under test, so the recount below does not share any logic with the tool.
    """
    with module_path.open(encoding="utf-8") as handle:
        module = json.load(handle)
    groups = next(
        outcome["groups"] for outcome in module["outcomes"] if outcome["name"] == "Stage group"
    )
    return {text: label for label, texts in groups.items() for text in texts}


def _recovered_parts(module_path: Path) -> list[dict]:
    """The parts of the "Recovered" event of the example, read from the file.

    The parts are input data of the example, so the recount below does not share any
    logic with the tool.
    """
    with module_path.open(encoding="utf-8") as handle:
        module = json.load(handle)
    outcome = next(o for o in module["outcomes"] if o["name"] == "Recovered")
    return outcome["match"]


def _module_with_op(directory: Path, op: str) -> Path:
    """The COVID-19 example with the "Recovered" event in the given ``op``."""
    with COVID.open(encoding="utf-8") as handle:
        module = json.load(handle)
    for outcome in module["outcomes"]:
        if outcome["name"] == "Recovered":
            outcome["op"] = op
    path = directory / f"covid19-{op}.json"
    path.write_text(json.dumps(module), encoding="utf-8")
    return path


def _part_days(
    sample: Path, part: dict, index_dates: dict[str, date]
) -> dict[str, int]:
    """The cohort patients of a part of an event, and the days to their first record.

    A record counts when it is of one of the codes, of a patient of the cohort, and
    inside the window: on or after the index date and on or before the reference date.
    The filters are exact texts of ``VALUE`` and of ``REASONCODE``, and a part that
    counts on the end of its record needs a ``STOP`` and counts on that date.
    """
    reference = date.fromisoformat(REFERENCE_DATE)
    table = part["table"]
    column = "STOP" if part.get("at") == "stop" else DATE_COLUMNS.get(table, "START")
    values = set(part.get("values", ()))
    reasons = set(part.get("reasons", ()))
    days: dict[str, int] = {}
    for row in _rows(sample / f"{table}.csv"):
        if row["CODE"] not in part["codes"]:
            continue
        index = index_dates.get(row["PATIENT"])
        if index is None:
            continue
        if values and row.get("VALUE", "") not in values:
            continue
        if reasons and row.get("REASONCODE", "") not in reasons:
            continue
        recorded = _day(row[column])
        if recorded is None or recorded > reference:
            continue
        day = (recorded - index).days
        if day < 0:
            continue
        patient = row["PATIENT"]
        if patient not in days or day < days[patient]:
            days[patient] = day
    return days


def _outcome_days(parts: list[dict[str, int]], op: str) -> dict[str, int]:
    """The patients of an event of several parts, and the day each of them reaches it.

    With ``any`` a patient counts with one part and reaches the outcome on the earliest
    of them; with ``all`` the patient needs every part and reaches it on the latest.
    """
    if op == "all":
        common = set(parts[0])
        for part in parts[1:]:
            common &= set(part)
        return {
            patient: max(part[patient] for part in parts) for patient in common
        }
    days: dict[str, int] = {}
    for part in parts:
        for patient, day in part.items():
            if patient not in days or day < days[patient]:
                days[patient] = day
    return days


def _median(days: list[int]) -> float | None:
    """The median of the days of some patients, or ``None`` when there are none."""
    if not days:
        return None
    ordered = sorted(days)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


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
        crosses=loaded.crosses,
        reference_date=REFERENCE_DATE,
    )
    return report.to_dict()


def _outcome(report: dict, name: str) -> dict:
    """The outcome of the indexed cohort named ``name``."""
    outcomes = report["indexed_cohort"]["outcomes"]
    return next(outcome for outcome in outcomes if outcome["name"] == name)


def _cross(report: dict, outcome: str, by: str) -> dict:
    """The crossing of ``outcome`` by ``by`` in the report."""
    crosses = report["indexed_cohort"]["crosses"]
    return next(
        cross for cross in crosses if cross["outcome"] == outcome and cross["by"] == by
    )


def _stratum(cross: dict, label: str) -> dict:
    """The cell of ``label`` in the crossing."""
    return next(stratum for stratum in cross["strata"] if stratum["label"] == label)


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
        if outcome["name"] in recount
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


def test_i7_the_patient_with_two_stage_rows_counts_once_in_its_group(sample, breast):
    groups = _stage_groups(BREAST_CANCER)
    assert groups["American Joint Committee on Cancer stage IIA (qualifier value)"] == "Stage 2"
    assert groups["Stage 2 (qualifier value)"] == "Stage 2"
    index_dates = _index_dates(sample, BREAST_CANCER_CODE)
    labels = _value_labels(sample, STAGE_CODE, index_dates, groups)
    assert len(labels) == 1
    assert set(next(iter(labels.values()))) == {"Stage 2"}
    values = _outcome(breast, "Stage group")["values"]
    assert [(share["value"], share["patients"]) for share in values] == [("Stage 2", 1)]


def test_i8_the_stage_cross_is_the_independent_recount(sample, breast):
    index_dates = _index_dates(sample, BREAST_CANCER_CODE)
    labels = _value_labels(sample, STAGE_CODE, index_dates, _stage_groups(BREAST_CANCER))
    strata = _value_strata(labels, set(index_dates))
    deaths = _death_members(sample, index_dates)
    with_cause = _cause_members(sample, deaths, {BREAST_CANCER_CAUSE})
    cross = _cross(breast, "Death", "Stage group")
    measured = {
        stratum["label"]: (stratum["numerator"], stratum["denominator"])
        for stratum in cross["strata"]
    }
    recount = {
        label: (len(members & deaths), len(members)) for label, members in strata.items()
    }
    assert measured == recount
    assert measured == {"Stage 2": (0, 1), "no value": (0, 0)}
    assert cross["patients_in_several_strata"] == 0
    causes = {stratum["label"]: stratum["with_cause"] for stratum in cross["strata"]}
    assert causes == {label: len(members & with_cause) for label, members in strata.items()}


def test_i9_the_covid_cross_is_the_independent_recount(sample, covid):
    index_dates = _index_dates(sample, COVID_CODE)
    icu = _event_members(sample, "encounters", {"305351004"}, index_dates)
    deaths = _death_members(sample, index_dates)
    strata = {
        "with ICU admission": icu,
        "without ICU admission": set(index_dates) - icu,
    }
    cross = _cross(covid, "Death", "ICU admission")
    measured = {
        stratum["label"]: (stratum["numerator"], stratum["denominator"])
        for stratum in cross["strata"]
    }
    recount = {
        label: (len(members & deaths), len(members)) for label, members in strata.items()
    }
    assert measured == recount
    assert measured == {"with ICU admission": (0, 0), "without ICU admission": (0, 4)}
    assert cross["patients_in_several_strata"] == 0


def test_i10_the_denominators_of_a_cross_add_up_to_the_cohort(sample, covid, breast):
    for report in (covid, breast):
        cross = report["indexed_cohort"]["crosses"][0]
        assert cross["patients_in_several_strata"] == 0
        denominators = sum(stratum["denominator"] for stratum in cross["strata"])
        assert denominators == report["indexed_cohort"]["patients"]


def test_i11_the_parts_of_the_recovered_outcome_are_the_independent_recount(sample, covid):
    index_dates = _index_dates(sample, COVID_CODE)
    parts = _recovered_parts(COVID)
    recounted = [_part_days(sample, part, index_dates) for part in parts]
    measured = _outcome(covid, "Recovered")["matches"]
    assert [(match["name"], match["patients"]) for match in measured] == [
        (part["name"], len(days)) for part, days in zip(parts, recounted)
    ]


def test_i12_the_recovered_outcome_is_the_independent_recount(sample, covid):
    index_dates = _index_dates(sample, COVID_CODE)
    parts = _recovered_parts(COVID)
    recounted = [_part_days(sample, part, index_dates) for part in parts]
    days = _outcome_days(recounted, "any")
    outcome = _outcome(covid, "Recovered")
    assert outcome["table"] is None
    assert outcome["codes"] == []
    assert outcome["op"] == "any"
    assert outcome["breakdown"]["total"]["numerator"] == len(days)
    assert outcome["days_to_first_record"]["median"] == _median(list(days.values()))


def test_i13_the_recovered_outcome_of_every_part_is_the_independent_recount(
    sample, tmp_path
):
    index_dates = _index_dates(sample, COVID_CODE)
    parts = _recovered_parts(COVID)
    recounted = [_part_days(sample, part, index_dates) for part in parts]
    days = _outcome_days(recounted, "all")
    report = _validate(sample, _module_with_op(tmp_path, "all"))
    outcome = _outcome(report, "Recovered")
    assert outcome["op"] == "all"
    assert outcome["breakdown"]["total"]["numerator"] == len(days)
    assert outcome["days_to_first_record"]["median"] == _median(list(days.values()))


def test_i14_the_covid_19_that_ended_is_the_independent_recount(sample, covid):
    index_dates = _index_dates(sample, COVID_CODE)
    part = {
        "name": "COVID-19 ended",
        "table": "conditions",
        "codes": {"840539006"},
        "at": "stop",
    }
    days = _part_days(sample, part, index_dates)
    outcome = _outcome(covid, "COVID-19 ended")
    assert outcome["breakdown"]["total"]["numerator"] == len(days)
    assert outcome["days_to_first_record"]["median"] == _median(list(days.values()))
