"""Compare two datasets over the official sample and recount the changes with ``csv``.

The command under test pairs the figures of two module validations. This file checks
that pairing against the dataset itself, never against the code under test: the
alive patients, the point and lifetime prevalence of myocardial infarction, the first
events and the person-days of its incidence window are recounted here with the
standard ``csv`` module and ``datetime`` only, from ``patients.csv`` and
``conditions.csv``.

The candidate of the main case is a copy of the sample with the myocardial infarction
rows of two patients removed: one holds the only first event inside the five-year
window, the other is a prior case. Every other table is a symlink to the sample, so
only ``conditions.csv`` differs and the population cannot change. The recount expects
point and lifetime prevalence to fall from 6 to 4 of 99 alive patients and the
incidence to go from 1 event in 169,062 person-days to 0 in 171,679.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterator
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from synthea_quality.compare.cli import main

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
MYOCARDIAL_INFARCTION = REPOSITORY / "examples" / "myocardial_infarction.json"
COVID = REPOSITORY / "examples" / "covid19.json"

#: Both sides are measured at this date so the recount needs no metadata.
REFERENCE_DATE = "2026-01-01"
#: The incidence window of the command when ``--window-years`` is not given.
WINDOW_YEARS = 5
#: A year in days, as the incidence report counts person-time.
DAYS_PER_YEAR = 365.25
#: The line of an incidence rate: events per thousand person-years.
PER_THOUSAND = 1000.0
#: The limit of one GitHub comment; the Markdown has to fit in one.
GITHUB_COMMENT_LIMIT = 65_536
#: The reports keep six decimals of a value and of a difference; the recount compares
#: them at that precision.
REPORTED_DECIMALS = 1e-6
#: The codes of the myocardial infarction example.
MI_CODES = {"22298006", "401303003", "401314000"}
#: Two alive patients with myocardial infarction rows: the first one holds the only
#: first event inside the window (2023-11-02, in the 65+ band); the second one is a
#: prior case (2020-03-13, before the window), in the 45-64 band. Both are male.
REMOVED_PATIENTS = {
    "1d63466e-0cfd-eb12-77aa-d153393355e5",
    "9c5b16dc-8a47-896c-4d74-f96f8c98891b",
}

#: Words the report never states (RULES section 2.6): differences are facts, not verdicts.
VERDICT = re.compile(r"pass|fail|\bok\b|match|significant", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Run:
    """One run of the command: its exit code, its JSON, its Markdown and its console."""

    exit_code: int
    data: dict[str, Any]
    markdown: str
    console: str


def _rows(path: Path) -> Iterator[dict[str, str]]:
    """The rows of one CSV file as read by the standard ``csv`` module."""
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _day(value: str) -> date | None:
    """The date of a CSV value, or ``None`` when it is empty or not ISO ``YYYY-MM-DD``."""
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _comparison(
    data: dict[str, Any], block: str, element: str, measure: str, stratum: str = ""
) -> dict[str, Any]:
    """The comparison of one figure, named by its four key parts."""
    for comparison in data["comparisons"]:
        key = comparison["key"]
        if (key["block"], key["element"], key["measure"], key["stratum"]) == (
            block,
            element,
            measure,
            stratum,
        ):
            return comparison
    raise AssertionError(f"no comparison for {block}/{element}/{measure}/{stratum!r}")


def _alive(directory: Path) -> set[str]:
    """The alive patients: those whose ``DEATHDATE`` is empty."""
    return {
        row["Id"] for row in _rows(directory / "patients.csv") if not row["DEATHDATE"].strip()
    }


def _prevalence(directory: Path) -> tuple[set[str], set[str]]:
    """The alive patients with point and with lifetime prevalence of the condition.

    Point: a record starting on or before the reference date that has no ``STOP`` or
    one after it. Lifetime: any record starting on or before the reference date. A
    patient counts once, whichever of the codes they have.
    """
    reference = date.fromisoformat(REFERENCE_DATE)
    alive = _alive(directory)
    point: set[str] = set()
    lifetime: set[str] = set()
    for row in _rows(directory / "conditions.csv"):
        patient = row["PATIENT"]
        if row["CODE"] not in MI_CODES or patient not in alive:
            continue
        start = _day(row["START"])
        if start is None or start > reference:
            continue
        lifetime.add(patient)
        stop = _day(row["STOP"])
        if not row["STOP"].strip() or (stop is not None and stop > reference):
            point.add(patient)
    return point, lifetime


def _incidence(directory: Path) -> tuple[int, int]:
    """The first events and the person-days of the incidence window, recounted.

    Entry is the later of the window start and the birth date; the patient leaves at
    the earliest of the reference date, the death and the first event. A record before
    entry is a prior case: the patient is not at risk. Person-time is whole days.
    """
    reference = date.fromisoformat(REFERENCE_DATE)
    window_start = reference.replace(year=reference.year - WINDOW_YEARS)
    starts: dict[str, list[date]] = {}
    for row in _rows(directory / "conditions.csv"):
        if row["CODE"] not in MI_CODES:
            continue
        start = _day(row["START"])
        if start is not None:
            starts.setdefault(row["PATIENT"], []).append(start)

    events = 0
    person_days = 0
    for patient in _rows(directory / "patients.csv"):
        patient_id = patient["Id"]
        birth = _day(patient["BIRTHDATE"])
        death = _day(patient["DEATHDATE"])
        unusable_death = bool(patient["DEATHDATE"].strip()) and death is None
        if birth is None or birth > reference or unusable_death:
            continue
        entry = max(window_start, birth)
        exit_ = reference if death is None else min(reference, death)
        if exit_ < entry:
            continue
        ordered = sorted(starts.get(patient_id, []))
        if any(start < entry for start in ordered):
            continue
        event = next((start for start in ordered if entry <= start <= exit_), None)
        if event is not None:
            events += 1
            exit_ = event
        person_days += (exit_ - entry).days
    return events, person_days


def _rate(events: int, person_days: int) -> float:
    """Events per thousand person-years."""
    return events / person_days * DAYS_PER_YEAR * PER_THOUSAND


#: The age bands of the reports: lower bound in completed years and label.
AGE_BANDS = ((0, "0-4"), (5, "5-17"), (18, "18-44"), (45, "45-64"), (65, "65+"))


def _age_band(birth: date, reference: date) -> str:
    """The age band of a patient born on ``birth``, in completed years at ``reference``."""
    age = reference.year - birth.year - ((reference.month, reference.day) < (birth.month, birth.day))
    return next(label for lower, label in reversed(AGE_BANDS) if age >= lower)


def _strata(directory: Path, patient_id: str) -> set[str]:
    """The ``GENDER`` and age-band strata of one patient, recounted from ``patients.csv``."""
    reference = date.fromisoformat(REFERENCE_DATE)
    for row in _rows(directory / "patients.csv"):
        if row["Id"] == patient_id:
            birth = _day(row["BIRTHDATE"])
            assert birth is not None, patient_id
            return {f"GENDER:{row['GENDER']}", f"age_band:{_age_band(birth, reference)}"}
    raise AssertionError(f"no patient {patient_id}")


def _first_event_in_window(directory: Path, patient_id: str) -> bool:
    """Whether the patient's first MI record falls inside the incidence window."""
    reference = date.fromisoformat(REFERENCE_DATE)
    window_start = reference.replace(year=reference.year - WINDOW_YEARS)
    starts = [
        start
        for row in _rows(directory / "conditions.csv")
        if row["PATIENT"] == patient_id and row["CODE"] in MI_CODES
        for start in (_day(row["START"]),)
        if start is not None
    ]
    return bool(starts) and window_start <= min(starts) <= reference


def _without_mi_rows(sample: Path, directory: Path) -> int:
    """A copy of ``sample`` with the MI rows of ``REMOVED_PATIENTS`` left out.

    Every table but ``conditions.csv`` is a symbolic link to the sample, so the copy
    differs from it in those rows alone. Returns how many rows were left out.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for path in sample.iterdir():
        if path.name != "conditions.csv" and path.is_file():
            (directory / path.name).symlink_to(path)
    removed = 0
    kept: list[dict[str, str]] = []
    with (sample / "conditions.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if row["PATIENT"] in REMOVED_PATIENTS and row["CODE"] in MI_CODES:
                removed += 1
            else:
                kept.append(row)
    with (directory / "conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return removed


def _compare(baseline: Path, candidate: Path, module: Path, output: Path) -> Run:
    """Run the command as a user would and read back the two reports it wrote."""
    console = io.StringIO()
    exit_code = 0
    with redirect_stdout(console):
        exit_code = main(
            [
                str(baseline),
                str(candidate),
                "--module",
                str(module),
                "--reference-date",
                REFERENCE_DATE,
                "--output-dir",
                str(output),
            ]
        )
    markdown = (output / "synthea_compare.md").read_text(encoding="utf-8")
    data = json.loads((output / "synthea_compare.json").read_text(encoding="utf-8"))
    return Run(exit_code=exit_code, data=data, markdown=markdown, console=console.getvalue())


@pytest.fixture(scope="module")
def mi_candidate(sample: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The sample without the MI rows of the two chosen patients."""
    directory = tmp_path_factory.mktemp("compare-candidate")
    assert _without_mi_rows(sample, directory) == 4
    return directory


@pytest.fixture(scope="module")
def mi_run(
    sample: Path, mi_candidate: Path, tmp_path_factory: pytest.TempPathFactory
) -> Run:
    """The comparison of the sample and its reduced copy with the MI example."""
    return _compare(
        sample, mi_candidate, MYOCARDIAL_INFARCTION, tmp_path_factory.mktemp("compare-mi")
    )


@pytest.fixture(scope="module")
def self_run(sample: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """The sample compared with itself and the MI example."""
    return _compare(sample, sample, MYOCARDIAL_INFARCTION, tmp_path_factory.mktemp("compare-self"))


@pytest.fixture(scope="module")
def covid_run(sample: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """The sample compared with itself and the COVID-19 example."""
    return _compare(sample, sample, COVID, tmp_path_factory.mktemp("compare-covid"))


def test_c1_the_sample_against_itself_reports_no_change(self_run: Run) -> None:
    """The same dataset on both sides: every figure is unchanged."""
    assert self_run.exit_code == 0
    assert self_run.data["compared"] == 46
    assert self_run.data["changed"] == 0
    assert self_run.data["unchanged"] == 46
    assert self_run.data["skipped"] == 0
    assert self_run.data["only_in_baseline"] == 0
    assert self_run.data["only_in_candidate"] == 0
    assert {comparison["state"] for comparison in self_run.data["comparisons"]} == {
        "unchanged"
    }


def test_c2_the_mi_prevalence_is_the_independent_recount(
    sample: Path, mi_candidate: Path, mi_run: Run
) -> None:
    """Point and lifetime prevalence match the ``csv`` recount before and after."""
    assert _alive(sample) == _alive(mi_candidate)
    assert len(_alive(sample)) == 99
    base_point, base_lifetime = _prevalence(sample)
    cand_point, cand_lifetime = _prevalence(mi_candidate)
    assert (len(base_point), len(base_lifetime)) == (6, 6)
    assert (len(cand_point), len(cand_lifetime)) == (4, 4)

    for measure, baseline, candidate in (
        ("point", len(base_point), len(cand_point)),
        ("lifetime", len(base_lifetime), len(cand_lifetime)),
    ):
        comparison = _comparison(mi_run.data, "conditions", "Myocardial infarction", measure)
        assert comparison["state"] == "changed"
        assert comparison["baseline"]["unit"] == "proportion"
        assert comparison["baseline"]["numerator"] == baseline
        assert comparison["baseline"]["denominator"] == 99
        assert comparison["candidate"]["numerator"] == candidate
        assert comparison["candidate"]["denominator"] == 99
        assert comparison["baseline"]["value"] == pytest.approx(
            baseline / 99, abs=REPORTED_DECIMALS
        )
        assert comparison["candidate"]["value"] == pytest.approx(
            candidate / 99, abs=REPORTED_DECIMALS
        )
        assert comparison["difference"] == pytest.approx(
            (candidate - baseline) / 99, abs=REPORTED_DECIMALS
        )
        assert comparison["relative_difference"] == pytest.approx(
            (candidate - baseline) / baseline
        )


def test_c3_the_mi_incidence_is_the_independent_recount(
    sample: Path, mi_candidate: Path, mi_run: Run
) -> None:
    """First events and person-days match the ``csv`` recount before and after."""
    base_events, base_days = _incidence(sample)
    cand_events, cand_days = _incidence(mi_candidate)
    assert (base_events, base_days) == (1, 169_062)
    assert (cand_events, cand_days) == (0, 171_679)

    comparison = _comparison(mi_run.data, "conditions", "Myocardial infarction", "incidence")
    assert comparison["state"] == "changed"
    assert comparison["baseline"]["unit"] == "per_1000_person_years"
    assert (comparison["baseline"]["numerator"], comparison["baseline"]["denominator"]) == (
        base_events,
        base_days,
    )
    assert (comparison["candidate"]["numerator"], comparison["candidate"]["denominator"]) == (
        cand_events,
        cand_days,
    )
    base_rate = _rate(base_events, base_days)
    cand_rate = _rate(cand_events, cand_days)
    assert comparison["baseline"]["value"] == pytest.approx(
        base_rate, abs=REPORTED_DECIMALS
    )
    assert comparison["candidate"]["value"] == pytest.approx(
        cand_rate, abs=REPORTED_DECIMALS
    )
    assert comparison["difference"] == pytest.approx(
        cand_rate - base_rate, abs=REPORTED_DECIMALS
    )
    assert comparison["relative_difference"] == pytest.approx(-1.0)


def test_c4_the_figures_that_do_not_depend_on_the_removed_rows_are_unchanged(
    sample: Path, mi_run: Run
) -> None:
    """Only the figures of the two removed patients change; nothing is lost silently."""
    data = mi_run.data
    # The changed figures, recounted: point and lifetime prevalence in total and in
    # the sex and age band of each removed patient, and the incidence in total and in
    # the strata of the patient whose first event was inside the window.
    expected = {("point", ""), ("lifetime", ""), ("incidence", "")}
    for patient in REMOVED_PATIENTS:
        strata = _strata(sample, patient)
        expected |= {(measure, stratum) for measure in ("point", "lifetime") for stratum in strata}
        if _first_event_in_window(sample, patient):
            expected |= {("incidence", stratum) for stratum in strata}
    changed = [comparison for comparison in data["comparisons"] if comparison["state"] == "changed"]
    assert {(c["key"]["block"], c["key"]["element"]) for c in changed} == {("conditions", "Myocardial infarction")}
    assert {(c["key"]["measure"], c["key"]["stratum"]) for c in changed} == expected
    assert len(expected) == data["changed"] == 11
    assert data["only_in_baseline"] == 0
    assert data["only_in_candidate"] == 0
    assert data["skipped"] == 0
    population = [
        comparison for comparison in data["comparisons"] if comparison["key"]["block"] == "population"
    ]
    assert len(population) == 22
    assert {comparison["state"] for comparison in population} == {"unchanged"}
    # Both patients are male, in the 45-64 and 65+ bands: every other stratum keeps both
    # its numerator and its denominator.
    for stratum in ("age_band:0-4", "age_band:5-17", "age_band:18-44", "GENDER:F"):
        for measure in ("point", "lifetime"):
            comparison = _comparison(
                data, "conditions", "Myocardial infarction", measure, stratum
            )
            assert comparison["state"] == "unchanged"
            assert comparison["baseline"] == comparison["candidate"]


def test_c5_the_covid_indexed_cohort_appears_in_the_comparison(covid_run: Run) -> None:
    """With ``examples/covid19.json`` the indexed cohort is part of the comparison."""
    assert covid_run.exit_code == 0
    indexed = [
        comparison
        for comparison in covid_run.data["comparisons"]
        if comparison["key"]["block"] == "indexed"
    ]
    assert len(indexed) == 58
    assert {comparison["state"] for comparison in indexed} == {"unchanged"}
    cohort = _comparison(covid_run.data, "indexed", "cohort", "patients")
    assert cohort["baseline"]["value"] == 4
    assert cohort["candidate"]["value"] == 4
    admission = _comparison(
        covid_run.data, "indexed", "Hospital admission", "patients", "total:all"
    )
    assert admission["baseline"]["value"] == pytest.approx(0.25)
    assert admission["candidate"]["value"] == pytest.approx(0.25)


def _assert_no_verdict(text: str) -> None:
    """Fail with the surrounding text when a verdict word is in ``text``."""
    found = VERDICT.search(text)
    if found is not None:
        start = max(0, found.start() - 40)
        raise AssertionError(f"{found.group(0)!r} in the output: {text[start:found.end() + 40]!r}")


def test_c6_the_reports_and_the_console_state_no_verdict(
    mi_run: Run, covid_run: Run
) -> None:
    """No pass, fail, ok, match or significant in the Markdown, the JSON or the console."""
    for run in (mi_run, covid_run):
        for text in (run.markdown, json.dumps(run.data), run.console):
            _assert_no_verdict(text)


def test_c7_the_markdown_of_the_mi_pair_fits_in_one_github_comment(mi_run: Run) -> None:
    """The report of the changed pair is shorter than one GitHub comment."""
    assert len(mi_run.markdown) < GITHUB_COMMENT_LIMIT


def test_c8_two_runs_write_the_same_report(
    sample: Path, mi_candidate: Path, tmp_path: Path, mi_run: Run
) -> None:
    """The same two datasets and module give the same report apart from ``generated_at``."""
    again = _compare(sample, mi_candidate, MYOCARDIAL_INFARCTION, tmp_path / "again")
    assert again.exit_code == 0
    assert again.markdown == mi_run.markdown
    first = {key: value for key, value in mi_run.data.items() if key != "generated_at"}
    second = {key: value for key, value in again.data.items() if key != "generated_at"}
    assert first == second
