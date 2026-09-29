"""End-to-end run of ``synthea-incidence`` over the official Synthea sample.

No notebook of synthetichealth/module-validation computes incidence over person-time
(``MI Module Validation.ipynb`` calls a lifetime proportion an "incidence rate"; the
COVID-19 notebook counts cases per day without a denominator), so the check is an
**independent recount**: the definitions are re-implemented here with the ``csv`` module
and ``datetime`` only — nothing is imported from the tool's computation — and the events,
prior cases and person-days of myocardial infarction and of hypertension must match
exactly, in total, by sex and by age band.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-incidence")
MI_CODES = ("22298006", "401303003", "401314000")
HYPERTENSION_CODES = ("59621000",)
BANDS = (0, 5, 18, 45, 65)


@pytest.fixture(scope="module")
def sample() -> Path:
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if not (candidate / "conditions.csv").is_file():
        pytest.skip(
            "official sample not available; run `python scripts/fetch_official_sample.py` "
            "or set SYNTHEA_QUALITY_SAMPLE to an extracted csv-latest directory"
        )
    return candidate


def command() -> list[str]:
    if CONSOLE_SCRIPT.exists():
        return [str(CONSOLE_SCRIPT)]
    return [sys.executable, "-m", "synthea_quality.incidence.cli"]


def run(sample: Path, out: Path, *extra: str) -> dict:
    completed = subprocess.run(
        [
            *command(),
            str(sample),
            "--condition", f"Myocardial infarction={','.join(MI_CODES)};acute",
            "--condition", f"Hypertension={','.join(HYPERTENSION_CODES)}",
            "--output-dir", str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((out / "synthea_incidence.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report(sample, tmp_path_factory) -> dict:
    return run(sample, tmp_path_factory.mktemp("incidence"))


def read(sample: Path, table: str) -> list[dict[str, str]]:
    with (sample / f"{table}.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def birthday(birth: date, years: int) -> date:
    try:
        return birth.replace(year=birth.year + years)
    except ValueError:
        return date(birth.year + years, 3, 1)


def age(birth: date, day: date) -> int:
    return day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))


def recount(sample: Path, reference: date, window_start: date, codes, alive_only=False) -> dict:
    """Events, prior cases and person-days, re-derived from the definitions."""
    starts: dict[str, list[date]] = {}
    for row in read(sample, "conditions"):
        if row["CODE"] in codes and row["START"]:
            starts.setdefault(row["PATIENT"], []).append(date.fromisoformat(row["START"]))
    result = {"events": 0, "days": 0, "prior": 0, "followed": 0,
              "sex": {}, "band_days": [0] * len(BANDS), "band_events": [0] * len(BANDS)}
    for patient in read(sample, "patients"):
        if alive_only and patient["DEATHDATE"]:
            continue
        birth = date.fromisoformat(patient["BIRTHDATE"])
        death = date.fromisoformat(patient["DEATHDATE"]) if patient["DEATHDATE"] else None
        if death is not None and death < window_start:
            continue
        result["followed"] += 1
        entry = max(window_start, birth)
        exit_ = min(reference, death) if death else reference
        mine = sorted(starts.get(patient["Id"], []))
        if mine and mine[0] < entry:
            result["prior"] += 1
            continue
        event = next((s for s in mine if s <= exit_), None)
        end = event or exit_
        days = (end - entry).days
        result["days"] += days
        sex = result["sex"].setdefault(patient["GENDER"], [0, 0])
        sex[1] += days
        for i, low in enumerate(BANDS):
            lo = max(entry, birthday(birth, low))
            hi = end if i + 1 == len(BANDS) else min(end, birthday(birth, BANDS[i + 1]))
            result["band_days"][i] += max(0, (hi - lo).days)
        if event is not None:
            result["events"] += 1
            sex[0] += 1
            band = max(i for i, low in enumerate(BANDS) if age(birth, event) >= low)
            result["band_events"][band] += 1
    return result


def condition(report: dict, name: str) -> dict:
    return next(c for c in report["conditions"] if c["name"] == name)


@pytest.mark.parametrize(
    "name, codes", [("Myocardial infarction", MI_CODES), ("Hypertension", HYPERTENSION_CODES)]
)
def test_rates_match_an_independent_recount(sample, report, name, codes):
    reference = date.fromisoformat(report["window"]["end"])
    start = date.fromisoformat(report["window"]["start"])
    expected = recount(sample, reference, start, set(codes))
    got = condition(report, name)
    assert report["cohort"]["followed"] == expected["followed"]
    assert got["metrics"]["prior_cases"] == expected["prior"]
    assert (got["rate"]["events"], got["rate"]["person_days"]) == (
        expected["events"], expected["days"]
    )
    sex = {s["value"]: (s["events"], s["person_days"]) for s in got["strata"]
           if s["dimension"] == "GENDER"}
    assert sex == {k: tuple(v) for k, v in expected["sex"].items()}
    bands = [s for s in got["strata"] if s["dimension"] == "age_band"]
    assert [s["person_days"] for s in bands] == expected["band_days"]
    assert [s["events"] for s in bands] == expected["band_events"]
    assert sum(expected["band_days"]) == expected["days"]


def test_the_sample_has_a_single_new_myocardial_infarction(report):
    mi = condition(report, "Myocardial infarction")
    assert mi["rate"]["events"] == 1
    assert mi["acute"] is True
    # the NSTEMI written the same day as the MI is the same episode, not a repeated one
    assert mi["metrics"]["records_on_the_event_day"] == 1
    assert mi["metrics"]["later_records_not_counted"] == 0
    low, high = mi["rate"]["ci95_low"], mi["rate"]["ci95_high"]
    assert low < mi["rate"]["per_1000_person_years"] < high and high / max(low, 1e-9) > 100


def test_the_deceased_are_followed_until_death_unless_alive_only(sample, report, tmp_path):
    reference = date.fromisoformat(report["window"]["end"])
    start = date.fromisoformat(report["window"]["start"])
    alive = run(sample, tmp_path, "--alive-only")
    expected = recount(sample, reference, start, set(HYPERTENSION_CODES), alive_only=True)
    got = condition(alive, "Hypertension")
    assert alive["population"] == "alive"
    assert (got["rate"]["events"], got["rate"]["person_days"]) == (
        expected["events"], expected["days"]
    )
    assert alive["cohort"]["followed"] < report["cohort"]["followed"]
    assert report["cohort"]["excluded"]["died_before_window"] == sum(
        1 for p in read(sample, "patients")
        if p["DEATHDATE"] and date.fromisoformat(p["DEATHDATE"]) < start
    )
