"""End-to-end run of ``synthea-observations`` over the official Synthea sample.

The check is an **independent recount**: the rules are re-implemented here with the
``csv`` module, ``datetime`` and ``math`` only — nothing is imported from the tool — and
the report must match exactly (to the six decimals it keeps):

* systolic and diastolic blood pressure among the alive and among the alive patients with
  hypertension active at the reference date: n, minimum, the 5th, 25th, 50th, 75th and
  95th percentiles (type 7) and maximum, in total, by sex and by age band;
* the patients below, within and above the example's reference range;
* what happened to every row of those codes;
* the codes written in several units and with several ``TYPE`` values over the whole table.

It also checks that ``synthea-validate-module`` gives, key for key, the observations
``synthea-observations`` gives with the same module file.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "hypertension.json"
DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
BIN = Path(sys.executable).parent
BANDS = (0, 5, 18, 45, 65)
HYPERTENSION = "59621000"
PERCENTILES = (5, 25, 50, 75, 95)


@pytest.fixture(scope="module")
def sample() -> Path:
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if not (candidate / "observations.csv").is_file():
        pytest.skip(
            "official sample not available; run `python scripts/fetch_official_sample.py` "
            "or set SYNTHEA_QUALITY_SAMPLE to an extracted csv-latest directory"
        )
    return candidate


def run(command: str, module: str, *arguments: str) -> None:
    script = BIN / command
    base = [str(script)] if script.exists() else [sys.executable, "-m", module]
    completed = subprocess.run([*base, *arguments], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="module")
def reports(sample, tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("observations")
    run("synthea-observations", "synthea_quality.observations.cli",
        str(sample), "--module", str(EXAMPLE), "--output-dir", str(out / "observations"))
    run("synthea-validate-module", "synthea_quality.validate.cli",
        str(sample), "--module", str(EXAMPLE), "--output-dir", str(out / "validate"))
    return {
        "observations": json.loads(
            (out / "observations" / "synthea_observations.json").read_text("utf-8")
        ),
        "validate": json.loads(
            (out / "validate" / "synthea_module_validation.json").read_text("utf-8")
        ),
    }


# --------------------------------------------------------------------------- #
# the independent recount
# --------------------------------------------------------------------------- #


def rows(sample: Path, name: str):
    with (sample / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def reference_date(sample: Path) -> date:
    days = [
        value[:10] for row in rows(sample, "encounters") for value in (row["START"], row["STOP"])
        if value
    ]
    return date.fromisoformat(max(days))


def age(birth: date, day: date) -> int:
    return day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))


def band(years: int) -> str:
    low = max(b for b in BANDS if b <= years)
    index = BANDS.index(low)
    return f"{low}+" if index == len(BANDS) - 1 else f"{low}-{BANDS[index + 1] - 1}"


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * p / 100
    low = math.floor(h)
    if low + 1 >= len(ordered):
        return ordered[-1]
    return ordered[low] + (h - low) * (ordered[low + 1] - ordered[low])


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, **{k: None for k in ("min", "p5", "p25", "median", "p75", "p95", "max")}}
    p5, p25, p50, p75, p95 = (round(percentile(values, p), 6) for p in PERCENTILES)
    return {"n": len(values), "min": round(min(values), 6), "p5": p5, "p25": p25,
            "median": p50, "p75": p75, "p95": p95, "max": round(max(values), 6)}


@pytest.fixture(scope="module")
def recount(sample) -> dict:
    ref = reference_date(sample)
    patients = {row["Id"]: row for row in rows(sample, "patients")}
    alive = {pid for pid, row in patients.items() if not row["DEATHDATE"]}
    hypertensive = {
        row["PATIENT"] for row in rows(sample, "conditions")
        if row["CODE"] == HYPERTENSION and row["PATIENT"] in alive
        and row["START"][:10] <= ref.isoformat()
        and (not row["STOP"] or row["STOP"][:10] > ref.isoformat())
    }
    by_patient: dict[tuple[str, str], dict[str, list]] = defaultdict(dict)
    fates: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    units_by_code: dict[str, set] = defaultdict(set)
    types_by_code: dict[str, set] = defaultdict(set)
    for row in rows(sample, "observations"):
        code = row["CODE"]
        types_by_code[code].add(row["TYPE"])
        if row["TYPE"] == "numeric":
            units_by_code[code].add(row["UNITS"])
        if row["PATIENT"] not in patients:
            fates[code]["unknown_patient"] += 1
            continue
        if row["PATIENT"] not in alive:
            fates[code]["deceased"] += 1
            continue
        if row["DATE"][:10] > ref.isoformat():
            fates[code]["after_reference"] += 1
            continue
        if row["TYPE"] != "numeric":
            fates[code]["not_numeric_type"] += 1
            continue
        try:
            value = float(row["VALUE"])
        except ValueError:
            fates[code]["value_unparseable"] += 1
            continue
        fates[code]["used"] += 1
        key = (code, row["UNITS"])
        latest = by_patient[key].get(row["PATIENT"])
        if latest is None or row["DATE"] > latest[0]:
            by_patient[key][row["PATIENT"]] = [row["DATE"], [value]]
        elif row["DATE"] == latest[0]:
            latest[1].append(value)
    values = {
        key: {pid: statistics.median(vals) for pid, (_, vals) in people.items()}
        for key, people in by_patient.items()
    }
    return {
        "ref": ref,
        "patients": patients,
        "alive": alive,
        "hypertensive": hypertensive,
        "values": values,
        "fates": fates,
        "mixed_units": sorted(c for c, u in units_by_code.items() if len(u) > 1),
        "mixed_types": sorted(c for c, t in types_by_code.items() if len(t) > 1),
    }


def expected_group(recount: dict, code: str, members: set) -> dict:
    ref = recount["ref"]
    values = {p: v for p, v in recount["values"][(code, "mm[Hg]")].items() if p in members}
    strata = []
    ages = {p: age(date.fromisoformat(recount["patients"][p]["BIRTHDATE"]), ref) for p in members}
    for label in ("0-4", "5-17", "18-44", "45-64", "65+"):
        group = {p for p in members if band(ages[p]) == label}
        strata.append(("age_band", label, len(group),
                       summary([v for p, v in values.items() if p in group])))
    for sex in sorted({recount["patients"][p]["GENDER"] for p in members}):
        group = {p for p in members if recount["patients"][p]["GENDER"] == sex}
        strata.append(("GENDER", sex, len(group),
                       summary([v for p, v in values.items() if p in group])))
    return {"summary": summary(list(values.values())), "strata": strata, "values": values}


def check(observation: dict, expected: dict) -> None:
    (group,) = observation["groups"]
    assert group["units"] == "mm[Hg]"
    assert group["summary"] == expected["summary"]
    got = [(s["dimension"], s["value"], s["patients"], s["summary"]) for s in group["strata"]]
    assert got == expected["strata"]


# --------------------------------------------------------------------------- #
# the tests
# --------------------------------------------------------------------------- #


def test_the_reference_date_and_the_population(reports, recount):
    data = reports["observations"]
    assert data["reference_date"]["value"] == recount["ref"].isoformat()
    assert data["alive"] == len(recount["alive"]) == 99
    assert len(recount["hypertensive"]) == 17


@pytest.mark.parametrize("index, code", [(0, "8480-6"), (1, "8462-4")])
def test_blood_pressure_among_the_alive(reports, recount, index, code):
    observation = reports["observations"]["observations"][index]
    expected = expected_group(recount, code, recount["alive"])
    check(observation, expected)
    rng = observation["groups"][0]["reference_range"]
    values = expected["values"].values()
    assert rng["below"] == sum(v < rng["low"] for v in values)
    assert rng["above"] == sum(v > rng["high"] for v in values)
    assert rng["within"] == len(values) - rng["below"] - rng["above"]
    assert rng["basis"] == "synthea-configuration" and rng["note"]


@pytest.mark.parametrize("index, code", [(2, "8480-6"), (3, "8462-4")])
def test_blood_pressure_among_the_hypertensive(reports, recount, index, code):
    observation = reports["observations"]["observations"][index]
    assert observation["cohort"] == {"condition": "Hypertension", "rule": "point"}
    assert observation["population"] == len(recount["hypertensive"])
    check(observation, expected_group(recount, code, recount["hypertensive"]))


@pytest.mark.parametrize("code", ["8480-6", "8462-4"])
def test_every_row_is_accounted_for(reports, recount, code):
    metrics = reports["observations"]["observations"][0 if code == "8480-6" else 1]["metrics"]
    for fate in ("unknown_patient", "deceased", "after_reference", "not_numeric_type",
                 "value_unparseable", "used"):
        assert metrics[f"rows_{fate}"] == recount["fates"][code][fate], fate


def test_codes_in_several_units_and_types(reports, recount):
    general = reports["observations"]["general"]
    assert [item["code"] for item in general["mixed_units"]] == recount["mixed_units"]
    assert [item["code"] for item in general["mixed_types"]] == recount["mixed_types"]
    assert {"788-0", "33914-3", "89579-7"} <= set(recount["mixed_units"])
    assert len(recount["mixed_types"]) == 7


def test_the_general_table_matches_the_recount(reports, recount):
    general = reports["observations"]["general"]["rows"]
    rows_by_key = {(r["code"], r["units"] or ""): r for r in general}
    assert set(rows_by_key) == set(recount["values"])
    for key, people in recount["values"].items():
        assert rows_by_key[key]["summary"] == summary(list(people.values())), key


def test_validate_module_gives_the_same_observations(reports):
    assert reports["validate"]["observations"] == reports["observations"]["observations"]
    assert reports["validate"]["alive"] == reports["observations"]["alive"]
