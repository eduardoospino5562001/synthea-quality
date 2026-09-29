"""The medications of ``examples/hypertension.json`` on the official Synthea sample.

The check is an **independent recount**: the rules are re-implemented here with the
``csv`` module and ``datetime`` only — nothing is imported from the tool — and
``synthea-validate-module`` must match it exactly:

* the cohort: alive patients with hypertension active at the reference date;
* for each medication, the patients of the cohort with a record active at the reference
  date and with any record started by then, their Wilson interval (recomputed here too),
  those whose record gives hypertension as ``REASONCODE``, and the records without ``STOP``;
* the note for a code that never appears in ``medications.csv`` (losartan, ``979485``).

The cohort must also be, key for key, the numerator ``synthea-prevalence`` reports for
hypertension with the same module file.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "hypertension.json"
DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
BIN = Path(sys.executable).parent
HYPERTENSION = "59621000"
Z = 1.959963984540054
MEDICATIONS = {
    "Lisinopril": ("314076",),
    "Hydrochlorothiazide": ("310798",),
    "Amlodipine": ("308136",),
    "Losartan": ("979485",),
    "Any of the four": ("314076", "310798", "308136", "979485"),
}


@pytest.fixture(scope="module")
def sample() -> Path:
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if not (candidate / "medications.csv").is_file():
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
    out = tmp_path_factory.mktemp("medications")
    run("synthea-validate-module", "synthea_quality.validate.cli",
        str(sample), "--module", str(EXAMPLE), "--output-dir", str(out / "validate"))
    run("synthea-prevalence", "synthea_quality.prevalence.cli",
        str(sample), "--conditions", str(EXAMPLE), "--output-dir", str(out / "prevalence"))
    return {
        "validate": json.loads(
            (out / "validate" / "synthea_module_validation.json").read_text("utf-8")
        ),
        "prevalence": json.loads(
            (out / "prevalence" / "synthea_prevalence.json").read_text("utf-8")
        ),
        "markdown": (out / "validate" / "synthea_module_validation.md").read_text("utf-8"),
    }


def rows(sample: Path, name: str):
    with (sample / f"{name}.csv").open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = (Z / (1 + Z * Z / n)) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return round(low, 6), round(high, 6)


@pytest.fixture(scope="module")
def recount(sample) -> dict:
    ref = max(
        value[:10] for row in rows(sample, "encounters")
        for value in (row["START"], row["STOP"]) if value
    )
    alive = {row["Id"] for row in rows(sample, "patients") if not row["DEATHDATE"]}
    cohort = {
        row["PATIENT"] for row in rows(sample, "conditions")
        if row["CODE"] == HYPERTENSION and row["PATIENT"] in alive and row["START"][:10] <= ref
        and (not row["STOP"] or row["STOP"][:10] > ref)
    }
    table_codes = set()
    counts = {}
    records = list(rows(sample, "medications"))
    for record in records:
        table_codes.add(record["CODE"])
    for name, codes in MEDICATIONS.items():
        ever, active, ever_reason, active_reason, used, without_stop = (
            set(), set(), set(), set(), 0, 0,
        )
        for record in records:
            if record["CODE"] not in codes or record["PATIENT"] not in cohort:
                continue
            if record["START"][:10] > ref:
                continue
            used += 1
            without_stop += not record["STOP"]
            is_active = not record["STOP"] or record["STOP"][:10] > ref
            ever.add(record["PATIENT"])
            if is_active:
                active.add(record["PATIENT"])
            if record["REASONCODE"] == HYPERTENSION:
                ever_reason.add(record["PATIENT"])
                if is_active:
                    active_reason.add(record["PATIENT"])
        counts[name] = {
            "active": len(active), "ever": len(ever), "active_with_reason": len(active_reason),
            "ever_with_reason": len(ever_reason), "records": used,
            "records_without_stop": without_stop,
            "missing": [c for c in codes if c not in table_codes],
        }
    return {"ref": date.fromisoformat(ref), "cohort": cohort, "counts": counts}


def test_the_cohort_is_the_prevalence_numerator(reports, recount):
    (hypertension,) = reports["prevalence"]["conditions"]
    assert hypertension["point"]["numerator"] == len(recount["cohort"]) == 17
    for medication in reports["validate"]["medications"]:
        assert medication["active"]["denominator"] == hypertension["point"]["numerator"]
        assert medication["cohort"] == {"condition": "Hypertension", "rule": "point"}
    assert reports["validate"]["conditions"][0]["prevalence"] == hypertension


@pytest.mark.parametrize("index, name", list(enumerate(MEDICATIONS)))
def test_each_medication_matches_the_recount(reports, recount, index, name):
    medication = reports["validate"]["medications"][index]
    expected = recount["counts"][name]
    n = len(recount["cohort"])
    assert medication["name"] == name
    for measure in ("active", "ever"):
        share = medication[measure]
        assert (share["numerator"], share["denominator"]) == (expected[measure], n)
        assert (share["ci95_low"], share["ci95_high"]) == wilson(expected[measure], n)
    assert medication["active_with_reason"] == expected["active_with_reason"]
    assert medication["ever_with_reason"] == expected["ever_with_reason"]
    assert medication["metrics"]["records"] == expected["records"]
    assert medication["metrics"]["records_without_stop"] == expected["records_without_stop"]
    missing_note = any("No record of" in note for note in medication["notes"])
    assert missing_note == bool(expected["missing"])


def test_the_example_on_the_sample(reports):
    shares = {m["name"]: (m["active"]["numerator"], m["ever"]["numerator"])
              for m in reports["validate"]["medications"]}
    assert shares == {
        "Lisinopril": (14, 14), "Hydrochlorothiazide": (11, 11), "Amlodipine": (5, 5),
        "Losartan": (0, 0), "Any of the four": (17, 17),
    }
    assert "No record of 979485 in medications.csv" in reports["markdown"]
