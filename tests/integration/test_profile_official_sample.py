"""End-to-end run of ``synthea-profile`` over the official Synthea sample.

What this proves: that the profile command, run as a user runs it, accounts for every
patient of the dataset Synthea recommends — the total matches the rows of
``patients.csv`` as the standard library's CSV reader counts them, and alive plus
deceased equals the total — that the approximate reference date is attributed as
such, and that the most common codes of ``conditions`` and ``observations`` (the largest
table) match an independent count made with the standard library's CSV reader. It deliberately pins no demographic number: those describe the sample, and the
acceptance baseline in ``docs/acceptance.md`` already pins the data itself.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-profile")


@pytest.fixture(scope="module")
def sample() -> Path:
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if not (candidate.is_dir() and (candidate / "patients.csv").is_file()):
        pytest.skip(
            "official sample not available; run `python scripts/fetch_official_sample.py` "
            "or set SYNTHEA_QUALITY_SAMPLE to an extracted csv-latest directory"
        )
    return candidate


def command() -> list[str]:
    if CONSOLE_SCRIPT.exists():
        return [str(CONSOLE_SCRIPT)]
    return [sys.executable, "-m", "synthea_quality.profile.cli"]


def run_profile(sample: Path, out: Path) -> tuple[subprocess.CompletedProcess, dict]:
    completed = subprocess.run(
        [*command(), str(sample), "--output-dir", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed, json.loads((out / "synthea_profile.json").read_text(encoding="utf-8"))


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        return sum(1 for row in reader if row)


def digest_of(directory: Path) -> str:
    return hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in sorted(directory.glob("*.csv")))
    ).hexdigest()


@pytest.fixture(scope="module")
def profiled(sample: Path, tmp_path_factory):
    before = digest_of(sample)
    out = tmp_path_factory.mktemp("profile")
    completed, data = run_profile(sample, out)
    assert digest_of(sample) == before, "the profile must never modify the dataset"
    return completed, data, out


def section(data: dict, section_id: str) -> dict:
    return next(s for s in data["sections"] if s["section_id"] == section_id)


def test_every_patient_is_accounted_for(sample, profiled):
    _, data, _ = profiled
    population = section(data, "population")
    assert population["status"] == "COMPUTED"
    metrics = population["metrics"]
    assert metrics["total"] == csv_rows(sample / "patients.csv")
    assert metrics["alive"] + metrics["deceased"] == metrics["total"]
    patients_input = next(i for i in data["inputs"] if i["table"] == "patients")
    assert patients_input["rows"] == metrics["total"]


def test_every_section_is_computed_and_the_reference_is_an_attributed_approximation(profiled):
    _, data, _ = profiled
    assert [s["section_id"] for s in data["sections"] if s["status"] != "COMPUTED"] == []
    reference = data["reference_date"]
    assert reference["source"] == "max_encounter_date"
    assert reference["approximate"] is True
    assert reference["detail"].startswith("APPROXIMATION")


def test_alive_distributions_and_age_bands_add_up(profiled):
    _, data, _ = profiled
    alive = section(data, "population")["metrics"]["alive"]
    for column in ("GENDER", "RACE", "ETHNICITY", "STATE"):
        rows = section(data, f"distribution.{column}")["distributions"][column]
        assert sum(row["count"] for row in rows) == alive
    county = section(data, "distribution.COUNTY")
    shown = sum(row["count"] for row in county["distributions"]["COUNTY"])
    assert shown + county["metrics"]["other_count"] == alive
    age = section(data, "age")
    excluded = sum(age["metrics"]["excluded"].values())
    assert sum(r["count"] for r in age["distributions"]["age_band"]) + excluded == alive


def test_same_dataset_same_profile_apart_from_the_timestamp(sample, profiled, tmp_path):
    _, first, _ = profiled
    _, second = run_profile(sample, tmp_path)
    first = dict(first, generated_at=None)
    second = dict(second, generated_at=None)
    assert first == second


# --------------------------------------------------------------------------- #
# most common codes, checked against an independent count with the csv module
# --------------------------------------------------------------------------- #


def independent_code_counts(sample: Path, table: str) -> dict:
    """Distinct alive patients and records per code, counted without pandas."""
    with (sample / "patients.csv").open(newline="", encoding="utf-8-sig") as handle:
        alive = {row["Id"] for row in csv.DictReader(handle) if not row["DEATHDATE"]}
    patients: dict[tuple, set] = {}
    records: dict[tuple, int] = {}
    with (sample / f"{table}.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["PATIENT"] not in alive or not row["CODE"]:
                continue
            key = (row.get("SYSTEM") or None, row["CODE"])
            patients.setdefault(key, set()).add(row["PATIENT"])
            records[key] = records.get(key, 0) + 1
    return {key: (len(patients[key]), records[key]) for key in patients}


@pytest.mark.parametrize("table", ["conditions", "observations"])
def test_top_codes_match_an_independent_count(sample, profiled, table):
    _, data, _ = profiled
    codes = section(data, f"codes.{table}")
    assert codes["status"] == "COMPUTED"
    expected = independent_code_counts(sample, table)
    assert codes["metrics"]["distinct_codes_alive"] == len(expected)
    for row in codes["codes"]:
        assert (row["patients"], row["records"]) == expected[(row["system"], row["code"])]
    ranked = sorted(
        expected.items(), key=lambda item: (-item[1][0], -item[1][1], item[0][0] or "", item[0][1])
    )
    listed = [(row["system"], row["code"]) for row in codes["codes"]]
    assert listed == [key for key, _ in ranked[: len(listed)]]


def test_every_clinical_table_is_profiled(profiled):
    _, data, _ = profiled
    for table in (
        "conditions", "medications", "procedures", "observations",
        "immunizations", "allergies", "careplans",
    ):
        assert section(data, f"codes.{table}")["status"] == "COMPUTED"
