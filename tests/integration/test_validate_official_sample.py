"""End-to-end run of ``synthea-validate-module`` over the official Synthea sample.

The command promises the numbers of the dedicated commands, not new ones: with the example
module file — which ``synthea-prevalence`` and ``synthea-incidence`` read as it is, with
``--conditions`` — its prevalence and incidence results must be identical, key for key,
to those the two commands write. Both of
those are themselves checked against independent recounts in their own integration tests.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "myocardial_infarction.json"
DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
BIN = Path(sys.executable).parent


@pytest.fixture(scope="module")
def sample() -> Path:
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if not (candidate / "conditions.csv").is_file():
        pytest.skip(
            "official sample not available; run `python scripts/fetch_official_sample.py` "
            "or set SYNTHEA_QUALITY_SAMPLE to an extracted csv-latest directory"
        )
    return candidate


def run(command: str, module: str, *arguments: str) -> None:
    script = BIN / command
    base = [str(script)] if script.exists() else [sys.executable, "-m", module]
    completed = subprocess.run(
        [*base, *arguments], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="module")
def reports(sample, tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("validate")
    run("synthea-validate-module", "synthea_quality.validate.cli",
        str(sample), "--module", str(EXAMPLE), "--output-dir", str(out / "validate"))
    run("synthea-prevalence", "synthea_quality.prevalence.cli",
        str(sample), "--conditions", str(EXAMPLE), "--output-dir", str(out / "prevalence"))
    run("synthea-incidence", "synthea_quality.incidence.cli",
        str(sample), "--conditions", str(EXAMPLE), "--output-dir", str(out / "incidence"))

    def read(name: str, file: str) -> dict:
        return json.loads((out / name / file).read_text(encoding="utf-8"))

    return {
        "validate": read("validate", "synthea_module_validation.json"),
        "prevalence": read("prevalence", "synthea_prevalence.json"),
        "incidence": read("incidence", "synthea_incidence.json"),
        "markdown": (out / "validate" / "synthea_module_validation.md").read_text("utf-8"),
    }


def test_prevalence_is_identical_to_synthea_prevalence(reports):
    mine = reports["validate"]["conditions"][0]["prevalence"]
    theirs = reports["prevalence"]["conditions"][0]
    assert mine == theirs
    # the module file's incidence value is left to the incidence report
    assert [e["measure"] for e in mine["expected"]] == ["lifetime"]
    assert reports["validate"]["alive"] == reports["prevalence"]["alive"]


def test_incidence_is_identical_to_synthea_incidence(reports):
    mine = reports["validate"]["conditions"][0]["incidence"]
    theirs = reports["incidence"]["conditions"][0]
    assert mine == theirs
    assert reports["validate"]["incidence_window"] == reports["incidence"]["window"]
    assert reports["validate"]["cohort"] == reports["incidence"]["cohort"]


def test_the_example_report_on_the_sample(reports):
    data = reports["validate"]
    mi = data["conditions"][0]
    assert data["module"]["name"] == "Myocardial infarction"
    assert (mi["prevalence"]["lifetime"]["numerator"], data["alive"]) == (6, 99)
    assert mi["incidence"]["rate"]["events"] == 1
    positions = [e["position_in_ci95"] for e in (*mi["prevalence"]["expected"],
                                                *mi["incidence"]["expected"])]
    assert positions and set(positions) <= {"inside", "outside"}
    text = reports["markdown"]
    assert "illustrative — replace with a cited reference" in text
    assert "6 of 8 records have no STOP date" in text
