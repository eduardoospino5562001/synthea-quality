"""End-to-end run of ``synthea-prevalence`` over the official Synthea sample.

Two independent checks:

1. **A recount with the standard library.** Point and lifetime prevalence of myocardial
   infarction, and of every code of the general table, are recounted with the ``csv``
   module from the definitions alone (alive = empty ``DEATHDATE``; ``START <= ref``;
   active = ``STOP`` empty or after ``ref``) and must match exactly.

2. **The module-validation notebook.** ``MI Module Validation.ipynb``
   (synthetichealth/module-validation) computes a "Synthea MI Incidence Rate" as the
   distinct patients with any condition whose description contains "myocardial
   infarction", over every patient in ``patients.csv``. That logic is reproduced here and
   the difference with the tool is asserted, not hidden: the notebook's denominator
   includes the deceased and has no time window, so on this sample it gives 7/108 while
   the tool's lifetime prevalence over the same four codes gives 7/99. The README
   documents every difference.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-prevalence")

#: The three acute myocardial infarction codes the sample contains.
MI_CODES = ("22298006", "401303003", "401314000")
#: The four codes the notebook's text search "myocardial infarction" selects on this sample.
NOTEBOOK_MI_CODES = (*MI_CODES, "399211009")


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
    return [sys.executable, "-m", "synthea_quality.prevalence.cli"]


@pytest.fixture(scope="module")
def report(sample, tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("prevalence")
    completed = subprocess.run(
        [
            *command(),
            str(sample),
            "--condition", f"Myocardial infarction={','.join(MI_CODES)};acute",
            "--condition", f"MI as in the notebook={','.join(NOTEBOOK_MI_CODES)}",
            "--output-dir", str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads((out / "synthea_prevalence.json").read_text(encoding="utf-8"))


def read(sample: Path, table: str) -> list[dict[str, str]]:
    with (sample / f"{table}.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def recount(sample: Path, reference: str, codes=None) -> dict:
    """{(system, code): (point patients, lifetime patients)} among the alive, with csv only."""
    alive = {row["Id"] for row in read(sample, "patients") if not row["DEATHDATE"]}
    point: dict[tuple, set] = {}
    lifetime: dict[tuple, set] = {}
    for row in read(sample, "conditions"):
        if row["PATIENT"] not in alive or row["START"] > reference:
            continue
        key = (row["SYSTEM"], row["CODE"]) if codes is None else "group"
        if codes is not None and row["CODE"] not in codes:
            continue
        lifetime.setdefault(key, set()).add(row["PATIENT"])
        if not row["STOP"] or row["STOP"] > reference:
            point.setdefault(key, set()).add(row["PATIENT"])
    return {k: (len(point.get(k, ())), len(v)) for k, v in lifetime.items()}, len(alive)


def condition(report: dict, name: str) -> dict:
    return next(c for c in report["conditions"] if c["name"] == name)


def test_mi_prevalence_matches_an_independent_recount(sample, report):
    counts, alive = recount(sample, report["reference_date"]["value"], set(MI_CODES))
    mi = condition(report, "Myocardial infarction")
    assert report["alive"] == alive
    assert (mi["point"]["numerator"], mi["lifetime"]["numerator"]) == counts["group"]
    assert mi["point"]["denominator"] == mi["lifetime"]["denominator"] == alive
    strata = [s for s in mi["strata"] if s["dimension"] == "GENDER"]
    assert sum(s["lifetime"]["numerator"] for s in strata) == mi["lifetime"]["numerator"]
    assert sum(s["lifetime"]["denominator"] for s in strata) == alive


def test_every_row_of_the_general_table_matches_an_independent_recount(sample, report):
    counts, alive = recount(sample, report["reference_date"]["value"])
    rows = report["general"]["rows"]
    assert rows and not any(row["social"] for row in rows)
    for row in rows:
        assert (row["point"]["numerator"], row["lifetime"]["numerator"]) == counts[
            (row["system"], row["code"])
        ]
        assert row["point"]["denominator"] == alive
    excluded = report["general"]["metrics"]["social_codes_in_data"]
    assert len(rows) + excluded == len(counts)


def test_the_notebook_logic_and_its_documented_difference(sample, report):
    # MI Module Validation.ipynb, cells 2, 5, 9 and 11, with the standard library.
    patients = read(sample, "patients")
    conditions = read(sample, "conditions")
    matched = [r for r in conditions if "myocardial infarction" in r["DESCRIPTION"].lower()]
    notebook_patients = {r["PATIENT"] for r in matched}
    tot_patients = len({r["Id"] for r in patients})
    notebook_rate = len(notebook_patients) / tot_patients

    assert sorted({r["CODE"] for r in matched}) == sorted(NOTEBOOK_MI_CODES)
    assert (len(notebook_patients), tot_patients) == (7, 108)

    tool = condition(report, "MI as in the notebook")["lifetime"]
    assert (tool["numerator"], tool["denominator"]) == (7, 99)
    # Same patients (every MI patient of this sample is alive); the rate differs only by
    # the denominator, which in the notebook includes the 9 deceased patients.
    alive = {r["Id"] for r in patients if not r["DEATHDATE"]}
    assert notebook_patients <= alive
    assert tool["denominator"] == len(alive)
    assert tot_patients - len(alive) == 9  # the deceased the notebook keeps in its denominator
    assert round(notebook_rate, 6) == round(7 / 108, 6) != tool["rate"]


def test_mi_declared_acute_gets_the_unstopped_records_note(sample, report):
    # On this sample the STEMI and NSTEMI codes never have a STOP: no Synthea module ends
    # them (heart/stemi_pathway.json and heart/nsteacs_pathway.json have no ConditionEnd).
    mi = condition(report, "Myocardial infarction")
    assert mi["acute"] is True
    unstopped = [
        r for r in read(sample, "conditions")
        if r["CODE"] in MI_CODES and not r["STOP"]
        and r["PATIENT"] in {p["Id"] for p in read(sample, "patients") if not p["DEATHDATE"]}
    ]
    assert mi["metrics"]["records_without_stop"] == len(unstopped) == 6
    assert mi["metrics"]["records"] == 8
    assert (
        "6 of 8 records have no STOP date, so point prevalence counts every past event as "
        "still active; for an acute condition, lifetime prevalence is the meaningful measure."
    ) in mi["notes"]
    notebook = condition(report, "MI as in the notebook")
    assert notebook["acute"] is False
    assert not any("for an acute condition" in note for note in notebook["notes"])
