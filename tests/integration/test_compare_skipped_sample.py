"""Compare the official sample with a copy that has no ``observations.csv``, end to end.

This is the case of the audit that found skipped outcomes leaving figures ``only_in_*``
without a reason. The copy is the sample with every table but ``observations.csv`` as a
symbolic link, so only what reads observations can differ. The outcomes that read it are
taken from the module file itself, never from the tool: a ``value`` outcome, and a
``death`` outcome with ``cause_values`` (the cause of death is an observation). Every
figure of those outcomes must come out ``skipped`` with the reason of the side without
the table, in both directions; the Markdown must list each of them with that reason; and
the size of the indexed cohort, recounted here with ``csv``, is compared as usual.

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from synthea_quality.compare.cli import main

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
BREAST_CANCER = REPOSITORY / "examples" / "breast_cancer.json"
#: Both sides are measured at this date so the recount needs no metadata.
REFERENCE_DATE = "2026-01-01"
#: The table left out of the copy, and the reason the tool gives for it.
MISSING_TABLE = "observations.csv"
REASON = "observations.csv is not in the dataset"


def _without_observations(sample: Path, directory: Path) -> Path:
    """A copy of ``sample`` with every table but ``observations.csv`` as a symbolic link."""
    for path in sample.iterdir():
        if path.name != MISSING_TABLE and path.is_file():
            (directory / path.name).symlink_to(path)
    return directory


def _run(baseline: Path, candidate: Path, output: Path) -> tuple[dict[str, Any], list[str]]:
    """Run the command as a user would; return its JSON and the lines of its Markdown."""
    with redirect_stdout(io.StringIO()):
        exit_code = main(
            [
                str(baseline),
                str(candidate),
                "--module",
                str(BREAST_CANCER),
                "--reference-date",
                REFERENCE_DATE,
                "--output-dir",
                str(output),
            ]
        )
    assert exit_code == 0
    data = json.loads((output / "synthea_compare.json").read_text(encoding="utf-8"))
    markdown = (output / "synthea_compare.md").read_text(encoding="utf-8").splitlines()
    return data, markdown


def _outcomes_reading_observations(module_file: Path) -> set[str]:
    """The outcomes of the module file that need ``observations.csv``."""
    module = json.loads(module_file.read_text(encoding="utf-8"))
    return {
        outcome["name"]
        for outcome in module["outcomes"]
        if outcome["type"] == "value" or (outcome["type"] == "death" and outcome.get("cause_values"))
    }


def _index_cohort(sample: Path, module_file: Path) -> int:
    """The patients, alive or deceased, with a record of the index condition on or before
    the reference date, recounted with ``csv`` from ``conditions.csv``."""
    module = json.loads(module_file.read_text(encoding="utf-8"))
    index = module["index"]["condition"]
    codes = {code for condition in module["conditions"] if condition["name"] == index for code in condition["codes"]}
    reference = date.fromisoformat(REFERENCE_DATE)
    with (sample / "conditions.csv").open(newline="", encoding="utf-8") as handle:
        return len(
            {
                row["PATIENT"]
                for row in csv.DictReader(handle)
                if row["CODE"] in codes and date.fromisoformat(row["START"][:10]) <= reference
            }
        )


def _skipped(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [comparison for comparison in data["comparisons"] if comparison["state"] == "skipped"]


@pytest.fixture(scope="module")
def runs(sample: Path, tmp_path_factory: pytest.TempPathFactory) -> list[tuple[dict[str, Any], list[str]]]:
    """The comparison in both directions: the sample against the copy, and back."""
    copy = _without_observations(sample, tmp_path_factory.mktemp("no-observations"))
    output = tmp_path_factory.mktemp("compare-skipped")
    return [_run(sample, copy, output / "forward"), _run(copy, sample, output / "backward")]


def test_s1_nothing_is_one_sided_and_every_skipped_figure_has_the_reason(runs) -> None:
    for data, _ in runs:
        assert (data["only_in_baseline"], data["only_in_candidate"]) == (0, 0)
        assert _skipped(data)
        assert all(comparison["reason"] and REASON in comparison["reason"] for comparison in _skipped(data))


def test_s2_the_skipped_figures_are_those_of_the_outcomes_that_read_observations(runs) -> None:
    expected = _outcomes_reading_observations(BREAST_CANCER)
    assert expected == {"Stage group", "Death"}
    for data, _ in runs:
        elements = {(c["key"]["block"], c["key"]["element"]) for c in _skipped(data)}
        assert elements == {("indexed", name) for name in expected}


def test_s3_both_directions_give_the_same_states(runs) -> None:
    (forward, _), (backward, _) = runs
    states = [
        {tuple(c["key"].values()): (c["state"], c["reason"]) for c in data["comparisons"]}
        for data in (forward, backward)
    ]
    assert states[0] == states[1]


def test_s4_the_markdown_lists_every_skipped_figure_with_its_reason(runs) -> None:
    for data, markdown in runs:
        rows = [line for line in markdown if line.startswith("| ") and line.endswith(" |") and REASON in line]
        assert len(rows) == len(_skipped(data))
        for comparison in _skipped(data):
            key = comparison["key"]
            start = f"| {key['element']} | {key['measure']} | {key['stratum']} |"
            assert any(row.startswith(start) and row.endswith(f" {comparison['reason']} |") for row in rows), start


def test_s5_the_cohort_is_compared_and_is_the_csv_recount(sample: Path, runs) -> None:
    patients = _index_cohort(sample, BREAST_CANCER)
    assert patients > 0
    for data, _ in runs:
        [cohort] = [
            c for c in data["comparisons"] if (c["key"]["block"], c["key"]["element"]) == ("indexed", "cohort")
        ]
        assert cohort["state"] == "unchanged"
        assert cohort["baseline"]["value"] == cohort["candidate"]["value"] == patients
