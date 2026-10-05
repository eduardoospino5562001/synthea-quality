"""End-to-end acceptance run of the whole product over the official Synthea sample.

What this proves
----------------
That somebody outside the project can point the tool at the dataset Synthea
recommends and get the documented result, using only the public command line: every
run here starts a real process (the ``synthea-quality`` console script), never an
internal function. It covers the whole product on one dataset: discovery, the contract
comparison, all 155 checks, both reports, the exit codes, determinism, the untouched
source files and the reference-evidence attribution.

The data
--------
The dataset is the official sample: ``synthea-sample-data/downloads/latest``, produced
from synthea commit ``d9d07a6e`` (2026-08-17) and published 2026-08-18. It is **not**
stored in this repository; fetch it with::

    python scripts/fetch_official_sample.py          # ~/synthea-sample-data/csv-latest

or point the tests at an existing extraction with::

    SYNTHEA_QUALITY_SAMPLE=/path/to/csv-latest .venv/bin/pytest -m integration

When the dataset is absent these tests skip with that message instead of failing, so a
clean checkout stays green. ``docs/acceptance.md`` documents the checksums, the
baseline numbers and how to re-baseline.

Running it
----------
::

    .venv/bin/pytest -m integration                 # just the acceptance runs
    .venv/bin/pytest                                # the whole suite

The first test is a tripwire: it compares the dataset against the digest the baseline
was measured on. If Synthea republishes ``latest``, that test fails first and says what
to do, so the numbers below are never silently applied to different data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Where the fetch script leaves the sample unless told otherwise.
DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"

#: Digest of the dataset every number below was measured on (see docs/acceptance.md).
BASELINE_DATASET_DIGEST = "7ecb7460dd31f9921f433a9c8d341f265e7b55b2a2ae9f0b32199deb8760769d"
BASELINE_FILES = 18
BASELINE_BYTES = 62_862_715

BASELINE_TOTAL_CHECKS = 155
BASELINE_STATUS = {
    "PASS": 143,
    "WARNING": 6,
    "FAIL": 0,
    "NOT_APPLICABLE": 3,
    "SKIPPED": 3,
    "ERROR": 0,
}
BASELINE_CONTRACT = "COMPATIBLE"
BASELINE_CONTRACT_TABLES = 19
BASELINE_TABLES_OBSERVED = 18
BASELINE_MISSING_TABLES = ["patient_expenses"]
#: The six warnings this dataset produces, pinned so a data change cannot pass quietly.
BASELINE_WARNING_CHECKS = (
    "duplicates.observations",
    "duplicates.supplies",
    "empty_columns.allergies",
    "empty_columns.claims",
    "empty_columns.claims_transactions",
    "empty_columns.payers",
)
#: The one check a single corrupted encounter interval must produce.
EXPECTED_FAILURE_CHECK = "temporal.start_le_stop.encounters"

#: The console script a user runs, next to the interpreter running the tests.
CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-quality")

#: Generous ceilings: they catch a pathological regression, not normal variation.
MAX_RUNTIME_SECONDS = 180
MAX_PEAK_MEMORY_MB = 2048


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def available_sample() -> Path | None:
    """The extracted official sample, or ``None`` when it is not there."""
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if candidate.is_dir() and any(candidate.glob("*.csv")):
        return candidate
    return None


@pytest.fixture(scope="module")
def sample() -> Path:
    found = available_sample()
    if found is None:
        pytest.skip(
            "official sample not available; run `python scripts/fetch_official_sample.py` "
            "or set SYNTHEA_QUALITY_SAMPLE to an extracted csv-latest directory"
        )
    return found


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_digest(directory: Path) -> str:
    lines = [
        f"{path.name}\0{sha256_of_file(path)}" for path in sorted(directory.glob("*.csv"))
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def cli_command() -> list[str]:
    """The command a user types: the console script, falling back to ``python -m``."""
    if CONSOLE_SCRIPT.exists():
        return [str(CONSOLE_SCRIPT)]
    return [sys.executable, "-m", "synthea_quality"]


def run_cli(dataset: Path, output_dir: Path) -> subprocess.CompletedProcess:
    """Run the tool the way a person would, as a real process."""
    return subprocess.run(
        [*cli_command(), str(dataset), "--output-dir", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


def read_reports(output_dir: Path) -> tuple[str, dict]:
    markdown = (output_dir / "synthea_quality_report.md").read_text(encoding="utf-8")
    payload = json.loads((output_dir / "synthea_quality_report.json").read_text(encoding="utf-8"))
    return markdown, payload


def counts_from_checks(payload: dict) -> dict[str, int]:
    """Recount the statuses from the ``checks`` array, independently of the summary."""
    counts = {status: 0 for status in BASELINE_STATUS}
    for check in payload["checks"]:
        counts[check["status"]] += 1
    return counts


def warning_check_ids(payload: dict) -> list[str]:
    return sorted(
        check["check_id"] for check in payload["checks"] if check["status"] == "WARNING"
    )


def git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout


# --------------------------------------------------------------------------- #
# 1. the dataset itself
# --------------------------------------------------------------------------- #


def test_the_dataset_is_the_documented_baseline(sample: Path) -> None:
    """Tripwire: if Synthea republishes ``latest``, this fails before the numbers do."""
    files = sorted(sample.glob("*.csv"))
    total = sum(path.stat().st_size for path in files)

    assert len(files) == BASELINE_FILES
    assert total == BASELINE_BYTES
    assert dataset_digest(sample) == BASELINE_DATASET_DIGEST, (
        "the official sample is not the dataset this baseline was measured on; "
        "re-run the acceptance checks and update docs/acceptance.md, or fetch the "
        "documented revision with scripts/fetch_official_sample.py"
    )


# --------------------------------------------------------------------------- #
# 2. the main acceptance run
# --------------------------------------------------------------------------- #


def test_a_full_run_from_the_command_line(sample: Path, tmp_path: Path) -> None:
    output = tmp_path / "reports"

    result = run_cli(sample, output)

    # 6. exit code 0 on a dataset with no failure
    assert result.returncode == 0
    assert result.stderr == ""

    # 8. both reports are written, with the predictable names
    markdown, payload = read_reports(output)

    # 1. the run is complete and reports what it did
    assert "synthea-quality" in result.stdout
    assert f"Tables:   {BASELINE_TABLES_OBSERVED} of {BASELINE_CONTRACT_TABLES}" in result.stdout
    assert f"Contract: {BASELINE_CONTRACT}" in result.stdout
    assert "Checks:   155 total" in result.stdout
    assert str(output / "synthea_quality_report.md") in result.stdout
    assert str(output / "synthea_quality_report.json") in result.stdout

    # 2. discovery: the 18 files of the sample, none unknown
    assert payload["dataset"]["tables_observed"] == BASELINE_TABLES_OBSERVED
    assert payload["dataset"]["unknown_files"] == []
    assert len(payload["tables"]) == BASELINE_TABLES_OBSERVED
    assert {table["name"] for table in payload["tables"]} >= {
        "patients",
        "encounters",
        "observations",
        "claims_transactions",
    }
    # 3. contract: compatible over the observed tables, 18 of 19, and no version claim
    assert payload["dataset"]["contract_status"] == BASELINE_CONTRACT
    assert payload["dataset"]["contract_tables"] == BASELINE_CONTRACT_TABLES
    assert payload["dataset"]["missing_known_tables"] == BASELINE_MISSING_TABLES
    assert "18 of the 19 contract tables were observed" in payload["dataset"]["contract_summary"]
    assert payload["dataset"]["load_errors"] == []

    # 4. every check ran exactly once
    identifiers = [check["check_id"] for check in payload["checks"]]
    assert len(identifiers) == BASELINE_TOTAL_CHECKS
    assert len(set(identifiers)) == BASELINE_TOTAL_CHECKS

    # 5. the documented statuses, both as declared and as recounted from the checks
    assert payload["summary"]["by_status"] == BASELINE_STATUS
    assert counts_from_checks(payload) == BASELINE_STATUS
    assert warning_check_ids(payload) == sorted(BASELINE_WARNING_CHECKS)
    assert payload["summary"]["has_failures"] is False
    assert payload["dataset"]["generated_at"] and payload["dataset"]["tool_version"]

    # 7. the JSON is complete and self-consistent
    assert payload["schema_version"] == 1
    assert sum(payload["summary"]["by_category"].values()) == BASELINE_TOTAL_CHECKS
    assert sum(payload["summary"]["findings_by_severity"].values()) == BASELINE_STATUS["WARNING"]
    assert all("message" in check and "metrics" in check for check in payload["checks"])

    # 8. the Markdown is a report a person can read, and it does not over-claim
    assert markdown.startswith("# Synthea dataset quality report")
    assert "Verdict: no check failed" in markdown
    assert "| **total** | **155** |" in markdown
    assert "not proof that the dataset was produced by one exact" in markdown
    assert "confirmed version" not in markdown.lower()

    # 10. reference evidence is attributed to the reference dataset, never to this one
    head, _, section = markdown.partition("## Documented relations not applied")
    assert "not on the dataset analysed in this report" in section
    assert f"`{sample}`" in section
    for relation in payload["dataset"]["unresolved_relations"]:
        assert relation["reference_evidence"] not in head
        assert "2026-08" in relation["reference_dataset"]
        assert relation["pending"].strip()


def test_the_json_and_the_markdown_agree(sample: Path, tmp_path: Path) -> None:
    """9. coherence: every number in the Markdown comes from the JSON."""
    output = tmp_path / "reports"
    run_cli(sample, output)
    markdown, payload = read_reports(output)
    counts = counts_from_checks(payload)

    for status, expected in BASELINE_STATUS.items():
        assert f"| `{status}` | {expected} |" in markdown
    assert f"| **total** | **{len(payload['checks'])}** |" in markdown
    assert f"**{payload['dataset']['contract_status']}**" in markdown
    assert f"Tables observed: **{payload['dataset']['tables_observed']}**" in markdown
    for relation in payload["dataset"]["unresolved_relations"]:
        assert relation["relation"] in markdown
    # a failing check would have to appear in the findings section, not only in the table
    assert counts["FAIL"] == 0 and "## Findings" in markdown


def test_a_second_run_produces_the_same_reports(sample: Path, tmp_path: Path) -> None:
    """12. determinism: same input, same output; only the timestamp may differ."""
    first, second = tmp_path / "first", tmp_path / "second"

    run_cli(sample, first)
    run_cli(sample, second)
    markdown_one, payload_one = read_reports(first)
    markdown_two, payload_two = read_reports(second)

    stamp_one = payload_one["dataset"]["generated_at"]
    stamp_two = payload_two["dataset"]["generated_at"]
    payload_one["dataset"].pop("generated_at")
    payload_two["dataset"].pop("generated_at")
    assert payload_one == payload_two
    assert markdown_one.replace(stamp_one, "") == markdown_two.replace(stamp_two, "")


def test_the_source_dataset_is_never_modified(sample: Path, tmp_path: Path) -> None:
    """11. the tool reads: every byte of every CSV is the same afterwards."""
    before = {path.name: sha256_of_file(path) for path in sorted(sample.glob("*.csv"))}
    listing_before = sorted(path.name for path in sample.iterdir())

    run_cli(sample, tmp_path / "reports")

    after = {path.name: sha256_of_file(path) for path in sorted(sample.glob("*.csv"))}
    assert after == before
    assert sorted(path.name for path in sample.iterdir()) == listing_before


def test_nothing_lands_in_the_repository(sample: Path, tmp_path: Path) -> None:
    """14. a run with an explicit output directory must not touch the source tree."""
    if shutil.which("git") is None:
        pytest.skip("git is not available to inspect the working tree")
    before = git_status()

    run_cli(sample, tmp_path / "reports")

    assert git_status() == before
    assert "synthea_quality_report" not in git_status()
    assert not (REPOSITORY_ROOT / "synthea_quality_report.md").exists()
    assert not (REPOSITORY_ROOT / "synthea_quality_report.json").exists()


def test_runtime_and_peak_memory_are_reported_and_within_budget(
    sample: Path, tmp_path: Path
) -> None:
    """13. time and memory are measured in a fresh interpreter and documented."""
    measure = (
        "import resource, subprocess, sys, time;"
        "start = time.perf_counter();"
        "finished = subprocess.run(sys.argv[1:], capture_output=True);"
        "print(finished.returncode, int((time.perf_counter() - start) * 1000),"
        " resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)"
    )
    output = tmp_path / "reports"
    result = subprocess.run(
        [sys.executable, "-c", measure, *cli_command(), str(sample), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    code, milliseconds, peak_kb = (int(value) for value in result.stdout.split())
    seconds, peak_mb = milliseconds / 1000, peak_kb / 1024

    print(
        f"\nacceptance run: exit {code}, {seconds:.1f} s wall, "
        f"~{peak_mb:,.0f} MB peak RSS for the child process"
    )

    assert code == 0
    assert seconds < MAX_RUNTIME_SECONDS
    assert peak_mb < MAX_PEAK_MEMORY_MB


# --------------------------------------------------------------------------- #
# 3. a second acceptance run: one known corruption
# --------------------------------------------------------------------------- #


def test_one_corrupted_interval_produces_exactly_one_failure(sample: Path, tmp_path: Path) -> None:
    """A single, known violation must change one number and nothing else."""
    corrupted = tmp_path / "corrupted"
    corrupted.mkdir()
    for path in sorted(sample.glob("*.csv")):
        shutil.copy2(path, corrupted / path.name)

    # One row, one value: the first encounter now ends before it starts.
    target = corrupted / "encounters.csv"
    lines = target.read_text(encoding="utf-8-sig").splitlines()
    fields = lines[1].split(",")
    header = lines[0].split(",")
    fields[header.index("STOP")] = "1990-01-01T00:00:00Z"
    lines[1] = ",".join(fields)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output = tmp_path / "reports"
    result = run_cli(corrupted, output)
    markdown, payload = read_reports(output)

    # exit code 1 for a data defect, and the failure is the expected one
    assert result.returncode == 1
    assert result.stderr == ""
    failures = [check for check in payload["checks"] if check["status"] == "FAIL"]
    assert len(failures) == 1
    assert failures[0]["check_id"] == EXPECTED_FAILURE_CHECK
    assert failures[0]["metrics"]["violations"] == 1
    assert failures[0]["metrics"]["evaluated"] == 5571
    assert failures[0]["severity"] == "MEDIUM"

    # everything else is exactly the baseline, shifted by that single failure
    counts = counts_from_checks(payload)
    assert counts == {**BASELINE_STATUS, "PASS": BASELINE_STATUS["PASS"] - 1, "FAIL": 1}
    assert len(payload["checks"]) == BASELINE_TOTAL_CHECKS
    assert payload["summary"]["has_failures"] is True

    # the reports say the same thing: the failure is up front and explained
    assert "FAIL temporal.start_le_stop.encounters" in result.stdout
    assert "`FAIL` — `temporal.start_le_stop.encounters`" in markdown
    assert "1 of 5571 row(s)" in markdown
    assert "Verdict: **1 check(s) failed**" in markdown


# --------------------------------------------------------------------------- #
# 4. a third acceptance run: a usage error
# --------------------------------------------------------------------------- #


def test_a_missing_dataset_is_a_clean_usage_error(tmp_path: Path) -> None:
    """Exit code 2, one readable message, no traceback, nothing written."""
    missing = tmp_path / "not-a-dataset"
    output = tmp_path / "reports"
    before = git_status()

    result = subprocess.run(
        [*cli_command(), str(missing), "--output-dir", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "dataset directory does not exist" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    assert not output.exists()
    assert git_status() == before
