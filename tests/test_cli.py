"""Tests for the command line interface.

Everything runs through :func:`synthea_quality.cli.main`, which returns the exit code
instead of exiting, so each scenario can be asserted directly. One test goes through
a real subprocess to prove the installed entry point works the way a user runs it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthea_quality import __version__
from synthea_quality.cli import (
    EXIT_ERROR,
    EXIT_FINDINGS,
    EXIT_OK,
    JSON_NAME,
    MARKDOWN_NAME,
    main,
)
from synthea_quality.schema.tables import tables_by_name


def write_table(directory: Path, name: str, rows: list[dict[str, str]]) -> Path:
    """Write a table with the contract's header so the dataset stays compatible."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def tmp_cwd(tmp_path: Path, monkeypatch) -> Path:
    """Run the CLI from a temporary directory.

    The output directory defaults to the current directory, so a test that omits
    ``--output-dir`` must not be run from the repository: it would write reports into
    the source tree.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def clean_dataset(directory: Path) -> Path:
    write_table(directory, "patients", [{"Id": "p1", "BIRTHDATE": "1980-01-01"}])
    write_table(
        directory,
        "encounters",
        [
            {
                "Id": "e1",
                "START": "2020-01-01T08:00:00Z",
                "STOP": "2020-01-01T09:00:00Z",
                "PATIENT": "p1",
            }
        ],
    )
    return directory


def failing_dataset(directory: Path) -> Path:
    """A dataset with one deterministic violation: an encounter that ends before it starts."""
    clean_dataset(directory)
    write_table(
        directory,
        "encounters",
        [
            {
                "Id": "e1",
                "START": "2020-01-01T10:00:00Z",
                "STOP": "2020-01-01T09:00:00Z",
                "PATIENT": "p1",
            }
        ],
    )
    return directory


def warning_only_dataset(directory: Path) -> Path:
    """A dataset with warnings but no failure: a repeated supplies row."""
    clean_dataset(directory)
    write_table(
        directory,
        "supplies",
        [
            {"DATE": "2021-05-05", "PATIENT": "p1", "CODE": "abc"},
            {"DATE": "2021-05-05", "PATIENT": "p1", "CODE": "abc"},
        ],
    )
    return directory


# --------------------------------------------------------------------------- #
# help and version
# --------------------------------------------------------------------------- #


def test_help_describes_the_interface_and_exit_codes(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "DATASET-DIRECTORY" in out
    assert "--output-dir" in out
    assert "--version" in out
    assert "exit codes:" in out
    assert "Warnings never change the exit code" in out
    assert "no prevalence, incidence or statistical rule" in out


def test_version_prints_the_tool_version(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"synthea-quality {__version__}"


def test_no_arguments_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == EXIT_ERROR
    captured = capsys.readouterr()
    assert "DATASET-DIRECTORY" in captured.err
    assert "Traceback" not in captured.err


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #


def test_valid_directory_runs_and_writes_both_reports(tmp_path: Path, capsys) -> None:
    data = clean_dataset(tmp_path / "data")
    out = tmp_path / "reports"

    code = main([str(data), "--output-dir", str(out)])

    assert code == EXIT_OK
    assert (out / MARKDOWN_NAME).exists()
    assert (out / JSON_NAME).exists()
    printed = capsys.readouterr().out
    assert "Tables:" in printed
    assert "Contract:" in printed
    assert "Checks:" in printed
    assert str(out / MARKDOWN_NAME) in printed
    assert str(out / JSON_NAME) in printed


def test_missing_directory_is_a_controlled_error(tmp_path: Path, capsys) -> None:
    """argparse turns a bad argument into exit code 2 with a plain message."""
    with pytest.raises(SystemExit) as exit_info:
        main([str(tmp_path / "nope")])

    assert exit_info.value.code == EXIT_ERROR
    captured = capsys.readouterr()
    assert "dataset directory does not exist" in captured.err
    assert "Traceback" not in captured.err


def test_a_file_instead_of_a_directory_is_a_controlled_error(tmp_path: Path, capsys) -> None:
    target = tmp_path / "not_a_dir.csv"
    target.write_text("Id\np1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        main([str(target)])

    assert exit_info.value.code == EXIT_ERROR
    captured = capsys.readouterr()
    assert "not a directory" in captured.err
    assert "Traceback" not in captured.err


def test_output_directory_is_created_when_missing(tmp_path: Path) -> None:
    data = clean_dataset(tmp_path / "data")
    out = tmp_path / "deep" / "nested" / "reports"

    assert not out.exists()
    code = main([str(data), "--output-dir", str(out)])

    assert code == EXIT_OK
    assert (out / JSON_NAME).exists()


def test_paths_with_spaces_work(tmp_path: Path) -> None:
    data = clean_dataset(tmp_path / "my dataset" / "csv output")
    out = tmp_path / "my reports"

    code = main([str(data), "--output-dir", str(out)])

    assert code == EXIT_OK
    assert (out / MARKDOWN_NAME).exists()
    assert (out / JSON_NAME).exists()


# --------------------------------------------------------------------------- #
# exit codes
# --------------------------------------------------------------------------- #


def test_clean_dataset_exits_zero(tmp_path: Path, tmp_cwd: Path) -> None:
    code = main([str(clean_dataset(tmp_path / "data"))])

    assert code == EXIT_OK
    # the default output directory is the current one
    assert (tmp_cwd / MARKDOWN_NAME).exists()
    assert (tmp_cwd / JSON_NAME).exists()


def test_dataset_with_a_failed_check_exits_one(tmp_path: Path, tmp_cwd: Path, capsys) -> None:
    code = main([str(failing_dataset(tmp_path / "data"))])

    assert code == EXIT_FINDINGS
    printed = capsys.readouterr().out
    assert "Failures (1):" in printed
    assert "FAIL temporal.start_le_stop.encounters" in printed


def test_warnings_alone_exit_zero(tmp_path: Path, tmp_cwd: Path, capsys) -> None:
    code = main([str(warning_only_dataset(tmp_path / "data"))])

    assert code == EXIT_OK
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "Warnings:" in printed


def test_a_check_that_errored_exits_two(
    tmp_path: Path, tmp_cwd: Path, monkeypatch, capsys
) -> None:
    """ERROR means the tool could not do its job, which is not a data defect."""
    import dataclasses

    from synthea_quality.models import CheckResult, Severity, Status

    data = clean_dataset(tmp_path / "data")
    real_build = __import__(
        "synthea_quality.reporting.build", fromlist=["build_report"]
    ).build_report

    def with_error(*args, **kwargs):
        report = real_build(*args, **kwargs)
        broken = CheckResult(
            check_id="dates.patients.BIRTHDATE",
            status=Status.ERROR,
            severity=Severity.MEDIUM,
            message="the tool could not complete this check",
            table="patients",
        )
        return dataclasses.replace(report, checks=(*report.checks, broken))

    monkeypatch.setattr("synthea_quality.cli.build_report", with_error)

    code = main([str(data)])

    assert code == EXIT_ERROR
    assert "ERROR dates.patients.BIRTHDATE" in capsys.readouterr().out


def test_unexpected_errors_are_not_silenced(
    tmp_path: Path, tmp_cwd: Path, monkeypatch, capsys
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("something broke inside")

    monkeypatch.setattr("synthea_quality.cli.build_report", boom)

    code = main([str(clean_dataset(tmp_path / "data"))])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "unexpected error: RuntimeError: something broke inside" in captured.err
    assert "Traceback" in captured.err


# --------------------------------------------------------------------------- #
# reports and the source dataset
# --------------------------------------------------------------------------- #


def test_generated_reports_are_complete(tmp_path: Path) -> None:
    data = clean_dataset(tmp_path / "data")
    out = tmp_path / "reports"

    main([str(data), "--output-dir", str(out)])

    markdown = (out / MARKDOWN_NAME).read_text(encoding="utf-8")
    payload = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert markdown.startswith("# Synthea dataset quality report")
    assert "## Documented relations not applied" in markdown
    assert payload["schema_version"] == 1
    assert payload["summary"]["checks_total"] == len(payload["checks"])
    assert payload["dataset"]["data_dir"] == str(data)


def test_many_missing_tables_are_summarised_not_dumped(tmp_path: Path, capsys) -> None:
    data = clean_dataset(tmp_path / "data")  # two tables present, seventeen missing

    main([str(data), "--output-dir", str(tmp_path / "reports")])

    printed = capsys.readouterr().out
    line = next(line for line in printed.splitlines() if line.startswith("Tables:"))
    listed = line.split("missing: ")[1].split(" and ")[0]

    assert "and 11 more" in line
    # exactly the first six names of the contract order, then a count
    assert len(listed.split(", ")) == 6
    assert listed.startswith("allergies")
    assert "supplies" not in line


def test_the_dataset_is_never_modified(tmp_path: Path) -> None:
    data = clean_dataset(tmp_path / "data")
    out = tmp_path / "reports"
    before = {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime)
        for path in sorted(data.iterdir())
    }

    main([str(data), "--output-dir", str(out)])

    after = {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime)
        for path in sorted(data.iterdir())
    }
    assert after == before
    assert sorted(path.name for path in data.iterdir()) == sorted(before)


def test_reports_go_into_the_output_directory_only(tmp_path: Path) -> None:
    data = clean_dataset(tmp_path / "data")
    out = tmp_path / "reports"

    main([str(data), "--output-dir", str(out)])

    assert sorted(path.name for path in out.iterdir()) == sorted([MARKDOWN_NAME, JSON_NAME])


# --------------------------------------------------------------------------- #
# the entry point a user actually runs
# --------------------------------------------------------------------------- #


def test_the_module_entry_point_runs_as_a_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "synthea_quality", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == EXIT_OK
    assert result.stdout.strip() == f"synthea-quality {__version__}"


def test_the_console_script_is_declared() -> None:
    """The entry point exists in the packaged metadata, whatever the local install."""
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert "[project.scripts]" in pyproject
    assert 'synthea-quality = "synthea_quality.cli:run"' in pyproject


# --------------------------------------------------------------------------- #
# A dataset that cannot be checked is an input error, never a silent success
# --------------------------------------------------------------------------- #


def test_a_directory_without_any_synthea_table_is_an_input_error(tmp_path: Path, capsys) -> None:
    """An empty directory must not become 155 skipped checks and exit code 0.

    Measured before the fix: an empty directory produced a complete report of 155
    ``SKIPPED`` checks and returned ``EXIT_OK``, which reads as "nothing was wrong".
    """
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "out"

    code = main([str(dataset), "--output-dir", str(output)])

    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
    assert "patients.csv" in captured.err  # the message says what a dataset needs
    assert not (output / MARKDOWN_NAME).exists()
    assert not (output / JSON_NAME).exists()


def test_a_directory_with_only_unknown_files_is_an_input_error(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "notes.txt").write_text("not a dataset\n", encoding="utf-8")
    output = tmp_path / "out"

    code = main([str(dataset), "--output-dir", str(output)])

    assert code == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_a_structurally_broken_table_is_reported_and_exits_one(tmp_path: Path) -> None:
    """A short row used to be padded with nulls and the run exited 0."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "patients.csv").write_text(
        "Id,BIRTHDATE,GENDER\np1,1980-01-01,M\np2,1990-01-01\n",
        encoding="utf-8",
        newline="",
    )
    output = tmp_path / "out"

    code = main([str(dataset), "--output-dir", str(output)])

    assert code == EXIT_FINDINGS
    report = json.loads((output / JSON_NAME).read_text(encoding="utf-8"))
    structural = [
        check for check in report["checks"] if check["check_id"] == "structure.patients"
    ]
    assert len(structural) == 1
    assert structural[0]["status"] == "FAIL"
    patients_statuses = {
        check["status"] for check in report["checks"] if check["table"] == "patients"
    }
    assert "PASS" not in patients_statuses
    markdown = (output / MARKDOWN_NAME).read_text(encoding="utf-8")
    assert "structure.patients" in markdown


def test_an_unreadable_table_is_recorded_and_the_run_continues(tmp_path: Path, capsys) -> None:
    """An OS failure on one table must not lose the analysis of the others."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "patients.csv").write_text(
        "Id,BIRTHDATE\np1,1980-01-01\n", encoding="utf-8", newline=""
    )
    unreadable = dataset / "encounters.csv"
    unreadable.write_text(
        "Id,START,STOP,PATIENT\ne1,2020-01-01T00:00:00Z,,p1\n", encoding="utf-8", newline=""
    )
    unreadable.chmod(0o000)
    try:
        unreadable.open("rb").close()
    except OSError:
        pass
    else:
        pytest.skip("this user can read a 0o000 file (running as root?)")

    output = tmp_path / "out"
    try:
        code = main([str(dataset), "--output-dir", str(output)])
    finally:
        unreadable.chmod(0o644)

    captured = capsys.readouterr()
    assert code == EXIT_ERROR  # the analysis is incomplete, so the run is not "ok"
    assert "Traceback" not in captured.err
    # the terminal summary names the table whose analysis is missing
    assert "Unreadable: encounters" in captured.out
    report = json.loads((output / JSON_NAME).read_text(encoding="utf-8"))
    assert [error["table"] for error in report["dataset"]["load_errors"]] == ["encounters"]
    lost = [check for check in report["checks"] if check["table"] == "encounters"]
    assert lost
    assert all(check["status"] != "PASS" for check in lost)
    # the table that could be read is still analysed
    patients = [check for check in report["checks"] if check["table"] == "patients"]
    assert any(check["status"] == "PASS" for check in patients)


def test_a_header_only_table_reports_zero_rows_in_the_json(tmp_path: Path) -> None:
    """``rows`` must be 0 for an empty table, not unknown."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "patients.csv").write_text("Id,BIRTHDATE\n", encoding="utf-8", newline="")
    output = tmp_path / "out"

    main([str(dataset), "--output-dir", str(output)])

    report = json.loads((output / JSON_NAME).read_text(encoding="utf-8"))
    summary = next(table for table in report["tables"] if table["name"] == "patients")
    assert summary["rows"] == 0
