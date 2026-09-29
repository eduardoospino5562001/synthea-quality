"""Tests for the ``synthea-incidence`` command line interface."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthea_quality import __version__
from synthea_quality.incidence.cli import EXIT_ERROR, EXIT_OK, main
from synthea_quality.incidence.render import JSON_NAME, MARKDOWN_NAME
from synthea_quality.schema.tables import tables_by_name

CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-incidence")
SNOMED = "http://snomed.info/sct"


def write_table(directory: Path, name: str, rows) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "F"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [{"PATIENT": "d1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2022-01-01"}],
    )
    return directory


def read(out: Path) -> dict:
    return json.loads((out / JSON_NAME).read_text(encoding="utf-8"))


def test_writes_both_files_and_summarises(tmp_path, capsys):
    out = tmp_path / "out"
    code = main(
        [
            str(dataset(tmp_path / "csv")),
            "--condition", "Hypertension=59621000",
            "--expected", "Hypertension:incidence=100",
            "--output-dir", str(out),
        ]
    )
    assert code == EXIT_OK
    assert (out / MARKDOWN_NAME).is_file()
    data = read(out)
    assert data["population"] == "all"
    assert data["conditions"][0]["rate"]["events"] == 1
    assert data["conditions"][0]["expected"][0]["position_in_ci95"] in ("inside", "outside")
    printed = capsys.readouterr().out
    assert "Window:     2021-01-01 to 2026-01-01" in printed
    assert "Condition:  Hypertension: 1 event(s) in" in printed


def test_alive_only_and_window_years(tmp_path):
    out = tmp_path / "out"
    assert main([str(dataset(tmp_path / "csv")), "--condition", "H=59621000", "--alive-only",
                 "--window-years", "2", "--output-dir", str(out)]) == EXIT_OK
    data = read(out)
    assert data["population"] == "alive"
    assert data["window"] == {"years": 2, "start": "2024-01-01", "end": "2026-01-01"}
    assert data["conditions"][0]["rate"]["events"] == 0


def test_a_condition_is_required(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path / "csv"))])
    assert raised.value.code == EXIT_ERROR
    assert "at least one condition" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--condition", "H=1", "--window-years", "0"],
        ["--condition", "H=1", "--reference-date", "2026-01-01", "--metadata", "m.json"],
        ["--condition", "H=1", "--age-bands", "4"],
    ],
)
def test_usage_errors(tmp_path, arguments):
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path / "csv")), *arguments])
    assert raised.value.code == EXIT_ERROR


def test_prevalence_measures_are_rejected_here(tmp_path, capsys):
    code = main([str(dataset(tmp_path / "csv")), "--condition", "H=1", "--expected", "H:point=0.1",
                 "--output-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR
    assert "measure must be one of ['incidence']" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0 and __version__ in capsys.readouterr().out


def test_the_console_script_runs_as_installed(tmp_path):
    if not CONSOLE_SCRIPT.exists():
        pytest.skip("synthea-incidence is not installed next to this interpreter")
    completed = subprocess.run(
        [str(CONSOLE_SCRIPT), str(dataset(tmp_path / "csv")), "--condition", "H=59621000",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
