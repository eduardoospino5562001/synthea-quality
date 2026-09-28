"""Tests for the ``synthea-prevalence`` command line interface."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthea_quality import __version__
from synthea_quality.prevalence.cli import EXIT_ERROR, EXIT_OK, main
from synthea_quality.prevalence.render import JSON_NAME, MARKDOWN_NAME
from synthea_quality.schema.tables import tables_by_name

CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-prevalence")
SNOMED = "http://snomed.info/sct"


def write_table(directory: Path, name: str, rows: list[dict[str, str]]) -> Path:
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
            {"Id": "a2", "BIRTHDATE": "1990-01-01", "GENDER": "F"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "a1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2010-01-01",
             "DESCRIPTION": "Essential hypertension (disorder)"},
            {"PATIENT": "a2", "CODE": "73595000", "SYSTEM": SNOMED, "START": "2020-01-01",
             "DESCRIPTION": "Stress (finding)"},
        ],
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
            "--expected", "Hypertension:point=0.3",
            "--output-dir", str(out),
        ]
    )
    assert code == EXIT_OK
    assert (out / MARKDOWN_NAME).is_file()
    data = read(out)
    assert data["conditions"][0]["point"]["numerator"] == 1
    assert data["conditions"][0]["expected"][0]["position_in_ci95"] == "inside"
    printed = capsys.readouterr().out
    assert "Condition:  Hypertension: point 1/2, lifetime 1/2" in printed
    assert "General:    1 codes (social codes excluded)" in printed


def test_include_social_and_top(tmp_path):
    out = tmp_path / "out"
    assert main([str(dataset(tmp_path / "csv")), "--include-social", "--top", "1",
                 "--output-dir", str(out)]) == EXIT_OK
    data = read(out)
    assert data["general"]["include_social"] is True
    assert len(data["general"]["rows"]) == 2 and data["general"]["top"] == 1


def test_conditions_file(tmp_path):
    spec = tmp_path / "c.json"
    spec.write_text(json.dumps({"conditions": [
        {"name": "Hypertension", "codes": ["59621000"], "expected": {"lifetime": 0.9}}
    ]}), encoding="utf-8")
    out = tmp_path / "out"
    assert main([str(dataset(tmp_path / "csv")), "--conditions", str(spec),
                 "--output-dir", str(out)]) == EXIT_OK
    assert read(out)["conditions"][0]["expected"][0]["position_in_ci95"] == "inside"


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--condition", "no-equals"], "NAME=CODE"),
        (["--expected", "Missing:point=0.1"], "no --condition"),
        (["--conditions", "absent.json"], "could not be read"),
    ],
)
def test_bad_definitions_are_errors_without_traceback(tmp_path, capsys, arguments, message):
    code = main([str(dataset(tmp_path / "csv")), *arguments, "--output-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert message in err and "Traceback" not in err
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--reference-date", "2026-01-01", "--metadata", "m.json"],
        ["--top", "0"],
        ["--age-bands", "5,18"],
        ["--reference-date", "yesterday"],
    ],
)
def test_usage_errors(tmp_path, arguments):
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path / "csv")), *arguments])
    assert raised.value.code == EXIT_ERROR


def test_an_unreadable_table_writes_an_incomplete_report_and_exits_two(tmp_path):
    directory = dataset(tmp_path / "csv")
    (directory / "conditions.csv").write_bytes(b"PATIENT,CODE\n\xff\xfe,1\n")
    out = tmp_path / "out"
    assert main([str(directory), "--output-dir", str(out)]) == EXIT_ERROR
    assert "This report is incomplete" in (out / MARKDOWN_NAME).read_text("utf-8")


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0 and __version__ in capsys.readouterr().out


def test_the_console_script_runs_as_installed(tmp_path):
    if not CONSOLE_SCRIPT.exists():
        pytest.skip("synthea-prevalence is not installed next to this interpreter")
    completed = subprocess.run(
        [str(CONSOLE_SCRIPT), str(dataset(tmp_path / "csv")), "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
