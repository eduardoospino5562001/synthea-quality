"""Tests for the ``synthea-observations`` command line interface."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthea_quality import __version__
from synthea_quality.observations.cli import EXIT_ERROR, EXIT_OK, main
from synthea_quality.observations.render import JSON_NAME, MARKDOWN_NAME

from synthea_quality.schema.tables import tables_by_name

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples" / "hypertension.json"
CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-observations")

SNOMED = "http://snomed.info/sct"
SBP = "8480-6"


def write_table(directory: Path, name: str, rows, columns=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def reading(patient, value, day="2026-01-01T00:00:00Z"):
    return {"DATE": day, "PATIENT": patient, "CODE": SBP, "DESCRIPTION": "Systolic",
            "VALUE": value, "UNITS": "mm[Hg]", "TYPE": "numeric"}


def dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-06-01", "GENDER": "F"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "M"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [{"PATIENT": "a1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2015-01-01"}],
    )
    write_table(
        directory,
        "observations",
        [reading("a1", "150"), reading("a2", "120", "2020-01-01T00:00:00Z"), reading("d1", "170")],
    )
    return directory



def test_writes_both_files_and_summarises(tmp_path, capsys):
    out = tmp_path / "out"
    code = main([str(dataset(tmp_path / "csv")), "--observation", "SBP=8480-6",
                 "--observation", "none", "--output-dir", str(out)])
    assert code == EXIT_OK
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert [o["name"] for o in data["observations"]] == ["SBP", "none"]
    assert (out / MARKDOWN_NAME).read_text("utf-8").startswith("# Synthea observation values")
    printed = capsys.readouterr().out
    assert "Observation: SBP: 2 patient(s), median 135 mm[Hg]" in printed
    assert "Observation: none: not computed — no row of code none" in printed
    assert "General:     1 code and unit pairs" in printed


def test_the_example_module_file(tmp_path):
    out = tmp_path / "out"
    code = main([str(dataset(tmp_path / "csv")), "--module", str(EXAMPLE),
                 "--output-dir", str(out)])
    assert code == EXIT_OK
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert data["module_file"] == str(EXAMPLE)
    assert [o["cohort"] for o in data["observations"]][2] == {
        "condition": "Hypertension", "rule": "point",
    }


def test_a_name_used_twice_is_an_error(tmp_path, capsys):
    code = main([str(dataset(tmp_path / "csv")), "--observation", "8480-6",
                 "--observation", "8480-6", "--output-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR
    assert "more than once" in capsys.readouterr().err


def test_an_unusable_module_file_is_an_error_without_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"observations": [{"code": "1", "cohort": "Nope"}]}', encoding="utf-8")
    code = main([str(dataset(tmp_path / "csv")), "--module", str(bad),
                 "--output-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "do not define" in err and "Traceback" not in err


def test_an_unreadable_table_exits_2_after_writing(tmp_path):
    directory = dataset(tmp_path / "csv")
    (directory / "observations.csv").write_text("DATE,PATIENT,CODE,VALUE\n1,2\n", encoding="utf-8")
    out = tmp_path / "out"
    assert main([str(directory), "--output-dir", str(out)]) == EXIT_ERROR
    assert (out / MARKDOWN_NAME).exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--lookback-years", "0"],
        ["--top", "0"],
        ["--reference-date", "2026-01-01", "--metadata", "m.json"],
        ["--age-bands", "7"],
    ],
)
def test_usage_errors(tmp_path, arguments):
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path)), *arguments])
    assert raised.value.code == EXIT_ERROR


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0 and __version__ in capsys.readouterr().out


def test_the_console_script_runs_as_installed(tmp_path):
    if not CONSOLE_SCRIPT.exists():
        pytest.skip("synthea-observations is not installed next to this interpreter")
    completed = subprocess.run(
        [str(CONSOLE_SCRIPT), str(dataset(tmp_path / "csv")), "--observation", "8480-6",
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
