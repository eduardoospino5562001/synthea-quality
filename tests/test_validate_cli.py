"""Tests for the ``synthea-validate-module`` command line interface."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthea_quality import __version__
from synthea_quality.schema.tables import tables_by_name
from synthea_quality.validate.cli import EXIT_ERROR, EXIT_OK, main
from synthea_quality.validate.models import load_module_file
from synthea_quality.validate.render import JSON_NAME, MARKDOWN_NAME

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples" / "myocardial_infarction.json"
CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-validate-module")
SNOMED = "http://snomed.info/sct"


def write_table(directory: Path, name: str, rows) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)


def dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1970-01-01", "GENDER": "F"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [{"PATIENT": "a1", "CODE": "401314000", "SYSTEM": SNOMED, "START": "2023-01-01"}],
    )
    return directory


def test_the_example_module_file_is_valid_and_says_its_values_are_illustrative():
    module, (mi,) = load_module_file(EXAMPLE)
    assert module.name == "Myocardial infarction"
    assert "myocardial_infarction.json" in module.synthea_modules
    assert mi.acute
    assert {e.measure for e in mi.expected} == {"lifetime", "incidence"}
    assert all(e.source == "illustrative — replace with a cited reference" for e in mi.expected)


def test_writes_both_files_and_summarises(tmp_path, capsys):
    out = tmp_path / "out"
    code = main(
        [str(dataset(tmp_path / "csv")), "--module", str(EXAMPLE), "--output-dir", str(out)]
    )
    assert code == EXIT_OK
    assert (out / MARKDOWN_NAME).read_text("utf-8").startswith(
        "# Module validation: Myocardial infarction"
    )
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert data["conditions"][0]["prevalence"]["lifetime"]["numerator"] == 1
    printed = capsys.readouterr().out
    assert "Module:     Myocardial infarction" in printed
    assert "Condition:  Myocardial infarction: lifetime 1/2, 1 new case(s)" in printed


def test_the_module_file_is_required(tmp_path):
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path))])
    assert raised.value.code == EXIT_ERROR


def test_an_unusable_module_file_is_an_error_without_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"conditions": []}', encoding="utf-8")
    code = main([str(dataset(tmp_path / "csv")), "--module", str(bad),
                 "--output-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "non-empty 'conditions' list" in err and "Traceback" not in err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--window-years", "0"],
        ["--reference-date", "2026-01-01", "--metadata", "m.json"],
        ["--age-bands", "7"],
    ],
)
def test_usage_errors(tmp_path, arguments):
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path)), "--module", str(EXAMPLE), *arguments])
    assert raised.value.code == EXIT_ERROR


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0 and __version__ in capsys.readouterr().out


def test_the_console_script_runs_as_installed(tmp_path):
    if not CONSOLE_SCRIPT.exists():
        pytest.skip("synthea-validate-module is not installed next to this interpreter")
    completed = subprocess.run(
        [str(CONSOLE_SCRIPT), str(dataset(tmp_path / "csv")), "--module", str(EXAMPLE),
         "--output-dir", str(tmp_path / "o")],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
