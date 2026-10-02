"""Tests for the ``synthea-profile`` command line interface."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthea_quality import __version__
from synthea_quality.profile.cli import EXIT_ERROR, EXIT_OK, main
from synthea_quality.profile.render import JSON_NAME, MARKDOWN_NAME
from synthea_quality.schema.tables import tables_by_name

CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-profile")


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
            {"Id": "p1", "BIRTHDATE": "2000-01-01", "GENDER": "F"},
            {"Id": "p2", "BIRTHDATE": "1930-01-01", "DEATHDATE": "2010-01-01", "GENDER": "M"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    return directory


@pytest.fixture
def tmp_cwd(tmp_path: Path, monkeypatch) -> Path:
    """The output directory defaults to the current one: never write into the repository."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_writes_both_files_and_exits_zero(tmp_path, capsys):
    out = tmp_path / "reports"
    assert main([str(dataset(tmp_path / "csv")), "--output-dir", str(out)]) == EXIT_OK
    assert (out / MARKDOWN_NAME).is_file()
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["tool_version"] == __version__
    assert data["data_dir"] == str(tmp_path / "csv")
    assert data["reference_date"]["source"] == "max_encounter_date"
    printed = capsys.readouterr().out
    assert "Reference:  2026-01-01 (APPROXIMATION, source: max_encounter_date)" in printed
    assert "Patients:   2 total - 1 alive, 1 deceased" in printed


def test_output_directory_defaults_to_the_current_directory(tmp_cwd):
    assert main([str(dataset(tmp_cwd / "csv"))]) == EXIT_OK
    assert (tmp_cwd / MARKDOWN_NAME).is_file()
    assert (tmp_cwd / JSON_NAME).is_file()


def test_reference_date_and_metadata_are_mutually_exclusive(tmp_path, capsys):
    metadata = tmp_path / "run.json"
    metadata.write_text(json.dumps({"endTime": "20260101"}), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main(
            [
                str(dataset(tmp_path / "csv")),
                "--reference-date", "2026-01-01",
                "--metadata", str(metadata),
                "--output-dir", str(tmp_path / "out"),
            ]
        )
    assert raised.value.code == EXIT_ERROR
    assert "not allowed with argument" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--reference-date", "2026-1-1"],
        ["--age-bands", "5,18"],
        ["--age-bands", "0,x"],
        ["--top-counties", "0"],
        ["--top", "0"],
        ["--top", "many"],
    ],
)
def test_invalid_options_are_usage_errors(tmp_path, arguments):
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path)), *arguments])
    assert raised.value.code == EXIT_ERROR


def test_missing_directory_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path / "absent")])
    assert raised.value.code == EXIT_ERROR


def test_an_unusable_metadata_file_is_an_error_without_traceback(tmp_path, capsys):
    code = main(
        [
            str(dataset(tmp_path / "csv")),
            "--metadata", str(tmp_path / "absent.json"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith("error: metadata file") and "Traceback" not in err


def test_a_metadata_file_with_an_oversized_number_is_invalid_json_without_traceback(
    tmp_path, capsys
):
    metadata = tmp_path / "big.json"
    metadata.write_text(
        '{"endTime": "20260101", "exporter.years_of_history": ' + "9" * 5000 + "}",
        encoding="utf-8",
    )
    code = main(
        [
            str(dataset(tmp_path / "csv")),
            "--metadata", str(metadata),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert str(metadata) in err and "not valid JSON" in err
    assert "Traceback" not in err
    assert not (tmp_path / "out").exists()


def test_an_inconsistent_metadata_end_time_is_a_note_not_an_error(tmp_path, capsys):
    metadata = tmp_path / "run.json"
    metadata.write_text(json.dumps({"endTime": "20251231"}), encoding="utf-8")
    out = tmp_path / "out"
    code = main(
        [str(dataset(tmp_path / "csv")), "--metadata", str(metadata), "--output-dir", str(out)]
    )
    assert code == EXIT_OK
    assert "Note:       The metadata endTime (2025-12-31) is earlier" in capsys.readouterr().out
    assert "earlier than the latest encounter" in (out / MARKDOWN_NAME).read_text("utf-8")


def test_options_reach_the_profile(tmp_path):
    out = tmp_path / "out"
    code = main(
        [
            str(dataset(tmp_path / "csv")),
            "--reference-date", "2020-01-01",
            "--age-bands", "0,18",
            "--output-dir", str(out),
        ]
    )
    assert code == EXIT_OK
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert data["reference_date"]["source"] == "user"
    assert data["age_bands"] == [0, 18]
    age = next(s for s in data["sections"] if s["section_id"] == "age")
    assert [row["value"] for row in age["distributions"]["age_band"]] == ["0-17", "18+"]


def test_missing_patients_still_writes_a_profile_and_exits_zero(tmp_path, capsys):
    directory = tmp_path / "csv"
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    assert main([str(directory), "--output-dir", str(tmp_path / "out")]) == EXIT_OK
    assert "not profiled — patients.csv is not in the dataset" in capsys.readouterr().out


def test_an_unreadable_patients_table_writes_an_incomplete_profile_and_exits_two(tmp_path):
    directory = dataset(tmp_path / "csv")
    (directory / "patients.csv").write_bytes(b"Id,BIRTHDATE\n\xff\xfe,2000-01-01\n")
    out = tmp_path / "out"
    assert main([str(directory), "--output-dir", str(out)]) == EXIT_ERROR
    assert "This profile is incomplete" in (out / MARKDOWN_NAME).read_text("utf-8")


def test_a_directory_without_known_tables_is_an_error(tmp_path, capsys):
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
    assert main([str(tmp_path), "--output-dir", str(tmp_path / "out")]) == EXIT_ERROR
    assert "no Synthea CSV table" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_the_console_script_runs_as_installed(tmp_path):
    if not CONSOLE_SCRIPT.exists():
        pytest.skip("synthea-profile is not installed next to this interpreter")
    out = tmp_path / "out"
    completed = subprocess.run(
        [str(CONSOLE_SCRIPT), str(dataset(tmp_path / "csv")), "--output-dir", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr
    assert (out / JSON_NAME).is_file()


def test_top_reaches_the_code_sections(tmp_path):
    directory = dataset(tmp_path / "csv")
    write_table(
        directory,
        "medications",
        [{"PATIENT": "p1", "CODE": str(code), "DESCRIPTION": f"drug {code}"} for code in range(5)],
    )
    out = tmp_path / "out"
    assert main([str(directory), "--top", "2", "--output-dir", str(out)]) == EXIT_OK
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    medications = next(s for s in data["sections"] if s["section_id"] == "codes.medications")
    assert [row["code"] for row in medications["codes"]] == ["0", "1"]
    assert medications["metrics"]["top"] == 2
    default = main([str(directory), "--output-dir", str(tmp_path / "default")])
    assert default == EXIT_OK
