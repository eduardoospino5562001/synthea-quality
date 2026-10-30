"""Tests for the ``synthea-compare`` command line interface (cli.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from csv_tables import write_table

from synthea_quality import __version__
from synthea_quality.compare.cli import EXIT_ERROR, EXIT_OK, main
from synthea_quality.compare.render import BLOCK_TITLES, BLOCKS, JSON_NAME, MARKDOWN_NAME

REPOSITORY = Path(__file__).resolve().parents[1]
EXAMPLE = REPOSITORY / "examples" / "myocardial_infarction.json"
CONSOLE_SCRIPT = Path(sys.executable).with_name("synthea-compare")
SNOMED = "http://snomed.info/sct"


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


def test_writes_both_files_and_summarises(tmp_path, capsys):
    base = dataset(tmp_path / "base")
    cand = dataset(tmp_path / "cand")
    out = tmp_path / "out"
    code = main(
        [
            str(base),
            str(cand),
            "--module",
            str(EXAMPLE),
            "--output-dir",
            str(out),
            "--reference-date",
            "2026-01-01",
        ]
    )
    assert code == EXIT_OK
    assert (out / MARKDOWN_NAME).exists()
    assert (out / JSON_NAME).exists()
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    printed = capsys.readouterr().out
    assert "Baseline:   " in printed
    assert "Candidate:  " in printed
    assert "Reports:" in printed
    assert str(out / MARKDOWN_NAME) in printed
    assert str(out / JSON_NAME) in printed


def test_same_directory_twice_is_valid(tmp_path, capsys):
    base = dataset(tmp_path / "csv")
    out = tmp_path / "out"
    code = main(
        [str(base), str(base), "--module", str(EXAMPLE), "--output-dir", str(out),
         "--reference-date", "2026-01-01"]
    )
    assert code == EXIT_OK
    assert (out / MARKDOWN_NAME).exists()
    assert (out / JSON_NAME).exists()


def test_the_module_file_is_required(tmp_path):
    base = dataset(tmp_path / "csv")
    with pytest.raises(SystemExit) as raised:
        main([str(base), str(base)])
    assert raised.value.code == EXIT_ERROR


def test_baseline_directory_that_does_not_exist_is_exit_2(tmp_path):
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path / "nope"), str(tmp_path / "nope"), "--module", str(EXAMPLE)])
    assert raised.value.code == EXIT_ERROR


def test_candidate_that_is_a_file_not_a_directory_is_exit_2(tmp_path):
    base = dataset(tmp_path / "base")
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        main([str(base), str(not_a_dir), "--module", str(EXAMPLE)])
    assert raised.value.code == EXIT_ERROR


def test_reference_date_and_metadata_cannot_be_combined(tmp_path):
    base = dataset(tmp_path / "base")
    cand = dataset(tmp_path / "cand")
    code = main(
        [
            str(base),
            str(cand),
            "--module",
            str(EXAMPLE),
            "--reference-date",
            "2026-01-01",
            "--metadata-baseline",
            str(tmp_path / "m.json"),
        ]
    )
    assert code == EXIT_ERROR


def test_metadata_requires_both_sides(tmp_path):
    base = dataset(tmp_path / "base")
    cand = dataset(tmp_path / "cand")
    code = main(
        [str(base), str(cand), "--module", str(EXAMPLE),
         "--metadata-baseline", str(tmp_path / "m.json")]
    )
    assert code == EXIT_ERROR


def test_version(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0 and __version__ in capsys.readouterr().out


def test_the_console_script_runs_as_installed(tmp_path):
    if not CONSOLE_SCRIPT.exists():
        pytest.skip("synthea-compare is not installed next to this interpreter")
    base = dataset(tmp_path / "base")
    cand = dataset(tmp_path / "cand")
    completed = subprocess.run(
        [
            str(CONSOLE_SCRIPT), str(base), str(cand), "--module", str(EXAMPLE),
            "--output-dir", str(tmp_path / "o"), "--reference-date", "2026-01-01",
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == EXIT_OK, completed.stderr


def test_the_console_prints_one_counts_line_per_block_like_the_markdown(tmp_path, capsys):
    base = dataset(tmp_path / "base")
    out = tmp_path / "out"
    code = main(
        [str(base), str(base), "--module", str(EXAMPLE), "--output-dir", str(out),
         "--reference-date", "2026-01-01"]
    )
    assert code == EXIT_OK
    printed = capsys.readouterr().out.splitlines()
    markdown = (out / MARKDOWN_NAME).read_text(encoding="utf-8").splitlines()
    for block in BLOCKS:
        title = BLOCK_TITLES[block]
        if f"## {title}" not in markdown:
            continue
        counts = markdown[markdown.index(f"## {title}") + 2]
        # The console repeats the block's line of the Markdown, without its final period.
        assert f"{title}:  {counts.removesuffix('.')}" in printed, title
    assert not any(" compared, " in line for line in printed)


def test_the_module_example_of_the_help_has_balanced_brackets(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    printed = capsys.readouterr().out
    example = printed.split("module file (see examples/myocardial_infarction.json):")[1].split("\n\n")[0]
    assert example.count("[") == example.count("]")
    assert example.count("{") == example.count("}")
    assert "}]]," not in example
