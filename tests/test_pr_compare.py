"""Use fake builds but the real comparison CLI to check the shell contract."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from csv_tables import write_table

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "pr_compare.sh"
EXAMPLE = REPOSITORY / "examples" / "myocardial_infarction.json"


def fake_build(directory: Path, *, fail: bool = False, metadata: bool = True) -> Path:
    """Export a small fixed dataset so tests need neither Java nor Gradle."""
    directory.mkdir()
    source = directory / "dataset"
    write_table(source, "patients", [
        {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
        {"Id": "a2", "BIRTHDATE": "1970-01-01", "GENDER": "F"},
    ])
    write_table(source, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(source, "conditions", [{
        "PATIENT": "a1", "CODE": "401314000", "SYSTEM": "http://snomed.info/sct",
        "START": "2023-01-01",
    }])
    executable = directory / "run_synthea"
    executable.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + ("export FAKE_SYNTHEA_FAIL=1\n" if fail else "")
        + 'BUILD="$(cd "$(dirname "$0")" && pwd)"\n'
        'printf "%s\\n" "$@" > "$BUILD/args.txt"\n'
        '[ "${FAKE_SYNTHEA_FAIL:-0}" != 1 ] || exit 1\n'
        'out=""\ncsv=false\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        '    --exporter.baseDirectory=*) out="${arg#*=}" ;;\n'
        '    --exporter.csv.export=true) csv=true ;;\n'
        '  esac\ndone\n'
        '[ "$csv" = true ] || exit 1\n'
        'mkdir -p "$out/csv"\n'
        'cp "$BUILD/dataset/"*.csv "$out/csv/"\n'
        + ('mkdir -p "$out/metadata"\n'
            'printf \'{"referenceTime": "20260101", "endTime": "20260101"}\\n\' > "$out/metadata/run.json"\n'
           if metadata else ""),
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return directory


@pytest.fixture
def comparison(tmp_path: Path) -> list[str]:
    """Keep build paths with spaces to exercise shell argument quoting."""
    base = fake_build(tmp_path / "base build")
    pr = fake_build(tmp_path / "pr build")
    return [str(base), str(pr), str(EXAMPLE), "7", "1000", str(tmp_path / "output")]


def run_script(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Isolate fake build failures from the caller's environment."""
    environment = os.environ.copy()
    environment.pop("FAKE_SYNTHEA_FAIL", None)
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments], capture_output=True, text=True,
        check=False, env=environment,
    )


def test_passes_seed_population_and_csv_flags(comparison: list[str]) -> None:
    """Both sides must receive identical generation settings."""
    result = run_script(comparison)
    assert result.returncode == 0, result.stderr
    for build, side in zip(comparison[:2], ("base", "pr"), strict=True):
        assert (Path(build) / "args.txt").read_text().splitlines() == [
            "-s", "7", "-cs", "7", "-p", "1000", "--exporter.csv.export=true",
            f"--exporter.baseDirectory={comparison[5]}/{side}/",
        ]


def test_writes_both_report_files(comparison: list[str]) -> None:
    """Successful comparison must leave both promised report formats."""
    result = run_script(comparison)
    assert result.returncode == 0, result.stderr
    for name in ("synthea_compare.md", "synthea_compare.json"):
        assert (Path(comparison[5]) / name).is_file()


def test_wrong_number_of_arguments_is_exit_1(comparison: list[str]) -> None:
    """Usage errors must be distinct from generation failures."""
    assert run_script(comparison[:5]).returncode == 1


def test_non_empty_out_dir_is_exit_1(comparison: list[str]) -> None:
    """Even hidden files must prevent reuse of an output directory."""
    output = Path(comparison[5])
    output.mkdir()
    (output / ".keep").write_text("keep", encoding="utf-8")
    assert run_script(comparison).returncode == 1


@pytest.mark.parametrize("seed", ["abc", "0"])
def test_seed_must_be_a_positive_integer(comparison: list[str], seed: str) -> None:
    """Reject nonnumeric and zero seeds before generating anything."""
    comparison[3] = seed
    assert run_script(comparison).returncode == 1


def test_synthea_failure_on_pr_side_is_exit_2(comparison: list[str], tmp_path: Path) -> None:
    """Identify which generator failed without attempting comparison."""
    comparison[1] = str(fake_build(tmp_path / "failing pr", fail=True))
    result = run_script(comparison)
    assert result.returncode == 2
    assert "pr" in result.stdout + result.stderr


def test_missing_metadata_is_exit_2(comparison: list[str], tmp_path: Path) -> None:
    """Do not silently fall back to an approximate reference date."""
    comparison[0] = str(fake_build(tmp_path / "no metadata", metadata=False))
    assert run_script(comparison).returncode == 2


def test_compare_failure_is_exit_3(comparison: list[str], tmp_path: Path) -> None:
    """Distinguish comparison errors from generator errors."""
    module = tmp_path / "invalid.json"
    module.write_text("{", encoding="utf-8")
    comparison[2] = str(module)
    result = run_script(comparison)
    assert result.returncode == 3
    assert result.stderr
