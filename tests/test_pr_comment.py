"""Tests for the character-limited pull request comment formatter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from pr_comment import MAX_COMMENT_CHARS, build_comment  # noqa: E402

NOTICE = "_The full report is in the synthea-compare-report artifact of this workflow._"


def test_short_report_is_returned_unchanged() -> None:
    report = "# Report\n\nok\n"
    assert build_comment(report) == report


def test_report_of_exactly_the_limit_is_unchanged() -> None:
    report = "x" * 65_536
    assert MAX_COMMENT_CHARS == 65_536
    assert build_comment(report) == report


def test_long_report_is_cut_at_a_line_boundary() -> None:
    lines = [f"{index:04d}" + "x" * 46 for index in range(2_000)]
    report = "\n".join(lines)
    comment = build_comment(report)
    assert len(comment) <= MAX_COMMENT_CHARS
    assert comment.endswith("\n\n" + NOTICE)
    kept_lines = comment[: -len("\n\n" + NOTICE)].splitlines()
    assert kept_lines
    assert kept_lines == lines[: len(kept_lines)]


def test_one_line_longer_than_the_limit_is_cut_by_characters() -> None:
    comment = build_comment("x" * 70_000)
    assert len(comment) <= MAX_COMMENT_CHARS
    assert comment.endswith("\n\n" + NOTICE)
    assert comment[: -len("\n\n" + NOTICE)] == "x" * (
        MAX_COMMENT_CHARS - len("\n\n" + NOTICE)
    )


def test_leading_newline_does_not_empty_the_comment() -> None:
    report = "\n" + "x" * 80_000
    suffix = "\n\n" + NOTICE
    comment = build_comment(report)
    assert len(comment) > len(suffix)
    assert len(comment) <= MAX_COMMENT_CHARS
    assert comment.endswith(suffix)


def test_multibyte_text_counts_characters_not_bytes() -> None:
    report = "\n".join(["é" * 1_000] * 40)
    assert len(report.encode("utf-8")) > MAX_COMMENT_CHARS
    assert len(report) <= MAX_COMMENT_CHARS
    assert build_comment(report) == report


def test_empty_report_is_value_error() -> None:
    for report in ("", "   \n"):
        with pytest.raises(ValueError, match="^report is empty$"):
            build_comment(report)


def test_cli_prints_the_comment(tmp_path: Path) -> None:
    report = "# Report\n\nok\n"
    path = tmp_path / "report.md"
    path.write_text(report, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "pr_comment.py"), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == report
    assert result.stderr == ""


def test_cli_without_file_is_exit_1() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "pr_comment.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stderr
