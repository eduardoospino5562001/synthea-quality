"""Tests for the shared reader of the exported history."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from synthea_quality.export_history import (
    ExportHistory,
    notice_lines,
    read_export_history,
    synthea_cutoff,
)

REF = date(2026, 8, 17)


def metadata_file(directory: Path, payload) -> Path:
    path = directory / "run.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_without_metadata_nothing_is_known_and_there_is_no_notice():
    history = read_export_history(None, REF)
    assert history.status == "no_metadata"
    assert history.years is None and history.cutoff is None
    assert history.to_dict() == {
        "years_of_history": None,
        "status": "no_metadata",
        "metadata_file": None,
        "cutoff": None,
        "reason": None,
    }
    assert notice_lines(history) == []


@pytest.mark.parametrize("value", [0, "0"])
def test_zero_means_the_whole_history_and_gives_no_notice(tmp_path, value):
    path = metadata_file(tmp_path, {"endTime": "20260817", "exporter.years_of_history": value})
    history = read_export_history(path, REF)
    assert (history.status, history.years, history.cutoff) == ("read", 0, None)
    assert notice_lines(history) == []


@pytest.mark.parametrize("value", [10, "10"])
def test_an_integer_and_its_string_form_are_valid(tmp_path, value):
    path = metadata_file(tmp_path, {"endTime": "20260817", "exporter.years_of_history": value})
    history = read_export_history(path, REF)
    assert history.status == "read" and history.years == 10
    assert history.cutoff == REF - timedelta(days=365 * 10)
    assert history.metadata_file == str(path)
    assert ExportHistory.from_dict(history.to_dict()) == history


@pytest.mark.parametrize("value", ["abc", "-1", -1, 2.5, 10.0, True, [10], {"n": 10}])
def test_a_negative_decimal_or_textual_value_is_invalid(tmp_path, value):
    path = metadata_file(tmp_path, {"endTime": "20260817", "exporter.years_of_history": value})
    history = read_export_history(path, REF)
    assert history.status == "invalid" and history.years is None
    assert "not a non-negative integer" in (history.reason or "")
    assert len(notice_lines(history)) == 1
    assert "Exported history unknown" in notice_lines(history)[0]


def test_a_missing_key_is_reported_without_raising(tmp_path):
    path = metadata_file(tmp_path, {"endTime": "20260817"})
    history = read_export_history(path, REF)
    assert (history.status, history.reason) == ("missing_key", "the key is absent")
    assert "Exported history unknown" in "\n".join(notice_lines(history))


def test_an_unreadable_file_is_invalid_without_raising(tmp_path):
    history = read_export_history(tmp_path / "absent.json", REF)
    assert history.status == "invalid" and "could not be read" in (history.reason or "")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert read_export_history(bad, REF).status == "invalid"
    array = metadata_file(tmp_path, [1, 2])
    assert read_export_history(array, REF).status == "invalid"


def test_the_cutoff_subtracts_365_days_per_year_even_over_a_leap_day():
    # 2024-03-01 minus 365 days is 2023-03-02: the leap day is inside the span,
    # so a calendar-year subtraction (2023-03-01) would land a day off.
    assert synthea_cutoff(date(2024, 3, 1), 1) == date(2023, 3, 2)
    assert synthea_cutoff(REF, 10) == REF - timedelta(days=3650)
    with pytest.raises(ValueError):
        synthea_cutoff(REF, -1)


def test_the_notice_names_the_setting_file_and_cutoff(tmp_path):
    path = metadata_file(tmp_path, {"endTime": "20260817", "exporter.years_of_history": "10"})
    lines = notice_lines(read_export_history(path, REF))
    text = "\n".join(lines)
    assert "Time-filtered dataset" in text
    assert "`exporter.years_of_history = 10`" in text
    assert f"(read from `{path}`)" in text
    assert (REF - timedelta(days=3650)).isoformat() in text
    assert "may rest on incomplete data" in text


def test_without_a_reference_date_there_is_no_cutoff_but_the_notice_stays(tmp_path):
    path = metadata_file(tmp_path, {"endTime": "20260817", "exporter.years_of_history": 10})
    history = read_export_history(path, None)
    assert history.cutoff is None
    assert "an unknown date" in "\n".join(notice_lines(history))
