"""Tests for strict date parsing and the reference date resolution."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from synthea_quality.discovery import discover_dataset
from synthea_quality.loader import DatasetLoader
from synthea_quality.profile.dates import parse_date_only, parse_timestamps
from synthea_quality.profile.models import ReferenceSource
from synthea_quality.profile.reference import (
    ReferenceDateError,
    latest_encounter,
    parse_reference_date,
    reference_from_metadata,
    reference_from_user,
    resolve_reference_date,
)
from synthea_quality.schema.tables import tables_by_name
from synthea_quality.structure import validate_tables


def write_table(directory: Path, name: str, rows: list[dict[str, str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def text(*values: str | None) -> pd.Series:
    """A column as the loader produces it: text, with an empty field as the only null."""
    return pd.Series(values, dtype="str")


def write_metadata(path: Path, end_time: object) -> Path:
    path.write_text(json.dumps({"runID": "x", "endTime": end_time}), encoding="utf-8")
    return path


def resolve(directory: Path, **kwargs):
    discovery = discover_dataset(directory)
    structure = validate_tables({t.name: t.path for t in discovery.tables})
    return resolve_reference_date(discovery, DatasetLoader(), structure=structure, **kwargs)


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #


def test_date_only_counts_empty_and_unparseable_values_separately():
    parsed = parse_date_only(text("2020-01-31", None, "2021-02-30", "2020-1-5", "junk"))
    assert parsed.valid == 1
    assert parsed.empty == 1
    assert parsed.unparseable == 3
    assert parsed.values.iloc[0] == pd.Timestamp("2020-01-31")


def test_date_only_rejects_a_timestamp_shape():
    assert parse_date_only(text("2020-01-31T00:00:00Z")).unparseable == 1


def test_timestamps_require_the_confirmed_iso_shape():
    parsed = parse_timestamps(
        text("2026-08-17T00:50:14Z", "2026-08-17 00:50:14", "2026-08-17", None)
    )
    assert parsed.valid == 1
    assert parsed.unparseable == 2
    assert parsed.empty == 1


def test_date_parsing_rejects_non_ascii_digits() -> None:
    """Non-ASCII digits stay unparseable: strptime already rejected them, and the ASCII shape check now agrees."""
    assert parse_date_only(text("٢٠٢٠-٠١-٠١")).unparseable == 1
    assert parse_date_only(text("٢٠٢٠-٠١-٠١")).valid == 0
    assert parse_date_only(text("２０２０-０１-０１")).unparseable == 1
    assert parse_date_only(text("2020-01-01")).valid == 1


def test_timestamp_parsing_rejects_non_ascii_digits() -> None:
    """Non-ASCII timestamp digits stay unparseable: strptime rejected them; the ASCII check agrees."""
    assert parse_timestamps(text("٢٠٢٠-٠١-٠١T٠٠:٠٠:٠٠Z")).unparseable == 1
    assert parse_timestamps(text("٢٠٢٠-٠١-٠١T٠٠:٠٠:٠٠Z")).valid == 0
    assert parse_timestamps(text("２０２０-０１-０１T００:００:００Z")).unparseable == 1
    assert parse_timestamps(text("2026-08-17T00:50:14Z")).valid == 1


# --------------------------------------------------------------------------- #
# explicit sources
# --------------------------------------------------------------------------- #


def test_parse_reference_date_is_strict():
    assert parse_reference_date("2026-08-17").isoformat() == "2026-08-17"
    for bad in ("2026-8-17", "20260817", "2026-02-30", ""):
        with pytest.raises(ValueError):
            parse_reference_date(bad)


def test_reference_from_user_is_exact_and_attributed():
    reference = reference_from_user("2020-06-01")
    assert reference.value == "2020-06-01"
    assert reference.source is ReferenceSource.USER
    assert reference.approximate is False
    with pytest.raises(ReferenceDateError):
        reference_from_user("06/01/2020")


def test_reference_from_metadata_reads_end_time(tmp_path):
    reference = reference_from_metadata(write_metadata(tmp_path / "m.json", "20260818"))
    assert reference.value == "2026-08-18"
    assert reference.source is ReferenceSource.SYNTHEA_METADATA
    assert reference.approximate is False
    assert "endTime=20260818" in reference.detail
    assert "MetadataExporter" in reference.detail


@pytest.mark.parametrize("end_time", [None, 20260818, "2026-08-18", "20260230"])
def test_reference_from_metadata_rejects_a_missing_or_malformed_end_time(tmp_path, end_time):
    path = write_metadata(tmp_path / "m.json", end_time)
    with pytest.raises(ReferenceDateError, match="endTime"):
        reference_from_metadata(path)


def test_reference_from_metadata_rejects_unreadable_or_non_object_files(tmp_path):
    with pytest.raises(ReferenceDateError, match="could not be read"):
        reference_from_metadata(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ReferenceDateError, match="not valid JSON"):
        reference_from_metadata(bad)
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ReferenceDateError, match="not a JSON object"):
        reference_from_metadata(array)


# --------------------------------------------------------------------------- #
# approximation from encounters
# --------------------------------------------------------------------------- #


def test_latest_encounter_scans_start_and_stop_and_counts_bad_values():
    frame = pd.DataFrame(
        {
            "START": text("2020-01-01T00:00:00Z", "2021-05-05T10:00:00Z", "bad"),
            "STOP": text("2020-01-01T01:00:00Z", "2021-05-06T09:30:00Z", None),
        }
    )
    latest = latest_encounter(frame)
    assert latest.timestamp == datetime(2021, 5, 6, 9, 30)
    assert latest.values == 4
    assert latest.unparseable == 1


def test_resolution_falls_back_to_the_latest_encounter(tmp_path):
    write_table(
        tmp_path,
        "encounters",
        [
            {"Id": "e1", "START": "2020-01-01T00:00:00Z", "STOP": "2020-01-01T01:00:00Z"},
            {"Id": "e2", "START": "2021-12-31T23:00:00Z", "STOP": "2022-01-01T00:10:00Z"},
            # the unset-stop sentinel is older than everything and cannot win the max
            {"Id": "e3", "START": "2021-06-01T00:00:00Z", "STOP": "1970-01-01T00:00:00Z"},
        ],
    )
    resolution = resolve(tmp_path)
    assert resolution.reference is not None
    assert resolution.reference.value == "2022-01-01"
    assert resolution.reference.source is ReferenceSource.MAX_ENCOUNTER_DATE
    assert resolution.reference.approximate is True
    assert resolution.reference.detail.startswith("APPROXIMATION")
    assert resolution.encounters.rows == 3
    assert resolution.encounters.state.value == "read"
    assert resolution.notes == ()


def test_resolution_without_encounters_has_a_reason_and_no_date(tmp_path):
    write_table(tmp_path, "patients", [{"Id": "p1"}])
    resolution = resolve(tmp_path)
    assert resolution.reference is None
    assert "encounters.csv is not in the dataset" in resolution.reason
    assert resolution.encounters.state.value == "absent"


def test_resolution_with_no_parseable_encounter_timestamp(tmp_path):
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "soon", "STOP": ""}])
    resolution = resolve(tmp_path)
    assert resolution.reference is None
    assert "no parseable" in resolution.reason


def test_resolution_does_not_trust_a_structurally_broken_encounters_file(tmp_path):
    path = write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2020-01-01T00:00:00Z"}])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("e2,2030-01-01T00:00:00Z\n")  # a short row pandas would pad
    resolution = resolve(tmp_path)
    assert resolution.reference is None
    assert "field count" in resolution.reason
    assert resolution.encounters.state.value == "unreadable"


def test_an_explicit_date_wins_over_the_approximation(tmp_path):
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2020-01-01T00:00:00Z"}])
    resolution = resolve(tmp_path, user_date="2019-01-01")
    assert resolution.reference.value == "2019-01-01"
    assert resolution.reference.source is ReferenceSource.USER
    assert resolution.notes == ()
    assert resolution.encounters is None  # nothing is read to second-guess the user


def test_metadata_earlier_than_the_latest_encounter_adds_a_note(tmp_path):
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2020-03-02T00:00:00Z"}])
    metadata = write_metadata(tmp_path / "run.json", "20200301")
    resolution = resolve(tmp_path, metadata_path=metadata)
    assert resolution.reference.value == "2020-03-01"
    assert resolution.reference.source is ReferenceSource.SYNTHEA_METADATA
    assert len(resolution.notes) == 1
    assert "earlier than the latest encounter (2020-03-02)" in resolution.notes[0]
    assert resolution.encounters.used_for.startswith("consistency")


def test_metadata_on_or_after_the_latest_encounter_adds_no_note(tmp_path):
    write_table(tmp_path, "encounters", [{"Id": "e1", "START": "2020-03-01T23:59:59Z"}])
    metadata = write_metadata(tmp_path / "run.json", "20200301")
    assert resolve(tmp_path, metadata_path=metadata).notes == ()


def test_explicit_sources_are_mutually_exclusive(tmp_path):
    write_table(tmp_path, "encounters", [])
    with pytest.raises(ValueError, match="not both"):
        resolve(tmp_path, user_date="2020-01-01", metadata_path=tmp_path / "m.json")


def test_metadata_without_encounters_is_used_and_records_the_absent_table(tmp_path):
    write_table(tmp_path, "patients", [{"Id": "p1"}])
    metadata = write_metadata(tmp_path / "run.json", "20200301")
    resolution = resolve(tmp_path, metadata_path=metadata)
    assert resolution.reference.value == "2020-03-01"
    assert resolution.notes == ()
    assert resolution.encounters.state.value == "absent"
