"""Tests for the user-supplied code exclusion list of ``synthea-prevalence``."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from synthea_quality.prevalence.build import build_prevalence
from synthea_quality.prevalence.cli import EXIT_ERROR, EXIT_OK, main
from synthea_quality.prevalence.compute import build_cohort, general_table, prepare_records
from synthea_quality.prevalence.exclusion import (
    ExclusionListError,
    codes_absent_from_data,
    read_exclusion_file,
)
from synthea_quality.prevalence.render import JSON_NAME
from synthea_quality.schema.tables import tables_by_name

REF = date(2026, 8, 17)
BANDS = (0, 18, 65)
SNOMED = "http://snomed.info/sct"


def write_list(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# the reader
# --------------------------------------------------------------------------- #


def test_comments_blank_lines_and_trailing_comments_are_ignored(tmp_path):
    path = write_list(
        tmp_path, "exclude.txt",
        "# a comment\n\n   \n73595000 # Stress (finding)\n160903007\n",
    )
    exclusion = read_exclusion_file(path)
    assert exclusion.codes == ("160903007", "73595000")
    assert exclusion.duplicates == 0
    assert exclusion.path == str(path)
    assert exclusion.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    described = exclusion.describe()
    assert described["id"] == "file:exclude.txt"
    assert described["source"] == "file"
    assert described["codes"] == 2
    assert [entry["code"] for entry in described["entries"]] == ["160903007", "73595000"]


def test_repeated_codes_are_kept_once_and_counted(tmp_path):
    path = write_list(tmp_path, "exclude.txt", "73595000\n160903007\n73595000\n73595000\n")
    exclusion = read_exclusion_file(path)
    assert exclusion.codes == ("160903007", "73595000")
    assert exclusion.duplicates == 2


def test_extra_text_after_the_code_is_an_error_with_line_number(tmp_path):
    path = write_list(tmp_path, "exclude.txt", "73595000\n160903007 Stress\n")
    with pytest.raises(ExclusionListError, match=r":2:.*extra text"):
        read_exclusion_file(path)


def test_an_invalid_code_is_an_error_with_line_number(tmp_path):
    path = write_list(tmp_path, "exclude.txt", "73595000\nabc!\n")
    with pytest.raises(ExclusionListError, match=r":2:.*invalid code"):
        read_exclusion_file(path)


def test_a_file_without_codes_is_valid_and_excludes_nothing(tmp_path):
    path = write_list(tmp_path, "exclude.txt", "# only comments\n\n")
    exclusion = read_exclusion_file(path)
    assert exclusion.codes == () and exclusion.duplicates == 0


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(ExclusionListError, match="could not be read"):
        read_exclusion_file(tmp_path / "absent.txt")


def test_a_non_utf8_file_is_an_error(tmp_path):
    path = tmp_path / "exclude.txt"
    path.write_bytes(b"73595000\n\xff\xfe\n")
    with pytest.raises(ExclusionListError, match="not UTF-8"):
        read_exclusion_file(path)


def test_a_utf8_bom_is_accepted(tmp_path):
    plain = write_list(tmp_path, "plain.txt", "73595000 # Stress\n160903007\n")
    bom = tmp_path / "bom.txt"
    bom.write_bytes(b"\xef\xbb\xbf" + plain.read_bytes())
    assert read_exclusion_file(bom).codes == read_exclusion_file(plain).codes


def test_absent_codes_are_listed_in_stable_order():
    from synthea_quality.prevalence.exclusion import ExclusionList

    listed = ExclusionList(path="f.txt", sha256="0" * 64, codes=("b", "a", "c"), duplicates=0)
    assert codes_absent_from_data(listed, {"b"}) == ("a", "c")


def test_the_example_file_holds_the_built_in_codes():
    from synthea_quality.prevalence.social import SOCIAL_CODES

    example = Path(__file__).resolve().parent.parent / "examples" / "social_codes.txt"
    exclusion = read_exclusion_file(example)
    assert sorted(exclusion.codes) == sorted(entry.code for entry in SOCIAL_CODES)
    assert exclusion.duplicates == 0


# --------------------------------------------------------------------------- #
# the computation over a small dataset
# --------------------------------------------------------------------------- #

PATIENT_COLUMNS = ("Id", "BIRTHDATE", "DEATHDATE", "GENDER")
CONDITION_COLUMNS = ("PATIENT", "CODE", "START", "STOP", "SYSTEM", "DESCRIPTION")


def text_frame(rows: list[dict[str, str]], columns) -> object:
    import pandas as pd

    data = pd.DataFrame([[row.get(c, "") for c in columns] for row in rows], columns=list(columns))
    return data.replace("", None).astype("str") if len(data) else data.astype("str")


def small_dataset():
    import pandas as pd  # noqa: F401 - keeps the helper's import local and explicit

    patients = text_frame(
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-01-01", "GENDER": "F"},
        ],
        PATIENT_COLUMNS,
    )
    conditions = text_frame(
        [
            {"PATIENT": "a1", "CODE": "73595000", "START": "2020-01-01", "SYSTEM": SNOMED,
             "DESCRIPTION": "Stress (finding)"},
            {"PATIENT": "a2", "CODE": "160903007", "START": "2020-01-01", "SYSTEM": SNOMED,
             "DESCRIPTION": "Full-time employment (finding)"},
            {"PATIENT": "a1", "CODE": "59621000", "START": "2020-01-01", "SYSTEM": SNOMED,
             "DESCRIPTION": "Essential hypertension (disorder)"},
        ],
        CONDITION_COLUMNS,
    )
    cohort = build_cohort(patients, REF, BANDS)
    return prepare_records(conditions, cohort, REF), cohort


def test_a_file_with_two_present_codes_and_one_absent(tmp_path):
    records, cohort = small_dataset()
    path = write_list(tmp_path, "exclude.txt", "73595000\n160903007\n99999999\n")
    exclusion = read_exclusion_file(path)
    table = general_table(records, cohort, exclude=exclusion)
    assert [row.code for row in table.rows] == ["59621000"]
    assert table.metrics["social_codes_in_data"] == 2
    assert table.metrics["social_records_in_data"] == 2
    assert any(str(path) in note and "are excluded" in note for note in table.notes)
    assert any("`99999999`" in note for note in table.notes)
    assert table.include_social is False


def test_an_empty_file_excludes_nothing_and_says_so(tmp_path):
    records, cohort = small_dataset()
    path = write_list(tmp_path, "exclude.txt", "# nothing listed\n")
    table = general_table(records, cohort, exclude=read_exclusion_file(path))
    assert sorted(row.code for row in table.rows) == ["160903007", "59621000", "73595000"]
    assert table.metrics["social_codes_in_data"] == 0
    assert any("0 code(s) listed" in note for note in table.notes)


def test_an_exclusion_file_and_include_social_cannot_be_combined(tmp_path):
    records, cohort = small_dataset()
    path = write_list(tmp_path, "exclude.txt", "73595000\n")
    with pytest.raises(ValueError):
        general_table(records, cohort, include_social=True, exclude=read_exclusion_file(path))
    with pytest.raises(ValueError):
        build_prevalence(tmp_path, include_social=True, exclude_codes=str(path))


# --------------------------------------------------------------------------- #
# the report and the command line
# --------------------------------------------------------------------------- #


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


def test_the_report_records_the_file_and_what_was_absent(tmp_path):
    path = write_list(tmp_path / "lists", "exclude.txt", "73595000\n99999999\n")
    report = build_prevalence(
        dataset(tmp_path / "csv"), exclude_codes=path, generated_at="2026-10-02T00:00:00+00:00"
    )
    assert report.social_list["source"] == "file"
    assert report.social_list["id"] == "file:exclude.txt"
    assert report.social_list["listed_absent_from_data"] == ["99999999"]
    assert [row.code for row in report.general.rows] == ["59621000"]


def test_without_the_option_the_built_in_list_is_recorded(tmp_path):
    report = build_prevalence(
        dataset(tmp_path / "csv"), generated_at="2026-10-02T00:00:00+00:00"
    )
    assert report.social_list["source"] == "built-in"
    assert "listed_absent_from_data" not in report.social_list
    assert [row.code for row in report.general.rows] == ["59621000"]


def test_cli_exits_two_with_both_options(tmp_path):
    path = write_list(tmp_path, "exclude.txt", "73595000\n")
    with pytest.raises(SystemExit) as raised:
        main([str(dataset(tmp_path / "csv")), "--include-social",
              "--exclude-codes", str(path)])
    assert raised.value.code == EXIT_ERROR


def test_cli_exits_two_with_an_unusable_file_and_says_so_without_a_traceback(tmp_path, capsys):
    code = main([str(dataset(tmp_path / "csv")), "--exclude-codes",
                 str(tmp_path / "absent.txt"), "--output-dir", str(tmp_path / "o")])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "exclusion file" in err and "Traceback" not in err
    assert not (tmp_path / "o").exists()


def test_cli_writes_the_report_with_the_file(tmp_path, capsys):
    path = write_list(tmp_path, "exclude.txt", "73595000\n")
    out = tmp_path / "out"
    code = main([str(dataset(tmp_path / "csv")), "--exclude-codes", str(path),
                 "--output-dir", str(out)])
    assert code == EXIT_OK
    data = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    assert data["social_list"]["source"] == "file"
    assert [row["code"] for row in data["general"]["rows"]] == ["59621000"]
    assert f"excluded via {path}" in capsys.readouterr().out
