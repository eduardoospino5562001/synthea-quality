"""[TESTS] Structural validation: every row must line up with the header.

The defect these tests pin down was measured before the fix, on a dataset whose
`patients.csv` held one row with four fields under a three-field header:

* ``pk.patients.Id`` reported ``PASS`` even though the file's rows did not line up with
  its columns, because the key checks load a column subset and pandas only complains
  about a long row when it reads every column;
* a *short* row was worse: pandas padded it with nulls, so ``nulls.patients`` reported
  ``PASS`` with an invented null cell and nothing anywhere said the row was short;
* both datasets ended with exit code ``0``.

A structural defect is dataset corruption, not a tool failure, so it is a ``FAIL``; and
no other check of that table may claim ``PASS``, because its rows cannot be attributed
to its columns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synthea_quality.checks.keys import run_key_checks
from synthea_quality.checks.quality import run_quality_checks
from synthea_quality.checks.temporal import run_temporal_checks
from synthea_quality.cli import EXIT_FINDINGS, EXIT_OK, exit_code_for
from synthea_quality.errors import TableLoadError
from synthea_quality.loader import load_table
from synthea_quality.models import Severity, Status
from synthea_quality.reporting.build import build_report
from synthea_quality.structure import validate_structure, validate_tables

HEADER = "Id,BIRTHDATE,GENDER\n"
CLEAN_ROWS = "p1,1980-01-01,M\np2,1990-01-01,F\n"
LONG_ROWS = "p1,1980-01-01,M\np2,1990-01-01,F,EXTRA\n"
SHORT_ROWS = "p1,1980-01-01,M\np2,1990-01-01\n"

ENCOUNTERS = (
    "Id,START,STOP,PATIENT\n"
    "e1,2020-01-01T00:00:00Z,2020-01-02T00:00:00Z,p1\n"
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def dataset(directory: Path, patients_rows: str) -> Path:
    """A small dataset whose ``patients.csv`` holds the rows under test."""
    write(directory / "patients.csv", HEADER + patients_rows)
    write(directory / "encounters.csv", ENCOUNTERS)
    return directory


def statuses_by_table(report) -> dict[str, set[Status]]:
    grouped: dict[str, set[Status]] = {}
    for check in report.checks:
        if check.table:
            grouped.setdefault(check.table, set()).add(check.status)
    return grouped


# --------------------------------------------------------------------------- #
# The validator itself
# --------------------------------------------------------------------------- #


def test_a_clean_file_has_no_structural_defect(tmp_path: Path) -> None:
    path = write(tmp_path / "patients.csv", HEADER + CLEAN_ROWS)

    report = validate_structure(path)

    assert report.ok
    assert report.defects_total == 0
    assert report.defects == ()
    assert report.rows_checked == 2
    assert report.header_fields == 3


def test_a_long_row_is_a_structural_defect(tmp_path: Path) -> None:
    path = write(tmp_path / "patients.csv", HEADER + LONG_ROWS)

    report = validate_structure(path)

    assert not report.ok
    assert report.defects_total == 1
    defect = report.defects[0]
    assert defect.row == 2  # the second data row, i.e. line 3 of the file
    assert defect.fields == 4
    assert "4" in report.as_reason() and "3" in report.as_reason()


def test_a_short_row_is_a_structural_defect(tmp_path: Path) -> None:
    path = write(tmp_path / "patients.csv", HEADER + SHORT_ROWS)

    report = validate_structure(path)

    assert not report.ok
    assert report.defects_total == 1
    assert report.defects[0].row == 2
    assert report.defects[0].fields == 2


def test_a_header_only_file_is_structurally_clean(tmp_path: Path) -> None:
    path = write(tmp_path / "patients.csv", HEADER)

    report = validate_structure(path)

    assert report.ok
    assert report.rows_checked == 0


def test_quoted_fields_holding_commas_and_newlines_are_not_defects(tmp_path: Path) -> None:
    """The validator must parse CSV properly, not split on commas or lines."""
    text = (
        'Id,DESCRIPTION,GENDER\n'
        'p1,"a value, with a comma",M\n'
        'p2,"a value\nwith a new line",F\n'
    )
    path = write(tmp_path / "patients.csv", text)

    report = validate_structure(path)

    assert report.ok
    assert report.rows_checked == 2


def test_a_crlf_file_is_not_a_defect(tmp_path: Path) -> None:
    path = tmp_path / "patients.csv"
    path.write_bytes((HEADER + CLEAN_ROWS).replace("\n", "\r\n").encode("utf-8"))

    report = validate_structure(path)

    assert report.ok
    assert report.rows_checked == 2


def test_defect_samples_are_bounded_but_the_count_is_complete(tmp_path: Path) -> None:
    path = write(tmp_path / "patients.csv", HEADER + "p1,1980-01-01,M,X\n" * 50)

    report = validate_structure(path, sample_limit=3)

    assert report.defects_total == 50
    assert len(report.defects) == 3
    assert [defect.row for defect in report.defects] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# The gate: no check of a structurally broken table may pass
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rows", [LONG_ROWS, SHORT_ROWS], ids=["long-row", "short-row"])
def test_no_check_of_a_broken_table_passes(tmp_path: Path, rows: str) -> None:
    broken = dataset(tmp_path, rows)

    report = build_report(broken)
    by_table = statuses_by_table(report)

    assert by_table["patients"], "the fixture must produce checks for patients"
    assert Status.PASS not in by_table["patients"]
    assert Status.ERROR not in by_table["patients"]


@pytest.mark.parametrize("rows", [LONG_ROWS, SHORT_ROWS], ids=["long-row", "short-row"])
def test_a_broken_table_is_a_failure_not_an_internal_error(tmp_path: Path, rows: str) -> None:
    broken = dataset(tmp_path, rows)

    report = build_report(broken)
    structural = [check for check in report.checks if check.check_id.startswith("structure.")]

    assert len(structural) == 1
    assert structural[0].status is Status.FAIL
    assert structural[0].severity is Severity.HIGH
    assert structural[0].table == "patients"
    assert structural[0].metrics["defect_rows"] == 1
    assert exit_code_for(report) == EXIT_FINDINGS


@pytest.mark.parametrize("rows", [LONG_ROWS, SHORT_ROWS], ids=["long-row", "short-row"])
def test_the_reason_names_the_table_and_the_rows(tmp_path: Path, rows: str) -> None:
    broken = dataset(tmp_path, rows)

    results = {
        check.check_id: check
        for check in (*run_key_checks(broken), *run_quality_checks(broken), *run_temporal_checks(broken))
    }

    pk = results["pk.patients.Id"]
    assert pk.status is Status.SKIPPED
    assert "field count" in pk.message
    assert "patients" in pk.message
    quality = results["nulls.patients"]
    assert quality.status is Status.SKIPPED
    assert "field count" in quality.message


def test_a_structurally_broken_parent_table_is_not_used_for_foreign_keys(
    tmp_path: Path,
) -> None:
    """A parent whose rows do not line up cannot be trusted to resolve references."""
    broken = dataset(tmp_path, SHORT_ROWS)

    results = {check.check_id: check for check in run_key_checks(broken)}

    assert results["fk.encounters.PATIENT->patients.Id"].status is Status.SKIPPED
    assert "field count" in results["fk.encounters.PATIENT->patients.Id"].message


def test_a_clean_dataset_keeps_every_verdict_and_gains_no_structure_check(
    tmp_path: Path,
) -> None:
    """A well-formed dataset must not change: the gate is silent when there is nothing to say."""
    clean = dataset(tmp_path, CLEAN_ROWS)

    report = build_report(clean)

    assert not [check for check in report.checks if check.check_id.startswith("structure.")]
    assert exit_code_for(report) == EXIT_OK
    assert Status.PASS in statuses_by_table(report)["patients"]


def test_a_broken_table_is_not_reported_as_an_unreadable_file(tmp_path: Path) -> None:
    """It is corruption of the dataset, so it is a finding, not a load error."""
    broken = dataset(tmp_path, LONG_ROWS)

    report = build_report(broken)

    assert report.load_errors == ()
    assert report.findings, "the structural defect must appear among the findings"


# --------------------------------------------------------------------------- #
# Blank lines: the gate must agree with the loader, which skips them
# --------------------------------------------------------------------------- #


def test_a_blank_line_is_not_a_structural_defect(tmp_path: Path) -> None:
    """pandas reads with ``skip_blank_lines=True``, so the validator has to agree.

    Measured before the fix: a blank line between records was reported as a structural
    FAIL — with the table's other checks skipped — because ``csv.reader`` answers ``[]``
    for a blank line and its field count then differs from the header's. A false positive
    about a file the loader reads without complaint.
    """
    path = write(tmp_path / "patients.csv", HEADER + "p1,1980-01-01,M\n\np2,1990-01-01,F\n")

    report = validate_structure(path)

    assert report.ok
    assert report.defects == ()
    assert report.rows_checked == 2


def test_a_whitespace_only_line_is_not_a_structural_defect(tmp_path: Path) -> None:
    for filler in ("   ", "\t", " \t ", "\r"):
        path = write(
            tmp_path / "patients.csv",
            HEADER + f"p1,1980-01-01,M\n{filler}\np2,1990-01-01,F\n",
        )

        report = validate_structure(path)

        assert report.ok, repr(filler)
        assert report.rows_checked == 2, repr(filler)


def test_blank_lines_between_records_and_at_the_end_are_ignored(tmp_path: Path) -> None:
    path = write(
        tmp_path / "patients.csv",
        HEADER + "p1,1980-01-01,M\n\n\n\np2,1990-01-01,F\n   \n",
    )

    report = validate_structure(path)

    assert report.ok
    assert report.rows_checked == 2


def test_a_row_of_empty_fields_is_a_row_and_not_a_blank_line(tmp_path: Path) -> None:
    """`,,` is a row of three empty fields; commas are not a blank line."""
    matching = write(tmp_path / "patients.csv", HEADER + "p1,1980-01-01,M\n,,\n")
    too_many = write(tmp_path / "other.csv", HEADER + "p1,1980-01-01,M\n,,,\n")

    counted = validate_structure(matching)
    defect = validate_structure(too_many)

    assert counted.ok
    assert counted.rows_checked == 2, "the empty-fields row is a real data row"
    assert defect.defects_total == 1
    assert defect.defects[0].row == 2
    assert defect.defects[0].fields == 4


def test_a_quoted_whitespace_value_is_a_row_and_not_a_blank_line(tmp_path: Path) -> None:
    """Quoted content is data: pandas keeps ``"   "`` as a value, so the validator must."""
    path = write(tmp_path / "patients.csv", HEADER + 'p1,1980-01-01,M\n"   ",1990-01-01,F\n')

    report = validate_structure(path)

    assert report.ok
    assert report.rows_checked == 2


def test_a_blank_line_inside_a_quoted_field_keeps_the_record_whole(tmp_path: Path) -> None:
    """A quoted value may contain a blank line; pandas keeps it, and so does the counter."""
    path = write(
        tmp_path / "patients.csv",
        'Id,DESCRIPTION\np1,"a value\n\nstill the same value"\n',
    )

    report = validate_structure(path)

    assert report.ok
    assert report.rows_checked == 1


@pytest.mark.parametrize(
    "rows",
    [
        CLEAN_ROWS,
        "p1,1980-01-01,M\n\np2,1990-01-01,F\n",
        "p1,1980-01-01,M\n   \np2,1990-01-01,F\n",
        "p1,1980-01-01,M\n\n\np2,1990-01-01,F\n\n",
        "p1,1980-01-01,M\n,,\n",
        'p1,1980-01-01,M\n"   ",1990-01-01,F\n',
    ],
    ids=["clean", "blank-line", "whitespace-line", "many-blank-lines", "empty-fields", "quoted-spaces"],
)
def test_the_validator_counts_the_same_rows_the_loader_reads(
    tmp_path: Path, rows: str
) -> None:
    """The gate protects the checks, so it must not disagree with the loader about rows."""
    path = write(tmp_path / "patients.csv", HEADER + rows)

    report = validate_structure(path)
    loaded = load_table(path)

    assert report.ok
    assert report.rows_checked == len(loaded.frame) == 2


# --------------------------------------------------------------------------- #
# Regression: a decoding error while reading the header
# --------------------------------------------------------------------------- #
#
# The first ``next(reader)`` decodes a whole buffer of the file, not only the header
# line. In a small file the invalid bytes of a later row are in that buffer, so the
# ``UnicodeDecodeError`` was raised while reading the header — outside the ``except``
# that translates it for the data rows — and escaped as a bare exception.
# ``validate_tables`` only catches ``TableLoadError``, so a caller that validated a
# table before reading its header crashed instead of recording the table as unreadable.


@pytest.mark.parametrize(
    "content",
    [
        b"Id,BIRTHDATE\n\xff\xfe,2000-01-01\n",  # invalid bytes in a row, same buffer
        b"Id,BIRTH\xffDATE\np1,2000-01-01\n",  # invalid bytes in the header itself
    ],
)
def test_invalid_utf8_near_the_header_is_a_load_error(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "patients.csv"
    path.write_bytes(content)
    with pytest.raises(TableLoadError, match="not valid UTF-8"):
        validate_structure(path)


def test_validate_tables_skips_a_table_that_is_not_utf8(tmp_path: Path) -> None:
    broken = tmp_path / "patients.csv"
    broken.write_bytes(b"Id,BIRTHDATE\n\xff\xfe,2000-01-01\n")
    clean = write(tmp_path / "encounters.csv", ENCOUNTERS)
    reports = validate_tables({"patients": broken, "encounters": clean})
    assert list(reports) == ["encounters"]
