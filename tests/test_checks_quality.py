"""Behaviour of the data quality checks, plus the runner that drives them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from synthea_quality.checks.quality import (
    check_date_values,
    check_duplicate_rows,
    check_empty_columns,
    check_nulls,
    run_quality_checks,
)
from synthea_quality.models import Severity, Status
from synthea_quality.schema.quality import DATE_FORMAT, TIMESTAMP_FORMAT, DateColumn

DATE_RULE = DateColumn("patients", "BIRTHDATE", DATE_FORMAT)
TIMESTAMP_RULE = DateColumn("encounters", "START", TIMESTAMP_FORMAT)

CLEAN = pd.DataFrame(
    {
        "Id": ["p1", "p2", "p3"],
        "BIRTHDATE": ["1980-01-01", "1990-02-02", "2000-03-03"],
        # some patients have no death date, but the column is not empty
        "DEATHDATE": [None, "2021-06-06", None],
    }
)


# --------------------------------------------------------------------------- #
# a clean table
# --------------------------------------------------------------------------- #


def test_clean_table_passes_every_quality_check() -> None:
    assert check_duplicate_rows("patients", CLEAN).status is Status.PASS
    assert check_empty_columns("patients", CLEAN).status is Status.PASS
    nulls = check_nulls("patients", CLEAN)

    assert nulls.status is Status.PASS
    assert nulls.severity.value == "LOW"


def test_clean_table_summarises_its_nulls() -> None:
    result = check_nulls("patients", CLEAN)

    assert result.metrics["rows"] == 3
    assert result.metrics["columns"] == 3
    assert result.metrics["columns_with_nulls"] == 1
    assert result.metrics["null_cells"] == 2
    assert result.metrics["most_null_columns"] == [
        {"column": "DEATHDATE", "nulls": 2, "null_pct": 66.6667}
    ]


# --------------------------------------------------------------------------- #
# duplicate rows
# --------------------------------------------------------------------------- #


def test_duplicate_row_is_a_warning_with_a_sample() -> None:
    frame = pd.DataFrame({"Id": ["p1", "p2", "p1"], "CODE": ["a", "b", "a"]})

    result = check_duplicate_rows("conditions", frame)

    assert result.status is Status.WARNING
    assert result.metrics["duplicated_rows"] == 2
    assert result.metrics["duplicate_groups"] == 1
    assert result.metrics["redundant_rows"] == 1
    assert result.samples[0]["row"] == 1
    assert result.samples[0]["values"] == {"Id": "p1", "CODE": "a"}


def test_duplicate_rows_are_never_a_failure_because_synthea_repeats_rows() -> None:
    """36 observations and 22 supplies rows repeat exactly in the official sample."""
    frame = pd.DataFrame({"DATE": ["2020-01-01"] * 4})

    result = check_duplicate_rows("observations", frame)

    assert result.status is Status.WARNING
    assert result.severity.value == "MEDIUM"


def test_duplicate_rows_on_a_keyed_table_point_at_the_key_check() -> None:
    frame = pd.DataFrame({"Id": ["p1", "p1"]})

    result = check_duplicate_rows("patients", frame)

    assert result.status is Status.WARNING
    assert result.metadata["has_primary_key"] is True
    assert "primary key Id" in result.message
    assert "already reports this as a failure" in result.message


def test_duplicate_row_samples_are_limited() -> None:
    frame = pd.DataFrame({"CODE": [f"c{i}" for i in range(10)] * 2})

    result = check_duplicate_rows("supplies", frame, sample_limit=2)

    assert result.metrics["duplicate_groups"] == 10
    assert len(result.samples) == 2


# --------------------------------------------------------------------------- #
# empty columns and nulls
# --------------------------------------------------------------------------- #


def test_fully_empty_column_is_a_warning() -> None:
    frame = pd.DataFrame({"Id": ["p1"], "MODIFIER1": [None], "MODIFIER2": [None]})

    result = check_empty_columns("claims_transactions", frame, sample_limit=1)

    assert result.status is Status.WARNING
    assert result.metrics["empty_columns"] == 2
    assert result.metrics["columns"] == 3
    assert result.samples == ({"column": "MODIFIER1"},)


def test_optional_column_with_legitimate_nulls_is_not_a_warning() -> None:
    """DEATHDATE is null for 99 of 108 patients in the official sample."""
    result = check_empty_columns("patients", CLEAN)

    assert result.status is Status.PASS
    assert result.metrics["empty_columns"] == 0


def test_a_column_that_is_empty_for_every_row_is_not_a_failure() -> None:
    """A run with no deaths leaves DEATHDATE empty: worth reporting, not an error."""
    frame = pd.DataFrame({"Id": ["p1", "p2"], "DEATHDATE": [None, None]})

    result = check_empty_columns("patients", frame)

    assert result.status is Status.WARNING
    # LOW: whether an empty column matters depends on the column's purpose, which
    # the data alone cannot settle, so no stronger impact can be justified.
    assert result.severity is Severity.LOW
    assert result.samples == ({"column": "DEATHDATE"},)


def test_severity_reflects_impact_and_not_merely_determinism() -> None:
    """A malformed date is deterministic, but its impact is a single row.

    HIGH is reserved for breaches of the relational contract (duplicate or null
    primary keys, orphan foreign keys), which break joins for the whole dataset.
    """
    broken_date = check_date_values(DATE_RULE, pd.Series(["31/12/1980"]))
    duplicate_key = check_duplicate_rows("patients", pd.DataFrame({"Id": ["p1", "p1"]}))

    assert broken_date.status is Status.FAIL
    assert broken_date.severity is Severity.MEDIUM
    assert broken_date.severity is not Severity.HIGH
    assert duplicate_key.severity is Severity.MEDIUM


def test_nulls_are_counted_and_never_treated_as_errors() -> None:
    frame = pd.DataFrame({"ENCOUNTER": [None, "e1", None], "UNITS": [None, None, None]})

    result = check_nulls("observations", frame)

    assert result.status is Status.PASS
    assert result.metrics["null_cells"] == 5
    assert result.metrics["columns_with_nulls"] == 2
    assert "not treated as errors" in result.message
    assert result.metadata["assumption"].startswith("an empty field")


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #


def test_valid_dates_pass() -> None:
    values = pd.Series(["1980-01-01", "1990-02-02", None])

    result = check_date_values(DATE_RULE, values)

    assert result.status is Status.PASS
    assert result.metrics["valid"] == 2
    assert result.metrics["invalid"] == 0
    assert result.metrics["nulls"] == 1


def test_malformed_date_fails_with_a_bounded_sample() -> None:
    values = pd.Series(["1980-01-01", "01/02/1990", "1980-1-1", "not a date", "1990-02-30T00:00:00Z"])

    result = check_date_values(DATE_RULE, values, sample_limit=2)

    assert result.status is Status.FAIL
    assert result.metrics["invalid"] == 4
    assert result.metrics["invalid_pct"] == 80.0
    assert len(result.samples) == 2
    assert result.samples[0] == {"value": "01/02/1990", "row": 2}
    # the message states the format that was expected, in readable terms
    assert "the YYYY-MM-DD date format" in result.message


def test_date_column_with_no_value_is_not_applicable() -> None:
    result = check_date_values(DATE_RULE, pd.Series([None, None]))

    assert result.status is Status.NOT_APPLICABLE
    assert result.metrics["nulls"] == 2
    assert "no non-null value" in result.message


def test_timestamp_column_rejects_a_date_only_value() -> None:
    """Each column documents one shape; the other shape is reported, not coerced."""
    rule = TIMESTAMP_RULE

    assert check_date_values(rule, pd.Series(["2020-01-01T10:00:00Z"])).status is Status.PASS
    reported = check_date_values(rule, pd.Series(["2020-01-01"]))

    assert reported.status is Status.FAIL
    assert reported.metadata["accepted_format"] == "iso8601-utc"


def test_date_check_records_the_documentation_it_relies_on() -> None:
    result = check_date_values(DATE_RULE, pd.Series(["1980-01-01"]))

    assert result.metadata["documented_as"].startswith("CSV File Data Dictionary")
    assert result.metadata["accepted_format"] == DATE_FORMAT.identifier


# --------------------------------------------------------------------------- #
# empty tables
# --------------------------------------------------------------------------- #


def test_checks_of_an_empty_table_are_not_applicable() -> None:
    empty = pd.DataFrame({"Id": pd.Series([], dtype="str"), "FIRST": pd.Series([], dtype="str")})

    for result in (
        check_duplicate_rows("patients", empty),
        check_empty_columns("patients", empty),
        check_nulls("patients", empty),
    ):
        assert result.status is Status.NOT_APPLICABLE


# --------------------------------------------------------------------------- #
# the runner over a dataset directory
# --------------------------------------------------------------------------- #


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_runner_reports_a_clean_dataset_as_clean(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,BIRTHDATE,DEATHDATE\np1,1980-01-01,\np2,1990-02-02,2000-01-01\n")

    by_id = {result.check_id: result for result in run_quality_checks(tmp_path)}

    assert by_id["duplicates.patients"].status is Status.PASS
    assert by_id["empty_columns.patients"].status is Status.PASS
    assert by_id["nulls.patients"].status is Status.PASS
    assert by_id["dates.patients.BIRTHDATE"].status is Status.PASS
    assert by_id["dates.patients.DEATHDATE"].status is Status.PASS


def test_runner_gives_a_verdict_per_table_of_a_partly_broken_dataset(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,BIRTHDATE,DEATHDATE\np1,1980-01-01,\np2,31/12/1990,2020-01-01\n")
    write(tmp_path, "conditions.csv", "START,STOP,PATIENT\n2020-01-01,,p1\n2020-01-01,,p1\n")
    write(tmp_path, "supplies.csv", "DATE,PATIENT,CODE\n2021-05-05,p1,abc\n")

    by_id = {result.check_id: result for result in run_quality_checks(tmp_path)}

    # patients: a broken date is the only failure
    assert by_id["dates.patients.BIRTHDATE"].status is Status.FAIL
    assert by_id["dates.patients.DEATHDATE"].status is Status.PASS
    # a date column with no value at all cannot be judged, and is not called clean
    assert by_id["dates.patients.DEATHDATE"].metrics["values"] == 1
    # conditions: no dates problem, but a repeated row
    assert by_id["dates.conditions.START"].status is Status.PASS
    assert by_id["duplicates.conditions"].status is Status.WARNING
    # supplies: clean
    assert by_id["dates.supplies.DATE"].status is Status.PASS
    assert by_id["duplicates.supplies"].status is Status.PASS
    # a table nobody touched must not be reported as clean
    assert by_id["nulls.medications"].status is Status.SKIPPED


def test_runner_does_not_call_an_uncheckable_date_clean(tmp_path: Path) -> None:
    """No value to check is NOT_APPLICABLE, never PASS: nothing was validated."""
    write(tmp_path, "patients.csv", "Id,BIRTHDATE,DEATHDATE\np1,1980-01-01,\n")

    by_id = {result.check_id: result for result in run_quality_checks(tmp_path)}

    assert by_id["dates.patients.BIRTHDATE"].status is Status.PASS
    assert by_id["dates.patients.DEATHDATE"].status is Status.NOT_APPLICABLE


def test_runner_skips_checks_of_a_missing_table(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,BIRTHDATE\np1,1980-01-01\n")

    by_id = {result.check_id: result for result in run_quality_checks(tmp_path)}

    skipped = by_id["duplicates.conditions"]
    assert skipped.status is Status.SKIPPED
    assert "not present in this dataset" in skipped.message
    assert by_id["dates.conditions.START"].status is Status.SKIPPED


def test_runner_skips_a_date_rule_whose_column_is_absent(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,FIRST\np1,Ana\n")

    by_id = {result.check_id: result for result in run_quality_checks(tmp_path)}

    assert by_id["nulls.patients"].status is Status.PASS
    assert by_id["dates.patients.BIRTHDATE"].status is Status.SKIPPED
    assert "has no column 'BIRTHDATE'" in by_id["dates.patients.BIRTHDATE"].message


def test_runner_reports_an_unreadable_table_without_inventing_a_verdict(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,BIRTHDATE\np1,1980-01-01\n")
    write(tmp_path, "conditions.csv", "")

    by_id = {result.check_id: result for result in run_quality_checks(tmp_path)}
    result = by_id["duplicates.conditions"]

    assert result.status is Status.SKIPPED
    assert "could not be read" in result.message


def test_runner_covers_every_table_check_and_date_rule(tmp_path: Path) -> None:
    from synthea_quality.schema.quality import DATE_COLUMNS

    write(tmp_path, "patients.csv", "Id,BIRTHDATE,DEATHDATE\np1,1980-01-01,\n")

    results = run_quality_checks(tmp_path)
    expected = 3 * 19 + len(DATE_COLUMNS)

    assert len(results) == expected
    assert len({result.check_id for result in results}) == expected


def test_runner_results_are_deterministic_and_sorted(tmp_path: Path) -> None:
    write(tmp_path, "patients.csv", "Id,BIRTHDATE\np1,1980-01-01\n")

    first = [result.check_id for result in run_quality_checks(tmp_path)]
    second = [result.check_id for result in run_quality_checks(tmp_path)]

    assert first == second == sorted(first)


def test_runner_does_not_load_two_tables_at_once(tmp_path: Path, monkeypatch) -> None:
    """Tables are processed one at a time; nothing keeps a previous table alive."""
    import synthea_quality.checks.quality as module

    write(tmp_path, "patients.csv", "Id,BIRTHDATE\np1,1980-01-01\n")
    write(tmp_path, "supplies.csv", "DATE,PATIENT\n2021-05-05,p1\n")
    live: list[int] = []
    original = module.load_table

    def tracking(file_path, **kwargs):
        loaded = original(file_path, **kwargs)
        live.append(id(loaded.frame))
        return loaded

    monkeypatch.setattr(module, "load_table", tracking)
    results = run_quality_checks(tmp_path)

    assert results
    assert len(live) == 2  # one load per present table, none repeated
    assert len(set(live)) == 2


def test_quality_checks_do_not_duplicate_the_key_checks(tmp_path: Path) -> None:
    """Row uniqueness belongs to the key checks; quality must not report it again."""
    write(tmp_path, "patients.csv", "Id,BIRTHDATE\np1,1980-01-01\np1,1990-01-01\n")

    results = run_quality_checks(tmp_path)

    assert all(not result.check_id.startswith(("pk.", "fk.")) for result in results)
