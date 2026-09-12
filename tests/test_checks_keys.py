"""Tests for primary key and foreign key checks.

Covers the unit behaviour of both checks plus the runner that drives them against
a dataset directory, including the cases where no verdict must be invented.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from synthea_quality.checks.keys import (
    check_foreign_key,
    check_primary_key,
    run_key_checks,
)
from synthea_quality.models import Status
from synthea_quality.schema.keys import ForeignKeyRule, PrimaryKeyRule

PATIENT_PK = PrimaryKeyRule("patients", "Id")
CONDITION_PATIENT_FK = ForeignKeyRule("conditions", "PATIENT", "patients")


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


# --------------------------------------------------------------------------- #
# primary key
# --------------------------------------------------------------------------- #


def test_valid_primary_key_passes() -> None:
    result = check_primary_key(PATIENT_PK, pd.Series(["a", "b", "c"]))

    assert result.status is Status.PASS
    assert result.check_id == "pk.patients.Id"
    assert result.metrics["rows"] == 3
    assert result.metrics["nulls"] == 0
    assert result.metrics["duplicate_rows"] == 0
    assert result.samples == ()
    assert result.metadata["confirmed_by"]


def test_null_primary_key_fails() -> None:
    result = check_primary_key(PATIENT_PK, pd.Series(["a", None, "c"]))

    assert result.status is Status.FAIL
    assert result.metrics["nulls"] == 1
    assert result.metrics["duplicate_rows"] == 0
    assert "null value" in result.message
    # row numbers are data rows: index 1 is the second row after the header
    assert result.samples == ({"row": 2},)


def test_duplicated_primary_key_fails_and_distinguishes_the_two_defects() -> None:
    result = check_primary_key(PATIENT_PK, pd.Series(["a", "a", "b", None]))

    assert result.status is Status.FAIL
    assert result.metrics["duplicate_rows"] == 2
    assert result.metrics["duplicate_values"] == 1
    assert result.metrics["nulls"] == 1
    assert "null value" in result.message and "duplicated value" in result.message
    assert {"value": "a", "occurrences": 2} in result.samples
    assert {"row": 4} in result.samples


def test_primary_key_samples_are_limited() -> None:
    values = pd.Series([f"v{i}" for i in range(50)] * 2)

    result = check_primary_key(PATIENT_PK, values, sample_limit=3)

    assert result.metrics["duplicate_rows"] == 100
    assert result.metrics["duplicate_values"] == 50
    assert len(result.samples) == 3
    assert result.metadata["sample_limit"] == 3


# --------------------------------------------------------------------------- #
# foreign key
# --------------------------------------------------------------------------- #


def test_valid_foreign_key_passes() -> None:
    parents = frozenset({"p1", "p2"})

    result = check_foreign_key(CONDITION_PATIENT_FK, pd.Series(["p1", "p2", "p1"]), parents)

    assert result.status is Status.PASS
    assert result.check_id == "fk.conditions.PATIENT->patients.Id"
    assert result.metrics["references"] == 3
    assert result.metrics["valid"] == 3
    assert result.metrics["invalid"] == 0
    assert result.metrics["invalid_pct"] == 0.0


def test_many_rows_may_point_to_the_same_parent() -> None:
    parents = frozenset({"p1"})

    result = check_foreign_key(CONDITION_PATIENT_FK, pd.Series(["p1"] * 5), parents)

    assert result.status is Status.PASS
    assert result.metrics["references"] == 5


def test_orphan_foreign_key_fails_and_reports_counts_and_samples() -> None:
    parents = frozenset({"p1"})

    result = check_foreign_key(
        CONDITION_PATIENT_FK, pd.Series(["p1", "ghost", "ghost2", "p1"]), parents
    )

    assert result.status is Status.FAIL
    assert result.metrics["references"] == 4
    assert result.metrics["valid"] == 2
    assert result.metrics["invalid"] == 2
    assert result.metrics["invalid_pct"] == 50.0
    assert "2 of 4 reference(s) (50.0%)" in result.message
    assert result.samples == ({"value": "ghost", "row": 2}, {"value": "ghost2", "row": 3})


def test_orphan_samples_are_limited_even_when_the_count_is_large() -> None:
    parents = frozenset({"p1"})
    values = pd.Series(["ghost"] * 100 + ["p1"])

    result = check_foreign_key(CONDITION_PATIENT_FK, values, parents, sample_limit=2)

    assert result.metrics["invalid"] == 100
    assert len(result.samples) == 2


def test_null_references_are_allowed_and_not_counted_as_orphans() -> None:
    parents = frozenset({"p1"})

    result = check_foreign_key(CONDITION_PATIENT_FK, pd.Series(["p1", None, float("nan")]), parents)

    assert result.status is Status.PASS
    assert result.metrics["null_references"] == 2
    assert result.metrics["references"] == 1
    assert result.metrics["invalid"] == 0


def test_missing_values_are_exactly_what_the_loader_produces() -> None:
    """The loader maps only the empty field to a missing value, and a value keeps its text.

    So a reference of ``"NA"`` is a value to resolve, not an absent one: the two
    modules must agree, otherwise a real reference could be skipped as null.
    """
    parents = frozenset({"p1"})

    result = check_foreign_key(CONDITION_PATIENT_FK, pd.Series(["NA", "p1"]), parents)

    assert result.status is Status.FAIL
    assert result.metrics["null_references"] == 0
    assert result.metrics["invalid"] == 1
    assert result.samples == ({"value": "NA", "row": 1},)


def test_a_column_with_no_reference_at_all_is_not_applicable() -> None:
    """All null is legitimate for the optional columns; it is not a pass either."""
    result = check_foreign_key(CONDITION_PATIENT_FK, pd.Series([None, None]), frozenset({"p1"}))

    assert result.status is Status.NOT_APPLICABLE
    assert result.metrics["null_references"] == 2
    assert result.metrics["references"] == 0
    assert "no non-null value" in result.message


def test_missing_parent_key_skips_the_check_with_a_reason() -> None:
    result = check_foreign_key(
        CONDITION_PATIENT_FK,
        pd.Series(["p1"]),
        None,
        unavailable_reason="table 'patients' is not present in this dataset",
    )

    assert result.status is Status.SKIPPED
    assert "not present in this dataset" in result.message
    assert result.metrics == {}


# --------------------------------------------------------------------------- #
# the runner over a dataset directory
# --------------------------------------------------------------------------- #

PATIENTS_CSV = "Id\np1\np2\n"
ENCOUNTERS_CSV = "Id,PATIENT\ne1,p1\ne2,p2\n"
CONDITIONS_CSV = "PATIENT,ENCOUNTER\np1,e1\np2,e2\n"


def build(directory: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        write(directory, f"{name}.csv", text)
    return directory


def test_runner_passes_on_a_consistent_dataset(tmp_path: Path) -> None:
    build(tmp_path, {"patients": PATIENTS_CSV, "encounters": ENCOUNTERS_CSV, "conditions": CONDITIONS_CSV})

    results = run_key_checks(tmp_path)
    by_id = {result.check_id: result for result in results}

    assert by_id["pk.patients.Id"].status is Status.PASS
    assert by_id["pk.encounters.Id"].status is Status.PASS
    assert by_id["fk.conditions.PATIENT->patients.Id"].status is Status.PASS
    assert by_id["fk.conditions.ENCOUNTER->encounters.Id"].status is Status.PASS
    assert not [r for r in results if r.status is Status.FAIL]


def test_runner_reports_an_orphan_foreign_key(tmp_path: Path) -> None:
    build(
        tmp_path,
        {
            "patients": PATIENTS_CSV,
            "encounters": ENCOUNTERS_CSV,
            "conditions": "PATIENT,ENCOUNTER\np1,e1\nghost,e2\n",
        },
    )

    by_id = {result.check_id: result for result in run_key_checks(tmp_path)}
    orphans = by_id["fk.conditions.PATIENT->patients.Id"]

    assert orphans.status is Status.FAIL
    assert orphans.metrics["invalid"] == 1
    assert orphans.samples == ({"value": "ghost", "row": 2},)


def test_runner_skips_rules_of_a_missing_child_table(tmp_path: Path) -> None:
    build(tmp_path, {"patients": PATIENTS_CSV})

    by_id = {result.check_id: result for result in run_key_checks(tmp_path)}

    conditions = by_id["fk.conditions.PATIENT->patients.Id"]
    assert conditions.status is Status.SKIPPED
    assert "table 'conditions' is not present" in conditions.message
    assert by_id["pk.patients.Id"].status is Status.PASS
    assert by_id["pk.claims.Id"].status is Status.SKIPPED


def test_runner_skips_rules_of_a_missing_parent_table(tmp_path: Path) -> None:
    build(tmp_path, {"patients": PATIENTS_CSV, "encounters": "Id,PATIENT\ne1,p1\n", "conditions": "PATIENT,ENCOUNTER\np1,e1\n"})
    (tmp_path / "encounters.csv").unlink()

    by_id = {result.check_id: result for result in run_key_checks(tmp_path)}

    assert by_id["pk.patients.Id"].status is Status.PASS
    skipped = by_id["fk.conditions.ENCOUNTER->encounters.Id"]
    assert skipped.status is Status.SKIPPED
    assert "table 'encounters' is not present" in skipped.message


def test_runner_skips_when_a_rule_column_is_not_in_the_dataset_schema(tmp_path: Path) -> None:
    """An older dataset without the column cannot be judged by this rule."""
    build(
        tmp_path,
        {
            "patients": PATIENTS_CSV,
            "encounters": ENCOUNTERS_CSV,
            "conditions": "PATIENT\np1\n",
        },
    )

    by_id = {result.check_id: result for result in run_key_checks(tmp_path)}

    assert by_id["fk.conditions.PATIENT->patients.Id"].status is Status.PASS
    assert by_id["fk.conditions.ENCOUNTER->encounters.Id"].status is Status.SKIPPED


def test_runner_skips_a_table_that_cannot_be_read(tmp_path: Path) -> None:
    build(tmp_path, {"patients": PATIENTS_CSV, "encounters": ENCOUNTERS_CSV, "conditions": CONDITIONS_CSV})
    write(tmp_path, "conditions.csv", "")

    by_id = {result.check_id: result for result in run_key_checks(tmp_path)}
    result = by_id["fk.conditions.PATIENT->patients.Id"]

    assert result.status is Status.SKIPPED
    assert "could not be read" in result.message
    assert "empty" in result.message


def test_runner_does_not_treat_repeated_imaging_study_ids_as_a_defect(tmp_path: Path) -> None:
    """imaging_studies.Id repeats by design: no primary key rule may fire."""
    build(
        tmp_path,
        {
            "patients": PATIENTS_CSV,
            "encounters": ENCOUNTERS_CSV,
            "imaging_studies": "Id,PATIENT,ENCOUNTER\ns1,p1,e1\ns1,p1,e1\ns1,p2,e2\n",
        },
    )

    results = run_key_checks(tmp_path)

    assert not [result for result in results if "imaging_studies" in result.check_id and result.status is Status.FAIL]
    assert not [result for result in results if result.check_id.startswith("pk.imaging_studies")]
    by_id = {result.check_id: result for result in results}
    assert by_id["fk.imaging_studies.PATIENT->patients.Id"].status is Status.PASS


def test_runner_results_are_deterministic_and_sorted(tmp_path: Path) -> None:
    build(tmp_path, {"patients": PATIENTS_CSV, "encounters": ENCOUNTERS_CSV, "conditions": CONDITIONS_CSV})

    first = [result.check_id for result in run_key_checks(tmp_path)]
    second = [result.check_id for result in run_key_checks(tmp_path)]

    assert first == second == sorted(first)


def test_runner_holds_only_compact_parent_keys(tmp_path: Path, monkeypatch) -> None:
    """Child frames must not be cached: memory is bounded by the parent key sets."""
    import synthea_quality.checks.keys as module

    build(tmp_path, {"patients": PATIENTS_CSV, "encounters": ENCOUNTERS_CSV, "conditions": CONDITIONS_CSV})
    loaded: list[tuple[str, ...]] = []
    original = module.load_table

    def tracking(file_path, *, columns=None, **kwargs):
        loaded.append(tuple(columns) if columns is not None else ())
        return original(file_path, columns=columns, **kwargs)

    monkeypatch.setattr(module, "load_table", tracking)
    results = run_key_checks(tmp_path)

    assert results  # sanity: the run produced results
    # conditions is a child only: loaded once, for exactly the columns its rules need
    conditions_loads = [columns for columns in loaded if set(columns) == {"PATIENT", "ENCOUNTER"}]
    assert len(conditions_loads) == 1
    # parent keys are read one column at a time, never as whole tables
    assert ("Id",) in loaded
