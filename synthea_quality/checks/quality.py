"""[CHECKS] Data quality: duplicate rows, empty columns, nulls and date shapes.

This module deliberately keeps two kinds of finding apart:

**Deterministic violations** (``FAIL``) — a value that contradicts the confirmed
schema. Today that means one thing: a column documented as holding a date contains
something that is neither of the two shapes Synthea writes. The value cannot be
interpreted, so every later analysis of that column is wrong.

**Informative metrics** (``PASS`` or ``WARNING``) — facts about the data that a
human should weigh, with no assumption that they are errors:

* ``nulls.<table>`` (``PASS``) counts nulls per column and never treats a null as a
  defect. The dictionary marks many of these columns optional, and the official
  sample proves nulls are normal: 99 of 108 patients have no death date, 3,060 of
  68,648 observations have no encounter.
* ``empty_columns.<table>`` (``WARNING``) flags a column the file contains but never
  fills. That is often by design: the dictionary calls
  ``claims_transactions.MODIFIER1``/``MODIFIER2`` "Unused", and ``allergies.STOP``,
  ``claims.REFERRINGPROVIDERID``, ``payers.ADDRESS`` and the payer address block are
  empty in the official sample too.
* ``duplicates.<table>`` (``WARNING``) flags rows that repeat an earlier row exactly.
  Measured on the official sample, this is normal for tables without a key
  (36 rows in ``observations``, 22 in ``supplies``) — a repeated measurement is
  plausible — while no keyed table has any. For a table with a documented primary
  key the key checks already report the duplicate as ``FAIL``, so this check stays
  informative and says so instead of duplicating the verdict.

No clinical rule is invented here: nothing judges whether a code, a value or a
diagnosis is medically plausible.

Memory
------
These checks need every column of a table, so unlike the key checks they load whole
tables. They still process one table at a time and release it before reading the
next, which is why a dataset is never held in memory at once. The cost of a single
table is measured: see the Scalability section of the README.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import TableLoadError
from synthea_quality.loader import load_table
from synthea_quality.models import DEFAULT_SAMPLE_LIMIT, CheckResult, Severity, Status
from synthea_quality.schema.keys import primary_key_for
from synthea_quality.schema.quality import DateColumn, date_columns_for
from synthea_quality.schema.tables import SYNTHEA_TABLES

#: A date that cannot be interpreted is a row-level defect: it makes that row's
#: value unusable, but it breaks no join and no whole table, so it is not ``HIGH``.
#: ``HIGH`` stays reserved for violations of the relational contract (see models).
DATE_SEVERITY = Severity.MEDIUM
#: Repeated rows are row-level anomalies, not broken joins.
ROW_SEVERITY = Severity.MEDIUM
#: Null counts are recorded, not judged.
NULL_SEVERITY = Severity.LOW
#: An empty column is informative: whether it matters depends on its purpose, which
#: the data alone cannot settle.
EMPTY_COLUMN_SEVERITY = Severity.LOW


def check_duplicate_rows(
    table: str,
    frame: pd.DataFrame,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Report rows that repeat an earlier row exactly.

    ``WARNING`` rather than ``FAIL``: no table in the dictionary is declared unique
    on its whole row, so an exact repeat is a strong hint, not a contract breach.
    """
    rows = int(len(frame))
    check_id = f"duplicates.{table}"

    if rows == 0:
        return _not_applicable(
            check_id, table, f"table {table} has no row, so no row can be duplicated"
        )

    duplicated_mask = frame.duplicated(keep=False)
    duplicated_rows = int(duplicated_mask.sum())

    metrics: dict[str, object] = {
        "rows": rows,
        "duplicated_rows": duplicated_rows,
        "duplicate_groups": 0,
        "redundant_rows": 0,
        "sample_limit": sample_limit,
    }

    if duplicated_rows == 0:
        return CheckResult(
            check_id=check_id,
            status=Status.PASS,
            severity=ROW_SEVERITY,
            message=f"no row of {table} repeats another row exactly ({rows} row(s) checked)",
            table=table,
            metrics=metrics,
            metadata={"rule": "exact duplicate row"},
        )

    representatives = frame[duplicated_mask].drop_duplicates()
    duplicate_groups = int(len(representatives))
    redundant_rows = duplicated_rows - duplicate_groups
    metrics.update({"duplicate_groups": duplicate_groups, "redundant_rows": redundant_rows})

    samples = tuple(
        {"row": int(index) + 1, "values": {str(k): _text(v) for k, v in row.items()}}
        for index, row in representatives.head(sample_limit).iterrows()
    )

    key = primary_key_for(table)
    clarification = (
        f"; {table} has primary key {key.column}, so its key check already reports this "
        "as a failure"
        if key is not None
        else ""
    )
    return CheckResult(
        check_id=check_id,
        status=Status.WARNING,
        severity=ROW_SEVERITY,
        message=(
            f"{duplicated_rows} row(s) of {table} belong to {duplicate_groups} repeated "
            f"row value(s) ({redundant_rows} redundant row(s)){clarification}"
        ),
        table=table,
        metrics=metrics,
        samples=samples,
        metadata={"rule": "exact duplicate row", "has_primary_key": key is not None},
    )


def check_empty_columns(
    table: str,
    frame: pd.DataFrame,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Report columns the file contains but never fills.

    ``WARNING``: an entirely empty column is never a relational breach, and several
    such columns are expected (see the module docstring).
    """
    rows = int(len(frame))
    check_id = f"empty_columns.{table}"

    if rows == 0:
        return _not_applicable(
            check_id, table, f"table {table} has no row, so its columns cannot be judged"
        )

    empty = [str(column) for column in frame.columns if bool(frame[column].isna().all())]
    metrics = {"rows": rows, "columns": int(len(frame.columns)), "empty_columns": len(empty)}

    if not empty:
        return CheckResult(
            check_id=check_id,
            status=Status.PASS,
            severity=EMPTY_COLUMN_SEVERITY,
            message=f"every column of {table} holds at least one value",
            table=table,
            metrics=metrics,
            metadata={"rule": "column with no value at all"},
        )

    return CheckResult(
        check_id=check_id,
        status=Status.WARNING,
        severity=EMPTY_COLUMN_SEVERITY,
        message=(
            f"{len(empty)} column(s) of {table} hold no value at all: {empty[:sample_limit]}"
            f"{' ...' if len(empty) > sample_limit else ''}"
        ),
        table=table,
        metrics=metrics,
        samples=tuple({"column": name} for name in empty[:sample_limit]),
        metadata={"rule": "column with no value at all", "sample_limit": sample_limit},
    )


def check_nulls(
    table: str,
    frame: pd.DataFrame,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Record how many nulls each column has, without judging them.

    Always ``PASS`` unless there is nothing to measure: the check exists to inform,
    and the dictionary marks many columns optional, so a null is not an error here.
    """
    rows = int(len(frame))
    check_id = f"nulls.{table}"

    if rows == 0:
        return _not_applicable(
            check_id, table, f"table {table} has no row, so it has no null to count"
        )

    counts = frame.isna().sum()
    columns_with_nulls = int((counts > 0).sum())
    null_cells = int(counts.sum())

    most_null = [
        {
            "column": str(column),
            "nulls": int(count),
            "null_pct": round(100.0 * int(count) / rows, 4),
        }
        for column, count in counts.sort_values(ascending=False).head(sample_limit).items()
        if int(count) > 0
    ]

    return CheckResult(
        check_id=check_id,
        status=Status.PASS,
        severity=NULL_SEVERITY,
        message=(
            f"informational: {null_cells} null cell(s) in {columns_with_nulls} of "
            f"{len(frame.columns)} column(s) of {table}; nulls are not treated as errors"
        ),
        table=table,
        metrics={
            "rows": rows,
            "columns": int(len(frame.columns)),
            "null_cells": null_cells,
            "columns_with_nulls": columns_with_nulls,
            "most_null_columns": most_null,
            "sample_limit": sample_limit,
        },
        metadata={
            "rule": "nulls are recorded, never assumed to be errors",
            "assumption": "an empty field is the only missing value (see the loader)",
        },
    )


def check_date_values(
    rule: DateColumn,
    values: pd.Series,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Check that every non-null value of a date column has the confirmed shape.

    Nulls are skipped: a date the generator chose not to write is not a malformed
    date. Anything else that does not match the documented format is a ``FAIL``,
    because no later interpretation of that value can be trusted.
    """
    target = values.dropna()
    total = int(len(target))
    metadata = {
        "rule": rule.check_id,
        "accepted_format": rule.format.identifier,
        "documented_as": rule.documented_as,
        "sample_limit": sample_limit,
    }

    if total == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.NOT_APPLICABLE,
            severity=DATE_SEVERITY,
            message=(
                f"{rule.table}.{rule.column} has no non-null value to check as a date"
            ),
            table=rule.table,
            metrics={"rows": int(len(values)), "nulls": int(len(values) - total), "values": 0},
            metadata=metadata,
        )

    text = target.astype(str)
    matches = text.str.fullmatch(rule.format.pattern).fillna(False)
    invalid = int((~matches).sum())
    valid = total - invalid
    invalid_pct = round(100.0 * invalid / total, 6)

    metrics = {
        "rows": int(len(values)),
        "values": total,
        "nulls": int(len(values) - total),
        "valid": valid,
        "invalid": invalid,
        "invalid_pct": invalid_pct,
    }

    if invalid == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.PASS,
            severity=DATE_SEVERITY,
            message=(
                f"all {total} value(s) of {rule.table}.{rule.column} match {rule.format.label}"
            ),
            table=rule.table,
            metrics=metrics,
            metadata=metadata,
        )

    offenders = text[~matches]
    samples = tuple(
        {"value": str(value), "row": int(index) + 1}
        for index, value in offenders.head(sample_limit).items()
    )
    return CheckResult(
        check_id=rule.check_id,
        status=Status.FAIL,
        severity=DATE_SEVERITY,
        message=(
            f"{invalid} of {total} value(s) ({invalid_pct}%) in {rule.table}.{rule.column} "
            f"do not match {rule.format.label}"
        ),
        table=rule.table,
        metrics=metrics,
        samples=samples,
        metadata=metadata,
    )


def run_quality_checks(
    data_dir: str | Path,
    *,
    discovery: DiscoveryResult | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> tuple[CheckResult, ...]:
    """Run the data quality checks over the dataset in ``data_dir``.

    One table is loaded at a time and released before the next is read. Results are
    sorted by ``check_id`` so two runs over the same data produce the same report.
    """
    found = discovery if discovery is not None else discover_dataset(data_dir)
    paths = {table.name: table.path for table in found.tables}
    results: list[CheckResult] = []

    for table in sorted(paths):
        try:
            frame = load_table(paths[table]).frame
        except TableLoadError as exc:
            results.extend(
                _skipped(check_id, table, f"table '{table}' could not be read: {exc}")
                for check_id in _table_check_ids(table)
            )
            results.extend(
                _skipped(rule.check_id, table, f"table '{table}' could not be read: {exc}")
                for rule in date_columns_for(table)
            )
            continue

        columns = tuple(str(column) for column in frame.columns)
        results.append(check_nulls(table, frame, sample_limit=sample_limit))
        results.append(check_empty_columns(table, frame, sample_limit=sample_limit))
        results.append(check_duplicate_rows(table, frame, sample_limit=sample_limit))

        for rule in date_columns_for(table):
            if rule.column not in columns:
                results.append(
                    _skipped(
                        rule.check_id,
                        table,
                        f"table '{table}' has no column '{rule.column}' in this dataset",
                    )
                )
                continue
            results.append(
                check_date_values(
                    rule, cast(pd.Series, frame[rule.column]), sample_limit=sample_limit
                )
            )

        # Release the table before loading the next one.
        del frame

    for spec in SYNTHEA_TABLES:
        if spec.name in paths:
            continue
        reason = f"table '{spec.name}' is not present in this dataset, so it cannot be checked"
        results.extend(
            _skipped(check_id, spec.name, reason) for check_id in _table_check_ids(spec.name)
        )
        results.extend(
            _skipped(rule.check_id, spec.name, reason)
            for rule in date_columns_for(spec.name)
        )

    return tuple(sorted(results, key=lambda result: result.check_id))


def _table_check_ids(table: str) -> tuple[str, ...]:
    """The per-table quality checks, so a missing table reports them consistently."""
    return (f"duplicates.{table}", f"empty_columns.{table}", f"nulls.{table}")


def _not_applicable(check_id: str, table: str, message: str) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        status=Status.NOT_APPLICABLE,
        severity=ROW_SEVERITY,
        message=message,
        table=table,
    )


def _skipped(check_id: str, table: str, reason: str) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        status=Status.SKIPPED,
        severity=ROW_SEVERITY,
        message=f"check not run for {check_id}: {reason}",
        table=table,
    )


def _text(value: object) -> str | None:
    """Render a cell for a sample without turning a null into the text 'nan'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
