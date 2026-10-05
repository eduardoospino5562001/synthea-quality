"""[CHECKS] Primary key uniqueness and foreign key referential integrity.

Both checks work on data that has already been read; they never touch the file
system. The rules they apply come from :mod:`synthea_quality.schema.keys`, which
derives them from Synthea's own data dictionary, so a check can only ever report
a relationship the project has confirmed.

Statuses
--------
``PASS``            every value is unique and non-null (PK), or every non-null
                    reference resolves (FK).
``FAIL``            duplicate or null primary key, or an orphan reference. Both are
                    deterministic violations of the relational contract.
``NOT_APPLICABLE``  there is nothing to validate: no non-null value at all in the
                    column (all references are null, which is legitimate for the
                    optional columns the dictionary marks ``required: false``).
``SKIPPED``         the rule could not be applied to this dataset, with the reason
                    in the message: the child or parent table is absent, the column
                    is not part of this dataset's schema, or a file could not be
                    read. No verdict is invented for those cases.

Row numbers in samples are data rows: 1 is the first row after the header, so it
corresponds to line 2 of the file.

Memory
------
The key checks load the parent keys once and reduce them to compact sets, then
process one child table at a time and release it; child frames are never cached.
That is deliberate, and the reason is measured: loading a real 1.78 GB
``observations.csv`` (10,209,651 rows x 9 columns) peaked at ~1.64 GB of RSS in
24.7 s (see the slice-5 measurement). Chunking is therefore **not needed for this
MVP**, which targets sample-sized datasets, but the scalability limit is
demonstrated rather than theoretical: a dataset whose large tables are loaded whole
in memory, or a table larger than that sample, will need chunked/streamed processing
before this tool can be called scalable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import TableLoadError
from synthea_quality.loader import load_table, read_header
from synthea_quality.models import DEFAULT_SAMPLE_LIMIT, CheckResult, Severity, Status
from synthea_quality.schema.keys import (
    FOREIGN_KEYS,
    PRIMARY_KEYS,
    ForeignKeyRule,
    PrimaryKeyRule,
    parent_columns,
)
from synthea_quality.structure import (
    StructureReport,
    gate_reason,
    validate_tables,
)

#: A broken relational contract affects joins and every downstream metric.
SEVERITY = Severity.HIGH


def check_primary_key(
    rule: PrimaryKeyRule,
    values: pd.Series,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Check that ``values`` is a valid primary key: non-null and unique.

    Nulls and duplicates are reported separately, because they are different
    defects with different causes.
    """
    rows = int(len(values))
    null_mask = values.isna()
    nulls = int(null_mask.sum())
    non_null = values[~null_mask]

    duplicate_mask = non_null.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_values = int(non_null[duplicate_mask].nunique())

    metrics: dict[str, object] = {
        "rows": rows,
        "nulls": nulls,
        "distinct_values": int(non_null.nunique()),
        "duplicate_rows": duplicate_rows,
        "duplicate_values": duplicate_values,
    }
    metadata = _metadata(rule, sample_limit)

    problems: list[str] = []
    # Nulls and duplicates are different defects, so each keeps its own samples: a table
    # with many null rows used to fill the whole budget and leave the duplicated values
    # invisible in the report, half of the finding with no evidence behind it.
    null_samples: list[Mapping[str, object]] = []
    duplicate_samples: list[Mapping[str, object]] = []

    if nulls:
        problems.append(f"{nulls} null value(s)")
        null_rows = [int(index) + 1 for index in values.index[null_mask][:sample_limit]]
        null_samples = [{"row": row} for row in null_rows]

    if duplicate_rows:
        problems.append(
            f"{duplicate_values} duplicated value(s) covering {duplicate_rows} row(s)"
        )
        counts = non_null[duplicate_mask].value_counts()
        duplicate_samples = [
            {"value": str(value), "occurrences": int(count)}
            for value, count in counts.head(sample_limit).items()
        ]

    if problems:
        message = (
            f"{' and '.join(problems)} in primary key {rule.table}.{rule.column}"
            f" (expected every value to be unique and non-null)"
        )
        return CheckResult(
            check_id=rule.check_id,
            status=Status.FAIL,
            severity=SEVERITY,
            message=message,
            table=rule.table,
            metrics=metrics,
            samples=tuple((*null_samples, *duplicate_samples)),
            metadata=metadata,
        )

    return CheckResult(
        check_id=rule.check_id,
        status=Status.PASS,
        severity=SEVERITY,
        message=(
            f"all {rows} value(s) of {rule.table}.{rule.column} are unique and non-null"
        ),
        table=rule.table,
        metrics=metrics,
        metadata=metadata,
    )


def check_foreign_key(
    rule: ForeignKeyRule,
    values: pd.Series,
    parent_values: frozenset[str] | None,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    unavailable_reason: str | None = None,
) -> CheckResult:
    """Check that every non-null reference in ``values`` exists in ``parent_values``.

    Null references are counted but never treated as orphans: whether a null is
    acceptable depends on the column, and the dictionary marks several of these
    columns optional (the official sample has 3,060 null ``observations.ENCOUNTER``
    among others).

    ``parent_values=None`` means the parent key could not be built; the reason must
    be given in ``unavailable_reason`` and the result is ``SKIPPED``, never a guess.
    """
    target = f"{rule.parent_table}.{rule.parent_column}"
    metadata = _metadata(rule, sample_limit)

    if parent_values is None:
        return _skipped(
            rule,
            unavailable_reason
            or f"the values of {target} could not be read, so the reference cannot be validated",
        )

    references = values.dropna()
    total = int(len(references))
    metrics: dict[str, object] = {
        "rows": int(len(values)),
        "references": total,
        "null_references": int(len(values) - total),
    }

    if total == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.NOT_APPLICABLE,
            severity=SEVERITY,
            message=(
                f"{rule.table}.{rule.column} has no non-null value to validate against {target}"
            ),
            table=rule.table,
            metrics={**metrics, "valid": 0, "invalid": 0, "invalid_pct": 0.0},
            metadata=metadata,
        )

    invalid_mask = ~references.isin(parent_values)
    invalid = int(invalid_mask.sum())
    valid = total - invalid
    invalid_pct = round(100.0 * invalid / total, 6)

    metrics.update({"valid": valid, "invalid": invalid, "invalid_pct": invalid_pct})

    if invalid == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.PASS,
            severity=SEVERITY,
            message=(
                f"all {total} non-null reference(s) in {rule.table}.{rule.column} "
                f"resolve to {target}"
            ),
            table=rule.table,
            metrics=metrics,
            metadata=metadata,
        )

    offending = references[invalid_mask]
    samples = tuple(
        {"value": str(value), "row": int(index) + 1}
        for index, value in offending.head(sample_limit).items()
    )
    return CheckResult(
        check_id=rule.check_id,
        status=Status.FAIL,
        severity=SEVERITY,
        message=(
            f"{invalid} of {total} reference(s) ({invalid_pct}%) in "
            f"{rule.table}.{rule.column} do not resolve to {target}"
        ),
        table=rule.table,
        metrics=metrics,
        samples=samples,
        metadata=metadata,
    )


def run_key_checks(
    data_dir: str | Path,
    *,
    discovery: DiscoveryResult | None = None,
    structure: Mapping[str, StructureReport] | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> tuple[CheckResult, ...]:
    """Run every confirmed key rule against the dataset in ``data_dir``.

    Parent keys are built first (one column per target table, reduced to a set),
    then each table is loaded once for the columns its rules need and released
    before the next one. Results are sorted by ``check_id`` so two runs over the
    same data produce the same report.

    :param structure: structural report per table. A table whose rows do not line up
        with its header cannot have its rows attributed to its columns, so neither its
        own rules nor the references that point at it produce a verdict. Built from
        the files when not given.
    """
    found = discovery if discovery is not None else discover_dataset(data_dir)
    paths = {table.name: table.path for table in found.tables}
    structures = (
        structure
        if structure is not None
        else validate_tables(paths, sample_limit=sample_limit)
    )

    headers: dict[str, tuple[str, ...] | None] = {}
    problems: dict[str, str] = {}
    for name, path in paths.items():
        try:
            headers[name] = read_header(path)
        except TableLoadError as exc:
            headers[name] = None
            problems[name] = str(exc)

    parent_keys, parent_reasons = _build_parent_keys(paths, headers, problems, structures)

    results: list[CheckResult] = []

    wanted: dict[str, set[str]] = {}
    for rule in (*PRIMARY_KEYS, *FOREIGN_KEYS):
        if rule.table in paths:
            wanted.setdefault(rule.table, set()).add(rule.column)

    for table in sorted(wanted):
        path = paths[table]
        header = headers[table]
        if header is None:
            reason = problems.get(table, f"table '{table}' could not be read")
            results.extend(_unavailable(rule, reason) for rule in _rules_for(table))
            continue

        structural = gate_reason(structures, table)
        if structural is not None:
            results.extend(_skipped(rule, structural) for rule in _rules_for(table))
            continue

        # Availability is decided per rule, not per table: one missing column must
        # not suppress the checks whose columns are present.
        columns = sorted(wanted[table])
        present = [column for column in columns if column in header]
        absent = [column for column in columns if column not in header]

        frame: pd.DataFrame | None = None
        if present:
            try:
                frame = load_table(path, columns=present).frame
            except TableLoadError as exc:
                results.extend(_unavailable(rule, str(exc)) for rule in _rules_for(table))
                continue

        rule: PrimaryKeyRule | ForeignKeyRule
        for rule in _rules_for(table):
            if rule.column in absent:
                results.append(
                    _skipped(
                        rule,
                        f"table '{table}' has no column '{rule.column}' in this dataset",
                    )
                )
                continue
            if frame is None or rule.column not in present:  # defensive; see present/absent
                results.append(_skipped(rule, f"column '{rule.column}' is not available"))
                continue
            values = cast(pd.Series, frame[rule.column])
            if isinstance(rule, PrimaryKeyRule):
                results.append(check_primary_key(rule, values, sample_limit=sample_limit))
            else:
                target = (rule.parent_table, rule.parent_column)
                results.append(
                    check_foreign_key(
                        rule,
                        values,
                        parent_keys.get(target),
                        sample_limit=sample_limit,
                        unavailable_reason=parent_reasons.get(target),
                    )
                )
        # Release this table before loading the next one.
        del frame

    for rule in (*PRIMARY_KEYS, *FOREIGN_KEYS):
        if rule.table not in paths:
            results.append(
                _skipped(
                    rule,
                    f"table '{rule.table}' is not present in this dataset, so this rule does not apply",
                )
            )

    return tuple(sorted(results, key=lambda result: result.check_id))


def _build_parent_keys(
    paths: Mapping[str, Path],
    headers: Mapping[str, tuple[str, ...] | None],
    problems: Mapping[str, str],
    structure: Mapping[str, StructureReport] | None = None,
) -> tuple[dict[tuple[str, str], frozenset[str] | None], dict[tuple[str, str], str]]:
    """Read one column per foreign key target and reduce it to a set of values.

    A parent table whose rows do not line up with its header is not read at all: a key
    set built from misaligned rows would resolve references that do not resolve, so its
    dependants are skipped with that reason instead.
    """
    keys: dict[tuple[str, str], frozenset[str] | None] = {}
    reasons: dict[tuple[str, str], str] = {}

    for table, column in parent_columns():
        target = (table, column)
        keys[target] = None
        path = paths.get(table)
        if path is None:
            reasons[target] = f"table '{table}' is not present in this dataset"
            continue
        header = headers.get(table)
        if header is None:
            reasons[target] = problems.get(table, f"table '{table}' could not be read")
            continue
        structural = gate_reason(structure, table)
        if structural is not None:
            reasons[target] = structural
            continue
        if column not in header:
            reasons[target] = f"table '{table}' has no column '{column}' in this dataset"
            continue
        try:
            series = load_table(path, columns=[column]).frame[column]
        except TableLoadError as exc:
            reasons[target] = str(exc)
            continue
        keys[target] = frozenset(series.dropna().unique().tolist())

    return keys, reasons


def _rules_for(table: str) -> tuple[PrimaryKeyRule | ForeignKeyRule, ...]:
    """Every key rule declared on ``table``, primary key first."""
    return (*_all_primary(table), *_foreign_key_rules(table))


def _all_primary(table: str) -> tuple[PrimaryKeyRule, ...]:
    return tuple(rule for rule in PRIMARY_KEYS if rule.table == table)


def _foreign_key_rules(table: str) -> tuple[ForeignKeyRule, ...]:
    return tuple(rule for rule in FOREIGN_KEYS if rule.table == table)


def _unavailable(rule: PrimaryKeyRule | ForeignKeyRule, reason: str) -> CheckResult:
    """The table exists but could not be read, so no verdict is produced."""
    return _skipped(rule, f"table '{rule.table}' could not be read: {reason}")


def _skipped(rule: PrimaryKeyRule | ForeignKeyRule, reason: str) -> CheckResult:
    kind = "primary key" if isinstance(rule, PrimaryKeyRule) else "foreign key"
    return CheckResult(
        check_id=rule.check_id,
        status=Status.SKIPPED,
        severity=SEVERITY,
        message=f"{kind} check not run for {rule.table}.{rule.column}: {reason}",
        table=rule.table,
        metadata=_metadata(rule, None),
    )


def _metadata(
    rule: PrimaryKeyRule | ForeignKeyRule, sample_limit: int | None
) -> Mapping[str, object]:
    """Provenance of the rule, so a report can show why the check exists."""
    metadata: dict[str, object] = {
        "rule": rule.check_id,
        "confirmed_by": rule.confirmed_by,
    }
    if isinstance(rule, ForeignKeyRule):
        metadata["target"] = f"{rule.parent_table}.{rule.parent_column}"
    if sample_limit is not None:
        metadata["sample_limit"] = sample_limit
    return metadata
