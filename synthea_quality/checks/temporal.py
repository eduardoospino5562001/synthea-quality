"""[CHECKS] Temporal validation: order of two dates that belong together.

Only relations the sources confirm are enforced, and only where the evidence says a
failure would be meaningful. Dates are compared as text: every value first has to
match the confirmed format of its column (from
:mod:`synthea_quality.schema.quality`), and ISO-8601 strings with zero padding sort
correctly as strings, so no second date parser exists in this module.

Three relations are checked:

* ``temporal.start_le_stop.<table>`` — ``FAIL`` when a start is later than its stop.
  A null stop is allowed and counted, never failed: it means the event is still open.
  Equal values pass.
* ``temporal.birth_le_death.patients`` — ``FAIL`` when a death date precedes the
  birth date. Rows without a death date (the patient was alive when the simulation
  ended) are counted and skipped.
* ``temporal.event_after_birth.<table>.<column>`` — ``WARNING`` when an event the
  patient experienced is dated before the patient was born. It is a sanity check
  rather than a documented guarantee, so it informs instead of failing.

Rows whose dates cannot be interpreted are excluded from the comparison and counted
in the metrics, because comparing an unreadable value would produce a verdict about
nothing; the date checks already report those rows as a defect.

Memory
------
The patient birth dates are the only cross-table state: two columns of ``patients``
held as one series while the other tables are processed. Each other table is loaded
one at a time with just the columns its rules need, and released immediately after.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import TableLoadError
from synthea_quality.loader import load_table, read_header
from synthea_quality.models import DEFAULT_SAMPLE_LIMIT, CheckResult, Severity, Status
from synthea_quality.schema.quality import date_rule_for
from synthea_quality.schema.tables import SYNTHEA_TABLES
from synthea_quality.schema.temporal import (
    EVENT_DATE_RULES,
    INTERVAL_RULES,
    LIFE_SPAN_RULES,
    EventDateRule,
    IntervalRule,
    LifeSpanRule,
    event_date_rules_for,
    interval_rules_for,
    life_span_rules_for,
)
from synthea_quality.structure import StructureReport, gate_reason, validate_tables

#: A wrong order is a row-level inconsistency, not a broken join.
ORDER_SEVERITY = Severity.MEDIUM
#: Reported for a human to look at, so the impact is the same but the status is WARNING.
EVENT_SEVERITY = Severity.MEDIUM

#: Columns loaded only to make a sample traceable, in order of preference. At most
#: ``MAX_IDENTIFIERS`` of them are added, so a sample says which record to open
#: without paying for the whole table.
IDENTIFIER_COLUMNS = ("Id", "PATIENT", "PATIENTID", "ENCOUNTER", "CLAIMID", "ID")
MAX_IDENTIFIERS = 2


def check_interval(
    rule: IntervalRule,
    frame: pd.DataFrame,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Check that ``start`` never follows ``stop``."""
    start = cast(pd.Series, frame[rule.start_column])
    stop = cast(pd.Series, frame[rule.stop_column])
    start_ok = _usable(start, rule.table, rule.start_column)
    stop_ok = _usable(stop, rule.table, rule.stop_column)

    comparable = start_ok & stop_ok
    metrics: dict[str, object] = {
        "rows": int(len(frame)),
        "evaluated": int(comparable.sum()),
        "null_start": int(start.isna().sum()),
        "null_stop": int(stop.isna().sum()),
        "unusable_start": int((start.notna() & ~start_ok).sum()),
        "unusable_stop": int((stop.notna() & ~stop_ok).sum()),
    }
    metadata = _metadata(rule, sample_limit)

    if metrics["evaluated"] == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.NOT_APPLICABLE,
            severity=ORDER_SEVERITY,
            message=(
                f"no row of {rule.table} has both a usable {rule.start_column} and "
                f"{rule.stop_column} to compare ({metrics['null_stop']} row(s) have a null "
                f"{rule.stop_column}, which means the event is still open)"
            ),
            table=rule.table,
            metrics={**metrics, "violations": 0, "violation_pct": 0.0},
            metadata=metadata,
        )

    starts = start[comparable]
    stops = stop[comparable]
    violating = starts > stops
    violations = int(violating.sum())
    metrics.update(
        {
            "violations": violations,
            "violation_pct": round(100.0 * violations / int(comparable.sum()), 6),
        }
    )

    if violations == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.PASS,
            severity=ORDER_SEVERITY,
            message=(
                f"{rule.start_column} <= {rule.stop_column} holds in all "
                f"{metrics['evaluated']} comparable row(s) of {rule.table}"
            ),
            table=rule.table,
            metrics=metrics,
            metadata=metadata,
        )

    samples = _samples(frame, starts.index[violating][:sample_limit], rule)
    return CheckResult(
        check_id=rule.check_id,
        status=Status.FAIL,
        severity=ORDER_SEVERITY,
        message=(
            f"{violations} of {metrics['evaluated']} row(s) ({metrics['violation_pct']}%) of "
            f"{rule.table} have {rule.start_column} after {rule.stop_column}"
        ),
        table=rule.table,
        metrics=metrics,
        samples=samples,
        metadata=metadata,
    )


def check_life_span(
    rule: LifeSpanRule,
    frame: pd.DataFrame,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Check that a patient is not recorded as dying before being born."""
    birth = cast(pd.Series, frame[rule.birth_column])
    death = cast(pd.Series, frame[rule.death_column])
    birth_ok = _usable(birth, rule.table, rule.birth_column)
    death_ok = _usable(death, rule.table, rule.death_column)

    comparable = birth_ok & death_ok
    metrics: dict[str, object] = {
        "rows": int(len(frame)),
        "evaluated": int(comparable.sum()),
        "null_birth": int(birth.isna().sum()),
        "null_death": int(death.isna().sum()),
        "unusable_birth": int((birth.notna() & ~birth_ok).sum()),
        "unusable_death": int((death.notna() & ~death_ok).sum()),
    }
    metadata = _metadata(rule, sample_limit)

    if metrics["evaluated"] == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.NOT_APPLICABLE,
            severity=ORDER_SEVERITY,
            message=(
                f"no row of {rule.table} has both a usable {rule.birth_column} and "
                f"{rule.death_column} to compare ({metrics['null_death']} row(s) have no "
                f"{rule.death_column}, which means the patient was alive at the end of the "
                f"simulation)"
            ),
            table=rule.table,
            metrics={**metrics, "violations": 0, "violation_pct": 0.0},
            metadata=metadata,
        )

    births = birth[comparable]
    deaths = death[comparable]
    violating = deaths < births
    violations = int(violating.sum())
    metrics.update(
        {
            "violations": violations,
            "violation_pct": round(100.0 * violations / int(comparable.sum()), 6),
        }
    )

    if violations == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.PASS,
            severity=ORDER_SEVERITY,
            message=(
                f"{rule.birth_column} <= {rule.death_column} holds in all "
                f"{metrics['evaluated']} comparable row(s) of {rule.table}"
            ),
            table=rule.table,
            metrics=metrics,
            metadata=metadata,
        )

    samples = _samples(frame, births.index[violating][:sample_limit], rule)
    return CheckResult(
        check_id=rule.check_id,
        status=Status.FAIL,
        severity=ORDER_SEVERITY,
        message=(
            f"{violations} of {metrics['evaluated']} row(s) ({metrics['violation_pct']}%) of "
            f"{rule.table} have {rule.death_column} before {rule.birth_column}"
        ),
        table=rule.table,
        metrics=metrics,
        samples=samples,
        metadata=metadata,
    )


def check_event_after_birth(
    rule: EventDateRule,
    frame: pd.DataFrame,
    birth_dates: Mapping[str, str] | None,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> CheckResult:
    """Check that an event is not dated before the patient was born.

    ``birth_dates`` maps a patient identifier to that patient's birth date; ``None``
    means the dates could not be read, in which case the check is ``SKIPPED`` instead
    of guessing.
    """
    if birth_dates is None:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.SKIPPED,
            severity=EVENT_SEVERITY,
            message=(
                f"event date check not run for {rule.table}.{rule.column}: the birth dates "
                f"of table 'patients' could not be built"
            ),
            table=rule.table,
            metadata=_metadata(rule, sample_limit),
        )

    events = cast(pd.Series, frame[rule.column])
    event_ok = _usable(events, rule.table, rule.column)
    mapped = frame[rule.patient_column].map(birth_dates)
    known_patient = mapped.notna()
    comparable = event_ok & known_patient

    metrics: dict[str, object] = {
        "rows": int(len(frame)),
        "evaluated": int(comparable.sum()),
        "null_event": int(events.isna().sum()),
        "unusable_event": int((events.notna() & ~event_ok).sum()),
        "event_without_birth_date": int((event_ok & ~known_patient).sum()),
    }
    metadata = _metadata(rule, sample_limit)

    if metrics["evaluated"] == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.NOT_APPLICABLE,
            severity=EVENT_SEVERITY,
            message=(
                f"no row of {rule.table} has both a usable {rule.column} and a known birth "
                f"date to compare"
            ),
            table=rule.table,
            metrics={**metrics, "violations": 0, "violation_pct": 0.0},
            metadata=metadata,
        )

    event_values = events[comparable]
    birth_values = mapped[comparable]
    violating = event_values < birth_values
    violations = int(violating.sum())
    metrics.update(
        {
            "violations": violations,
            "violation_pct": round(100.0 * violations / int(comparable.sum()), 6),
        }
    )

    if violations == 0:
        return CheckResult(
            check_id=rule.check_id,
            status=Status.PASS,
            severity=EVENT_SEVERITY,
            message=(
                f"no {rule.table}.{rule.column} falls before the patient's birth date "
                f"({metrics['evaluated']} row(s) compared)"
            ),
            table=rule.table,
            metrics=metrics,
            metadata=metadata,
        )

    samples = _samples(frame, event_values.index[violating][:sample_limit], rule, birth_values)
    return CheckResult(
        check_id=rule.check_id,
        status=Status.WARNING,
        severity=EVENT_SEVERITY,
        message=(
            f"{violations} of {metrics['evaluated']} row(s) ({metrics['violation_pct']}%) of "
            f"{rule.table} have {rule.column} before the patient's birth date"
        ),
        table=rule.table,
        metrics=metrics,
        samples=samples,
        metadata=metadata,
    )


def run_temporal_checks(
    data_dir: str | Path,
    *,
    discovery: DiscoveryResult | None = None,
    structure: Mapping[str, StructureReport] | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> tuple[CheckResult, ...]:
    """Run every confirmed temporal rule against the dataset in ``data_dir``.

    :param structure: structural report per table. Dates read from rows that do not line
        up with the header are dates of unknown columns, so a broken table is skipped
        rather than compared — including ``patients``, whose birth dates feed the event
        checks of every other table.
    """
    found = discovery if discovery is not None else discover_dataset(data_dir)
    paths = {table.name: table.path for table in found.tables}
    structures = (
        structure
        if structure is not None
        else validate_tables(paths, sample_limit=sample_limit)
    )

    headers: dict[str, tuple[str, ...]] = {}
    problems: dict[str, str] = {}
    for name, path in paths.items():
        try:
            headers[name] = read_header(path)
        except TableLoadError as exc:
            problems[name] = str(exc)

    birth_dates, birth_problem = _patient_birth_dates(paths, headers, problems, structures)
    results: list[CheckResult] = []

    for table in sorted(set(paths) & _tables_with_rules()):
        rules = (*interval_rules_for(table), *life_span_rules_for(table), *event_date_rules_for(table))
        header = headers.get(table)
        if header is None:
            reason = problems.get(table, f"table '{table}' could not be read")
            results.extend(
                _skipped(rule.check_id, table, f"table '{table}' could not be read: {reason}")
                for rule in rules
            )
            continue

        structural = gate_reason(structures, table)
        if structural is not None:
            results.extend(_skipped(rule.check_id, table, structural) for rule in rules)
            continue

        columns = _columns_needed(rules, header)
        try:
            frame = load_table(paths[table], columns=columns).frame
        except TableLoadError as exc:
            results.extend(
                _skipped(rule.check_id, table, f"table '{table}' could not be read: {exc}")
                for rule in rules
            )
            continue

        for rule in rules:
            missing = _missing_columns(rule, header)
            if missing:
                results.append(
                    _skipped(
                        rule.check_id,
                        table,
                        f"table '{table}' has no column(s) {sorted(missing)} in this dataset",
                    )
                )
            elif isinstance(rule, IntervalRule):
                results.append(check_interval(rule, frame, sample_limit=sample_limit))
            elif isinstance(rule, LifeSpanRule):
                results.append(check_life_span(rule, frame, sample_limit=sample_limit))
            elif birth_dates is None:
                results.append(
                    _skipped(
                        rule.check_id,
                        table,
                        birth_problem or "the patient birth dates could not be built",
                    )
                )
            else:
                results.append(
                    check_event_after_birth(
                        rule, frame, birth_dates, sample_limit=sample_limit
                    )
                )
        del frame

    for spec in SYNTHEA_TABLES:
        if spec.name in paths:
            continue
        reason = f"table '{spec.name}' is not present in this dataset, so this rule does not apply"
        for rule in (
            *interval_rules_for(spec.name),
            *life_span_rules_for(spec.name),
            *event_date_rules_for(spec.name),
        ):
            results.append(_skipped(rule.check_id, spec.name, reason))

    return tuple(sorted(results, key=lambda result: result.check_id))


def _patient_birth_dates(
    paths: Mapping[str, Path],
    headers: Mapping[str, tuple[str, ...]],
    problems: Mapping[str, str],
    structure: Mapping[str, StructureReport] | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    """Map patient identifier to a usable birth date, or explain why it is missing.

    Two ways of losing the answer, both reported instead of guessed: the table cannot be
    read or lacks a column, and — the reason this function is careful — the patient
    identifiers are not unique, in which case "the patient's birth date" has no single
    value and picking one row (the last, say) would be an invented verdict.
    """
    header = headers.get("patients")
    if "patients" not in paths:
        return None, "table 'patients' is not present in this dataset"
    if header is None:
        return None, f"table 'patients' could not be read: {problems.get('patients', 'unknown')}"
    structural = gate_reason(structure, "patients")
    if structural is not None:
        return None, structural
    missing = [column for column in ("Id", "BIRTHDATE") if column not in header]
    if missing:
        return None, f"table 'patients' has no column(s) {missing} in this dataset"
    try:
        frame = load_table(paths["patients"], columns=["Id", "BIRTHDATE"]).frame
    except TableLoadError as exc:
        return None, f"table 'patients' could not be read: {exc}"

    identifiers = frame["Id"]
    repeated = int(identifiers.dropna().duplicated().sum())
    if repeated:
        return None, (
            f"patients.Id is not unique: {repeated} repeated value(s), so a single birth "
            f"date per patient cannot be chosen"
        )

    usable = _usable(frame["BIRTHDATE"], "patients", "BIRTHDATE")
    selected = frame[usable]
    return (
        {str(key): str(value) for key, value in zip(selected["Id"], selected["BIRTHDATE"])},
        None,
    )


def _tables_with_rules() -> set[str]:
    return {
        rule.table
        for rule in (*INTERVAL_RULES, *LIFE_SPAN_RULES, *EVENT_DATE_RULES)
    }


def _columns_needed(
    rules: Sequence[IntervalRule | LifeSpanRule | EventDateRule], header: tuple[str, ...]
) -> list[str]:
    """Rule columns plus a bounded set of identifiers, all present in the file."""
    wanted: list[str] = []
    for rule in rules:
        for column in _rule_columns(rule):
            if column not in wanted:
                wanted.append(column)
    identifiers = [column for column in IDENTIFIER_COLUMNS if column in header]
    for column in identifiers[:MAX_IDENTIFIERS]:
        if column not in wanted:
            wanted.append(column)
    return [column for column in wanted if column in header]


def _rule_columns(rule: IntervalRule | LifeSpanRule | EventDateRule) -> tuple[str, ...]:
    if isinstance(rule, IntervalRule):
        return (rule.start_column, rule.stop_column)
    if isinstance(rule, LifeSpanRule):
        return (rule.birth_column, rule.death_column)
    return (rule.column, rule.patient_column)


def _missing_columns(
    rule: IntervalRule | LifeSpanRule | EventDateRule, header: tuple[str, ...]
) -> set[str]:
    return {column for column in _rule_columns(rule) if column not in header}


def _usable(values: pd.Series, table: str, column: str) -> pd.Series:
    """Mask of values that are present and match the confirmed format of the column."""
    rule = date_rule_for(table, column)
    if rule is None:  # pragma: no cover - the catalogue refuses such a rule at import
        raise ValueError(f"no confirmed date format for {table}.{column}")
    text = values.astype(str)
    return values.notna() & text.str.fullmatch(rule.format.pattern).fillna(False)


def _samples(
    frame: pd.DataFrame,
    indices: Sequence[object],
    rule: IntervalRule | LifeSpanRule | EventDateRule,
    birth_values: pd.Series | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Turn offending rows into traceable samples: row number, values, identifiers."""
    columns = [column for column in _rule_columns(rule) if column in frame.columns]
    for column in IDENTIFIER_COLUMNS[:MAX_IDENTIFIERS]:
        if column in frame.columns and column not in columns:
            columns.append(column)

    samples: list[Mapping[str, object]] = []
    for index in indices:
        row = frame.loc[index]
        values = {column: _text(row[column]) for column in columns}
        if birth_values is not None:
            values["patient_birth_date"] = _text(birth_values.loc[index])
        samples.append({"row": int(index) + 1, "values": values})  # type: ignore[call-overload]
    return tuple(samples)


def _metadata(
    rule: IntervalRule | LifeSpanRule | EventDateRule, sample_limit: int | None
) -> Mapping[str, object]:
    metadata: dict[str, object] = {
        "rule": rule.check_id,
        "documented_as": rule.documented_as,
    }
    if sample_limit is not None:
        metadata["sample_limit"] = sample_limit
    if isinstance(rule, IntervalRule):
        metadata["relation"] = f"{rule.start_column} <= {rule.stop_column}"
    elif isinstance(rule, LifeSpanRule):
        metadata["relation"] = f"{rule.birth_column} <= {rule.death_column}"
    else:
        metadata["relation"] = f"{rule.column} >= patients.BIRTHDATE"
    return metadata


def _skipped(check_id: str, table: str, reason: str) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        status=Status.SKIPPED,
        severity=ORDER_SEVERITY,
        message=f"temporal check not run for {check_id}: {reason}",
        table=table,
    )


def _text(value: object) -> str | None:
    """Render a cell for a sample without turning a null into the text 'nan'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
