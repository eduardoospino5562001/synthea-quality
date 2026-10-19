"""[INDEXED] Build the indexed cohort of a module file over one dataset.

The only file of the package that touches the file system. It opens no dataset of its
own: the caller (``synthea-validate-module``) has already opened it and read ``patients``
and ``conditions``, so this module only loads what the outcomes add — the table of each
``event`` outcome (``PATIENT``, ``CODE`` and the table's date column) and
``observations.csv`` when a ``value`` outcome or a ``death`` outcome with ``cause_values``
needs it. Each table is loaded once, however many outcomes read it.

A table that cannot be loaded does not sink the report: the outcome that needed it comes
out ``SKIPPED`` with the table and the reason, and the other outcomes are still measured
(RULES §2.2). When the cohort itself cannot be built (no reference date, no patients, no
condition records), every outcome is ``SKIPPED`` with that reason and the cohort's
``reason`` says why.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from synthea_quality.dataset import DatasetContext
from synthea_quality.indexed.compute import (
    DATE_COLUMNS,
    IndexedCohort,
    build_indexed_cohort,
    measure_death,
    measure_event,
    measure_value,
)
from synthea_quality.indexed.definitions import (
    DeathOutcome,
    EventOutcome,
    IndexSpec,
    Outcome,
    ValueOutcome,
)
from synthea_quality.indexed.models import (
    DeathResult,
    EventResult,
    IndexedCohortResult,
    SkippedOutcome,
    ValueResult,
)
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.profile.models import TableInput

#: The columns of ``observations.csv`` a ``value`` or a ``death`` outcome needs.
OBSERVATION_VALUE_COLUMNS = ("DATE", "PATIENT", "CODE", "VALUE")
#: The table every ``value`` outcome and every death cause reads.
OBSERVATIONS = "observations"
#: What the report says the extra tables were read for.
INDEXED_USED_FOR = "indexed cohort outcomes"

OutcomeResult = EventResult | ValueResult | DeathResult | SkippedOutcome


def indexed_tables(outcomes: Sequence[Outcome]) -> tuple[str, ...]:
    """The tables the outcomes need, without repeats and in the order they appear.

    An ``event`` outcome names its table; a ``value`` outcome and a ``death`` outcome
    with ``cause_values`` read ``observations``. A ``death`` outcome without a cause
    needs no table at all: its dates come from ``patients``.
    """
    tables: list[str] = []
    for outcome in outcomes:
        if isinstance(outcome, EventOutcome):
            table = outcome.table
        elif isinstance(outcome, ValueOutcome) or outcome.cause_values:
            table = OBSERVATIONS
        else:
            continue
        if table not in tables:
            tables.append(table)
    return tuple(tables)


def skipped_indexed(
    index: IndexSpec, outcomes: Sequence[Outcome], reference: str | None, reason: str
) -> IndexedCohortResult:
    """An indexed cohort that could not be built: no patients, every outcome skipped.

    ``reference`` is ``YYYY-MM-DD``, or ``None`` when the dataset gave no reference date.
    """
    return IndexedCohortResult(
        index_condition=index.condition,
        reference_date=reference,
        patients=0,
        rows_matched=0,
        rows_unknown_patient=0,
        rows_start_unusable=0,
        rows_after_reference=0,
        outcomes=tuple(
            SkippedOutcome(outcome.name, _outcome_type(outcome), reason)
            for outcome in outcomes
        ),
        reason=reason,
    )


def indexed_cohort_result(
    context: DatasetContext,
    patients: pd.DataFrame,
    conditions: pd.DataFrame,
    definitions: Sequence[ConditionDefinition],
    index: IndexSpec,
    outcomes: Sequence[Outcome],
    reference: date,
    age_bands: Sequence[int],
    observations: pd.DataFrame | None = None,
) -> tuple[IndexedCohortResult, tuple[TableInput, ...]]:
    """The indexed cohort of ``index`` and every outcome, with the tables it loaded.

    ``observations`` is the frame the caller already read, if any: it is then used as is
    and not loaded a second time. The returned :class:`TableInput` are only the tables
    loaded here, so the report lists each one once.

    :raises StopIteration: ``definitions`` has no condition named ``index.condition``;
        the module file's loader rejects that case before this function runs.
    """
    definition = next(d for d in definitions if d.name == index.condition)
    cohort = build_indexed_cohort(patients, conditions, definition, reference, age_bands)
    frames: dict[str, pd.DataFrame | None] = {}
    reasons: dict[str, str] = {}
    inputs: list[TableInput] = []
    for table in indexed_tables(outcomes):
        if table == OBSERVATIONS and observations is not None:
            frames[table] = observations
            continue
        frame, table_input = _load_table(context, table, _reads_value(outcomes))
        frames[table] = frame
        inputs.append(table_input)
        if frame is None:
            assert table_input.reason is not None
            reasons[table] = table_input.reason
    measured = tuple(
        _measure(outcome, cohort, frames, reasons, patients, reference, age_bands)
        for outcome in outcomes
    )
    return (
        IndexedCohortResult(
            index_condition=index.condition,
            reference_date=reference.isoformat(),
            patients=cohort.size,
            rows_matched=cohort.rows_matched,
            rows_unknown_patient=cohort.rows_unknown_patient,
            rows_start_unusable=cohort.rows_start_unusable,
            rows_after_reference=cohort.rows_after_reference,
            outcomes=measured,
        ),
        tuple(inputs),
    )


def _load_table(
    context: DatasetContext, table: str, reads_value: bool
) -> tuple[pd.DataFrame | None, TableInput]:
    """Read the columns ``table`` needs for the outcomes, and what happened to it.

    ``observations`` is always read with ``VALUE`` when it has one, but only requires it
    when ``reads_value``: an ``event`` outcome on ``observations`` does not read it.
    """
    columns: tuple[str, ...]
    if table == OBSERVATIONS:
        wanted = OBSERVATION_VALUE_COLUMNS
        if reads_value:
            columns = wanted
        else:
            columns = tuple(c for c in wanted if c != "VALUE")
        return context.load(table, wanted, columns, INDEXED_USED_FOR)
    columns = ("PATIENT", "CODE", DATE_COLUMNS.get(table, "START"))
    return context.load(table, columns, columns, INDEXED_USED_FOR)


def _reads_value(outcomes: Sequence[Outcome]) -> bool:
    """Whether an outcome reads ``VALUE``: a ``value``, or a ``death`` with ``cause_values``."""
    return any(
        isinstance(outcome, ValueOutcome)
        or (isinstance(outcome, DeathOutcome) and bool(outcome.cause_values))
        for outcome in outcomes
    )


def _measure(
    outcome: Outcome,
    cohort: IndexedCohort,
    frames: Mapping[str, pd.DataFrame | None],
    reasons: Mapping[str, str],
    patients: pd.DataFrame,
    reference: date,
    age_bands: Sequence[int],
) -> OutcomeResult:
    """One outcome, or ``SKIPPED`` with the reason when its table is not available."""
    if isinstance(outcome, DeathOutcome) and not outcome.cause_values:
        return measure_death(cohort, patients, None, outcome, reference, age_bands)
    table = _table_of(outcome)
    frame = frames.get(table)
    if frame is None:
        return SkippedOutcome(outcome.name, _outcome_type(outcome), reasons[table])
    if isinstance(outcome, EventOutcome):
        return measure_event(cohort, frame, outcome, reference, age_bands)
    if isinstance(outcome, ValueOutcome):
        return measure_value(cohort, frame, outcome, reference, age_bands)
    return measure_death(cohort, patients, frame, outcome, reference, age_bands)


def _table_of(outcome: Outcome) -> str:
    """The table ``outcome`` reads: its own for an event, ``observations`` otherwise."""
    return outcome.table if isinstance(outcome, EventOutcome) else OBSERVATIONS


def _outcome_type(outcome: Outcome) -> str:
    """The type of ``outcome`` as the module file names it."""
    if isinstance(outcome, EventOutcome):
        return "event"
    if isinstance(outcome, ValueOutcome):
        return "value"
    return "death"
