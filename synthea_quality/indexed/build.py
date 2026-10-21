"""[INDEXED] Build the indexed cohort of a module file over one dataset.

The only file of the package that touches the file system. It opens no dataset of its
own: the caller (``synthea-validate-module``) has already opened it and read ``patients``
and ``conditions``, so this module only loads what the outcomes add — the table of each
part of each ``event`` outcome (``PATIENT``, ``CODE`` and the table's date column) and
``observations.csv`` when a ``value`` outcome or a ``death`` outcome with ``cause_values``
needs it. Each table is loaded once, however many outcomes or parts read it.

A part of an ``event`` outcome can filter on a column its table has: ``VALUE``,
``REASONCODE`` or the ``STOP`` it counts on. Those columns are asked for, never
required: a table without them still serves the outcomes that do not need them, and only
the outcome whose part needs the missing column comes out ``SKIPPED``.

A table that cannot be loaded does not sink the report: the outcome that needed it comes
out ``SKIPPED`` with the table and the reason, and the other outcomes are still measured
(RULES §2.2). When the cohort itself cannot be built (no reference date, no patients, no
condition records), every outcome is ``SKIPPED`` with that reason and the cohort's
``reason`` says why.

The crossings of the module file's ``by`` list come after the outcomes: a crossing needs
the patients of both of its outcomes, so it is measured from the same tables the
outcomes were measured from and never reads a table of its own. A crossing whose
outcome could not be measured is ``SKIPPED`` with that reason, like the outcome itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from synthea_quality.dataset import DatasetContext
from synthea_quality.indexed.compute import (
    DATE_COLUMNS,
    STOP_COLUMN,
    IndexedCohort,
    OutcomeMembers,
    build_indexed_cohort,
    death_members,
    event_members,
    measure_cross,
    measure_death,
    measure_event,
    measure_value,
    skipped_cross,
    value_members,
)
from synthea_quality.indexed.definitions import (
    CrossSpec,
    DeathOutcome,
    EventMatch,
    EventOutcome,
    IndexSpec,
    Outcome,
    ValueOutcome,
)
from synthea_quality.indexed.models import (
    CrossResult,
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
#: The column of a record that holds its reason; a part can filter on it.
REASON_COLUMN = "REASONCODE"
#: The column of an observation that holds its value; a part can filter on it.
VALUE_COLUMN = "VALUE"

OutcomeResult = EventResult | ValueResult | DeathResult | SkippedOutcome


def indexed_tables(outcomes: Sequence[Outcome]) -> tuple[str, ...]:
    """The tables the outcomes need, without repeats and in the order they appear.

    An ``event`` outcome reads the table of each of its parts; a ``value`` outcome and
    a ``death`` outcome with ``cause_values`` read ``observations``. A ``death`` outcome
    without a cause needs no table at all: its dates come from ``patients``.
    """
    tables: list[str] = []
    for outcome in outcomes:
        if isinstance(outcome, EventOutcome):
            candidates = [part.table for part in outcome.parts]
        elif isinstance(outcome, ValueOutcome) or outcome.cause_values:
            candidates = [OBSERVATIONS]
        else:
            continue
        for table in candidates:
            if table not in tables:
                tables.append(table)
    return tuple(tables)


def _wanted_columns(outcomes: Sequence[Outcome]) -> dict[str, tuple[str, ...]]:
    """The optional columns each table needs, asked for but never required.

    A part of an event filters on ``VALUE`` or on ``REASONCODE``, or counts on the
    ``STOP`` of its record: the table of that part is asked for that column. The column
    is never required, so a table without it still serves the outcomes that do not need
    it, and only the outcome whose part needs it is left out.
    """
    columns: dict[str, list[str]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, EventOutcome):
            continue
        for part in outcome.parts:
            wanted = columns.setdefault(part.table, [])
            for column in _part_columns(part):
                if column not in wanted:
                    wanted.append(column)
    return {table: tuple(wanted) for table, wanted in columns.items()}


def _part_columns(part: EventMatch) -> tuple[str, ...]:
    """The optional columns a part of an event needs to be measured."""
    columns: list[str] = []
    if part.values:
        columns.append(VALUE_COLUMN)
    if part.reasons:
        columns.append(REASON_COLUMN)
    if part.at == "stop":
        columns.append(STOP_COLUMN)
    return tuple(columns)


def skipped_indexed(
    index: IndexSpec,
    outcomes: Sequence[Outcome],
    reference: str | None,
    reason: str,
    crosses: Sequence[CrossSpec] = (),
) -> IndexedCohortResult:
    """An indexed cohort that could not be built: no patients, every outcome skipped.

    ``reference`` is ``YYYY-MM-DD``, or ``None`` when the dataset gave no reference date.
    Every crossing is skipped with the same reason: without a cohort there is nothing
    to cross.
    """
    by_name = _outcomes_by_name(outcomes)
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
        crosses=tuple(
            skipped_cross(
                cross, _by_type(by_name[cross.by]), cross.outcome, reason
            )
            for cross in crosses
        ),
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
    crosses: Sequence[CrossSpec] = (),
) -> tuple[IndexedCohortResult, tuple[TableInput, ...]]:
    """The indexed cohort of ``index``, every outcome and every crossing.

    ``observations`` is the frame the caller already read, if any: it is then used as is
    and not loaded a second time. The returned :class:`TableInput` are only the tables
    loaded here, so the report lists each one once.

    :raises StopIteration: ``definitions`` has no condition named ``index.condition``;
        the module file's loader rejects that case before this function runs.
    :raises KeyError: a crossing names an outcome that ``outcomes`` does not have; the
        module file's loader rejects that case before this function runs.
    """
    definition = next(d for d in definitions if d.name == index.condition)
    cohort = build_indexed_cohort(patients, conditions, definition, reference, age_bands)
    frames: dict[str, pd.DataFrame | None] = {}
    reasons: dict[str, str] = {}
    inputs: list[TableInput] = []
    wanted = _wanted_columns(outcomes)
    for table in indexed_tables(outcomes):
        if table == OBSERVATIONS and observations is not None:
            frames[table] = observations
            continue
        frame, table_input = _load_table(context, table, wanted.get(table, ()))
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
            crosses=_crosses(
                cohort, crosses, outcomes, measured, frames, patients, reference
            ),
        ),
        tuple(inputs),
    )


def _crosses(
    cohort: IndexedCohort,
    crosses: Sequence[CrossSpec],
    outcomes: Sequence[Outcome],
    measured: Sequence[OutcomeResult],
    frames: Mapping[str, pd.DataFrame | None],
    patients: pd.DataFrame,
    reference: date,
) -> tuple[CrossResult, ...]:
    """The crossings of the module file, in its order, with the outcomes' members.

    A crossing whose outcome could not be measured is skipped with that reason, the
    measured one first: the report says which outcome is missing, not only that
    something is. The members of an outcome are computed once, however many crossings
    use it, and from the tables its own measure used.
    """
    by_name = _outcomes_by_name(outcomes)
    results = {outcome.name: outcome for outcome in measured}
    members = _members(
        cohort, crosses, by_name, results, frames, patients, reference
    )
    computed: list[CrossResult] = []
    for cross in crosses:
        by_type = _by_type(by_name[cross.by])
        missing = _missing_outcome(cross, results)
        if missing is not None:
            name, reason = missing
            computed.append(skipped_cross(cross, by_type, name, reason))
            continue
        computed.append(
            measure_cross(
                cohort,
                cross,
                members[cross.outcome],
                members[cross.by],
                by_type,
            )
        )
    return tuple(computed)


def _missing_outcome(
    cross: CrossSpec, results: Mapping[str, OutcomeResult]
) -> tuple[str, str] | None:
    """The outcome of the crossing that could not be measured, with its reason.

    The measured outcome comes first: the report names the outcome that is missing.
    """
    for name in (cross.outcome, cross.by):
        result = results[name]
        if isinstance(result, SkippedOutcome):
            return name, result.reason
    return None


def _members(
    cohort: IndexedCohort,
    crosses: Sequence[CrossSpec],
    by_name: Mapping[str, Outcome],
    results: Mapping[str, OutcomeResult],
    frames: Mapping[str, pd.DataFrame | None],
    patients: pd.DataFrame,
    reference: date,
) -> dict[str, OutcomeMembers]:
    """The patients of every outcome a crossing uses, computed once each.

    An outcome that could not be measured has no members: its crossing is skipped
    before this is asked for it.
    """
    members: dict[str, OutcomeMembers] = {}
    for cross in crosses:
        for name in (cross.outcome, cross.by):
            if name in members or isinstance(results[name], SkippedOutcome):
                continue
            members[name] = _outcome_members(
                by_name[name], cohort, frames, patients, reference
            )
    return members


def _outcome_members(
    outcome: Outcome,
    cohort: IndexedCohort,
    frames: Mapping[str, pd.DataFrame | None],
    patients: pd.DataFrame,
    reference: date,
) -> OutcomeMembers:
    """The patients of an outcome, from the tables its own measure read.

    The tables are the ones already loaded for the outcomes: a crossing never reads a
    table of its own, so the numbers of a crossing and of its outcome cannot disagree.
    """
    if isinstance(outcome, EventOutcome):
        return event_members(cohort, _tables_of_parts(outcome, frames), outcome, reference)
    if isinstance(outcome, ValueOutcome):
        frame = frames[OBSERVATIONS]
        assert frame is not None
        return value_members(cohort, frame, outcome, reference)
    observations = frames.get(OBSERVATIONS) if outcome.cause_values else None
    return death_members(cohort, patients, observations, outcome, reference)


def _outcomes_by_name(outcomes: Sequence[Outcome]) -> dict[str, Outcome]:
    """Each outcome of the module file by its name."""
    return {outcome.name: outcome for outcome in outcomes}


def _by_type(by: Outcome) -> str:
    """The kind of strata an outcome gives: one per value, or two for an event.

    :raises ValueError: the outcome is a death, which cannot give strata; the module
        file's loader rejects that case before this function runs.
    """
    if isinstance(by, ValueOutcome):
        return "value"
    if isinstance(by, EventOutcome):
        return "event"
    raise ValueError(f"outcome {by.name!r} is a death and cannot give the strata of a cross")


def _load_table(
    context: DatasetContext, table: str, optional: Sequence[str]
) -> tuple[pd.DataFrame | None, TableInput]:
    """Read the columns ``table`` needs for the outcomes, and what happened to it.

    ``observations`` is read with ``VALUE`` when it has one but never requires it: only the
    outcomes that read ``VALUE`` need it, and :func:`_measure` skips those without it.
    The optional columns of the table are asked for and never required.
    """
    if table == OBSERVATIONS:
        columns = tuple(c for c in OBSERVATION_VALUE_COLUMNS if c != VALUE_COLUMN)
        return context.load(table, OBSERVATION_VALUE_COLUMNS, columns, INDEXED_USED_FOR)
    columns = ("PATIENT", "CODE", DATE_COLUMNS.get(table, "START"))
    return context.load(table, (*columns, *optional), columns, INDEXED_USED_FOR)


def _measure(
    outcome: Outcome,
    cohort: IndexedCohort,
    frames: Mapping[str, pd.DataFrame | None],
    reasons: Mapping[str, str],
    patients: pd.DataFrame,
    reference: date,
    age_bands: Sequence[int],
) -> OutcomeResult:
    """One outcome, or ``SKIPPED`` with the reason when it could not be measured."""
    if isinstance(outcome, DeathOutcome) and not outcome.cause_values:
        return measure_death(cohort, patients, None, outcome, reference, age_bands)
    if isinstance(outcome, EventOutcome):
        skipped = _skipped_event(outcome, frames, reasons)
        if skipped is not None:
            return skipped
        tables = _tables_of_parts(outcome, frames)
        return measure_event(cohort, tables, outcome, reference, age_bands)
    frame = frames.get(OBSERVATIONS)
    if frame is None:
        return SkippedOutcome(outcome.name, _outcome_type(outcome), reasons[OBSERVATIONS])
    if VALUE_COLUMN not in frame.columns:
        reason = f"{OBSERVATIONS}.csv has no VALUE column(s)"
        return SkippedOutcome(outcome.name, _outcome_type(outcome), reason)
    if isinstance(outcome, ValueOutcome):
        return measure_value(cohort, frame, outcome, reference, age_bands)
    return measure_death(cohort, patients, frame, outcome, reference, age_bands)


def _skipped_event(
    outcome: EventOutcome,
    frames: Mapping[str, pd.DataFrame | None],
    reasons: Mapping[str, str],
) -> SkippedOutcome | None:
    """Why an event could not be measured, or ``None`` when it can be.

    The parts are read in order: the first part whose table is missing, or whose table
    lacks the column that part filters on, leaves the whole outcome out with its reason.
    """
    for part in outcome.parts:
        frame = frames.get(part.table)
        if frame is None:
            return SkippedOutcome(outcome.name, "event", reasons[part.table])
        column = _missing_column(part, frame)
        if column is not None:
            return SkippedOutcome(
                outcome.name,
                "event",
                f"{part.table}.csv has no {column} column(s), which part {part.name!r} "
                "needs",
            )
    return None


def _missing_column(part: EventMatch, frame: pd.DataFrame) -> str | None:
    """The column a part filters on that its table does not have, if there is one."""
    for column, needed in (
        (VALUE_COLUMN, bool(part.values)),
        (REASON_COLUMN, bool(part.reasons)),
        (STOP_COLUMN, part.at == "stop"),
    ):
        if needed and column not in frame.columns:
            return column
    return None


def _tables_of_parts(
    outcome: EventOutcome, frames: Mapping[str, pd.DataFrame | None]
) -> dict[str, pd.DataFrame]:
    """The table of every part of ``outcome``, ready to measure it."""
    tables: dict[str, pd.DataFrame] = {}
    for part in outcome.parts:
        frame = frames[part.table]
        assert frame is not None
        tables[part.table] = frame
    return tables


def _outcome_type(outcome: Outcome) -> str:
    """The type of ``outcome`` as the module file names it."""
    if isinstance(outcome, EventOutcome):
        return "event"
    if isinstance(outcome, ValueOutcome):
        return "value"
    return "death"
