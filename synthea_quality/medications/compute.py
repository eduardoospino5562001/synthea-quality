"""[MEDICATIONS] The share of a cohort with each medication, from frames already loaded.

Definitions (``ref`` is the reference date, a calendar day; ``medications.csv`` writes
``START`` and ``STOP`` as UTC timestamps, compared by their day):

* a record counts for **ever** when its ``START`` day is on or before ``ref``;
* it counts for **active** when, in addition, ``STOP`` is empty or its day is after
  ``ref`` — the rule of point prevalence;
* the numerator is the number of **distinct patients of the cohort** with at least one
  such record of any of the medication's codes; the denominator is the cohort.

What is counted, then left out (every count is in the result): rows of patients outside
the alive cohort, rows whose ``START`` is empty or unparseable, rows that start after the
reference date; a row whose ``STOP`` is present but unparseable is kept for *ever* and
left out of *active*.

A record without ``STOP`` stays active forever: Synthea writes no ``STOP`` for a chronic
prescription that is never ended, so *active* then means "prescribed and never stopped".
Every result says how many of its records have no ``STOP``. A code that appears nowhere in
``medications.csv`` gets a note, since its 0 says nothing about the module.

``REASONCODE`` is compared with the codes of the cohort's condition, by code only (the
column has no system).

This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import pandas as pd

from synthea_quality.condition_cohort import select_cohort
from synthea_quality.medications.definitions import MedicationDefinition
from synthea_quality.medications.models import MedicationResult
from synthea_quality.prevalence.compute import Cohort, Records, most_frequent_descriptions
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.prevalence.models import ExpectedComparison, Rate
from synthea_quality.profile.dates import parse_timestamps
from synthea_quality.profile.models import SectionStatus

#: Columns of ``medications.csv`` this module uses, and those it cannot do without.
MEDICATION_COLUMNS = ("START", "STOP", "PATIENT", "CODE", "DESCRIPTION", "REASONCODE")
REQUIRED_MEDICATION_COLUMNS = ("START", "STOP", "PATIENT", "CODE")


@dataclass(frozen=True, slots=True, eq=False)
class MedicationRecords:
    """The alive patients' medication records that can be placed in time."""

    #: PATIENT, CODE, REASONCODE ("" when absent), ever, active, has_stop (bools)
    frame: pd.DataFrame
    #: Every CODE of the whole table, any patient, any date.
    table_codes: frozenset[str]
    descriptions: dict[str, str | None]
    has_reason: bool
    metrics: dict[str, int]


def prepare_medications(
    medications: pd.DataFrame, population: Cohort, reference: date
) -> MedicationRecords:
    """Place every medication row in time relative to ``reference`` and count what is left out."""
    index = medications.index
    has_reason = "REASONCODE" in medications.columns
    coded = medications["CODE"].notna()
    alive = medications["PATIENT"].isin(population.alive_ids)
    start = parse_timestamps(medications["START"]).values.dt.normalize()
    stop = parse_timestamps(medications["STOP"]).values.dt.normalize()
    stop_missing = medications["STOP"].isna()
    stop_bad = ~stop_missing & stop.isna()
    ref = pd.Timestamp(reference)

    usable = alive & coded & start.notna()
    ever = usable & (start <= ref)
    active = ever & ~stop_bad & (stop_missing | (stop > ref))
    frame = pd.DataFrame(
        {
            "PATIENT": medications["PATIENT"],
            "CODE": medications["CODE"],
            "REASONCODE": medications["REASONCODE"].fillna("")
            if has_reason
            else pd.Series("", index=index),
            "ever": ever,
            "active": active,
            "has_stop": ~stop_missing,
        }
    )[ever]
    descriptions = most_frequent_descriptions(medications, pd.Series("", index=index), coded)
    metrics = {
        "rows": len(medications),
        "rows_alive": int(alive.sum()),
        "rows_not_alive": int((~alive).sum()),
        "rows_without_code": int((alive & ~coded).sum()),
        "rows_start_unusable": int((alive & coded & start.isna()).sum()),
        "rows_after_reference": int((usable & ~(start <= ref)).sum()),
        "rows_stop_unparseable": int((ever & stop_bad).sum()),
        "rows_used": int(ever.sum()),
    }
    return MedicationRecords(
        frame=frame,
        table_codes=frozenset(medications.loc[coded, "CODE"]),
        descriptions={code: text for (_, code), text in descriptions.items()},
        has_reason=has_reason,
        metrics=metrics,
    )


def medication_use(
    definition: MedicationDefinition,
    records: MedicationRecords,
    population: Cohort,
    *,
    members: pd.Index | None = None,
    condition: ConditionDefinition | None = None,
) -> MedicationResult:
    """The share of ``members`` (default: every alive patient) with ``definition``.

    :param condition: the cohort's condition, whose codes ``REASONCODE`` is compared with.
    """
    people = population.alive_ids if members is None else members
    frame = records.frame
    rows = frame[frame["CODE"].isin(definition.codes) & frame["PATIENT"].isin(people)]
    ever_ids = pd.Index(rows["PATIENT"].unique())
    active_ids = pd.Index(rows.loc[rows["active"], "PATIENT"].unique())
    active = Rate(len(active_ids), len(people))
    ever = Rate(len(ever_ids), len(people))

    active_reason = ever_reason = None
    if condition is not None and records.has_reason:
        reason_codes = {ref.code for ref in condition.codes}
        with_reason = rows[rows["REASONCODE"].isin(reason_codes)]
        ever_reason = int(with_reason["PATIENT"].nunique())
        active_reason = int(with_reason.loc[with_reason["active"], "PATIENT"].nunique())

    without_stop = int((~rows["has_stop"]).sum())
    metrics: dict[str, Any] = {
        "records": len(rows),
        "records_by_code": {code: int((rows["CODE"] == code).sum()) for code in definition.codes},
        "records_without_stop": without_stop,
        "records_with_stop": len(rows) - without_stop,
    }
    notes: list[str] = []
    missing = [code for code in definition.codes if code not in records.table_codes]
    if missing:
        notes.append(
            f"No record of {', '.join(missing)} in medications.csv (any patient, any date): "
            f"its 0 says the code is absent from the data, not that the cohort goes without it."
        )
    if without_stop:
        notes.append(
            f"{without_stop} of {len(rows)} records have no STOP: they count as active at the "
            f"reference date (a prescription that is never ended, as Synthea writes chronic "
            f"ones)."
        )
    if condition is not None and not records.has_reason:
        notes.append("medications.csv has no REASONCODE column: reasons are not counted.")

    by_measure = {"active": active, "ever": ever}
    return MedicationResult(
        name=definition.name,
        codes=definition.codes,
        status=SectionStatus.COMPUTED,
        cohort=definition.cohort,
        descriptions={
            code: records.descriptions.get(code)
            for code in definition.codes
            if code in records.table_codes
        },
        active=active,
        ever=ever,
        active_with_reason=active_reason,
        ever_with_reason=ever_reason,
        expected=tuple(
            ExpectedComparison(expected, by_measure[expected.measure])
            for expected in definition.expected
        ),
        metrics=metrics,
        notes=tuple(notes),
    )


def skipped_medication(definition: MedicationDefinition, reason: str) -> MedicationResult:
    return MedicationResult(
        name=definition.name,
        codes=definition.codes,
        status=SectionStatus.SKIPPED,
        reason=reason,
        cohort=definition.cohort,
    )


def medication_results(
    medications: Sequence[MedicationDefinition],
    conditions: Sequence[ConditionDefinition],
    records: MedicationRecords,
    population: Cohort,
    *,
    condition_records: Records | None = None,
    records_reason: str | None = None,
) -> tuple[MedicationResult, ...]:
    """Every medication asked for, each among the alive or among its cohort."""
    by_name: Mapping[str, ConditionDefinition] = {c.name: c for c in conditions}
    results = []
    for definition in medications:
        members = condition = None
        if definition.cohort is not None:
            members, reason = select_cohort(
                definition.cohort, by_name, condition_records, records_reason
            )
            if members is None:
                results.append(skipped_medication(definition, reason or "no cohort"))
                continue
            condition = by_name[definition.cohort.condition]
        results.append(
            medication_use(
                definition, records, population, members=members, condition=condition
            )
        )
    return tuple(results)
