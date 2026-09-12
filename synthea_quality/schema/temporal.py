"""[SCHEMA] Temporal rules confirmed by Synthea's documentation and code.

Every rule here states a relation the generator itself guarantees. Nothing is
inferred from the shape of a column name, and no clinical expectation is encoded:
the rules are about the *order* of two dates the dictionary describes as the
beginning and the end of one thing.

Two invariants hold the catalogue together:

* a temporal rule may only reference columns that already have a confirmed date
  rule in :mod:`synthea_quality.schema.quality`. That is what keeps date parsing in
  one place: the temporal checks reuse the confirmed format instead of re-deriving
  it, and a rule for a column with an unknown format cannot be written at all;
* every rule carries the documentation it comes from.

What is enforced
----------------
``start <= stop`` for the seven tables whose STOP the dictionary describes as the
end of the same event (allergies, careplans, conditions, devices, encounters,
medications, procedures). Measured on the official 2026-08 sample: 0 inversions in
26,678 evaluable pairs. Equal values are legitimate and common (995 medication rows,
489 condition rows), so equality passes.

``birth <= death`` for ``patients``, and only on rows that have a death date: a null
death date means the patient was alive when the simulation ended. Measured: 9 of 108
patients have a death date, 0 inversions.

``event >= birth`` for the eleven tables that have a date of an event the patient
experienced. Reported as ``WARNING``, not ``FAIL``: this is the one relation here
that is a sanity check rather than a documented guarantee.

What is documented but NOT enforced
-----------------------------------
``claims_transactions.FROMDATE <= TODATE``
    The dictionary calls these the transaction's start and end date, but on the
    official sample 6,555 of 85,047 rows (7.7%) carry ``1970-01-01T00:00:00Z`` in
    ``TODATE`` — the epoch, which is what ``iso8601Timestamp`` writes for an unset
    long (``CSVExporter.java``: ``this.stop = claimEntry.entry.stop``). Enforcing the
    relation would report 6,541 false inversions. Every other date column in the
    sample contains no such sentinel, which is why the seven enforced pairs are safe.

``payer_transitions.START_DATE <= END_DATE``
    Clean on the sample (3,815 of 3,815), but its end date goes through the same
    ``iso8601Timestamp`` of a possibly-unset value that produces the sentinel above,
    so an open-ended plan could invert the pair. Left out until the encoding of a
    missing end is confirmed, per "when in doubt, leave it out".

Columns deliberately not used as patient events
-----------------------------------------------
``claims.SERVICEDATE``, ``claims.CURRENTILLNESSDATE``, ``claims.LASTBILLEDDATE*``,
``claims_transactions.FROMDATE``/``TODATE``
    Billing artifacts. They are derived from encounters and already covered there,
    and the dictionary describes several of them as claim administration.
``payer_transitions.START_DATE``/``END_DATE``
    Coverage periods, not clinical events.
``imaging_studies``, ``allergies.STOP``…: STOP columns are not used for the event
check either, only the primary event date of each table, to avoid reporting the same
row twice.

Not implemented at all (out of scope, not accidentally missing)
--------------------------------------------------------------
Events after death, clinical duration bounds, medical sequences and age/disease
rules are not encoded anywhere, because the sources do not state them as guarantees
and a wrong rule would produce failures for legitimate data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from synthea_quality.schema.quality import date_rule_for
from synthea_quality.schema.tables import REFERENCE_DATASET, TableSpec, tables_by_name


@dataclass(frozen=True, slots=True)
class IntervalRule:
    """``start`` and ``stop`` bound the same event, so ``start`` cannot follow ``stop``."""

    table: str
    start_column: str
    stop_column: str
    documented_as: str

    @property
    def check_id(self) -> str:
        return f"temporal.start_le_stop.{self.table}"


@dataclass(frozen=True, slots=True)
class LifeSpanRule:
    """A patient cannot die before being born."""

    table: str
    birth_column: str
    death_column: str
    documented_as: str

    @property
    def check_id(self) -> str:
        return f"temporal.birth_le_death.{self.table}"


@dataclass(frozen=True, slots=True)
class EventDateRule:
    """A date column that records something that happened to the patient."""

    table: str
    column: str
    patient_column: str
    documented_as: str

    @property
    def check_id(self) -> str:
        return f"temporal.event_after_birth.{self.table}.{self.column}"


@dataclass(frozen=True, slots=True)
class UnresolvedTemporalRelation:
    """A relation the documentation suggests but that is not safe to enforce yet.

    As in the key catalogue, the five things a reader must tell apart are stored
    apart, and the quantitative evidence is attributed to the dataset it was
    measured on rather than to the dataset a report is about.
    """

    relation: str
    documented_as: str
    implemented_as: str
    reference_dataset: str
    reference_evidence: str
    why_not_enforced: str
    pending: str


#: Pairs the dictionary documents as the start and end of the same event.
INTERVAL_RULES: tuple[IntervalRule, ...] = (
    IntervalRule(
        "allergies",
        "START",
        "STOP",
        "dictionary: 'The date the allergy was diagnosed' / 'The date the allergy ended'",
    ),
    IntervalRule(
        "careplans",
        "START",
        "STOP",
        "dictionary: 'The date the care plan was initiated' / 'The date the care plan ended'",
    ),
    IntervalRule(
        "conditions",
        "START",
        "STOP",
        "dictionary: 'The date the condition was diagnosed' / 'The date the condition resolved'",
    ),
    IntervalRule(
        "devices",
        "START",
        "STOP",
        "dictionary: 'The date and time the device was associated' / '... the device was removed'",
    ),
    IntervalRule(
        "encounters",
        "START",
        "STOP",
        "dictionary: 'The date and time the encounter started' / '... the encounter concluded'",
    ),
    IntervalRule(
        "medications",
        "START",
        "STOP",
        "dictionary: 'The date and time the medication was prescribed' / '... the prescription ended'",
    ),
    IntervalRule(
        "procedures",
        "START",
        "STOP",
        "dictionary: 'The date and time the procedure was performed' / '... the procedure was completed'",
    ),
)

#: A null death date means the patient was alive at the end of the simulation, so
#: only rows with a death date are compared.
LIFE_SPAN_RULES: tuple[LifeSpanRule, ...] = (
    LifeSpanRule(
        "patients",
        "BIRTHDATE",
        "DEATHDATE",
        "dictionary: 'The date the patient was born' / 'The date the patient died' "
        "(required: false)",
    ),
)

#: Primary event date of each table that records something happening to the patient.
EVENT_DATE_RULES: tuple[EventDateRule, ...] = (
    EventDateRule(
        "allergies", "START", "PATIENT", "dictionary: 'The date the allergy was diagnosed'"
    ),
    EventDateRule(
        "careplans", "START", "PATIENT", "dictionary: 'The date the care plan was initiated'"
    ),
    EventDateRule(
        "conditions", "START", "PATIENT", "dictionary: 'The date the condition was diagnosed'"
    ),
    EventDateRule(
        "devices",
        "START",
        "PATIENT",
        "dictionary: 'The date and time the device was associated to the patient'",
    ),
    EventDateRule(
        "encounters",
        "START",
        "PATIENT",
        "dictionary: 'The date and time the encounter started'",
    ),
    EventDateRule(
        "imaging_studies",
        "DATE",
        "PATIENT",
        "dictionary: 'The date and time the imaging study was conducted'",
    ),
    EventDateRule(
        "immunizations",
        "DATE",
        "PATIENT",
        "dictionary: 'The date the immunization was administered'",
    ),
    EventDateRule(
        "medications",
        "START",
        "PATIENT",
        "dictionary: 'The date and time the medication was prescribed'",
    ),
    EventDateRule(
        "observations",
        "DATE",
        "PATIENT",
        "dictionary: 'The date and time the observation was performed'",
    ),
    EventDateRule(
        "procedures",
        "START",
        "PATIENT",
        "dictionary: 'The date and time the procedure was performed'",
    ),
    EventDateRule(
        "supplies", "DATE", "PATIENT", "dictionary: 'The date the supplies were used'"
    ),
)

#: Relations that are documented but not applied; see the module docstring for the why.
UNRESOLVED_TEMPORAL_RELATIONS: tuple[UnresolvedTemporalRelation, ...] = (
    UnresolvedTemporalRelation(
        relation="claims_transactions.FROMDATE <= TODATE",
        documented_as=(
            "dictionary: From Date 'Transaction start date' / To Date 'Transaction end date'"
        ),
        implemented_as=(
            "CSVExporter.java: ClaimTransaction.toString() writes FROMDATE from the "
            "transaction start and TODATE from its stop ('this.stop = claimEntry.entry.stop'), "
            "and an unset stop is the value 0, which iso8601Timestamp renders as "
            "'1970-01-01T00:00:00Z'"
        ),
        reference_dataset=REFERENCE_DATASET,
        reference_evidence=(
            "6,555 of 85,047 rows (7.7%) carry '1970-01-01T00:00:00Z' in TODATE, which "
            "inverts the pair in 6,541 rows; no other date column of that sample contains "
            "that value"
        ),
        why_not_enforced=(
            "the epoch is what an unset stop time looks like in this table, so the "
            "inversion means 'no end' rather than 'ends before it starts'; enforcing the "
            "relation would report 6,541 false failures on that reference dataset"
        ),
        pending=(
            "confirmation of how a missing end is meant to be interpreted in this table, "
            "or an explicit sentinel rule; until then the relation is neither enforced nor "
            "declared a Synthea defect"
        ),
    ),
    UnresolvedTemporalRelation(
        relation="payer_transitions.START_DATE <= END_DATE",
        documented_as=(
            "dictionary (under the stale names START_YEAR/END_YEAR): 'The year the "
            "coverage started (inclusive)' / '... the year the coverage ended (inclusive)'"
        ),
        implemented_as=(
            "CSVExporter.java:1044 exportPayerTransition writes START_DATE with "
            "iso8601Timestamp(planRecord.getStartTime()) and END_DATE with "
            "iso8601Timestamp(planRecord.getStopTime()), the same writer that turns an "
            "unset value into the epoch; its own comment still says START_YEAR/END_YEAR"
        ),
        reference_dataset=REFERENCE_DATASET,
        reference_evidence=(
            "3,815 of 3,815 rows are ordered correctly and none is null, so enforcing the "
            "relation would not fire on that reference dataset"
        ),
        why_not_enforced=(
            "the end date goes through the same possibly-unset timestamp path that "
            "produces the epoch sentinel in claims_transactions, so an open-ended plan "
            "could invert the pair"
        ),
        pending=(
            "evidence that a missing coverage end is written as an empty field rather than "
            "the epoch, as it is for conditions.stop"
        ),
    ),
)


def interval_rules_for(table: str) -> tuple[IntervalRule, ...]:
    """Interval rules declared for ``table``."""
    return tuple(rule for rule in INTERVAL_RULES if rule.table == table)


def life_span_rules_for(table: str) -> tuple[LifeSpanRule, ...]:
    """Life span rules declared for ``table``."""
    return tuple(rule for rule in LIFE_SPAN_RULES if rule.table == table)


def event_date_rules_for(table: str) -> tuple[EventDateRule, ...]:
    """Event date rules declared for ``table``."""
    return tuple(rule for rule in EVENT_DATE_RULES if rule.table == table)


class _CatalogueError(ValueError):
    """A temporal rule references a table, column or format that is not confirmed."""


def _validate_catalogue() -> None:
    """Fail loudly at import time rather than produce a verdict about nothing."""
    known: Mapping[str, TableSpec] = tables_by_name()
    seen: set[str] = set()

    for rule in INTERVAL_RULES:
        _check_column(known, rule.table, rule.start_column, rule)
        _check_column(known, rule.table, rule.stop_column, rule)
        _check_unique(seen, rule.check_id)

    for rule in LIFE_SPAN_RULES:
        _check_column(known, rule.table, rule.birth_column, rule)
        _check_column(known, rule.table, rule.death_column, rule)
        _check_unique(seen, rule.check_id)

    for rule in EVENT_DATE_RULES:
        _check_column(known, rule.table, rule.column, rule)
        _check_column(known, rule.table, rule.patient_column, rule)
        _check_unique(seen, rule.check_id)

    for unresolved in UNRESOLVED_TEMPORAL_RELATIONS:
        for field_name in (
            "relation",
            "documented_as",
            "implemented_as",
            "reference_dataset",
            "reference_evidence",
            "why_not_enforced",
            "pending",
        ):
            if not getattr(unresolved, field_name).strip():
                raise _CatalogueError(
                    f"unresolved temporal relation {unresolved.relation} needs a "
                    f"'{field_name}' note"
                )


def _check_column(
    known: Mapping[str, TableSpec],
    table: str,
    column: str,
    rule: IntervalRule | LifeSpanRule | EventDateRule,
) -> None:
    spec = known.get(table)
    if spec is None:
        raise _CatalogueError(f"temporal rule {rule.check_id} refers to unknown table '{table}'")
    if column not in spec.columns:
        raise _CatalogueError(
            f"temporal rule {rule.check_id} refers to column '{column}', which is not "
            f"part of table '{table}'"
        )
    if not rule.documented_as.strip():
        raise _CatalogueError(f"temporal rule {rule.check_id} has no documentation note")


def _check_unique(seen: set[str], check_id: str) -> None:
    if check_id in seen:
        raise _CatalogueError(f"duplicate temporal rule {check_id}")
    seen.add(check_id)


def date_columns_used() -> tuple[tuple[str, str], ...]:
    """Every ``(table, column)`` compared by a temporal rule.

    All of them must have a confirmed date format, which is what allows the checks
    to reuse one definition of "this value is a date" instead of a second parser.
    """
    used: set[tuple[str, str]] = set()
    for rule in INTERVAL_RULES:
        used.add((rule.table, rule.start_column))
        used.add((rule.table, rule.stop_column))
    for rule in LIFE_SPAN_RULES:
        used.add((rule.table, rule.birth_column))
        used.add((rule.table, rule.death_column))
    for rule in EVENT_DATE_RULES:
        used.add((rule.table, rule.column))
    return tuple(sorted(used))


def _validate_formats() -> None:
    """A temporal rule may only compare columns whose date format is confirmed."""
    for table, column in date_columns_used():
        if date_rule_for(table, column) is None:
            raise _CatalogueError(
                f"temporal rules compare {table}.{column}, which has no confirmed date "
                "format in schema.quality"
            )


_validate_catalogue()
_validate_formats()
