"""[SCHEMA] Confirmed primary and foreign key rules for Synthea CSV tables.

Every rule here is confirmed against Synthea's own documentation, never guessed
from a column name:

* primary keys come from the ``:key:`` rows of the official *CSV File Data
  Dictionary* (https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary),
  which describe the column as "Primary Key. Unique Identifier of ...";
* foreign keys come from the ``:old_key:`` rows of the same document, which state
  "Foreign key to the ...".

Two facts are deliberately *not* modelled as rules:

``imaging_studies.Id``
    The dictionary documents it as a "Non-unique identifier of the imaging study.
    An imaging study may have multiple rows." On the official sample (2026-08) 65
    of 478 rows repeat it, which is expected. It is not a primary key.

``claims_transactions.PATIENTINSURANCEID``
    Documented as a foreign key to the payer transition member id, but **not applied
    as a constraint**. See :data:`UNRESOLVED_FOREIGN_KEYS`: the dictionary states the
    relationship, the reference dataset (see
    :data:`~synthea_quality.schema.tables.REFERENCE_DATASET`) shows 170 references
    (3 patients) that match no ``payer_transitions.MEMBERID``, all dated before 1970,
    and ``CSVExporter.java`` writes the value from the claim's plan record
    (``this.memberId = claim.getPlanRecordMemberId()``) while exporting only the plans
    that ended on or after 1970-01-01 (``exportPayerTransitions(person, 0L, time)``).
    Reported upstream as synthetichealth/synthea#1725. Until a maintainer confirms the
    intended semantics, the relationship is neither enforced (which would emit ``FAIL``
    for legitimate data) nor declared wrong.

The catalogue is validated on import against the generated table catalogue: a rule
pointing at a table or column that does not exist is a bug here, not a dataset
defect, so it raises immediately instead of producing a wrong verdict later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from synthea_quality.schema.tables import REFERENCE_DATASET, TableSpec, tables_by_name

#: Provenance recorded on every rule derived from Synthea's own documentation.
DATA_DICTIONARY = "Synthea CSV File Data Dictionary (:key: / :old_key:)"


@dataclass(frozen=True, slots=True)
class PrimaryKeyRule:
    """A column documented as the unique identifier of its table."""

    table: str
    column: str
    confirmed_by: str = DATA_DICTIONARY

    @property
    def check_id(self) -> str:
        """Stable identifier used in reports, e.g. ``pk.patients.Id``."""
        return f"pk.{self.table}.{self.column}"


@dataclass(frozen=True, slots=True)
class ForeignKeyRule:
    """A column documented as a foreign key to another table's key."""

    table: str
    column: str
    parent_table: str
    parent_column: str = "Id"
    confirmed_by: str = DATA_DICTIONARY

    @property
    def check_id(self) -> str:
        """Stable identifier, e.g. ``fk.conditions.PATIENT->patients.Id``."""
        return f"fk.{self.table}.{self.column}->{self.parent_table}.{self.parent_column}"


@dataclass(frozen=True, slots=True)
class UnresolvedForeignKey:
    """A documented relationship that is not applied as a constraint yet.

    Kept separate from :data:`FOREIGN_KEYS` on purpose: the relationship is not
    declared wrong, it is simply not safe to enforce, so no ``FAIL`` is produced
    for it until a maintainer confirms the intended semantics.

    The five things a reader must be able to tell apart are stored apart, and the
    quantitative evidence is attributed to the dataset it came from:
    ``documented_as`` (documentation), ``implemented_as`` (generator code),
    ``reference_dataset`` + ``reference_evidence`` (a measurement on that dataset,
    never on the dataset a report is about), ``why_not_enforced`` and ``pending``.
    """

    rule: ForeignKeyRule
    #: What Synthea's own documentation states about the relationship.
    documented_as: str
    #: What the generator source does.
    implemented_as: str
    #: Which dataset produced ``reference_evidence``.
    reference_dataset: str
    #: The measurement made on ``reference_dataset``, not on the reported dataset.
    reference_evidence: str
    #: Why this tool does not turn the above into a failure.
    why_not_enforced: str
    #: What is still needed before it can become a rule.
    pending: str


#: Columns the data dictionary marks as primary keys (``:key:``).
PRIMARY_KEYS: tuple[PrimaryKeyRule, ...] = (
    PrimaryKeyRule("careplans", "Id"),
    PrimaryKeyRule("claims", "Id"),
    PrimaryKeyRule("claims_transactions", "ID"),
    PrimaryKeyRule("encounters", "Id"),
    PrimaryKeyRule("organizations", "Id"),
    PrimaryKeyRule("patients", "Id"),
    PrimaryKeyRule("payers", "Id"),
    PrimaryKeyRule("providers", "Id"),
)

#: Columns the data dictionary marks as foreign keys (``:old_key:``).
FOREIGN_KEYS: tuple[ForeignKeyRule, ...] = (
    ForeignKeyRule("allergies", "PATIENT", "patients"),
    ForeignKeyRule("allergies", "ENCOUNTER", "encounters"),
    ForeignKeyRule("careplans", "PATIENT", "patients"),
    ForeignKeyRule("careplans", "ENCOUNTER", "encounters"),
    ForeignKeyRule("claims", "PATIENTID", "patients"),
    ForeignKeyRule("claims", "PROVIDERID", "providers"),
    ForeignKeyRule("claims", "PRIMARYPATIENTINSURANCEID", "payers"),
    ForeignKeyRule("claims", "SECONDARYPATIENTINSURANCEID", "payers"),
    ForeignKeyRule("claims", "REFERRINGPROVIDERID", "providers"),
    ForeignKeyRule("claims", "APPOINTMENTID", "encounters"),
    ForeignKeyRule("claims", "SUPERVISINGPROVIDERID", "providers"),
    ForeignKeyRule("claims_transactions", "CLAIMID", "claims"),
    ForeignKeyRule("claims_transactions", "PATIENTID", "patients"),
    ForeignKeyRule("claims_transactions", "PLACEOFSERVICE", "organizations"),
    ForeignKeyRule("claims_transactions", "APPOINTMENTID", "encounters"),
    ForeignKeyRule("claims_transactions", "PROVIDERID", "providers"),
    ForeignKeyRule("claims_transactions", "SUPERVISINGPROVIDERID", "providers"),
    ForeignKeyRule("conditions", "PATIENT", "patients"),
    ForeignKeyRule("conditions", "ENCOUNTER", "encounters"),
    ForeignKeyRule("devices", "PATIENT", "patients"),
    ForeignKeyRule("devices", "ENCOUNTER", "encounters"),
    ForeignKeyRule("encounters", "PATIENT", "patients"),
    ForeignKeyRule("encounters", "ORGANIZATION", "organizations"),
    ForeignKeyRule("encounters", "PROVIDER", "providers"),
    ForeignKeyRule("encounters", "PAYER", "payers"),
    ForeignKeyRule("imaging_studies", "PATIENT", "patients"),
    ForeignKeyRule("imaging_studies", "ENCOUNTER", "encounters"),
    ForeignKeyRule("immunizations", "PATIENT", "patients"),
    ForeignKeyRule("immunizations", "ENCOUNTER", "encounters"),
    ForeignKeyRule("medications", "PATIENT", "patients"),
    ForeignKeyRule("medications", "PAYER", "payers"),
    ForeignKeyRule("medications", "ENCOUNTER", "encounters"),
    ForeignKeyRule("observations", "PATIENT", "patients"),
    ForeignKeyRule("observations", "ENCOUNTER", "encounters"),
    ForeignKeyRule("payer_transitions", "PATIENT", "patients"),
    ForeignKeyRule("payer_transitions", "PAYER", "payers"),
    ForeignKeyRule("payer_transitions", "SECONDARY_PAYER", "payers"),
    ForeignKeyRule("procedures", "PATIENT", "patients"),
    ForeignKeyRule("procedures", "ENCOUNTER", "encounters"),
    ForeignKeyRule("providers", "ORGANIZATION", "organizations"),
    ForeignKeyRule("supplies", "PATIENT", "patients"),
    ForeignKeyRule("supplies", "ENCOUNTER", "encounters"),
)

#: Documented relationships that are **not applied** yet (evidence in each entry).
#:
#: These are not declared invalid: the documentation states the relationship, the
#: data shows it does not hold as a strict constraint, and the pending item is a
#: maintainer's confirmation. Until then they produce no verdict at all.
UNRESOLVED_FOREIGN_KEYS: tuple[UnresolvedForeignKey, ...] = (
    UnresolvedForeignKey(
        rule=ForeignKeyRule(
            "claims_transactions", "PATIENTINSURANCEID", "payer_transitions", "MEMBERID"
        ),
        documented_as=(
            "CSV File Data Dictionary, claims_transactions: 'Patient Insurance ID ... "
            "Foreign key to the Payer Transitions table member ID'"
        ),
        implemented_as=(
            "CSVExporter.java:1566 sets 'this.memberId = claim.getPlanRecordMemberId()' and "
            "line 1686 writes it to PATIENTINSURANCEID, while payer_transitions.MEMBERID is "
            "produced by a separate export that line 274 calls with a cutoff of 0L "
            "(1970-01-01): line 173 keeps only the plans whose stop time is on or after it"
        ),
        reference_dataset=REFERENCE_DATASET,
        reference_evidence=(
            "170 of 79,453 non-null references (5 distinct values, belonging to 3 patients) "
            "match no payer_transitions.MEMBERID, all dated before 1970 (FROMDATE 1942-11-26 "
            "to 1961-08-25): 149 belong to a patient with no payer_transitions row (born 1960, "
            "died 1961) and 21 to two patients whose rows start in 1969"
        ),
        why_not_enforced=(
            "applying it as a strict foreign key would emit FAIL for data the generator "
            "produces on purpose, and a false alarm is worse than a missing check"
        ),
        pending=(
            "a maintainer's confirmation of whether PATIENTINSURANCEID is meant to reference "
            "payer_transitions.MEMBERID (reported as synthetichealth/synthea#1725); until "
            "then the relationship is neither enforced nor declared wrong"
        ),
    ),
)


def primary_key_for(table: str) -> PrimaryKeyRule | None:
    """Return the primary key rule of ``table``, or ``None`` if it has none."""
    return next((rule for rule in PRIMARY_KEYS if rule.table == table), None)


def foreign_keys_for(table: str) -> tuple[ForeignKeyRule, ...]:
    """Return the foreign key rules declared on ``table``."""
    return tuple(rule for rule in FOREIGN_KEYS if rule.table == table)


def parent_columns() -> tuple[tuple[str, str], ...]:
    """Distinct ``(table, column)`` pairs used as foreign key targets, sorted."""
    return tuple(sorted({(rule.parent_table, rule.parent_column) for rule in FOREIGN_KEYS}))


class _CatalogueError(ValueError):
    """A key rule points at something the table catalogue does not declare."""


def _validate_catalogue() -> None:
    """Fail loudly at import time if a rule contradicts the table catalogue."""
    known = tables_by_name()
    seen: set[tuple[str, str, str]] = set()

    for rule in PRIMARY_KEYS:
        _check_rule(known, rule.table, rule.column, "primary key")
        _check_unique(seen, "pk", rule.table, rule.column)

    for rule in FOREIGN_KEYS:
        _check_rule(known, rule.table, rule.column, "foreign key")
        _check_rule(known, rule.parent_table, rule.parent_column, "foreign key target")
        _check_unique(seen, "fk", rule.table, rule.column)

    for unresolved in UNRESOLVED_FOREIGN_KEYS:
        _check_rule(known, unresolved.rule.table, unresolved.rule.column, "unresolved rule")
        _check_rule(
            known,
            unresolved.rule.parent_table,
            unresolved.rule.parent_column,
            "unresolved rule target",
        )
        for field_name in (
            "documented_as",
            "implemented_as",
            "reference_dataset",
            "reference_evidence",
            "why_not_enforced",
            "pending",
        ):
            if not getattr(unresolved, field_name).strip():
                raise _CatalogueError(
                    f"unresolved rule {unresolved.rule.check_id} needs a '{field_name}' note"
                )


def _check_rule(
    known: Mapping[str, TableSpec], table: str, column: str, role: str
) -> None:
    spec = known.get(table)
    if spec is None:
        raise _CatalogueError(f"{role} refers to unknown table '{table}'")
    if column not in spec.columns:
        raise _CatalogueError(
            f"{role} refers to column '{column}', which is not part of table '{table}'"
        )


def _check_unique(
    seen: set[tuple[str, str, str]], kind: str, table: str, column: str
) -> None:
    key = (kind, table, column)
    if key in seen:
        raise _CatalogueError(f"duplicate {kind} rule for {table}.{column}")
    seen.add(key)


_validate_catalogue()
