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
    The dictionary calls it a foreign key to the payer transition member id, but
    it cannot be enforced as one. On the official sample 170 references (5 distinct
    values, belonging to 3 patients) match no ``payer_transitions.MEMBERID``, and
    those 3 patients have no rows in ``payer_transitions`` at all: the column is
    written from the claim's plan record (``CSVExporter.java``:
    ``this.memberId = claim.getPlanRecordMemberId()``), while
    ``payer_transitions.MEMBERID`` is an independent export. Enforcing it would
    report legitimate data as broken, so it is listed in
    ``REJECTED_FOREIGN_KEYS`` with the reason and left for the maintainers.

The catalogue is validated on import against the generated table catalogue: a rule
pointing at a table or column that does not exist is a bug here, not a dataset
defect, so it raises immediately instead of producing a wrong verdict later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from synthea_quality.schema.tables import TableSpec, tables_by_name

#: Provenance recorded on every rule derived from Synthea's own documentation.
DATA_DICTIONARY = "Synthea CSV File Data Dictionary (:key: / :old_key:)"

#: Documented relationship that is intentionally not enforced. See module docstring.
REJECTION_REASON = (
    "not enforceable as a constraint: on the official 2026-08 sample 170 references "
    "(5 values, 3 patients) match no payer_transitions.MEMBERID, and those patients "
    "have no payer_transitions rows; CSVExporter.java writes the column from "
    "claim.getPlanRecordMemberId(), a different source than payer_transitions.MEMBERID"
)


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
class RejectedForeignKey:
    """A documented relationship that this tool refuses to enforce, with the reason."""

    rule: ForeignKeyRule
    reason: str


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

#: Documented relationship that is deliberately not enforced (evidence in the reason).
REJECTED_FOREIGN_KEYS: tuple[RejectedForeignKey, ...] = (
    RejectedForeignKey(
        ForeignKeyRule(
            "claims_transactions", "PATIENTINSURANCEID", "payer_transitions", "MEMBERID"
        ),
        REJECTION_REASON,
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

    for rejected in REJECTED_FOREIGN_KEYS:
        if not rejected.reason.strip():
            raise _CatalogueError(f"rejected rule {rejected.rule.check_id} needs a reason")


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
