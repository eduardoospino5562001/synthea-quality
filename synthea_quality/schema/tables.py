"""Known Synthea CSV table names for the current schema contract.

This module holds the *table catalogue* only (which files Synthea writes). The
per-column contracts live next to it, added when the schema checks are built.

Every fact here is derived from the Synthea source, not guessed:

* Source: ``src/main/java/org/mitre/synthea/export/CSVConstants.java`` in
  ``synthetichealth/synthea`` — the class declares one ``*_KEY`` constant per
  table (19 of them), and each table is written as ``<key>.csv``.
* Commit used as the reference for the contract below:
  ``d9d07a6eef91ee5144293b42ab64224d84d124f8`` (master, 2026-08-17).
* Cross-checked against the official CSV sample published by the project
  (``synthea-sample-data/downloads/latest``, generated from the same commit):
  the 18 files present there match these names exactly.

**No table is mandatory.** The Synthea CSV exporter decides which files to write
from the ``exporter.csv.included_files`` / ``exporter.csv.excluded_files``
settings, so any table can legitimately be absent from a dataset. Discovery must
therefore never treat a missing table as an error. The only file excluded by
default in Synthea's own configuration is ``patient_expenses.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Identifier of the schema contract described by this module.
CURRENT_CONTRACT_ID = "synthea-csv-2026-08"

#: Where the table catalogue comes from (recorded for provenance).
SOURCE_REPOSITORY = "synthetichealth/synthea"
SOURCE_COMMIT = "d9d07a6eef91ee5144293b42ab64224d84d124f8"
SOURCE_FILE = "src/main/java/org/mitre/synthea/export/CSVConstants.java"


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One table of the Synthea CSV export."""

    name: str
    file_name: str
    #: False for files Synthea excludes by default (``exporter.csv.excluded_files``).
    included_by_default: bool = True

    @property
    def is_optional(self) -> bool:
        """Whether Synthea leaves this file out of a default CSV export."""
        return not self.included_by_default


#: All tables declared by ``CSVConstants``, in the order of the source file.
SYNTHEA_TABLES: tuple[TableSpec, ...] = (
    TableSpec("patients", "patients.csv"),
    TableSpec("allergies", "allergies.csv"),
    TableSpec("medications", "medications.csv"),
    TableSpec("conditions", "conditions.csv"),
    TableSpec("careplans", "careplans.csv"),
    TableSpec("observations", "observations.csv"),
    TableSpec("procedures", "procedures.csv"),
    TableSpec("immunizations", "immunizations.csv"),
    TableSpec("encounters", "encounters.csv"),
    TableSpec("imaging_studies", "imaging_studies.csv"),
    TableSpec("devices", "devices.csv"),
    TableSpec("supplies", "supplies.csv"),
    TableSpec("organizations", "organizations.csv"),
    TableSpec("providers", "providers.csv"),
    TableSpec("payers", "payers.csv"),
    TableSpec("payer_transitions", "payer_transitions.csv"),
    TableSpec("claims", "claims.csv"),
    TableSpec("claims_transactions", "claims_transactions.csv"),
    TableSpec("patient_expenses", "patient_expenses.csv", included_by_default=False),
)


def tables_by_name() -> dict[str, TableSpec]:
    """Return the catalogue indexed by table name."""
    return {spec.name: spec for spec in SYNTHEA_TABLES}
