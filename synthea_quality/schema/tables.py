"""[SCHEMA] Known Synthea CSV tables and their columns, for the current contract.

Every fact in this module is derived from the Synthea source, not guessed:

* Source: ``src/main/java/org/mitre/synthea/export/CSVConstants.java`` in
  ``synthetichealth/synthea``. That class declares one ``*_KEY`` constant per
  table (19 of them, each written as ``<key>.csv``) and one ``*_HEADER_LINE``
  constant holding the exact CSV header it writes.
* Reference commit: ``d9d07a6eef91ee5144293b42ab64224d84d124f8`` (master,
  2026-08-17).
* Cross-checked against the official CSV sample published by the project
  (``synthea-sample-data/downloads/latest``, generated from the same commit):
  the 18 files present there match these names and columns exactly.

The ``columns`` tuples below are the expected header of each table, **in the
order Synthea writes it**. They were extracted mechanically from that source file
rather than typed by hand, because a typo in a contract would produce a false
positive on every dataset.

**No table is mandatory.** The exporter decides which files to write from the
``exporter.csv.included_files`` / ``exporter.csv.excluded_files`` settings, so a
table can legitimately be absent from a dataset, and its absence says nothing
about whether the dataset matches this contract. The only file excluded by
default in Synthea's own configuration is ``patient_expenses.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Identifier of the schema contract described by this module.
CONTRACT_ID = "synthea-csv-2026-08"

#: Where the table catalogue comes from (recorded for provenance).
SOURCE_REPOSITORY = "synthetichealth/synthea"
SOURCE_COMMIT = "d9d07a6eef91ee5144293b42ab64224d84d124f8"
SOURCE_FILE = "src/main/java/org/mitre/synthea/export/CSVConstants.java"


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One table of the Synthea CSV export: file, expected columns, default status."""

    name: str
    file_name: str
    columns: tuple[str, ...]
    #: False for files Synthea excludes by default (``exporter.csv.excluded_files``).
    included_by_default: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.file_name:
            raise ValueError("name and file_name must be non-empty")
        if not self.columns:
            raise ValueError(f"table {self.name!r} must declare its columns")
        duplicates = sorted({c for c in self.columns if self.columns.count(c) > 1})
        if duplicates:
            raise ValueError(f"table {self.name!r} declares duplicate columns: {duplicates}")

    @property
    def is_optional(self) -> bool:
        """Whether Synthea leaves this file out of a default CSV export."""
        return not self.included_by_default


#: All tables declared by ``CSVConstants``, in the order of the source file.
SYNTHEA_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        'patients',
        'patients.csv',
        columns=('Id', 'BIRTHDATE', 'DEATHDATE', 'SSN', 'DRIVERS', 'PASSPORT', 'PREFIX', 'FIRST', 'MIDDLE', 'LAST', 'SUFFIX', 'MAIDEN', 'MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'BIRTHPLACE', 'ADDRESS', 'CITY', 'STATE', 'COUNTY', 'FIPS', 'ZIP', 'LAT', 'LON', 'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',),
    ),
    TableSpec(
        'allergies',
        'allergies.csv',
        columns=('START', 'STOP', 'PATIENT', 'ENCOUNTER', 'CODE', 'SYSTEM', 'DESCRIPTION', 'TYPE', 'CATEGORY', 'REACTION1', 'DESCRIPTION1', 'SEVERITY1', 'REACTION2', 'DESCRIPTION2', 'SEVERITY2',),
    ),
    TableSpec(
        'medications',
        'medications.csv',
        columns=('START', 'STOP', 'PATIENT', 'PAYER', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'BASE_COST', 'PAYER_COVERAGE', 'DISPENSES', 'TOTALCOST', 'REASONCODE', 'REASONDESCRIPTION',),
    ),
    TableSpec(
        'conditions',
        'conditions.csv',
        columns=('START', 'STOP', 'PATIENT', 'ENCOUNTER', 'SYSTEM', 'CODE', 'DESCRIPTION',),
    ),
    TableSpec(
        'careplans',
        'careplans.csv',
        columns=('Id', 'START', 'STOP', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'REASONCODE', 'REASONDESCRIPTION',),
    ),
    TableSpec(
        'observations',
        'observations.csv',
        columns=('DATE', 'PATIENT', 'ENCOUNTER', 'CATEGORY', 'CODE', 'DESCRIPTION', 'VALUE', 'UNITS', 'TYPE',),
    ),
    TableSpec(
        'procedures',
        'procedures.csv',
        columns=('START', 'STOP', 'PATIENT', 'ENCOUNTER', 'SYSTEM', 'CODE', 'DESCRIPTION', 'BASE_COST', 'REASONCODE', 'REASONDESCRIPTION',),
    ),
    TableSpec(
        'immunizations',
        'immunizations.csv',
        columns=('DATE', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'BASE_COST',),
    ),
    TableSpec(
        'encounters',
        'encounters.csv',
        columns=('Id', 'START', 'STOP', 'PATIENT', 'ORGANIZATION', 'PROVIDER', 'PAYER', 'ENCOUNTERCLASS', 'CODE', 'DESCRIPTION', 'BASE_ENCOUNTER_COST', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE', 'REASONCODE', 'REASONDESCRIPTION',),
    ),
    TableSpec(
        'imaging_studies',
        'imaging_studies.csv',
        columns=('Id', 'DATE', 'PATIENT', 'ENCOUNTER', 'SERIES_UID', 'BODYSITE_CODE', 'BODYSITE_DESCRIPTION', 'MODALITY_CODE', 'MODALITY_DESCRIPTION', 'INSTANCE_UID', 'SOP_CODE', 'SOP_DESCRIPTION', 'PROCEDURE_CODE',),
    ),
    TableSpec(
        'devices',
        'devices.csv',
        columns=('START', 'STOP', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'UDI',),
    ),
    TableSpec(
        'supplies',
        'supplies.csv',
        columns=('DATE', 'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'QUANTITY',),
    ),
    TableSpec(
        'organizations',
        'organizations.csv',
        columns=('Id', 'NAME', 'ADDRESS', 'CITY', 'STATE', 'ZIP', 'LAT', 'LON', 'PHONE', 'REVENUE', 'UTILIZATION', 'NPI',),
    ),
    TableSpec(
        'providers',
        'providers.csv',
        columns=('Id', 'ORGANIZATION', 'NAME', 'GENDER', 'SPECIALITY', 'ADDRESS', 'CITY', 'STATE', 'ZIP', 'LAT', 'LON', 'ENCOUNTERS', 'PROCEDURES', 'NPI',),
    ),
    TableSpec(
        'payers',
        'payers.csv',
        columns=('Id', 'NAME', 'OWNERSHIP', 'ADDRESS', 'CITY', 'STATE_HEADQUARTERED', 'ZIP', 'PHONE', 'AMOUNT_COVERED', 'AMOUNT_UNCOVERED', 'REVENUE', 'COVERED_ENCOUNTERS', 'UNCOVERED_ENCOUNTERS', 'COVERED_MEDICATIONS', 'UNCOVERED_MEDICATIONS', 'COVERED_PROCEDURES', 'UNCOVERED_PROCEDURES', 'COVERED_IMMUNIZATIONS', 'UNCOVERED_IMMUNIZATIONS', 'UNIQUE_CUSTOMERS', 'QOLS_AVG', 'MEMBER_MONTHS',),
    ),
    TableSpec(
        'payer_transitions',
        'payer_transitions.csv',
        columns=('PATIENT', 'MEMBERID', 'START_DATE', 'END_DATE', 'PAYER', 'SECONDARY_PAYER', 'PLAN_OWNERSHIP', 'OWNER_NAME',),
    ),
    TableSpec(
        'claims',
        'claims.csv',
        columns=('Id', 'PATIENTID', 'PROVIDERID', 'PRIMARYPATIENTINSURANCEID', 'SECONDARYPATIENTINSURANCEID', 'DEPARTMENTID', 'PATIENTDEPARTMENTID', 'DIAGNOSIS1', 'DIAGNOSIS2', 'DIAGNOSIS3', 'DIAGNOSIS4', 'DIAGNOSIS5', 'DIAGNOSIS6', 'DIAGNOSIS7', 'DIAGNOSIS8', 'REFERRINGPROVIDERID', 'APPOINTMENTID', 'CURRENTILLNESSDATE', 'SERVICEDATE', 'SUPERVISINGPROVIDERID', 'STATUS1', 'STATUS2', 'STATUSP', 'OUTSTANDING1', 'OUTSTANDING2', 'OUTSTANDINGP', 'LASTBILLEDDATE1', 'LASTBILLEDDATE2', 'LASTBILLEDDATEP', 'HEALTHCARECLAIMTYPEID1', 'HEALTHCARECLAIMTYPEID2',),
    ),
    TableSpec(
        'claims_transactions',
        'claims_transactions.csv',
        columns=('ID', 'CLAIMID', 'CHARGEID', 'PATIENTID', 'TYPE', 'AMOUNT', 'METHOD', 'FROMDATE', 'TODATE', 'PLACEOFSERVICE', 'PROCEDURECODE', 'MODIFIER1', 'MODIFIER2', 'DIAGNOSISREF1', 'DIAGNOSISREF2', 'DIAGNOSISREF3', 'DIAGNOSISREF4', 'UNITS', 'DEPARTMENTID', 'NOTES', 'UNITAMOUNT', 'TRANSFEROUTID', 'TRANSFERTYPE', 'PAYMENTS', 'ADJUSTMENTS', 'TRANSFERS', 'OUTSTANDING', 'APPOINTMENTID', 'LINENOTE', 'PATIENTINSURANCEID', 'FEESCHEDULEID', 'PROVIDERID', 'SUPERVISINGPROVIDERID',),
    ),
    TableSpec(
        'patient_expenses',
        'patient_expenses.csv',
        columns=('PATIENT_ID', 'YEAR', 'PAYER_ID', 'HEALTHCARE_EXPENSES', 'INSURANCE_COSTS', 'COVERED_COSTS',),
        included_by_default=False,
    ),
)


def tables_by_name() -> dict[str, TableSpec]:
    """Return the catalogue indexed by table name."""
    return {spec.name: spec for spec in SYNTHEA_TABLES}
