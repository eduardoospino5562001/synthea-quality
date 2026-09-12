"""[REPORTING] Aggregate one dataset into a single structured report.

This is the only reporter that reads the dataset. It discovers the tables, reads
their headers, compares them with the versioned contract, validates that every row has
as many fields as its header, runs the four confirmed check families and collects
everything into a :class:`~synthea_quality.models.DatasetReport`.

Three deliberate choices:

* the check families receive the discovery result and the structural validation this
  module already computed, so the directory and the row shapes are inspected once;
* a report never invents a verdict for something that could not be read: tables
  whose header cannot be read are recorded in ``load_errors``, and the checks
  themselves report the tables they had to skip;
* a directory with no known table at all is an input error
  (:class:`~synthea_quality.errors.EmptyDatasetError`), not a report of skipped
  checks that would read as "nothing was wrong".

Table row counts in the report come from the quality checks, which load every
present table anyway; nothing is loaded a second time just to fill a number.
"""

from __future__ import annotations

from pathlib import Path

from synthea_quality.checks.keys import run_key_checks
from synthea_quality.checks.quality import run_quality_checks
from synthea_quality.checks.temporal import run_temporal_checks
from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import EmptyDatasetError, TableLoadError
from synthea_quality.loader import read_header
from synthea_quality.models import (
    DEFAULT_SAMPLE_LIMIT,
    DatasetReport,
    LoadError,
    TableSummary,
    UnresolvedRelation,
)
from synthea_quality.schema.contract import assess_contract
from synthea_quality.schema.keys import UNRESOLVED_FOREIGN_KEYS
from synthea_quality.schema.tables import SYNTHEA_TABLES
from synthea_quality.schema.temporal import UNRESOLVED_TEMPORAL_RELATIONS
from synthea_quality.structure import validate_tables


def build_report(
    data_dir: str | Path,
    *,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    discovery: DiscoveryResult | None = None,
    generated_at: str | None = None,
) -> DatasetReport:
    """Run every check over ``data_dir`` and aggregate the results.

    :param sample_limit: how many offending items each check keeps as a sample.
    :param discovery: reuse an existing discovery result instead of inspecting the
        directory again.
    :param generated_at: timestamp to record; the current UTC time by default.
    :raises EmptyDatasetError: the directory holds none of the tables of the contract,
        so there would be nothing to report about.
    """
    data_path = Path(data_dir)
    found = discovery if discovery is not None else discover_dataset(data_path)

    if not found.tables:
        raise EmptyDatasetError(
            f"no Synthea CSV table was found in {data_path}: a dataset directory has to "
            f"hold at least one of the {len(SYNTHEA_TABLES)} tables the schema contract "
            f"describes (for example patients.csv or encounters.csv)"
        )

    headers: dict[str, tuple[str, ...]] = {}
    load_errors: list[LoadError] = []
    for table in found.tables:
        try:
            headers[table.name] = read_header(table.path)
        except TableLoadError as exc:
            load_errors.append(LoadError(table=table.name, reason=str(exc)))

    assessment = assess_contract(headers)
    schema_match = {match.table: match.is_exact for match in assessment.matches}

    # Row shapes are validated once, before any check reads a table: a table whose rows
    # do not line up with its header cannot have its values attributed to its columns.
    structures = validate_tables(
        {table.name: table.path for table in found.tables}, sample_limit=sample_limit
    )

    checks = tuple(
        sorted(
            (
                *run_key_checks(
                    data_path,
                    discovery=found,
                    structure=structures,
                    sample_limit=sample_limit,
                ),
                *run_quality_checks(
                    data_path,
                    discovery=found,
                    structure=structures,
                    sample_limit=sample_limit,
                ),
                *run_temporal_checks(
                    data_path,
                    discovery=found,
                    structure=structures,
                    sample_limit=sample_limit,
                ),
                # One finding per malformed table, not one per check that had to give up.
                *(
                    structure.to_check_result()
                    for structure in structures.values()
                    if not structure.ok
                ),
            ),
            key=lambda check: check.check_id,
        )
    )

    tables = tuple(
        _table_summary(table, headers.get(table.name), schema_match.get(table.name), checks)
        for table in found.tables
    )

    report = DatasetReport(
        data_dir=str(data_path),
        tables=tables,
        checks=checks,
        unknown_files=found.unknown_csv_files,
        missing_known_tables=tuple(spec.name for spec in found.missing_tables),
        schema_contract=found.contract_id or assessment.contract_id,
        contract_tables=assessment.contract_table_count,
        contract_status=assessment.status,
        contract_summary=assessment.summary,
        contract_findings=assessment.reasons,
        load_errors=tuple(load_errors),
        unresolved_relations=unresolved_relations(),
    )
    if generated_at is not None:
        report = _with_timestamp(report, generated_at)
    return report


def unresolved_relations() -> tuple[UnresolvedRelation, ...]:
    """Documented relationships the tool does not enforce, with their evidence.

    They are part of every report so a reader can see what was considered and left
    out, and so that their absence never looks like an oversight.
    """
    foreign_keys = tuple(
        UnresolvedRelation(
            kind="foreign_key",
            relation=entry.rule.check_id,
            documented_as=entry.documented_as,
            implemented_as=entry.implemented_as,
            reference_dataset=entry.reference_dataset,
            reference_evidence=entry.reference_evidence,
            why_not_enforced=entry.why_not_enforced,
            pending=entry.pending,
        )
        for entry in UNRESOLVED_FOREIGN_KEYS
    )
    temporal = tuple(
        UnresolvedRelation(
            kind="temporal",
            relation=entry.relation,
            documented_as=entry.documented_as,
            implemented_as=entry.implemented_as,
            reference_dataset=entry.reference_dataset,
            reference_evidence=entry.reference_evidence,
            why_not_enforced=entry.why_not_enforced,
            pending=entry.pending,
        )
        for entry in UNRESOLVED_TEMPORAL_RELATIONS
    )
    return foreign_keys + temporal


def _table_summary(
    table,
    header: tuple[str, ...] | None,
    schema_match: bool | None,
    checks,
) -> TableSummary:
    """Describe one table, taking its row count from the check that already loaded it."""
    rows = next(
        (
            int(check.metrics["rows"])
            for check in checks
            if check.check_id == f"nulls.{table.name}" and "rows" in check.metrics
        ),
        None,
    )
    return TableSummary(
        name=table.name,
        file_name=table.file_name,
        rows=rows,
        columns=header or (),
        schema_match=schema_match,
    )


def _with_timestamp(report: DatasetReport, generated_at: str) -> DatasetReport:
    """Return the same report with a given timestamp (used by tests and callers)."""
    return DatasetReport(
        data_dir=report.data_dir,
        tables=report.tables,
        checks=report.checks,
        unknown_files=report.unknown_files,
        missing_known_tables=report.missing_known_tables,
        schema_contract=report.schema_contract,
        generated_at=generated_at,
        tool_version=report.tool_version,
        schema_version=report.schema_version,
        contract_tables=report.contract_tables,
        contract_status=report.contract_status,
        contract_summary=report.contract_summary,
        contract_findings=report.contract_findings,
        load_errors=report.load_errors,
        unresolved_relations=report.unresolved_relations,
    )
