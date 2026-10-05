"""Structured result models for dataset quality checks.

These models are the contract between the check logic and the report writers:
checks produce :class:`CheckResult` objects, the runner collects them into a
:class:`DatasetReport`, and the report module renders that report as Markdown or
JSON. Nothing in this module knows about file formats or about how results are
displayed.

Design notes
------------
``severity`` describes the impact a check *would* have if it failed; it is a
property of the check itself, not of the observed outcome. ``status`` is the
observed outcome. Keeping those two concepts apart avoids inventing arbitrary
"how bad is this failure" ratings at run time.

The JSON layout of a report is versioned through ``REPORT_SCHEMA_VERSION`` so
that downstream consumers can rely on it. Adding optional keys is backwards
compatible; changing or removing keys requires a new version.

Only the standard library is imported here: importing the models must stay cheap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from synthea_quality import __version__
from synthea_quality.schema.contract import ContractStatus

#: Version of the JSON layout produced by :meth:`DatasetReport.to_dict`.
REPORT_SCHEMA_VERSION = 1

#: Default number of offending items kept in ``CheckResult.samples``.
#: A report must never embed millions of identifiers, so every check takes a
#: (configurable) limit and reports the full count in ``metrics`` instead.
DEFAULT_SAMPLE_LIMIT = 5


class Status(str, Enum):  # noqa: UP042 - keep (str, Enum); StrEnum needs a separate decision
    """Outcome of a single check.

    ``PASS``            the dataset satisfies the check.
    ``WARNING``         something worth a human look, but legitimate in Synthea
                        data (for example a null in an optional column).
    ``FAIL``            a deterministic contract violation (duplicate primary
                        key, orphan foreign key, missing required column).
    ``NOT_APPLICABLE``  the check cannot apply to this dataset by definition
                        (for example a foreign key whose target table is absent).
    ``SKIPPED``         the check was not run in this execution (no confirmed
                        rule for this table, or a prerequisite failed).
    ``ERROR``           the *tool* could not complete the check. This signals a
                        problem with the tool or its inputs, never a data defect.
    """

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


class Severity(str, Enum):  # noqa: UP042 - keep (str, Enum); StrEnum needs a separate decision
    """Impact a check would have if it failed.

    ``HIGH``    breaks the relational contract or the meaning of a whole table:
                missing schema columns, duplicate or null primary keys, orphan
                foreign keys. A failure here means joins or a table's contents
                cannot be trusted.
    ``MEDIUM``  row-level inconsistencies that do not break joins: inverted date
                intervals, birth after death, a value in a documented date column
                that matches no date format, missing required values, exact
                duplicate rows.
    ``LOW``     anomalies that are informative but do not corrupt data:
                unexpected extra files or columns, nulls in optional columns,
                a column that is empty for every row, empty tables.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def rank(self) -> int:
        """Numeric rank used to pick the most severe finding (HIGH > MEDIUM > LOW)."""
        return _SEVERITY_RANK[self]

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
}


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


#: Categories a check belongs to, in the order a report should present them.
#: The category is derived from the check identifier, so adding a check to an
#: existing family needs no change here.
CHECK_CATEGORIES: tuple[str, ...] = (
    "schema",
    "primary_keys",
    "foreign_keys",
    "data_quality",
    "temporal",
    "other",
)

_CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("schema.", "schema"),
    ("pk.", "primary_keys"),
    ("fk.", "foreign_keys"),
    ("dates.", "data_quality"),
    ("duplicates.", "data_quality"),
    ("empty_columns.", "data_quality"),
    ("nulls.", "data_quality"),
    ("structure.", "schema"),
    ("temporal.", "temporal"),
)


def category_of(check_id: str) -> str:
    """Category a check belongs to, or ``other`` for an unrecognised identifier."""
    for prefix, category in _CATEGORY_PREFIXES:
        if check_id.startswith(prefix):
            return category
    return "other"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a single check, structured and JSON-serialisable.

    ``check_id`` is a stable identifier (``pk.patients.Id``,
    ``fk.conditions.PATIENT->patients.Id``, ``temporal.start_le_stop.conditions``).
    ``metrics`` holds the numbers behind the verdict; ``samples`` holds a bounded
    list of offending rows; ``metadata`` documents the rule that was applied
    (columns used, thresholds, schema contract version).
    """

    check_id: str
    status: Status
    severity: Severity
    message: str
    table: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    samples: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id.strip():
            raise ValueError("check_id must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        if not isinstance(self.status, Status):
            raise TypeError(f"status must be a Status value, got {type(self.status).__name__}")
        if not isinstance(self.severity, Severity):
            raise TypeError(
                f"severity must be a Severity value, got {type(self.severity).__name__}"
            )
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple of mappings")
        for sample in self.samples:
            if not isinstance(sample, Mapping):
                raise TypeError("each sample must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation of this result."""
        return {
            "check_id": self.check_id,
            "table": self.table,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "metrics": dict(self.metrics),
            "samples": [dict(sample) for sample in self.samples],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckResult:
        """Rebuild a result from :meth:`to_dict` output.

        Unknown keys are ignored so that a reader keeps working if optional keys
        are added within the same report schema version. Unknown ``status`` or
        ``severity`` values are a hard error instead of being silently dropped.
        """
        return cls(
            check_id=data["check_id"],
            status=Status(data["status"]),
            severity=Severity(data["severity"]),
            message=data["message"],
            table=data.get("table"),
            metrics=dict(data.get("metrics") or {}),
            samples=tuple(dict(sample) for sample in (data.get("samples") or ())),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class TableSummary:
    """What is known about one table of the dataset.

    ``rows`` and ``columns`` are unknown (``None`` / empty) until the table has
    been loaded, so a summary can also describe a table that failed to load.
    """

    name: str
    file_name: str
    rows: int | None = None
    columns: tuple[str, ...] = ()
    schema_match: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.file_name, str) or not self.file_name.strip():
            raise ValueError("file_name must be a non-empty string")
        if self.rows is not None and self.rows < 0:
            raise ValueError("rows cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file_name": self.file_name,
            "rows": self.rows,
            "columns": list(self.columns),
            "schema_match": self.schema_match,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TableSummary:
        return cls(
            name=data["name"],
            file_name=data["file_name"],
            rows=data.get("rows"),
            columns=tuple(data.get("columns") or ()),
            schema_match=data.get("schema_match"),
        )


@dataclass(frozen=True, slots=True)
class LoadError:
    """A table the tool could not read at all.

    Only reading failures detected while inspecting the dataset are listed here; a
    table whose content turns out to be malformed when a check loads it appears as
    ``SKIPPED`` in that check, with the reason in its message.
    """

    table: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.table, str) or not self.table.strip():
            raise ValueError("table must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LoadError:
        return cls(table=data["table"], reason=data["reason"])


@dataclass(frozen=True, slots=True)
class AnomalousEntry:
    """A path whose name could be a table but which cannot be read as one.

    Recorded rather than dropped: without it, a table name that resolves to a directory,
    a broken symbolic link or anything else that is not a regular file is
    indistinguishable from a file the generator never wrote, and a reader is left to
    guess. It is **not** a dataset defect — the table still counts as missing, and no
    check fails because of it — so it stays out of ``load_errors``, which drives the exit
    code.
    """

    #: Name found on disk, exactly as written (may differ in case from the catalogue).
    file_name: str
    #: Readable explanation of why the name produced no table.
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.file_name, str) or not self.file_name.strip():
            raise ValueError("file_name must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {"file_name": self.file_name, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnomalousEntry:
        return cls(file_name=data["file_name"], reason=data["reason"])


@dataclass(frozen=True, slots=True)
class UnresolvedRelation:
    """A relationship the documentation describes but the tool does not enforce.

    It is reported so a reader knows it was considered, and it must never be
    presented as a dataset defect: it is pending a maintainer's confirmation.

    Five different things are kept apart, and the quantitative evidence is
    attributed to the dataset that produced it:

    * ``documented_as`` — what Synthea's documentation states;
    * ``implemented_as`` — what the generator source does;
    * ``reference_dataset`` + ``reference_evidence`` — a measurement made on that
      dataset, never on the dataset a report is about;
    * ``why_not_enforced`` — why it is not applied as a constraint;
    * ``pending`` — the question only a maintainer can answer.
    """

    kind: str
    relation: str
    documented_as: str
    implemented_as: str
    reference_dataset: str
    reference_evidence: str
    why_not_enforced: str
    pending: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must be a non-empty string")
        for name in (
            "relation",
            "documented_as",
            "implemented_as",
            "reference_dataset",
            "reference_evidence",
            "why_not_enforced",
            "pending",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "relation": self.relation,
            "documented_as": self.documented_as,
            "implemented_as": self.implemented_as,
            "reference_dataset": self.reference_dataset,
            "reference_evidence": self.reference_evidence,
            "why_not_enforced": self.why_not_enforced,
            "pending": self.pending,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnresolvedRelation:
        return cls(
            kind=data["kind"],
            relation=data["relation"],
            documented_as=data["documented_as"],
            implemented_as=data["implemented_as"],
            reference_dataset=data["reference_dataset"],
            reference_evidence=data["reference_evidence"],
            why_not_enforced=data["why_not_enforced"],
            pending=data["pending"],
        )


@dataclass(frozen=True, slots=True)
class DatasetReport:
    """Complete result of one run of the tool over one dataset directory."""

    data_dir: str
    tables: tuple[TableSummary, ...] = ()
    checks: tuple[CheckResult, ...] = ()
    unknown_files: tuple[str, ...] = ()
    missing_known_tables: tuple[str, ...] = ()
    schema_contract: str | None = None
    generated_at: str = field(default_factory=utc_now_iso)
    tool_version: str = __version__
    schema_version: int = REPORT_SCHEMA_VERSION
    #: Number of tables the contract describes, so a report can state how much of
    #: it was actually observed.
    contract_tables: int = 0
    #: ``COMPATIBLE`` / ``INCOMPATIBLE`` / ``UNKNOWN``, or ``None`` when no contract
    #: was assessed at all. Deliberately never "the dataset is version X".
    contract_status: ContractStatus | None = None
    #: One-line statement of what was compared against the contract.
    contract_summary: str | None = None
    #: Per-table deviations from the contract, as written by the contract check.
    contract_findings: tuple[str, ...] = ()
    load_errors: tuple[LoadError, ...] = ()
    #: Names that could be a table but are not readable files: reported, never a defect.
    anomalous_entries: tuple[AnomalousEntry, ...] = ()
    #: Documented relations that are not enforced (see the schema catalogue).
    unresolved_relations: tuple[UnresolvedRelation, ...] = ()

    @property
    def counts_by_status(self) -> dict[str, int]:
        """Number of checks per status, including statuses with zero checks."""
        counts = {status.value: 0 for status in Status}
        for check in self.checks:
            counts[check.status.value] += 1
        return counts

    @property
    def has_failures(self) -> bool:
        """True when at least one check failed or could not be run."""
        return any(check.status in (Status.FAIL, Status.ERROR) for check in self.checks)

    @property
    def highest_severity_finding(self) -> Severity | None:
        """Most severe outcome among the checks that actually reported something."""
        findings = [
            check
            for check in self.checks
            if check.status in (Status.FAIL, Status.WARNING, Status.ERROR)
        ]
        if not findings:
            return None
        return max(findings, key=lambda check: check.severity.rank).severity

    @property
    def findings(self) -> tuple[CheckResult, ...]:
        """Checks that reported something to look at, most severe and urgent first."""
        selected = [
            check
            for check in self.checks
            if check.status in (Status.FAIL, Status.WARNING, Status.ERROR)
        ]
        order = {Status.FAIL: 0, Status.ERROR: 1, Status.WARNING: 2}
        return tuple(
            sorted(selected, key=lambda check: (order[check.status], -check.severity.rank, check.check_id))
        )

    @property
    def findings_by_severity(self) -> dict[str, int]:
        """How many findings (FAIL, WARNING, ERROR) of each severity, zero included."""
        counts = {severity.value: 0 for severity in Severity}
        for check in self.findings:
            counts[check.severity.value] += 1
        return counts

    @property
    def checks_by_category(self) -> dict[str, tuple[CheckResult, ...]]:
        """Checks grouped by category, in ``CHECK_CATEGORIES`` order."""
        grouped: dict[str, list[CheckResult]] = {name: [] for name in CHECK_CATEGORIES}
        for check in self.checks:
            grouped[category_of(check.check_id)].append(check)
        return {name: tuple(grouped[name]) for name in CHECK_CATEGORIES}

    @property
    def counts_by_category(self) -> dict[str, int]:
        """Number of checks per category, including empty categories."""
        return {name: len(checks) for name, checks in self.checks_by_category.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation of the whole report."""
        highest = self.highest_severity_finding
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "data_dir": self.data_dir,
                "generated_at": self.generated_at,
                "tool_version": self.tool_version,
                "schema_contract": self.schema_contract,
                "contract_tables": self.contract_tables,
                "contract_status": self.contract_status.value if self.contract_status else None,
                "contract_summary": self.contract_summary,
                "contract_findings": list(self.contract_findings),
                "unknown_files": list(self.unknown_files),
                "missing_known_tables": list(self.missing_known_tables),
                "tables_observed": len(self.tables),
                "load_errors": [error.to_dict() for error in self.load_errors],
                "anomalous_entries": [
                    entry.to_dict() for entry in self.anomalous_entries
                ],
                "unresolved_relations": [
                    relation.to_dict() for relation in self.unresolved_relations
                ],
            },
            "tables": [table.to_dict() for table in self.tables],
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "checks_total": len(self.checks),
                "by_status": self.counts_by_status,
                "by_category": self.counts_by_category,
                "findings_by_severity": self.findings_by_severity,
                "highest_severity_finding": highest.value if highest else None,
                "has_failures": self.has_failures,
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the report as JSON (UTF-8 friendly, no ASCII escaping)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetReport:
        """Rebuild a report from :meth:`to_dict` output.

        Keys added within the same report schema version are optional here, so a
        report written by an earlier build of the tool still loads.
        """
        version = data.get("schema_version")
        if version != REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported report schema_version {version!r}; "
                f"this version of the tool reads {REPORT_SCHEMA_VERSION}"
            )
        dataset = data.get("dataset") or {}
        status = dataset.get("contract_status")
        return cls(
            data_dir=dataset["data_dir"],
            tables=tuple(TableSummary.from_dict(item) for item in (data.get("tables") or ())),
            checks=tuple(CheckResult.from_dict(item) for item in (data.get("checks") or ())),
            unknown_files=tuple(dataset.get("unknown_files") or ()),
            missing_known_tables=tuple(dataset.get("missing_known_tables") or ()),
            schema_contract=dataset.get("schema_contract"),
            generated_at=dataset["generated_at"],
            tool_version=dataset["tool_version"],
            schema_version=version,
            contract_tables=int(dataset.get("contract_tables") or 0),
            contract_status=ContractStatus(status) if status else None,
            contract_summary=dataset.get("contract_summary"),
            contract_findings=tuple(dataset.get("contract_findings") or ()),
            load_errors=tuple(
                LoadError.from_dict(item) for item in (dataset.get("load_errors") or ())
            ),
            anomalous_entries=tuple(
                AnomalousEntry.from_dict(item)
                for item in (dataset.get("anomalous_entries") or ())
            ),
            unresolved_relations=tuple(
                UnresolvedRelation.from_dict(item)
                for item in (dataset.get("unresolved_relations") or ())
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> DatasetReport:
        """Rebuild a report from its JSON representation."""
        return cls.from_dict(json.loads(text))
