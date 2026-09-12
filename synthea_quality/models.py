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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from synthea_quality import __version__

#: Version of the JSON layout produced by :meth:`DatasetReport.to_dict`.
REPORT_SCHEMA_VERSION = 1

#: Default number of offending items kept in ``CheckResult.samples``.
#: A report must never embed millions of identifiers, so every check takes a
#: (configurable) limit and reports the full count in ``metrics`` instead.
DEFAULT_SAMPLE_LIMIT = 5


class Status(str, Enum):
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


class Severity(str, Enum):
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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckResult":
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
    def from_dict(cls, data: Mapping[str, Any]) -> "TableSummary":
        return cls(
            name=data["name"],
            file_name=data["file_name"],
            rows=data.get("rows"),
            columns=tuple(data.get("columns") or ()),
            schema_match=data.get("schema_match"),
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
                "unknown_files": list(self.unknown_files),
                "missing_known_tables": list(self.missing_known_tables),
            },
            "tables": [table.to_dict() for table in self.tables],
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "checks_total": len(self.checks),
                "by_status": self.counts_by_status,
                "highest_severity_finding": highest.value if highest else None,
                "has_failures": self.has_failures,
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise the report as JSON (UTF-8 friendly, no ASCII escaping)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetReport":
        """Rebuild a report from :meth:`to_dict` output."""
        version = data.get("schema_version")
        if version != REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported report schema_version {version!r}; "
                f"this version of the tool reads {REPORT_SCHEMA_VERSION}"
            )
        dataset = data.get("dataset") or {}
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
        )

    @classmethod
    def from_json(cls, text: str) -> "DatasetReport":
        """Rebuild a report from its JSON representation."""
        return cls.from_dict(json.loads(text))
