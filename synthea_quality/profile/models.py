"""[PROFILE] Structured result of a dataset profile.

These models are the contract between the profile computations and the renderers,
the same role :mod:`synthea_quality.models` plays for the quality checks. They are
deliberately separate from it: a profile describes a dataset and has no verdict, so
it carries neither ``Status`` nor ``Severity``.

Design notes
------------
* A :class:`ProfileSection` is ``COMPUTED`` or ``SKIPPED``. A skipped section always
  says why (a missing table, a missing column, a table whose rows do not line up with
  its header, no reference date) and carries no numbers, so an absent measurement can
  never be read as a zero.
* Provenance travels with the result: the reference date records its source and
  whether it is an approximation, and :attr:`DatasetProfile.inputs` records which
  tables were read and how many rows each had.
* :meth:`DatasetProfile.to_dict` is deterministic: the key order is fixed here, the
  sections keep the order the builder produced, distributions are ordered by the code
  that computes them, and nested metric mappings are written with sorted keys. The
  only value that differs between two runs over the same input is ``generated_at``.
* The JSON layout is versioned through ``PROFILE_SCHEMA_VERSION``, independently of the
  quality report's ``schema_version``: they are two different public contracts.

Only the standard library is imported here: importing the models must stay cheap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from synthea_quality import __version__
from synthea_quality.models import utc_now_iso

#: Version of the JSON layout produced by :meth:`DatasetProfile.to_dict`.
PROFILE_SCHEMA_VERSION = 1


class SectionStatus(str, Enum):
    """Whether a profile section could be computed.

    ``COMPUTED``  the numbers describe the dataset.
    ``SKIPPED``   the section could not be computed; ``reason`` says why.
    """

    COMPUTED = "COMPUTED"
    SKIPPED = "SKIPPED"


class ReferenceSource(str, Enum):
    """Where the reference date ("end of the simulation") came from."""

    #: Given explicitly by the person running the profile.
    USER = "user"
    #: ``endTime`` of a Synthea run metadata file (written by ``MetadataExporter``).
    SYNTHEA_METADATA = "synthea_metadata"
    #: The latest ``START``/``STOP`` in ``encounters.csv``: an approximation.
    MAX_ENCOUNTER_DATE = "max_encounter_date"


def _normalise_json(value: Any, *, path: str) -> Any:
    """Validate a JSON-compatible value and copy it with sorted mapping keys."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings")
        return {key: _normalise_json(value[key], path=f"{path}.{key}") for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    raise TypeError(f"{path} contains {type(value).__name__}, which is not JSON serialisable")


@dataclass(frozen=True, slots=True)
class ReferenceDate:
    """The date a profile measures ages at, with its provenance."""

    #: ``YYYY-MM-DD``.
    value: str
    source: ReferenceSource
    #: Human-readable account of how the value was obtained.
    detail: str
    #: True when the value is an estimate rather than the recorded end of the simulation.
    approximate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source, ReferenceSource):
            object.__setattr__(self, "source", ReferenceSource(self.source))
        if not self.value or not self.detail:
            raise ValueError("a reference date needs a value and a detail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source.value,
            "approximate": self.approximate,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceDate":
        return cls(
            value=data["value"],
            source=ReferenceSource(data["source"]),
            detail=data["detail"],
            approximate=bool(data["approximate"]),
        )


@dataclass(frozen=True, slots=True)
class CategoryCount:
    """One value of a distribution.

    ``value`` is ``None`` for rows where the column is empty, so an empty field is
    counted in the open instead of being dropped from the distribution.
    """

    value: str | None
    count: int
    #: Share of the section's denominator, in percent, rounded to two decimals.
    percent: float

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "count": self.count, "percent": self.percent}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CategoryCount":
        return cls(value=data["value"], count=int(data["count"]), percent=float(data["percent"]))


def percent(count: int, total: int) -> float:
    """``count`` as a percentage of ``total``, rounded to two decimals (0 when empty)."""
    if total <= 0:
        return 0.0
    return round(100.0 * count / total, 2)


@dataclass(frozen=True, slots=True)
class ProfileSection:
    """One part of a profile: a set of metrics and, optionally, distributions."""

    #: Stable identifier, e.g. ``population`` or ``distribution.GENDER``.
    section_id: str
    title: str
    status: SectionStatus
    #: Why the section was skipped; ``None`` when it was computed.
    reason: str | None = None
    #: JSON-compatible values; nested mappings are allowed.
    metrics: Mapping[str, Any] = field(default_factory=dict)
    #: Named, ordered distributions (``age_band``, ``GENDER`` ...).
    distributions: Mapping[str, tuple[CategoryCount, ...]] = field(default_factory=dict)
    #: Observations a reader needs to interpret the numbers correctly.
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, SectionStatus):
            object.__setattr__(self, "status", SectionStatus(self.status))
        if not self.section_id or not self.title:
            raise ValueError("a section needs an identifier and a title")
        if self.status is SectionStatus.SKIPPED:
            if not self.reason:
                raise ValueError(f"skipped section {self.section_id!r} must state a reason")
            if self.metrics or self.distributions:
                raise ValueError(f"skipped section {self.section_id!r} must carry no numbers")
        elif self.reason is not None:
            raise ValueError(f"computed section {self.section_id!r} must not carry a reason")
        object.__setattr__(
            self, "metrics", _normalise_json(dict(self.metrics), path=f"{self.section_id}.metrics")
        )
        object.__setattr__(
            self,
            "distributions",
            {name: tuple(rows) for name, rows in self.distributions.items()},
        )
        object.__setattr__(self, "notes", tuple(self.notes))

    @classmethod
    def skipped(cls, section_id: str, title: str, reason: str) -> "ProfileSection":
        return cls(section_id=section_id, title=title, status=SectionStatus.SKIPPED, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "status": self.status.value,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "distributions": {
                name: [row.to_dict() for row in rows] for name, rows in self.distributions.items()
            },
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProfileSection":
        return cls(
            section_id=data["section_id"],
            title=data["title"],
            status=SectionStatus(data["status"]),
            reason=data.get("reason"),
            metrics=data.get("metrics", {}),
            distributions={
                name: tuple(CategoryCount.from_dict(row) for row in rows)
                for name, rows in data.get("distributions", {}).items()
            },
            notes=tuple(data.get("notes", ())),
        )


@dataclass(frozen=True, slots=True)
class TableInput:
    """A table the profile read, and what for."""

    table: str
    #: Data rows in the table; ``None`` when it could not be read.
    rows: int | None
    used_for: str

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table, "rows": self.rows, "used_for": self.used_for}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableInput":
        return cls(table=data["table"], rows=data.get("rows"), used_for=data["used_for"])


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """The complete descriptive profile of one dataset."""

    data_dir: str
    #: The reference date, or ``None`` when none could be determined.
    reference_date: ReferenceDate | None
    #: Lower bounds of the age bands, e.g. ``(0, 5, 18, 45, 65)``.
    age_bands: tuple[int, ...]
    sections: tuple[ProfileSection, ...]
    inputs: tuple[TableInput, ...] = ()
    #: Why there is no reference date, when there is none.
    reference_reason: str | None = None
    #: Observations about the run as a whole (e.g. an inconsistent metadata endTime).
    notes: tuple[str, ...] = ()
    generated_at: str = field(default_factory=utc_now_iso)
    tool_version: str = __version__

    def __post_init__(self) -> None:
        if (self.reference_date is None) == (self.reference_reason is None):
            raise ValueError("exactly one of reference_date and reference_reason must be set")
        ids = [section.section_id for section in self.sections]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate section identifiers: {duplicates}")
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "age_bands", tuple(self.age_bands))
        object.__setattr__(self, "notes", tuple(self.notes))

    def section(self, section_id: str) -> ProfileSection:
        """Return the section named ``section_id``."""
        for candidate in self.sections:
            if candidate.section_id == section_id:
                return candidate
        raise KeyError(section_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "tool_version": self.tool_version,
            "generated_at": self.generated_at,
            "data_dir": self.data_dir,
            "reference_date": (
                self.reference_date.to_dict() if self.reference_date is not None else None
            ),
            "reference_reason": self.reference_reason,
            "age_bands": list(self.age_bands),
            "inputs": [item.to_dict() for item in self.inputs],
            "notes": list(self.notes),
            "sections": [section.to_dict() for section in self.sections],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetProfile":
        version = data.get("schema_version")
        if version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profile schema_version {version!r}; "
                f"this tool reads version {PROFILE_SCHEMA_VERSION}"
            )
        reference = data.get("reference_date")
        return cls(
            data_dir=data["data_dir"],
            reference_date=ReferenceDate.from_dict(reference) if reference else None,
            reference_reason=data.get("reference_reason"),
            age_bands=tuple(data["age_bands"]),
            sections=tuple(ProfileSection.from_dict(s) for s in data["sections"]),
            inputs=tuple(TableInput.from_dict(i) for i in data.get("inputs", ())),
            notes=tuple(data.get("notes", ())),
            generated_at=data["generated_at"],
            tool_version=data["tool_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> "DatasetProfile":
        return cls.from_dict(json.loads(text))
