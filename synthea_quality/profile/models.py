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
  quality report's ``schema_version``: they are two different public contracts. As for
  the quality report, adding optional keys keeps the version (``codes`` and
  ``multi_description_codes`` were added that way, and a profile written without them
  still loads); changing or removing keys requires a new one.

Only the standard library is imported here: importing the models must stay cheap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from synthea_quality import __version__
from synthea_quality.export_history import ExportHistory
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
class CodeCount:
    """One code of a clinical table and how many alive patients have it."""

    #: ``SYSTEM`` of the code when the table has that column, ``None`` otherwise. A code
    #: is identified by ``(system, code)``: the same number in SNOMED and RxNorm is two
    #: different codes.
    system: str | None
    code: str
    #: The most frequent description of the code over the whole table.
    description: str | None
    #: Distinct alive patients with at least one record of the code.
    patients: int
    #: ``patients`` as a percentage of the alive patients.
    percent: float
    #: Records of the code belonging to alive patients.
    records: int
    #: How many distinct descriptions the code has in the whole table.
    description_variants: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "code": self.code,
            "description": self.description,
            "patients": self.patients,
            "percent": self.percent,
            "records": self.records,
            "description_variants": self.description_variants,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CodeCount":
        return cls(
            system=data.get("system"),
            code=data["code"],
            description=data.get("description"),
            patients=int(data["patients"]),
            percent=float(data["percent"]),
            records=int(data["records"]),
            description_variants=int(data.get("description_variants", 1)),
        )


@dataclass(frozen=True, slots=True)
class DescriptionCount:
    """One description a code is written with, and on how many records."""

    description: str | None
    records: int

    def to_dict(self) -> dict[str, Any]:
        return {"description": self.description, "records": self.records}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DescriptionCount":
        return cls(description=data.get("description"), records=int(data["records"]))


@dataclass(frozen=True, slots=True)
class CodeDescriptions:
    """A code written with more than one description, with every variant.

    ``descriptions`` are ordered by records, most frequent first (then alphabetically),
    so the first one is the description the profile shows for the code.
    """

    system: str | None
    code: str
    descriptions: tuple[DescriptionCount, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptions", tuple(self.descriptions))
        if len(self.descriptions) < 2:
            raise ValueError(f"code {self.code!r} needs at least two descriptions to be listed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "code": self.code,
            "descriptions": [item.to_dict() for item in self.descriptions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CodeDescriptions":
        return cls(
            system=data.get("system"),
            code=data["code"],
            descriptions=tuple(DescriptionCount.from_dict(d) for d in data["descriptions"]),
        )


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
    #: Most common codes of a clinical table, in order.
    codes: tuple[CodeCount, ...] = ()
    #: Codes of that table written with more than one description.
    multi_description_codes: tuple[CodeDescriptions, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, SectionStatus):
            object.__setattr__(self, "status", SectionStatus(self.status))
        if not self.section_id or not self.title:
            raise ValueError("a section needs an identifier and a title")
        if self.status is SectionStatus.SKIPPED:
            if not self.reason:
                raise ValueError(f"skipped section {self.section_id!r} must state a reason")
            if self.metrics or self.distributions or self.codes or self.multi_description_codes:
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
        object.__setattr__(self, "codes", tuple(self.codes))
        object.__setattr__(self, "multi_description_codes", tuple(self.multi_description_codes))

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
            "codes": [row.to_dict() for row in self.codes],
            "multi_description_codes": [item.to_dict() for item in self.multi_description_codes],
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
            codes=tuple(CodeCount.from_dict(row) for row in data.get("codes", ())),
            multi_description_codes=tuple(
                CodeDescriptions.from_dict(item)
                for item in data.get("multi_description_codes", ())
            ),
        )


class InputState(str, Enum):
    """What happened to a table the profile needed."""

    #: Read; its numbers are in the profile.
    READ = "read"
    #: Not in the dataset. Legitimate (Synthea can omit files): the sections that need
    #: it are skipped, and the profile is still complete about what the dataset holds.
    ABSENT = "absent"
    #: Present but could not be read, or its rows do not line up with its header. The
    #: profile is incomplete: the sections that need it are skipped.
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class TableInput:
    """A table the profile needed, what for, and whether it could be read."""

    table: str
    used_for: str
    state: InputState
    #: Data rows read; ``None`` unless the table was read.
    rows: int | None = None
    #: Why it was not read, when it was not.
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, InputState):
            object.__setattr__(self, "state", InputState(self.state))
        if (self.state is InputState.READ) != (self.rows is not None):
            raise ValueError("rows must be given exactly when the table was read")
        if (self.state is InputState.READ) != (self.reason is None):
            raise ValueError("a table that was not read needs a reason, and only then")

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "used_for": self.used_for,
            "state": self.state.value,
            "rows": self.rows,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableInput":
        return cls(
            table=data["table"],
            used_for=data["used_for"],
            state=InputState(data["state"]),
            rows=data.get("rows"),
            reason=data.get("reason"),
        )


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
    #: What the metadata file says about the exported history (no notice without it).
    export_history: ExportHistory = field(default_factory=ExportHistory.no_metadata)
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

    @property
    def incomplete(self) -> bool:
        """True when a table the profile needed is present but could not be read."""
        return any(item.state is InputState.UNREADABLE for item in self.inputs)

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
            "export_history": self.export_history.to_dict(),
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
        history = data.get("export_history")
        return cls(
            data_dir=data["data_dir"],
            reference_date=ReferenceDate.from_dict(reference) if reference else None,
            reference_reason=data.get("reference_reason"),
            age_bands=tuple(data["age_bands"]),
            sections=tuple(ProfileSection.from_dict(s) for s in data["sections"]),
            inputs=tuple(TableInput.from_dict(i) for i in data.get("inputs", ())),
            notes=tuple(data.get("notes", ())),
            export_history=ExportHistory.from_dict(history) if history else ExportHistory.no_metadata(),
            generated_at=data["generated_at"],
            tool_version=data["tool_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> "DatasetProfile":
        return cls.from_dict(json.loads(text))
