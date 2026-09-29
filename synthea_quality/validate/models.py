"""[VALIDATE] The module file and the structured result of a module validation.

The module file is the conditions JSON of ``synthea-prevalence`` with an optional
``module`` block describing the module::

    {"module": {"name": "Myocardial infarction",
                "synthea_modules": ["myocardial_infarction.json"]},
     "conditions": [{"name": "Myocardial infarction",
                     "codes": ["22298006", "401303003", "401314000"],
                     "acute": true,
                     "expected": {"lifetime": 0.03, "incidence": 2.5,
                                  "source": "..."}}]}

Expected values may use the three measures (``point`` and ``lifetime`` as proportions,
``incidence`` per 1,000 person-years). The report embeds the prevalence and incidence
results as their own commands produce them (``to_dict``), so there is one JSON layout per
result, versioned here as a whole by ``MODULE_VALIDATION_SCHEMA_VERSION``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthea_quality import __version__
from synthea_quality.incidence.models import ConditionIncidence, Window
from synthea_quality.models import utc_now_iso
from synthea_quality.prevalence.definitions import (
    ConditionDefinition,
    DefinitionError,
    assemble,
)
from synthea_quality.prevalence.models import (
    ALL_MEASURES,
    CodeRef,
    ConditionResult,
    ExpectedComparison,
)
from synthea_quality.profile.models import ProfileSection, ReferenceDate, TableInput

MODULE_VALIDATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    """What the module file says about the module (all optional)."""

    name: str | None = None
    synthea_modules: tuple[str, ...] = ()
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "synthea_modules": list(self.synthea_modules),
            "description": self.description,
        }


def load_module_file(path: str | Path) -> tuple[ModuleInfo, tuple[ConditionDefinition, ...]]:
    """The module block and the conditions of a module file.

    :raises DefinitionError: the file, its ``module`` block or its conditions cannot be
        used as given.
    """
    file_path = Path(path)
    definitions = assemble((), file_path, (), measures=ALL_MEASURES)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    block = data.get("module", {})
    if not isinstance(block, dict):
        raise DefinitionError(f"{file_path}: 'module' must be an object")
    unknown = sorted(set(block) - {"name", "synthea_modules", "description"})
    if unknown:
        raise DefinitionError(f"{file_path}: unknown key(s) in 'module': {unknown}")
    name, modules, description = (
        block.get("name"),
        block.get("synthea_modules", []),
        block.get("description"),
    )
    if name is not None and not isinstance(name, str):
        raise DefinitionError(f"{file_path}: module 'name' must be a string")
    if description is not None and not isinstance(description, str):
        raise DefinitionError(f"{file_path}: module 'description' must be a string")
    if not isinstance(modules, list) or not all(isinstance(m, str) for m in modules):
        raise DefinitionError(f"{file_path}: 'synthea_modules' must be a list of strings")
    return ModuleInfo(name, tuple(modules), description), definitions


@dataclass(frozen=True, slots=True)
class ConditionValidation:
    """One condition of the module: its prevalence and its incidence."""

    name: str
    codes: tuple[CodeRef, ...]
    acute: bool
    prevalence: ConditionResult
    incidence: ConditionIncidence

    @property
    def references(self) -> tuple[ExpectedComparison, ...]:
        """Every expected value with its observed counterpart, point, lifetime, incidence."""
        return (*self.prevalence.expected, *self.incidence.expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "codes": [code.to_dict() for code in self.codes],
            "acute": self.acute,
            "prevalence": self.prevalence.to_dict(),
            "incidence": self.incidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModuleValidationReport:
    """The complete result of one module validation."""

    data_dir: str
    module_file: str
    module: ModuleInfo
    reference_date: ReferenceDate | None
    reference_reason: str | None
    age_bands: tuple[int, ...]
    alive: int | None
    population: tuple[ProfileSection, ...]
    conditions: tuple[ConditionValidation, ...]
    incidence_window: Window | None = None
    incidence_population: str = "all"
    history: dict[str, Any] = field(default_factory=dict)
    cohort: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[TableInput, ...] = ()
    notes: tuple[str, ...] = ()
    generated_at: str = field(default_factory=utc_now_iso)
    tool_version: str = __version__

    @property
    def incomplete(self) -> bool:
        return any(item.state.value == "unreadable" for item in self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODULE_VALIDATION_SCHEMA_VERSION,
            "tool_version": self.tool_version,
            "generated_at": self.generated_at,
            "data_dir": self.data_dir,
            "module_file": self.module_file,
            "module": self.module.to_dict(),
            "reference_date": self.reference_date.to_dict() if self.reference_date else None,
            "reference_reason": self.reference_reason,
            "age_bands": list(self.age_bands),
            "alive": self.alive,
            "incidence_window": self.incidence_window.to_dict() if self.incidence_window else None,
            "incidence_population": self.incidence_population,
            "history": _sorted(self.history),
            "cohort": _sorted(self.cohort),
            "inputs": [item.to_dict() for item in self.inputs],
            "notes": list(self.notes),
            "population": [section.to_dict() for section in self.population],
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _sorted(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sorted(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sorted(item) for item in value]
    return value
