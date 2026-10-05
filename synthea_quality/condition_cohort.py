"""[COHORT] A population restricted to the patients who have a condition.

An analysis can be limited to the alive patients with a condition of the same module file
(for example the blood pressure of the patients with hypertension)::

    "cohort": {"condition": "Hypertension", "rule": "point"}

or, for the default rule, just ``"cohort": "Hypertension"``. The condition is named, not
repeated: it must be one of the file's ``conditions``, so its codes are written once and
the cohort is exactly the numerator of that condition's prevalence
(:func:`synthea_quality.prevalence.compute.patients_with`).

Rules
-----
``point`` (the default)
    a record active at the reference date: ``START <= ref`` and ``STOP`` empty or after it.
    It answers "has the condition now".
``lifetime``
    a record started on or before the reference date, ended or not.

The rule is stated in every result that uses a cohort; it is never inferred.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from synthea_quality.prevalence.compute import Records, patients_with
from synthea_quality.prevalence.definitions import ConditionDefinition, DefinitionError
from synthea_quality.prevalence.models import MEASURES

#: How a condition selects its patients; the same two measures as prevalence.
RULES = MEASURES
DEFAULT_RULE = "point"


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """The alive patients with ``condition`` at the reference date, under ``rule``."""

    condition: str
    rule: str = DEFAULT_RULE

    def __post_init__(self) -> None:
        if not self.condition or not self.condition.strip():
            raise ValueError("a cohort needs a condition name")
        if self.rule not in RULES:
            raise ValueError(f"a cohort rule must be one of {list(RULES)}, not {self.rule!r}")

    def describe(self) -> str:
        if self.rule == "point":
            return f"alive patients with {self.condition} active at the reference date"
        return f"alive patients with a record of {self.condition} on or before the reference date"

    def to_dict(self) -> dict[str, Any]:
        return {"condition": self.condition, "rule": self.rule}


def parse_cohort(raw: Any, *, origin: str, condition_names: Collection[str]) -> CohortSpec | None:
    """The cohort of a module file entry, or ``None`` when it has none.

    :raises DefinitionError: the cohort is malformed or names a condition the file does
        not define.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        name, rule = raw, DEFAULT_RULE
    elif isinstance(raw, Mapping):
        unknown = sorted(set(raw) - {"condition", "rule"})
        if unknown:
            raise DefinitionError(f"{origin}: unknown key(s) in 'cohort': {unknown}")
        name, rule = raw.get("condition"), raw.get("rule", DEFAULT_RULE)
    else:
        raise DefinitionError(f"{origin}: 'cohort' must be a condition name or an object")
    if not isinstance(name, str) or not name.strip():
        raise DefinitionError(f"{origin}: a cohort needs a condition name")
    if rule not in RULES:
        raise DefinitionError(f"{origin}: a cohort rule must be one of {list(RULES)}, not {rule!r}")
    if name.strip() not in condition_names:
        raise DefinitionError(
            f"{origin}: the cohort names {name.strip()!r}, which the file's 'conditions' do not "
            f"define"
        )
    return CohortSpec(name.strip(), rule)


def select_cohort(
    spec: CohortSpec,
    definitions: Mapping[str, ConditionDefinition],
    records: Records | None,
    unavailable: str | None = None,
) -> tuple[pd.Index | None, str | None]:
    """The patients of ``spec``, or ``None`` and why they cannot be selected.

    :param records: the alive patients' condition records placed at the reference date
        (:func:`synthea_quality.prevalence.compute.prepare_records`), or ``None`` when
        there are none; ``unavailable`` then says why.
    """
    if records is None:
        return None, f"the cohort ({spec.describe()}) cannot be selected: {unavailable}"
    return patients_with(definitions[spec.condition], records, spec.rule), None
