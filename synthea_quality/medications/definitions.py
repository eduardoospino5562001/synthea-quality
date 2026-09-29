"""[MEDICATIONS] The medications of a module file.

::

    {"conditions": [{"name": "Hypertension", "codes": ["59621000"]}],
     "medications": [
       {"name": "Lisinopril", "codes": ["314076"],
        "cohort": {"condition": "Hypertension", "rule": "point"},
        "expected": {"active": 0.8, "ever": 0.85, "source": "..."}}]}

``codes``
    One or more codes of ``medications.csv``, which has no ``SYSTEM`` column (Synthea
    writes RxNorm codes there). A patient counts once, whichever of the codes they have.
``cohort`` (optional)
    A condition of the same file (see :mod:`synthea_quality.condition_cohort`); without
    it, the denominator is every alive patient.
``expected`` (optional)
    ``active`` and/or ``ever`` as proportions, and a ``source``: shown inside or outside
    the observed 95% CI, never as a verdict.

Anything ambiguous is an input error, never a guess: a repeated name, a name with no
code, a ``SYSTEM|CODE`` code, an unknown key, a proportion outside ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping

from synthea_quality.condition_cohort import CohortSpec, parse_cohort
from synthea_quality.prevalence.definitions import DefinitionError
from synthea_quality.prevalence.models import MEDICATION_MEASURES, Expected

_KEYS = {"name", "codes", "cohort", "expected"}


@dataclass(frozen=True, slots=True)
class MedicationDefinition:
    """A named medication: its codes, the cohort it is measured in and expected shares."""

    name: str
    codes: tuple[str, ...]
    cohort: CohortSpec | None = None
    expected: tuple[Expected, ...] = ()


def load_medications(
    data: Mapping[str, Any], *, origin: str, condition_names: Collection[str]
) -> tuple[MedicationDefinition, ...]:
    """The ``medications`` of a parsed module file (none when the key is absent)."""
    items = data.get("medications")
    if items is None:
        return ()
    if not isinstance(items, list) or not items:
        raise DefinitionError(f"{origin}: 'medications' must be a non-empty list")
    definitions = tuple(
        _definition(item, origin=f"{origin} medications[{index}]",
                    condition_names=condition_names)
        for index, item in enumerate(items)
    )
    names = [d.name for d in definitions]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise DefinitionError(f"{origin}: medication(s) defined more than once: {repeated}")
    return definitions


def _definition(
    item: Any, *, origin: str, condition_names: Collection[str]
) -> MedicationDefinition:
    if not isinstance(item, Mapping):
        raise DefinitionError(f"{origin} is not an object")
    unknown = sorted(set(item) - _KEYS)
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s): {unknown}")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise DefinitionError(f"{origin}: a medication needs a name")
    codes = item.get("codes")
    if not isinstance(codes, list) or not codes:
        raise DefinitionError(f"{origin}: medication {name.strip()!r} needs at least one code")
    cleaned: list[str] = []
    for code in codes:
        if not isinstance(code, str) or not code.strip():
            raise DefinitionError(f"{origin}: a medication code must be a non-empty string")
        if "|" in code:
            raise DefinitionError(
                f"{origin}: medications.csv has no SYSTEM column; give the code alone, "
                f"not {code!r}"
            )
        if code.strip() not in cleaned:
            cleaned.append(code.strip())
    return MedicationDefinition(
        name=name.strip(),
        codes=tuple(cleaned),
        cohort=parse_cohort(item.get("cohort"), origin=origin, condition_names=condition_names),
        expected=_expected(item.get("expected"), origin=origin),
    )


def _expected(block: Any, *, origin: str) -> tuple[Expected, ...]:
    if block is None:
        return ()
    if not isinstance(block, Mapping):
        raise DefinitionError(f"{origin}: 'expected' must be an object")
    unknown = sorted(set(block) - {*MEDICATION_MEASURES, "source"})
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s) in 'expected': {unknown}")
    source = block.get("source")
    if source is not None and not isinstance(source, str):
        raise DefinitionError(f"{origin}: expected 'source' must be a string")
    expected = []
    for measure in MEDICATION_MEASURES:
        if measure not in block:
            continue
        value = block[measure]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DefinitionError(f"{origin}: expected {measure} {value!r} is not a number")
        try:
            expected.append(Expected(measure, float(value), source))
        except ValueError as exc:
            raise DefinitionError(f"{origin}: {exc}") from exc
    return tuple(expected)
