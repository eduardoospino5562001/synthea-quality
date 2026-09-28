"""[PREVALENCE] The conditions asked for: names, codes and optional expected values.

A disease often has several codes (myocardial infarction: STEMI, NSTEMI, unspecified),
so a condition is a **name with one or more codes**, and a patient counts once for the
name whichever of its codes they have.

Three ways to give them, which can be combined:

``--condition "Myocardial infarction=22298006,401303003,401314000"``
    Repeatable. Codes are SNOMED CT unless written ``SYSTEM|CODE``
    (``http://snomed.info/sct|22298006``).

``--conditions FILE.json``
    Meant to live next to a module, for example::

        {"conditions": [
          {"name": "Myocardial infarction",
           "codes": ["22298006", "401303003", "401314000"],
           "expected": {"lifetime": 0.03, "source": "CDC 2019"}}
        ]}

    ``system`` may be given per condition (default SNOMED CT), and a code may be an
    object ``{"system": ..., "code": ...}``.

``--expected "Myocardial infarction:lifetime=0.03"``
    Repeatable. Adds a reference value to a condition defined by one of the above.

Anything ambiguous is an input error, never a guess: a repeated name, a name with no
code, an expected value for a condition that was not defined or given twice, a value
outside ``[0, 1]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from synthea_quality.errors import SyntheaQualityError
from synthea_quality.prevalence.models import (
    MEASURES,
    SNOMED_CT,
    CodeRef,
    Expected,
    normalise_system,
)


class DefinitionError(SyntheaQualityError):
    """A condition definition or expected value cannot be used as given."""


@dataclass(frozen=True, slots=True)
class ConditionDefinition:
    """A named condition: its codes and the expected values to show next to it."""

    name: str
    codes: tuple[CodeRef, ...]
    expected: tuple[Expected, ...] = ()


def parse_condition_option(text: str) -> ConditionDefinition:
    """Parse ``NAME=CODE[,CODE...]`` (a code may be ``SYSTEM|CODE``)."""
    name, separator, codes = text.partition("=")
    if not separator:
        raise DefinitionError(f"--condition needs NAME=CODE[,CODE...], got {text!r}")
    return _definition(name, [c for c in codes.split(",")], SNOMED_CT, (), origin="--condition")


def load_conditions_file(path: str | Path) -> list[ConditionDefinition]:
    """Read the conditions of a ``--conditions`` JSON file."""
    file_path = Path(path)
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise DefinitionError(f"conditions file {file_path} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DefinitionError(f"conditions file {file_path} is not valid JSON: {exc}") from exc
    items = data.get("conditions") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise DefinitionError(f"conditions file {file_path} needs a non-empty 'conditions' list")
    definitions = []
    for index, item in enumerate(items):
        origin = f"{file_path} conditions[{index}]"
        if not isinstance(item, dict):
            raise DefinitionError(f"{origin} is not an object")
        system = item.get("system", SNOMED_CT)
        expected = _expected_block(item.get("expected"), origin)
        definitions.append(
            _definition(item.get("name", ""), item.get("codes"), system, expected, origin=origin)
        )
    return definitions


def parse_expected_option(text: str) -> tuple[str, Expected]:
    """Parse ``NAME:MEASURE=VALUE`` into the condition name and its expected value."""
    head, separator, value = text.rpartition("=")
    name, colon, measure = head.rpartition(":")
    if not separator or not colon or not name.strip():
        raise DefinitionError(f"--expected needs NAME:MEASURE=VALUE, got {text!r}")
    return name.strip(), _expected(measure.strip(), value.strip(), None, origin="--expected")


def assemble(
    condition_options: Iterable[str] = (),
    conditions_file: str | Path | None = None,
    expected_options: Iterable[str] = (),
) -> tuple[ConditionDefinition, ...]:
    """Every condition asked for, in the order given, with its expected values attached."""
    definitions: list[ConditionDefinition] = []
    if conditions_file is not None:
        definitions.extend(load_conditions_file(conditions_file))
    definitions.extend(parse_condition_option(text) for text in condition_options)

    names = [d.name for d in definitions]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise DefinitionError(f"condition name(s) defined more than once: {repeated}")

    by_name = {d.name: d for d in definitions}
    for text in expected_options:
        name, expected = parse_expected_option(text)
        if name not in by_name:
            raise DefinitionError(
                f"--expected refers to {name!r}, which no --condition or --conditions defines"
            )
        current = by_name[name]
        if any(e.measure == expected.measure for e in current.expected):
            raise DefinitionError(f"{name!r} has more than one expected {expected.measure} value")
        by_name[name] = ConditionDefinition(
            current.name, current.codes, (*current.expected, expected)
        )
    return tuple(by_name[name] for name in names)


def _definition(
    name: Any, codes: Any, system: Any, expected: tuple[Expected, ...], *, origin: str
) -> ConditionDefinition:
    if not isinstance(name, str) or not name.strip():
        raise DefinitionError(f"{origin}: a condition needs a name")
    if not isinstance(codes, list) or not codes:
        raise DefinitionError(f"{origin}: condition {name.strip()!r} needs at least one code")
    refs: list[CodeRef] = []
    for raw in codes:
        ref = _code(raw, system, origin=f"{origin} ({name.strip()})")
        if ref not in refs:
            refs.append(ref)
    return ConditionDefinition(name.strip(), tuple(refs), expected)


def _code(raw: Any, default_system: Any, *, origin: str) -> CodeRef:
    if isinstance(raw, Mapping):
        system, code = raw.get("system", default_system), raw.get("code")
    elif isinstance(raw, str):
        system, _, code = raw.rpartition("|") if "|" in raw else (default_system, "", raw)
    else:
        raise DefinitionError(f"{origin}: a code must be a string or an object, got {raw!r}")
    if not isinstance(code, str) or not code.strip():
        raise DefinitionError(f"{origin}: empty code")
    if system is not None and not isinstance(system, str):
        raise DefinitionError(f"{origin}: a system must be a string")
    return CodeRef(code.strip(), normalise_system(system) if system else None)


def _expected_block(block: Any, origin: str) -> tuple[Expected, ...]:
    if block is None:
        return ()
    if not isinstance(block, dict):
        raise DefinitionError(f"{origin}: 'expected' must be an object")
    unknown = sorted(set(block) - {*MEASURES, "source"})
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s) in 'expected': {unknown}")
    source = block.get("source")
    return tuple(
        _expected(measure, block[measure], source, origin=origin)
        for measure in MEASURES
        if measure in block
    )


def _expected(measure: str, value: Any, source: Any, *, origin: str) -> Expected:
    if measure not in MEASURES:
        raise DefinitionError(f"{origin}: measure must be one of {list(MEASURES)}, not {measure!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DefinitionError(
            f"{origin}: expected {measure} value {value!r} is not a number"
        ) from exc
    try:
        return Expected(measure, number, source if isinstance(source, str) else None)
    except ValueError as exc:
        raise DefinitionError(f"{origin}: {exc}") from exc
