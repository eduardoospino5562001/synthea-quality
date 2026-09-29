"""[OBSERVATIONS] The observations asked for: a code, and optionally a cohort and a range.

An observation is **one code** of ``observations.csv``, which has no ``SYSTEM`` column
(Synthea writes LOINC codes there). Two ways to ask for them:

``--observation CODE`` or ``--observation "NAME=CODE"``
    Repeatable. Without a name, the report uses the code's most frequent description.

``observations`` in a module file (``--module``, ``synthea-validate-module``)::

    {"conditions": [{"name": "Hypertension", "codes": ["59621000"]}],
     "observations": [
       {"name": "Systolic blood pressure", "code": "8480-6",
        "reference_range": {"low": 100, "high": 139, "units": "mm[Hg]",
                            "basis": "synthea-configuration",
                            "source": "Synthea configuration (biometrics.yml, ...)"}},
       {"name": "Systolic blood pressure, hypertension", "code": "8480-6",
        "cohort": {"condition": "Hypertension", "rule": "point"},
        "lookback_years": 3}]}

``cohort`` (optional)
    Limits the patients to those with a condition of the same file; see
    :mod:`synthea_quality.condition_cohort`.
``lookback_years`` (optional)
    Leaves out a patient whose latest value is older than this many years before the
    reference date. They are counted, never dropped silently.
``reference_range`` (optional)
    ``low`` and/or ``high`` (at least one), ``units`` (required: a range is compared only
    with values in the same units, never converted), ``source`` and ``basis``.
    ``basis`` says what the range is: ``synthea-configuration`` (the generator's own
    settings, which the report says are not a clinical norm) or ``external`` (a
    published or clinical reference). The report shows the range next to the observed
    distribution and counts the patients below, within and above it — no verdict.

Anything ambiguous is an input error, never a guess: a repeated name, an empty code, an
unknown key, a range without units or with ``low > high``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping

from synthea_quality.condition_cohort import CohortSpec, parse_cohort
from synthea_quality.prevalence.definitions import (
    ConditionDefinition,
    DefinitionError,
    assemble,
)
from synthea_quality.prevalence.models import ALL_MEASURES

#: What a reference range can be.
BASES = ("synthea-configuration", "external")

#: Said next to a range whose basis is the generator's configuration.
CONFIGURATION_NOTE = (
    "This range is Synthea's own configuration, not a clinical norm: the comparison shows "
    "how the output relates to the generator's settings, not whether the values are "
    "clinically plausible."
)

_OBSERVATION_KEYS = {"name", "code", "cohort", "reference_range", "lookback_years"}
_RANGE_KEYS = {"low", "high", "units", "source", "basis"}


@dataclass(frozen=True, slots=True)
class ReferenceRange:
    """A range to show next to the observed values. It is never a verdict."""

    units: str
    low: float | None = None
    high: float | None = None
    source: str | None = None
    basis: str | None = None

    def __post_init__(self) -> None:
        if not self.units or not self.units.strip():
            raise ValueError("a reference range needs its units")
        if self.low is None and self.high is None:
            raise ValueError("a reference range needs a low or a high bound")
        for bound in (self.low, self.high):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("reference range bounds must be finite numbers")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"reference range low {self.low} is above high {self.high}")
        if self.basis is not None and self.basis not in BASES:
            raise ValueError(f"a reference range basis must be one of {list(BASES)}")

    @property
    def note(self) -> str | None:
        return CONFIGURATION_NOTE if self.basis == "synthea-configuration" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": None if self.low is None else float(self.low),
            "high": None if self.high is None else float(self.high),
            "units": self.units,
            "source": self.source,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class ObservationDefinition:
    """One code of ``observations.csv`` to describe, with its optional cohort and range."""

    code: str
    #: ``None`` when not given: the report then uses the code's most frequent description.
    name: str | None = None
    cohort: CohortSpec | None = None
    reference_range: ReferenceRange | None = None
    lookback_years: int | None = None

    @property
    def key(self) -> str:
        """What identifies the definition in a report: its name, or its code."""
        return self.name if self.name is not None else self.code


def parse_observation_option(text: str) -> ObservationDefinition:
    """Parse ``CODE`` or ``NAME=CODE``."""
    name, separator, code = text.rpartition("=")
    if separator and not name.strip():
        raise DefinitionError(f"--observation needs CODE or NAME=CODE, got {text!r}")
    return _definition(
        {"name": name if separator else None, "code": code}, origin="--observation",
        condition_names=(),
    )


def load_observations(
    data: Mapping[str, Any], *, origin: str, condition_names: Collection[str]
) -> tuple[ObservationDefinition, ...]:
    """The ``observations`` of a parsed module file (none when the key is absent)."""
    items = data.get("observations")
    if items is None:
        return ()
    if not isinstance(items, list) or not items:
        raise DefinitionError(f"{origin}: 'observations' must be a non-empty list")
    definitions = tuple(
        _definition(item, origin=f"{origin} observations[{index}]", condition_names=condition_names)
        for index, item in enumerate(items)
    )
    return unique(definitions)


def load_module_observations(
    path: str | Path,
) -> tuple[tuple[ConditionDefinition, ...], tuple[ObservationDefinition, ...]]:
    """The conditions (possibly none) and the observations of a module file.

    :raises DefinitionError: the file cannot be read or used as given.
    """
    file_path = Path(path)
    data = read_module_json(file_path)
    conditions = (
        assemble((), file_path, (), measures=ALL_MEASURES) if "conditions" in data else ()
    )
    observations = load_observations(
        data, origin=str(file_path), condition_names={c.name for c in conditions}
    )
    return conditions, observations


def read_module_json(path: Path) -> dict[str, Any]:
    """A module file as a JSON object.

    :raises DefinitionError: it cannot be read, is not JSON or is not an object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise DefinitionError(f"module file {path} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DefinitionError(f"module file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DefinitionError(f"module file {path} must hold a JSON object")
    return data


def unique(definitions: tuple[ObservationDefinition, ...]) -> tuple[ObservationDefinition, ...]:
    """``definitions`` unchanged when every name (or unnamed code) is used once.

    :raises DefinitionError: two definitions would share a heading in the report.
    """
    keys = [d.key for d in definitions]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    if repeated:
        raise DefinitionError(f"observation(s) defined more than once: {repeated}")
    return definitions


def _definition(
    item: Any, *, origin: str, condition_names: Collection[str]
) -> ObservationDefinition:
    if not isinstance(item, Mapping):
        raise DefinitionError(f"{origin} is not an object")
    unknown = sorted(set(item) - _OBSERVATION_KEYS)
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s): {unknown}")
    code, name = item.get("code"), item.get("name")
    if not isinstance(code, str) or not code.strip():
        raise DefinitionError(f"{origin}: an observation needs a code")
    if "|" in code:
        raise DefinitionError(
            f"{origin}: observations.csv has no SYSTEM column; give the code alone, not {code!r}"
        )
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise DefinitionError(f"{origin}: an observation name must be a non-empty string")
    lookback = item.get("lookback_years")
    if lookback is not None and (
        isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1
    ):
        raise DefinitionError(f"{origin}: 'lookback_years' must be an integer of at least 1")
    return ObservationDefinition(
        code=code.strip(),
        name=name.strip() if name is not None else None,
        cohort=parse_cohort(item.get("cohort"), origin=origin, condition_names=condition_names),
        reference_range=_range(item.get("reference_range"), origin=origin),
        lookback_years=lookback,
    )


def _range(raw: Any, *, origin: str) -> ReferenceRange | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise DefinitionError(f"{origin}: 'reference_range' must be an object")
    unknown = sorted(set(raw) - _RANGE_KEYS)
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s) in 'reference_range': {unknown}")
    bounds = {}
    for key in ("low", "high"):
        value = raw.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise DefinitionError(f"{origin}: reference range {key} {value!r} is not a number")
        bounds[key] = None if value is None else float(value)
    units, source, basis = raw.get("units"), raw.get("source"), raw.get("basis")
    if not isinstance(units, str):
        raise DefinitionError(
            f"{origin}: a reference range needs its 'units' (values are never converted)"
        )
    if source is not None and not isinstance(source, str):
        raise DefinitionError(f"{origin}: reference range 'source' must be a string")
    try:
        return ReferenceRange(units.strip(), bounds["low"], bounds["high"], source, basis)
    except ValueError as exc:
        raise DefinitionError(f"{origin}: {exc}") from exc
