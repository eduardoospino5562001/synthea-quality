"""[INDEXED] The index of a module file and the outcomes measured from it.

::

    {"conditions": [{"name": "COVID-19", "codes": ["840539006"]}],
     "index": {"condition": "COVID-19"},
     "outcomes": [
       {"name": "Hospital admission", "type": "event", "table": "encounters",
        "codes": ["1505002"]},
       {"name": "Stage group", "type": "value", "code": "21908-9", "pick": "first"},
       {"name": "Death", "type": "death",
        "cause_values": ["Malignant neoplasm of breast (disorder)"]}]}

``index``
    A condition of the same file (see :mod:`synthea_quality.condition_cohort`). Its
    earliest ``START`` in or before the reference date is the patient's index date. The
    indexed cohort is every patient with that condition, alive or deceased: a ``death``
    outcome would be impossible otherwise.
``outcomes``
    What is measured from each patient's index date. Every outcome has a unique ``name``
    and an optional ``window``; the window is counted in days from the index date, from
    day 0 with no end by default.
``type: event``
    ``table`` and ``codes``: patients with a record of one of the codes in the window.
``type: value``
    ``code`` of ``observations`` and ``pick`` (``first`` or ``last``): the value on the
    earliest or latest date of the window.
``type: death``
    ``cause_values`` (optional): exact ``VALUE`` texts of observation ``69453-9``.

Anything ambiguous is an input error, never a guess: an ``index`` without ``outcomes``
or the reverse, an empty ``outcomes`` list, a repeated name, an unknown key or type, a
table outside :data:`EVENT_TABLES`, an empty ``codes`` list, a bad ``pick``, or a window
whose ``from_days`` is after its ``to_days``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from synthea_quality.prevalence.definitions import DefinitionError

#: The kinds of outcome an indexed cohort can measure.
OUTCOME_TYPES = ("event", "value", "death")
#: The tables an ``event`` outcome may read.
EVENT_TABLES = (
    "conditions",
    "encounters",
    "procedures",
    "medications",
    "careplans",
    "devices",
    "immunizations",
    "observations",
)
#: How a ``value`` outcome picks one date per patient.
PICKS = ("first", "last")
DEFAULT_PICK = "first"

_COMMON_KEYS = {"name", "type", "window"}
_TYPE_KEYS = {
    "event": {"table", "codes"},
    "value": {"code", "pick"},
    "death": {"cause_values"},
}


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """The condition whose earliest start is a patient's index date."""

    condition: str


@dataclass(frozen=True, slots=True)
class Window:
    """Days from the index date; ``to_days`` is ``None`` for an open window."""

    from_days: int = 0
    to_days: int | None = None


@dataclass(frozen=True, slots=True)
class EventOutcome:
    """Patients with a record of one of ``codes`` in ``table`` within the window."""

    name: str
    table: str
    codes: tuple[str, ...]
    window: Window = field(default_factory=Window)


@dataclass(frozen=True, slots=True)
class ValueOutcome:
    """The value of an observation on the first or last date of the window."""

    name: str
    code: str
    pick: str = DEFAULT_PICK
    window: Window = field(default_factory=Window)


@dataclass(frozen=True, slots=True)
class DeathOutcome:
    """Deaths within the window, optionally narrowed by their recorded cause."""

    name: str
    cause_values: tuple[str, ...] = ()
    window: Window = field(default_factory=Window)


Outcome = EventOutcome | ValueOutcome | DeathOutcome


def load_indexed(
    data: Mapping[str, Any], *, origin: str, condition_names: Collection[str]
) -> tuple[IndexSpec | None, tuple[Outcome, ...]]:
    """The ``index`` and ``outcomes`` of a parsed module file (both or neither).

    A file without an indexed cohort has no ``index`` and no ``outcomes``; the two keys
    are only meaningful together, so one without the other is an input error.

    :raises DefinitionError: the index names a condition the file does not define, the
        two keys are incomplete, or an outcome is malformed.
    """
    if "index" not in data and "outcomes" not in data:
        return None, ()
    if "index" not in data:
        raise DefinitionError(f"{origin}: 'outcomes' needs an 'index'")
    index = _index(data["index"], origin=origin, condition_names=condition_names)
    if "outcomes" not in data:
        raise DefinitionError(f"{origin}: 'index' needs a non-empty 'outcomes' list")
    return index, _outcomes(data["outcomes"], origin=origin)


def _index(
    raw: Any, *, origin: str, condition_names: Collection[str]
) -> IndexSpec:
    if not isinstance(raw, Mapping):
        raise DefinitionError(f"{origin}: 'index' must be an object")
    unknown = sorted(set(raw) - {"condition"})
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s) in 'index': {unknown}")
    condition = raw.get("condition")
    if not isinstance(condition, str) or not condition.strip():
        raise DefinitionError(f"{origin}: 'index' needs a condition name")
    condition = condition.strip()
    if condition not in condition_names:
        raise DefinitionError(
            f"{origin}: 'index' names {condition!r}, which the file's 'conditions' do not "
            f"define"
        )
    return IndexSpec(condition)


def _outcomes(raw: Any, *, origin: str) -> tuple[Outcome, ...]:
    if not isinstance(raw, list) or not raw:
        raise DefinitionError(f"{origin}: 'outcomes' must be a non-empty list")
    outcomes = tuple(
        _outcome(item, origin=f"{origin} outcomes[{index}]")
        for index, item in enumerate(raw)
    )
    names = [outcome.name for outcome in outcomes]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise DefinitionError(f"{origin}: outcome(s) defined more than once: {repeated}")
    return outcomes


def _outcome(item: Any, *, origin: str) -> Outcome:
    if not isinstance(item, Mapping):
        raise DefinitionError(f"{origin} is not an object")
    name = _name(item, origin=origin)
    kind = item.get("type")
    if kind not in OUTCOME_TYPES:
        raise DefinitionError(
            f"{origin}: outcome {name!r} needs a type, one of {list(OUTCOME_TYPES)}, not "
            f"{kind!r}"
        )
    unknown = sorted(set(item) - _COMMON_KEYS - _TYPE_KEYS[kind])
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s) in outcome {name!r}: {unknown}")
    window = _window(item.get("window"), origin=origin)
    if kind == "event":
        return _event_outcome(name, item, origin=origin, window=window)
    if kind == "value":
        return _value_outcome(name, item, origin=origin, window=window)
    return _death_outcome(name, item, origin=origin, window=window)


def _name(item: Mapping[str, Any], *, origin: str) -> str:
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise DefinitionError(f"{origin}: an outcome needs a name")
    return name.strip()


def _window(raw: Any, *, origin: str) -> Window:
    if raw is None:
        return Window()
    if not isinstance(raw, Mapping):
        raise DefinitionError(f"{origin}: 'window' must be an object")
    unknown = sorted(set(raw) - {"from_days", "to_days"})
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s) in 'window': {unknown}")
    from_days = raw.get("from_days", 0)
    to_days = raw.get("to_days")
    if isinstance(from_days, bool) or not isinstance(from_days, int):
        raise DefinitionError(f"{origin}: 'from_days' must be an integer, not {from_days!r}")
    if to_days is not None and (isinstance(to_days, bool) or not isinstance(to_days, int)):
        raise DefinitionError(f"{origin}: 'to_days' must be an integer or null")
    if to_days is not None and from_days > to_days:
        raise DefinitionError(
            f"{origin}: 'from_days' ({from_days}) is after 'to_days' ({to_days})"
        )
    return Window(from_days, to_days)


def _event_outcome(
    name: str, item: Mapping[str, Any], *, origin: str, window: Window
) -> EventOutcome:
    table = item.get("table")
    if table not in EVENT_TABLES:
        raise DefinitionError(
            f"{origin}: outcome {name!r} needs a table, one of {list(EVENT_TABLES)}, not "
            f"{table!r}"
        )
    return EventOutcome(name, table, _codes(item.get("codes"), origin=origin, name=name), window)


def _codes(raw: Any, *, origin: str, name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise DefinitionError(f"{origin}: outcome {name!r} needs a non-empty 'codes' list")
    cleaned: list[str] = []
    for code in raw:
        if not isinstance(code, str) or not code.strip():
            raise DefinitionError(f"{origin}: outcome {name!r} codes must be non-empty strings")
        if code.strip() in cleaned:
            raise DefinitionError(f"{origin}: outcome {name!r} repeats code {code.strip()!r}")
        cleaned.append(code.strip())
    return tuple(cleaned)


def _value_outcome(
    name: str, item: Mapping[str, Any], *, origin: str, window: Window
) -> ValueOutcome:
    code = item.get("code")
    if not isinstance(code, str) or not code.strip():
        raise DefinitionError(f"{origin}: outcome {name!r} needs one observation 'code'")
    pick = item.get("pick", DEFAULT_PICK)
    if pick not in PICKS:
        raise DefinitionError(
            f"{origin}: outcome {name!r} pick must be one of {list(PICKS)}, not {pick!r}"
        )
    return ValueOutcome(name, code.strip(), pick, window)


def _death_outcome(
    name: str, item: Mapping[str, Any], *, origin: str, window: Window
) -> DeathOutcome:
    raw = item.get("cause_values", [])
    if not isinstance(raw, list) or not all(
        isinstance(value, str) and value.strip() for value in raw
    ):
        raise DefinitionError(
            f"{origin}: outcome {name!r} 'cause_values' must be a list of non-empty strings"
        )
    return DeathOutcome(name, tuple(value.strip() for value in raw), window)
