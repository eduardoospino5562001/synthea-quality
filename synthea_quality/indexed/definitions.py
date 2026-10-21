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
    An event of several tables or with several rules uses ``match``: a list of parts,
    each with its own ``table``, ``codes`` and filters. ``op`` says how the parts
    combine: ``any`` (the default) counts a patient with one part, ``all`` needs every
    part.
``type: value``
    ``code`` of ``observations`` and ``pick`` (``first`` or ``last``): the value on the
    earliest or latest date of the window. ``groups`` (optional) joins several exact
    ``VALUE`` texts under one label, so that a patient with two rows of the same day
    counts once.
``type: death``
    ``cause_values`` (optional): exact ``VALUE`` texts of observation ``69453-9``.
``by`` (optional)
    A list of ``{"outcome": X, "by": Y}``: the patients of X in each stratum of Y. Y
    has to be a ``value`` or an ``event``; X can be of any type.

The filters of an event
    A part of an event narrows the records it accepts. ``values`` keeps the records
    whose ``VALUE`` is one of the exact texts (only ``observations`` has a ``VALUE``);
    ``reasons`` keeps the records whose ``REASONCODE`` is one of the exact codes
    (only ``encounters``, ``procedures``, ``medications`` and ``careplans`` have one);
    ``at`` says when the record counts, on its ``START`` date (the default) or on its
    ``STOP`` date (only the tables that have a ``STOP``). The three filters work the
    same in an event of one table and in a part of a ``match``.

Anything ambiguous is an input error, never a guess: an ``index`` without ``outcomes``
or the reverse, an empty ``outcomes`` list, a repeated name, an unknown key or type, a
table outside :data:`EVENT_TABLES`, an empty ``codes`` list, a bad ``pick``, a window
whose ``from_days`` is after its ``to_days``, a malformed ``groups`` object, or a
malformed ``by`` list. The same goes for the parts of an event: a ``match`` together
with the keys of an event of one table, a ``match`` of less than two parts, an ``op``
without a ``match`` or outside :data:`EVENT_OPS`, an unknown key in a part, a table or
a list of codes that a part cannot use, a null list of ``values`` or ``reasons``, a
filter on a table that has no such column, an ``at`` outside :data:`EVENT_AT`, a
repeated part name, and two parts that say the same thing.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
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
#: How the parts of a ``match`` combine: one part is enough, or all of them are needed.
EVENT_OPS = ("any", "all")
DEFAULT_OP = "any"
#: When a record of an event counts: on its start date or on its stop date.
EVENT_AT = ("start", "stop")
DEFAULT_AT = "start"
#: The tables whose records have a ``STOP`` date, the ones ``at: stop`` can use.
STOP_TABLES = (
    "conditions",
    "encounters",
    "procedures",
    "medications",
    "careplans",
    "devices",
)
#: The tables whose records have a ``REASONCODE``, the ones ``reasons`` can use.
REASON_TABLES = ("encounters", "procedures", "medications", "careplans")
#: The tables whose records have a ``VALUE``, the ones ``values`` can use.
VALUE_TABLES = ("observations",)
#: The label of the stratum with no value; a group of a ``value`` outcome cannot use it,
#: because the report would show two different strata with the same name.
NO_VALUE_LABEL = "no value"

_COMMON_KEYS = {"name", "type", "window"}
_TYPE_KEYS = {
    "event": {"table", "codes", "values", "reasons", "at", "match", "op"},
    "value": {"code", "pick", "groups"},
    "death": {"cause_values"},
}
#: The keys of one part of an event outcome.
_PART_KEYS = {"name", "table", "codes", "values", "reasons", "at"}
#: The keys of an event outcome that a part cannot have: they describe the whole
#: outcome, and a ``match`` says the outcome has parts instead.
_EVENT_WHOLE_KEYS = {"table", "codes", "values", "reasons", "at"}
#: What makes two parts of an outcome the same rule.
_PartKey = tuple[str, frozenset[str], frozenset[str], frozenset[str], str]


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
class EventMatch:
    """One part of an event outcome: a table, its codes and its filters.

    A part is a rule of its own: the records of its codes in its table, narrowed by the
    exact ``VALUE`` texts (``values``), by the exact ``REASONCODE`` codes (``reasons``)
    and by the date they count on (``at``).
    """

    #: The label of the part in the report.
    name: str
    #: The table the part reads.
    table: str
    #: The codes that select a record.
    codes: tuple[str, ...]
    #: The exact ``VALUE`` texts a record must have; only ``observations`` has them.
    values: tuple[str, ...] = ()
    #: The exact ``REASONCODE`` codes a record must have.
    reasons: tuple[str, ...] = ()
    #: When the record counts: on its ``START`` or on its ``STOP`` date.
    at: str = DEFAULT_AT


@dataclass(frozen=True, slots=True)
class EventOutcome:
    """Patients with a record of one of ``codes`` in ``table`` within the window.

    An outcome of one table keeps its ``table`` and ``codes``. An outcome of several
    parts has ``matches`` instead, and no table of its own; ``op`` says whether one
    part is enough (``any``) or all of them are needed (``all``).
    """

    name: str
    table: str | None
    codes: tuple[str, ...]
    window: Window = field(default_factory=Window)
    #: The exact ``VALUE`` texts a record must have; only ``observations`` has them.
    values: tuple[str, ...] = ()
    #: The exact ``REASONCODE`` codes a record must have.
    reasons: tuple[str, ...] = ()
    #: When the records count: on their ``START`` or on their ``STOP`` date.
    at: str = DEFAULT_AT
    #: The parts of a composite outcome; empty for an outcome of one table.
    matches: tuple[EventMatch, ...] = ()
    #: How the parts combine: ``any`` or ``all``.
    op: str = DEFAULT_OP

    @property
    def parts(self) -> tuple[EventMatch, ...]:
        """The parts of the outcome: its ``matches``, or the one of its own table.

        The part of an outcome of one table takes the filters of the outcome and the
        default name ``"<table>: <code>, <code>"``.
        """
        if self.matches:
            return self.matches
        assert self.table is not None
        return (
            EventMatch(
                name=f"{self.table}: {', '.join(self.codes)}",
                table=self.table,
                codes=self.codes,
                values=self.values,
                reasons=self.reasons,
                at=self.at,
            ),
        )


@dataclass(frozen=True, slots=True)
class ValueOutcome:
    """The value of an observation on the first or last date of the window."""

    name: str
    code: str
    pick: str = DEFAULT_PICK
    window: Window = field(default_factory=Window)
    #: The exact ``VALUE`` texts of each group, in the order of the module file.
    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class DeathOutcome:
    """Deaths within the window, optionally narrowed by their recorded cause."""

    name: str
    cause_values: tuple[str, ...] = ()
    window: Window = field(default_factory=Window)


@dataclass(frozen=True, slots=True)
class CrossSpec:
    """One crossing: the patients of ``outcome`` in each stratum of ``by``."""

    outcome: str
    by: str


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


def load_crosses(
    data: Mapping[str, Any], *, origin: str, outcomes: Sequence[Outcome]
) -> tuple[CrossSpec, ...]:
    """The ``by`` crossings of a parsed module file (none without the key).

    ``outcomes`` are the outcomes of the same file: a crossing names two of them, and
    the one that gives the strata has to be a ``value`` or an ``event``.

    :raises DefinitionError: a crossing cannot be used as given.
    """
    if "by" not in data:
        return ()
    if not outcomes:
        raise DefinitionError(f"{origin}: 'by' needs an 'index' and its 'outcomes'")
    raw = data["by"]
    if not isinstance(raw, list) or not raw:
        raise DefinitionError(f"{origin}: 'by' must be a non-empty list")
    defined = {outcome.name: outcome for outcome in outcomes}
    crosses: list[CrossSpec] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        cross = _cross(item, origin=f"{origin} by[{index}]", outcomes=defined)
        pair = (cross.outcome, cross.by)
        if pair in seen:
            raise DefinitionError(
                f"{origin}: the pair ({cross.outcome!r}, {cross.by!r}) is crossed more "
                f"than once"
            )
        seen.add(pair)
        crosses.append(cross)
    return tuple(crosses)


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
    """Read an event outcome: one part of one table, or several parts with ``match``.

    An outcome has either ``table`` and ``codes`` or ``match``, never both: a part and
    a whole outcome would then say the same thing in two ways. The filters are the
    same in the two forms, so a part is read by the same rule in both.
    """
    if "match" in item:
        return _matched_event(name, item, origin=origin, window=window)
    if "op" in item:
        raise DefinitionError(
            _event_message(origin, name, "", "'op' needs a 'match' of several parts")
        )
    part = _event_part(
        {key: item[key] for key in _EVENT_WHOLE_KEYS if key in item},
        origin=origin,
        name=name,
        where="",
    )
    return EventOutcome(
        name, part.table, part.codes, window, part.values, part.reasons, part.at
    )


def _matched_event(
    name: str, item: Mapping[str, Any], *, origin: str, window: Window
) -> EventOutcome:
    """Read an event outcome of several parts and the way they combine."""
    mixed = sorted(set(item) & _EVENT_WHOLE_KEYS)
    if mixed:
        raise DefinitionError(
            _event_message(origin, name, "", f"'match' cannot be used with {mixed}")
        )
    raw = item.get("match")
    if not isinstance(raw, list) or len(raw) < 2:
        raise DefinitionError(
            _event_message(
                origin,
                name,
                "",
                "needs a 'match' list of at least two parts; an outcome of one table "
                "uses 'table' and 'codes'",
            )
        )
    op = item.get("op", DEFAULT_OP)
    if op not in EVENT_OPS:
        raise DefinitionError(
            _event_message(
                origin, name, "", f"'op' must be one of {list(EVENT_OPS)}, not {op!r}"
            )
        )
    parts = tuple(
        _event_part(
            part, origin=origin, name=name, where=f"part {index} of 'match'"
        )
        for index, part in enumerate(raw, start=1)
    )
    _check_part_names(parts, origin=origin, name=name)
    _check_repeated_parts(parts, origin=origin, name=name)
    return EventOutcome(name, None, (), window, matches=parts, op=op)


def _event_part(
    item: Any, *, origin: str, name: str, where: str
) -> EventMatch:
    """Read one part of an event outcome: its table, codes and filters.

    The same rule reads the part of an outcome of one table and the parts of a
    ``match``: both have the same keys and both are checked the same way.
    """
    if not isinstance(item, Mapping):
        raise DefinitionError(_event_message(origin, name, where, "is not an object"))
    unknown = sorted(set(item) - _PART_KEYS)
    if unknown:
        raise DefinitionError(
            _event_message(origin, name, where, f"unknown key(s): {unknown}")
        )
    table = item.get("table")
    if table not in EVENT_TABLES:
        raise DefinitionError(
            _event_message(
                origin,
                name,
                where,
                f"needs a table, one of {list(EVENT_TABLES)}, not {table!r}",
            )
        )
    codes = _codes(item.get("codes"), origin=origin, name=name, where=where)
    values = _texts(item, origin=origin, name=name, where=where, key="values")
    reasons = _texts(item, origin=origin, name=name, where=where, key="reasons")
    at = item.get("at", DEFAULT_AT)
    if at not in EVENT_AT:
        raise DefinitionError(
            _event_message(
                origin, name, where, f"'at' must be one of {list(EVENT_AT)}, not {at!r}"
            )
        )
    if values and table not in VALUE_TABLES:
        raise DefinitionError(
            _event_message(
                origin,
                name,
                where,
                f"'values' needs a table of {list(VALUE_TABLES)}, not {table!r}",
            )
        )
    if reasons and table not in REASON_TABLES:
        raise DefinitionError(
            _event_message(
                origin,
                name,
                where,
                f"'reasons' needs a table of {list(REASON_TABLES)}, not {table!r}",
            )
        )
    if at == "stop" and table not in STOP_TABLES:
        raise DefinitionError(
            _event_message(
                origin,
                name,
                where,
                f"'at: stop' needs a table of {list(STOP_TABLES)}, not {table!r}",
            )
        )
    return EventMatch(
        name=_part_name(
            item.get("name"),
            default=f"{table}: {', '.join(codes)}",
            origin=origin,
            name=name,
            where=where,
        ),
        table=table,
        codes=codes,
        values=values,
        reasons=reasons,
        at=at,
    )


def _part_name(
    raw: Any, *, default: str, origin: str, name: str, where: str
) -> str:
    """The name of a part: the one of the module file, or the default one."""
    if raw is None:
        return default
    if not isinstance(raw, str) or not raw.strip():
        raise DefinitionError(
            _event_message(origin, name, where, "a part needs a non-empty name")
        )
    return raw.strip()


def _check_part_names(
    parts: Sequence[EventMatch], *, origin: str, name: str
) -> None:
    """Every part of an outcome has its own name, the default one included.

    Two parts with the same name would show the same label twice in the report, and
    the reader could not tell which part each row describes.
    """
    seen: dict[str, int] = {}
    for index, part in enumerate(parts, start=1):
        if part.name in seen:
            raise DefinitionError(
                f"{origin}: outcome {name!r}: part {index} of 'match' repeats the name "
                f"{part.name!r} of part {seen[part.name]} of 'match'"
            )
        seen[part.name] = index


def _check_repeated_parts(
    parts: Sequence[EventMatch], *, origin: str, name: str
) -> None:
    """Two parts of the same outcome cannot say the same thing.

    The codes, the values and the reasons are compared as sets: the order of a list
    does not change what a part accepts.
    """
    seen: dict[_PartKey, int] = {}
    for index, part in enumerate(parts, start=1):
        key = (
            part.table,
            frozenset(part.codes),
            frozenset(part.values),
            frozenset(part.reasons),
            part.at,
        )
        if key in seen:
            raise DefinitionError(
                f"{origin}: outcome {name!r}: part {index} of 'match' is the same as "
                f"part {seen[key]} of 'match'"
            )
        seen[key] = index


def _event_message(origin: str, name: str, where: str, message: str) -> str:
    """The message of an event outcome, or of one of its parts."""
    if where:
        return f"{origin}: outcome {name!r}, {where}: {message}"
    return f"{origin}: outcome {name!r} {message}"


def _codes(raw: Any, *, origin: str, name: str, where: str = "") -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise DefinitionError(
            _event_message(origin, name, where, "needs a non-empty 'codes' list")
        )
    cleaned: list[str] = []
    for code in raw:
        if not isinstance(code, str) or not code.strip():
            raise DefinitionError(
                _event_message(origin, name, where, "codes must be non-empty strings")
            )
        if code.strip() in cleaned:
            raise DefinitionError(
                _event_message(origin, name, where, f"repeats code {code.strip()!r}")
            )
        cleaned.append(code.strip())
    return tuple(cleaned)


def _texts(
    item: Mapping[str, Any], *, origin: str, name: str, where: str, key: str
) -> tuple[str, ...]:
    """The exact texts a part filters on, stripped and without repeats.

    An absent key filters on nothing. A key with a null value is an error, like an
    empty list: a filter that was left out by mistake would otherwise read as a
    filter that keeps every record.
    """
    if key not in item:
        return ()
    raw = item[key]
    if not isinstance(raw, list) or not raw:
        raise DefinitionError(
            _event_message(
                origin, name, where, f"'{key}' must be a non-empty list of texts"
            )
        )
    cleaned: list[str] = []
    for text in raw:
        if not isinstance(text, str) or not text.strip():
            raise DefinitionError(
                _event_message(
                    origin, name, where, f"'{key}' texts must be non-empty strings"
                )
            )
        if text.strip() in cleaned:
            raise DefinitionError(
                _event_message(
                    origin, name, where, f"'{key}' repeats the text {text.strip()!r}"
                )
            )
        cleaned.append(text.strip())
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
    groups = _groups(item.get("groups"), origin=origin, name=name)
    return ValueOutcome(name, code.strip(), pick, window, groups)


def _groups(
    raw: Any, *, origin: str, name: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The groups of a ``value`` outcome: a label and the exact ``VALUE`` texts it joins.

    The groups keep the order of the module file, so that the report lists them the way
    the file does. A text belongs to one group only and a group cannot be empty. A label
    is unique once stripped: two labels cleaned into the same one would merge two groups
    without saying so. A label is not a text of *another* group either, because a
    ``VALUE`` with that text would then count under a different label than the group
    that names it; a label that is a text of its own group is fine.
    """
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or not raw:
        raise DefinitionError(
            f"{origin}: outcome {name!r} 'groups' must be a non-empty object"
        )
    groups: list[tuple[str, tuple[str, ...]]] = []
    owner: dict[str, str] = {}
    for label, texts in raw.items():
        if not isinstance(label, str) or not label.strip():
            raise DefinitionError(
                f"{origin}: outcome {name!r} 'groups' needs a non-empty label"
            )
        label = label.strip()
        if label == NO_VALUE_LABEL:
            raise DefinitionError(
                f"{origin}: outcome {name!r} 'groups' cannot use the label "
                f"{NO_VALUE_LABEL!r}"
            )
        if any(existing == label for existing, _ in groups):
            raise DefinitionError(
                f"{origin}: outcome {name!r} 'groups' repeats the label {label!r}"
            )
        if not isinstance(texts, list) or not texts:
            raise DefinitionError(
                f"{origin}: outcome {name!r} group {label!r} needs a non-empty list of "
                f"texts"
            )
        cleaned: list[str] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise DefinitionError(
                    f"{origin}: outcome {name!r} group {label!r} texts must be non-empty "
                    f"strings"
                )
            text = text.strip()
            if text in cleaned:
                raise DefinitionError(
                    f"{origin}: outcome {name!r} group {label!r} repeats the text {text!r}"
                )
            if text in owner:
                raise DefinitionError(
                    f"{origin}: outcome {name!r} 'groups' uses the text {text!r} in more "
                    f"than one group"
                )
            owner[text] = label
            cleaned.append(text)
        groups.append((label, tuple(cleaned)))
    for label, _ in groups:
        used_as = owner.get(label)
        if used_as is not None and used_as != label:
            raise DefinitionError(
                f"{origin}: outcome {name!r} 'groups' uses the label {label!r} as a "
                f"text of group {used_as!r}"
            )
    return tuple(groups)


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


def _cross(
    item: Any, *, origin: str, outcomes: Mapping[str, Outcome]
) -> CrossSpec:
    """One crossing of the ``by`` list, with the two outcomes it names."""
    if not isinstance(item, Mapping):
        raise DefinitionError(f"{origin} is not an object")
    unknown = sorted(set(item) - {"outcome", "by"})
    if unknown:
        raise DefinitionError(f"{origin}: unknown key(s): {unknown}")
    outcome = _cross_name(item.get("outcome"), origin=origin, key="outcome")
    by = _cross_name(item.get("by"), origin=origin, key="by")
    for name in (outcome, by):
        if name not in outcomes:
            raise DefinitionError(f"{origin}: {name!r} is not an outcome of the file")
    if outcome == by:
        raise DefinitionError(
            f"{origin}: an outcome cannot be crossed with itself ({outcome!r})"
        )
    if not isinstance(outcomes[by], (EventOutcome, ValueOutcome)):
        raise DefinitionError(
            f"{origin}: {by!r} cannot give the strata of a cross, which need a 'value' "
            f"or an 'event'"
        )
    return CrossSpec(outcome, by)


def _cross_name(raw: Any, *, origin: str, key: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise DefinitionError(f"{origin} needs the {key!r} name")
    return raw.strip()
