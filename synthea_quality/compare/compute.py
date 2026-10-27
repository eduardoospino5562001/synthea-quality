"""[COMPARE] Extract the figures of a module validation and compare two of them.

The extraction reads the typed models of
:mod:`synthea_quality.validate.models` and turns every measured value into a
:class:`~synthea_quality.compare.models.Figure` with a stable key. Nothing is
recalculated here: the rates, intervals and counts come from
:func:`synthea_quality.validate.build.build_module_validation`. A skipped source
becomes a skipped figure with its reason, never a zero.

Two figures of the same report never share a key: a repeated key raises
:class:`ValueError` instead of silently overwriting one of them.

Pure functions: no pandas, no network, no file system.
"""

from __future__ import annotations

from typing import TypedDict

from synthea_quality.compare.models import (
    CHANGED,
    COMPUTED,
    COUNT,
    DAYS,
    ONLY_IN_BASELINE,
    ONLY_IN_CANDIDATE,
    PER_1000_PERSON_YEARS,
    PERCENT,
    PROPORTION,
    SKIPPED,
    UNCHANGED,
    Comparison,
    ComparisonKey,
    Figure,
)
from synthea_quality.indexed.models import (
    DeathResult,
    EventResult,
    SkippedOutcome,
    ValueResult,
)
from synthea_quality.prevalence.models import Rate
from synthea_quality.profile.models import SectionStatus
from synthea_quality.validate.models import ModuleValidationReport

#: Statistics of an observation summary, in the order the report lists them.
OBSERVATION_STATISTICS = ("n", "minimum", "p5", "p25", "median", "p75", "p95", "maximum")

#: The parts of an indexed breakdown, in the order they are reported.
BREAKDOWN_PARTS = ("total", "sex", "age")

#: Population sections whose metrics and distributions become figures.
POPULATION_SECTIONS = ("population", "age", "distribution.GENDER")

#: Measures of a condition's prevalence and incidence, with their units.
CONDITION_MEASURES = (
    ("point", PROPORTION),
    ("lifetime", PROPORTION),
    ("incidence", PER_1000_PERSON_YEARS),
)

class _ElementInfo(TypedDict):
    """Por elemento (block + element): medidas salteadas, medidas calculadas
    y una razón por medida salteada."""

    skipped: set[str]
    computed: set[str]
    reasons: dict[str, str]



def _key(block: str, element: str, measure: str, stratum: str = "") -> ComparisonKey:
    """The stable key of one figure."""
    return ComparisonKey(block=block, element=element, measure=measure, stratum=stratum)


def _rate_figure(unit: str, rate: Rate) -> Figure:
    """A figure from a rate-like result (value, interval, numerator, denominator)."""

    interval = rate.interval
    return Figure(
        unit=unit,
        value=rate.value,
        ci95_low=interval[0] if interval is not None else None,
        ci95_high=interval[1] if interval is not None else None,
        numerator=rate.numerator,
        denominator=rate.denominator,
        state=COMPUTED,
    )


def _skipped_figure(unit: str, reason: str) -> Figure:
    """A figure whose source could not be measured."""

    return Figure(unit=unit, state=SKIPPED, reason=reason)


def _population_figures(report: ModuleValidationReport) -> list[tuple[ComparisonKey, Figure]]:
    """The numeric metrics and the distribution counts of the profile sections."""

    pairs: list[tuple[ComparisonKey, Figure]] = []
    for section in report.population:
        if section.section_id not in POPULATION_SECTIONS:
            continue
        for name, value in section.metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            unit = PERCENT if str(name).endswith("_percent") else COUNT
            pairs.append(
                (
                    _key("population", section.section_id, str(name), ""),
                    Figure(unit=unit, value=value, state=COMPUTED),
                )
            )
        for dist_name, rows in section.distributions.items():
            for row in rows:
                stratum = f"{dist_name}:{row.value if row.value is not None else ''}"
                pairs.append(
                    (
                        _key("population", section.section_id, "count", stratum),
                        Figure(unit=COUNT, value=row.count, numerator=row.count, state=COMPUTED),
                    )
                )
    return pairs


def _condition_figures(report: ModuleValidationReport) -> list[tuple[ComparisonKey, Figure]]:
    """The prevalence and incidence of every condition, in total and by stratum."""

    pairs: list[tuple[ComparisonKey, Figure]] = []
    for condition in report.conditions:
        prevalence_skipped = condition.prevalence.status is SectionStatus.SKIPPED
        incidence_skipped = condition.incidence.status is SectionStatus.SKIPPED
        if prevalence_skipped:
            reason = condition.prevalence.reason or condition.incidence.reason or "skipped"
            for measure, unit in CONDITION_MEASURES:
                pairs.append((_key("conditions", condition.name, measure, ""), _skipped_figure(unit, reason)))
            continue
        point = condition.prevalence.point
        lifetime = condition.prevalence.lifetime
        if point is not None:
            pairs.append((_key("conditions", condition.name, "point", ""), _rate_figure(PROPORTION, point)))
        if lifetime is not None:
            pairs.append((_key("conditions", condition.name, "lifetime", ""), _rate_figure(PROPORTION, lifetime)))
        if incidence_skipped:
            pairs.append(
                (
                    _key("conditions", condition.name, "incidence", ""),
                    _skipped_figure(PER_1000_PERSON_YEARS, condition.incidence.reason or "skipped"),
                )
            )
        else:
            rate = condition.incidence.rate
            if rate is not None:
                pairs.append(
                    (
                        _key("conditions", condition.name, "incidence", ""),
                        Figure(
                            unit=PER_1000_PERSON_YEARS,
                            value=rate.value,
                            ci95_low=(rate.interval[0] if rate.interval is not None else None),
                            ci95_high=(rate.interval[1] if rate.interval is not None else None),
                            numerator=rate.events,
                            denominator=rate.person_days,
                            state=COMPUTED,
                        ),
                    )
                )
        for stratum in condition.prevalence.strata:
            stratum_key = f"{stratum.dimension}:{stratum.value if stratum.value is not None else ''}"
            for measure, rate_value in (("point", stratum.point), ("lifetime", stratum.lifetime)):
                pairs.append(
                    (_key("conditions", condition.name, measure, stratum_key), _rate_figure(PROPORTION, rate_value))
                )
        for inc_stratum in condition.incidence.strata:
            stratum_key = f"{inc_stratum.dimension}:{inc_stratum.value if inc_stratum.value is not None else ''}"
            key = _key("conditions", condition.name, "incidence", stratum_key)
            interval = inc_stratum.rate.interval
            pairs.append(
                (
                    key,
                    Figure(
                        unit=PER_1000_PERSON_YEARS,
                        value=inc_stratum.rate.value,
                        ci95_low=interval[0] if interval is not None else None,
                        ci95_high=interval[1] if interval is not None else None,
                        numerator=inc_stratum.rate.events,
                        denominator=inc_stratum.rate.person_days,
                        state=COMPUTED,
                    ),
                )
            )
    return pairs


def _observation_figures(report: ModuleValidationReport) -> list[tuple[ComparisonKey, Figure]]:
    """The statistics of every observation summary, in total and by stratum."""

    pairs: list[tuple[ComparisonKey, Figure]] = []
    for observation in report.observations:
        if observation.status is SectionStatus.SKIPPED:
            pairs.append(
                (
                    _key("observations", observation.name, "value", ""),
                    _skipped_figure(COUNT, observation.reason or "skipped"),
                )
            )
            continue
        for group in observation.groups:
            units = group.units if group.units else ""
            unit = units if units else COUNT
            numerator = group.summary.n
            for statistic in OBSERVATION_STATISTICS:
                value = getattr(group.summary, statistic)
                stat_unit = COUNT if statistic == "n" else unit
                pairs.append(
                    (
                        _key("observations", observation.name, statistic, units),
                        Figure(unit=stat_unit, value=value, numerator=numerator, state=COMPUTED),
                    )
                )
            for stratum in group.strata:
                parts = []
                if units:
                    parts.append(units)
                parts.append(f"{stratum.dimension}:{stratum.value if stratum.value is not None else ''}")
                stratum_key = "|".join(parts)
                for statistic in OBSERVATION_STATISTICS:
                    value = getattr(stratum.summary, statistic)
                    stat_unit = COUNT if statistic == "n" else unit
                    pairs.append(
                        (
                            _key("observations", observation.name, statistic, stratum_key),
                            Figure(
                                unit=stat_unit,
                                value=value,
                                numerator=stratum.patients,
                                state=COMPUTED,
                            ),
                        )
                    )
    return pairs


def _medication_figures(report: ModuleValidationReport) -> list[tuple[ComparisonKey, Figure]]:
    """The active and ever share of every medication."""

    pairs: list[tuple[ComparisonKey, Figure]] = []
    for medication in report.medications:
        if medication.status is SectionStatus.SKIPPED:
            reason = medication.reason or "skipped"
            for measure in ("active", "ever"):
                pairs.append((_key("medications", medication.name, measure, ""), _skipped_figure(PROPORTION, reason)))
            continue
        if medication.active is not None:
            pairs.append(
                (
                    _key("medications", medication.name, "active", ""),
                    _rate_figure(PROPORTION, medication.active),
                )
            )
        if medication.ever is not None:
            pairs.append(
                (
                    _key("medications", medication.name, "ever", ""),
                    _rate_figure(PROPORTION, medication.ever),
                )
            )
    return pairs


def _indexed_figures(report: ModuleValidationReport) -> list[tuple[ComparisonKey, Figure]]:
    """The size of the indexed cohort, its outcomes and its crossings."""

    pairs: list[tuple[ComparisonKey, Figure]] = []
    indexed = report.indexed
    if indexed is None:
        return pairs
    if indexed.reason is not None:
        pairs.append((_key("indexed", "cohort", "patients", ""), _skipped_figure(COUNT, indexed.reason)))
        return pairs
    pairs.append(
        (
            _key("indexed", "cohort", "patients", ""),
            Figure(unit=COUNT, value=indexed.patients, numerator=indexed.patients, state=COMPUTED),
        )
    )
    for outcome in indexed.outcomes:
        if isinstance(outcome, SkippedOutcome):
            pairs.append((_key("indexed", outcome.name, outcome.type, ""), _skipped_figure(COUNT, outcome.reason)))
            continue
        breakdown = outcome.breakdown
        parts = (
            ("total", (breakdown.total,)),
            ("sex", breakdown.by_sex),
            ("age", breakdown.by_age),
        )
        for part, strata in parts:
            for stratum in strata:
                stratum_key = f"{part}:{stratum.label}"
                key = _key("indexed", outcome.name, "patients", stratum_key)
                rate = Rate(stratum.patients, stratum.cohort)
                interval = rate.interval
                pairs.append(
                    (
                        key,
                        Figure(
                            unit=PROPORTION,
                            value=rate.value,
                            ci95_low=interval[0] if interval is not None else None,
                            ci95_high=interval[1] if interval is not None else None,
                            numerator=stratum.patients,
                            denominator=stratum.cohort,
                            state=COMPUTED,
                        ),
                    )
                )
        if isinstance(outcome, ValueResult):
            for share in outcome.values:
                key = _key("indexed", outcome.name, "value", f"value:{share.value}")
                pairs.append(
                    (
                        key,
                        Figure(unit=COUNT, value=share.patients, numerator=share.patients, state=COMPUTED),
                    )
                )
        if isinstance(outcome, (EventResult, DeathResult)):
            pairs.append(
                (
                    _key("indexed", outcome.name, "days_median", ""),
                    Figure(unit=DAYS, value=outcome.days.median, state=COMPUTED),
                )
            )
            if isinstance(outcome, DeathResult):
                pairs.append(
                    (
                        _key("indexed", outcome.name, "with_cause", ""),
                        Figure(unit=COUNT, value=outcome.with_cause, state=COMPUTED),
                    )
                )
    for cross in indexed.crosses:
        if cross.reason is not None:
            pairs.append((_key("indexed", cross.outcome, "cross", cross.by), _skipped_figure(PROPORTION, cross.reason)))
            continue
        for cross_stratum in cross.strata:
            key = _key("indexed", cross.outcome, "cross", f"{cross.by}:{cross_stratum.label}")
            rate = Rate(cross_stratum.numerator, cross_stratum.denominator)
            interval = rate.interval
            pairs.append(
                (
                    key,
                    Figure(
                        unit=PROPORTION,
                        value=rate.value,
                        ci95_low=interval[0] if interval is not None else None,
                        ci95_high=interval[1] if interval is not None else None,
                        numerator=cross_stratum.numerator,
                        denominator=cross_stratum.denominator,
                        state=COMPUTED,
                    ),
                )
            )
    return pairs


def extract_figures(report: ModuleValidationReport) -> dict[ComparisonKey, Figure]:
    """Every measurable figure of one report, keyed by its stable key.

    The order is population, conditions, observations, medications and the
    indexed cohort; within each block the order of the report. Two figures of
    the same report never share a key: a repeated key raises
    :class:`ValueError` naming it.
    """

    figures: dict[ComparisonKey, Figure] = {}
    for block in (
        _population_figures(report),
        _condition_figures(report),
        _observation_figures(report),
        _medication_figures(report),
        _indexed_figures(report),
    ):
        for key, figure in block:
            if key in figures:
                raise ValueError(f"duplicate comparison key: {key}")
            figures[key] = figure
    return figures


def _element_info(
    figures: dict[ComparisonKey, Figure],
) -> dict[tuple[str, str], _ElementInfo]:
    """Por elemento (block + element): las medidas salteadas, las medidas
    calculadas y una razón por medida salteada.

    Un elemento salteado tapa las cifras del otro lado cuando esa medida no se
    puede medir en ninguno de los dos lados: si el elemento está entero
    salteado se tapan todas sus claves; si sólo algunas medidas están
    salteadas (por ejemplo un cruce), se tapan las claves de esas medidas y
    las demás se comparan con sus propias claves.
    """

    info: dict[tuple[str, str], _ElementInfo] = {}
    for key, figure in figures.items():
        element = (key.block, key.element)
        if element not in info:
            info[element] = {"skipped": set(), "computed": set(), "reasons": {}}
        skipped, computed, reasons = (
            info[element]["skipped"],
            info[element]["computed"],
            info[element]["reasons"],
        )
        if figure.state == SKIPPED:
            skipped.add(key.measure)
            reasons.setdefault(key.measure, figure.reason or "skipped")
        else:
            computed.add(key.measure)
    return info


def _indexed_cohort_skip_reason(figures: dict[ComparisonKey, Figure]) -> str | None:
    """La razón por la que la cohorte indexada entera no se pudo construir, o None."""

    for key, figure in figures.items():
        if key.block == "indexed" and key.element == "cohort" and figure.state == SKIPPED:
            return figure.reason or "skipped"
    return None


def _covers(
    other_info: dict[tuple[str, str], _ElementInfo],
    element: tuple[str, str],
    measure: str,
    self_computed_measures: set[str],
) -> bool:
    """¿Tapa el lado ``other`` la cifra de ``measure`` de ``element``?

    El otro lado tapa la medida cuando la tiene salteada y, por ese elemento,
    no hay nada computado del otro lado que la contradiga: si el elemento está
    entero salteado se tapan todas sus claves; si sólo falta esa medida, se
    tapan las claves de esa medida y las demás se comparan por su propia clave.
    """

    data = other_info.get(element)
    if not data or not data["skipped"]:
        return False
    skipped, computed = data["skipped"], data["computed"]
    if computed:
        return measure in skipped
    if skipped & self_computed_measures:
        return measure in skipped
    return True


def _cover_reason(
    info: dict[tuple[str, str], _ElementInfo], element: tuple[str, str], measure: str
) -> str:
    """Razón de la cifra salteada de ``measure`` en ``element``, o cualquiera del elemento."""

    reasons = info[element]["reasons"]
    reason = reasons.get(measure)
    if reason is None:
        reason = next(iter(reasons.values()))
    return reason


def _skipped_comparison(
    key: ComparisonKey, baseline: Figure | None, candidate: Figure | None, reason: str
) -> Comparison:
    """Una comparación ``skipped``: sin diferencia, con la razón del lado que saltó."""

    if baseline is not None:
        unit = baseline.unit
    elif candidate is not None:
        unit = candidate.unit
    else:
        raise AssertionError("una comparación skipped siempre tiene un lado")
    return Comparison(
        key=key,
        unit=unit,
        baseline=baseline,
        candidate=candidate,
        difference=None,
        relative_difference=None,
        state=SKIPPED,
        reason=reason,
    )


def compare(baseline: dict[ComparisonKey, Figure], candidate: dict[ComparisonKey, Figure]) -> list[Comparison]:
    """Pair the figures of two reports by key, in a stable order.

    Every key is emitted exactly once: first the keys of the baseline mapping,
    in its order, then the keys found only in the candidate mapping, in its
    order. The result is symmetric: swapping the sides interchanges
    ``only_in_baseline`` and ``only_in_candidate`` and keeps every other state.

    A figure missing on the other side comes out ``skipped`` with the reason
    when the other side could not measure that element: an entirely skipped
    element covers every key of the same element on the other side, and a
    partially skipped element covers the keys of the measures it skipped. A
    skipped indexed cohort covers the whole ``indexed`` block of the other
    side. A skipped figure without a partner comes out ``skipped`` with its
    own reason. Only figures that genuinely exist on one side (for example a
    new age band) are ``only_in_*``.
    """

    comparisons: list[Comparison] = []

    base_info = _element_info(baseline)
    cand_info = _element_info(candidate)
    base_cohort_reason = _indexed_cohort_skip_reason(baseline)
    cand_cohort_reason = _indexed_cohort_skip_reason(candidate)

    for key in [*baseline, *(key for key in candidate if key not in baseline)]:
        element = (key.block, key.element)
        base_figure = baseline.get(key)
        cand_figure = candidate.get(key)
        base_measures = base_info[element]["computed"] if element in base_info else set()
        cand_measures = cand_info[element]["computed"] if element in cand_info else set()

        if base_figure is not None and cand_figure is not None:
            comparisons.append(_pair(key, base_figure, cand_figure))
            continue

        if base_figure is not None:
            # The figure exists only on the baseline side.
            if _covers(cand_info, element, key.measure, base_measures):
                comparisons.append(
                    _skipped_comparison(
                        key, base_figure, None, _cover_reason(cand_info, element, key.measure)
                    )
                )
            elif cand_cohort_reason is not None and key.block == "indexed":
                comparisons.append(_skipped_comparison(key, base_figure, None, cand_cohort_reason))
            elif base_figure.state == SKIPPED:
                comparisons.append(
                    _skipped_comparison(key, base_figure, None, base_figure.reason or "skipped")
                )
            else:
                comparisons.append(_pair(key, base_figure, None))
        else:
            # The figure exists only on the candidate side: the key comes from
            # the candidate mapping, so its figure is never None here.
            assert cand_figure is not None
            if _covers(base_info, element, key.measure, cand_measures):
                comparisons.append(
                    _skipped_comparison(
                        key, None, cand_figure, _cover_reason(base_info, element, key.measure)
                    )
                )
            elif base_cohort_reason is not None and key.block == "indexed":
                comparisons.append(_skipped_comparison(key, None, cand_figure, base_cohort_reason))
            elif cand_figure.state == SKIPPED:
                comparisons.append(
                    _skipped_comparison(key, None, cand_figure, cand_figure.reason or "skipped")
                )
            else:
                comparisons.append(_pair(key, None, cand_figure))
    return comparisons


def _pair(key: ComparisonKey, baseline: Figure | None, candidate: Figure | None) -> Comparison:
    """One baseline/candidate pair, with its difference and its state."""

    if baseline is None:
        if candidate is None:
            raise ValueError(f"cannot compare two missing figures: {key}")
        return Comparison(
            key=key,
            unit=candidate.unit,
            baseline=None,
            candidate=candidate,
            difference=None,
            relative_difference=None,
            state=ONLY_IN_CANDIDATE,
        )
    if candidate is None:
        return Comparison(
            key=key,
            unit=baseline.unit,
            baseline=baseline,
            candidate=None,
            difference=None,
            relative_difference=None,
            state=ONLY_IN_BASELINE,
        )
    if baseline.state == SKIPPED or candidate.state == SKIPPED:
        return Comparison(
            key=key,
            unit=baseline.unit,
            baseline=baseline,
            candidate=candidate,
            difference=None,
            relative_difference=None,
            state=SKIPPED,
            reason=baseline.reason or candidate.reason,
        )
    base_value = baseline.value
    cand_value = candidate.value
    if base_value is None or cand_value is None:
        state = UNCHANGED if base_value is None and cand_value is None else CHANGED
        return Comparison(
            key=key,
            unit=baseline.unit,
            baseline=baseline,
            candidate=candidate,
            difference=None,
            relative_difference=None,
            state=state,
        )
    difference = round(cand_value - base_value, 6)
    relative_difference = None if base_value == 0 else (cand_value - base_value) / base_value
    state = UNCHANGED if cand_value == base_value else CHANGED
    return Comparison(
        key=key,
        unit=baseline.unit,
        baseline=baseline,
        candidate=candidate,
        difference=difference,
        relative_difference=relative_difference,
        state=state,
    )


def largest_changes(comparisons: list[Comparison], n: int) -> list[Comparison]:
    """The ``n`` largest changes by |relative difference|, ties by key.

    Relative differences of ``None`` sort last; only ``"changed"`` comparisons
    are ranked. When ``n`` is greater than the number of changes they all fit.
    """

    changed = [comparison for comparison in comparisons if comparison.state == CHANGED]

    def sort_key(comparison: Comparison) -> tuple[bool, float, ComparisonKey]:
        relative = comparison.relative_difference
        return (relative is None, -(abs(relative) if relative is not None else 0.0), comparison.key)

    changed.sort(key=sort_key)
    return changed[: max(n, 0)]
