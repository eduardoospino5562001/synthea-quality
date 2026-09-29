"""[OBSERVATIONS] The distribution of observation values, from frames that are already loaded.

One value per patient
---------------------
A patient measured every month would otherwise weigh twelve times as much as one measured
once a year, and the patients measured most often are usually the sickest. So each alive
patient contributes **one value per code and units**: the latest whose ``DATE`` falls on or
before the reference date (a calendar day; a ``DATE`` on that day counts). When several
values share that latest ``DATE``, their median is used and the patient is counted as tied.

A latest value can be old. Every group reports how old the values are (median days before
the reference date, patients whose value is more than 1 and more than 3 years old), and a
definition may set ``lookback_years``: a patient whose latest value is older is left out of
the distribution and counted.

What is counted, then left out (per code, every count is in the result), in this order:

* rows of patients not in ``patients.csv``, then rows of deceased patients;
* rows whose ``DATE`` is empty or unparseable, then rows after the reference date;
* rows whose ``TYPE`` is not ``numeric`` (Synthea writes ``text`` for coded answers);
* rows with ``TYPE`` ``numeric`` whose ``VALUE`` is not a finite number.

Units
-----
``UNITS`` is part of what is described: the values of one code in two units are two
groups, **never converted** (a conversion would be a guess about what the unit strings
mean). The general table lists every code whose numeric rows use more than one unit and
every code written with more than one ``TYPE``, over the whole table.

Percentiles are linear interpolation between the closest ranks (type 7), the default of
``pandas.Series.quantile``; the 5th, 25th, 75th and 95th are given only with at least
``MIN_PATIENTS_FOR_PERCENTILES`` (10) values. Strata: age band at the reference date and
``GENDER`` of the patients described, as in the prevalence report.

This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from synthea_quality.incidence.compute import window_start
from synthea_quality.observations.definitions import ObservationDefinition, ReferenceRange
from synthea_quality.observations.models import (
    MIN_PATIENTS_FOR_PERCENTILES,
    PERCENTILES,
    GeneralRow,
    GeneralTable,
    ObservationResult,
    RangeComparison,
    UnitGroup,
    ValueStratum,
    ValueSummary,
)
from synthea_quality.prevalence.compute import Cohort, most_frequent_descriptions
from synthea_quality.profile.dates import parse_timestamps
from synthea_quality.profile.models import SectionStatus

#: Columns of ``observations.csv`` this module uses, and those it cannot do without.
OBSERVATION_COLUMNS = ("DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS", "TYPE")
REQUIRED_OBSERVATION_COLUMNS = ("DATE", "PATIENT", "CODE", "VALUE")

#: The ``TYPE`` of a row whose ``VALUE`` is a number.
NUMERIC_TYPE = "numeric"

#: What happens to a row, in the order the reasons are checked.
FATES = (
    "unknown_patient",
    "deceased",
    "date_unusable",
    "after_reference",
    "not_numeric_type",
    "value_unparseable",
    "used",
)

#: Ages of a latest value that every group counts.
STALE_YEARS = (1, 3)

DEFAULT_TOP = 30

_GROUP = ["CODE", "UNITS", "PATIENT"]


@dataclass(frozen=True, slots=True, eq=False)
class Values:
    """The latest usable value of every alive patient, and what happened to every row."""

    #: PATIENT, CODE, UNITS ("" when empty), DATE, VALUE, values (count sharing the DATE)
    latest: pd.DataFrame
    #: Rows per fate, per code.
    fates: dict[str, dict[str, int]]
    #: Rows per ``TYPE`` ("" when empty), per code, over the whole table.
    types: dict[str, dict[str, int]]
    #: Rows per ``UNITS`` of the rows with ``TYPE`` numeric, per code, over the whole table.
    units: dict[str, dict[str, int]]
    descriptions: dict[str, str | None]
    has_type: bool
    has_units: bool
    metrics: dict[str, int]
    notes: tuple[str, ...] = ()


def prepare_values(observations: pd.DataFrame, cohort: Cohort, reference: date) -> Values:
    """Classify every row of ``observations`` and keep each alive patient's latest values."""
    has_type = "TYPE" in observations.columns
    has_units = "UNITS" in observations.columns
    index = observations.index
    codes = observations["CODE"]
    coded = codes.notna()
    types = observations["TYPE"].fillna("") if has_type else pd.Series("", index=index)
    units = observations["UNITS"].fillna("") if has_units else pd.Series("", index=index)

    known = observations["PATIENT"].isin(cohort.patient_ids)
    alive = observations["PATIENT"].isin(cohort.alive_ids)
    dates = parse_timestamps(observations["DATE"]).values
    numbers = pd.to_numeric(observations["VALUE"], errors="coerce")
    finite = numbers.notna() & (numbers.abs() != float("inf"))
    numeric_type = types == NUMERIC_TYPE if has_type else pd.Series(True, index=index)

    fate = pd.Series("used", index=index, dtype="object")
    reasons = (
        ("unknown_patient", ~known),
        ("deceased", known & ~alive),
        ("date_unusable", dates.isna()),
        ("after_reference", dates.dt.normalize() > pd.Timestamp(reference)),
        ("not_numeric_type", ~numeric_type),
        ("value_unparseable", ~finite),
    )
    for name, mask in reversed(reasons):  # the first reason that applies wins
        fate[mask.fillna(False).astype(bool)] = name

    used = pd.DataFrame(
        {
            "PATIENT": observations["PATIENT"],
            "CODE": codes,
            "UNITS": units,
            "DATE": dates,
            "VALUE": numbers,
        }
    )[coded & (fate == "used")]
    last = used.groupby(_GROUP)["DATE"].transform("max")
    latest = (
        used[used["DATE"] == last]
        .groupby(_GROUP, sort=True)
        .agg(DATE=("DATE", "first"), VALUE=("VALUE", "median"), values=("VALUE", "size"))
        .reset_index()
    )

    descriptions = most_frequent_descriptions(observations, pd.Series("", index=index), coded)
    notes = []
    if not has_type:
        notes.append(
            "observations.csv has no TYPE column: every row whose VALUE is a number is used."
        )
    if not has_units:
        notes.append(
            "observations.csv has no UNITS column: values of one code cannot be told apart "
            "by unit."
        )
    metrics = {f"rows_{name}": int((fate[coded] == name).sum()) for name in FATES}
    metrics["rows"] = len(observations)
    metrics["rows_without_code"] = int((~coded).sum())
    return Values(
        latest=latest,
        fates=_counts(codes[coded], fate[coded], FATES),
        types=_counts(codes[coded], types[coded]),
        units=_counts(codes[coded & numeric_type], units[coded & numeric_type]),
        descriptions={code: text for (_, code), text in descriptions.items()},
        has_type=has_type,
        has_units=has_units,
        metrics=metrics,
        notes=tuple(notes),
    )


def summarise(values: pd.Series) -> ValueSummary:
    """n, minimum, the percentiles of :data:`PERCENTILES` and maximum of ``values``.

    Below :data:`MIN_PATIENTS_FOR_PERCENTILES` values, only the median of the percentiles.
    """
    values = values.dropna()
    if values.empty:
        return ValueSummary(0)
    quantiles = values.quantile([p / 100 for p in PERCENTILES]).tolist()
    p5, p25, median, p75, p95 = (float(q) for q in quantiles)
    if len(values) < MIN_PATIENTS_FOR_PERCENTILES:
        p5 = p25 = p75 = p95 = None
    return ValueSummary(
        n=len(values),
        minimum=float(values.min()),
        p5=p5,
        p25=p25,
        median=median,
        p75=p75,
        p95=p95,
        maximum=float(values.max()),
    )


# --------------------------------------------------------------------------- #
# observations asked for
# --------------------------------------------------------------------------- #


def observation_values(
    definition: ObservationDefinition,
    values: Values,
    population: Cohort,
    reference: date,
    *,
    members: pd.Index | None = None,
) -> ObservationResult:
    """The distribution of one code's latest values among ``members``, one group per unit.

    :param members: the patients to describe (a cohort of alive patients); ``None`` means
        every alive patient of ``population``.
    """
    code = definition.code
    description = values.descriptions.get(code)
    people = population.alive_ids if members is None else members
    common: dict[str, Any] = dict(
        name=definition.name or description or code,
        code=code,
        description=description,
        cohort=definition.cohort,
        population=len(people),
        lookback_years=definition.lookback_years,
        reference_range=definition.reference_range,
    )
    fates = values.fates.get(code)
    if fates is None:
        return ObservationResult(
            status=SectionStatus.SKIPPED,
            reason=f"no row of code {code} in observations.csv (any patient, any date)",
            **common,
        )
    metrics: dict[str, Any] = {
        **{f"rows_{name}": fates.get(name, 0) for name in FATES},
        "rows_by_type": values.types.get(code, {}),
    }
    rows = values.latest[(values.latest["CODE"] == code) & values.latest["PATIENT"].isin(people)]
    whom = "patient of the cohort" if members is not None else "alive patient"
    if rows.empty:
        return ObservationResult(
            status=SectionStatus.SKIPPED,
            reason=f"no {whom} has a usable numeric value of {code} on or before the "
            f"reference date ({_fate_text(fates)})",
            metrics=metrics,
            notes=_row_notes(fates, values.types.get(code, {})),
            **common,
        )

    counts = rows.groupby("UNITS")["PATIENT"].nunique()
    order = sorted(counts.index, key=lambda u: (-counts[u], u))
    lookback = (
        window_start(reference, definition.lookback_years) if definition.lookback_years else None
    )
    groups = tuple(
        _group(rows[rows["UNITS"] == units], units, definition.reference_range, population,
               people, reference, lookback)
        for units in order
    )
    in_several = int((rows.groupby("PATIENT")["UNITS"].nunique() > 1).sum())
    metrics["patients_in_several_units"] = in_several

    notes = list(_row_notes(fates, values.types.get(code, {})))
    if len(groups) > 1:
        notes.append(
            f"The values of {code} come in {len(groups)} units, each described separately and "
            f"never converted; {in_several} patient(s) have a latest value in more than one."
        )
    tied = sum(g.metrics["patients_tied_latest"] for g in groups)
    if tied:
        notes.append(
            f"{tied} patient(s) have several values on their latest DATE; the median of those "
            f"values is used."
        )
    if definition.lookback_years:
        outside = sum(g.metrics["patients_outside_lookback"] for g in groups)
        notes.append(
            f"Lookback of {definition.lookback_years} year(s) (from {lookback}): {outside} "
            f"patient(s) whose latest value is older are left out."
        )
    rng = definition.reference_range
    if rng is not None and not any((g.units or "") == rng.units for g in groups):
        notes.append(
            f"The reference range is in {rng.units!r} and no value of {code} is: it is not "
            f"compared (units are never converted)."
        )
    without_age = _without_age(population, people)
    if without_age:
        notes.append(
            f"{without_age} patient(s) without a known age are left out of the age strata "
            f"(and counted in the totals)."
        )
    return ObservationResult(
        status=SectionStatus.COMPUTED,
        groups=groups,
        metrics=metrics,
        notes=tuple(notes),
        **common,
    )


def _group(
    rows: pd.DataFrame,
    units: str,
    rng: ReferenceRange | None,
    population: Cohort,
    people: pd.Index,
    reference: date,
    lookback: date | None,
) -> UnitGroup:
    days = rows["DATE"].dt.normalize()
    ref = pd.Timestamp(reference)
    outside = days < pd.Timestamp(lookback) if lookback else pd.Series(False, index=rows.index)
    kept = rows[~outside]
    kept_days = days[~outside]
    by_patient = kept.set_index("PATIENT")["VALUE"]
    age = (ref - kept_days).dt.days
    metrics: dict[str, Any] = {
        "patients_with_value": len(kept),
        "patients_outside_lookback": int(outside.sum()),
        "patients_tied_latest": int((kept["values"] > 1).sum()),
        "latest_value_age_days_median": float(age.median()) if len(age) else None,
    }
    for years in STALE_YEARS:
        metrics[f"patients_latest_older_than_{years}y"] = int(
            (kept_days < pd.Timestamp(window_start(reference, years))).sum()
        )
    return UnitGroup(
        units=units or None,
        summary=summarise(by_patient),
        strata=_strata(by_patient, population, people),
        reference_range=None if rng is None else _compare(rng, units, by_patient),
        metrics=metrics,
    )


def _strata(values: pd.Series, population: Cohort, people: pd.Index) -> tuple[ValueStratum, ...]:
    strata: list[ValueStratum] = []
    if population.age_band is not None:
        bands = population.age_band[population.age_band.index.isin(people)]
        for label in population.band_labels:
            members = bands[bands == label].index
            strata.append(_stratum("age_band", label, members, values))
    if population.gender is not None:
        gender = population.gender[population.gender.index.isin(people)]
        labels = sorted(gender.dropna().unique()) + ([None] if gender.isna().any() else [])
        for label in labels:
            members = (
                gender[gender.isna()].index if label is None else gender[gender == label].index
            )
            strata.append(_stratum("GENDER", label, members, values))
    return tuple(strata)


def _stratum(
    dimension: str, value: str | None, members: pd.Index, values: pd.Series
) -> ValueStratum:
    return ValueStratum(
        dimension=dimension,
        value=value,
        patients=len(members),
        summary=summarise(values[values.index.isin(members)]),
    )


def _compare(rng: ReferenceRange, units: str, values: pd.Series) -> RangeComparison:
    if rng.units != units:
        shown = repr(units) if units else "no units"
        return RangeComparison(
            rng,
            SectionStatus.SKIPPED,
            reason=f"the range is in {rng.units!r}; these values are in {shown} and are never "
            f"converted",
        )
    below = int((values < rng.low).sum()) if rng.low is not None else 0
    above = int((values > rng.high).sum()) if rng.high is not None else 0
    return RangeComparison(
        rng, SectionStatus.COMPUTED, below=below, within=len(values) - below - above, above=above
    )


def _without_age(population: Cohort, people: pd.Index) -> int:
    if population.age_band is None:
        return 0
    bands = population.age_band[population.age_band.index.isin(people)]
    return int(bands.isna().sum())


def _row_notes(fates: dict[str, int], types: dict[str, int]) -> tuple[str, ...]:
    notes = []
    if fates.get("not_numeric_type"):
        other = ", ".join(
            f"{t or '(empty)'} {n}" for t, n in sorted(types.items()) if t != NUMERIC_TYPE
        )
        notes.append(
            f"{fates['not_numeric_type']} row(s) of alive patients, dated on or before the "
            f"reference date, have a TYPE other than numeric and are not used (rows by TYPE, "
            f"all patients: {other})."
        )
    if fates.get("value_unparseable"):
        notes.append(
            f"{fates['value_unparseable']} row(s) with TYPE numeric have a VALUE that is not a "
            f"finite number and are not used."
        )
    if fates.get("date_unusable"):
        notes.append(f"{fates['date_unusable']} row(s) have an empty or unparseable DATE.")
    return tuple(notes)


def _fate_text(fates: dict[str, int]) -> str:
    parts = [f"{name.replace('_', ' ')} {count}" for name, count in fates.items() if count]
    return "rows: " + ", ".join(parts)


def _counts(keys: pd.Series, values: pd.Series, order: tuple[str, ...] = ()) -> dict:
    """``{key: {value: rows}}``, with every name of ``order`` present (0 when absent)."""
    if keys.empty:
        return {}
    counted = pd.DataFrame({"k": keys, "v": values}).groupby(["k", "v"]).size()
    result: dict[str, dict[str, int]] = {}
    for (key, value), count in counted.items():
        result.setdefault(str(key), {})[str(value)] = int(count)
    if order:
        result = {
            key: {name: inner.get(name, 0) for name in order} for key, inner in result.items()
        }
    return result


# --------------------------------------------------------------------------- #
# general table
# --------------------------------------------------------------------------- #


def general_table(
    values: Values,
    population: Cohort,
    reference: date,
    *,
    lookback_years: int | None = None,
    top: int = DEFAULT_TOP,
) -> GeneralTable:
    """Every numeric (code, units) with at least one alive patient, by patients."""
    if top < 1:
        raise ValueError("top must be at least 1")
    latest = values.latest
    outside = 0
    if lookback_years:
        start = pd.Timestamp(window_start(reference, lookback_years))
        old = latest["DATE"].dt.normalize() < start
        outside = int(old.sum())
        latest = latest[~old]
    mixed_units = {code for code, units in values.units.items() if len(units) > 1}
    mixed_types = {code for code, types in values.types.items() if len(types) > 1}
    rows = [
        GeneralRow(
            code=str(code),
            units=units or None,
            description=values.descriptions.get(code),
            summary=summarise(group["VALUE"]),
            mixed_units=code in mixed_units,
            other_types=code in mixed_types,
        )
        for (code, units), group in latest.groupby(["CODE", "UNITS"], sort=False)
    ]
    rows.sort(key=lambda r: (-r.summary.n, r.code, r.units or ""))
    notes = [
        "Sorted by patients with a value, then code and units; the JSON lists every row.",
        "Codes with more than one unit, or more than one TYPE, are listed below over the "
        "whole table (all patients, all dates).",
        *values.notes,
    ]
    if lookback_years:
        notes.append(
            f"Lookback of {lookback_years} year(s): {outside} latest value(s) older than that are "
            f"left out."
        )
    metrics = {
        "alive": population.size,
        "codes": len({r.code for r in rows}),
        "code_units": len(rows),
        "values_outside_lookback": outside,
        **values.metrics,
    }
    return GeneralTable(
        status=SectionStatus.COMPUTED,
        rows=tuple(rows),
        top=top,
        mixed_units=tuple(
            {
                "code": code,
                "description": values.descriptions.get(code),
                "units": values.units[code],
            }
            for code in sorted(mixed_units)
        ),
        mixed_types=tuple(
            {
                "code": code,
                "description": values.descriptions.get(code),
                "types": values.types[code],
            }
            for code in sorted(mixed_types)
        ),
        metrics=metrics,
        notes=tuple(notes),
    )
