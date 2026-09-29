"""Tests for the distribution of observation values."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.observations.compute import (
    OBSERVATION_COLUMNS,
    general_table,
    observation_values,
    prepare_values,
    summarise,
)
from synthea_quality.observations.definitions import ObservationDefinition, ReferenceRange
from synthea_quality.prevalence.compute import PATIENT_COLUMNS, build_cohort
from synthea_quality.profile.models import SectionStatus

REF = date(2026, 8, 17)
BANDS = (0, 18, 65)
SBP = "8480-6"


def text_frame(rows, columns) -> pd.DataFrame:
    """Rows as the loader produces them: text, an empty field as the only null."""
    data = pd.DataFrame([[row.get(c, "") for c in columns] for row in rows], columns=list(columns))
    return data.replace("", None).astype("str") if len(data) else data.astype("str")


def people():
    return text_frame(
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},  # 76
            {"Id": "a2", "BIRTHDATE": "1990-01-01", "GENDER": "F"},  # 36
            {"Id": "a3", "BIRTHDATE": "2010-01-01", "GENDER": "F"},  # 16
            {"Id": "a4", "BIRTHDATE": "1980-01-01"},  # 46, no GENDER
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2020-01-01", "GENDER": "M"},
        ],
        PATIENT_COLUMNS,
    ).replace("nan", None)


def obs(patient, value, day="2026-01-01T10:00:00Z", code=SBP, units="mm[Hg]", kind="numeric",
        description="Systolic Blood Pressure"):
    return {"PATIENT": patient, "CODE": code, "VALUE": value, "DATE": day, "UNITS": units,
            "TYPE": kind, "DESCRIPTION": description}


def frame(*rows, columns=OBSERVATION_COLUMNS):
    return text_frame(list(rows), columns).replace("nan", None)


def run(*rows, definition=None, members=None, columns=OBSERVATION_COLUMNS):
    population = build_cohort(people(), REF, BANDS)
    values = prepare_values(frame(*rows, columns=columns), population, REF)
    result = observation_values(
        definition or ObservationDefinition(SBP), values, population, REF, members=members
    )
    return result, values, population


# --------------------------------------------------------------------------- #
# one value per patient
# --------------------------------------------------------------------------- #


def test_the_latest_value_on_or_before_the_reference_date():
    result, values, _ = run(
        obs("a1", "150", "2020-01-01T00:00:00Z"),
        obs("a1", "130", "2026-08-17T23:59:59Z"),  # on the reference day: counts
        obs("a1", "999", "2026-08-18T00:00:00Z"),  # after: left out, counted
    )
    (group,) = result.groups
    assert group.summary.n == 1 and group.summary.median == 130
    assert result.metrics["rows_used"] == 2
    assert result.metrics["rows_after_reference"] == 1


def test_values_sharing_the_latest_date_use_their_median():
    result, _, _ = run(
        obs("a1", "120", "2026-01-01T10:00:00Z"),
        obs("a1", "130", "2026-01-01T10:00:00Z"),
        obs("a1", "200", "2025-01-01T10:00:00Z"),
    )
    (group,) = result.groups
    assert group.summary.median == 125
    assert group.metrics["patients_tied_latest"] == 1
    assert any("median of those values" in note for note in result.notes)


def test_every_left_out_row_is_counted():
    result, values, _ = run(
        obs("a1", "120"),
        obs("d1", "180"),  # deceased
        obs("zz", "180"),  # not in patients.csv
        obs("a2", "Positive", kind="text"),
        obs("a2", "abc"),  # numeric TYPE, not a number
        obs("a2", "inf"),
        obs("a3", "110", day="yesterday"),
        obs("a3", "110", day=""),
    )
    m = result.metrics
    assert (m["rows_used"], m["rows_deceased"], m["rows_unknown_patient"]) == (1, 1, 1)
    assert (m["rows_not_numeric_type"], m["rows_value_unparseable"], m["rows_date_unusable"]) == (
        1, 2, 2,
    )
    assert m["rows_by_type"] == {"numeric": 7, "text": 1}
    assert any("TYPE other than numeric" in note for note in result.notes)
    assert any("not a finite number" in note for note in result.notes)
    assert values.metrics["rows"] == 8


def test_units_are_separate_groups_and_never_converted():
    result, values, _ = run(
        obs("a1", "10", units="mL/min"),
        obs("a2", "20", units="mL/min"),
        obs("a2", "30", units="mL/min/{1.73_m2}"),
        obs("a3", "40", units=""),
    )
    assert [(g.units, g.summary.n) for g in result.groups] == [
        ("mL/min", 2), (None, 1), ("mL/min/{1.73_m2}", 1),
    ]
    assert result.metrics["patients_in_several_units"] == 1
    assert any("never converted" in note for note in result.notes)


def test_a_lookback_leaves_old_values_out_and_counts_them():
    definition = ObservationDefinition(SBP, lookback_years=2)
    result, _, _ = run(
        obs("a1", "120", "2025-01-01T00:00:00Z"),
        obs("a2", "140", "2024-08-17T00:00:00Z"),  # exactly 2 years: kept
        obs("a3", "160", "2024-08-16T00:00:00Z"),  # older: left out
        definition=definition,
    )
    (group,) = result.groups
    assert group.summary.n == 2
    assert group.metrics["patients_outside_lookback"] == 1
    assert any("Lookback of 2 year(s) (from 2024-08-17): 1 patient" in n for n in result.notes)


def test_how_old_the_latest_values_are():
    result, _, _ = run(
        obs("a1", "1", "2026-08-07T00:00:00Z"),  # 10 days
        obs("a2", "1", "2025-08-01T00:00:00Z"),  # > 1 year
        obs("a3", "1", "2022-01-01T00:00:00Z"),  # > 3 years
    )
    m = result.groups[0].metrics
    assert m["patients_latest_older_than_1y"] == 2
    assert m["patients_latest_older_than_3y"] == 1
    assert m["latest_value_age_days_median"] == (date(2026, 8, 17) - date(2025, 8, 1)).days


# --------------------------------------------------------------------------- #
# statistics, strata and ranges
# --------------------------------------------------------------------------- #


def test_percentiles_are_type_7():
    s = summarise(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]))
    assert (s.n, s.minimum, s.maximum, s.median) == (10, 1.0, 100.0, 5.5)
    assert s.p25 == pytest.approx(3.25) and s.p75 == pytest.approx(7.75)
    assert s.p5 == pytest.approx(1.45) and s.p95 == pytest.approx(59.05)
    assert summarise(pd.Series([], dtype=float)).n == 0


def test_below_ten_values_only_n_minimum_median_and_maximum():
    s = summarise(pd.Series([float(v) for v in range(1, 10)]))
    assert (s.n, s.minimum, s.median, s.maximum) == (9, 1.0, 5.0, 9.0)
    assert s.p5 is s.p25 is s.p75 is s.p95 is None
    one = summarise(pd.Series([7.0]))
    assert one.minimum == one.median == one.maximum == 7.0 and one.p25 is None


def test_strata_by_age_band_and_sex():
    result, _, _ = run(obs("a1", "150"), obs("a2", "120"), obs("a3", "100"), obs("a4", "130"))
    strata = {(s.dimension, s.value): s for s in result.groups[0].strata}
    assert [k for k in strata if k[0] == "age_band"] == [
        ("age_band", "0-17"), ("age_band", "18-64"), ("age_band", "65+"),
    ]
    assert strata[("age_band", "18-64")].patients == 2
    assert strata[("age_band", "18-64")].summary.median == 125
    assert strata[("GENDER", "F")].summary.n == 2
    assert strata[("GENDER", None)].summary.median == 130  # empty GENDER, its own stratum


def test_a_reference_range_counts_below_within_and_above():
    rng = ReferenceRange("mm[Hg]", 100, 139, basis="synthea-configuration")
    result, _, _ = run(
        obs("a1", "99"), obs("a2", "100"), obs("a3", "139"), obs("a4", "140"),
        definition=ObservationDefinition(SBP, reference_range=rng),
    )
    comparison = result.groups[0].reference_range
    assert comparison.status is SectionStatus.COMPUTED
    assert (comparison.below, comparison.within, comparison.above) == (1, 2, 1)
    assert comparison.to_dict()["note"] == rng.note


def test_a_range_in_other_units_is_not_compared():
    rng = ReferenceRange("kPa", 13, 18)
    result, _, _ = run(obs("a1", "120"), definition=ObservationDefinition(SBP, reference_range=rng))
    comparison = result.groups[0].reference_range
    assert comparison.status is SectionStatus.SKIPPED
    assert "never converted" in comparison.reason
    assert any("no value of 8480-6 is" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# populations and skipped results
# --------------------------------------------------------------------------- #


def test_a_cohort_limits_the_patients():
    definition = ObservationDefinition(SBP, cohort=CohortSpec("Hypertension"))
    result, _, _ = run(
        obs("a1", "150"), obs("a2", "120"), definition=definition, members=pd.Index(["a1", "a3"]),
    )
    assert result.population == 2
    assert result.groups[0].summary.n == 1 and result.groups[0].summary.median == 150
    ages = [s for s in result.groups[0].strata if s.dimension == "age_band"]
    assert sum(s.patients for s in ages) == 2


def test_a_code_absent_from_the_table_is_skipped():
    result, _, _ = run(obs("a1", "120", code="other"))
    assert result.status is SectionStatus.SKIPPED
    assert "no row of code 8480-6" in result.reason
    assert result.name == SBP


def test_a_code_with_only_text_values_is_skipped_with_the_counts():
    result, _, _ = run(obs("a1", "Positive", kind="text"), obs("d1", "12"))
    assert result.status is SectionStatus.SKIPPED
    assert "not numeric type 1" in result.reason and "deceased 1" in result.reason
    assert result.metrics["rows_not_numeric_type"] == 1


def test_the_name_defaults_to_the_most_frequent_description():
    result, _, _ = run(obs("a1", "120"), obs("a2", "120", description="Other"),
                       obs("a3", "120"))
    assert result.name == result.description == "Systolic Blood Pressure"
    named, _, _ = run(obs("a1", "120"), definition=ObservationDefinition(SBP, name="SBP"))
    assert named.name == "SBP"


def test_without_type_and_units_columns():
    columns = ("DATE", "PATIENT", "CODE", "VALUE")
    population = build_cohort(people(), REF, BANDS)
    values = prepare_values(frame(obs("a1", "120"), obs("a2", "x"), columns=columns),
                            population, REF)
    assert not values.has_type and not values.has_units
    assert len(values.notes) == 2
    result = observation_values(ObservationDefinition(SBP), values, population, REF)
    assert result.groups[0].units is None and result.groups[0].summary.n == 1
    assert result.metrics["rows_value_unparseable"] == 1


def test_a_patient_without_age_is_counted():
    patients = text_frame([{"Id": "a1", "BIRTHDATE": "2030-01-01", "GENDER": "M"}],
                          PATIENT_COLUMNS).replace("nan", None)
    population = build_cohort(patients, REF, BANDS)
    values = prepare_values(frame(obs("a1", "120")), population, REF)
    result = observation_values(ObservationDefinition(SBP), values, population, REF)
    assert any("without a known age" in note for note in result.notes)


# --------------------------------------------------------------------------- #
# general table
# --------------------------------------------------------------------------- #


def test_the_general_table():
    population = build_cohort(people(), REF, BANDS)
    values = prepare_values(
        frame(
            obs("a1", "120"), obs("a2", "130"),
            obs("a1", "1", code="33914-3", units="mL/min", description="GFR"),
            obs("d1", "2", code="33914-3", units="mL/min/{1.73_m2}", description="GFR"),
            obs("a1", "2", code="2514-8", units="{presence}", description="Ketones"),
            obs("a2", "Negative", code="2514-8", units="", kind="text", description="Ketones"),
            obs("a3", "1", "2020-01-01T00:00:00Z", code="old", units="x", description="Old"),
        ),
        population, REF,
    )
    table = general_table(values, population, REF, top=2)
    assert [(r.code, r.units, r.summary.n) for r in table.rows] == [
        (SBP, "mm[Hg]", 2), ("2514-8", "{presence}", 1), ("33914-3", "mL/min", 1),
        ("old", "x", 1),
    ]
    assert table.rows[2].mixed_units and table.rows[1].other_types
    # over the whole table: the deceased patient's unit counts
    assert table.mixed_units == (
        {"code": "33914-3", "description": "GFR",
         "units": {"mL/min": 1, "mL/min/{1.73_m2}": 1}},
    )
    assert table.mixed_types[0]["types"] == {"numeric": 1, "text": 1}
    assert table.metrics["codes"] == 4

    recent = general_table(values, population, REF, lookback_years=3)
    assert "old" not in {r.code for r in recent.rows}
    assert recent.metrics["values_outside_lookback"] == 1
    with pytest.raises(ValueError):
        general_table(values, population, REF, top=0)
