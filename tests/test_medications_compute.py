"""Tests for the share of a cohort with each medication."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from synthea_quality.condition_cohort import CohortSpec
from synthea_quality.medications.compute import (
    MEDICATION_COLUMNS,
    medication_results,
    medication_use,
    prepare_medications,
)
from synthea_quality.medications.definitions import MedicationDefinition
from synthea_quality.prevalence.compute import (
    CONDITION_COLUMNS,
    PATIENT_COLUMNS,
    build_cohort,
    prepare_records,
)
from synthea_quality.prevalence.definitions import parse_condition_option
from synthea_quality.prevalence.models import Expected
from synthea_quality.profile.models import SectionStatus

REF = date(2026, 8, 17)
HTN = parse_condition_option("Hypertension=59621000")
LISINOPRIL = MedicationDefinition("Lisinopril", ("314076",), CohortSpec("Hypertension"))


def text_frame(rows, columns) -> pd.DataFrame:
    """Rows as the loader produces them: text, an empty field as the only null."""
    data = pd.DataFrame([[row.get(c, "") for c in columns] for row in rows], columns=list(columns))
    return data.replace("", None)


def population():
    patients = text_frame(
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1960-01-01", "GENDER": "F"},
            {"Id": "a3", "BIRTHDATE": "1970-01-01", "GENDER": "F"},
            {"Id": "a4", "BIRTHDATE": "1990-01-01", "GENDER": "M"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2020-01-01", "GENDER": "M"},
        ],
        PATIENT_COLUMNS,
    )
    return build_cohort(patients, REF, (0, 18, 65))


def rx(patient, start, stop="", code="314076", reason="59621000"):
    return {"PATIENT": patient, "START": start, "STOP": stop, "CODE": code,
            "REASONCODE": reason, "DESCRIPTION": "lisinopril 10 MG Oral Tablet"}


def records(*rows, columns=MEDICATION_COLUMNS):
    return prepare_medications(text_frame(list(rows), columns), population(), REF)


def hypertensive():
    frame = text_frame(
        [{"PATIENT": p, "CODE": "59621000", "SYSTEM": "http://snomed.info/sct",
          "START": "2015-01-01"} for p in ("a1", "a2", "a3")],
        CONDITION_COLUMNS,
    )
    return prepare_records(frame, population(), REF)


def use(
    *rows,
    definition=LISINOPRIL,
    members=pd.Index(["a1", "a2", "a3"]),  # noqa: B008 - shared read-only default for a test helper
    condition=HTN,
):
    return medication_use(
        definition, records(*rows), population(), members=members, condition=condition
    )


def test_active_and_ever_at_the_reference_date():
    result = use(
        rx("a1", "2020-01-01T00:00:00Z"),  # no STOP: active
        rx("a2", "2020-01-01T00:00:00Z", "2026-08-17T10:00:00Z"),  # stops on ref day: ever only
        rx("a3", "2026-08-17T23:00:00Z", "2026-08-18T00:00:00Z"),  # starts on ref day: active
        rx("a4", "2020-01-01T00:00:00Z"),  # outside the cohort
        rx("d1", "2010-01-01T00:00:00Z"),  # deceased
    )
    assert (result.active.numerator, result.active.denominator) == (2, 3)
    assert (result.ever.numerator, result.ever.denominator) == (3, 3)
    assert result.active.interval is not None
    assert result.metrics["records"] == 3 and result.metrics["records_without_stop"] == 1
    assert any("1 of 3 records have no STOP" in note for note in result.notes)
    assert result.descriptions == {"314076": "lisinopril 10 MG Oral Tablet"}


def test_a_patient_counts_once_whichever_code():
    definition = MedicationDefinition("Any", ("314076", "310798"), CohortSpec("Hypertension"))
    result = use(
        rx("a1", "2020-01-01T00:00:00Z"), rx("a1", "2021-01-01T00:00:00Z", code="310798"),
        definition=definition,
    )
    assert result.ever.numerator == 1
    assert result.metrics["records_by_code"] == {"314076": 1, "310798": 1}


def test_reasoncode_of_the_cohort_condition():
    result = use(
        rx("a1", "2020-01-01T00:00:00Z"),
        rx("a2", "2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
        rx("a3", "2020-01-01T00:00:00Z", reason="38341003"),
    )
    assert (result.active_with_reason, result.ever_with_reason) == (1, 2)


def test_no_reason_without_a_cohort_or_a_column():
    plain = MedicationDefinition("Lisinopril", ("314076",))
    result = use(rx("a1", "2020-01-01T00:00:00Z"), definition=plain, members=None, condition=None)
    assert result.ever.denominator == 4  # every alive patient
    assert result.active_with_reason is result.ever_with_reason is None
    columns = tuple(c for c in MEDICATION_COLUMNS if c != "REASONCODE")
    no_column = medication_use(
        LISINOPRIL, records(rx("a1", "2020-01-01T00:00:00Z"), columns=columns), population(),
        members=pd.Index(["a1"]), condition=HTN,
    )
    assert no_column.ever_with_reason is None
    assert any("no REASONCODE column" in note for note in no_column.notes)


def test_a_code_absent_from_the_table_gets_a_note():
    losartan = MedicationDefinition("Losartan", ("979485",), CohortSpec("Hypertension"))
    result = use(rx("a4", "2020-01-01T00:00:00Z"), definition=losartan)
    assert (result.ever.numerator, result.ever.denominator) == (0, 3)
    assert any("No record of 979485 in medications.csv" in note for note in result.notes)
    assert result.descriptions == {}
    present = use(rx("a4", "2020-01-01T00:00:00Z"))  # in the table, outside the cohort
    assert not any("No record" in note for note in present.notes)


def test_rows_left_out_are_counted():
    prepared = records(
        rx("a1", "2020-01-01T00:00:00Z"),
        rx("a1", "2027-01-01T00:00:00Z"),  # after the reference date
        rx("a2", ""),  # no START
        rx("a2", "2020-01-01T00:00:00Z", "soon"),  # unparseable STOP: ever, not active
        rx("d1", "2020-01-01T00:00:00Z"),
    )
    m = prepared.metrics
    assert (m["rows"], m["rows_not_alive"], m["rows_after_reference"]) == (5, 1, 1)
    assert (m["rows_start_unusable"], m["rows_stop_unparseable"], m["rows_used"]) == (1, 1, 2)
    result = medication_use(LISINOPRIL, prepared, population(), members=pd.Index(["a2"]),
                            condition=HTN)
    assert (result.active.numerator, result.ever.numerator) == (0, 1)


def test_expected_shares_are_placed_against_the_ci():
    definition = MedicationDefinition(
        "Lisinopril", ("314076",), CohortSpec("Hypertension"),
        (Expected("active", 0.5), Expected("ever", 0.01)),
    )
    result = use(rx("a1", "2020-01-01T00:00:00Z"), definition=definition)
    assert [(e.expected.measure, e.position) for e in result.expected] == [
        ("active", "inside"), ("ever", "outside"),
    ]


def test_results_select_each_cohort():
    prepared = records(rx("a1", "2020-01-01T00:00:00Z"), rx("a4", "2020-01-01T00:00:00Z"))
    plain = MedicationDefinition("All alive", ("314076",))
    cohort, everyone = medication_results(
        [LISINOPRIL, plain], [HTN], prepared, population(), condition_records=hypertensive()
    )
    assert (cohort.ever.numerator, cohort.ever.denominator) == (1, 3)
    assert cohort.ever_with_reason == 1
    assert (everyone.ever.numerator, everyone.ever.denominator) == (2, 4)
    (skipped,) = medication_results(
        [LISINOPRIL], [HTN], prepared, population(), records_reason="conditions.csv is absent"
    )
    assert skipped.status is SectionStatus.SKIPPED
    assert "conditions.csv is absent" in skipped.reason


def test_an_empty_cohort_has_no_share():
    result = use(rx("a1", "2020-01-01T00:00:00Z"), members=pd.Index([]))
    assert result.active.denominator == 0 and result.active.value is None
    with pytest.raises(ValueError):
        from synthea_quality.medications.models import MedicationResult

        MedicationResult("x", ("1",), SectionStatus.COMPUTED)
