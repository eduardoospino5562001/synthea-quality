"""Tests for the most common codes of a clinical table among the alive patients."""

from __future__ import annotations

import pandas as pd
import pytest

from synthea_quality.profile.codes import (
    CLINICAL_TABLES,
    HISTORICAL_NOTE,
    MULTI_DESCRIPTION_LIMIT,
    columns_to_load,
    profile_codes,
    skipped_code_sections,
)
from synthea_quality.profile.models import SectionStatus
from synthea_quality.profile.ranking import tie_at_cut

ALIVE = pd.Index(["a1", "a2", "a3", "a4"])
EVERYONE = pd.Index(["a1", "a2", "a3", "a4", "d1"])


def frame(rows: list[tuple], columns=("PATIENT", "CODE", "DESCRIPTION")) -> pd.DataFrame:
    """A table as the loader produces it: text, an empty field as the only null."""
    data = pd.DataFrame([list(r) for r in rows], columns=list(columns), dtype="str")
    return data.replace("", None).astype("str") if len(data) else data


def compute(rows, table="conditions", columns=("PATIENT", "CODE", "DESCRIPTION"), top=20):
    return profile_codes(
        table, frame(rows, columns), alive_ids=ALIVE, patient_ids=EVERYONE, top=top
    )


def summary(section):
    return [(r.system, r.code, r.description, r.patients, r.records) for r in section.codes]


def test_columns_to_load_requires_patient_and_code_and_adds_what_exists():
    assert columns_to_load(("PATIENT", "CODE", "X")) == ["PATIENT", "CODE"]
    assert columns_to_load(("SYSTEM", "CODE", "DESCRIPTION", "PATIENT")) == [
        "PATIENT", "CODE", "SYSTEM", "DESCRIPTION",
    ]
    assert columns_to_load(("PATIENT", "DESCRIPTION")) is None


def test_counts_distinct_alive_patients_not_rows():
    section = compute(
        [
            ("a1", "10", "Flu"), ("a1", "10", "Flu"), ("a1", "10", "Flu"),
            ("a2", "20", "Cold"), ("a3", "20", "Cold"),
        ]
    )
    assert summary(section) == [(None, "20", "Cold", 2, 2), (None, "10", "Flu", 1, 3)]
    assert section.codes[0].percent == 50.0  # 2 of the 4 alive patients


def test_deceased_unknown_and_codeless_rows_are_counted_then_left_out():
    section = compute(
        [
            ("a1", "10", "Flu"),
            ("d1", "10", "Flu"),     # deceased
            ("zz", "10", "Flu"),     # not in patients.csv
            ("a2", "", "Nothing"),   # alive, no code
        ]
    )
    assert summary(section) == [(None, "10", "Flu", 1, 1)]
    m = section.metrics
    assert (m["rows"], m["rows_alive"], m["rows_deceased"]) == (4, 2, 1)
    assert m["rows_unknown_patient"] == 1
    assert m["rows_without_code"] == 1
    assert m["patients_with_records"] == 1
    assert m["denominator"] == 4
    assert any(note.startswith("Left out: 1 row(s) of deceased") for note in section.notes)


def test_every_section_says_the_count_is_historical():
    section = compute([("a1", "10", "Flu")])
    assert section.notes[0] == HISTORICAL_NOTE
    assert "not patients in whom it is active at the reference date" in HISTORICAL_NOTE


def test_order_is_patients_then_records_then_code():
    section = compute(
        [
            ("a1", "30", "C"), ("a2", "30", "C"),
            ("a1", "20", "B"), ("a2", "20", "B"), ("a2", "20", "B"),
            ("a3", "10", "A"), ("a4", "10", "A"),
        ]
    )
    assert [r.code for r in section.codes] == ["20", "10", "30"]


def test_a_cut_inside_a_patient_tie_is_reported():
    section = compute(
        [("a1", "10", "A"), ("a2", "10", "A"), ("a1", "20", "B"), ("a1", "30", "C")],
        top=2,
    )
    assert [r.code for r in section.codes] == ["10", "20"]
    assert section.metrics["tie_at_cut"] == {"count": 1, "values": 2, "listed": 1}
    assert (
        "2 codes tie at 1 patients; ties are broken by records, then by code "
        "(1 listed, 1 not listed)."
    ) in section.notes
    assert "tie_at_cut" not in compute([("a1", "10", "A"), ("a1", "20", "B")], top=5).metrics


def test_distinct_codes_are_counted_over_the_table_and_over_the_alive():
    section = compute([("a1", "10", "A"), ("d1", "20", "B"), ("a2", "30", "C")], top=1)
    assert section.metrics["distinct_codes"] == 3
    assert section.metrics["distinct_codes_alive"] == 2
    assert section.metrics["shown"] == 1


def test_the_most_frequent_description_over_the_whole_table_is_shown():
    section = compute(
        [
            ("a1", "10", "flu"),       # the alive patients only ever write "flu"
            ("d1", "10", "Influenza"),
            ("d1", "10", "Influenza"),
            ("a2", "20", "b"), ("a3", "20", "a"),   # tie: alphabetical
            ("a4", "30", ""), ("a4", "30", ""), ("a1", "30", "Thing"),  # empty loses ties
        ]
    )
    by_code = {r.code: r for r in section.codes}
    assert by_code["10"].description == "Influenza"
    assert by_code["10"].description_variants == 2
    assert by_code["20"].description == "a"
    assert by_code["30"].description is None  # two empty records beat one described


def test_codes_with_several_descriptions_are_listed_with_every_variant():
    section = compute(
        [
            ("a1", "20", "Tdap"), ("a2", "20", "Tdap"), ("a3", "20", "tetanus toxoid"),
            ("a1", "10", "Urea nitrogen"), ("a2", "10", "Urea Nitrogen"),
            ("a1", "10", "Urea nitrogen"),
            ("a4", "30", "single"),
        ]
    )
    assert section.metrics["codes_with_multiple_descriptions"] == 2
    listed = [
        (item.code, [(d.description, d.records) for d in item.descriptions])
        for item in section.multi_description_codes
    ]
    assert listed == [
        ("10", [("Urea nitrogen", 2), ("Urea Nitrogen", 1)]),
        ("20", [("Tdap", 2), ("tetanus toxoid", 1)]),
    ]


def test_the_list_of_ambiguous_codes_is_bounded_but_the_count_is_complete():
    rows = []
    for number in range(MULTI_DESCRIPTION_LIMIT + 3):
        rows += [("a1", f"{number:03d}", "x"), ("a1", f"{number:03d}", "y")]
    section = compute(rows)
    assert section.metrics["codes_with_multiple_descriptions"] == MULTI_DESCRIPTION_LIMIT + 3
    assert len(section.multi_description_codes) == MULTI_DESCRIPTION_LIMIT
    assert section.multi_description_codes[0].code == "000"
    assert any("the first 10" in note for note in section.notes)


def test_a_code_is_identified_by_system_and_code_when_the_table_has_a_system():
    columns = ("PATIENT", "CODE", "SYSTEM", "DESCRIPTION")
    section = compute(
        [
            ("a1", "123", "SNOMED-CT", "Peanut"),
            ("a2", "123", "SNOMED-CT", "Peanut"),
            ("a3", "123", "RxNorm", "Penicillin"),
            ("a4", "999", "", "No system"),
        ],
        table="allergies",
        columns=columns,
    )
    assert summary(section) == [
        ("SNOMED-CT", "123", "Peanut", 2, 2),
        (None, "999", "No system", 1, 1),
        ("RxNorm", "123", "Penicillin", 1, 1),
    ]
    assert section.metrics["code_identity"] == "SYSTEM+CODE"
    assert section.metrics["codes_with_multiple_descriptions"] == 0


def test_a_table_without_description_still_ranks_its_codes():
    section = compute([("a1", "10"), ("a2", "10")], columns=("PATIENT", "CODE"))
    assert summary(section) == [(None, "10", None, 2, 2)]
    assert any("no DESCRIPTION column" in note for note in section.notes)


def test_an_empty_table_computes_zeros():
    section = compute([])
    assert section.status is SectionStatus.COMPUTED
    assert section.codes == ()
    assert section.metrics["rows"] == 0


def test_top_must_be_positive():
    with pytest.raises(ValueError):
        compute([("a1", "10", "A")], top=0)


def test_skipped_sections_cover_every_clinical_table():
    sections = skipped_code_sections("no DEATHDATE")
    assert [s.section_id for s in sections] == [f"codes.{t}" for t, _ in CLINICAL_TABLES]
    assert all(s.status is SectionStatus.SKIPPED for s in sections)


def test_tie_at_cut():
    assert tie_at_cut([5, 3], [3, 3, 1]) == {"count": 3, "values": 3, "listed": 1}
    assert tie_at_cut([5, 3], [2]) is None
    assert tie_at_cut([], [1]) is None
    assert tie_at_cut([1], []) is None
