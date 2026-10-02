"""Tests for the profile's structured result models."""

from __future__ import annotations

import json

import pytest

from synthea_quality import __version__
from synthea_quality.profile.models import (
    PROFILE_SCHEMA_VERSION,
    CategoryCount,
    CodeCount,
    CodeDescriptions,
    DescriptionCount,
    DatasetProfile,
    InputState,
    ProfileSection,
    ReferenceDate,
    ReferenceSource,
    SectionStatus,
    TableInput,
    percent,
)


def reference() -> ReferenceDate:
    return ReferenceDate(
        value="2026-08-17",
        source=ReferenceSource.MAX_ENCOUNTER_DATE,
        detail="latest encounters.START/STOP",
        approximate=True,
    )


def sample_profile(**overrides) -> DatasetProfile:
    values = dict(
        data_dir="/data/csv",
        reference_date=reference(),
        age_bands=(0, 5, 18, 45, 65),
        sections=(
            ProfileSection(
                section_id="population",
                title="Population",
                status=SectionStatus.COMPUTED,
                metrics={"total": 3, "alive": 2, "deceased": 1},
            ),
            ProfileSection(
                section_id="distribution.GENDER",
                title="Gender",
                status=SectionStatus.COMPUTED,
                metrics={"denominator": 2},
                distributions={
                    "GENDER": (CategoryCount("F", 1, 50.0), CategoryCount(None, 1, 50.0))
                },
            ),
            ProfileSection.skipped("distribution.STATE", "State", "no STATE column"),
        ),
        inputs=(TableInput("patients", "every section", InputState.READ, rows=3),),
        generated_at="2026-09-28T00:00:00+00:00",
    )
    values.update(overrides)
    return DatasetProfile(**values)


def test_percent_rounds_to_two_decimals_and_handles_an_empty_denominator():
    assert percent(1, 3) == 33.33
    assert percent(2, 3) == 66.67
    assert percent(0, 0) == 0.0


def test_skipped_section_requires_a_reason_and_no_numbers():
    with pytest.raises(ValueError, match="reason"):
        ProfileSection("x", "X", SectionStatus.SKIPPED)
    with pytest.raises(ValueError, match="no numbers"):
        ProfileSection("x", "X", SectionStatus.SKIPPED, reason="r", metrics={"a": 1})


def test_computed_section_must_not_carry_a_reason():
    with pytest.raises(ValueError, match="reason"):
        ProfileSection("x", "X", SectionStatus.COMPUTED, reason="why")


def test_metrics_reject_values_json_cannot_represent():
    with pytest.raises(ValueError, match="NaN"):
        ProfileSection("x", "X", SectionStatus.COMPUTED, metrics={"a": float("nan")})
    with pytest.raises(TypeError):
        ProfileSection("x", "X", SectionStatus.COMPUTED, metrics={"a": object()})


def test_nested_metric_keys_are_sorted_for_determinism():
    section = ProfileSection(
        "x", "X", SectionStatus.COMPUTED, metrics={"b": {"z": 1, "a": 2}, "a": 0}
    )
    assert list(section.metrics) == ["a", "b"]
    assert list(section.metrics["b"]) == ["a", "z"]


def test_profile_needs_exactly_one_of_reference_date_and_reason():
    with pytest.raises(ValueError, match="exactly one"):
        sample_profile(reference_date=None)
    with pytest.raises(ValueError, match="exactly one"):
        sample_profile(reference_reason="also a reason")
    assert sample_profile(reference_date=None, reference_reason="none").reference_date is None


def test_profile_rejects_duplicate_section_ids():
    section = ProfileSection.skipped("a", "A", "r")
    with pytest.raises(ValueError, match="duplicate"):
        sample_profile(sections=(section, section))


def test_to_dict_carries_version_provenance_and_fixed_key_order():
    data = sample_profile().to_dict()
    assert list(data) == [
        "schema_version",
        "tool_version",
        "generated_at",
        "data_dir",
        "reference_date",
        "reference_reason",
        "age_bands",
        "inputs",
        "notes",
        "export_history",
        "sections",
    ]
    assert data["schema_version"] == PROFILE_SCHEMA_VERSION
    assert data["tool_version"] == __version__
    assert data["reference_date"] == {
        "value": "2026-08-17",
        "source": "max_encounter_date",
        "approximate": True,
        "detail": "latest encounters.START/STOP",
    }
    gender = data["sections"][1]["distributions"]["GENDER"]
    assert gender[1] == {"value": None, "count": 1, "percent": 50.0}


def test_json_round_trip_reproduces_the_same_text():
    profile = sample_profile()
    text = profile.to_json()
    again = DatasetProfile.from_json(text)
    assert again == profile
    assert again.to_json() == text


def test_from_dict_rejects_an_unknown_schema_version():
    data = json.loads(sample_profile().to_json())
    data["schema_version"] = PROFILE_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema_version"):
        DatasetProfile.from_dict(data)


def test_section_lookup_by_identifier():
    profile = sample_profile()
    assert profile.section("population").metrics["alive"] == 2
    with pytest.raises(KeyError):
        profile.section("nope")


def test_table_input_rows_and_reason_follow_the_state():
    absent = TableInput("encounters", "x", InputState.ABSENT, reason="not in the dataset")
    assert absent.rows is None
    with pytest.raises(ValueError):
        TableInput("patients", "x", InputState.READ)
    with pytest.raises(ValueError):
        TableInput("patients", "x", InputState.UNREADABLE, rows=1, reason="r")
    with pytest.raises(ValueError):
        TableInput("patients", "x", InputState.ABSENT)


def test_profile_is_incomplete_only_when_a_needed_table_is_unreadable():
    absent = TableInput("encounters", "x", InputState.ABSENT, reason="not in the dataset")
    broken = TableInput("patients", "x", InputState.UNREADABLE, reason="not UTF-8")
    assert not sample_profile(inputs=(absent,)).incomplete
    assert sample_profile(inputs=(absent, broken)).incomplete


def code_section() -> ProfileSection:
    return ProfileSection(
        section_id="codes.medications",
        title="Medications",
        status=SectionStatus.COMPUTED,
        metrics={"denominator": 2},
        codes=(CodeCount(None, "123", "Aspirin", 2, 100.0, 5, description_variants=2),),
        multi_description_codes=(
            CodeDescriptions(
                None,
                "123",
                (DescriptionCount("Aspirin", 4), DescriptionCount("aspirin 81 MG", 1)),
            ),
        ),
    )


def test_code_rows_round_trip_through_json():
    profile = sample_profile(sections=(code_section(),))
    data = json.loads(profile.to_json())
    section = data["sections"][0]
    assert section["codes"] == [
        {
            "system": None,
            "code": "123",
            "description": "Aspirin",
            "patients": 2,
            "percent": 100.0,
            "records": 5,
            "description_variants": 2,
        }
    ]
    assert section["multi_description_codes"][0]["descriptions"][1] == {
        "description": "aspirin 81 MG",
        "records": 1,
    }
    assert DatasetProfile.from_json(profile.to_json()) == profile


def test_a_profile_written_before_codes_existed_still_loads():
    data = json.loads(sample_profile().to_json())
    for section in data["sections"]:
        del section["codes"], section["multi_description_codes"]
    assert DatasetProfile.from_dict(data) == sample_profile()


def test_a_skipped_section_carries_no_codes():
    with pytest.raises(ValueError, match="no numbers"):
        ProfileSection(
            "codes.x", "X", SectionStatus.SKIPPED, reason="r",
            codes=(CodeCount(None, "1", "d", 1, 1.0, 1),),
        )


def test_a_code_is_listed_as_ambiguous_only_with_two_descriptions():
    with pytest.raises(ValueError, match="at least two"):
        CodeDescriptions(None, "1", (DescriptionCount("only", 3),))
