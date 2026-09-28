"""Tests for the JSON and Markdown renderings of a profile."""

from __future__ import annotations

import json

import pytest

from synthea_quality.profile.models import (
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
)
from synthea_quality.profile.render import (
    dumps,
    loads,
    render_markdown,
    write_json,
    write_markdown,
)


def computed(section_id, title, metrics, distributions=None, notes=()):
    return ProfileSection(
        section_id=section_id,
        title=title,
        status=SectionStatus.COMPUTED,
        metrics=metrics,
        distributions=distributions or {},
        notes=notes,
    )


def profile(**overrides) -> DatasetProfile:
    values = dict(
        data_dir="/data/csv",
        reference_date=ReferenceDate(
            value="2026-08-17",
            source=ReferenceSource.MAX_ENCOUNTER_DATE,
            detail="APPROXIMATION: the latest encounters.START/STOP value",
            approximate=True,
        ),
        age_bands=(0, 18),
        inputs=(
            TableInput("patients", "population and demographic sections", InputState.READ, 3),
            TableInput("encounters", "reference date approximation", InputState.READ, 2),
        ),
        sections=(
            computed(
                "population",
                "Population",
                {
                    "total": 3,
                    "alive": 2,
                    "alive_percent": 66.67,
                    "deceased": 1,
                    "deceased_percent": 33.33,
                    "deceased_with_unparseable_deathdate": 0,
                    "deaths_after_reference_date": 0,
                },
                notes=("Alive: DEATHDATE is empty.",),
            ),
            computed(
                "age",
                "Age at the reference date",
                {
                    "alive": 2,
                    "denominator": 2,
                    "reference_date": "2026-08-17",
                    "median": 40.5,
                    "min": 5,
                    "max": 76,
                    "excluded": {
                        "birthdate_empty": 0,
                        "birthdate_unparseable": 0,
                        "born_after_reference_date": 0,
                    },
                },
                {"age_band": (CategoryCount("0-17", 1, 50.0), CategoryCount("18+", 1, 50.0))},
            ),
            computed(
                "distribution.COUNTY",
                "County",
                {
                    "denominator": 2,
                    "distinct_values": 2,
                    "empty": 0,
                    "shown": 1,
                    "other_values": 1,
                    "other_count": 1,
                    "other_percent": 50.0,
                },
                {"COUNTY": (CategoryCount("Suffolk | County", 1, 50.0),)},
            ),
            computed(
                "distribution.ETHNICITY",
                "Ethnicity",
                {"denominator": 2, "distinct_values": 1, "empty": 1},
                {"ETHNICITY": (CategoryCount("hispanic", 1, 50.0), CategoryCount(None, 1, 50.0))},
            ),
            ProfileSection.skipped("distribution.STATE", "State", "patients.csv has no STATE"),
            computed(
                "date_range",
                "Date range",
                {
                    "BIRTHDATE": {
                        "min": "1950-01-01", "max": "2021-01-01",
                        "valid": 3, "empty": 0, "unparseable": 0,
                    },
                    "DEATHDATE": {
                        "min": None, "max": None, "valid": 0, "empty": 3, "unparseable": 0,
                    },
                },
            ),
            computed(
                "completeness",
                "Empty and unparseable values",
                {
                    "BIRTHDATE": {"rows": 3, "empty": 0, "unparseable": 0},
                    "GENDER": {"rows": 3, "empty": 1, "unparseable": None},
                },
            ),
        ),
        generated_at="2026-09-28T00:00:00+00:00",
    )
    values.update(overrides)
    return DatasetProfile(**values)


def test_json_round_trip_and_trailing_newline():
    text = dumps(profile())
    assert text.endswith("}\n")
    assert loads(text) == profile()
    assert dumps(loads(text)) == text
    assert json.loads(text)["schema_version"] == 1


def test_markdown_states_the_reference_date_and_that_it_is_an_approximation():
    text = render_markdown(profile())
    assert "**2026-08-17** — an **approximation**, source `max_encounter_date`." in text
    assert "> APPROXIMATION: the latest encounters.START/STOP value" in text


def test_markdown_for_an_exact_reference_date():
    exact = ReferenceDate("2026-08-18", ReferenceSource.SYNTHEA_METADATA, "endTime=20260818", False)
    text = render_markdown(profile(reference_date=exact, notes=("endTime is early.",)))
    assert "**2026-08-18** — exact, source `synthea_metadata`." in text
    assert "- endTime is early." in text


def test_markdown_without_a_reference_date_shows_the_reason():
    text = render_markdown(profile(reference_date=None, reference_reason="no encounters"))
    assert "## Reference date\n\n`SKIPPED` — no encounters" in text


def test_markdown_renders_every_section_in_order():
    text = render_markdown(profile())
    order = [
        "## Scope",
        "## Reference date",
        "## Population",
        "## Age at the reference date",
        "## County",
        "## Ethnicity",
        "## State",
        "## Date range",
        "## Empty and unparseable values",
    ]
    positions = [text.index(title) for title in order]
    assert positions == sorted(positions)


def test_markdown_tables_and_values():
    text = render_markdown(profile())
    assert "| Alive | 2 | 66.67 |" in text
    assert "Median **40.5**, minimum **5**, maximum **76** years." in text
    assert "| 0-17 | 1 | 50.00 |" in text
    assert "| *(empty)* | 1 | 50.00 |" in text
    assert "| Suffolk \\| County | 1 | 50.00 |" in text
    assert "| *(1 other value(s))* | 1 | 50.00 |" in text
    assert "| `DEATHDATE` | — | — | 0 | 3 | 0 |" in text
    assert "| `GENDER` | 3 | 1 | — |" in text
    assert "## State\n\n`SKIPPED` — patients.csv has no STATE" in text


def test_markdown_lists_absent_tables_and_flags_an_incomplete_profile():
    inputs = (
        TableInput("patients", "x", InputState.UNREADABLE, reason="not UTF-8"),
        TableInput("encounters", "y", InputState.ABSENT, reason="not in the dataset"),
    )
    text = render_markdown(profile(inputs=inputs))
    assert "- `patients`: **unreadable** — not UTF-8" in text
    assert "- `encounters`: **absent** — not in the dataset" in text
    assert "**This profile is incomplete:**" in text
    assert "incomplete" not in render_markdown(profile()).split("## Reference date")[0]


def test_markdown_has_no_verdict_vocabulary():
    text = render_markdown(profile())
    for word in ("PASS", "FAIL", "WARNING", "severity"):
        assert word not in text


@pytest.mark.parametrize("writer, suffix", [(write_json, ".json"), (write_markdown, ".md")])
def test_writers_create_the_directory(tmp_path, writer, suffix):
    path = writer(profile(), tmp_path / "nested" / f"out{suffix}")
    assert path.exists() and path.read_text(encoding="utf-8").endswith("\n")


def code_section(**overrides) -> ProfileSection:
    values = dict(
        section_id="codes.allergies",
        title="Allergies: most common codes",
        status=SectionStatus.COMPUTED,
        metrics={
            "denominator": 4,
            "patients_with_records": 3,
            "rows": 6,
            "rows_alive": 5,
            "distinct_codes": 3,
            "distinct_codes_alive": 2,
            "shown": 2,
            "codes_with_multiple_descriptions": 1,
            "code_identity": "SYSTEM+CODE",
        },
        notes=("Historical count, not active.", "Ordered by patients."),
        codes=(
            CodeCount("SNOMED-CT", "123", "Peanut | nut", 3, 75.0, 4, description_variants=2),
            CodeCount(None, "999", None, 1, 25.0, 1),
        ),
        multi_description_codes=(
            CodeDescriptions(
                "SNOMED-CT",
                "123",
                (DescriptionCount("Peanut | nut", 3), DescriptionCount("peanut", 1)),
            ),
        ),
    )
    values.update(overrides)
    return ProfileSection(**values)


def test_code_section_leads_with_the_historical_note_then_the_table():
    text = render_markdown(profile(sections=(code_section(),)))
    body = text.split("## Allergies: most common codes\n\n", 1)[1]
    assert body.startswith("> Historical count, not active.")
    assert "| # | SYSTEM | CODE | DESCRIPTION | Patients | % alive | Records |" in body
    assert (
        "| 1 | `SNOMED-CT` | `123` | Peanut \\| nut *(2 descriptions)* | 3 | 75.00 | 4 |"
        in body
    )
    assert "| 2 | — | `999` | — | 1 | 25.00 | 1 |" in body
    assert body.index("| # |") < body.index("- Ordered by patients.")
    assert body.count("Historical count, not active.") == 1


def test_code_section_lists_every_description_variant():
    body = render_markdown(profile(sections=(code_section(),)))
    assert "Codes written with more than one description: **1**." in body
    assert "| `SNOMED-CT` `123` | Peanut \\| nut | 3 |" in body
    assert "|  | peanut | 1 |" in body


def test_code_section_without_system_has_no_system_column():
    section = code_section(
        section_id="codes.medications",
        title="Medications: most common codes",
        metrics={**code_section().metrics, "code_identity": "CODE"},
        codes=(CodeCount(None, "313782", "Acetaminophen", 2, 50.0, 3),),
        multi_description_codes=(),
    )
    body = render_markdown(profile(sections=(section,)))
    assert "| # | CODE | DESCRIPTION | Patients | % alive | Records |" in body
    assert "| 1 | `313782` | Acetaminophen | 2 | 50.00 | 3 |" in body
    assert "more than one description" not in body


def test_a_skipped_code_section_shows_its_reason():
    section = ProfileSection.skipped(
        "codes.careplans", "Care plans: most common codes", "careplans.csv is not in the dataset"
    )
    text = render_markdown(profile(sections=(section,)))
    assert "## Care plans: most common codes\n\n`SKIPPED` — careplans.csv is not in the dataset" in text
