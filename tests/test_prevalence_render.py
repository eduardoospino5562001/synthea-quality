"""Tests for the JSON and Markdown renderings of a prevalence report."""

from __future__ import annotations

import json
import re

from synthea_quality.prevalence.models import (
    CodeRef,
    ConditionResult,
    Expected,
    ExpectedComparison,
    GeneralRow,
    GeneralTable,
    PrevalenceReport,
    Rate,
    Stratum,
)
from synthea_quality.prevalence.render import dumps, render_markdown, write_json, write_markdown
from synthea_quality.profile.models import (
    InputState,
    ReferenceDate,
    ReferenceSource,
    SectionStatus,
    TableInput,
)

SNOMED = "http://snomed.info/sct"


def condition(**overrides) -> ConditionResult:
    values = dict(
        name="Myocardial infarction",
        codes=(CodeRef("22298006", SNOMED), CodeRef("401314000", SNOMED)),
        status=SectionStatus.COMPUTED,
        point=Rate(0, 99),
        lifetime=Rate(7, 99),
        strata=(
            Stratum("age_band", "65+", Rate(0, 10), Rate(5, 10)),
            Stratum("age_band", "100+", Rate(0, 0), Rate(0, 0)),
            Stratum("GENDER", None, Rate(0, 1), Rate(1, 1)),
        ),
        expected=(
            ExpectedComparison(Expected("lifetime", 0.05, "CDC"), Rate(7, 99)),
            ExpectedComparison(Expected("point", 0.5), Rate(0, 99)),
        ),
        metrics={"records_by_code": {f"{SNOMED}|22298006": 3, f"{SNOMED}|401314000": 4}},
        notes=("Point prevalence is far below lifetime prevalence: …",),
    )
    values.update(overrides)
    return ConditionResult(**values)


def report(**overrides) -> PrevalenceReport:
    values = dict(
        data_dir="/data/csv",
        reference_date=ReferenceDate(
            "2026-08-17", ReferenceSource.MAX_ENCOUNTER_DATE, "APPROXIMATION: latest", True
        ),
        reference_reason=None,
        age_bands=(0, 65, 100),
        alive=99,
        conditions=(condition(),),
        general=GeneralTable(
            SectionStatus.COMPUTED,
            rows=(
                GeneralRow(SNOMED, "59621000", "Essential hypertension (disorder)",
                           Rate(17, 99), Rate(17, 99)),
                GeneralRow(SNOMED, "73595000", "Stress (finding)", Rate(9, 99), Rate(60, 99),
                           social=True),
                GeneralRow(SNOMED, "1", "Rare | thing", Rate(0, 99), Rate(1, 99)),
            ),
            top=2,
            include_social=True,
            notes=("Social and administrative codes are included.",),
        ),
        social_list={"id": "synthea-d9d07a6e-social-1", "codes": 21},
        inputs=(TableInput("patients", "cohort", InputState.READ, rows=108),),
        generated_at="2026-09-28T00:00:00+00:00",
    )
    values.update(overrides)
    return PrevalenceReport(**values)


def test_json_carries_version_definitions_and_provenance():
    data = json.loads(dumps(report()))
    assert data["schema_version"] == 1
    assert data["reference_date"]["source"] == "max_encounter_date"
    assert data["social_list"]["id"] == "synthea-d9d07a6e-social-1"
    mi = data["conditions"][0]
    assert mi["lifetime"]["numerator"] == 7 and mi["lifetime"]["ci95_low"] is not None
    assert mi["expected"][0]["position_in_ci95"] == "inside"
    assert data["general"]["rows"][1]["social"] is True
    assert dumps(report()).endswith("}\n")


def test_markdown_states_definitions_before_results():
    text = render_markdown(report())
    assert text.index("## Definitions") < text.index("## Condition: Myocardial infarction")
    assert "Wilson score interval" in text


def test_condition_totals_and_strata():
    text = render_markdown(report())
    assert "| Lifetime | 7 | 99 | 7.07% | 3.47–13.88% |" in text
    assert "| Point | 0 | 99 | 0.00% | 0.00–3.74% |" in text
    assert "| 65+ | 10 | 0 (0.00%) | 0.00–27.75% | 5 (50.00%) | 23.66–76.34% |" in text
    assert "| 100+ | 0 | 0 (—) | — | 0 (—) | — |" in text
    assert "| *(empty)* | 1 |" in text
    assert "Records used, by code:" in text
    assert "- Point prevalence is far below lifetime prevalence" in text


def test_expected_values_are_described_by_position_never_as_a_verdict():
    text = render_markdown(report())
    assert "| Lifetime | 5.00% | 7.07% | +2.07 pp | inside the 95% CI | CDC |" in text
    assert "| Point | 50.00% | 0.00% | -50.00 pp | outside the 95% CI | — |" in text
    assert not re.search(r"\b(pass|fail|ok|match)", text, re.IGNORECASE)


def test_the_full_report_uses_no_verdict_words():
    text = render_markdown(report()) + render_markdown(report(conditions=()))
    assert not re.search(r"\b(pass(ed|es)?|fail(ed|s)?|ok|match(ed|es)?)\b", text, re.IGNORECASE)


def test_general_table_lists_the_top_rows_and_marks_social_codes():
    text = render_markdown(report())
    assert "Codes with at least one alive patient: **3**, all in `http://snomed.info/sct`" in text
    assert "| 1 | `59621000` | Essential hypertension (disorder) | 17 (17.17%) |" in text
    assert "Stress (finding) *(social/administrative)*" in text
    assert "Rare \\| thing" not in text  # beyond --top


def test_skipped_parts_show_their_reason():
    skipped = report(
        reference_date=None,
        reference_reason="no reference date: none was given",
        alive=None,
        conditions=(
            ConditionResult("MI", (CodeRef("1"),), SectionStatus.SKIPPED, reason="no reference"),
        ),
        general=GeneralTable(SectionStatus.SKIPPED, reason="no reference"),
    )
    text = render_markdown(skipped)
    assert "## Reference date\n\n`SKIPPED` — no reference date: none was given" in text
    assert "Codes: `1`\n\n`SKIPPED` — no reference" in text
    assert "## All conditions by point prevalence\n\n`SKIPPED` — no reference" in text


def test_without_conditions_asked_the_report_says_how_to_ask():
    text = render_markdown(report(conditions=()))
    assert "Use `--condition NAME=CODE[,CODE...]`" in text


def test_writers(tmp_path):
    assert write_json(report(), tmp_path / "a" / "p.json").read_text("utf-8").endswith("\n")
    assert write_markdown(report(), tmp_path / "a" / "p.md").read_text("utf-8").startswith("#")
