"""Tests for the JSON and Markdown renderings of an incidence report."""

from __future__ import annotations

import json
import re

from synthea_quality.incidence.models import (
    ConditionIncidence,
    IncidenceRate,
    IncidenceReport,
    IncidenceStratum,
    Window,
)
from synthea_quality.incidence.render import (
    dumps,
    render_condition,
    render_markdown,
    write_json,
    write_markdown,
)
from synthea_quality.prevalence.models import CodeRef, Expected, ExpectedComparison
from synthea_quality.profile.models import (
    InputState,
    ReferenceDate,
    ReferenceSource,
    SectionStatus,
    TableInput,
)

YEAR = 365.25


def mi(**overrides) -> ConditionIncidence:
    rate = IncidenceRate(1, 169_266)
    values = dict(
        name="Myocardial infarction",
        codes=(CodeRef("22298006", "http://snomed.info/sct"),),
        status=SectionStatus.COMPUTED,
        acute=True,
        rate=rate,
        strata=(
            IncidenceStratum("age_band", "65+", IncidenceRate(1, 5_297)),
            IncidenceStratum("age_band", "0-4", IncidenceRate(0, 0)),
            IncidenceStratum("GENDER", None, IncidenceRate(0, 3_000)),
        ),
        expected=(ExpectedComparison(Expected("incidence", 3.0, "CDC"), rate),),
        metrics={"followed": 101, "prior_cases": 5, "at_risk": 96,
                 "records_on_the_event_day": 1, "later_records_not_counted": 0},
        notes=("5 patient(s) with a record of the condition before entering the window …",),
    )
    values.update(overrides)
    return ConditionIncidence(**values)


def report(**overrides) -> IncidenceReport:
    values = dict(
        data_dir="/data/csv",
        reference_date=ReferenceDate(
            "2026-08-17", ReferenceSource.MAX_ENCOUNTER_DATE, "APPROXIMATION: latest", True
        ),
        reference_reason=None,
        window=Window(5, "2021-08-17", "2026-08-17"),
        population="all",
        age_bands=(0, 65),
        conditions=(mi(),),
        history={"years_of_history": None},
        cohort={"followed": 101, "excluded": {"died_before_window": 7, "birthdate_unusable": 0}},
        inputs=(TableInput("patients", "population", InputState.READ, rows=108),),
        notes=("Incidence counts every patient, deceased included, until their death: …",),
        generated_at="2026-09-28T00:00:00+00:00",
    )
    values.update(overrides)
    return IncidenceReport(**values)


def test_json_is_versioned_and_carries_person_days():
    data = json.loads(dumps(report()))
    assert data["schema_version"] == 1
    assert data["population"] == "all"
    rate = data["conditions"][0]["rate"]
    assert rate["person_days"] == 169_266 and rate["events"] == 1
    assert data["conditions"][0]["expected"][0]["position_in_ci95"] == "inside"


def test_markdown_states_population_window_and_definitions_before_numbers():
    text = render_markdown(report())
    order = ["## Scope", "## Reference date", "## Window, population and exported history",
             "## Definitions", "## Condition: Myocardial infarction (declared acute)"]
    positions = [text.index(title) for title in order]
    assert positions == sorted(positions)
    assert "- Population: every patient, deceased included until their death" in text
    assert "  - left out, died before the window started: 7" in text
    assert "empty or unparseable BIRTHDATE" not in text  # zero counts are not listed
    assert "Window: **2021-08-17** to **2026-08-17** (5 years)." in text
    assert "exact (Garwood) Poisson interval" in text


def test_condition_rate_strata_and_counts():
    text = render_markdown(report())
    years = f"{169_266 / YEAR:.2f}"
    assert f"| 1 | {years} | 2.16 | 0.05–12.02 |" in text
    assert "| 0-4 | 0 | 0.00 | — | — |" in text
    assert "| *(empty)* | 0 |" in text
    assert "prior cases (not at risk): 5; at risk: 96" in text
    assert "1 on the day of a first event, 0 on a later day" in text


def test_reference_values_are_positions_not_verdicts():
    text = render_markdown(report())
    assert "| 3.00 | 2.16 | -0.84 | inside the 95% CI | CDC |" in text
    assert not re.search(r"\b(pass(ed|es)?|fail(ed|s)?|ok|match(ed|es)?)\b", text, re.I)


def test_alive_only_population_and_skipped_conditions():
    skipped = mi(status=SectionStatus.SKIPPED, reason="no reference date", rate=None, strata=(),
                 expected=())
    text = render_markdown(report(population="alive", conditions=(skipped,), window=None))
    assert "the patients alive at the end of the simulation (`--alive-only`)" in text
    assert "`SKIPPED` — no reference date" in text
    assert "No window" in text


def test_writers(tmp_path):
    assert write_json(report(), tmp_path / "x" / "i.json").read_text("utf-8").endswith("\n")
    assert write_markdown(report(), tmp_path / "x" / "i.md").read_text("utf-8").startswith("#")


def test_a_condition_renders_under_any_heading_level():
    text = render_condition(mi(), level=3, title_prefix="Incidence: ")
    assert text.startswith("### Incidence: Myocardial infarction (declared acute)\n\nCodes:")
    assert "#### By age band (person-years split between bands)" in text
    assert render_condition(mi()) in render_markdown(report())


def _unescaped_pipes(line: str) -> int:
    return sum(1 for i, ch in enumerate(line) if ch == "|" and (i == 0 or line[i - 1] != "\\"))


def _tables(text: str) -> list[list[str]]:
    tables: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("| "):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def test_expected_source_with_pipes_and_line_breaks_stays_in_its_cell():
    from synthea_quality.incidence.models import IncidenceRate

    nasty = "CDC | line\nbreak\rreturn"
    rate = IncidenceRate(1, 169_266)
    bad = mi(expected=(ExpectedComparison(Expected("incidence", 3.0, nasty), rate),))
    text = render_condition(bad)
    assert "CDC \\| line break return" in text
    assert "\r" not in text
    tables = _tables(text)
    assert tables
    for table in tables:
        wanted = _unescaped_pipes(table[0])
        assert all(_unescaped_pipes(line) == wanted for line in table)


def test_tiny_rates_and_person_years_are_never_shown_as_zero():
    from synthea_quality.incidence.models import IncidenceRate

    tiny = IncidenceRate(1, 100_000_000)
    cond = mi(rate=tiny, strata=(), expected=(), metrics={})
    text = render_condition(cond)
    assert "| 1 | 273785.08 | 0.0037 | 9.2e-05–0.02 |" in text


def test_one_day_of_follow_up_is_not_shown_as_zero_person_years():
    from synthea_quality.incidence.models import IncidenceRate

    one_day = IncidenceRate(0, 1)
    cond = mi(rate=one_day, strata=(), expected=(), metrics={})
    text = render_condition(cond)
    assert "| 0 | 0.0027 | 0.00 |" in text


def test_a_tiny_difference_is_never_shown_as_zero():
    from synthea_quality.incidence.models import IncidenceRate

    rate = IncidenceRate(1, 100_000_000)
    cond = mi(
        rate=rate,
        strata=(),
        expected=(ExpectedComparison(Expected("incidence", 0.0, "CDC"), rate),),
        metrics={},
    )
    text = render_condition(cond)
    assert "| 0.00 | 0.0037 | +0.0037 | outside the 95% CI | CDC |" in text
