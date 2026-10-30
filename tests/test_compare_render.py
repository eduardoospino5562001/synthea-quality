"""Tests for rendering the comparison as Markdown and JSON (render.py)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from synthea_quality import __version__
from synthea_quality.compare.compute import compare
from synthea_quality.compare.models import (
    CHANGED,
    COMPUTED,
    COUNT,
    NOT_COMPARED,
    NUMBER,
    ONLY_IN_BASELINE,
    ONLY_IN_CANDIDATE,
    PER_1000_PERSON_YEARS,
    PERCENT,
    PROPORTION,
    SKIPPED,
    UNCHANGED,
    YEARS,
    Comparison,
    ComparisonKey,
    Figure,
)
from synthea_quality.compare.render import (
    ComparisonResult,
    render_json,
    render_markdown,
    summary_counts,
)
from synthea_quality.export_history import ExportHistory
from synthea_quality.validate.models import ModuleInfo, ModuleValidationReport

GENERATED_AT = "2026-09-29T00:00:00+00:00"


def _report(generated_at: str = GENERATED_AT):
    """A minimal ModuleValidationReport with a fixed generated_at."""
    return ModuleValidationReport(
        data_dir="/data",
        module_file="module.json",
        module=ModuleInfo("Test module"),
        reference_date=None,
        reference_reason="no encounters",
        age_bands=(0,),
        alive=2,
        population=(),
        conditions=(),
        generated_at=generated_at,
    )


def _comparison(key, *, state=None, base=0.5, cand=0.5, unit=PROPORTION, reason=None):
    base_fig = Figure(unit=unit, value=base, state=COMPUTED) if base is not None else None
    cand_fig = Figure(unit=unit, value=cand, state=COMPUTED) if cand is not None else None
    if state is None:
        if base is None or cand is None:
            state = UNCHANGED if base is None and cand is None else CHANGED
        elif base == cand:
            state = UNCHANGED
        else:
            state = CHANGED
    difference = None if (base is None or cand is None) else round(cand - base, 6)
    relative_difference = None if (base is None or base == 0) else (cand - base) / base
    return Comparison(
        key=key,
        unit=unit,
        baseline=base_fig,
        candidate=cand_fig,
        difference=difference,
        relative_difference=relative_difference,
        state=state,
        reason=reason,
    )


def _result(comparisons, generated_at=GENERATED_AT):
    base = _report(generated_at)
    cand = _report(generated_at)
    return ComparisonResult(
        module_file="module.json",
        module=ModuleInfo("Test module"),
        baseline_dir="/baseline",
        candidate_dir="/candidate",
        baseline_report=base,
        candidate_report=cand,
        comparisons=comparisons,
        generated_at=generated_at,
    )


def test_markdown_against_itself_reports_zero_changes():
    key = ComparisonKey("conditions", "MI", "point")
    result = _result([_comparison(key, base=0.05, cand=0.05)])
    markdown = render_markdown(result)
    assert (
        "1 figures: 1 compared (0 changed, 1 unchanged), 0 skipped, 0 only in the baseline, 0 only in the candidate."
        in markdown
    )
    assert "No figures changed." in markdown
    assert "Conditions" in markdown


def test_markdown_lists_only_changed_rows_per_block():
    keys = [
        ComparisonKey("conditions", "MI", "point"),
        ComparisonKey("conditions", "HTN", "point"),
        ComparisonKey("observations", "8480-6", "n", "mm[Hg]"),
    ]
    comparisons = [
        _comparison(keys[0], base=0.05, cand=0.08),  # changed
        _comparison(keys[1], base=0.5, cand=0.5),  # unchanged
        _comparison(keys[2], base=2, cand=3),  # changed
    ]
    result = _result(comparisons)
    markdown = render_markdown(result)
    conditions = markdown.split("## Conditions")[1].split("##")[0]
    assert "MI" in conditions
    assert "HTN" not in conditions  # unchanged, not listed
    observations = markdown.split("## Observations")[1].split("##")[0]
    assert "8480-6" in observations


def test_markdown_largest_changes_ordered_by_relative_difference():
    high = _comparison(
        ComparisonKey("conditions", "MI", "point"),
        base=0.04, cand=0.06,  # relative +0.5
    )
    low = _comparison(
        ComparisonKey("conditions", "HTN", "point"),
        base=0.5, cand=0.55,  # relative +0.1
    )
    tie = _comparison(
        ComparisonKey("conditions", "MI", "lifetime"),
        base=0.04, cand=0.06,  # relative +0.5, same as high, tie broken by key
    )
    null_rel = _comparison(
        ComparisonKey("indexed", "cohort", "patients"),
        base=0, cand=3, unit=COUNT,  # relative None -> last
    )
    result = _result([low, high, tie, null_rel])
    markdown = render_markdown(result, top=4)
    largest = markdown.split("## Largest changes")[1].split("##")[0]
    mi_lifetime = largest.index("MI")
    mi_point = largest.index("MI", mi_lifetime + 1)
    htn = largest.index("HTN")
    cohort = largest.index("cohort")
    assert mi_lifetime < mi_point < htn < cohort  # ties by key, null last


def test_markdown_top_greater_than_changes_returns_all():
    comparisons = [_comparison(ComparisonKey("conditions", "MI", "point"), base=0.05, cand=0.08)]
    result = _result(comparisons)
    markdown = render_markdown(result, top=100)
    assert "## Largest changes (top 100)" in markdown
    # The one change is in the table, not only counted in the summary.
    assert "| conditions | MI | point |  | 5.00% | 8.00% | +3.00 pp | +60.00% |" in markdown.splitlines()


def test_markdown_skipped_population_section_is_not_a_metric():
    skipped_key = ComparisonKey("population", "age", "age", "")
    comparisons = [
        _comparison(
            skipped_key,
            base=None,
            cand=None,
            unit=COUNT,
            state=SKIPPED,
            reason="no reference date: none was given",
        ),
        _comparison(ComparisonKey("population", "population", "alive"), base=2, cand=2),
    ]
    result = _result(comparisons)
    markdown = render_markdown(result)
    assert "Section age not measured: no reference date: none was given." in markdown
    assert "Section age not measured:" in markdown


def test_markdown_reference_date_difference_note():
    from synthea_quality.profile.models import ReferenceDate, ReferenceSource
    base = ModuleValidationReport(
        data_dir="/data", module_file="module.json", module=ModuleInfo("Test module"),
        reference_date=None, reference_reason="no encounters", age_bands=(0,),
        alive=2, population=(), conditions=(), generated_at=GENERATED_AT,
    )
    cand = ModuleValidationReport(
        data_dir="/data", module_file="module.json", module=ModuleInfo("Test module"),
        reference_date=ReferenceDate("2026-02-01", ReferenceSource.MAX_ENCOUNTER_DATE, "x", False),
        reference_reason=None, age_bands=(0,), alive=2, population=(), conditions=(),
        generated_at=GENERATED_AT,
    )
    result = ComparisonResult(
        module_file="module.json",
        module=ModuleInfo("Test module"),
        baseline_dir="/baseline",
        candidate_dir="/candidate",
        baseline_report=base,
        candidate_report=cand,
        comparisons=[],
        generated_at=GENERATED_AT,
    )
    markdown = render_markdown(result)
    assert (
        "> Note: the reference dates differ (baseline none, candidate 2026-02-01): this changes who is alive and "
        "their ages." in markdown.splitlines()
    )


def test_markdown_no_verdict_words():
    key = ComparisonKey("conditions", "MI", "point")
    result = _result([_comparison(key, base=0.05, cand=0.08)])
    markdown = render_markdown(result)
    import re

    for word in ("pass", "fail", "\bok\b", "match", "significant"):
        assert not re.search(word, markdown, re.IGNORECASE), f"found {word!r} in Markdown"


def test_json_schema_version_and_fields():
    key = ComparisonKey("conditions", "MI", "point")
    result = _result([_comparison(key, base=0.05, cand=0.08)])
    data = render_json(result)
    import json

    parsed = json.loads(data)
    assert parsed["schema_version"] == 1
    assert parsed["tool_version"] == __version__
    assert parsed["generated_at"] == GENERATED_AT
    assert parsed["compared"] == 1
    assert parsed["changed"] == 1
    assert parsed["unchanged"] == 0
    assert parsed["baseline"]["data_dir"] == "/baseline"
    assert parsed["candidate"]["data_dir"] == "/candidate"
    assert parsed["module"]["name"] == "Test module"
    assert len(parsed["comparisons"]) == 1
    assert parsed["comparisons"][0]["key"] == {
        "block": "conditions",
        "element": "MI",
        "measure": "point",
        "stratum": "",
    }
    assert parsed["comparisons"][0]["baseline"]["value"] == 0.05
    assert parsed["comparisons"][0]["candidate"]["value"] == 0.08


def test_json_deterministic_except_generated_at():
    key = ComparisonKey("conditions", "MI", "point")
    result_a = _result([_comparison(key, base=0.05, cand=0.08)], generated_at="t1")
    result_b = _result([_comparison(key, base=0.05, cand=0.08)], generated_at="t2")
    json_a = render_json(result_a)
    json_b = render_json(result_b)
    assert json_a.replace("t1", "GENERATED") == json_b.replace("t2", "GENERATED")


def test_markdown_deterministic_except_generated_at():
    key = ComparisonKey("conditions", "MI", "point")
    result_a = _result([_comparison(key, base=0.05, cand=0.08)], generated_at="t1")
    result_b = _result([_comparison(key, base=0.05, cand=0.08)], generated_at="t2")
    md_a = render_markdown(result_a)
    md_b = render_markdown(result_b)
    assert md_a.replace("t1", "GENERATED") == md_b.replace("t2", "GENERATED")


def test_markdown_unicode_and_pipe_names_keep_distinct_keys():
    keys = [
        ComparisonKey("observations", "H\u2764\ufe0ftension", "n", "mm[Hg]"),
        ComparisonKey("observations", "H|tension", "n", "mm[Hg]"),
    ]
    comparisons = [
        _comparison(keys[0], base=2, cand=3),
        _comparison(keys[1], base=2, cand=4),
    ]
    result = _result(comparisons)
    markdown = render_markdown(result)
    assert "H\u2764\ufe0ftension" in markdown
    assert "H\\|tension" in markdown  # the vertical bar is escaped inside the cell


def test_skipped_figures_are_counted_apart_from_the_compared_ones():
    key = ComparisonKey("conditions", "MI", "incidence")
    comparisons = [
        _comparison(key, base=None, cand=None, unit=PER_1000_PERSON_YEARS, state=SKIPPED, reason="no condition records")
    ]
    result = _result(comparisons)
    markdown = render_markdown(result)
    assert (
        "1 figures: 0 compared (0 changed, 0 unchanged), 1 skipped, 0 only in the baseline, 0 only in the candidate."
        in markdown
    )


def test_markdown_table_cells_keep_one_column_count_per_row():
    """A cell of the element or the stratum with '|', Unicode or a line break:
    every row of the table keeps the same cell count as the header."""
    keys = [
        ComparisonKey("observations", "Systolic blood pressure", "median", "mm[Hg]|age_band:18-44"),
        ComparisonKey("observations", "H\u2764\ufe0ftension|with a pipe", "n", "mm[Hg]"),
        ComparisonKey("observations", "Systolic blood pressure", "p25", "mm[Hg]\nage_band:65-74"),
    ]
    comparisons = [
        _comparison(keys[0], base=120, cand=122),
        _comparison(keys[1], base=2, cand=3, unit=COUNT),
        _comparison(keys[2], base=90, cand=95),
    ]
    result = _result(comparisons)
    markdown = render_markdown(result)

    table = markdown.split("## Observations")[1].split("##")[0]
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    header_cells = rows[0].strip().strip("|").split(" | ")
    for row in rows[1:]:
        if row.strip().startswith("| ---"):
            continue
        cells = row.strip().strip("|").split(" | ")
        assert len(cells) == len(header_cells), f"{row!r} has {len(cells)} cells, the header {len(header_cells)}"

    # Pipes, line breaks and Unicode all land in a single cell.
    assert "mm[Hg]\\|age_band:18-44" in markdown
    assert "H\u2764\ufe0ftension\\|with a pipe" in markdown
    assert "mm[Hg] age_band:65-74" in markdown


def test_markdown_states_the_units_once_under_largest_changes():
    """One legend line (exactly once) under 'Largest changes' states the units:
    percentage points for proportions, per 1,000 person-years for rates, 2
    decimals for ages, observation values and plain numbers, integers for
    counts and days. No verdict words."""
    result = _result([_comparison(ComparisonKey("conditions", "MI", "point"), base=0.05, cand=0.08)])
    markdown = render_markdown(result)
    legend = (
        "pp = percentage points; incidence per 1,000 person-years; ages, observation values and plain numbers "
        "with 2 decimals; counts and days as integers."
    )
    assert markdown.count(legend) == 1
    section = markdown.split("## Largest changes")[1].split("##")[0]
    assert legend in section


@pytest.mark.parametrize(
    ("key", "base", "cand", "unit", "row"),
    [
        (
            ComparisonKey("population", "population", "alive_percent"),
            91.67,
            92.59,
            PERCENT,
            "| population | population | alive_percent |  | 91.67% | 92.59% | +0.92 pp | +1.00% |",
        ),
        (
            ComparisonKey("population", "age", "median"),
            35.5,
            36.0,
            YEARS,
            "| population | age | median |  | 35.50 | 36.00 | +0.50 | +1.41% |",
        ),
        (
            ComparisonKey("observations", "Score", "median"),
            1.5,
            2.25,
            NUMBER,
            "| observations | Score | median |  | 1.50 | 2.25 | +0.75 | +50.00% |",
        ),
        (
            ComparisonKey("observations", "Systolic", "median", "mm[Hg]"),
            119.0,
            125.5,
            "mm[Hg]",
            "| observations | Systolic | median | mm[Hg] | 119.00 | 125.50 | +6.50 | +5.46% |",
        ),
        (
            ComparisonKey("conditions", "MI", "incidence"),
            2.16,
            3.5,
            PER_1000_PERSON_YEARS,
            "| conditions | MI | incidence |  | 2.16 | 3.50 | +1.34 | +62.04% |",
        ),
        (
            ComparisonKey("conditions", "MI", "point"),
            0.05,
            0.08,
            PROPORTION,
            "| conditions | MI | point |  | 5.00% | 8.00% | +3.00 pp | +60.00% |",
        ),
    ],
    ids=["percent", "years", "number", "observation-units", "rate", "proportion"],
)
def test_each_unit_is_formatted_in_the_largest_changes_table(key, base, cand, unit, row):
    """A percent is not multiplied by 100 again; ages, plain numbers and the units
    of an observation have 2 fixed decimals; proportions and rates keep their format."""
    markdown = render_markdown(_result([_comparison(key, base=base, cand=cand, unit=unit)]))
    assert row in markdown.splitlines()


def test_summary_counts_of_no_figures():
    assert summary_counts([]) == (
        "0 figures: 0 compared (0 changed, 0 unchanged), 0 skipped, 0 only in the baseline, 0 only in the candidate"
    )


def test_summary_counts_count_skipped_and_one_sided_figures_apart_from_compared():
    comparisons = [
        _comparison(ComparisonKey("conditions", "A", "point"), base=0.05, cand=0.08),
        _comparison(ComparisonKey("conditions", "B", "point"), base=0.05, cand=0.05),
        _comparison(ComparisonKey("conditions", "C", "point"), base=0.05, cand=0.05),
        _comparison(ComparisonKey("conditions", "D", "point"), base=None, cand=None, state=SKIPPED, reason="r"),
        _comparison(ComparisonKey("conditions", "E", "point"), base=None, cand=None, state=ONLY_IN_BASELINE),
        _comparison(ComparisonKey("conditions", "F", "point"), base=None, cand=None, state=ONLY_IN_BASELINE),
        _comparison(ComparisonKey("conditions", "G", "point"), base=None, cand=None, state=ONLY_IN_CANDIDATE),
    ]
    assert summary_counts(comparisons) == (
        "7 figures: 3 compared (1 changed, 2 unchanged), 1 skipped, 2 only in the baseline, 1 only in the candidate"
    )


def test_the_summary_and_every_block_use_the_same_counts_line():
    comparisons = [
        _comparison(ComparisonKey("population", "population", "alive"), base=2, cand=3, unit=COUNT),
        _comparison(ComparisonKey("conditions", "MI", "point"), base=0.05, cand=0.05),
        _comparison(ComparisonKey("conditions", "MI", "lifetime"), base=None, cand=None, state=SKIPPED, reason="r"),
    ]
    lines = render_markdown(_result(comparisons)).splitlines()
    summary = lines[lines.index("## Summary") + 2]
    population = lines[lines.index("## Population") + 2]
    conditions = lines[lines.index("## Conditions") + 2]
    assert summary == (
        "3 figures: 2 compared (1 changed, 1 unchanged), 1 skipped, 0 only in the baseline, 0 only in the candidate."
    )
    assert population == (
        "1 figures: 1 compared (1 changed, 0 unchanged), 0 skipped, 0 only in the baseline, 0 only in the candidate."
    )
    assert conditions == (
        "2 figures: 1 compared (0 changed, 1 unchanged), 1 skipped, 0 only in the baseline, 0 only in the candidate."
    )
    assert not any(line.startswith("Compared ") for line in lines)


def test_skipped_and_one_sided_figures_are_listed_with_why_they_were_not_compared():
    comparisons = [
        _comparison(ComparisonKey("conditions", "MI", "point"), base=0.05, cand=0.08),
        _comparison(
            ComparisonKey("conditions", "MI", "lifetime"), base=None, cand=None, state=SKIPPED,
            reason="no condition records",
        ),
        _comparison(
            ComparisonKey("conditions", "MI", "incidence", "sex:F"), base=None, cand=None, state=ONLY_IN_BASELINE
        ),
        _comparison(
            ComparisonKey("conditions", "MI", "incidence", "sex:M"), base=None, cand=None, state=ONLY_IN_CANDIDATE
        ),
    ]
    lines = render_markdown(_result(comparisons)).splitlines()
    section = lines[lines.index("## Conditions"):]
    assert "Not compared:" in section
    assert "| Element | Measure | Stratum | Baseline | Candidate | Why not compared |" in section
    assert "| MI | lifetime |  | — | — | no condition records |" in section
    assert "| MI | incidence | sex:F | — | — | only in the baseline |" in section
    assert "| MI | incidence | sex:M | — | — | only in the candidate |" in section
    # The changed figure stays in the table of changes, not in this one.
    assert not any(line.startswith("| MI | point |") and line.endswith("|  |") for line in section)


def test_a_block_with_every_figure_compared_has_no_not_compared_table():
    comparisons = [_comparison(ComparisonKey("conditions", "MI", "point"), base=0.05, cand=0.05)]
    markdown = render_markdown(_result(comparisons))
    assert "Not compared:" not in markdown
    assert "Why not compared" not in markdown


def test_a_figure_covered_by_a_skipped_side_shows_the_measured_value_and_the_reason():
    reason = "observations.csv is not in the dataset"
    baseline = {
        ComparisonKey("indexed", "Death", "patients", "total:all"): Figure(
            unit=PROPORTION, value=0.5, numerator=1, denominator=2
        ),
    }
    candidate = {
        ComparisonKey("indexed", "Death", "death", ""): Figure(unit=COUNT, state=SKIPPED, reason=reason),
    }
    lines = render_markdown(_result(compare(baseline, candidate))).splitlines()
    assert f"| Death | patients | total:all | 50.00% | — | {reason} |" in lines
    assert f"| Death | death |  | — | — | {reason} |" in lines


def _result_with(baseline: ModuleValidationReport, candidate: ModuleValidationReport, comparisons=()):
    """A result whose two sides are the given reports."""
    return ComparisonResult(
        module_file="module.json",
        module=ModuleInfo("Test module"),
        baseline_dir="/baseline",
        candidate_dir="/candidate",
        baseline_report=baseline,
        candidate_report=candidate,
        comparisons=list(comparisons),
        generated_at=GENERATED_AT,
    )


def test_the_markdown_and_the_json_say_what_is_not_compared():
    result = _result([_comparison(ComparisonKey("conditions", "MI", "point"), base=0.05, cand=0.05)])
    lines = render_markdown(result).splitlines()
    assert "## Parts not compared" in lines
    for part in NOT_COMPARED:
        assert f"- {part}" in lines
    assert json.loads(render_json(result))["not_compared"] == list(NOT_COMPARED)


def test_the_notes_of_each_side_are_in_the_markdown():
    baseline = replace(_report(), notes=("The exported history is unknown.",))
    candidate = replace(_report(), notes=("Incidence counts every patient.", "The exported history is unknown."))
    lines = render_markdown(_result_with(baseline, candidate)).splitlines()
    notes = lines[lines.index("## Notes"):]
    assert notes[:9] == [
        "## Notes",
        "",
        "Baseline:",
        "- The exported history is unknown.",
        "",
        "Candidate:",
        "- Incidence counts every patient.",
        "- The exported history is unknown.",
        "",
    ]


def test_without_notes_there_is_no_notes_section():
    assert "## Notes" not in render_markdown(_result([]))


def test_a_different_years_of_history_changes_the_exported_records_not_the_patients():
    history = ExportHistory(years=10, status="read", metadata_file="meta.json", cutoff=None)
    candidate = replace(_report(), export_history=history)
    markdown = render_markdown(_result_with(_report(), candidate))
    assert (
        "> Note: the years of history differ (baseline None, candidate 10): this changes which records were "
        "exported (lifetime prevalence, ever use and prior cases look back)." in markdown.splitlines()
    )
    assert "alive" not in markdown.split("## Reference date")[1].split("## Summary")[0]


def test_the_json_counts_the_figures_apart_from_the_compared_ones():
    comparisons = [
        _comparison(ComparisonKey("conditions", "A", "point"), base=0.05, cand=0.08),
        _comparison(ComparisonKey("conditions", "B", "point"), base=0.05, cand=0.05),
        _comparison(ComparisonKey("conditions", "C", "point"), base=None, cand=None, state=SKIPPED, reason="r"),
        _comparison(ComparisonKey("conditions", "D", "point"), base=None, cand=None, state=ONLY_IN_BASELINE),
    ]
    data = json.loads(render_json(_result(comparisons)))
    assert (data["figures"], data["compared"], data["changed"], data["unchanged"]) == (4, 2, 1, 1)
    assert (data["skipped"], data["only_in_baseline"], data["only_in_candidate"]) == (1, 1, 0)
