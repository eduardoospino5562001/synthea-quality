"""Tests for rendering the comparison as Markdown and JSON (render.py)."""

from __future__ import annotations

from synthea_quality.compare.models import (
    CHANGED,
    COMPUTED,
    COUNT,
    PER_1000_PERSON_YEARS,
    PROPORTION,
    SKIPPED,
    UNCHANGED,
    Comparison,
    ComparisonKey,
    Figure,
)
from synthea_quality.compare.render import (
    ComparisonResult,
    render_json,
    render_markdown,
)
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
    assert "Compared 1 figures: 0 changed, 1 unchanged." in markdown
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
    assert "Compared 1 figures: 1 changed, 0 unchanged." in markdown


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
    assert "Note:" in markdown
    assert "reference dates differ" in markdown


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
    assert parsed["tool_version"] == "0.12.1"
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
    assert "H\\|tension" in markdown  # la barra vertical se escapa en la celda


def test_skipped_count_line_names_the_reason():
    key = ComparisonKey("conditions", "MI", "incidence")
    comparisons = [
        _comparison(key, base=None, cand=None, unit=PER_1000_PERSON_YEARS, state=SKIPPED, reason="no condition records")
    ]
    result = _result(comparisons)
    markdown = render_markdown(result)
    assert "1 could not be measured (skipped, with a reason)." in markdown


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
    percentage points for proportions, per 1,000 person-years for rates, integers
    for counts and days. No verdict words."""
    result = _result([_comparison(ComparisonKey("conditions", "MI", "point"), base=0.05, cand=0.08)])
    markdown = render_markdown(result)
    legend = "pp = percentage points; incidence per 1,000 person-years; counts and days as integers."
    assert markdown.count(legend) == 1
    section = markdown.split("## Largest changes")[1].split("##")[0]
    assert legend in section

