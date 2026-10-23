"""Tests for extracting figures from a module validation and comparing two reports."""

from __future__ import annotations

from pathlib import Path

import pytest
from csv_tables import write_table

from synthea_quality.compare.compute import compare, extract_figures, largest_changes
from synthea_quality.compare.models import (
    CHANGED,
    COUNT,
    ONLY_IN_BASELINE,
    ONLY_IN_CANDIDATE,
    PER_1000_PERSON_YEARS,
    PROPORTION,
    SKIPPED,
    UNCHANGED,
    Comparison,
    ComparisonKey,
    Figure,
)
from synthea_quality.incidence.models import ConditionIncidence, IncidenceRate
from synthea_quality.indexed.definitions import (
    CrossSpec,
    DeathOutcome,
    EventOutcome,
    IndexSpec,
    Window,
)
from synthea_quality.indexed.models import (
    Breakdown,
    CrossResult,
    CrossStratum,
    EventResult,
    IndexedCohortResult,
    SkippedOutcome,
    Stratum,
    ValueSummary,
)
from synthea_quality.medications.definitions import MedicationDefinition
from synthea_quality.observations.definitions import ObservationDefinition
from synthea_quality.prevalence.definitions import ALL_MEASURES, assemble
from synthea_quality.prevalence.models import ConditionResult, Rate
from synthea_quality.profile.models import ReferenceDate, ReferenceSource, SectionStatus
from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.models import (
    ConditionValidation,
    ModuleInfo,
    ModuleValidationReport,
)

SNOMED = "http://snomed.info/sct"
GENERATED_AT = "2026-09-29T00:00:00+00:00"


def reference() -> ReferenceDate:
    return ReferenceDate("2026-01-01", ReferenceSource.MAX_ENCOUNTER_DATE, "the latest encounter date", False)


def condition(name: str) -> ConditionValidation:
    """One computed condition: half the alive patients, one event over 365 person-days."""

    rate = Rate(1, 2)
    return ConditionValidation(
        name=name,
        codes=(),
        acute=False,
        prevalence=ConditionResult(name, (), SectionStatus.COMPUTED, point=rate, lifetime=rate),
        incidence=ConditionIncidence(name, (), SectionStatus.COMPUTED, rate=IncidenceRate(1, 365)),
    )


def test_a_zero_basis_makes_the_relative_difference_none():
    base = Figure(unit=PROPORTION, value=0.0, numerator=0, denominator=10)
    cand = Figure(unit=PROPORTION, value=0.03, numerator=3, denominator=100)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=0.03,
        relative_difference=None,
        state=CHANGED,
    )
    assert comparison.state == CHANGED
    assert comparison.difference == 0.03
    assert comparison.relative_difference is None


def test_equal_figures_are_unchanged():
    base = cand = Figure(unit=PROPORTION, value=0.05, numerator=5, denominator=100)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=0.0,
        relative_difference=0.0,
        state=UNCHANGED,
    )
    assert comparison.state == UNCHANGED
    assert comparison.difference == 0.0
    assert comparison.relative_difference == 0.0


def test_a_change_is_candidate_minus_baseline():
    base = Figure(unit=PROPORTION, value=0.06)
    cand = Figure(unit=PROPORTION, value=0.04)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=-0.02,
        relative_difference=(0.04 - 0.06) / 0.06,
        state=CHANGED,
    )
    assert comparison.difference == -0.02


def test_a_figure_found_on_one_side_only():
    base = {ComparisonKey("conditions", "MI", "point"): Figure(unit=PROPORTION, value=0.5)}
    comparisons = compare(base, {})
    assert len(comparisons) == 1
    assert comparisons[0].state == ONLY_IN_BASELINE
    assert comparisons[0].baseline.value == 0.5
    assert comparisons[0].candidate is None
    assert comparisons[0].difference is None
    assert comparisons[0].relative_difference is None

    comparisons = compare({}, base)
    assert comparisons[0].state == ONLY_IN_CANDIDATE
    assert comparisons[0].baseline is None
    assert comparisons[0].candidate.value == 0.5


def test_a_rate_without_person_time_is_a_change_without_a_difference():
    base = Figure(unit=PER_1000_PERSON_YEARS, value=2.16, numerator=1, denominator=365)
    cand = Figure(unit=PER_1000_PERSON_YEARS, value=None, numerator=0, denominator=0)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "incidence"),
        unit=PER_1000_PERSON_YEARS,
        baseline=base,
        candidate=cand,
        difference=None,
        relative_difference=None,
        state=CHANGED,
    )
    assert comparison.state == CHANGED
    assert comparison.difference is None
    assert comparison.relative_difference is None


def test_a_skipped_figure_on_one_side_makes_the_comparison_skipped():
    base = Figure(unit=PROPORTION, state=SKIPPED, reason="no condition records")
    cand = Figure(unit=PROPORTION, value=0.05)
    comparison = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=base,
        candidate=cand,
        difference=None,
        relative_difference=None,
        state=SKIPPED,
        reason="no condition records",
    )
    assert comparison.state == SKIPPED
    assert comparison.reason == "no condition records"
    assert comparison.difference is None
    assert comparison.relative_difference is None


def test_names_with_unicode_and_pipes_keep_distinct_keys():
    keys = [
        ComparisonKey("observations", "H\u2764\ufe0ftension", "n", "mm[Hg]"),
        ComparisonKey("observations", "H|tension", "n", "mm[Hg]"),
        ComparisonKey("observations", "H\u2764\ufe0ftension", "n|p5", "mm[Hg]"),
    ]
    assert len({key for key in keys}) == 3


def test_the_order_is_baseline_first_then_candidate_only():
    base = {
        ComparisonKey("population", "population", "alive", ""): Figure(unit=COUNT, value=1),
        ComparisonKey("population", "population", "total", ""): Figure(unit=COUNT, value=2),
    }
    cand = {
        ComparisonKey("population", "population", "total", ""): Figure(unit=COUNT, value=2),
        ComparisonKey("population", "population", "deceased", ""): Figure(unit=COUNT, value=3),
    }
    keys = [comparison.key for comparison in compare(base, cand)]
    assert [key.measure for key in keys] == ["alive", "total", "deceased"]


def test_largest_changes_rank_by_relative_difference_and_break_ties_by_key():
    high = Comparison(
        key=ComparisonKey("conditions", "MI", "point"),
        unit=PROPORTION,
        baseline=Figure(unit=PROPORTION, value=0.04),
        candidate=Figure(unit=PROPORTION, value=0.06),
        difference=0.02,
        relative_difference=0.5,
        state=CHANGED,
    )
    low = Comparison(
        key=ComparisonKey("conditions", "HTN", "point"),
        unit=PROPORTION,
        baseline=Figure(unit=PROPORTION, value=0.5),
        candidate=Figure(unit=PROPORTION, value=0.55),
        difference=0.05,
        relative_difference=0.1,
        state=CHANGED,
    )
    tie = Comparison(
        key=ComparisonKey("conditions", "MI", "lifetime"),
        unit=PROPORTION,
        baseline=Figure(unit=PROPORTION, value=0.04),
        candidate=Figure(unit=PROPORTION, value=0.06),
        difference=0.02,
        relative_difference=0.5,
        state=CHANGED,
    )
    unknown = Comparison(
        key=ComparisonKey("indexed", "cohort", "patients"),
        unit=COUNT,
        baseline=Figure(unit=COUNT, value=0),
        candidate=Figure(unit=COUNT, value=3),
        difference=3,
        relative_difference=None,
        state=CHANGED,
    )
    ranked = largest_changes([high, low, tie, unknown], 2)
    assert [c.key for c in ranked] == [
        ComparisonKey("conditions", "MI", "lifetime"),
        ComparisonKey("conditions", "MI", "point"),
    ]
    assert len(largest_changes([high, low, tie, unknown], 10)) == 4
    assert largest_changes([high], -1) == []


# --------------------------------------------------------------------------- #
# extraction on a real report built by build_module_validation, block by block
# --------------------------------------------------------------------------- #


def write_dataset(
    directory: Path,
    *,
    observations: bool = True,
    medications: bool = True,
    conditions: bool = True,
) -> Path:
    """A tiny Synthea export with patients, conditions, observations and medications.

    Cada tabla puede omitirse para simular que no se pudo leer.
    """

    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-06-01", "GENDER": "F"},
            {"Id": "d1", "BIRTHDATE": "1940-01-01", "DEATHDATE": "2024-01-01", "GENDER": "M"},
        ],
    )
    write_table(
        directory,
        "encounters",
        [
            {"Id": "e1", "PATIENT": "a1", "CODE": "1505002", "START": "2026-01-01T00:00:00Z"},
        ],
    )
    if conditions:
        write_table(
            directory,
            "conditions",
            [
                {
                    "PATIENT": "a1",
                    "CODE": "22298006",
                    "SYSTEM": SNOMED,
                    "START": "2023-01-01",
                    "STOP": "2023-01-08",
                    "DESCRIPTION": "Myocardial infarction (disorder)",
                },
                {
                    "PATIENT": "a1",
                    "CODE": "401314000",
                    "SYSTEM": SNOMED,
                    "START": "2023-02-01",
                    "STOP": "",
                    "DESCRIPTION": "NSTEMI (disorder)",
                },
                {
                    "PATIENT": "d1",
                    "CODE": "22298006",
                    "SYSTEM": SNOMED,
                    "START": "2022-06-01",
                    "STOP": "2022-06-05",
                    "DESCRIPTION": "Myocardial infarction (disorder)",
                },
                {
                    "PATIENT": "a2",
                    "CODE": "59621000",
                    "SYSTEM": SNOMED,
                    "START": "2015-01-01",
                    "STOP": "",
                    "DESCRIPTION": "Essential hypertension (disorder)",
                },
            ],
        )
    if observations:
        write_table(
            directory,
            "observations",
            [
                {
                    "DATE": "2026-01-01T00:00:00Z",
                    "PATIENT": "a1",
                    "CODE": "8480-6",
                    "VALUE": "150",
                    "UNITS": "mm[Hg]",
                    "TYPE": "numeric",
                },
                {
                    "DATE": "2026-01-01T00:00:00Z",
                    "PATIENT": "a2",
                    "CODE": "8480-6",
                    "VALUE": "120",
                    "UNITS": "mm[Hg]",
                    "TYPE": "numeric",
                },
            ],
        )
    if medications:
        write_table(
            directory,
            "medications",
            [
                {
                    "START": "2025-01-01T00:00:00Z",
                    "STOP": "",
                    "PATIENT": "a1",
                    "PAYER": "",
                    "ENCOUNTER": "",
                    "CODE": "314076",
                    "DESCRIPTION": "lisinopril",
                    "BASE_COST": "",
                    "PAYER_COVERAGE": "",
                    "DISPENSES": "",
                    "TOTALCOST": "",
                    "REASONCODE": "",
                    "REASONDESCRIPTION": "",
                },
            ],
        )
    return directory


def build_report(tmp_path: Path) -> ModuleValidationReport:
    return build_report_in(write_dataset(tmp_path))


def build_report_in(directory: Path) -> ModuleValidationReport:
    """Construye el reporte de validación sobre un directorio ya escrito, sin reescribir sus tablas.

    Se usa (en vez de ``build_report``) cuando se quiere omitir una tabla a propósito:
    ``build_report`` llama de nuevo a ``write_dataset`` y la restauraría.
    """

    return build_module_validation(
        directory,
        module=ModuleInfo("Cardio", ("myocardial_infarction.json",)),
        definitions=assemble(
            ["Myocardial infarction=22298006,401314000;acute", "Hypertension=59621000"],
            None,
            (),
            measures=ALL_MEASURES,
        ),
        module_file="module.json",
        observations=(ObservationDefinition("8480-6"),),
        medications=(MedicationDefinition("Lisinopril", ("314076",)),),
        index=IndexSpec("Myocardial infarction"),
        outcomes=(
            EventOutcome("ICU admission", "encounters", ("1505002",)),
            DeathOutcome("Death"),
        ),
        crosses=(CrossSpec("Death", "ICU admission"),),
        reference_date="2026-01-01",
        generated_at=GENERATED_AT,
    )


def figures_by_key(report: ModuleValidationReport) -> dict[ComparisonKey, Figure]:
    return extract_figures(report)


def test_population_yields_its_metrics_and_distribution_counts(tmp_path):
    figures = figures_by_key(build_report(tmp_path))
    assert figures[ComparisonKey("population", "population", "alive", "")].value == 2
    assert figures[ComparisonKey("population", "population", "total", "")].value == 3
    assert figures[ComparisonKey("population", "population", "deceased", "")].value == 1
    # the distribution describes the alive patients: one man and one woman.
    assert figures[ComparisonKey("population", "distribution.GENDER", "count", "GENDER:F")].value == 1
    assert figures[ComparisonKey("population", "distribution.GENDER", "count", "GENDER:M")].value == 1
    assert figures[ComparisonKey("population", "age", "alive", "")].value == 2
    for key in (
        ComparisonKey("population", "population", "alive", ""),
        ComparisonKey("population", "population", "total", ""),
        ComparisonKey("population", "distribution.GENDER", "count", "GENDER:F"),
    ):
        assert figures[key].unit == COUNT
        assert figures[key].state == "computed"


def test_conditions_yield_prevalence_incidence_in_total_and_by_stratum(tmp_path):
    figures = figures_by_key(build_report(tmp_path))
    # a1 is alive and has an open NSTEMI at the reference date: point = 1/2.
    mi_point = figures[ComparisonKey("conditions", "Myocardial infarction", "point", "")]
    assert mi_point.unit == PROPORTION
    assert mi_point.value == 0.5
    assert (mi_point.numerator, mi_point.denominator) == (1, 2)
    assert mi_point.ci95_low is not None and mi_point.ci95_high is not None
    # a2 has hypertension: point = 1/2.
    assert figures[ComparisonKey("conditions", "Hypertension", "point", "")].value == 0.5
    # incidence carries events and person-days in the per-1,000 person-years unit:
    # a1's first MI in 2023 and d1's MI in 2022 are both events in the window.
    incidence = figures[ComparisonKey("conditions", "Myocardial infarction", "incidence", "")]
    assert incidence.unit == PER_1000_PERSON_YEARS
    assert incidence.numerator == 2
    assert incidence.denominator > 0
    assert incidence.value is not None
    # strata: a1 is the only alive man and he has the condition.
    assert figures[ComparisonKey("conditions", "Myocardial infarction", "point", "GENDER:M")].value == 1.0
    assert figures[ComparisonKey("conditions", "Myocardial infarction", "point", "GENDER:F")].value == 0.0


def test_observations_yield_their_summary_statistics_in_units(tmp_path):
    figures = figures_by_key(build_report(tmp_path))
    base = ComparisonKey("observations", "8480-6", "n", "mm[Hg]")
    assert figures[base].unit == "mm[Hg]"
    assert figures[base].value == 2
    assert figures[base].numerator == 2
    assert figures[ComparisonKey("observations", "8480-6", "median", "mm[Hg]")].value == 135
    assert figures[ComparisonKey("observations", "8480-6", "minimum", "mm[Hg]")].value == 120
    assert figures[ComparisonKey("observations", "8480-6", "maximum", "mm[Hg]")].value == 150
    # n = 2 está por debajo del umbral de percentiles: se emiten con valor None para
    # que se emparejen con el otro lado de la comparación en vez de desaparecer.
    p5 = ComparisonKey("observations", "8480-6", "p5", "mm[Hg]")
    assert p5 in figures
    assert figures[p5].value is None
    assert figures[p5].state == "computed"
    for statistic in ("p25", "p75", "p95"):
        key = ComparisonKey("observations", "8480-6", statistic, "mm[Hg]")
        assert key in figures
        assert figures[key].value is None


def test_medications_yield_active_and_ever(tmp_path):
    figures = figures_by_key(build_report(tmp_path))
    assert figures[ComparisonKey("medications", "Lisinopril", "active", "")].value == 0.5
    assert figures[ComparisonKey("medications", "Lisinopril", "ever", "")].value == 0.5


def test_indexed_yields_cohort_size_outcomes_and_crosses(tmp_path):
    figures = figures_by_key(build_report(tmp_path))
    assert figures[ComparisonKey("indexed", "cohort", "patients", "")].value == 2
    assert figures[ComparisonKey("indexed", "ICU admission", "patients", "total:all")].value == 0.5
    assert figures[ComparisonKey("indexed", "ICU admission", "patients", "sex:M")].value == 0.5
    # empty bands have denominator 0: rate None, never a made-up 0.
    empty = figures[ComparisonKey("indexed", "ICU admission", "patients", "age:0-4")]
    assert empty.value is None
    assert (empty.numerator, empty.denominator) == (0, 0)
    assert figures[ComparisonKey("indexed", "Death", "patients", "total:all")].value == 0.5
    # the cross of deaths inside and outside the ICU-admission stratum.
    with_icu = figures[ComparisonKey("indexed", "Death", "cross", "ICU admission:with ICU admission")]
    assert with_icu.value == 0.0
    assert (with_icu.numerator, with_icu.denominator) == (0, 1)
    without_icu = figures[ComparisonKey("indexed", "Death", "cross", "ICU admission:without ICU admission")]
    assert without_icu.value == 1.0
    assert (without_icu.numerator, without_icu.denominator) == (1, 1)


def test_a_duplicated_key_raises_naming_it(tmp_path):
    report = ModuleValidationReport(
        data_dir="/x",
        module_file="module.json",
        module=ModuleInfo(),
        reference_date=reference(),
        reference_reason="x",
        age_bands=(0,),
        alive=2,
        population=(),
        conditions=(condition("MI"), condition("MI")),
    )
    with pytest.raises(ValueError, match="duplicate comparison key"):
        extract_figures(report)


def test_a_missing_condition_table_skips_the_condition_with_its_reason(tmp_path):
    directory = write_dataset(tmp_path)
    (directory / "conditions.csv").unlink()
    report = build_module_validation(
        directory,
        module=ModuleInfo(),
        definitions=assemble(["MI=22298006"], None, (), measures=ALL_MEASURES),
        module_file="module.json",
        generated_at=GENERATED_AT,
    )
    figures = figures_by_key(report)
    mi_point = figures[ComparisonKey("conditions", "MI", "point", "")]
    assert mi_point.state == SKIPPED
    assert mi_point.reason and "conditions" in mi_point.reason
    assert mi_point.value is None
    mi_incidence = figures[ComparisonKey("conditions", "MI", "incidence", "")]
    assert mi_incidence.state == SKIPPED
    # the population is not affected: its figures are still there.
    assert figures[ComparisonKey("population", "population", "alive", "")].value == 2


# --------------------------------------------------------------------------- #
# R-1 del REVIEW: un elemento SKIPPED en un lado cubre las claves del mismo
# elemento (block + element) del otro lado con su razón, y nunca aparece como
# only_in_*; solo lo que de verdad existe en un solo lado es only_in_*.
# --------------------------------------------------------------------------- #


def states_by_block(comparisons: list[Comparison]) -> dict[str, dict[str, int]]:
    states: dict[str, dict[str, int]] = {}
    for comparison in comparisons:
        states.setdefault(comparison.key.block, {}).setdefault(comparison.state, 0)
        states[comparison.key.block][comparison.state] += 1
    return states


def assert_no_only_in(comparisons: list[Comparison]) -> None:
    for comparison in comparisons:
        assert comparison.state not in (ONLY_IN_BASELINE, ONLY_IN_CANDIDATE), comparison.key


def test_a_skipped_observation_table_is_skipped_with_its_reason_and_covers_all_statistics(tmp_path):
    base = build_report_in(write_dataset(tmp_path / "base"))
    candidate = build_report_in(write_dataset(tmp_path / "candidate", observations=False))
    comparisons = compare(extract_figures(base), extract_figures(candidate))
    assert_no_only_in(comparisons)
    observed = [comparison for comparison in comparisons if comparison.key.block == "observations"]
    assert observed
    assert all(comparison.state == SKIPPED for comparison in observed)
    assert all(comparison.reason and "observation" in comparison.reason for comparison in observed)


def test_a_skipped_observation_table_on_the_baseline_covers_the_candidate(tmp_path):
    base = build_report_in(write_dataset(tmp_path / "base", observations=False))
    candidate = build_report_in(write_dataset(tmp_path / "candidate"))
    comparisons = compare(extract_figures(base), extract_figures(candidate))
    assert_no_only_in(comparisons)
    observed = [comparison for comparison in comparisons if comparison.key.block == "observations"]
    assert observed
    assert all(comparison.state == SKIPPED for comparison in observed)
    assert all(comparison.reason and "observation" in comparison.reason for comparison in observed)


def test_a_skipped_condition_covers_prevalence_and_incidence_with_its_reason(tmp_path):
    base = build_report_in(write_dataset(tmp_path / "base"))
    candidate = build_report_in(write_dataset(tmp_path / "candidate", conditions=False))
    comparisons = compare(extract_figures(base), extract_figures(candidate))
    assert_no_only_in(comparisons)
    conditions = [comparison for comparison in comparisons if comparison.key.block == "conditions"]
    assert conditions
    assert all(comparison.state == SKIPPED for comparison in conditions)
    assert all(comparison.reason and "condition" in comparison.reason for comparison in conditions)


def test_a_skipped_medication_covers_active_and_ever_with_its_reason(tmp_path):
    base = build_report_in(write_dataset(tmp_path / "base"))
    candidate = build_report_in(write_dataset(tmp_path / "candidate", medications=False))
    comparisons = compare(extract_figures(base), extract_figures(candidate))
    assert_no_only_in(comparisons)
    medications = [comparison for comparison in comparisons if comparison.key.block == "medications"]
    assert medications
    assert all(comparison.state == SKIPPED for comparison in medications)
    assert all(comparison.reason and "medication" in comparison.reason for comparison in medications)


def test_a_skipped_indexed_cohort_covers_the_whole_block_in_both_directions(tmp_path):
    full = build_report(tmp_path)
    no_conditions = build_report_in(write_dataset(tmp_path / "no_conditions", conditions=False))
    for baseline, candidate in ((full, no_conditions), (no_conditions, full)):
        comparisons = compare(extract_figures(baseline), extract_figures(candidate))
        assert_no_only_in(comparisons)
        indexed = [comparison for comparison in comparisons if comparison.key.block == "indexed"]
        assert indexed
        assert all(comparison.state == SKIPPED for comparison in indexed)
        assert all(comparison.reason for comparison in indexed)


def _indexed_report(*, indexed: IndexedCohortResult) -> ModuleValidationReport:
    return ModuleValidationReport(
        data_dir="/x",
        module_file="module.json",
        module=ModuleInfo(),
        reference_date=reference(),
        reference_reason="x",
        age_bands=(0,),
        alive=2,
        population=(),
        conditions=(),
        indexed=indexed,
    )


def test_a_skipped_indexed_outcome_and_cross_are_covered_with_its_reason():
    total = Stratum(label="all", patients=2, cohort=2)
    breakdown = Breakdown(total=total, by_sex=(), by_age=())
    days = ValueSummary(n=1, minimum=1096.0, median=1096.0, maximum=1096.0)
    icu_computed = EventResult(
        name="ICU admission",
        table="encounters",
        codes=("1505002",),
        window=Window(from_days=0, to_days=365),
        breakdown=breakdown,
        days=days,
        rows_matched=0,
        rows_date_unusable=0,
        rows_after_reference=0,
        rows_outside_window=0,
    )
    cross_computed = CrossResult(
        outcome="ICU admission",
        by="ICU admission",
        by_type="event",
        strata=(
            CrossStratum(label="with ICU admission", numerator=1, denominator=2, with_cause=None),
        ),
        patients_in_several_strata=0,
    )
    cross_skipped = CrossResult(
        outcome="ICU admission",
        by="ICU admission",
        by_type="event",
        strata=(),
        patients_in_several_strata=0,
        reason="no crossed patients",
    )
    full = _indexed_report(
        indexed=IndexedCohortResult(
            index_condition="Myocardial infarction",
            reference_date="2026-01-01",
            patients=2,
            rows_matched=0,
            rows_unknown_patient=0,
            rows_start_unusable=0,
            rows_after_reference=0,
            outcomes=(icu_computed,),
            crosses=(cross_computed,),
        )
    )
    skipped_outcome = _indexed_report(
        indexed=IndexedCohortResult(
            index_condition="Myocardial infarction",
            reference_date="2026-01-01",
            patients=2,
            rows_matched=0,
            rows_unknown_patient=0,
            rows_start_unusable=0,
            rows_after_reference=0,
            outcomes=(SkippedOutcome("ICU admission", "event", "no encounter records"),),
            crosses=(),
        )
    )
    skipped_cross = _indexed_report(
        indexed=IndexedCohortResult(
            index_condition="Myocardial infarction",
            reference_date="2026-01-01",
            patients=2,
            rows_matched=0,
            rows_unknown_patient=0,
            rows_start_unusable=0,
            rows_after_reference=0,
            outcomes=(icu_computed,),
            crosses=(cross_skipped,),
        )
    )
    for baseline, candidate, reason in (
        (full, skipped_outcome, "no encounter records"),
        (skipped_outcome, full, "no encounter records"),
        (full, skipped_cross, "no crossed patients"),
        (skipped_cross, full, "no crossed patients"),
    ):
        comparisons = compare(extract_figures(baseline), extract_figures(candidate))
        assert_no_only_in(comparisons)
        indexed = [comparison for comparison in comparisons if comparison.key.block == "indexed"]
        assert indexed
        covered = [comparison for comparison in indexed if comparison.state == SKIPPED]
        assert covered
        assert all(comparison.reason == reason for comparison in covered)


def test_a_skipped_incidence_with_a_computed_prevalence_is_a_skipped_incidence():
    computed = ModuleValidationReport(
        data_dir="/x",
        module_file="module.json",
        module=ModuleInfo(),
        reference_date=reference(),
        reference_reason="x",
        age_bands=(0,),
        alive=2,
        population=(),
        conditions=(
            ConditionValidation(
                name="Myocardial infarction",
                codes=(),
                acute=False,
                prevalence=ConditionResult(
                    "Myocardial infarction", (), SectionStatus.COMPUTED,
                    point=Rate(1, 2), lifetime=Rate(1, 2),
                ),
                incidence=ConditionIncidence(
                    "Myocardial infarction", (), SectionStatus.COMPUTED,
                    rate=IncidenceRate(1, 365),
                ),
            ),
        ),
    )
    skipped_incidence = ModuleValidationReport(
        data_dir="/x",
        module_file="module.json",
        module=ModuleInfo(),
        reference_date=reference(),
        reference_reason="x",
        age_bands=(0,),
        alive=2,
        population=(),
        conditions=(
            ConditionValidation(
                name="Myocardial infarction",
                codes=(),
                acute=False,
                prevalence=ConditionResult(
                    "Myocardial infarction", (), SectionStatus.COMPUTED,
                    point=Rate(1, 2), lifetime=Rate(1, 2),
                ),
                incidence=ConditionIncidence(
                    "Myocardial infarction", (), SectionStatus.SKIPPED,
                    reason="no condition records usable for incidence",
                ),
            ),
        ),
    )
    comparisons = compare(extract_figures(computed), extract_figures(skipped_incidence))
    states = [comparison.state for comparison in comparisons]
    assert states == [UNCHANGED, UNCHANGED, SKIPPED]
    assert comparisons[2].key == ComparisonKey("conditions", "Myocardial infarction", "incidence", "")
    assert comparisons[2].reason == "no condition records usable for incidence"
    assert_no_only_in(comparisons)


def test_a_statistic_without_a_value_pairs_instead_of_disappearing():
    key = ComparisonKey("observations", "8480-6", "p5", "mm[Hg]")
    base = {key: Figure(unit="mm[Hg]", value=None)}
    candidate = {key: Figure(unit="mm[Hg]", value=12.5)}
    [comparison] = compare(base, candidate)
    assert comparison.state == CHANGED
    assert comparison.difference is None
    assert comparison.relative_difference is None


def test_a_statistic_without_a_value_on_both_sides_is_unchanged():
    key = ComparisonKey("observations", "8480-6", "p95", "mm[Hg]")
    base = {key: Figure(unit="mm[Hg]", value=None)}
    candidate = {key: Figure(unit="mm[Hg]", value=None)}
    [comparison] = compare(base, candidate)
    assert comparison.state == UNCHANGED
    assert comparison.difference is None
    assert comparison.relative_difference is None
