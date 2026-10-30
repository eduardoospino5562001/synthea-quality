"""[COMPARE] Command line interface: ``synthea-compare``.

One command to compare two Synthea datasets: it runs ``build_module_validation``
once per dataset with the same module and the same parameters, pairs the figures
of the two reports, and writes ``synthea_compare.md`` and ``synthea_compare.json``
into ``--output-dir``.

Exit codes: ``0`` when the report was written, whatever was skipped; ``2`` when it
could not be completed (a usage error, an unreadable directory, a module file that
cannot be loaded, or a report that could not be written). There is no ``1``: the
report states no verdict.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from synthea_quality import __version__
from synthea_quality.compare.build import build
from synthea_quality.compare.compute import compare, extract_figures
from synthea_quality.compare.render import (
    BLOCK_TITLES,
    BLOCKS,
    JSON_NAME,
    MARKDOWN_NAME,
    ComparisonResult,
    summary_counts,
    write_json,
    write_markdown,
)
from synthea_quality.errors import SyntheaQualityError
from synthea_quality.incidence.compute import DEFAULT_WINDOW_YEARS
from synthea_quality.models import utc_now_iso
from synthea_quality.profile.cli import (
    _age_bands,
    _dataset_directory,
    _positive,
    _reference_date,
)
from synthea_quality.profile.population import DEFAULT_AGE_BANDS

EXIT_OK = 0
EXIT_ERROR = 2

_EPILOG = """\
module file (see examples/myocardial_infarction.json):
  {"module": {"name": "...", "synthea_modules": ["..."]},
   "conditions": [{"name": "...", "codes": ["..."], "acute": true}],
   "observations": [...], "medications": [...], "index": {"condition": "..."},
   "outcomes": [...], "by": [...]}

The same module is applied to both datasets. A table that cannot be read on one
side becomes skipped figures with their reasons; nothing is read as zero.

exit codes:
  0  the report was written (skipped parts do not change this)
  2  it could not be completed

The report describes differences; it does not decide whether they are acceptable.

example:
  synthea-compare ./baseline ./candidate --module my_module.json \\
      --reference-date 2026-01-01 --output-dir ./reports
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthea-compare",
        description=(
            "Compare two Synthea CSV datasets against the same module and write "
            "one Markdown and one JSON report of what changed: population, "
            "prevalence, incidence, observations, medications and the indexed "
            "cohort."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "baseline",
        metavar="BASELINE-DIRECTORY",
        type=_dataset_directory,
        help="directory holding the Synthea CSV files of the baseline (read only, never modified)",
    )
    parser.add_argument(
        "candidate",
        metavar="CANDIDATE-DIRECTORY",
        type=_dataset_directory,
        help="directory holding the Synthea CSV files of the candidate (read only, never modified)",
    )
    parser.add_argument(
        "--module",
        metavar="FILE",
        type=Path,
        required=True,
        help="module file: the conditions, observations and medications to compare, applied to both datasets",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIRECTORY",
        type=Path,
        default=Path.cwd(),
        help=f"directory for {MARKDOWN_NAME} and {JSON_NAME} (default: the current directory)",
    )
    source = parser.add_argument_group("reference date (one of)")
    source.add_argument(
        "--reference-date",
        metavar="YYYY-MM-DD",
        type=_reference_date,
        help="measure both datasets at this date instead of resolving it",
    )
    source.add_argument(
        "--metadata-baseline",
        metavar="FILE",
        type=Path,
        help="Synthea run metadata JSON of the baseline (requires --metadata-candidate)",
    )
    source.add_argument(
        "--metadata-candidate",
        metavar="FILE",
        type=Path,
        help="Synthea run metadata JSON of the candidate (requires --metadata-baseline)",
    )
    parser.add_argument(
        "--window-years",
        metavar="N",
        type=_positive,
        default=DEFAULT_WINDOW_YEARS,
        help=f"incidence window before the reference date (default: {DEFAULT_WINDOW_YEARS})",
    )
    parser.add_argument(
        "--age-bands",
        metavar="BOUNDS",
        type=_age_bands,
        default=DEFAULT_AGE_BANDS,
        help="comma-separated lower bounds of the age bands (default: "
        f"{','.join(str(b) for b in DEFAULT_AGE_BANDS)})",
    )
    parser.add_argument(
        "--top",
        metavar="N",
        type=_positive,
        default=20,
        help="how many largest changes the report lists (default: 20)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"synthea-compare (synthea-quality {__version__})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    args = build_parser().parse_args(argv)

    if args.reference_date is not None and (
        args.metadata_baseline is not None or args.metadata_candidate is not None
    ):
        print(
            "error: --reference-date cannot be combined with --metadata-baseline or --metadata-candidate",
            file=sys.stderr,
        )
        return EXIT_ERROR
    if (args.metadata_baseline is None) != (args.metadata_candidate is None):
        print(
            "error: --metadata-baseline and --metadata-candidate must be given together",
            file=sys.stderr,
        )
        return EXIT_ERROR

    try:
        result = build(
            args.baseline,
            args.candidate,
            args.module,
            reference_date=args.reference_date,
            metadata_baseline=args.metadata_baseline,
            metadata_candidate=args.metadata_candidate,
            window_years=args.window_years,
            age_bands=args.age_bands,
        )
        baseline_figures = extract_figures(result.baseline.report)
        candidate_figures = extract_figures(result.candidate.report)
        comparisons = compare(baseline_figures, candidate_figures)
        comparison_result = ComparisonResult(
            module_file=result.module_file,
            module=result.module,
            baseline_dir=result.baseline.data_dir,
            candidate_dir=result.candidate.data_dir,
            baseline_report=result.baseline.report,
            candidate_report=result.candidate.report,
            comparisons=comparisons,
            generated_at=utc_now_iso(),
        )
        markdown_path = write_markdown(
            comparison_result, Path(args.output_dir) / MARKDOWN_NAME, top=args.top
        )
        json_path = write_json(comparison_result, Path(args.output_dir) / JSON_NAME)
    except (SyntheaQualityError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("error: interrupted before the report was written", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - nothing unexpected is silently swallowed
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_ERROR

    print_summary(comparison_result, markdown_path, json_path)
    return EXIT_OK


def run() -> None:
    """Console-script entry point: exit with the code :func:`main` computed."""
    raise SystemExit(main())


def print_summary(
    result: ComparisonResult, markdown_path: Path, json_path: Path
) -> None:
    """Print the short terminal summary: one line per block and the report paths."""

    print(f"synthea-compare (synthea-quality {__version__})")
    print()
    print(f"Baseline:   {result.baseline_dir}")
    print(f"Candidate:  {result.candidate_dir}")
    print(f"Module:     {result.module_file}")
    for block in BLOCKS:
        block_rows = [c for c in result.comparisons if c.key.block == block]
        if not block_rows:
            continue
        print(f"{BLOCK_TITLES[block]}:  {summary_counts(block_rows)}")
    print()
    print("Reports:")
    print(f"  {markdown_path}")
    print(f"  {json_path}")
