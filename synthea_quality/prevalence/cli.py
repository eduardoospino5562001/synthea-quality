"""[PREVALENCE] Command line interface: ``synthea-prevalence``.

A separate command, like ``synthea-profile``: ``synthea-quality`` keeps its behaviour and
exit codes. It writes ``synthea_prevalence.md`` and ``synthea_prevalence.json`` into
``--output-dir``.

Exit codes
----------
``0``  the report was written, whatever was skipped.
``2``  it could not be completed: a usage error (including both ``--reference-date`` and
       ``--metadata``), an unusable condition definition, reference date or metadata
       file, a directory with no known table, a table it needs that is present but
       unreadable (the report is still written, marked incomplete), a write failure, or
       an unexpected error.

There is no ``1``: a prevalence report has no findings.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from synthea_quality import __version__
from synthea_quality.errors import SyntheaQualityError
from synthea_quality.prevalence.build import build_prevalence
from synthea_quality.prevalence.compute import DEFAULT_TOP
from synthea_quality.prevalence.definitions import assemble
from synthea_quality.prevalence.models import PrevalenceReport
from synthea_quality.prevalence.render import (
    JSON_NAME,
    MARKDOWN_NAME,
    write_json,
    write_markdown,
)
from synthea_quality.prevalence.social import SOCIAL_LIST_ID
from synthea_quality.profile.cli import (
    _age_bands,
    _dataset_directory,
    _positive,
    _reference_date,
)
from synthea_quality.profile.models import InputState, SectionStatus
from synthea_quality.profile.population import DEFAULT_AGE_BANDS

EXIT_OK = 0
EXIT_ERROR = 2

_EPILOG = f"""\
conditions:
  --condition "Myocardial infarction=22298006,401303003,401314000"   (repeatable)
  --conditions module_conditions.json                               (see the README)
  --expected "Myocardial infarction:lifetime=0.03"                  (repeatable)
  Codes are SNOMED CT unless written SYSTEM|CODE. A patient counts once per condition.

general table:
  every condition code among the alive, by point prevalence. The social and
  administrative codes of list {SOCIAL_LIST_ID} (SDoH screening and medication review)
  are excluded unless --include-social is given.

exit codes:
  0  the report was written (skipped parts do not change this)
  2  it could not be completed: a usage error, an unusable definition, reference date or
     metadata file, no known table, an unreadable table (the report is still written and
     marked incomplete), or a write failure

The report describes; a reference value is shown inside or outside the observed 95% CI,
never as a verdict. Incidence is not computed.

example:
  synthea-prevalence ./output/csv --condition "Hypertension=59621000" --output-dir ./reports
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthea-prevalence",
        description=(
            "Point and lifetime prevalence of conditions among the patients alive at the end "
            "of the simulation, in total and by age band and sex, with 95% Wilson intervals."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        metavar="DATASET-DIRECTORY",
        type=_dataset_directory,
        help="directory holding the Synthea CSV files (read only, never modified)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIRECTORY",
        type=Path,
        default=Path.cwd(),
        help=f"directory for {MARKDOWN_NAME} and {JSON_NAME} (default: the current directory)",
    )
    parser.add_argument(
        "--condition",
        metavar="NAME=CODES",
        action="append",
        default=[],
        help="a condition to measure: a name and its comma-separated codes (repeatable)",
    )
    parser.add_argument(
        "--conditions",
        metavar="FILE",
        type=Path,
        help="JSON file of conditions to measure, with optional expected values",
    )
    parser.add_argument(
        "--expected",
        metavar="NAME:MEASURE=VALUE",
        action="append",
        default=[],
        help="a reference prevalence (point or lifetime, as a proportion) to show (repeatable)",
    )
    parser.add_argument(
        "--include-social",
        action="store_true",
        help="keep the social and administrative codes in the general table",
    )
    parser.add_argument(
        "--top",
        metavar="N",
        type=_positive,
        default=DEFAULT_TOP,
        help=f"rows of the general table listed in the Markdown (default: {DEFAULT_TOP})",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--reference-date",
        metavar="YYYY-MM-DD",
        type=_reference_date,
        help="measure prevalence at this date instead of resolving it",
    )
    source.add_argument(
        "--metadata",
        metavar="FILE",
        type=Path,
        help="Synthea run metadata JSON whose endTime is the end of the simulation",
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
        "--version",
        action="version",
        version=f"synthea-prevalence (synthea-quality {__version__})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        definitions = assemble(args.condition, args.conditions, args.expected)
        report = build_prevalence(
            args.dataset,
            definitions=definitions,
            reference_date=args.reference_date,
            metadata=args.metadata,
            age_bands=args.age_bands,
            include_social=args.include_social,
            top=args.top,
        )
        markdown_path = write_markdown(report, Path(args.output_dir) / MARKDOWN_NAME)
        json_path = write_json(report, Path(args.output_dir) / JSON_NAME)
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

    print_summary(report, markdown_path, json_path)
    return EXIT_ERROR if report.incomplete else EXIT_OK


def run() -> None:
    raise SystemExit(main())


def print_summary(report: PrevalenceReport, markdown_path: Path, json_path: Path) -> None:
    print(f"synthea-prevalence (synthea-quality {report.tool_version})")
    print()
    print(f"Dataset:    {report.data_dir}")
    reference = report.reference_date
    if reference is None:
        print(f"Reference:  none — {report.reference_reason}")
    else:
        kind = "APPROXIMATION" if reference.approximate else "exact"
        print(f"Reference:  {reference.value} ({kind}, source: {reference.source.value})")
    print(f"Alive:      {report.alive if report.alive is not None else '—'}")
    for condition in report.conditions:
        if condition.status is SectionStatus.SKIPPED:
            print(f"Condition:  {condition.name}: not computed — {condition.reason}")
            continue
        point, lifetime = condition.point, condition.lifetime
        print(
            f"Condition:  {condition.name}: point {point.numerator}/{point.denominator}, "
            f"lifetime {lifetime.numerator}/{lifetime.denominator}"
        )
    general = report.general
    if general.status is SectionStatus.COMPUTED:
        social = "included" if general.include_social else "excluded"
        print(f"General:    {len(general.rows)} codes (social codes {social})")
    else:
        print(f"General:    not computed — {general.reason}")
    for item in report.inputs:
        if item.state is InputState.UNREADABLE:
            print(f"Incomplete: {item.table} could not be read — {item.reason}")
    print()
    print("Report:")
    print(f"  {markdown_path}")
    print(f"  {json_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    run()
