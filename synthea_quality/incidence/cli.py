"""[INCIDENCE] Command line interface: ``synthea-incidence``.

A separate command: the other three keep their behaviour. It writes
``synthea_incidence.md`` and ``synthea_incidence.json`` into ``--output-dir`` and needs at
least one condition (``--condition`` or ``--conditions``).

Exit codes: ``0`` when the report was written, whatever was skipped; ``2`` when it could
not be completed (a usage error, an unusable definition, reference date or metadata file,
no known table, an unreadable table — the report is still written and marked incomplete —
or a write failure). There is no ``1``: an incidence report has no findings.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from synthea_quality import __version__
from synthea_quality.errors import SyntheaQualityError
from synthea_quality.incidence.build import build_incidence
from synthea_quality.incidence.compute import DEFAULT_WINDOW_YEARS
from synthea_quality.incidence.models import IncidenceReport
from synthea_quality.incidence.render import (
    JSON_NAME,
    MARKDOWN_NAME,
    write_json,
    write_markdown,
)
from synthea_quality.prevalence.definitions import assemble
from synthea_quality.prevalence.models import INCIDENCE
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
conditions (the same definitions as synthea-prevalence):
  --condition "Myocardial infarction=22298006,401303003,401314000;acute"   (repeatable)
  --conditions module_conditions.json
  --expected "Myocardial infarction:incidence=2.5"   (per 1,000 person-years, repeatable)

population:
  every patient, deceased included until their death (default); --alive-only keeps only
  the patients alive at the end of the simulation, which biases the rate down for
  conditions that shorten life (survivor bias)

window:
  the last --window-years years before the reference date (default {DEFAULT_WINDOW_YEARS});
  use --metadata to know how many years Synthea exported (exporter.years_of_history)

exit codes:
  0  the report was written (skipped parts do not change this)
  2  it could not be completed

The report describes; a reference value is shown inside or outside the observed 95% CI.

example:
  synthea-incidence ./output/csv --condition "Hypertension=59621000" --output-dir ./reports
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthea-incidence",
        description=(
            "New cases per 1,000 person-years of conditions in a Synthea CSV dataset, in total "
            "and by age band and sex, with exact 95% Poisson intervals."
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
        help="a condition to measure: NAME=CODE[,CODE...][;acute] (repeatable)",
    )
    parser.add_argument(
        "--conditions", metavar="FILE", type=Path, help="JSON file of conditions to measure"
    )
    parser.add_argument(
        "--expected",
        metavar="NAME:incidence=VALUE",
        action="append",
        default=[],
        help="a reference incidence per 1,000 person-years to show (repeatable)",
    )
    parser.add_argument(
        "--window-years",
        metavar="N",
        type=_positive,
        default=DEFAULT_WINDOW_YEARS,
        help=f"length of the observation window before the reference date "
        f"(default: {DEFAULT_WINDOW_YEARS})",
    )
    parser.add_argument(
        "--alive-only",
        action="store_true",
        help="follow only the patients alive at the end of the simulation (survivor bias)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--reference-date",
        metavar="YYYY-MM-DD",
        type=_reference_date,
        help="end the window at this date instead of resolving it",
    )
    source.add_argument(
        "--metadata",
        metavar="FILE",
        type=Path,
        help="Synthea run metadata JSON: endTime and exporter.years_of_history",
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
        version=f"synthea-incidence (synthea-quality {__version__})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.condition and args.conditions is None:
        parser.error("give at least one condition with --condition or --conditions")
    try:
        definitions = assemble(
            args.condition, args.conditions, args.expected, measures=(INCIDENCE,)
        )
        report = build_incidence(
            args.dataset,
            definitions=definitions,
            reference_date=args.reference_date,
            metadata=args.metadata,
            window_years=args.window_years,
            alive_only=args.alive_only,
            age_bands=args.age_bands,
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


def print_summary(report: IncidenceReport, markdown_path: Path, json_path: Path) -> None:
    print(f"synthea-incidence (synthea-quality {report.tool_version})")
    print()
    print(f"Dataset:    {report.data_dir}")
    reference = report.reference_date
    if reference is None:
        print(f"Reference:  none — {report.reference_reason}")
    else:
        kind = "APPROXIMATION" if reference.approximate else "exact"
        print(f"Reference:  {reference.value} ({kind}, source: {reference.source.value})")
    if report.window is not None:
        print(f"Window:     {report.window.start} to {report.window.end}")
    print(f"Population: {report.population}, followed {report.cohort.get('followed', '—')}")
    for condition in report.conditions:
        if condition.status is SectionStatus.SKIPPED:
            print(f"Condition:  {condition.name}: not computed — {condition.reason}")
            continue
        rate = condition.rate
        value = "—" if rate.value is None else f"{rate.value:.2f}"
        print(
            f"Condition:  {condition.name}: {rate.events} event(s) in "
            f"{rate.person_years:.1f} person-years, {value} per 1,000"
        )
    for item in report.inputs:
        if item.state is InputState.UNREADABLE:
            print(f"Incomplete: {item.table} could not be read — {item.reason}")
    print()
    print("Report:")
    print(f"  {markdown_path}")
    print(f"  {json_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    run()
