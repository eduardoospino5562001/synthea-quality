"""[CLI] Command line entry point.

Intended use::

    synthea-quality <dataset-directory> [--output-dir <directory>]

The command runs the whole pipeline over a dataset directory and writes two reports
in the output directory, using predictable file names.

Exit codes (stable, so a script can rely on them)
-------------------------------------------------

``0``
    The run completed and no check failed. Warnings do not change this: a warning
    flags legitimate Synthea data that deserves a look, not a data defect.
``1``
    The run completed but at least one check **failed**: a deterministic violation
    of a confirmed rule, in the data.
``2``
    The tool could not complete the run: the input directory holds no table the tool
    knows how to check, a table could not be read at all (so the analysis is
    incomplete), a report could not be written, or a check ended in ``ERROR``, which
    by definition means the tool itself could not do its job.

This is not a statistical gate. Every check is deterministic; nothing here measures
prevalence, incidence or any distribution, and a deviation of a rate is not a
failure. The dataset is only read: the tool never writes inside it.

Unexpected failures are reported with their traceback on stderr rather than
swallowed, because a crash is a bug worth seeing.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from synthea_quality import __version__
from synthea_quality.errors import SyntheaQualityError
from synthea_quality.models import DatasetReport, Status
from synthea_quality.reporting import json_report
from synthea_quality.reporting.build import build_report
from synthea_quality.reporting.markdown import write_markdown

#: The run completed and no check failed.
EXIT_OK = 0
#: The run completed but at least one check failed.
EXIT_FINDINGS = 1
#: The tool could not complete the run.
EXIT_ERROR = 2

#: Predictable report file names inside the output directory.
MARKDOWN_NAME = "synthea_quality_report.md"
JSON_NAME = "synthea_quality_report.json"

#: How many failing checks the terminal summary lists before summarising the rest.
MAX_LISTED_FINDINGS = 10
#: How many missing table names the terminal summary lists before counting the rest.
MAX_LISTED_TABLES = 6

_EPILOG = """\
exit codes:
  0  the run completed and no check failed
  1  the run completed but at least one check failed (a data defect)
  2  the tool could not complete the run (no known table in the input directory, an
     unreadable table, a write failure, or a check errored)

Warnings never change the exit code: they flag legitimate Synthea data worth a look.
Only deterministic checks run here; no prevalence, incidence or statistical rule is
applied, and the dataset is never modified.

example:
  synthea-quality ./output/csv --output-dir ./reports
"""


def _dataset_directory(value: str) -> Path:
    """Validate the positional argument before anything else happens."""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"dataset directory does not exist: {path}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed so tests and docs describe one interface)."""
    parser = argparse.ArgumentParser(
        prog="synthea-quality",
        description=(
            "Check a Synthea CSV dataset against the confirmed rules and write a "
            "Markdown and a JSON report."
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
        help=(
            f"directory for {MARKDOWN_NAME} and {JSON_NAME}, created if missing "
            "(default: the current directory)"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"synthea-quality {__version__}",
        help="show the tool version and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)  # exits 2 on a usage error, 0 for --help/--version

    try:
        report = build_report(args.dataset)
        markdown_path = write_markdown(report, Path(args.output_dir) / MARKDOWN_NAME)
        json_path = json_report.dump(report, Path(args.output_dir) / JSON_NAME)
    except (SyntheaQualityError, OSError) as exc:
        # Expected and actionable: a plain message, no traceback.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("error: interrupted before the run finished", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - nothing unexpected is silently swallowed
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_ERROR

    print_summary(report, markdown_path, json_path)
    return exit_code_for(report)


def run() -> None:
    """Console-script entry point: exit with the code :func:`main` computed."""
    raise SystemExit(main())


def exit_code_for(report: DatasetReport) -> int:
    """Map a finished report to an exit code (see the module docstring).

    An unreadable table makes the run incomplete, so it is an error even when every
    table that could be read passed: returning ``0`` there would present an analysis
    with a hole in it as a clean dataset.
    """
    counts = report.counts_by_status
    if counts[Status.ERROR.value]:
        return EXIT_ERROR
    if report.load_errors:
        return EXIT_ERROR
    if counts[Status.FAIL.value]:
        return EXIT_FINDINGS
    return EXIT_OK


def print_summary(report: DatasetReport, markdown_path: Path, json_path: Path) -> None:
    """Print the short terminal summary: scope, verdict, counts and the report paths."""
    counts = report.counts_by_status
    missing = report.missing_known_tables

    print(f"synthea-quality {report.tool_version}")
    print()
    print(f"Dataset:  {report.data_dir}")
    missing_note = ""
    if missing:
        shown = ", ".join(missing[:MAX_LISTED_TABLES])
        more = f" and {len(missing) - MAX_LISTED_TABLES} more" if len(missing) > MAX_LISTED_TABLES else ""
        missing_note = f" (missing: {shown}{more})"
    print(
        f"Tables:   {len(report.tables)} of {report.contract_tables} described by the contract"
        f"{missing_note}"
    )
    contract = report.contract_status.value if report.contract_status else "UNKNOWN"
    print(f"Contract: {contract}")
    if report.contract_summary:
        print(f"          {report.contract_summary}")
    print(
        "Checks:   {total} total - {PASS} PASS, {WARNING} WARNING, {FAIL} FAIL, "
        "{NOT_APPLICABLE} NOT_APPLICABLE, {SKIPPED} SKIPPED, {ERROR} ERROR".format(
            total=len(report.checks), **counts
        )
    )
    if report.unknown_files:
        print(f"Unknown:  {', '.join(report.unknown_files)}")
    if report.load_errors:
        lost = ", ".join(error.table for error in report.load_errors[:MAX_LISTED_TABLES])
        more = (
            f" and {len(report.load_errors) - MAX_LISTED_TABLES} more"
            if len(report.load_errors) > MAX_LISTED_TABLES
            else ""
        )
        print(f"Unreadable: {lost}{more} (the analysis of those tables is incomplete)")
    for entry in report.anomalous_entries[:MAX_LISTED_TABLES]:
        print(f"Ignored:  {entry.reason}")

    failing = [
        check for check in report.findings if check.status in (Status.FAIL, Status.ERROR)
    ]
    if failing:
        print()
        print(f"Failures ({len(failing)}):")
        for check in failing[:MAX_LISTED_FINDINGS]:
            print(f"  {check.status.value} {check.check_id} ({check.severity.value})")
            print(f"      {check.message}")
        if len(failing) > MAX_LISTED_FINDINGS:
            print(f"  ... and {len(failing) - MAX_LISTED_FINDINGS} more, see the report")

    if counts[Status.WARNING.value]:
        print()
        print(
            f"Warnings: {counts[Status.WARNING.value]} (legitimate Synthea data worth a look; "
            "the report lists them with their metrics)"
        )

    print()
    print("Reports:")
    print(f"  {markdown_path}")
    print(f"  {json_path}")
