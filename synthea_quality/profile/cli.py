"""[PROFILE] Command line interface: ``synthea-profile``.

A separate command, so ``synthea-quality`` keeps exactly its behaviour, output and exit
codes. The profile writes two files with fixed names into ``--output-dir``:
``synthea_profile.md`` for a person and ``synthea_profile.json`` for automation.

Exit codes
----------
``0``  the profile was written. Skipped sections do not change it: a dataset without
       ``encounters.csv`` legitimately has no approximate reference date.
``2``  the profile could not be completed: a usage error (including both
       ``--reference-date`` and ``--metadata``), a directory with no known table, an
       unusable reference date or metadata file, a table the profile needs that is
       present but unreadable (the profile is still written, marked incomplete), a
       write failure, or an unexpected error.

There is no ``1``: a profile has no findings, so nothing in it is a failure.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from synthea_quality import __version__
from synthea_quality.errors import SyntheaQualityError
from synthea_quality.profile.build import build_profile
from synthea_quality.profile.codes import DEFAULT_TOP_CODES
from synthea_quality.profile.models import DatasetProfile, InputState, SectionStatus
from synthea_quality.profile.population import DEFAULT_AGE_BANDS, validate_age_bands
from synthea_quality.profile.reference import parse_reference_date
from synthea_quality.profile.render import (
    JSON_NAME,
    MARKDOWN_NAME,
    write_json,
    write_markdown,
)

EXIT_OK = 0
EXIT_ERROR = 2

_EPILOG = """\
reference date (the "end of the simulation" ages are measured at), in this order:
  --reference-date YYYY-MM-DD   a date you choose
  --metadata FILE               endTime from a Synthea output/metadata/*.json file
  (neither)                     the latest encounters.csv START/STOP, reported as an
                                approximation: the CSV export does not record the end
                                of the simulation

exit codes:
  0  the profile was written (skipped sections do not change this)
  2  the profile could not be completed: a usage error, no known table in the
     directory, an unusable reference date or metadata file, an unreadable table the
     profile needs (the profile is still written and marked incomplete), or a write
     failure

The profile describes the dataset; it states no verdict and never modifies the data.

example:
  synthea-profile ./output/csv --output-dir ./reports
"""


def _dataset_directory(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"dataset directory does not exist: {path}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path


def _reference_date(value: str) -> str:
    try:
        parse_reference_date(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _age_bands(value: str) -> tuple[int, ...]:
    try:
        bounds = tuple(int(part) for part in value.split(","))
        return validate_age_bands(bounds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{exc}; give increasing lower bounds starting at 0, e.g. 0,5,18,45,65"
        ) from exc


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed so tests and docs describe one interface)."""
    parser = argparse.ArgumentParser(
        prog="synthea-profile",
        description=(
            "Describe a Synthea CSV dataset — population, age and demographics of the "
            "patients alive at the end of the simulation, and the most common codes of "
            "each clinical table among them — and write a Markdown and a JSON profile."
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--reference-date",
        metavar="YYYY-MM-DD",
        type=_reference_date,
        help="measure ages at this date instead of resolving it",
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
        help=(
            "comma-separated lower bounds of the age bands (default: "
            f"{','.join(str(b) for b in DEFAULT_AGE_BANDS)})"
        ),
    )
    parser.add_argument(
        "--top-counties",
        metavar="N",
        type=_positive,
        default=None,
        help="how many counties to list (default: 10)",
    )
    parser.add_argument(
        "--top",
        metavar="N",
        type=_positive,
        default=DEFAULT_TOP_CODES,
        help=f"how many codes to list per clinical table (default: {DEFAULT_TOP_CODES})",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"synthea-profile (synthea-quality {__version__})",
        help="show the tool version and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)  # exits 2 on a usage error, 0 for --help/--version

    try:
        profile = build_profile(
            args.dataset,
            reference_date=args.reference_date,
            metadata=args.metadata,
            age_bands=args.age_bands,
            top_counties=args.top_counties,
            top_codes=args.top,
        )
        output_dir = Path(args.output_dir)
        markdown_path = write_markdown(profile, output_dir / MARKDOWN_NAME)
        json_path = write_json(profile, output_dir / JSON_NAME)
    except (SyntheaQualityError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("error: interrupted before the profile was written", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - nothing unexpected is silently swallowed
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return EXIT_ERROR

    print_summary(profile, markdown_path, json_path)
    return EXIT_ERROR if profile.incomplete else EXIT_OK


def run() -> None:
    """Console-script entry point: exit with the code :func:`main` computed."""
    raise SystemExit(main())


def print_summary(profile: DatasetProfile, markdown_path: Path, json_path: Path) -> None:
    """Print the short terminal summary: scope, reference date, population, report paths."""
    print(f"synthea-profile (synthea-quality {profile.tool_version})")
    print()
    print(f"Dataset:    {profile.data_dir}")
    reference = profile.reference_date
    if reference is None:
        print(f"Reference:  none — {profile.reference_reason}")
    else:
        kind = "APPROXIMATION" if reference.approximate else "exact"
        print(f"Reference:  {reference.value} ({kind}, source: {reference.source.value})")
    population = profile.section("population")
    if population.status is SectionStatus.COMPUTED:
        m = population.metrics
        print(f"Patients:   {m['total']} total - {m['alive']} alive, {m['deceased']} deceased")
    else:
        print(f"Patients:   not profiled — {population.reason}")
    skipped = [s.section_id for s in profile.sections if s.status is SectionStatus.SKIPPED]
    print(
        f"Sections:   {len(profile.sections) - len(skipped)} computed, {len(skipped)} skipped"
        + (f" ({', '.join(skipped)})" if skipped else "")
    )
    for note in profile.notes:
        print(f"Note:       {note}")
    for item in profile.inputs:
        if item.state is InputState.UNREADABLE:
            print(f"Incomplete: {item.table} could not be read — {item.reason}")
    print()
    print("Profile:")
    print(f"  {markdown_path}")
    print(f"  {json_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    run()
