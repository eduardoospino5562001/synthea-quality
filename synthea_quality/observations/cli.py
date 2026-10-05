"""[OBSERVATIONS] Command line interface: ``synthea-observations``.

Writes ``synthea_observations.md`` and ``synthea_observations.json`` into ``--output-dir``:
the observations asked for (``--observation``, or the ``observations`` of a ``--module``
file) and a general table of every numeric code among the alive patients.

Exit codes
----------
``0``  the report was written, whatever was skipped.
``2``  it could not be completed: a usage error, an unusable definition, module file,
       reference date or metadata file, a directory with no known table, a table it needs
       that is present but unreadable (the report is still written, marked incomplete), a
       write failure, or an unexpected error.

There is no ``1``: the report states no verdict.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from synthea_quality import __version__
from synthea_quality.errors import SyntheaQualityError
from synthea_quality.observations.build import build_observations
from synthea_quality.observations.compute import DEFAULT_TOP
from synthea_quality.observations.definitions import (
    load_module_observations,
    parse_observation_option,
    unique,
)
from synthea_quality.observations.models import ObservationsReport
from synthea_quality.observations.render import (
    JSON_NAME,
    MARKDOWN_NAME,
    write_json,
    write_markdown,
)
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

_EPILOG = """\
observations:
  --observation 8480-6                        (repeatable; or "NAME=CODE")
  --module module.json                        its "observations", with optional cohort,
                                              reference_range and lookback_years (see the
                                              README and examples/hypertension.json)

values:
  each alive patient contributes their latest value on or before the reference date, per
  code and units; units are never converted; rows that cannot be used are counted

general table:
  every numeric code and unit among the alive, by patients with a value, and the codes
  written in several units or with several TYPE values

exit codes:
  0  the report was written (skipped parts do not change this)
  2  it could not be completed

The report describes; a reference range is shown next to the observed values, never as a
verdict.

example:
  synthea-observations ./output/csv --observation 8480-6 --output-dir ./reports
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthea-observations",
        description=(
            "Distribution of numeric observation values among the patients alive at the end "
            "of the simulation: one latest value per patient, percentiles per unit, in total "
            "and by age band and sex."
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
        "--observation",
        metavar="CODE",
        action="append",
        default=[],
        help="an observation code to describe, or NAME=CODE (repeatable)",
    )
    parser.add_argument(
        "--module",
        metavar="FILE",
        type=Path,
        help="module file whose 'observations' (and the 'conditions' their cohorts name) to use",
    )
    parser.add_argument(
        "--lookback-years",
        metavar="N",
        type=_positive,
        help="leave out latest values older than N years before the reference date "
        "(an observation's own lookback_years wins)",
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
        help="describe the values at this date instead of resolving it",
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
        version=f"synthea-observations (synthea-quality {__version__})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        conditions, observations = (
            load_module_observations(args.module) if args.module is not None else ((), ())
        )
        observations = unique(
            (*observations, *(parse_observation_option(t) for t in args.observation))
        )
        report = build_observations(
            args.dataset,
            observations=observations,
            conditions=conditions,
            module_file=args.module,
            reference_date=args.reference_date,
            metadata=args.metadata,
            age_bands=args.age_bands,
            lookback_years=args.lookback_years,
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


def print_summary(report: ObservationsReport, markdown_path: Path, json_path: Path) -> None:
    print(f"synthea-observations (synthea-quality {report.tool_version})")
    print()
    print(f"Dataset:     {report.data_dir}")
    reference = report.reference_date
    if reference is None:
        print(f"Reference:   none — {report.reference_reason}")
    else:
        kind = "APPROXIMATION" if reference.approximate else "exact"
        print(f"Reference:   {reference.value} ({kind}, source: {reference.source.value})")
    print(f"Alive:       {report.alive if report.alive is not None else '—'}")
    for item in report.observations:
        if item.status is SectionStatus.SKIPPED:
            print(f"Observation: {item.name}: not computed — {item.reason}")
            continue
        parts = [
            f"{g.summary.n} patient(s), median {g.summary.median:g} {g.units or ''}".rstrip()
            if g.summary.n
            else f"no value in {g.units or 'no units'}"
            for g in item.groups
        ]
        print(f"Observation: {item.name}: {'; '.join(parts)}")
    general = report.general
    if general.status is SectionStatus.COMPUTED:
        print(
            f"General:     {len(general.rows)} code and unit pairs; "
            f"{len(general.mixed_units)} code(s) in several units, "
            f"{len(general.mixed_types)} with several TYPE"
        )
    else:
        print(f"General:     not computed — {general.reason}")
    for item in report.inputs:
        if item.state is InputState.UNREADABLE:
            print(f"Incomplete:  {item.table} could not be read — {item.reason}")
    print()
    print("Report:")
    print(f"  {markdown_path}")
    print(f"  {json_path}")


if __name__ == "__main__":  # pragma: no cover - exercised through a subprocess
    run()
