"""[VALIDATE] Command line interface: ``synthea-validate-module``.

One command to check a Synthea module: it reads a module file (the conditions JSON of
``synthea-prevalence``, with an optional ``module`` block and expected ``point``,
``lifetime`` and ``incidence`` values) and writes ``synthea_module_validation.md`` and
``synthea_module_validation.json`` into ``--output-dir``.

Exit codes: ``0`` when the report was written, whatever was skipped; ``2`` when it could
not be completed (a usage error, an unusable module file, reference date or metadata
file, no known table, an unreadable table — the report is still written and marked
incomplete — or a write failure). There is no ``1``: the report states no verdict.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Sequence

from synthea_quality import __version__
from synthea_quality.errors import SyntheaQualityError
from synthea_quality.incidence.compute import DEFAULT_WINDOW_YEARS
from synthea_quality.profile.cli import (
    _age_bands,
    _dataset_directory,
    _positive,
    _reference_date,
)
from synthea_quality.profile.models import InputState, SectionStatus
from synthea_quality.profile.population import DEFAULT_AGE_BANDS
from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.models import ModuleValidationReport, load_module
from synthea_quality.validate.render import (
    JSON_NAME,
    MARKDOWN_NAME,
    write_json,
    write_markdown,
)

EXIT_OK = 0
EXIT_ERROR = 2

_EPILOG = f"""\
module file (see examples/myocardial_infarction.json):
  {{"module": {{"name": "...", "synthea_modules": ["..."]}},
   "conditions": [{{"name": "...", "codes": ["..."], "acute": true,
                   "expected": {{"lifetime": 0.03, "incidence": 2.5, "source": "..."}}}}],
   "observations": [{{"name": "...", "code": "8480-6", "cohort": "...",
                     "reference_range": {{"low": 100, "high": 139, "units": "mm[Hg]"}}}}],
   "medications": [{{"name": "...", "codes": ["314076"], "cohort": "...",
                    "expected": {{"active": 0.8, "source": "..."}}}}]}}
  point and lifetime are proportions; incidence is per 1,000 person-years;
  observations and medications are optional (see examples/hypertension.json);
  a medication's active and ever expected values are proportions

populations:
  prevalence counts the patients alive at the end of the simulation; incidence follows
  every patient, deceased included, until their death, over the last --window-years
  (default {DEFAULT_WINDOW_YEARS}) before the reference date

exit codes:
  0  the report was written (skipped parts do not change this)
  2  it could not be completed

The report describes; a reference value is shown inside or outside the observed 95% CI.

example:
  synthea-validate-module ./output/csv --module my_module.json \\
      --metadata ./output/metadata/<run>.json --output-dir ./reports
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthea-validate-module",
        description=(
            "Validate a Synthea module against a generated population: population summary, "
            "prevalence and incidence of the module's conditions, the values of its "
            "observations, the share of a cohort with each medication, and reference values "
            "next to the observed ones, in one Markdown and one JSON report."
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
        "--module",
        metavar="FILE",
        type=Path,
        required=True,
        help="module file: the conditions, observations and medications to validate, with "
        "reference values",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIRECTORY",
        type=Path,
        default=Path.cwd(),
        help=f"directory for {MARKDOWN_NAME} and {JSON_NAME} (default: the current directory)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--reference-date",
        metavar="YYYY-MM-DD",
        type=_reference_date,
        help="measure at this date instead of resolving it",
    )
    source.add_argument(
        "--metadata",
        metavar="FILE",
        type=Path,
        help="Synthea run metadata JSON: endTime and exporter.years_of_history",
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
        "--version",
        action="version",
        version=f"synthea-validate-module (synthea-quality {__version__})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        loaded = load_module(args.module)
        report = build_module_validation(
            args.dataset,
            module=loaded.module,
            definitions=loaded.conditions,
            observations=loaded.observations,
            medications=loaded.medications,
            module_file=args.module,
            reference_date=args.reference_date,
            metadata=args.metadata,
            window_years=args.window_years,
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


def print_summary(report: ModuleValidationReport, markdown_path: Path, json_path: Path) -> None:
    print(f"synthea-validate-module (synthea-quality {report.tool_version})")
    print()
    print(f"Dataset:    {report.data_dir}")
    print(f"Module:     {report.module.name or report.module_file}")
    reference = report.reference_date
    if reference is None:
        print(f"Reference:  none — {report.reference_reason}")
    else:
        kind = "APPROXIMATION" if reference.approximate else "exact"
        print(f"Reference:  {reference.value} ({kind}, source: {reference.source.value})")
    print(f"Alive:      {report.alive if report.alive is not None else '—'}")
    for condition in report.conditions:
        prevalence, incidence = condition.prevalence, condition.incidence
        parts = []
        if prevalence.status is SectionStatus.COMPUTED:
            parts.append(
                f"lifetime {prevalence.lifetime.numerator}/{prevalence.lifetime.denominator}"
            )
        if incidence.status is SectionStatus.COMPUTED:
            parts.append(f"{incidence.rate.events} new case(s)")
        placed = sum(1 for r in condition.references if r.position is not None)
        if condition.references:
            parts.append(f"{placed} reference value(s) placed against the 95% CI")
        print(f"Condition:  {condition.name}: {', '.join(parts) or 'not computed'}")
    for item in report.observations:
        if item.status is SectionStatus.SKIPPED:
            print(f"Observation: {item.name}: not computed — {item.reason}")
            continue
        values = "; ".join(
            f"{g.summary.n} patient(s) in {g.units or 'no units'}" for g in item.groups
        )
        print(f"Observation: {item.name}: {values}")
    for item in report.medications:
        if item.status is SectionStatus.SKIPPED:
            print(f"Medication: {item.name}: not computed — {item.reason}")
            continue
        print(
            f"Medication: {item.name}: active {item.active.numerator}/{item.active.denominator}, "
            f"ever {item.ever.numerator}/{item.ever.denominator}"
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
