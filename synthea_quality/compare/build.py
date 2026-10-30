"""[COMPARE] Build the module validation of each dataset.

The module file is loaded once with :func:`synthea_quality.validate.models.load_module`
and :func:`synthea_quality.validate.build.build_module_validation` is run once per
dataset with the same module and the same parameters. Nothing is recalculated here:
the typed reports are carried as-is to the comparison. A table that cannot be read
on one side becomes SKIPPED figures with their reasons; a module that cannot be
loaded or a directory that cannot be read is an error the CLI turns into exit 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from synthea_quality.incidence.compute import DEFAULT_WINDOW_YEARS
from synthea_quality.profile.population import DEFAULT_AGE_BANDS
from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.models import (
    ModuleFile,
    ModuleInfo,
    ModuleValidationReport,
    load_module,
)


@dataclass(frozen=True, slots=True)
class Side:
    """One side of the comparison.

    ``data_dir`` is the directory that was read and ``report`` is the typed module
    validation of that directory. The reference date and the years of history are
    read from the report itself, so both sides are described the same way.
    """

    data_dir: str
    report: ModuleValidationReport


@dataclass(frozen=True, slots=True)
class BuildResult:
    """The module validation of both datasets, built with the same parameters."""

    module_file: str
    module: ModuleInfo
    baseline: Side
    candidate: Side


def build(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    module_file: str | Path,
    *,
    reference_date: str | None = None,
    metadata_baseline: str | Path | None = None,
    metadata_candidate: str | Path | None = None,
    window_years: int = DEFAULT_WINDOW_YEARS,
    age_bands: Sequence[int] = DEFAULT_AGE_BANDS,
    generated_at: str | None = None,
) -> BuildResult:
    """Build the module validation of each dataset.

    The same module and the same parameters are used on both sides, so every
    figure of one side has a counterpart on the other side unless a source was
    skipped (then it carries its reason, never a made-up zero).

    :raises DefinitionError: the module file cannot be loaded.
    :raises ValueError: both ``reference_date`` and ``metadata`` are given for a
        side, or the window/age bands are invalid.
    :raises EmptyDatasetError: a directory holds none of the known tables.
    :raises ReferenceDateError: the reference date or metadata file cannot be used.
    """

    loaded = load_module(module_file)
    module_file_str = str(module_file)

    baseline_report = _build_side(
        baseline_dir,
        loaded,
        module_file_str,
        reference_date=reference_date,
        metadata=metadata_baseline,
        window_years=window_years,
        age_bands=age_bands,
        generated_at=generated_at,
    )
    candidate_report = _build_side(
        candidate_dir,
        loaded,
        module_file_str,
        reference_date=reference_date,
        metadata=metadata_candidate,
        window_years=window_years,
        age_bands=age_bands,
        generated_at=generated_at,
    )
    return BuildResult(
        module_file=module_file_str,
        module=loaded.module,
        baseline=Side(data_dir=str(baseline_dir), report=baseline_report),
        candidate=Side(data_dir=str(candidate_dir), report=candidate_report),
    )


def _build_side(
    data_dir: str | Path,
    loaded: ModuleFile,
    module_file_str: str,
    *,
    reference_date: str | None,
    metadata: str | Path | None,
    window_years: int,
    age_bands: Sequence[int],
    generated_at: str | None,
) -> ModuleValidationReport:
    """Run the module validation on one dataset with the shared module."""
    return build_module_validation(
        data_dir,
        module=loaded.module,
        definitions=loaded.conditions,
        observations=loaded.observations,
        medications=loaded.medications,
        index=loaded.index,
        outcomes=loaded.outcomes,
        crosses=loaded.crosses,
        module_file=module_file_str,
        reference_date=reference_date,
        metadata=metadata,
        window_years=window_years,
        age_bands=age_bands,
        generated_at=generated_at,
    )
