"""[PREVALENCE] Build the prevalence report of one dataset directory.

The only module of the package that touches the file system. It opens the dataset with
:func:`synthea_quality.dataset.open_dataset`, the steps every analysis report shares:
discovery, structural validation of ``patients``, ``encounters`` and ``conditions``
before any of them is loaded, the reference date, and loading only the columns needed.

Nothing is guessed. Every rate needs the alive patients and the reference date, so
without either every result is ``SKIPPED`` with the reason; without ``conditions.csv``
(or its ``PATIENT``, ``CODE``, ``START`` or ``STOP`` column) every result is ``SKIPPED``
too. A table that is present but unreadable marks the report incomplete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from synthea_quality.dataset import open_dataset
from synthea_quality.discovery import DiscoveryResult
from synthea_quality.prevalence.compute import (
    CONDITION_COLUMNS,
    DEFAULT_TOP,
    PATIENT_COLUMNS,
    REQUIRED_CONDITION_COLUMNS,
    build_cohort,
    condition_prevalence,
    general_table,
    prepare_records,
)
from synthea_quality.prevalence.definitions import ConditionDefinition
from synthea_quality.prevalence.exclusion import ExclusionList, codes_absent_from_data, read_exclusion_file
from synthea_quality.prevalence.models import (
    ConditionResult,
    GeneralTable,
    PrevalenceReport,
)
from synthea_quality.prevalence.social import describe_list
from synthea_quality.profile.models import SectionStatus
from synthea_quality.profile.population import DEFAULT_AGE_BANDS, validate_age_bands

PREVALENCE_TABLES = ("patients", "encounters", "conditions")
PATIENTS_USED_FOR = "the alive cohort, ages and sex"
CONDITIONS_USED_FOR = "condition records"


def build_prevalence(
    data_dir: str | Path,
    *,
    definitions: Sequence[ConditionDefinition] = (),
    reference_date: str | None = None,
    metadata: str | Path | None = None,
    age_bands: Sequence[int] = DEFAULT_AGE_BANDS,
    include_social: bool = False,
    top: int = DEFAULT_TOP,
    exclude_codes: str | Path | None = None,
    discovery: DiscoveryResult | None = None,
    generated_at: str | None = None,
) -> PrevalenceReport:
    """Compute the prevalence of ``definitions`` and the general table for ``data_dir``.

    With ``exclude_codes`` the codes of that file are left out of the general table
    instead of the built-in social and administrative list.

    :raises ValueError: invalid age bands or ``top``, or both ``reference_date`` and
        ``metadata``, or both ``include_social`` and ``exclude_codes``.
    :raises EmptyDatasetError: the directory holds none of the known tables.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    :raises ExclusionListError: the exclusion file cannot be used as given.
    """
    bands = validate_age_bands(age_bands)
    if top < 1:
        raise ValueError("top must be at least 1")
    if include_social and exclude_codes is not None:
        raise ValueError("give either --include-social or --exclude-codes, not both")
    exclusion = read_exclusion_file(exclude_codes) if exclude_codes is not None else None
    context = open_dataset(
        data_dir,
        tables=PREVALENCE_TABLES,
        reference_date=reference_date,
        metadata=metadata,
        discovery=discovery,
    )
    resolution = context.resolution
    patients, patients_input = context.load(
        "patients", PATIENT_COLUMNS, ("Id", "DEATHDATE"), PATIENTS_USED_FOR
    )
    conditions, conditions_input = context.load(
        "conditions", CONDITION_COLUMNS, REQUIRED_CONDITION_COLUMNS, CONDITIONS_USED_FOR
    )
    inputs = [patients_input]
    if resolution.encounters is not None:
        inputs.append(resolution.encounters)
    inputs.append(conditions_input)

    if exclusion is None:
        social_list: dict = describe_list()
    else:
        described = exclusion.describe()
        if conditions is None:
            # No table to compare with: every listed code is absent from the data.
            described["listed_absent_from_data"] = list(exclusion.codes)
        else:
            present = {str(code) for code in conditions["CODE"].dropna().unique()}
            described["listed_absent_from_data"] = list(
                codes_absent_from_data(exclusion, present)
            )
        social_list = described

    common = dict(
        data_dir=str(context.data_dir),
        reference_date=resolution.reference,
        reference_reason=resolution.reason,
        age_bands=bands,
        social_list=social_list,
        inputs=tuple(inputs),
        **({"generated_at": generated_at} if generated_at is not None else {}),
    )

    reason = None
    if resolution.reference is None:
        reason = resolution.reason or "no reference date"
    elif patients is None:
        reason = f"the alive patients are unknown: {patients_input.reason}"
    elif conditions is None:
        reason = f"no condition records: {conditions_input.reason}"
    if reason is not None:
        return PrevalenceReport(
            alive=None,
            conditions=tuple(
                ConditionResult(
                    d.name, d.codes, SectionStatus.SKIPPED, reason=reason, acute=d.acute
                )
                for d in definitions
            ),
            general=GeneralTable(
                SectionStatus.SKIPPED, reason=reason, top=top, include_social=include_social
            ),
            notes=resolution.notes,
            **common,
        )

    reference = context.reference
    assert reference is not None
    cohort = build_cohort(patients, reference, bands)
    records = prepare_records(conditions, cohort, reference)
    if exclusion is not None:
        # The absent list is recomputed over the whole table (any patient, any date),
        # the scope the general table uses for its own note; the early return above
        # only covers a missing conditions table.
        present = {str(code) for _, code in records.table_codes}
        social_list = {
            **exclusion.describe(),
            "listed_absent_from_data": list(codes_absent_from_data(exclusion, present)),
        }
    return PrevalenceReport(
        alive=cohort.size,
        conditions=tuple(condition_prevalence(d, records, cohort) for d in definitions),
        general=general_table(
            records, cohort, include_social=include_social, top=top, exclude=exclusion
        ),
        notes=(*resolution.notes, *cohort.notes),
        **{**common, "social_list": social_list},
    )

