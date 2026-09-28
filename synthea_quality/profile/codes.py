"""[PROFILE] The most common codes of each clinical table, among the alive patients.

One section per table — conditions, medications, procedures, observations,
immunizations, allergies, careplans — each listing the top N codes by the number of
**distinct alive patients** with at least one record of the code.

What the numbers mean
---------------------
A patient counts for a code when they are alive at the end of the simulation
(:func:`~synthea_quality.profile.population.alive_patient_ids`) and have **at least one
record of that code, ever**. That is a historical count: a condition resolved twenty
years ago counts, and nothing here says the code is active at the reference date.
Telling those apart is prevalence, a later step. Every section repeats this, because
reading "patients with diabetes" as prevalence would be the natural mistake.

Choices a reader should know about
----------------------------------
* **Distinct patients, not rows.** A patient with 300 blood-pressure observations counts
  once. ``records`` is shown next to it for scale.
* **A code is ``(SYSTEM, CODE)``** when the table has a ``SYSTEM`` column
  (conditions, procedures, allergies), and ``CODE`` alone otherwise. ``allergies``
  mixes SNOMED-CT and RxNorm, and merging equal numbers from two systems would invent an
  equivalence.
* **One description per code, never hiding the others.** The description shown is the
  most frequent one for the code over the *whole table*, since it describes the code and
  not the cohort (ties: alphabetical). Codes written with more than one description are
  counted, and up to :data:`MULTI_DESCRIPTION_LIMIT` of them are listed with every
  variant and its number of records.
* **Deterministic order:** distinct patients, then records (both highest first), then
  system and code. When the top-N cut falls inside a group with the same number of
  patients, the section says so (``tie_at_cut``).
* **Nothing is dropped silently.** Rows of deceased patients, rows whose ``PATIENT`` is
  not in ``patients.csv``, and alive rows without a ``CODE`` are counted, then left out.

This module computes; it reads no file and renders nothing.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from synthea_quality.profile.models import (
    CodeCount,
    CodeDescriptions,
    DescriptionCount,
    ProfileSection,
    SectionStatus,
    percent,
)
from synthea_quality.profile.ranking import tie_at_cut

#: The clinical tables profiled, in report order, with their section titles.
CLINICAL_TABLES: tuple[tuple[str, str], ...] = (
    ("conditions", "Conditions"),
    ("medications", "Medications"),
    ("procedures", "Procedures"),
    ("observations", "Observations"),
    ("immunizations", "Immunizations"),
    ("allergies", "Allergies"),
    ("careplans", "Care plans"),
)

#: Columns a table must have to be profiled, and columns used when present.
REQUIRED_COLUMNS = ("PATIENT", "CODE")
OPTIONAL_COLUMNS = ("SYSTEM", "DESCRIPTION")

DEFAULT_TOP_CODES = 20
#: How many codes with several descriptions are listed per table (all are counted).
MULTI_DESCRIPTION_LIMIT = 10

HISTORICAL_NOTE = (
    "Patients are alive patients with at least one record of the code ever "
    "(historical), not patients in whom it is active at the reference date; that "
    "distinction belongs to prevalence."
)


def section_id(table: str) -> str:
    return f"codes.{table}"


def section_title(title: str) -> str:
    return f"{title}: most common codes"


def columns_to_load(header: Sequence[str]) -> list[str] | None:
    """The columns to read from a table with ``header``, or ``None`` if it cannot be profiled."""
    if any(column not in header for column in REQUIRED_COLUMNS):
        return None
    return [*REQUIRED_COLUMNS, *(c for c in OPTIONAL_COLUMNS if c in header)]


def skipped_code_sections(reason: str) -> tuple[ProfileSection, ...]:
    """Every clinical section, skipped for the same reason (e.g. alive patients unknown)."""
    return tuple(
        ProfileSection.skipped(section_id(table), section_title(title), reason)
        for table, title in CLINICAL_TABLES
    )


def skipped_code_section(table: str, reason: str) -> ProfileSection:
    title = dict(CLINICAL_TABLES)[table]
    return ProfileSection.skipped(section_id(table), section_title(title), reason)


def profile_codes(
    table: str,
    frame: pd.DataFrame,
    *,
    alive_ids: pd.Index,
    patient_ids: pd.Index,
    top: int = DEFAULT_TOP_CODES,
) -> ProfileSection:
    """The top-``top`` codes of ``table`` among the alive patients.

    :param frame: the table's ``PATIENT`` and ``CODE`` columns, plus ``SYSTEM`` and
        ``DESCRIPTION`` when the table has them, as text.
    :param alive_ids: identifiers of the alive patients.
    :param patient_ids: identifiers of every patient, to tell a deceased patient's row
        from a row whose ``PATIENT`` is unknown.
    """
    if top < 1:
        raise ValueError("top must be at least 1")
    title = dict(CLINICAL_TABLES)[table]
    has_system = "SYSTEM" in frame.columns
    has_description = "DESCRIPTION" in frame.columns
    keys = ["SYSTEM", "CODE"] if has_system else ["CODE"]
    denominator = len(alive_ids)

    alive_rows = frame["PATIENT"].isin(alive_ids)
    known_rows = frame["PATIENT"].isin(patient_ids)
    coded = frame["CODE"].notna()

    # One working layout for every table: an empty or absent SYSTEM becomes "" so that
    # grouping and joining never have to match a null key. The loader never produces ""
    # (an empty field is a null), so "" maps back to None without ambiguity.
    work = pd.DataFrame(
        {
            "PATIENT": frame["PATIENT"],
            "SYSTEM": frame["SYSTEM"].fillna("") if has_system else "",
            "CODE": frame["CODE"],
            "DESCRIPTION": (
                frame["DESCRIPTION"] if has_description else pd.Series(pd.NA, index=frame.index)
            ),
        }
    )

    # Descriptions are a property of the code, so they are counted over the whole table.
    variants = _description_variants(work[coded])
    per_key = variants.drop_duplicates(["SYSTEM", "CODE"], keep="first")[
        ["SYSTEM", "CODE", "DESCRIPTION"]
    ].merge(
        variants.groupby(["SYSTEM", "CODE"]).size().rename("variants").reset_index(),
        on=["SYSTEM", "CODE"],
    )

    cohort = work[alive_rows & coded]
    per_code = (
        cohort.groupby(["SYSTEM", "CODE"])
        .agg(patients=("PATIENT", "nunique"), records=("PATIENT", "size"))
        .reset_index()
        .merge(per_key, on=["SYSTEM", "CODE"], how="left")
        .sort_values(
            ["patients", "records", "SYSTEM", "CODE"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
    )
    shown = per_code.head(top)
    rest = per_code.iloc[len(shown):]

    rows = tuple(
        CodeCount(
            system=row.SYSTEM or None,
            code=str(row.CODE),
            description=_text(row.DESCRIPTION),
            patients=int(row.patients),
            percent=percent(int(row.patients), denominator),
            records=int(row.records),
            description_variants=int(row.variants),
        )
        for row in shown.itertuples(index=False)
    )
    ambiguous = _multi_description_codes(variants, per_key[per_key["variants"] > 1])

    metrics: dict[str, Any] = {
        "denominator": denominator,
        "rows": len(frame),
        "rows_alive": int(alive_rows.sum()),
        "rows_deceased": int((known_rows & ~alive_rows).sum()),
        "rows_unknown_patient": int((~known_rows).sum()),
        "rows_without_code": int((alive_rows & ~coded).sum()),
        "distinct_codes": len(per_key),
        "distinct_codes_alive": len(per_code),
        "patients_with_records": int(cohort["PATIENT"].nunique()),
        "shown": len(shown),
        "top": top,
        "codes_with_multiple_descriptions": len(ambiguous),
        "code_identity": "+".join(keys),
    }

    notes = [
        HISTORICAL_NOTE,
        "Ordered by distinct alive patients, then records, then "
        f"{'system and code' if has_system else 'code'}; percentages are shares of the "
        "alive patients.",
        f"A code is identified by {' + '.join(keys)}. The description shown is the most "
        "frequent one for the code over the whole table (ties: alphabetical).",
    ]
    if not has_description:
        notes.append(f"{table}.csv has no DESCRIPTION column; codes are shown without one.")
    tie = tie_at_cut(
        [int(p) for p in shown["patients"]], [int(p) for p in rest["patients"]]
    )
    if tie is not None:
        metrics["tie_at_cut"] = tie
        notes.append(
            f"{tie['values']} codes tie at {tie['count']} patients; ties are broken by "
            f"records, then by code ({tie['listed']} listed, "
            f"{tie['values'] - tie['listed']} not listed)."
        )
    excluded = []
    if metrics["rows_deceased"]:
        excluded.append(f"{metrics['rows_deceased']} row(s) of deceased patients")
    if metrics["rows_unknown_patient"]:
        excluded.append(
            f"{metrics['rows_unknown_patient']} row(s) whose PATIENT is not in patients.csv"
        )
    if metrics["rows_without_code"]:
        excluded.append(f"{metrics['rows_without_code']} alive row(s) without a CODE")
    if excluded:
        notes.append(f"Left out: {'; '.join(excluded)}.")
    if len(ambiguous) > MULTI_DESCRIPTION_LIMIT:
        notes.append(
            f"{len(ambiguous)} codes have more than one description; the first "
            f"{MULTI_DESCRIPTION_LIMIT} (by system and code) are listed."
        )

    return ProfileSection(
        section_id=section_id(table),
        title=section_title(title),
        status=SectionStatus.COMPUTED,
        metrics=metrics,
        notes=tuple(notes),
        codes=rows,
        multi_description_codes=tuple(ambiguous[:MULTI_DESCRIPTION_LIMIT]),
    )


def _description_variants(work: pd.DataFrame) -> pd.DataFrame:
    """Records per (system, code, description), most frequent first within each code.

    Within a code, equal counts are ordered alphabetically and an empty description
    comes last, so the first row of each code is the description the profile shows.
    """
    counted = (
        work.groupby(["SYSTEM", "CODE", "DESCRIPTION"], dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )
    counted["_empty"] = counted["DESCRIPTION"].isna()
    counted["_description"] = counted["DESCRIPTION"].fillna("")
    return counted.sort_values(
        ["SYSTEM", "CODE", "records", "_empty", "_description"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _multi_description_codes(
    variants: pd.DataFrame, ambiguous: pd.DataFrame
) -> list[CodeDescriptions]:
    """Every code with more than one description, ordered by system and code."""
    if ambiguous.empty:
        return []
    selected = variants.merge(ambiguous[["SYSTEM", "CODE"]], on=["SYSTEM", "CODE"])
    return [
        CodeDescriptions(
            system=system or None,
            code=str(code),
            descriptions=tuple(
                DescriptionCount(_text(row.DESCRIPTION), int(row.records))
                for row in group.itertuples(index=False)
            ),
        )
        for (system, code), group in selected.groupby(["SYSTEM", "CODE"], sort=True)
    ]


def _text(value: Any) -> str | None:
    """A loaded text value, or ``None`` for a null."""
    return None if pd.isna(value) else str(value)
