"""Shared helper to write a Synthea CSV table for tests.

Using the header the contract expects keeps each fixture contract-compatible:
the columns left out end up empty, as they would in a dataset where a module
does not populate them. An explicit ``columns`` subset is for the cases where
a test needs a header the contract does not have (a missing column, an older
layout); otherwise the contract's columns are used.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from synthea_quality.schema.tables import tables_by_name


def write_table(
    directory: Path,
    name: str,
    rows: list[dict[str, str]],
    columns: Sequence[str] | None = None,
) -> Path:
    """Write ``name``.csv in ``directory`` with the contract header (or ``columns``)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    fieldnames = list(columns) if columns is not None else tables_by_name()[name].columns
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
