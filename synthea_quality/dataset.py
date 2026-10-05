"""[DATASET] The steps every analysis report takes before it computes anything.

``synthea-prevalence``, ``synthea-incidence`` and ``synthea-validate-module`` all open a
dataset the same way, and this module is that one way:

1. discover the tables (a directory with none of the known tables is an input error);
2. validate the row structure of the tables the report reads, before any is loaded
   (only tables whose header can be read; the others fail again, with the loader's
   message, when loaded);
3. resolve the reference date, with its provenance;
4. load each table with only the columns asked for, recording what happened to it as a
   :class:`~synthea_quality.profile.models.TableInput` — read, absent (the table or a
   required column), or unreadable.

The loader caches what it reads, so two computations asking for the same columns of the
same table share one read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from synthea_quality.discovery import DiscoveryResult, discover_dataset
from synthea_quality.errors import EmptyDatasetError, TableLoadError
from synthea_quality.loader import DatasetLoader
from synthea_quality.profile.models import InputState, TableInput
from synthea_quality.profile.reference import (
    ReferenceResolution,
    parse_reference_date,
    resolve_reference_date,
)
from synthea_quality.schema.tables import SYNTHEA_TABLES
from synthea_quality.structure import StructureReport, gate_reason, validate_tables


@dataclass(frozen=True, slots=True, eq=False)
class DatasetContext:
    """A discovered, structurally validated dataset with its reference date."""

    data_dir: Path
    discovery: DiscoveryResult
    loader: DatasetLoader
    structure: dict[str, StructureReport]
    resolution: ReferenceResolution

    @property
    def reference(self) -> date | None:
        """The reference date as a date, or ``None`` when it could not be resolved."""
        if self.resolution.reference is None:
            return None
        return parse_reference_date(self.resolution.reference.value)

    def load(
        self,
        name: str,
        wanted: Sequence[str],
        required: Sequence[str],
        used_for: str,
    ) -> tuple[pd.DataFrame | None, TableInput]:
        """The ``wanted`` columns of table ``name`` that exist, or ``None`` and why not."""
        table = next((t for t in self.discovery.tables if t.name == name), None)
        if table is None:
            reason = f"{name}.csv is not in the dataset"
            return None, TableInput(name, used_for, InputState.ABSENT, reason=reason)
        gated = gate_reason(self.structure, name)
        if gated is not None:
            return None, TableInput(name, used_for, InputState.UNREADABLE, reason=gated)
        try:
            header = self.loader.read_header(table.path)
            missing = [c for c in required if c not in header]
            if missing:
                reason = f"{name}.csv has no {', '.join(missing)} column(s)"
                return None, TableInput(name, used_for, InputState.ABSENT, reason=reason)
            loaded = self.loader.load(
                table.path, table=name, columns=[c for c in wanted if c in header]
            )
        except TableLoadError as exc:
            reason = f"{name}.csv could not be read: {exc}"
            return None, TableInput(name, used_for, InputState.UNREADABLE, reason=reason)
        return loaded.frame, TableInput(name, used_for, InputState.READ, rows=loaded.rows)


def open_dataset(
    data_dir: str | Path,
    *,
    tables: Sequence[str],
    reference_date: str | None = None,
    metadata: str | Path | None = None,
    discovery: DiscoveryResult | None = None,
) -> DatasetContext:
    """Discover ``data_dir``, validate ``tables`` and resolve the reference date.

    :raises EmptyDatasetError: the directory holds none of the known tables.
    :raises ValueError: both ``reference_date`` and ``metadata`` were given.
    :raises ReferenceDateError: the explicit date or metadata file cannot be used.
    """
    data_path = Path(data_dir)
    found = discovery if discovery is not None else discover_dataset(data_path)
    if not found.tables:
        raise EmptyDatasetError(
            f"no Synthea CSV table was found in {data_path}: a dataset directory has to "
            f"hold at least one of the {len(SYNTHEA_TABLES)} tables the schema contract "
            f"describes (for example patients.csv or conditions.csv)"
        )
    loader = DatasetLoader()
    readable: dict[str, Path] = {}
    for table in found.tables:
        if table.name in tables:
            try:
                loader.read_header(table.path)
            except TableLoadError:
                continue
            readable[table.name] = table.path
    structure = validate_tables(readable)
    resolution = resolve_reference_date(
        found, loader, user_date=reference_date, metadata_path=metadata, structure=structure
    )
    return DatasetContext(data_path, found, loader, structure, resolution)
