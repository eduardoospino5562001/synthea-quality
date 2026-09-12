"""Schema contracts for Synthea CSV datasets.

This package is the single place where facts about the Synthea CSV schema live:
first the table catalogue (:mod:`synthea_quality.schema.tables`), later the
per-column contracts and their versioning. Keeping them here prevents schema
knowledge from spreading across the check modules.
"""

from __future__ import annotations

__all__ = ["tables"]
