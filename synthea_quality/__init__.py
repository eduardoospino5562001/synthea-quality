"""synthea_quality - data quality and integrity checks for Synthea CSV datasets.

Scope of the current development phase (Phase 1): structural data quality and
integrity only. Statistical validation (prevalence, incidence, disease rates)
is explicitly out of scope for now.

This module intentionally imports nothing but the standard library, so that
``import synthea_quality`` stays cheap. Heavier dependencies (pandas) are
imported by the modules that need them.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.9.0"
