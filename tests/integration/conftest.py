"""Shared fixture for the official Synthea sample.

The 5 integration files that check ``conditions.csv`` used the same directory
and the same skip message; the others check a different file (patients,
medications, observations) or use a different existence test, so they keep
their own fixture (see the REPORT for the comparison).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"


@pytest.fixture(scope="module")
def sample() -> Path:
    """The extracted official sample, or a skip when it is not there."""
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if not (candidate / "conditions.csv").is_file():
        pytest.skip(
            "official sample not available; run `python scripts/fetch_official_sample.py` "
            "or set SYNTHEA_QUALITY_SAMPLE to an extracted csv-latest directory"
        )
    return candidate
