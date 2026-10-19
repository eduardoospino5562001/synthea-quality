"""[CHECKS] Render one cell of a check sample.

The key, quality and temporal checks keep a bounded sample of the rows behind a
finding, and a report writes every sampled value as text. A null cell has to stay
empty: a naive ``str`` would write ``NaN`` and a reader could take it for a value
the dataset contains. Only a null is turned into ``None``; every other value,
including ``0`` and the empty string, is written as it stands.

The function used to be copied in the quality and temporal checks. It lives here
so both describe a sample the same way, and so the null rule has a single place.
"""

from __future__ import annotations

import pandas as pd


def sample_text(value: object) -> str | None:
    """Render a cell for a sample without turning a null into the text ``nan``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
