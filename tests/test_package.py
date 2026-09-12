"""Skeleton tests for Phase 1, slice 1.

These tests verify packaging facts rather than behaviour, because slice 1 only
creates the project structure. Behaviour tests start with the result models.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import synthea_quality

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_declared() -> None:
    assert synthea_quality.__version__


def test_version_matches_pyproject() -> None:
    """Keep the package constant and the project metadata in sync."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]

    assert synthea_quality.__version__ == project["version"]


def test_importing_the_package_does_not_import_pandas() -> None:
    """Importing the package root must stay lightweight.

    pandas is only needed once a dataset is actually loaded, so the package
    root must not pull it in at import time.
    """
    code = "import sys, synthea_quality; print('pandas' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False", result.stderr
