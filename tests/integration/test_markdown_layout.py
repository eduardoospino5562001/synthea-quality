"""Layout of every analysis report's Markdown on the official sample.

Two properties a reader relies on and a renderer can break without any test noticing:
every heading is preceded by a blank line (otherwise some Markdown renderers glue it to
the paragraph or list above), and no table repeats its header or separator row.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY / "examples" / "myocardial_infarction.json"
HYPERTENSION = REPOSITORY / "examples" / "hypertension.json"
DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
BIN = Path(sys.executable).parent
MI = "Myocardial infarction=22298006,401303003,401314000;acute"

#: Report name -> (Markdown file, command, module, arguments).
COMMANDS = {
    "profile": ("synthea_profile.md", "synthea-profile", "synthea_quality.profile.cli", ()),
    "prevalence": (
        "synthea_prevalence.md", "synthea-prevalence", "synthea_quality.prevalence.cli",
        ("--condition", MI),
    ),
    "incidence": (
        "synthea_incidence.md", "synthea-incidence", "synthea_quality.incidence.cli",
        ("--condition", MI, "--condition", "Hypertension=59621000"),
    ),
    "validate-mi": (
        "synthea_module_validation.md", "synthea-validate-module",
        "synthea_quality.validate.cli", ("--module", str(EXAMPLE)),
    ),
    "validate-hypertension": (
        "synthea_module_validation.md", "synthea-validate-module",
        "synthea_quality.validate.cli", ("--module", str(HYPERTENSION)),
    ),
    "observations": (
        "synthea_observations.md", "synthea-observations", "synthea_quality.observations.cli",
        ("--module", str(HYPERTENSION)),
    ),
}


@pytest.fixture(scope="module")
def sample() -> Path:
    candidate = Path(os.environ.get("SYNTHEA_QUALITY_SAMPLE", DEFAULT_SAMPLE_DIR))
    if not (candidate / "conditions.csv").is_file():
        pytest.skip(
            "official sample not available; run `python scripts/fetch_official_sample.py` "
            "or set SYNTHEA_QUALITY_SAMPLE to an extracted csv-latest directory"
        )
    return candidate


@pytest.fixture(scope="module", params=sorted(COMMANDS))
def markdown(request, sample, tmp_path_factory) -> list[str]:
    name, command, module, extra = COMMANDS[request.param]
    out = tmp_path_factory.mktemp(request.param)
    script = BIN / command
    base = [str(script)] if script.exists() else [sys.executable, "-m", module]
    completed = subprocess.run(
        [*base, str(sample), *extra, "--output-dir", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return (out / name).read_text(encoding="utf-8").split("\n")


def test_every_heading_follows_a_blank_line(markdown):
    glued = [
        f"line {i + 1}: {line!r} after {markdown[i - 1]!r}"
        for i, line in enumerate(markdown)
        if i > 0 and line.startswith("#") and markdown[i - 1].strip()
    ]
    assert glued == []


def test_no_table_repeats_its_header_or_separator(markdown):
    repeated = []
    for i, line in enumerate(markdown[:-1]):
        separator = markdown[i + 1]
        if line.startswith("|") and separator.startswith("|") and set(
            separator.replace("|", "").replace(":", "").replace(" ", "")
        ) <= {"-"}:
            j = i + 2
            while j < len(markdown) and markdown[j].startswith("|"):
                if markdown[j] in (line, separator):
                    repeated.append(f"line {j + 1} repeats the table header of line {i + 1}")
                j += 1
    assert repeated == []
