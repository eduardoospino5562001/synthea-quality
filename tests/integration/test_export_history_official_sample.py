"""The time-filtered dataset notice over the official Synthea sample.

With a metadata file carrying ``exporter.years_of_history = 10`` every analysis
report shows the notice right after its title, with the cut-off computed as the
reference date minus 365 x 10 days; with ``0`` there is no notice; and the numbers
are unchanged by the notice (the JSON without the provenance keys is identical to
the run without metadata, whose approximate reference date is the same day).

The dataset is fetched with ``python scripts/fetch_official_sample.py`` or pointed at
with ``SYNTHEA_QUALITY_SAMPLE``; without it these tests skip.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

DEFAULT_SAMPLE_DIR = Path.home() / "synthea-sample-data" / "csv-latest"
EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"

REFERENCE_ISO = "2026-08-17"
CUTOFF_ISO = (date.fromisoformat(REFERENCE_ISO) - timedelta(days=365 * 10)).isoformat()

COMMANDS = ("profile", "prevalence", "incidence", "observations", "validate")
MARKDOWN = {
    "profile": "synthea_profile.md",
    "prevalence": "synthea_prevalence.md",
    "incidence": "synthea_incidence.md",
    "observations": "synthea_observations.md",
    "validate": "synthea_module_validation.json",
}
JSON = {
    "profile": "synthea_profile.json",
    "prevalence": "synthea_prevalence.json",
    "incidence": "synthea_incidence.json",
    "observations": "synthea_observations.json",
    "validate": "synthea_module_validation.json",
}

#: Keys whose values describe provenance rather than the dataset's numbers.
VOLATILE = {
    "generated_at",
    "reference_date",
    "reference_reason",
    "notes",
    "inputs",
    "history",
    "export_history",
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


def run(command: str, sample: Path, out: Path, metadata: Path | None) -> int:
    args = [str(sample), "--output-dir", str(out)]
    if metadata is not None:
        args += ["--metadata", str(metadata)]
    if command == "profile":
        from synthea_quality.profile.cli import main
    elif command == "prevalence":
        from synthea_quality.prevalence.cli import main
    elif command == "incidence":
        from synthea_quality.incidence.cli import main
        args += ["--condition", "MI=22298006,401303003,401314000"]
    elif command == "observations":
        from synthea_quality.observations.cli import main
        args += ["--observation", "8480-6"]
    elif command == "validate":
        from synthea_quality.validate.cli import main
        args += ["--module", str(EXAMPLES / "myocardial_infarction.json")]
    return main(args)


def write_metadata(directory: Path, years) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    path.write_text(
        json.dumps({"endTime": "20260817", "exporter.years_of_history": years}),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def reports(sample, tmp_path_factory):
    made = {}
    for command in COMMANDS:
        for label, years in (("plain", None), ("ten", "10"), ("zero", "0")):
            out = tmp_path_factory.mktemp(f"{command}-{label}")
            meta = None if years is None else write_metadata(out / "meta", years)
            code = run(command, sample, out / "reports", meta)
            assert code == 0, (command, label)
            made[(command, label)] = out / "reports"
    return made


def read_json(reports, command: str, label: str) -> dict:
    return json.loads((reports[(command, label)] / JSON[command]).read_text(encoding="utf-8"))


def read_markdown(reports, command: str, label: str) -> str:
    name = MARKDOWN[command] if command != "validate" else "synthea_module_validation.md"
    return (reports[(command, label)] / name).read_text(encoding="utf-8")


def test_the_notice_is_first_after_the_title_in_every_report(reports):
    for command in COMMANDS:
        text = read_markdown(reports, command, "ten")
        assert text.split("\n\n")[2].startswith("> **Time-filtered dataset.**"), command
        assert "`exporter.years_of_history = 10`" in text, command
        assert CUTOFF_ISO in text, command
        data = read_json(reports, command, "ten")
        assert data["export_history"]["status"] == "read", command
        assert data["export_history"]["years_of_history"] == 10, command
        assert data["export_history"]["cutoff"] == CUTOFF_ISO, command


def test_no_notice_with_a_whole_history(reports):
    for command in COMMANDS:
        text = read_markdown(reports, command, "zero")
        assert "Time-filtered dataset" not in text, command
        assert "Exported history unknown" not in text, command
        assert read_json(reports, command, "zero")["export_history"]["years_of_history"] == 0


def test_the_numbers_do_not_change_with_the_notice(reports):
    for command in COMMANDS:
        with_notice = read_json(reports, command, "ten")
        without = read_json(reports, command, "plain")
        assert without["reference_date"]["value"] == with_notice["reference_date"]["value"]
        assert {
            key: value for key, value in with_notice.items() if key not in VOLATILE
        } == {
            key: value for key, value in without.items() if key not in VOLATILE
        }, command
