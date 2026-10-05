"""Tests for the time-filtered dataset notice at the top of every report."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from synthea_quality.incidence.build import build_incidence
from synthea_quality.observations.build import build_observations
from synthea_quality.observations.definitions import parse_observation_option
from synthea_quality.prevalence.build import build_prevalence
from synthea_quality.prevalence.definitions import assemble
from synthea_quality.profile.build import build_profile
from synthea_quality.schema.tables import tables_by_name
from synthea_quality.validate.build import build_module_validation
from synthea_quality.validate.models import load_module

GENERATED_AT = "2026-10-02T00:00:00+00:00"
SNOMED = "http://snomed.info/sct"
CUTOFF = (date(2026, 1, 1).toordinal() - 365 * 10)
CUTOFF_ISO = date.fromordinal(CUTOFF).isoformat()


def write_table(directory: Path, name: str, rows: list[dict[str, str]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tables_by_name()[name].columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def dataset(directory: Path) -> Path:
    write_table(
        directory,
        "patients",
        [
            {"Id": "a1", "BIRTHDATE": "1950-01-01", "GENDER": "M"},
            {"Id": "a2", "BIRTHDATE": "1990-01-01", "GENDER": "F"},
        ],
    )
    write_table(directory, "encounters", [{"Id": "e1", "START": "2026-01-01T00:00:00Z"}])
    write_table(
        directory,
        "conditions",
        [
            {"PATIENT": "a1", "CODE": "59621000", "SYSTEM": SNOMED, "START": "2010-01-01",
             "DESCRIPTION": "Essential hypertension (disorder)"},
        ],
    )
    write_table(
        directory,
        "observations",
        [
            {"PATIENT": "a1", "CODE": "8480-6", "DATE": "2025-06-01T00:00:00Z",
             "VALUE": "120", "UNITS": "mm[Hg]", "TYPE": "numeric",
             "DESCRIPTION": "Systolic blood pressure"},
        ],
    )
    return directory


def metadata(directory: Path, years) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    path.write_text(
        json.dumps({"endTime": "20260101", "exporter.years_of_history": years}),
        encoding="utf-8",
    )
    return path


def module_file(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "module.json"
    path.write_text(
        json.dumps(
            {
                "module": {"name": "Hypertension"},
                "conditions": [{"name": "Hypertension", "codes": ["59621000"]}],
            }
        ),
        encoding="utf-8",
    )
    return path


def build_all(data: Path, meta, generated_at: str = GENERATED_AT) -> dict:
    loaded = load_module(module_file(data.parent / "mod"))
    return {
        "profile": build_profile(data, metadata=meta, generated_at=generated_at),
        "prevalence": build_prevalence(data, metadata=meta, generated_at=generated_at),
        "incidence": build_incidence(
            data,
            definitions=assemble(["H=59621000"], None, [], measures=("incidence",)),
            metadata=meta,
            generated_at=generated_at,
        ),
        "observations": build_observations(
            data,
            observations=[parse_observation_option("8480-6")],
            metadata=meta,
            generated_at=generated_at,
        ),
        "validate": build_module_validation(
            data,
            module=loaded.module,
            definitions=loaded.conditions,
            module_file=str(data.parent / "mod" / "module.json"),
            metadata=meta,
            generated_at=generated_at,
        ),
    }


def first_after_title(markdown: str) -> str:
    return markdown.split("\n\n")[2]


def test_the_notice_is_first_after_the_title_in_every_report(tmp_path):
    from synthea_quality.incidence.render import render_markdown as incidence_md
    from synthea_quality.observations.render import render_markdown as observations_md
    from synthea_quality.prevalence.render import render_markdown as prevalence_md
    from synthea_quality.profile.render import render_markdown as profile_md
    from synthea_quality.validate.render import render_markdown as validate_md

    reports = build_all(dataset(tmp_path / "csv"), metadata(tmp_path / "meta", "10"))
    rendered = {
        "profile": profile_md(reports["profile"]),
        "prevalence": prevalence_md(reports["prevalence"]),
        "incidence": incidence_md(reports["incidence"]),
        "observations": observations_md(reports["observations"]),
        "validate": validate_md(reports["validate"]),
    }
    for name, text in rendered.items():
        assert first_after_title(text).startswith("> **Time-filtered dataset.**"), name
        assert "`exporter.years_of_history = 10`" in text, name
        assert CUTOFF_ISO in text, name
    for name, report in reports.items():
        assert report.to_dict()["export_history"]["status"] == "read", name
        assert report.to_dict()["export_history"]["cutoff"] == CUTOFF_ISO, name


def test_no_notice_with_a_whole_history_or_without_metadata(tmp_path):
    from synthea_quality.prevalence.render import render_markdown as prevalence_md

    data = dataset(tmp_path / "csv")
    for meta in (metadata(tmp_path / "meta", "0"), None):
        reports = build_all(data, meta)
        for name, report in reports.items():
            dumped = report.to_json()
            assert '"export_history"' in dumped, name
            status = report.to_dict()["export_history"]["status"]
            assert status == ("read" if meta is not None else "no_metadata"), name
    text = prevalence_md(build_prevalence(data, metadata=metadata(tmp_path / "m", 0)))
    assert "Time-filtered dataset" not in text
    assert "Exported history unknown" not in text
    text = prevalence_md(build_prevalence(data))
    assert "Time-filtered dataset" not in text


def test_an_unusable_value_is_announced_once_at_the_top(tmp_path):
    from synthea_quality.prevalence.render import render_markdown as prevalence_md

    data = dataset(tmp_path / "csv")
    report = build_prevalence(data, metadata=metadata(tmp_path / "meta", "abc"))
    text = prevalence_md(report)
    assert text.count("Exported history unknown") == 1
    assert first_after_title(text).startswith("> **Exported history unknown.**")
    assert report.to_dict()["export_history"]["status"] == "invalid"


def test_an_out_of_range_value_writes_the_report_with_the_unknown_notice(tmp_path):
    from synthea_quality.prevalence.cli import main
    from synthea_quality.prevalence.render import MARKDOWN_NAME

    out = tmp_path / "out"
    code = main(
        [str(dataset(tmp_path / "csv")), "--metadata", str(metadata(tmp_path / "meta", 1001)),
         "--output-dir", str(out)]
    )
    assert code == 0
    text = (out / MARKDOWN_NAME).read_text(encoding="utf-8")
    assert first_after_title(text).startswith("> **Exported history unknown.**")
