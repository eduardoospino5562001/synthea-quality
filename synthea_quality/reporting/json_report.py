"""[REPORTING] JSON serialisation of a report.

The layout belongs to :meth:`synthea_quality.models.DatasetReport.to_dict`, which
is versioned by ``schema_version``; this module only owns the file-facing details:
indentation, a trailing newline, UTF-8 without ASCII escaping, and reloading. It
reads no dataset and runs no check, and it embeds no Markdown: the JSON carries
structured statuses, messages, metrics and samples.

Determinism
-----------
The key order is fixed by the model, the checks are written in the order the report
holds them (the builder sorts them by ``check_id``), and the only value that differs
between two runs over the same data is ``generated_at``. Re-serialising what was
loaded reproduces the same text, so a report can be post-processed and written back
without drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthea_quality.models import DatasetReport


def to_payload(report: DatasetReport) -> dict[str, Any]:
    """Return the JSON-compatible payload of ``report``."""
    return report.to_dict()


def dumps(report: DatasetReport, *, indent: int | None = 2) -> str:
    """Serialise ``report`` as JSON text, ending with a newline."""
    return report.to_json(indent=indent) + "\n"


def loads(text: str) -> DatasetReport:
    """Rebuild a report from :func:`dumps` output."""
    return DatasetReport.from_json(text)


def dump(report: DatasetReport, path: str | Path, *, indent: int | None = 2) -> Path:
    """Write ``report`` to ``path`` as UTF-8 JSON and return the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(report, indent=indent), encoding="utf-8")
    return target


def load(path: str | Path) -> DatasetReport:
    """Read a report written by :func:`dump`."""
    return loads(Path(path).read_text(encoding="utf-8"))
