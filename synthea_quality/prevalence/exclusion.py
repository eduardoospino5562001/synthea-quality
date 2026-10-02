"""[PREVALENCE] A user-supplied list of condition codes to exclude from the general table.

The general table of ``synthea-prevalence`` leaves out, by default, the 21 social and
administrative codes of the built-in list (see :mod:`synthea_quality.prevalence.social`).
The maintainers of Synthea say there is no easy way to identify such codes, so the
tool accepts ``--exclude-codes FILE``, which **replaces** the built-in list: the codes
of the file are left out instead.

File format (UTF-8 text, one entry per line):

* the **first token** of a line is the code; a ``#`` starts a comment (a whole line or
  the end of one); blank lines are ignored;
* nothing may follow the code except the comment: extra text before the ``#`` is an
  error, so a stray column is never silently dropped;
* a code matches ``^[A-Za-z0-9][A-Za-z0-9.\\\\-]*$``; anything else is an error with
  ``file:line``;
* matching is **by CODE alone, in any SYSTEM**: ``conditions.csv`` writes SNOMED CT
  codes, and a list that names them should work whatever the ``SYSTEM`` column says;
* repeated codes are kept once and **counted** (``duplicates``);
* a file with no code is valid: it excludes nothing, and the report says so.

Errors (missing or unreadable file, non-UTF-8 bytes, an invalid line) raise
:class:`ExclusionListError`, which the command line reports as a usage error
(exit 2).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthea_quality.errors import SyntheaQualityError

#: A code is an alphanumeric token, optionally dotted or dashed (SNOMED, LOINC, RxNorm).
CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")


class ExclusionListError(SyntheaQualityError):
    """An exclusion file cannot be used as given."""


@dataclass(frozen=True, slots=True)
class ExclusionList:
    """The codes of an ``--exclude-codes`` file, deduplicated and ordered."""

    #: As the user gave it (kept for the report, so the provenance is exact).
    path: str
    #: Hex SHA-256 of the file bytes, so a report says which list it used.
    sha256: str
    #: Distinct codes, sorted for deterministic output.
    codes: tuple[str, ...]
    #: Lines repeating a code already seen.
    duplicates: int

    def describe(self) -> dict[str, Any]:
        """The list's identity and provenance, as recorded in a report."""
        return {
            "id": f"file:{Path(self.path).name}",
            "source": "file",
            "path": self.path,
            "sha256": self.sha256,
            "codes": len(self.codes),
            "duplicates": self.duplicates,
            "entries": [{"code": code} for code in self.codes],
        }


def read_exclusion_file(path: str | Path) -> ExclusionList:
    """Read the exclusion list of ``path``.

    :raises ExclusionListError: the file does not exist, cannot be read, is not
        UTF-8, or has a line with no usable code.
    """
    file_path = Path(path)
    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        raise ExclusionListError(
            f"exclusion file {file_path} could not be read: {exc}"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExclusionListError(
            f"exclusion file {file_path} is not UTF-8: {exc}"
        ) from exc
    sha256 = hashlib.sha256(raw).hexdigest()

    seen: set[str] = set()
    ordered: list[str] = []
    duplicates = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        content, _, _comment = line.partition("#")
        stripped = content.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) > 1:
            raise ExclusionListError(
                f"{file_path}:{lineno}: extra text after the code "
                f"{parts[0]!r}; write one code per line"
            )
        code = parts[0]
        if not CODE_PATTERN.match(code):
            raise ExclusionListError(
                f"{file_path}:{lineno}: invalid code {code!r}"
            )
        if code in seen:
            duplicates += 1
            continue
        seen.add(code)
        ordered.append(code)
    ordered.sort()
    return ExclusionList(
        path=str(path),
        sha256=sha256,
        codes=tuple(ordered),
        duplicates=duplicates,
    )
