#!/usr/bin/env python3
"""Limit comments while recognizing only the renderer's whole-line fold tags.

Data is escaped or placed behind a prefix, so tags inside code spans must never
change the structural depth used to keep the artifact notice visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_COMMENT_CHARS = 65_536
_NOTICE = "_The full report is in the synthea-compare-report artifact of this workflow._"


def _closing_details(text: str) -> str:
    """Count whole-line tags because prefixed data and code spans are not structure."""
    depth = 0
    for line in text.splitlines():
        if line == "<details>":
            depth += 1
        elif line == "</details>":
            depth = max(0, depth - 1)
    return "\n</details>" * depth


def _excerpt(report: str, budget: int) -> str:
    if budget == 0:
        return ""
    boundary = report.rfind("\n", 0, budget + 1)
    return (report[:boundary] if boundary > 0 else report[:budget]).rstrip("\r\n")


def build_comment(report_md: str) -> str:
    """Keep the artifact notice outside folded content within the character cap.

    Reserve closing tags before cutting again so even a boundary-length folded
    report leaves room for its notice. Only the renderer's whole-line tags count;
    escaped or prefixed data, including tags inside code spans, is not structure.
    """
    if not report_md.strip():
        raise ValueError("report is empty")
    if len(report_md) <= MAX_COMMENT_CHARS:
        return report_md

    suffix = "\n\n" + _NOTICE
    budget = MAX_COMMENT_CHARS - len(suffix)
    limit = budget
    excerpt = _excerpt(report_md, limit)
    closing = _closing_details(excerpt)
    while len(excerpt) + len(closing) > budget:
        limit = max(0, min(limit - 1, budget - len(closing)))
        excerpt = _excerpt(report_md, limit)
        closing = _closing_details(excerpt)
    return excerpt + closing + suffix


def main() -> int:
    """Print a report as a comment, or describe invalid input on standard error."""
    if len(sys.argv) != 2:
        print("Usage: pr_comment.py <report.md>", file=sys.stderr)
        return 1
    try:
        report_md = Path(sys.argv[1]).read_text(encoding="utf-8")
        comment = build_comment(report_md)
    except (OSError, UnicodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    sys.stdout.write(comment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
