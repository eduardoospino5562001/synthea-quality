#!/usr/bin/env python3
"""Format reports within GitHub's 65,536-character PR comment limit."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

MAX_COMMENT_CHARS = 65_536
_NOTICE = "_The full report is in the synthea-compare-report artifact of this workflow._"


class _DetailsDepth(HTMLParser):
    """Track real HTML tags without interpreting escaped report data as markup."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "details":
            self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "details":
            self.depth = max(0, self.depth - 1)


def _closing_details(text: str) -> str:
    parser = _DetailsDepth()
    parser.feed(text)
    return "\n</details>" * parser.depth


def _excerpt(report: str, budget: int) -> str:
    if budget == 0:
        return ""
    boundary = report.rfind("\n", 0, budget + 1)
    return (report[:boundary] if boundary > 0 else report[:budget]).rstrip("\r\n")


def build_comment(report_md: str) -> str:
    """Keep the artifact notice outside folded content within the character cap.

    Reserve closing tags before cutting again so even a boundary-length folded
    report leaves room for its notice. Escaped data never contributes tags.
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
