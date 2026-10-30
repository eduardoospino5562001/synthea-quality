#!/usr/bin/env python3
"""Format reports within GitHub's 65,536-character PR comment limit."""

import sys
from pathlib import Path

MAX_COMMENT_CHARS = 65_536
_NOTICE = "_The full report is in the synthea-compare-report artifact of this workflow._"


def build_comment(report_md: str) -> str:
    """Respect GitHub's character cap, retaining full lines and an artifact notice."""
    if not report_md.strip():
        raise ValueError("report is empty")
    if len(report_md) <= MAX_COMMENT_CHARS:
        return report_md

    suffix = "\n\n" + _NOTICE
    budget = MAX_COMMENT_CHARS - len(suffix)
    boundary = report_md.rfind("\n", 0, budget + 1)
    excerpt = report_md[:boundary] if boundary >= 0 else report_md[:budget]
    return excerpt.rstrip("\r\n") + suffix


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
