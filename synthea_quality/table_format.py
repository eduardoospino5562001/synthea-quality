"""[TABLE] Safe Markdown table cells shared by the analysis reports.

The five analysis renderers (prevalence, incidence, validate, medications and
observations) put text written by the user (expected-value sources, condition,
medication and observation names, cohort names, reference-range sources) into
Markdown table cells. A ``|`` splits the cell and a line break splits the table,
so every such cell goes through :func:`table_cell`. Titles, notes and text outside
tables are never escaped: the caller decides what is a cell.

Only Markdown is affected, never JSON. The previous copies (in prevalence and
observations) replaced ``|`` and ``\\n``; this one also turns ``\\r`` and ``\\r\\n``
into a space, so a carriage return cannot split a row either. Only the standard
library is imported here.
"""

from __future__ import annotations


def table_cell(value: str) -> str:
    """Make ``value`` safe for a Markdown table cell.

    ``|`` becomes ``\\|`` and any line break (``\\r\\n``, ``\\n``, ``\\r``) becomes
    one space. Anything else, including Unicode, is left unchanged.
    """
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("|", "\\|")
