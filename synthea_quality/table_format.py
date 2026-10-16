"""[TABLE] Shared Markdown number and cell formatting for the analysis reports.

The five analysis renderers (prevalence, incidence, validate, medications and
observations) put text written by the user (expected-value sources, condition,
medication and observation names, cohort names, reference-range sources) into
Markdown table cells. A ``|`` splits the cell and a line break splits the table,
so every such cell goes through :func:`table_cell`. Titles, notes and text outside
tables are never escaped: the caller decides what is a cell.

Numbers use :func:`fixed` (two decimals) and :func:`trimmed` (four decimals without
trailing zeros) so that a value other than zero is never shown as zero: when the
fixed text would read as zero, the value is shown with two significant digits
instead (``0.001``, ``3.2e-05``). An exact zero (including ``-0.0``) keeps its
current text, and any value that does not round to zero is formatted exactly as
before. Only Markdown is affected, never JSON.

The previous cell copies (in prevalence and observations) replaced ``|`` and
``\\n``; :func:`table_cell` also turns ``\\r`` and ``\\r\\n`` into a space, so a
carriage return cannot split a row either. Only the standard library is imported
here.
"""

from __future__ import annotations

#: Significant digits used when a non-zero value would otherwise read as zero.
SIGNIFICANT_DIGITS = 2


def table_cell(value: str) -> str:
    """Make ``value`` safe for a Markdown table cell.

    ``|`` becomes ``\\|`` and any line break (``\\r\\n``, ``\\n``, ``\\r``) becomes
    one space. Anything else, including Unicode, is left unchanged.
    """
    return value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def fixed(value: float, decimals: int, *, sign: bool = False) -> str:
    """Format ``value`` with ``decimals`` places, never showing a non-zero as zero.

    The result is ``f"{value:.{decimals}f}"`` (with a leading ``+`` when ``sign``),
    except when ``value`` is not zero and that text reads as zero: then the value
    is shown with :data:`SIGNIFICANT_DIGITS` significant digits instead. Percentages
    convert before calling (the rule applies to the shown number).
    """
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    if value != 0 and float(text) == 0:
        return f"{value:+.{SIGNIFICANT_DIGITS}g}" if sign else f"{value:.{SIGNIFICANT_DIGITS}g}"
    return text


def trimmed(value: float, decimals: int = 4) -> str:
    """Format ``value`` without trailing zeros, never showing a non-zero as zero.

    The result is ``value`` with ``decimals`` places and the trailing zeros removed
    (``120``, ``1.015``), except when ``value`` is not zero and that text reads as
    zero: then the value is shown with :data:`SIGNIFICANT_DIGITS` significant digits
    instead. Stripping applies only to the fixed text, so an exponent such as
    ``2e-10`` is kept intact.
    """
    text = f"{value:.{decimals}f}"
    if value != 0 and float(text) == 0:
        return f"{value:.{SIGNIFICANT_DIGITS}g}"
    stripped = text.rstrip("0").rstrip(".")
    return "0" if stripped == "-0" else stripped
