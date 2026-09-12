"""Exceptions raised by this tool.

The distinction matters for callers (and for the CLI exit codes):

* :class:`DiscoveryError`, :class:`LoadingError` … are *user or dataset input*
  problems. They are expected, they carry an actionable message, and the CLI
  reports them without a traceback.
* Anything else (``TypeError``, ``KeyError``, programming mistakes) is a bug in
  the tool and must stay visible.
"""

from __future__ import annotations


class SyntheaQualityError(Exception):
    """Base class for expected, user-actionable errors raised by this tool."""


class DiscoveryError(SyntheaQualityError):
    """The dataset directory cannot be inspected as given."""


class LoadingError(SyntheaQualityError):
    """A table could not be read from disk as a Synthea CSV file."""


class TableLoadError(LoadingError):
    """One specific table is unreadable, empty, malformed or not UTF-8.

    The failure is scoped to a single table on purpose: the caller can record it
    and keep analysing the rest of the dataset. Any operating-system failure while
    opening or reading the file (a missing file, a directory, a permission the user
    does not have) is reported as this error too, so a caller never has to catch
    ``OSError`` from the loader.
    """


class EmptyDatasetError(SyntheaQualityError):
    """The directory holds no table this tool knows how to check.

    An input error rather than a finding: with nothing to validate, a report would
    be a long list of ``SKIPPED`` checks that says nothing about the dataset. The
    caller must say so instead of returning such a report.
    """
