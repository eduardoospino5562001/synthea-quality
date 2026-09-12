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
