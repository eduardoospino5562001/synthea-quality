"""[CHECKS] Deterministic checks over a Synthea CSV dataset.

A check receives data that has already been read (by ``synthea_quality.loader``)
and returns a :class:`~synthea_quality.models.CheckResult`. Checks never read
files themselves, so they are cheap to test and cannot hide I/O failures.

Each module in this package covers one family of rules:

``keys``
    primary key uniqueness and foreign key referential integrity, using the rules
    confirmed in :mod:`synthea_quality.schema.keys`.
"""
