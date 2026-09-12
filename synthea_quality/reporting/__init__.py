"""[REPORTING] Build and render reports from structured results.

The pipeline this package keeps separate:

``data`` → ``checks`` → ``structured model`` → ``renderer``

* :mod:`synthea_quality.reporting.build` is the only module here that touches the
  file system: it inspects a dataset, runs the confirmed checks and aggregates
  their results into one :class:`~synthea_quality.models.DatasetReport`;
* :mod:`synthea_quality.reporting.json_report` serialises that report;
* :mod:`synthea_quality.reporting.markdown` renders it for a human.

Neither renderer reads a CSV, runs a check or knows how the scores were produced:
they only interpret a report that already exists. That keeps a rendered report
reproducible from its JSON alone.
"""
