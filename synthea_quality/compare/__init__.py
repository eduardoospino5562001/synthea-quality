"""[COMPARE] A readable report of what changes between two datasets.

The command ``synthea-compare`` runs ``build_module_validation`` once on each
dataset and compares the typed figures of the two results: population,
prevalence, incidence, observations, medications and the indexed cohort. It
describes differences and never judges them: there is no pass/fail, no
threshold and no verdict. Whether a change is acceptable is left to the next
task (K3).

Layers, like the other analysis packages:

``models``     the figure, its key and the comparison of one figure on two sides;
``compute``    extracting the figures of a report and comparing two lists of them;
``build``      running the module validation on each dataset;
``render``     Markdown and JSON;
``cli``        the ``synthea-compare`` command.

Only the standard library is imported by ``models`` and ``compute``.
"""

from __future__ import annotations

__all__ = ["models", "compute", "build", "render", "cli"]
