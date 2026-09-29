"""[MEDICATIONS] Which share of a cohort takes each medication.

A module that treats a condition prescribes medications; checking it means asking how
many of the patients with the condition have each of them. This package answers, for the
medications listed in a module file:

* **active** — alive patients of the cohort with a prescription started on or before the
  reference date and not stopped by then (``STOP`` empty or after it);
* **ever** — alive patients of the cohort with a prescription started on or before it;
* both over the patients of the cohort, with a 95% Wilson interval, and how many of those
  patients have a record whose ``REASONCODE`` is one of the cohort condition's codes.

A cohort is the alive patients with a condition of the same file
(:mod:`synthea_quality.condition_cohort`); without one, every alive patient. A code that
never appears in ``medications.csv`` and records without ``STOP`` get a note. It
describes and never judges: an expected share is shown inside or outside the 95% CI.

Layers:

``definitions``  the ``medications`` of a module file;
``models``       the structured result (stdlib only);
``compute``      the shares, from frames already loaded;
``render``       Markdown, for ``synthea-validate-module``, which is where medications are
                 reported (there is no separate command).
"""

from __future__ import annotations
