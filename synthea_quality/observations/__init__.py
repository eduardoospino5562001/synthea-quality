"""[OBSERVATIONS] The distribution of numeric observation values among the alive patients.

Besides how many patients have a condition, a module shapes the values it records: the
blood pressure of the hypertensive, the HbA1c of the diabetic. This package describes
them, code by code:

* **one value per patient** — the latest on or before the reference date, so a patient
  measured often weighs no more than one measured once;
* n, minimum, the 5th, 25th, 50th, 75th and 95th percentiles and maximum, **per unit**
  (units are never converted), in total and by age band and sex;
* optionally among the alive patients with a condition of the same module file (a
  cohort), and next to a reference range with the patients below, within and above it.

Rows that cannot be used — a ``TYPE`` other than ``numeric``, a ``VALUE`` that is not a
number, a date after the reference date — are counted and reported, never dropped
silently. It describes and never judges.

Layers:

``definitions``  the observations asked for (``--observation``, ``observations`` of a
                 module file), cohorts and reference ranges;
``models``       the structured result (stdlib only);
``compute``      the values, from frames already loaded;
``build``        the only module that reads the dataset;
``render``       JSON and Markdown;
``cli``          the ``synthea-observations`` command.

The alive cohort, ages and strata are those of :mod:`synthea_quality.prevalence.compute`;
a condition cohort is the numerator of that condition's prevalence
(:mod:`synthea_quality.condition_cohort`). ``synthea-validate-module`` renders the
observations of a module file with the same functions.
"""

from __future__ import annotations
