"""[PREVALENCE] Prevalence of conditions among the patients alive at the end of the simulation.

Validating a new Synthea module usually means a one-off notebook that loads the CSV
export, picks the module's condition codes and divides a count of patients by a count of
people. This package keeps that computation in one reproducible place, with the
definitions written down:

* **point prevalence** — alive patients in whom the condition is *active* at the
  reference date (``START <= ref`` and ``STOP`` empty or ``STOP > ref``), over the alive
  patients;
* **lifetime prevalence** — alive patients with at least one record whose
  ``START <= ref``, over the alive patients;
* both in total and, for the conditions asked for, by age band and by sex, each with a
  95% Wilson confidence interval.

It describes and never judges: there is no ``PASS``/``FAIL``, a value given with
``--expected`` is shown next to the observed one together with whether it lies inside
or outside the 95% CI, and nothing more. Incidence is a different measure and is not
computed here.

Layers:

``models``       the structured result and the Wilson interval (stdlib only);
``social``       the versioned list of social and administrative codes, with provenance;
``definitions``  the conditions asked for (``--condition``, ``--conditions FILE``,
                 ``--expected``);
``compute``      the rates, from frames already loaded;
``build``        the only module that reads the dataset;
``render``       JSON and Markdown;
``cli``          the ``synthea-prevalence`` command.

The alive cohort, ages and the reference date come from
:mod:`synthea_quality.profile.population` and :mod:`synthea_quality.profile.reference`,
so the profile and the prevalence can never disagree about who is alive or when the
simulation ended.
"""

from __future__ import annotations
