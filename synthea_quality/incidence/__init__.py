"""[INCIDENCE] New cases per 1,000 person-years of the conditions asked for.

Prevalence says how many people *have* a condition at a date; incidence says how fast
new cases *appear* over a period. It is a rate over time, so the population, the time
each person is at risk and the window all have to be written down:

* **window** — the last ``--window-years`` years (5 by default) before the reference date,
  ``[ref − N years, ref]``;
* **population** — every patient, **deceased included, until their death**. Keeping only
  the patients alive at the end (as the profile and the prevalence do) would drop the time
  and the events of everyone who died during the window, biasing the rate down for
  conditions that kill: survivor bias. ``--alive-only`` restricts to that cohort for
  comparison, and the report says which population it used;
* **at risk** — a patient with a record of the condition before the window is a prior
  case and is left out; the others enter at the start of the window (or at birth) and
  leave at the reference date, at death or at their **first** event, whichever is first;
* **rate** — first events over person-years, per 1,000, with an exact (Garwood) 95%
  Poisson interval, in total and by sex and by age band, the person-years of each patient
  being split between the bands they pass through.

It describes and never judges. Layers mirror the prevalence package: ``poisson`` (the
interval), ``models``, ``compute``, ``build``, ``render``, ``cli``; the definitions,
reference date, dates and ages are reused as they are.
"""

from __future__ import annotations
