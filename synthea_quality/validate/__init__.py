"""[VALIDATE] One report to validate a Synthea module: population, prevalence, incidence.

Checking a new module used to mean three commands (or a one-off notebook). This package
puts them behind one: ``synthea-validate-module DATASET --module FILE.json`` reads the
module's conditions — the same JSON as ``synthea-prevalence --conditions``, with an
optional ``module`` block and expected values for ``point``, ``lifetime`` and
``incidence`` — and writes one Markdown and one JSON report with:

* a summary of the population (the profile's population, age and sex sections);
* for each condition, point and lifetime prevalence among the patients alive at the end
  and incidence per 1,000 person-years (deceased included until death), each with its
  95% interval and strata;
* every expected value next to the observed one, inside or outside its 95% interval —
  never a verdict.

Nothing is computed here that the dedicated commands do not compute the same way: the
dataset is opened with :mod:`synthea_quality.dataset`, the numbers come from
:mod:`synthea_quality.prevalence.compute` and :mod:`synthea_quality.incidence.compute`,
the population from :mod:`synthea_quality.profile.demographics`, and each condition is
rendered by the prevalence and incidence renderers. The integration suite checks that
the numbers are identical to those of ``synthea-prevalence`` and ``synthea-incidence``.
"""

from __future__ import annotations
