"""[PROFILE] Descriptive profile of a Synthea CSV dataset.

The quality checks in :mod:`synthea_quality.checks` answer "is this dataset broken?".
This package answers a different question: "what does this dataset contain?". It
describes, it never judges:

* no ``PASS``/``FAIL``, no severity, no expected range and no tolerance. A section is
  either ``COMPUTED`` or ``SKIPPED`` with the reason it could not be computed, exactly
  as a quality check that cannot run is ``SKIPPED`` rather than guessed;
* it never changes what ``synthea-quality`` prints, writes or returns: the profile has
  its own command (``synthea-profile``), its own report files and its own versioned
  JSON layout;
* it reads data only through :class:`~synthea_quality.loader.DatasetLoader` (everything
  as text, only an empty field is a null) and validates each table's row structure
  first, like the checks do. Converting text into dates happens here, never in the
  loader.

Layers, in the order data flows through them:

``models``      the structured result (stdlib only, cheap to import);
``dates``       strict parsing of the date shapes Synthea writes;
``reference``   the "end of the simulation" date and where it came from;
``population``  who is alive and how old they are (reusable by later analyses);
``demographics`` the sections of the patient profile;
``build``       the only module that touches the dataset directory;
``render``      JSON and Markdown, from the model alone;
``cli``         the ``synthea-profile`` command.

``reference`` and ``population`` depend on neither ``render`` nor ``cli``: prevalence
and incidence, the next planned analyses, are expected to reuse them as they are.
"""

from __future__ import annotations
