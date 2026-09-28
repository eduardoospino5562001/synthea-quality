"""Extract the social and administrative condition codes from a Synthea checkout.

The list in ``synthea_quality/prevalence/social.py`` is not typed by hand: it is every
code a ``ConditionOnset`` state writes in the source modules below, with every module
file (among all of Synthea's modules) that also writes it, so each entry carries its
provenance.

    python scripts/extract_social_codes.py ~/src/synthea-upstream

prints the ``SOCIAL_CODES`` literal to paste into ``social.py``. The checkout is only
read. Record the commit it was taken from in ``SYNTHEA_COMMIT``.

Why these two modules
---------------------
* ``encounter/sdoh_hrsn.json`` — "SDoH HRSN", the social determinants of health /
  health-related social needs screening: employment, education, stress, social
  isolation, violence, housing, transport, criminal record, military service, migration.
* ``med_rec.json`` — "Medication Reconciliation": ``Medication review due (situation)``,
  an administrative marker written at wellness encounters.

Selecting by module rather than by the SNOMED semantic tag is deliberate: a
``(finding)`` filter would drop clinical findings such as Prediabetes or obesity
(written by ``wellness_encounters.json``) and keep ``Medication review due
(situation)`` and ``Refugee (person)``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SOURCE_MODULES = ("encounter/sdoh_hrsn.json", "med_rec.json")


def condition_onsets(module: dict) -> list[dict]:
    """Every code of every ``ConditionOnset`` state of a module, recursively."""
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "ConditionOnset":
                found.extend(node.get("codes", []))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(module.get("states", {}))
    return found


def main(checkout: Path) -> None:
    modules_dir = checkout / "src" / "main" / "resources" / "modules"
    every_module = {
        str(path.relative_to(modules_dir)): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(modules_dir.rglob("*.json"))
    }
    emitted_by: dict[str, set[str]] = {}
    for name, module in every_module.items():
        for code in condition_onsets(module):
            emitted_by.setdefault(code["code"], set()).add(name)

    entries: dict[str, tuple[str, str, str]] = {}
    for source in SOURCE_MODULES:
        for code in condition_onsets(every_module[source]):
            entries.setdefault(code["code"], (code["system"], code["display"], source))

    quote = json.dumps
    print("SOCIAL_CODES: tuple[SocialCode, ...] = (")
    for code, (system, display, source) in sorted(entries.items(), key=lambda e: e[1][1]):
        modules = [source, *sorted(emitted_by[code] - {source})]
        print("    SocialCode(")
        print(f"        {quote(code)},")
        print(f"        {quote(display)},")
        print(f"        {quote(system)},")
        print(f"        ({', '.join(quote(m) for m in modules)},),")
        print("    ),")
    print(")")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: extract_social_codes.py SYNTHEA-CHECKOUT")
    main(Path(sys.argv[1]).expanduser())
