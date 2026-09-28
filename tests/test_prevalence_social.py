"""Tests for the versioned list of social and administrative condition codes."""

from __future__ import annotations

from synthea_quality.prevalence.models import SNOMED_CT, normalise_system
from synthea_quality.prevalence.social import (
    SOCIAL_CODES,
    SOCIAL_LIST_ID,
    SOURCE_MODULES,
    describe_list,
    is_social,
)


def test_the_list_has_the_21_codes_of_the_two_source_modules():
    assert len(SOCIAL_CODES) == 21
    assert len({entry.code for entry in SOCIAL_CODES}) == 21
    assert all(entry.modules[0] in SOURCE_MODULES for entry in SOCIAL_CODES)
    assert sum(entry.modules[0] == "med_rec.json" for entry in SOCIAL_CODES) == 1


def test_every_entry_carries_its_provenance():
    by_code = {entry.code: entry for entry in SOCIAL_CODES}
    assert by_code["314529007"].modules == ("med_rec.json",)
    assert by_code["32911000"].modules == ("encounter/sdoh_hrsn.json", "homelessness.json")
    assert "encounter/hark_screening.json" in by_code["706893006"].modules


def test_the_list_is_not_the_snomed_finding_tag():
    displays = {entry.display for entry in SOCIAL_CODES}
    assert "Medication review due (situation)" in displays
    assert "Refugee (person)" in displays
    # clinical findings stay out of the list
    assert not is_social(SNOMED_CT, "714628002")  # Prediabetes (finding)
    assert not is_social(SNOMED_CT, "162864005")  # Body mass index 30+ - obesity (finding)
    assert not is_social(SNOMED_CT, "361055000")  # Misuses drugs (finding)
    assert not is_social(SNOMED_CT, "80583007")  # Severe anxiety (panic) (finding)


def test_membership_normalises_the_system_spelling():
    assert is_social("http://snomed.info/sct", "73595000")
    assert is_social("SNOMED-CT", "73595000")
    assert is_social(None, "73595000")
    assert not is_social("http://loinc.org", "73595000")
    assert normalise_system("SNOMED-CT") == SNOMED_CT
    assert normalise_system("http://example.org") == "http://example.org"


def test_the_description_records_version_and_commit():
    described = describe_list()
    assert described["id"] == SOCIAL_LIST_ID
    assert described["synthea_commit"] == "d9d07a6e"
    assert described["codes"] == 21 == len(described["entries"])
