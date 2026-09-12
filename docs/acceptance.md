# Acceptance baseline

This document is the reproducible contract of the tool with the dataset Synthea
recommends. `tests/integration/test_acceptance_official_sample.py` enforces every
number on this page by running the real command line as a user would.

## What it proves

That somebody outside the project can point the tool at the official sample and get
the documented result using only the public interface — discovery, the contract
comparison, all 155 checks, both reports, the exit codes, determinism, the
reference-evidence attribution and untouched source files — and that a single known
corruption is reported exactly, as a failure with exit code 1.

## The dataset

| | |
| --- | --- |
| Source | `synthetichealth/synthea-sample-data` → `downloads/latest` |
| URL | <https://github.com/synthetichealth/synthea-sample-data/tree/main/downloads/latest> |
| Archive | `synthea_sample_data_csv_latest.zip` (5,960,866 bytes), sha256 `d61417b551e5b0997c33851b339c157421751f0ea68c18ea686ceb1850907c35` |
| Dataset | 18 CSV files, 62,862,715 bytes |
| **Dataset digest** | `7ecb7460dd31f9921f433a9c8d341f265e7b55b2a2ae9f0b32199deb8760769d` |
| Produced by | Synthea `master` commit `d9d07a6e` (2026-08-17), published 2026-08-18 |
| Population | 100 patients |

The dataset digest is the anchor: sha256 over the sorted `name\0sha256(content)` lines
of the extracted CSV files. **The archive checksum is not used as the anchor**, because
`downloads/latest` is a moving target and an archive can be re-packaged without its
contents changing; the CSV files here are byte-identical to the copy this baseline was
first measured on, even though the archive size differs from an earlier note in this
project. The digest changes only when the data changes, which is when a re-baseline is
due.

## How to reproduce

```bash
# 1. get the dataset — it is never stored in this repository
python scripts/fetch_official_sample.py          # → ~/synthea-sample-data/csv-latest

# 2. run the acceptance checks
.venv/bin/pytest -m integration

# 3. or run the whole suite, which includes them
.venv/bin/pytest
```

An existing extraction can be used instead:

```bash
SYNTHEA_QUALITY_SAMPLE=/path/to/csv-latest .venv/bin/pytest -m integration
```

When the dataset is absent the acceptance tests **skip** with that message, so a clean
checkout stays green. The first test is a tripwire on the dataset digest: if Synthea
republishes `latest`, it fails before the numbers do and points here.

The flow a person actually runs is the command line itself:

```bash
.venv/bin/synthea-quality <dataset-directory> --output-dir <output-directory>
```

## Expected results — clean sample

| Check | Expected |
| --- | --- |
| Exit code | `0` |
| stdout | summary: tables, contract, counts, both report paths |
| stderr | empty |
| Tables observed | 18 of the 19 the contract describes |
| Missing tables | `patient_expenses` |
| Unknown CSV files | none |
| Load errors | none |
| Contract | `COMPATIBLE`, described as 18 of 19 tables observed, without claiming the dataset version |
| Total checks | 155, all identifiers unique |
| `PASS` | 143 |
| `WARNING` | 6 |
| `FAIL` | 0 |
| `NOT_APPLICABLE` | 3 |
| `SKIPPED` | 3 |
| `ERROR` | 0 |
| Warnings | `duplicates.observations`, `duplicates.supplies`, `empty_columns.allergies`, `empty_columns.claims`, `empty_columns.claims_transactions`, `empty_columns.payers` |
| Checks by category | `primary_keys` 8, `foreign_keys` 42, `data_quality` 86, `temporal` 19 |
| Findings by severity | `HIGH` 0, `MEDIUM` 2, `LOW` 4 |
| Reports | `synthea_quality_report.md` and `synthea_quality_report.json` in the output directory |
| Source files | unchanged (sha256 of every CSV identical before and after) |
| Repository | no new file, no report anywhere inside it |

The six warnings are part of the baseline on purpose: this dataset repeats rows in
`observations` and `supplies` and leaves some columns entirely empty, which the tool
reports as information rather than as defects.

## Expected results — one known corruption

A copy of the sample with exactly one corrupted value (the first `encounters` row has
`STOP` earlier than `START`):

| Check | Expected |
| --- | --- |
| Exit code | `1` |
| FAIL | exactly 1: `temporal.start_le_stop.encounters`, severity `MEDIUM`, `violations` 1 of 5571 evaluated |
| Everything else | unchanged from the baseline, with `PASS` 143 → 142 |
| Reports | written, with the failure in the findings section and in the terminal summary |

## Expected results — usage error

A path that does not exist:

| Check | Expected |
| --- | --- |
| Exit code | `2` |
| stderr | `synthea-quality: error: argument DATASET-DIRECTORY: dataset directory does not exist: …` |
| Traceback | none |
| Output directory | not created |
| Repository | unchanged |

## Measured cost

Measured on this machine for the clean run (a fresh interpreter runs the command line
as its child and reports `ru_maxrss`):

| Measurement | Value |
| --- | --- |
| Wall time, whole run | 5.5 s |
| Peak RSS, child process | ~146 MB |
| Acceptance test file itself | 9 tests, 45 s (it runs the command line eleven times) |

For comparison, the per-family measurements in the README were taken in-process on the
same dataset: 1.1 s for the key checks, 3.0 s for the quality checks, 0.6 s for the
temporal checks.

## Re-baselining

When the digest tripwire fails, the dataset changed and the numbers here may legitimately
differ. To re-baseline:

1. `python scripts/fetch_official_sample.py` and note the new dataset digest;
2. `.venv/bin/synthea-quality ~/synthea-sample-data/csv-latest --output-dir /tmp/rebaseline`
   and inspect both reports, paying attention to any new `FAIL`;
3. if the result is sound, update this page, the constants at the top of
   `tests/integration/test_acceptance_official_sample.py`, `BASELINE_DATASET_DIGEST` in
   `scripts/fetch_official_sample.py` and the Scalability table in the README;
4. run `.venv/bin/pytest` and commit the update on its own, with the new digest quoted
   in the message.

A new `FAIL` is not automatically a reason to re-baseline: confirm first whether the
sample or the check changed.

## Deliberately out of scope

The acceptance run makes no statistical claim. Nothing here measures prevalence,
incidence or any distribution, no tolerance is invented, and a rate deviation cannot
fail a run: every check is deterministic and every baseline number is exact.
