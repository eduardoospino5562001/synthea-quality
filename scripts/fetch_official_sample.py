"""Download and extract the official Synthea CSV sample used as the acceptance baseline.

Source
------
``synthetichealth/synthea-sample-data`` → ``downloads/latest``::

    https://github.com/synthetichealth/synthea-sample-data/tree/main/downloads/latest

The project regenerates that directory from Synthea ``master`` in its own release
workflow. The dataset this project's baseline was measured on is::

    18 CSV files, 62,862,715 bytes
    dataset digest 7ecb7460dd31f9921f433a9c8d341f265e7b55b2a2ae9f0b32199deb8760769d
    zip sha256   d61417b551e5b0997c33851b339c157421751f0ea68c18ea686ceb1850907c35
    produced from synthea commit d9d07a6e (2026-08-17), published 2026-08-18

Because ``latest`` is a moving target, this script reports both the archive checksum
and a digest of the extracted files. The file digest is the meaningful anchor: it does
not change if the project merely re-zips the same CSV content, and the acceptance test
compares against it. See ``docs/acceptance.md``.

The data is written outside the repository by default, and never to a path inside it,
so a downloaded dataset cannot end up versioned by accident.

Usage::

    python scripts/fetch_official_sample.py                    # ~/synthea-sample-data
    python scripts/fetch_official_sample.py --destination /data/synthea
    python scripts/fetch_official_sample.py --zip existing.zip # extract an archive you already have
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCE_URL = (
    "https://github.com/synthetichealth/synthea-sample-data/raw/main/downloads/latest/"
    "synthea_sample_data_csv_latest.zip"
)
ARCHIVE_NAME = "synthea_sample_data_csv_latest.zip"
DEFAULT_DESTINATION = Path.home() / "synthea-sample-data"

#: Documented checksums of the baseline dataset (see the module docstring).
BASELINE_ZIP_SHA256 = "d61417b551e5b0997c33851b339c157421751f0ea68c18ea686ceb1850907c35"
BASELINE_DATASET_DIGEST = "7ecb7460dd31f9921f433a9c8d341f265e7b55b2a2ae9f0b32199deb8760769d"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_digest(directory: Path) -> str:
    """Digest of the extracted CSVs, independent of how the archive was packaged."""
    lines = [
        f"{path.name}\0{sha256_of_file(path)}"
        for path in sorted(directory.glob("*.csv"))
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    return target


def extract(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    return destination


def report(destination: Path, archive: Path | None) -> int:
    """Print the digests and say whether this is the documented baseline dataset."""
    files = sorted(destination.glob("*.csv"))
    if not files:
        print(f"error: no CSV file found in {destination}", file=sys.stderr)
        return 2

    digest = dataset_digest(destination)
    total = sum(path.stat().st_size for path in files)
    print(f"\ndataset:        {destination}")
    print(f"files:          {len(files)}")
    print(f"bytes:          {total:,}")
    print(f"dataset digest: {digest}")
    if archive is not None:
        print(f"zip sha256:     {sha256_of_file(archive)}")

    if digest == BASELINE_DATASET_DIGEST:
        print("\nthis is the documented baseline dataset (see docs/acceptance.md)")
        return 0

    print(
        "\nwarning: this is NOT the documented baseline dataset.\n"
        f"  expected digest: {BASELINE_DATASET_DIGEST}\n"
        "  the project regenerates downloads/latest, so it may simply be newer.\n"
        "  Re-run the acceptance checks and update docs/acceptance.md with the new\n"
        "  numbers if you intend to re-baseline.",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract the official Synthea CSV sample (acceptance baseline)."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"directory for the archive and the extracted csv-latest (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="use this archive instead of downloading one",
    )
    args = parser.parse_args(argv)

    destination = args.destination.resolve()
    if REPOSITORY_ROOT in destination.parents or destination == REPOSITORY_ROOT:
        print(
            f"error: refusing to write the dataset inside the repository ({destination}); "
            "choose a path outside it so the data is never versioned by accident",
            file=sys.stderr,
        )
        return 2

    archive = args.zip if args.zip is not None else destination / ARCHIVE_NAME
    if args.zip is None and not archive.exists():
        download(SOURCE_URL, archive)
    elif not archive.exists():
        print(f"error: archive not found: {archive}", file=sys.stderr)
        return 2

    csv_dir = extract(archive, destination / "csv-latest")
    return report(csv_dir, archive)


if __name__ == "__main__":
    raise SystemExit(main())
