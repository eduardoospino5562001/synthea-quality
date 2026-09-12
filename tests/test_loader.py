"""Tests for safe CSV loading.

Normal paths, edge cases and failure paths, plus the two promises that matter most
for this tool: identifiers survive the read, and nothing is dropped silently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from synthea_quality import loader
from synthea_quality.errors import LoadingError, TableLoadError
from synthea_quality.loader import DatasetLoader, load_table, read_header

PATIENTS_TEXT = (
    "Id,BIRTHDATE,DEATHDATE,CODE,VALUE\n"
    "00022a9e-d5ba-4aae-9656-faedd47f967d,1984-12-30,,314076,7.2\n"
    "000e3181-56f9-4cb7-a66e-53dff6ce74d2,1955-01-02,,007,\n"
)


def write_bytes(directory: Path, name: str, payload: bytes) -> Path:
    path = directory / name
    path.write_bytes(payload)
    return path


def write_text(directory: Path, name: str, text: str, *, encoding: str = "utf-8") -> Path:
    path = directory / name
    path.write_text(text, encoding=encoding, newline="")
    return path


# --------------------------------------------------------------------------- #
# normal loading
# --------------------------------------------------------------------------- #


def test_loads_a_small_table_with_the_expected_shape(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)

    loaded = load_table(path)

    assert loaded.table == "patients"
    assert loaded.path == path
    assert loaded.rows == 2
    assert loaded.columns == ("Id", "BIRTHDATE", "DEATHDATE", "CODE", "VALUE")
    assert loaded.header == loaded.columns
    assert loaded.notes == ()


def test_table_name_can_be_overridden(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients_data.csv", PATIENTS_TEXT)

    assert load_table(path, table="patients").table == "patients"


def test_identifiers_and_codes_are_kept_as_text(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)

    frame = load_table(path).frame

    assert frame["Id"][0] == "00022a9e-d5ba-4aae-9656-faedd47f967d"
    # leading zeros and numeric-looking codes must not become numbers
    assert frame["CODE"][1] == "007"
    assert frame["CODE"][0] == "314076"
    assert all(isinstance(value, str) for value in frame["CODE"])


@pytest.mark.parametrize("token", ["NA", "N/A", "null", "None", "nan", "-", "NaN"])
def test_values_that_look_like_missing_tokens_by_default_are_preserved(
    tmp_path: Path, token: str
) -> None:
    """pandas would turn every one of these into NaN with its default NA list."""
    path = write_text(tmp_path, "conditions.csv", f"START,CODE\n2020-01-01,{token}\n")

    frame = load_table(path).frame

    assert frame["CODE"][0] == token
    assert not pd.isna(frame["CODE"][0])


def test_an_empty_field_is_missing(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)

    frame = load_table(path).frame

    assert pd.isna(frame["DEATHDATE"][0])
    assert pd.isna(frame["VALUE"][1])
    assert not pd.isna(frame["CODE"][1])


def test_reading_only_a_subset_of_columns_is_recorded(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)

    loaded = load_table(path, columns=("Id", "CODE"))

    assert loaded.columns == ("Id", "CODE")
    assert loaded.rows == 2
    assert loaded.header == ("Id", "BIRTHDATE", "DEATHDATE", "CODE", "VALUE")
    assert len(loaded.notes) == 1 and "2 of 5 columns" in loaded.notes[0]


def test_read_header_returns_the_first_line_only(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)

    assert read_header(path) == ("Id", "BIRTHDATE", "DEATHDATE", "CODE", "VALUE")


def test_requested_columns_come_back_in_the_requested_order(tmp_path: Path) -> None:
    """pandas returns a subset in file order; the loader normalises it.

    Without this, a caller asking for ``("VALUE", "Id")`` would silently receive
    ``("Id", "VALUE")``, and any code pairing the two column by column would be
    wrong.
    """
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)

    loaded = load_table(path, columns=("VALUE", "Id"))

    assert loaded.columns == ("VALUE", "Id")
    assert loaded.frame["VALUE"][0] == "7.2"


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #


def test_header_only_file_loads_with_zero_rows(tmp_path: Path) -> None:
    path = write_text(tmp_path, "supplies.csv", "DATE,PATIENT,QUANTITY\n")

    loaded = load_table(path)

    assert loaded.rows == 0
    assert loaded.columns == ("DATE", "PATIENT", "QUANTITY")


def test_utf8_bom_is_stripped(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT, encoding="utf-8-sig")

    loaded = load_table(path)

    assert loaded.header[0] == "Id"
    assert loaded.columns[0] == "Id"


def test_crlf_line_endings_and_accents_are_preserved(tmp_path: Path) -> None:
    path = write_bytes(
        tmp_path,
        "patients.csv",
        "Id,COUNTY\r\nabc,Middlesex\r\n".encode("utf-8") + "zzz,Mañana\r\n".encode("utf-8"),
    )

    frame = load_table(path).frame

    assert frame["COUNTY"][1] == "Mañana"
    assert frame["COUNTY"][0] == "Middlesex"


def test_quoted_fields_with_commas_are_read_correctly(tmp_path: Path) -> None:
    path = write_text(tmp_path, "observations.csv", 'DATE,DESCRIPTION\n2020-01-01,"Pain, 0-10"\n')

    frame = load_table(path).frame

    assert frame["DESCRIPTION"][0] == "Pain, 0-10"


# --------------------------------------------------------------------------- #
# failure paths
# --------------------------------------------------------------------------- #


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(TableLoadError, match="does not exist"):
        load_table(tmp_path / "nope.csv")


def test_directory_instead_of_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(TableLoadError, match="not a regular file"):
        load_table(tmp_path)


def test_broken_symlink_is_a_clear_error(tmp_path: Path) -> None:
    link = tmp_path / "patients.csv"
    link.symlink_to(tmp_path / "gone.csv")

    with pytest.raises(TableLoadError, match="broken symbolic link"):
        load_table(link)


def test_empty_file_is_a_clear_error(tmp_path: Path) -> None:
    path = write_bytes(tmp_path, "patients.csv", b"")

    with pytest.raises(TableLoadError, match="empty"):
        load_table(path)


def test_file_that_is_not_utf8_is_a_clear_error(tmp_path: Path) -> None:
    path = write_bytes(tmp_path, "patients.csv", "Id,N\n1,José\n".encode("latin-1"))

    with pytest.raises(TableLoadError, match="not valid UTF-8"):
        load_table(path)


def test_row_with_more_fields_than_the_header_is_rejected(tmp_path: Path) -> None:
    """pandas would drop the extra field and only warn; that must not pass as success."""
    path = write_text(tmp_path, "encounters.csv", "Id,PATIENT\n1,abc,EXTRA\n")

    with pytest.raises(TableLoadError, match="drop data"):
        load_table(path)


def test_unterminated_quote_is_rejected(tmp_path: Path) -> None:
    path = write_text(tmp_path, "encounters.csv", 'Id,PATIENT\n1,"abc\n')

    with pytest.raises(TableLoadError, match="not a well-formed CSV"):
        load_table(path)


def test_duplicate_column_names_in_the_header_are_rejected(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", "Id,Id\n1,2\n")

    with pytest.raises(TableLoadError, match="verbatim"):
        load_table(path)


def test_requesting_a_column_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)

    with pytest.raises(TableLoadError, match=r"has no column\(s\) \['NOPE'\]"):
        load_table(path, columns=("Id", "NOPE"))


def test_loading_errors_share_a_common_base() -> None:
    assert issubclass(TableLoadError, LoadingError)


# --------------------------------------------------------------------------- #
# the source file is never modified
# --------------------------------------------------------------------------- #


def test_loading_does_not_modify_the_source_file(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    load_table(path, columns=("Id",))
    read_header(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


# --------------------------------------------------------------------------- #
# DatasetLoader caching
# --------------------------------------------------------------------------- #


def test_dataset_loader_reads_each_table_only_once(tmp_path: Path, monkeypatch) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)
    calls: list[Path] = []
    original = loader._read_frame

    def counting_read_frame(file_path: Path, *, usecols: list[str] | None = None):
        calls.append(file_path)
        return original(file_path, usecols=usecols)

    monkeypatch.setattr(loader, "_read_frame", counting_read_frame)
    dataset = DatasetLoader()

    first = dataset.load(path)
    second = dataset.load(path)

    assert first is second
    assert len(calls) == 1
    assert dataset.cache_size == 1
    assert dataset.loaded_tables == ("patients",)


def test_dataset_loader_caches_headers(tmp_path: Path, monkeypatch) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)
    calls: list[Path] = []
    original = loader.read_header

    def counting_read_header(file_path):
        calls.append(Path(file_path))
        return original(file_path)

    monkeypatch.setattr(loader, "read_header", counting_read_header)
    dataset = DatasetLoader()

    assert dataset.read_header(path) == dataset.read_header(path)
    assert len(calls) == 1


def test_different_column_subsets_are_cached_separately(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)
    dataset = DatasetLoader()

    full = dataset.load(path)
    subset = dataset.load(path, columns=("Id",))

    assert full is not subset
    assert full.columns != subset.columns
    assert dataset.cache_size == 2
    # loading the same subset again reuses the cached entry
    assert dataset.load(path, columns=("Id",)) is subset


def test_clear_releases_cached_tables(tmp_path: Path) -> None:
    path = write_text(tmp_path, "patients.csv", PATIENTS_TEXT)
    dataset = DatasetLoader()
    dataset.load(path)
    dataset.read_header(path)

    dataset.clear()

    assert dataset.cache_size == 0
    assert dataset.loaded_tables == ()


# --------------------------------------------------------------------------- #
# A file the operating system refuses to open is not a crash
# --------------------------------------------------------------------------- #


def unreadable_file(directory: Path, name: str = "patients.csv") -> Path:
    """Return a file this user cannot read, or skip if the platform allows it anyway."""
    path = write_text(directory, name, PATIENTS_TEXT)
    path.chmod(0o000)
    try:
        path.open("rb").close()
    except OSError:
        return path
    pytest.skip("this user can read a 0o000 file (running as root?)")


def test_an_unreadable_file_raises_a_table_load_error(tmp_path: Path) -> None:
    """An expected OS failure must be reported like any other unreadable table.

    Measured before the fix: a 0o000 file raised ``PermissionError`` out of
    ``load_table`` and ``read_header``, escaped the loader's error contract, and
    aborted the whole run with no record of which table was lost.
    """
    path = unreadable_file(tmp_path)

    with pytest.raises(TableLoadError) as load_error:
        load_table(path)
    with pytest.raises(TableLoadError) as header_error:
        read_header(path)

    assert "could not be read" in str(load_error.value)
    assert str(path) in str(load_error.value)
    assert "could not be read" in str(header_error.value)


def test_an_unreadable_file_does_not_escape_the_loader_as_an_oserror(tmp_path: Path) -> None:
    """``TableLoadError`` is what callers catch; anything else would abort the run."""
    path = unreadable_file(tmp_path, "encounters.csv")

    with pytest.raises(TableLoadError):
        DatasetLoader().load(path)
