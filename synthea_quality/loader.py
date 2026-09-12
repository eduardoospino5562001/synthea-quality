"""[LOADER] Safe reading of Synthea CSV files.

Conventions this module establishes for the whole tool:

* **Everything is text.** Files are read with ``dtype=str``, so UUIDs, SNOMED
  codes and LOINC codes reach the checks exactly as written; type inference can
  never truncate an identifier or turn a code into a float.
* **Only an empty field is a null.** pandas' default list of missing-value tokens
  ("NA", "N/A", "null", "None", "-", ...) would turn real values into nulls, so it
  is disabled (``keep_default_na=False, na_values=[""]``). Verified on pandas
  3.0.5: with ``dtype=str`` alone, a ``CODE`` of ``"NA"`` silently becomes NaN.
* **Never modify the source.** Files are opened read-only; a test asserts the bytes
  are unchanged after loading.
* **Nothing is dropped silently.** Anything that would change or lose data is an
  error: rows whose field count does not match the header, a header pandas cannot
  reproduce verbatim, non-UTF-8 bytes, an empty file. Errors are scoped to one
  table (:class:`~synthea_quality.errors.TableLoadError`) so the caller can record
  the failure and continue with the rest of the dataset.

Known limitation: CSV cannot distinguish "a row with fewer fields than the header"
from "trailing empty fields", and pandas pads such rows with nulls. Detecting it
would need a second full pass over the file, so it is left to a future check.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from synthea_quality.errors import TableLoadError

#: ``utf-8-sig`` strips a UTF-8 BOM when present and behaves like ``utf-8`` without one.
ENCODING = "utf-8-sig"

#: Value treated as missing. See the module docstring for why this is not pandas' default.
MISSING_VALUES = ("",)


@dataclass(frozen=True, slots=True, eq=False)
class LoadedTable:
    """One CSV table read from disk.

    ``eq=False`` on purpose: comparing two loaded tables would compare DataFrames,
    which is ambiguous in pandas and almost never what a caller means.
    """

    table: str
    path: Path
    #: Header exactly as written in the file (BOM stripped, duplicates preserved).
    header: tuple[str, ...]
    frame: pd.DataFrame
    #: Deviations worth reporting, e.g. a column subset was loaded.
    notes: tuple[str, ...] = ()

    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.frame.columns)


def read_header(path: str | Path) -> tuple[str, ...]:
    """Return the header line of a CSV file, verbatim.

    Reads with the standard library rather than pandas so the result is exactly
    what the file contains: a BOM is stripped, but duplicate column names and
    unusual spacing are preserved for the caller to report.

    :raises TableLoadError: the file is missing, is not a regular file, is empty,
        is not UTF-8, or its header cannot be parsed as CSV.
    """
    file_path = _check_readable_file(path)
    try:
        with file_path.open(newline="", encoding=ENCODING) as handle:
            row = next(csv.reader(handle), None)
    except UnicodeDecodeError as exc:
        raise TableLoadError(f"{file_path} is not valid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise TableLoadError(f"{file_path} has an unreadable header: {exc}") from exc

    if row is None:
        raise TableLoadError(f"{file_path} is empty (no header line)")
    return tuple(row)


def load_table(
    path: str | Path,
    *,
    table: str | None = None,
    columns: Sequence[str] | None = None,
    header: Sequence[str] | None = None,
) -> LoadedTable:
    """Load one Synthea CSV file into a DataFrame of text.

    :param path: file to read. It is opened read-only and never modified.
    :param table: logical table name; defaults to the file name without suffix.
    :param columns: load only these columns (all of them by default). Requesting a
        column the file does not have is an error, not a silent omission.
    :param header: header already read by the caller, to avoid reading the first
        line twice (used by :class:`DatasetLoader`).
    :raises TableLoadError: the file cannot be read as a well-formed Synthea CSV.
    """
    file_path = _check_readable_file(path)
    header_tuple = tuple(header) if header is not None else read_header(file_path)
    table_name = table or file_path.stem

    notes: list[str] = []
    usecols: list[str] | None = None
    if columns is not None:
        requested = tuple(columns)
        missing = [column for column in requested if column not in header_tuple]
        if missing:
            raise TableLoadError(
                f"{file_path} has no column(s) {missing}; the file header is {list(header_tuple)}"
            )
        usecols = list(requested)

    frame = _read_frame(file_path, usecols=usecols)

    expected_columns = header_tuple if usecols is None else tuple(usecols)
    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != expected_columns:
        raise TableLoadError(
            f"{file_path} header could not be read verbatim: expected {list(expected_columns)}, "
            f"pandas produced {list(actual_columns)} (duplicate or malformed column names)"
        )

    if usecols is not None:
        notes.append(f"loaded {len(usecols)} of {len(header_tuple)} columns: {usecols}")

    return LoadedTable(
        table=table_name,
        path=file_path,
        header=header_tuple,
        frame=frame,
        notes=tuple(notes),
    )


class DatasetLoader:
    """Loads tables on demand and keeps them, so a table is read at most once.

    Checks share one loader instance per run instead of each reading the same CSV
    again. Headers are cached too, because assessing the schema needs the header of
    every table while the expensive data frames are only needed by some checks.
    """

    def __init__(self) -> None:
        self._headers: dict[str, tuple[str, ...]] = {}
        self._tables: dict[tuple[str, tuple[str, ...] | None], LoadedTable] = {}

    def read_header(self, path: str | Path) -> tuple[str, ...]:
        """Return (and cache) the header of ``path``."""
        key = str(path)
        cached = self._headers.get(key)
        if cached is None:
            cached = read_header(path)
            self._headers[key] = cached
        return cached

    def load(
        self,
        path: str | Path,
        *,
        table: str | None = None,
        columns: Sequence[str] | None = None,
    ) -> LoadedTable:
        """Return ``path`` loaded as a table, reading it only the first time.

        The cache key includes the requested column subset, so two different
        projections of the same file are cached separately.
        """
        key = (str(path), None if columns is None else tuple(columns))
        cached = self._tables.get(key)
        if cached is None:
            cached = load_table(
                path,
                table=table,
                columns=columns,
                header=self.read_header(path) if columns is None else None,
            )
            self._tables[key] = cached
        return cached

    @property
    def loaded_tables(self) -> tuple[str, ...]:
        """Names of the tables currently held in memory."""
        return tuple(sorted({loaded.table for loaded in self._tables.values()}))

    @property
    def cache_size(self) -> int:
        """Number of cached entries (a column subset is a separate entry)."""
        return len(self._tables)

    def clear(self) -> None:
        """Drop every cached header and table, releasing memory."""
        self._headers.clear()
        self._tables.clear()


def _check_readable_file(path: str | Path) -> Path:
    """Return ``path`` as a readable regular file, or raise a clear error."""
    file_path = Path(path)
    if file_path.is_symlink() and not file_path.exists():
        raise TableLoadError(f"broken symbolic link: {file_path}")
    if not file_path.exists():
        raise TableLoadError(f"file does not exist: {file_path}")
    if not file_path.is_file():
        raise TableLoadError(f"not a regular file: {file_path}")
    return file_path


def _read_frame(file_path: Path, *, usecols: list[str] | None) -> pd.DataFrame:
    """Read the CSV with the project's conventions, translating pandas errors."""
    try:
        with warnings.catch_warnings():
            # pandas only warns when a row's field count does not match the header
            # and it therefore drops fields. That is silent data loss, so make it
            # an error and report it.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(
                file_path,
                dtype=str,
                header=0,
                index_col=False,
                keep_default_na=False,
                na_values=list(MISSING_VALUES),
                encoding=ENCODING,
                on_bad_lines="error",
                usecols=usecols,
            )
    except pd.errors.EmptyDataError as exc:
        raise TableLoadError(f"{file_path} is empty (no header line)") from exc
    except pd.errors.ParserError as exc:
        raise TableLoadError(f"{file_path} is not a well-formed CSV: {exc}") from exc
    except pd.errors.ParserWarning as exc:
        raise TableLoadError(
            f"{file_path} has rows that do not match its header, which would drop data: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise TableLoadError(f"{file_path} is not valid UTF-8: {exc}") from exc
    except ValueError as exc:
        raise TableLoadError(f"{file_path} could not be parsed: {exc}") from exc
