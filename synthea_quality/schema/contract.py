"""[SCHEMA] Versioned schema contracts and header matching.

A :class:`SchemaContract` bundles the table catalogue of one Synthea version with
its provenance, so a second version can be added later without touching the check
modules. :data:`CURRENT_CONTRACT` is the only version implemented today; older
datasets are expected to fail the comparison and be reported as ``INCOMPATIBLE``
instead of being coerced into the current contract.

Agreement is reported as ``COMPATIBLE``, never as "this dataset is version X":
matching the tables that happened to be observed is only evidence of consistency,
while a deviation is positive evidence of a different version.

The matching logic is pure: it takes the header a caller read from a CSV file and
compares it with the contract. Reading files is the loader's job, which keeps this
module testable without touching the file system.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from synthea_quality.schema.tables import (
    CONTRACT_ID,
    SOURCE_COMMIT,
    SOURCE_FILE,
    SOURCE_REPOSITORY,
    SYNTHEA_TABLES,
    TableSpec,
)


class MatchKind(str, Enum):
    """How an actual CSV header relates to the contract."""

    EXACT = "EXACT"
    #: Different columns: one or more missing, unexpected, or repeated.
    COLUMNS_DIFFERENT = "COLUMNS_DIFFERENT"
    #: Same columns, written in a different order.
    ORDER_DIFFERS = "ORDER_DIFFERS"


class ContractStatus(str, Enum):
    """How well a dataset agrees with a contract, given the tables observed.

    The evidence is deliberately asymmetric:

    ``INCOMPATIBLE``  at least one observed table deviates from the contract. That
                      is positive evidence: a file written by that exact version
                      cannot deviate, so the dataset is not from this version.
    ``COMPATIBLE``    every observed table matches, i.e. nothing contradicts the
                      contract. This is *evidence of consistency only*, not proof
                      of the dataset's version: tables that were not observed
                      (Synthea can legitimately omit files) were never checked.
    ``UNKNOWN``       no table was observed, so there is no evidence either way.
    """

    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class HeaderMatch:
    """Comparison of one table's actual header against the contract."""

    table: str
    expected: tuple[str, ...]
    actual: tuple[str, ...]
    kind: MatchKind
    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    misplaced: tuple[str, ...] = ()
    duplicated: tuple[str, ...] = ()

    @property
    def is_exact(self) -> bool:
        return self.kind is MatchKind.EXACT

    def describe(self) -> str:
        """One-line human summary, used in reports and assessment reasons."""
        if self.is_exact:
            return f"{self.table}: header matches the contract ({len(self.expected)} columns)"

        problems: list[str] = []
        if self.missing:
            problems.append(f"missing {list(self.missing)}")
        if self.extra:
            problems.append(f"unexpected {list(self.extra)}")
        if self.duplicated:
            problems.append(f"duplicated {list(self.duplicated)}")
        if self.misplaced:
            problems.append(f"out of order {list(self.misplaced)}")
        return f"{self.table}: " + "; ".join(problems)


@dataclass(frozen=True, slots=True)
class ContractAssessment:
    """Result of comparing every supplied header against one contract."""

    contract_id: str
    matches: tuple[HeaderMatch, ...]
    #: Size of the contract, kept so that reports can state how much was observed.
    contract_table_count: int

    @property
    def status(self) -> ContractStatus:
        """Verdict for the tables that were actually observed.

        A deviation is conclusive (``INCOMPATIBLE``); agreement is not
        (``COMPATIBLE`` only, never "this dataset is version X"); observing
        nothing yields ``UNKNOWN``.
        """
        if not self.matches:
            return ContractStatus.UNKNOWN
        if all(match.is_exact for match in self.matches):
            return ContractStatus.COMPATIBLE
        return ContractStatus.INCOMPATIBLE

    @property
    def tables_assessed(self) -> tuple[str, ...]:
        return tuple(match.table for match in self.matches)

    @property
    def non_matching(self) -> tuple[HeaderMatch, ...]:
        return tuple(match for match in self.matches if not match.is_exact)

    @property
    def summary(self) -> str:
        """One-line statement of what was checked, without over-claiming."""
        observed = len(self.matches)
        if observed == 0:
            return (
                f"no known tables were observed, so contract '{self.contract_id}' "
                "cannot be assessed"
            )
        scope = (
            f"all {observed} known tables of the dataset were observed (the contract "
            f"has {self.contract_table_count})"
            if observed == self.contract_table_count
            else f"{observed} of the {self.contract_table_count} contract tables were observed"
        )
        if self.status is ContractStatus.COMPATIBLE:
            return f"{scope}; every observed table matches contract '{self.contract_id}'"
        return (
            f"{scope}; {len(self.non_matching)} do not match contract "
            f"'{self.contract_id}'"
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        """Human-readable explanations when the dataset is not ``COMPATIBLE``."""
        if self.status is ContractStatus.COMPATIBLE:
            return ()
        if self.status is ContractStatus.UNKNOWN:
            return (
                "no known tables were found in the dataset, so no contract can be "
                "assessed for it",
            )
        return tuple(match.describe() for match in self.non_matching)


@dataclass(frozen=True, slots=True)
class SchemaContract:
    """The column contract of one Synthea version, plus its provenance."""

    contract_id: str
    tables: tuple[TableSpec, ...]
    source_repository: str
    source_commit: str
    source_file: str

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.tables)

    def table(self, name: str) -> TableSpec:
        """Return one table of the contract.

        :raises KeyError: the table is not part of this contract (the caller asked
            about something Synthea does not export).
        """
        for spec in self.tables:
            if spec.name == name:
                return spec
        raise KeyError(f"contract '{self.contract_id}' has no table {name!r}")

    def match_header(self, table: str, actual: Sequence[str]) -> HeaderMatch:
        """Compare the header read from a CSV file with the contract for ``table``."""
        spec = self.table(table)
        expected = spec.columns
        actual_tuple = tuple(actual)

        missing = _multiset_difference(expected, actual_tuple)
        extra = _multiset_difference(actual_tuple, expected)
        duplicated = tuple(sorted({c for c in actual_tuple if actual_tuple.count(c) > 1}))

        if actual_tuple == expected:
            kind = MatchKind.EXACT
        elif missing or extra or duplicated:
            kind = MatchKind.COLUMNS_DIFFERENT
        else:
            kind = MatchKind.ORDER_DIFFERS

        misplaced = ()
        if kind is MatchKind.ORDER_DIFFERS:
            misplaced = tuple(
                column for index, column in enumerate(actual_tuple) if expected[index] != column
            )

        return HeaderMatch(
            table=table,
            expected=expected,
            actual=actual_tuple,
            kind=kind,
            missing=missing,
            extra=extra,
            misplaced=misplaced,
            duplicated=duplicated,
        )

    def assess(self, headers_by_table: Mapping[str, Sequence[str]]) -> ContractAssessment:
        """Compare several headers at once.

        Tables absent from ``headers_by_table`` are simply not assessed: a missing
        table is legitimate in a Synthea export and says nothing about whether the
        dataset matches the contract.
        """
        unknown = sorted(set(headers_by_table) - set(self.table_names))
        if unknown:
            raise ValueError(
                f"headers were given for tables that are not part of contract "
                f"'{self.contract_id}': {unknown}"
            )
        matches = tuple(self.match_header(name, headers_by_table[name]) for name in sorted(headers_by_table))
        return ContractAssessment(
            contract_id=self.contract_id,
            matches=matches,
            contract_table_count=len(self.tables),
        )


def _multiset_difference(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    """Columns present in ``left`` more often than in ``right``, ordered as in ``left``."""
    remaining = Counter(right)
    result: list[str] = []
    for column in left:
        if remaining[column] > 0:
            remaining[column] -= 1
        elif column not in result:
            result.append(column)
    return tuple(result)


#: The only schema version implemented today: Synthea CSV as of 2026-08.
CURRENT_CONTRACT = SchemaContract(
    contract_id=CONTRACT_ID,
    tables=SYNTHEA_TABLES,
    source_repository=SOURCE_REPOSITORY,
    source_commit=SOURCE_COMMIT,
    source_file=SOURCE_FILE,
)


def assess_contract(
    headers_by_table: Mapping[str, Sequence[str]],
    *,
    contract: SchemaContract = CURRENT_CONTRACT,
) -> ContractAssessment:
    """Assess headers against ``contract`` (the current one by default)."""
    return contract.assess(headers_by_table)
