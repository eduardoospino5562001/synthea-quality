"""[PROFILE] What a reader needs to know about a top-N cut.

A top-N list is ordered by a count and then by a deterministic tie-break. When the cut
falls between two different counts it is meaningful; when it falls inside a group of
equal counts, only the tie-break decides which members of that group make the list. The
profile never hides that: :func:`tie_at_cut` finds such a group so the section can say
so.
"""

from __future__ import annotations

from typing import Sequence


def tie_at_cut(shown: Sequence[int], rest: Sequence[int]) -> dict[str, int] | None:
    """The tie that straddles the cut, or ``None`` when the cut falls between counts.

    :param shown: the counts that made the list, in list order.
    :param rest: the counts that did not, in the same order.
    :returns: ``{"count": c, "values": n, "listed": k}`` — ``n`` entries share the count
        ``c`` at the cut and ``k`` of them are listed.
    """
    if not shown or not rest or shown[-1] != rest[0]:
        return None
    count = shown[-1]
    listed = sum(1 for c in shown if c == count)
    unlisted = sum(1 for c in rest if c == count)
    return {"count": count, "values": listed + unlisted, "listed": listed}
