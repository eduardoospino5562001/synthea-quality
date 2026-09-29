"""[INCIDENCE] Exact (Garwood) confidence interval for a Poisson count, standard library only.

For ``k`` observed events the two-sided ``1 − alpha`` interval for the expected count is
``[λ_low, λ_high]`` with

* ``P(X ≥ k; λ_low) = alpha / 2``  (``λ_low = 0`` when ``k = 0``),
* ``P(X ≤ k; λ_high) = alpha / 2``.

These are the chi-square quantiles ``χ²(alpha/2, 2k)/2`` and ``χ²(1 − alpha/2, 2k+2)/2``,
found here by bisection on the Poisson cumulative distribution computed in log space, so
no statistics library is needed. The interval is "exact" in the sense that its coverage is
never below the nominal level, which matters for the small counts a synthetic population
of a few hundred people produces.
"""

from __future__ import annotations

import math

#: Two-sided level of the intervals the tool reports.
ALPHA = 0.05
_ITERATIONS = 200


def poisson_cdf(k: int, lam: float) -> float:
    """``P(X ≤ k)`` for ``X ~ Poisson(lam)``."""
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0
    log_terms = [i * math.log(lam) - lam - math.lgamma(i + 1) for i in range(k + 1)]
    peak = max(log_terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in log_terms))


def exact_interval(k: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Exact two-sided ``1 − alpha`` interval for the mean of a Poisson count ``k``."""
    if k < 0:
        raise ValueError("a count cannot be negative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    half = alpha / 2
    low = 0.0 if k == 0 else _solve(lambda lam: 1.0 - poisson_cdf(k - 1, lam) - half, k)
    high = _solve(lambda lam: half - poisson_cdf(k, lam), k)
    return low, high


def _solve(increasing, k: int) -> float:
    """Root of a function of λ that increases from negative to positive."""
    lo, hi = 0.0, max(10.0, 2.0 * k + 10.0)
    while increasing(hi) < 0:
        hi *= 2
    for _ in range(_ITERATIONS):
        mid = (lo + hi) / 2
        if increasing(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, hi):
            break
    return (lo + hi) / 2
