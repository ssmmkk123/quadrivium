"""Monte Carlo and quasi-Monte Carlo integration.

Convergence is ``O(N^-1/2)`` regardless of dimension, which is why these are
the methods of choice once the dimension is large; the variance reduction
techniques here buy back a substantial constant factor.
"""

from __future__ import annotations

import numpy as np

from ..core.types import QuadratureResult
from ..core.utils import as_vector

__all__ = [
    "monte_carlo",
    "monte_carlo_nd",
    "stratified_sampling",
    "importance_sampling",
    "control_variates",
    "antithetic_variates",
    "quasi_monte_carlo",
    "halton_sequence",
    "sobol_sequence",
    "latin_hypercube",
    "hit_or_miss",
    "vegas_lite",
]


def monte_carlo(f, a: float, b: float, n: int = 100000, rng=None):
    """Plain Monte Carlo on an interval, with a standard error estimate."""
    rng = np.random.default_rng(rng)
    x = rng.uniform(a, b, n)
    y = np.array([f(xi) for xi in x])
    vol = b - a
    mean = float(np.mean(y))
    err = float(vol * np.std(y, ddof=1) / np.sqrt(n))
    return QuadratureResult(vol * mean, err, n, 1, True, "monte_carlo")


def monte_carlo_nd(f, lows, highs, n: int = 100000, rng=None):
    """Plain Monte Carlo over a box in any dimension."""
    rng = np.random.default_rng(rng)
    lows, highs = as_vector(lows), as_vector(highs)
    d = lows.size
    pts = rng.uniform(lows, highs, size=(n, d))
    y = np.array([f(p) for p in pts])
    vol = float(np.prod(highs - lows))
    err = float(vol * np.std(y, ddof=1) / np.sqrt(n))
    return QuadratureResult(vol * float(np.mean(y)), err, n, 1, True, "monte_carlo_nd")


def stratified_sampling(f, a: float, b: float, n: int = 10000, strata: int = 100,
                        rng=None):
    """Stratified Monte Carlo: sample uniformly within equal sub-intervals.

    Variance never exceeds plain Monte Carlo and is usually much lower.
    """
    rng = np.random.default_rng(rng)
    per = max(n // strata, 1)
    edges = np.linspace(a, b, strata + 1)
    total = 0.0
    var = 0.0
    for i in range(strata):
        x = rng.uniform(edges[i], edges[i + 1], per)
        y = np.array([f(xi) for xi in x])
        w = edges[i + 1] - edges[i]
        total += w * np.mean(y)
        var += w * w * np.var(y, ddof=1) / per if per > 1 else 0.0
    return QuadratureResult(float(total), float(np.sqrt(var)), per * strata, strata,
                            True, "stratified")


def importance_sampling(f, sampler, pdf, n: int = 100000, rng=None):
    """Importance sampling: draw from ``sampler`` and reweight by ``f/pdf``.

    Variance collapses when ``pdf`` is close to proportional to ``|f|``.
    """
    rng = np.random.default_rng(rng)
    x = sampler(n, rng)
    x = np.atleast_1d(x)
    w = np.array([f(xi) / pdf(xi) for xi in x])
    mean = float(np.mean(w))
    err = float(np.std(w, ddof=1) / np.sqrt(n))
    return QuadratureResult(mean, err, n, 1, True, "importance_sampling")


def control_variates(f, g, g_mean: float, a: float, b: float, n: int = 100000,
                     rng=None):
    """Control variates: subtract a correlated function with known mean."""
    rng = np.random.default_rng(rng)
    x = rng.uniform(a, b, n)
    fy = np.array([f(xi) for xi in x])
    gy = np.array([g(xi) for xi in x])
    cov = np.cov(fy, gy, ddof=1)
    c = -cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    vol = b - a
    adjusted = fy + c * (gy - g_mean / vol)
    mean = float(np.mean(adjusted))
    err = float(vol * np.std(adjusted, ddof=1) / np.sqrt(n))
    return QuadratureResult(vol * mean, err, n, 1, True, "control_variates")


def antithetic_variates(f, a: float, b: float, n: int = 100000, rng=None):
    """Antithetic variates: pair each sample with its mirror image."""
    rng = np.random.default_rng(rng)
    m = n // 2
    u = rng.uniform(0.0, 1.0, m)
    x1 = a + (b - a) * u
    x2 = a + (b - a) * (1.0 - u)
    y = np.array([0.5 * (f(p) + f(q)) for p, q in zip(x1, x2)])
    vol = b - a
    err = float(vol * np.std(y, ddof=1) / np.sqrt(m))
    return QuadratureResult(vol * float(np.mean(y)), err, 2 * m, 1, True, "antithetic")


def _van_der_corput(n: int, base: int):
    """``n``-th element of the van der Corput sequence in the given base."""
    q, bk = 0.0, 1.0 / base
    while n:
        n, rem = divmod(n, base)
        q += rem * bk
        bk /= base
    return q


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97, 101]


def halton_sequence(n: int, dim: int = 1, skip: int = 1):
    """Halton low-discrepancy sequence, one prime base per dimension."""
    if dim > len(_PRIMES):
        raise ValueError(f"Halton supports up to {len(_PRIMES)} dimensions here")
    out = np.empty((n, dim))
    for j in range(dim):
        base = _PRIMES[j]
        for i in range(n):
            out[i, j] = _van_der_corput(i + skip, base)
    return out[:, 0] if dim == 1 else out


def sobol_sequence(n: int, dim: int = 1):
    """Sobol-style low-discrepancy sequence.

    Uses direction numbers from primitive polynomials for the first few
    dimensions and falls back to Halton beyond them.
    """
    max_dim = 6
    if dim > max_dim:
        return halton_sequence(n, dim)
    bits = max(int(np.ceil(np.log2(max(n, 2)))) + 1, 2)
    # primitive polynomial degrees and initial direction numbers
    poly = [1, 3, 7, 11, 13, 19]
    minit = [[1], [1, 1], [1, 3, 7], [1, 1, 5], [1, 3, 1, 1], [1, 1, 3, 7]]
    V = np.zeros((dim, bits + 1), dtype=np.int64)
    for j in range(dim):
        deg = int(np.floor(np.log2(poly[j]))) if j > 0 else 0
        m = list(minit[j])
        if j == 0:
            for i in range(1, bits + 1):
                V[j, i] = 1 << (bits - i)
        else:
            for i in range(1, min(deg, bits) + 1):
                V[j, i] = m[i - 1] << (bits - i)
            for i in range(deg + 1, bits + 1):
                val = V[j, i - deg] ^ (V[j, i - deg] >> deg)
                for k in range(1, deg):
                    if (poly[j] >> (deg - k)) & 1:
                        val ^= V[j, i - k]
                V[j, i] = val
    out = np.zeros((n, dim))
    X = np.zeros(dim, dtype=np.int64)
    for i in range(1, n + 1):
        c = 1
        v = i - 1
        while v & 1:
            v >>= 1
            c += 1
        for j in range(dim):
            X[j] ^= V[j, min(c, bits)]
            out[i - 1, j] = X[j] / float(1 << bits)
    return out[:, 0] if dim == 1 else out


def latin_hypercube(n: int, dim: int = 1, rng=None):
    """Latin hypercube sample: one point per stratum in every dimension."""
    rng = np.random.default_rng(rng)
    out = np.empty((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.uniform(size=n)) / n
    return out[:, 0] if dim == 1 else out


def quasi_monte_carlo(f, lows, highs, n: int = 4096, sequence: str = "halton"):
    """Quasi-Monte Carlo: low-discrepancy points give ``O((log N)^d / N)``.

    Substantially better than plain Monte Carlo for smooth integrands in
    moderate dimension.
    """
    lows, highs = as_vector(lows), as_vector(highs)
    d = lows.size
    if sequence == "halton":
        u = halton_sequence(n, d)
    elif sequence == "sobol":
        u = sobol_sequence(n, d)
    elif sequence == "lhs":
        u = latin_hypercube(n, d)
    else:
        raise ValueError("sequence must be 'halton', 'sobol' or 'lhs'")
    u = np.atleast_2d(u.reshape(n, d))
    pts = lows + u * (highs - lows)
    y = np.array([f(p if d > 1 else p[0]) for p in pts])
    vol = float(np.prod(highs - lows))
    return QuadratureResult(vol * float(np.mean(y)), None, n, 1, True,
                            f"qmc_{sequence}")


def hit_or_miss(f, a: float, b: float, ymax: float, n: int = 100000, rng=None):
    """Hit-or-miss Monte Carlo: the area under the curve by rejection counting."""
    rng = np.random.default_rng(rng)
    x = rng.uniform(a, b, n)
    y = rng.uniform(0.0, ymax, n)
    hits = np.array([y[i] <= f(x[i]) for i in range(n)])
    p = float(np.mean(hits))
    area = (b - a) * ymax
    err = area * np.sqrt(max(p * (1 - p), 0.0) / n)
    return QuadratureResult(area * p, float(err), n, 1, True, "hit_or_miss")


def vegas_lite(f, a: float, b: float, n: int = 20000, iterations: int = 5,
               bins: int = 50, rng=None):
    """Simplified VEGAS: adapt a piecewise-constant sampling density to ``|f|``.

    Captures the essential idea of importance sampling on a learned grid.
    """
    rng = np.random.default_rng(rng)
    edges = np.linspace(a, b, bins + 1)
    estimate = 0.0
    err = None
    for _ in range(iterations):
        widths = np.diff(edges)
        per = max(n // bins, 2)
        contrib = np.zeros(bins)
        variances = np.zeros(bins)
        for i in range(bins):
            xs = rng.uniform(edges[i], edges[i + 1], per)
            ys = np.array([f(x) for x in xs])
            contrib[i] = widths[i] * np.mean(ys)
            variances[i] = widths[i] ** 2 * np.var(ys, ddof=1) / per
        estimate = float(np.sum(contrib))
        err = float(np.sqrt(np.sum(variances)))
        # refine: put more bins where |contribution| is large
        weight = np.abs(contrib) + 1e-12
        cdf = np.concatenate([[0.0], np.cumsum(weight)])
        cdf /= cdf[-1]
        targets = np.linspace(0.0, 1.0, bins + 1)
        edges = np.interp(targets, cdf, edges)
    return QuadratureResult(estimate, err, n * iterations, bins, True, "vegas_lite")
