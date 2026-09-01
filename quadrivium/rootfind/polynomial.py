"""Polynomial evaluation, deflation, and root finding.

Coefficients are ordered from the highest degree down, matching
``numpy.polyval``: ``[a_n, ..., a_1, a_0]`` represents
``a_n x^n + ... + a_1 x + a_0``.
"""

from __future__ import annotations

import numpy as np

from ..core.types import RootResult
from ..core.utils import as_vector

__all__ = [
    "horner",
    "horner_derivative",
    "synthetic_division",
    "deflate",
    "durand_kerner",
    "aberth_ehrlich",
    "bairstow",
    "laguerre_root",
    "companion_roots",
    "jenkins_traub_like",
    "polynomial_roots",
    "sturm_chain",
    "count_real_roots",
    "root_bounds",
    "newton_polynomial",
]


def horner(coeffs, x):
    """Evaluate a polynomial by Horner's rule (``n`` multiplications).

    Works for real or complex coefficients and for scalar or array ``x``.
    """
    c = np.asarray(coeffs)
    xa = np.asarray(x)
    dtype = np.result_type(c.dtype, xa.dtype, np.float64)
    result = np.full(xa.shape, c[0], dtype=dtype)
    for a in c[1:]:
        result = result * xa + a
    return result[()] if xa.ndim == 0 else result


def horner_derivative(coeffs, x):
    """Evaluate ``(p(x), p'(x))`` in a single pass of synthetic division."""
    c = np.asarray(coeffs)
    xa = np.asarray(x)
    dtype = np.result_type(c.dtype, xa.dtype, np.float64)
    p = np.full(xa.shape, c[0], dtype=dtype)
    dp = np.zeros(xa.shape, dtype=dtype)
    for a in c[1:]:
        dp = dp * xa + p
        p = p * xa + a
    if xa.ndim == 0:
        return p[()], dp[()]
    return p, dp


def synthetic_division(coeffs, r):
    """Divide by ``(x - r)``; returns ``(quotient, remainder)``."""
    c = np.asarray(coeffs)
    dtype = np.result_type(c.dtype, np.asarray(r).dtype, np.float64)
    c = c.astype(dtype)
    q = np.zeros(len(c) - 1, dtype=dtype)
    acc = c[0]
    for i in range(1, len(c)):
        q[i - 1] = acc
        acc = c[i] + acc * r
    return q, acc


def deflate(coeffs, root):
    """Remove a known root, returning the deflated coefficient array."""
    q, _ = synthetic_division(coeffs, root)
    return q


def root_bounds(coeffs):
    """Cauchy and Fujiwara bounds on the moduli of all roots."""
    c = np.asarray(coeffs, dtype=float)
    if c[0] == 0:
        raise ValueError("leading coefficient must be nonzero")
    a = c[1:] / c[0]
    n = a.size
    cauchy = 1.0 + np.max(np.abs(a)) if n else 1.0
    terms = [abs(a[i]) ** (1.0 / (i + 1)) for i in range(n)]
    if terms:
        terms[-1] = (abs(a[-1]) / 2.0) ** (1.0 / n) if n else 0.0
    fujiwara = 2.0 * max(terms) if terms else 0.0
    lower = 0.0
    if n and a[-1] != 0:
        lower = 1.0 / (1.0 + np.max(np.abs(c[:-1] / c[-1])))
    return {"cauchy": float(cauchy), "fujiwara": float(fujiwara), "lower": float(lower)}


def durand_kerner(coeffs, tol: float = 1e-14, max_iter: int = 500):
    """Durand-Kerner (Weierstrass): all roots at once by simultaneous iteration."""
    c = np.asarray(coeffs, dtype=complex)
    c = c / c[0]
    n = len(c) - 1
    if n < 1:
        return np.array([])
    radius = 1.0 + float(np.max(np.abs(c[1:]))) if n else 1.0
    z = radius * np.exp(2j * np.pi * np.arange(n) / n) * np.exp(0.5j)
    for k in range(max_iter):
        z_old = z.copy()
        for i in range(n):
            denom = np.prod([z[i] - z[j] for j in range(n) if j != i]) if n > 1 else 1.0
            if abs(denom) < 1e-300:
                continue
            z[i] = z[i] - horner(c, z[i]) / denom
        if np.max(np.abs(z - z_old)) < tol:
            break
    return _clean_roots(z)


def aberth_ehrlich(coeffs, tol: float = 1e-14, max_iter: int = 500):
    """Aberth-Ehrlich: cubically convergent simultaneous root iteration."""
    c = np.asarray(coeffs, dtype=complex)
    c = c / c[0]
    n = len(c) - 1
    if n < 1:
        return np.array([])
    dc = c[:-1] * np.arange(n, 0, -1)
    radius = 1.0 + float(np.max(np.abs(c[1:])))
    z = radius * np.exp(2j * np.pi * (np.arange(n) + 0.5) / n)
    for k in range(max_iter):
        z_old = z.copy()
        for i in range(n):
            p = horner(c, z[i])
            dp = horner(dc, z[i]) if n > 1 else dc[0] if dc.size else 1.0
            if abs(dp) < 1e-300:
                continue
            ratio = p / dp
            s = sum(1.0 / (z[i] - z[j]) for j in range(n) if j != i and abs(z[i] - z[j]) > 1e-300)
            denom = 1.0 - ratio * s
            if abs(denom) < 1e-300:
                continue
            z[i] = z[i] - ratio / denom
        if np.max(np.abs(z - z_old)) < tol:
            break
    return _clean_roots(z)


def laguerre_root(coeffs, x0=0.0, tol: float = 1e-14, max_iter: int = 200):
    """Laguerre's method for a single polynomial root.

    Cubically convergent and near-globally convergent for polynomials with real
    roots. Named ``laguerre_root`` to keep the bare name ``laguerre`` for the
    Laguerre *polynomial* in :mod:`numethods.approx.orthopoly`.
    """
    c = np.asarray(coeffs, dtype=complex)
    n = len(c) - 1
    dc = c[:-1] * np.arange(n, 0, -1)
    d2c = dc[:-1] * np.arange(n - 1, 0, -1) if n > 1 else np.array([0j])
    x = complex(x0)
    for k in range(max_iter):
        p = horner(c, x)
        if abs(p) < tol:
            break
        G = horner(dc, x) / p
        H = G * G - horner(d2c, x) / p
        disc = np.sqrt((n - 1) * (n * H - G * G) + 0j)
        d1, d2 = G + disc, G - disc
        denom = d1 if abs(d1) >= abs(d2) else d2
        if abs(denom) < 1e-300:
            break
        a = n / denom
        x = x - a
        if abs(a) < tol * max(1.0, abs(x)):
            break
    return x.real if abs(x.imag) < 1e-12 else x


def bairstow(coeffs, r: float = 1.0, s: float = 1.0, tol: float = 1e-14,
             max_iter: int = 500):
    """Bairstow's method: extracts real quadratic factors, so complex pairs
    are found without leaving real arithmetic."""
    a = np.asarray(coeffs, dtype=float)
    a = a / a[0]
    roots = []
    while len(a) > 3:
        n = len(a) - 1
        for _ in range(max_iter):
            b = np.zeros(n + 1)
            c = np.zeros(n + 1)
            b[0] = a[0]
            b[1] = a[1] + r * b[0]
            for i in range(2, n + 1):
                b[i] = a[i] + r * b[i - 1] + s * b[i - 2]
            c[0] = b[0]
            c[1] = b[1] + r * c[0]
            for i in range(2, n):
                c[i] = b[i] + r * c[i - 1] + s * c[i - 2]
            det = c[n - 2] * c[n - 2] - c[n - 3] * c[n - 1] if n >= 3 else c[n - 2] ** 2
            if abs(det) < 1e-300:
                r += 0.5
                s += 0.5
                continue
            if n >= 3:
                dr = (-b[n - 1] * c[n - 2] + b[n] * c[n - 3]) / det
                ds = (-b[n] * c[n - 2] + b[n - 1] * c[n - 1]) / det
            else:
                dr = -b[n - 1] / c[n - 2]
                ds = -b[n] / c[n - 2]
            r += dr
            s += ds
            if abs(dr) < tol and abs(ds) < tol:
                break
        disc = r * r + 4 * s
        if disc >= 0:
            roots.extend([(r + np.sqrt(disc)) / 2, (r - np.sqrt(disc)) / 2])
        else:
            roots.extend([complex(r / 2, np.sqrt(-disc) / 2), complex(r / 2, -np.sqrt(-disc) / 2)])
        a = b[: n - 1]
    if len(a) == 3:
        disc = a[1] ** 2 - 4 * a[0] * a[2]
        if disc >= 0:
            roots.extend([(-a[1] + np.sqrt(disc)) / (2 * a[0]),
                          (-a[1] - np.sqrt(disc)) / (2 * a[0])])
        else:
            roots.extend([complex(-a[1] / (2 * a[0]), np.sqrt(-disc) / (2 * a[0])),
                          complex(-a[1] / (2 * a[0]), -np.sqrt(-disc) / (2 * a[0]))])
    elif len(a) == 2:
        roots.append(-a[1] / a[0])
    return _clean_roots(np.array(roots, dtype=complex))


def companion_roots(coeffs):
    """Roots as eigenvalues of the companion matrix (the standard approach)."""
    c = np.asarray(coeffs, dtype=float)
    c = c[np.argmax(c != 0):] if np.any(c != 0) else c
    n = len(c) - 1
    if n < 1:
        return np.array([])
    C = np.zeros((n, n))
    C[0, :] = -c[1:] / c[0]
    C[1:, :-1] = np.eye(n - 1)
    return _clean_roots(np.linalg.eigvals(C))


def jenkins_traub_like(coeffs, tol: float = 1e-14):
    """Deflation-based solver: Laguerre for one root, deflate, repeat.

    Shares the shape of Jenkins-Traub (find, deflate, polish) without the
    three-stage shift strategy.
    """
    c = np.asarray(coeffs, dtype=complex)
    n = len(c) - 1
    roots = []
    work = c.copy()
    for _ in range(n):
        if len(work) <= 2:
            if len(work) == 2:
                roots.append(-work[1] / work[0])
            break
        r = laguerre_root(work, 0.0, tol=tol)
        roots.append(r)
        work = deflate(work, r)
    # polish against the original polynomial
    polished = [complex(laguerre_root(c, r, tol=tol)) for r in roots]
    return _clean_roots(np.array(polished, dtype=complex))


def polynomial_roots(coeffs, method: str = "companion"):
    """Find all roots. ``method`` selects the algorithm."""
    return {
        "companion": companion_roots,
        "durand_kerner": durand_kerner,
        "aberth": aberth_ehrlich,
        "bairstow": bairstow,
        "jenkins_traub": jenkins_traub_like,
    }[method](coeffs)


def newton_polynomial(coeffs, x0: float, tol: float = 1e-14, max_iter: int = 200):
    """Newton's method specialised to polynomials via Horner's rule."""
    x = float(x0)
    for k in range(1, max_iter + 1):
        p, dp = horner_derivative(coeffs, x)
        if abs(dp) < 1e-300:
            return RootResult(x, p, k, False, k, "newton_polynomial", [], "zero derivative")
        step = p / dp
        x -= step
        if abs(step) < tol * max(1.0, abs(x)):
            return RootResult(x, horner(coeffs, x), k, True, k, "newton_polynomial", [], "converged")
    return RootResult(x, horner(coeffs, x), max_iter, False, max_iter,
                      "newton_polynomial", [], "maximum iterations reached")


def sturm_chain(coeffs):
    """Sturm chain of a squarefree polynomial, for isolating real roots."""
    p = np.trim_zeros(np.asarray(coeffs, dtype=float), "f")
    n = len(p) - 1
    dp = p[:-1] * np.arange(n, 0, -1)
    chain = [p, dp]
    while len(chain[-1]) > 1:
        a, b = chain[-2], chain[-1]
        _, rem = np.polydiv(a, b)
        rem = np.trim_zeros(np.atleast_1d(rem), "f")
        if rem.size == 0 or np.all(np.abs(rem) < 1e-300):
            break
        chain.append(-rem)
    return chain


def count_real_roots(coeffs, a: float, b: float) -> int:
    """Number of distinct real roots in ``(a, b]`` by Sturm's theorem."""
    chain = sturm_chain(coeffs)

    def sign_changes(x):
        vals = [horner(c, x) for c in chain]
        vals = [v for v in vals if abs(v) > 1e-300]
        return sum(1 for i in range(len(vals) - 1) if np.sign(vals[i]) != np.sign(vals[i + 1]))

    return sign_changes(a) - sign_changes(b)


def _clean_roots(z):
    """Sort roots and drop negligible imaginary parts."""
    z = np.asarray(z, dtype=complex)
    if z.size == 0:
        return z
    # Round the real part before sorting so a conjugate pair whose real parts
    # differ only by rounding noise still comes out in a stable order.
    scale = max(1.0, float(np.max(np.abs(z))))
    key_real = np.round(z.real / scale, 9) * scale
    z = z[np.lexsort((z.imag, key_real))]
    if np.all(np.abs(z.imag) < 1e-9 * np.maximum(1.0, np.abs(z.real))):
        return z.real.copy()
    return z
