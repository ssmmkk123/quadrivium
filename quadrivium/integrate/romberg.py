"""Romberg integration and Richardson-accelerated quadrature."""

from __future__ import annotations

from .. import numeric as np

from ..core.types import QuadratureResult
from ..core.utils import CountedFunction

__all__ = ["romberg", "romberg_table", "richardson_quadrature", "euler_maclaurin"]


def romberg_table(f, a: float, b: float, levels: int = 10, tol: float = 1e-12):
    """Full Romberg tableau of trapezoid estimates and their extrapolations."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    R = np.zeros((levels, levels))
    h = b - a
    R[0, 0] = 0.5 * h * (fc(a) + fc(b))
    used = 1
    for i in range(1, levels):
        h /= 2.0
        # reuse previous points: only the new midpoints are evaluated
        total = sum(fc(a + (2 * k - 1) * h) for k in range(1, 2 ** (i - 1) + 1))
        R[i, 0] = 0.5 * R[i - 1, 0] + h * total
        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (4**j - 1)
        used = i + 1
        if i > 1 and abs(R[i, i] - R[i - 1, i - 1]) < tol * max(1.0, abs(R[i, i])):
            break
    return R[:used, :used], fc.calls


def romberg(f, a: float, b: float, levels: int = 12, tol: float = 1e-12):
    """Romberg integration: Richardson extrapolation of the trapezoid rule.

    Each column of the tableau raises the order by two, so column ``j`` is
    ``O(h^(2j+2))`` accurate.
    """
    R, calls = romberg_table(f, a, b, levels, tol)
    n = R.shape[0]
    err = abs(R[n - 1, n - 1] - R[n - 2, n - 2]) if n > 1 else None
    return QuadratureResult(float(R[n - 1, n - 1]), err, calls, 2 ** (n - 1),
                            err is not None and err < tol * max(1.0, abs(R[n - 1, n - 1])),
                            "romberg")


def richardson_quadrature(rule, f, a: float, b: float, n: int = 8, levels: int = 5,
                          order: int = 2):
    """Richardson-extrapolate any composite rule by repeated halving of ``h``."""
    T = np.zeros((levels, levels))
    for i in range(levels):
        T[i, 0] = rule(f, a, b, n * 2**i).value
        for j in range(1, i + 1):
            p = 2.0 ** (order + 2 * (j - 1))
            T[i, j] = (p * T[i, j - 1] - T[i - 1, j - 1]) / (p - 1.0)
    err = abs(T[levels - 1, levels - 1] - T[levels - 2, levels - 2]) if levels > 1 else None
    return QuadratureResult(float(T[levels - 1, levels - 1]), err, 0, n * 2 ** (levels - 1),
                            True, "richardson_quadrature")


def euler_maclaurin(f, a: float, b: float, n: int = 100, terms: int = 2, df=None):
    """Euler-Maclaurin corrected trapezoid rule.

    Adds derivative corrections at the endpoints; ``terms`` counts Bernoulli
    corrections (each adds two orders of accuracy).
    """
    from ..diff.finite import central_difference
    from .newton_cotes import trapezoid_rule

    base = trapezoid_rule(f, a, b, n)
    h = (b - a) / n
    value = base.value
    bern = [1.0 / 12.0, -1.0 / 720.0, 1.0 / 30240.0]
    for k in range(min(terms, len(bern))):
        order = 2 * k + 1
        if df is not None and k == 0:
            da, db = df(a), df(b)
        else:
            da = central_difference(f, a, 1e-3, order=1) if order == 1 else \
                central_difference(f, a, 1e-2, order=3)
            db = central_difference(f, b, 1e-3, order=1) if order == 1 else \
                central_difference(f, b, 1e-2, order=3)
        value -= bern[k] * h ** (2 * k + 2) * (db - da)
    return QuadratureResult(float(value), None, base.function_calls + 4 * terms, n,
                            True, "euler_maclaurin")
