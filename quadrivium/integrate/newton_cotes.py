"""Newton-Cotes quadrature: the interpolatory rules on equispaced nodes."""

from __future__ import annotations

from .. import numeric as np

from ..core.types import QuadratureResult
from ..core.utils import CountedFunction, as_vector

__all__ = [
    "rectangle_rule",
    "midpoint_rule",
    "trapezoid_rule",
    "simpson_rule",
    "simpson38_rule",
    "boole_rule",
    "newton_cotes_weights",
    "newton_cotes",
    "composite_trapezoid",
    "composite_simpson",
    "composite_midpoint",
    "trapezoid_data",
    "simpson_data",
    "cumulative_trapezoid",
    "corrected_trapezoid",
]


def rectangle_rule(f, a: float, b: float, n: int = 100, side: str = "left"):
    """Riemann sum with left, right or midpoint sampling."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    h = (b - a) / n
    if side == "left":
        x = a + h * np.arange(n)
    elif side == "right":
        x = a + h * np.arange(1, n + 1)
    elif side == "mid":
        x = a + h * (np.arange(n) + 0.5)
    else:
        raise ValueError("side must be 'left', 'right' or 'mid'")
    total = h * np.sum([fc(xi) for xi in x])
    return QuadratureResult(float(total), None, fc.calls, n, True, f"rectangle_{side}")


def midpoint_rule(f, a: float, b: float, n: int = 100):
    """Composite midpoint rule; second-order accurate and open."""
    res = rectangle_rule(f, a, b, n, "mid")
    res.method = "midpoint"
    return res


def trapezoid_rule(f, a: float, b: float, n: int = 100):
    """Composite trapezoid rule, error ``O(h^2)``."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    x = np.linspace(a, b, n + 1)
    y = np.array([fc(xi) for xi in x])
    h = (b - a) / n
    total = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return QuadratureResult(float(total), None, fc.calls, n, True, "trapezoid")


def simpson_rule(f, a: float, b: float, n: int = 100):
    """Composite Simpson's 1/3 rule, error ``O(h^4)``; ``n`` must be even."""
    if n % 2:
        n += 1
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    x = np.linspace(a, b, n + 1)
    y = np.array([fc(xi) for xi in x])
    h = (b - a) / n
    total = h / 3.0 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return QuadratureResult(float(total), None, fc.calls, n, True, "simpson")


def simpson38_rule(f, a: float, b: float, n: int = 99):
    """Composite Simpson's 3/8 rule; ``n`` must be a multiple of 3."""
    n = n + (3 - n % 3) % 3
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    x = np.linspace(a, b, n + 1)
    y = np.array([fc(xi) for xi in x])
    h = (b - a) / n
    total = 0.0
    for i in range(0, n, 3):
        total += 3 * h / 8 * (y[i] + 3 * y[i + 1] + 3 * y[i + 2] + y[i + 3])
    return QuadratureResult(float(total), None, fc.calls, n, True, "simpson38")


def boole_rule(f, a: float, b: float, n: int = 100):
    """Composite Boole's rule, error ``O(h^6)``; ``n`` must be a multiple of 4."""
    n = n + (4 - n % 4) % 4
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    x = np.linspace(a, b, n + 1)
    y = np.array([fc(xi) for xi in x])
    h = (b - a) / n
    total = 0.0
    for i in range(0, n, 4):
        total += 2 * h / 45 * (7 * y[i] + 32 * y[i + 1] + 12 * y[i + 2]
                               + 32 * y[i + 3] + 7 * y[i + 4])
    return QuadratureResult(float(total), None, fc.calls, n, True, "boole")


def newton_cotes_weights(n: int, closed: bool = True):
    """Weights of the degree-``n`` Newton-Cotes rule on ``[0, 1]``.

    Computed by exactly integrating the Lagrange basis. Note that closed rules
    develop negative weights for ``n >= 8`` and become unstable.
    """
    if closed:
        x = np.linspace(0.0, 1.0, n + 1)
    else:
        x = (np.arange(1, n + 2)) / (n + 2.0)
    w = np.zeros(x.size)
    for i in range(x.size):
        others = np.delete(x, i)
        coeffs = np.poly(others) if others.size else np.array([1.0])
        integral = np.polyval(np.polyint(coeffs), 1.0) - np.polyval(np.polyint(coeffs), 0.0)
        denom = np.prod(x[i] - others) if others.size else 1.0
        w[i] = integral / denom
    return x, w


def newton_cotes(f, a: float, b: float, n: int = 4, closed: bool = True):
    """Single-panel Newton-Cotes rule of arbitrary degree ``n``."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    t, w = newton_cotes_weights(n, closed)
    x = a + (b - a) * t
    y = np.array([fc(xi) for xi in x])
    return QuadratureResult(float((b - a) * (w @ y)), None, fc.calls, 1, True,
                            f"newton_cotes_{n}")


def composite_trapezoid(f, a, b, n=100):
    """Alias of :func:`trapezoid_rule` under its composite name."""
    return trapezoid_rule(f, a, b, n)


def composite_simpson(f, a, b, n=100):
    """Alias of :func:`simpson_rule` under its composite name."""
    return simpson_rule(f, a, b, n)


def composite_midpoint(f, a, b, n=100):
    """Alias of :func:`midpoint_rule` under its composite name."""
    return midpoint_rule(f, a, b, n)


def trapezoid_data(x, y) -> float:
    """Trapezoid rule applied to tabulated data (non-uniform spacing allowed)."""
    x, y = as_vector(x), as_vector(y)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def simpson_data(x, y) -> float:
    """Simpson's rule on tabulated data; falls back to trapezoid on the last
    interval when the number of intervals is odd."""
    x, y = as_vector(x), as_vector(y)
    n = x.size - 1
    total = 0.0
    i = 0
    while i + 2 <= n:
        h1 = x[i + 1] - x[i]
        h2 = x[i + 2] - x[i + 1]
        h = h1 + h2
        total += (h / 6.0) * ((2 - h2 / h1) * y[i] + (h**2 / (h1 * h2)) * y[i + 1]
                              + (2 - h1 / h2) * y[i + 2])
        i += 2
    if i < n:
        total += (x[n] - x[n - 1]) * (y[n] + y[n - 1]) / 2.0
    return float(total)


def cumulative_trapezoid(x, y, initial: float = 0.0):
    """Running integral of tabulated data by the trapezoid rule."""
    x, y = as_vector(x), as_vector(y)
    inc = np.diff(x) * (y[:-1] + y[1:]) / 2.0
    return np.concatenate([[initial], initial + np.cumsum(inc)])


def corrected_trapezoid(f, a: float, b: float, n: int = 100, df=None):
    """Euler-Maclaurin corrected trapezoid rule.

    Adding the endpoint derivative term raises the order from 2 to 4.
    """
    from ..diff.finite import central_difference

    base = trapezoid_rule(f, a, b, n)
    h = (b - a) / n
    dfa = df(a) if df is not None else central_difference(f, a, 1e-5)
    dfb = df(b) if df is not None else central_difference(f, b, 1e-5)
    value = base.value - h**2 / 12.0 * (dfb - dfa)
    return QuadratureResult(float(value), None, base.function_calls + 4, n, True,
                            "corrected_trapezoid")
