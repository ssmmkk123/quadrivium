"""One-dimensional minimization."""

from __future__ import annotations

from ._history import History, monitor

from .. import numeric as np

from ..core.exceptions import BracketError
from ..core.types import OptimizeResult
from ..core.utils import CountedFunction, numerical_derivative

__all__ = [
    "golden_section",
    "fibonacci_search",
    "ternary_search",
    "parabolic_interpolation",
    "brent_minimize",
    "newton_minimize_1d",
    "bracket_minimum",
    "line_minimize",
]

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def bracket_minimum(f, a: float = 0.0, b: float = 1.0, growth: float = 1.618,
                    max_iter: int = 100):
    """Expand outward until three points bracket a minimum."""
    fa, fb = f(a), f(b)
    if fb > fa:
        a, b, fa, fb = b, a, fb, fa
    c = b + growth * (b - a)
    fc = f(c)
    for _ in range(max_iter):
        if fc > fb:
            return (a, b, c) if a < c else (c, b, a)
        a, fa, b, fb = b, fb, c, fc
        c = b + growth * (b - a)
        fc = f(c)
    raise BracketError("failed to bracket a minimum")


def golden_section(f, a: float, b: float, tol: float = 1e-10, max_iter: int = 500):
    """Golden section search: reduces the interval by ``0.618`` each step.

    Needs no derivatives and is guaranteed for a unimodal function.
    """
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    f_c, f_d = fc(c), fc(d)
    for k in range(1, max_iter + 1):
        if abs(b - a) < tol:
            break
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - _GOLDEN * (b - a)
            f_c = fc(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + _GOLDEN * (b - a)
            f_d = fc(d)
    xm = 0.5 * (a + b)
    return OptimizeResult(xm, float(fc(xm)), None, None, k, abs(b - a) < tol,
                          fc.calls, 0, "golden_section")


def fibonacci_search(f, a: float, b: float, n: int = 40):
    """Fibonacci search: the optimal fixed-budget interval reduction."""
    fc = CountedFunction(f)
    fib = [1, 1]
    while len(fib) < n + 2:
        fib.append(fib[-1] + fib[-2])
    a, b = float(a), float(b)
    c = a + (b - a) * fib[n - 1] / fib[n + 1]
    d = a + (b - a) * fib[n] / fib[n + 1]
    f_c, f_d = fc(c), fc(d)
    for k in range(1, n - 1):
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = a + (b - a) * fib[n - k - 1] / fib[n - k + 1]
            f_c = fc(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + (b - a) * fib[n - k] / fib[n - k + 1]
            f_d = fc(d)
    xm = 0.5 * (a + b)
    return OptimizeResult(xm, float(fc(xm)), None, None, n, True, fc.calls, 0,
                          "fibonacci")


def ternary_search(f, a: float, b: float, tol: float = 1e-10, max_iter: int = 500):
    """Ternary search: split into thirds, discard one. Simple but slower than golden."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    for k in range(1, max_iter + 1):
        if abs(b - a) < tol:
            break
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if fc(m1) < fc(m2):
            b = m2
        else:
            a = m1
    xm = 0.5 * (a + b)
    return OptimizeResult(xm, float(fc(xm)), None, None, k, abs(b - a) < tol,
                          fc.calls, 0, "ternary")


def parabolic_interpolation(f, a: float, b: float, c=None, tol: float = 1e-10,
                            max_iter: int = 100):
    """Successive parabolic interpolation through three points."""
    fc = CountedFunction(f)
    if c is None:
        a, b, c = float(a), 0.5 * (a + b), float(b)
    x0, x1, x2 = float(a), float(b), float(c)
    f0, f1, f2 = fc(x0), fc(x1), fc(x2)
    xm = x1
    for k in range(1, max_iter + 1):
        denom = (x1 - x0) * (f1 - f2) - (x1 - x2) * (f1 - f0)
        if abs(denom) < 1e-300:
            break
        xm = x1 - 0.5 * (((x1 - x0) ** 2 * (f1 - f2) - (x1 - x2) ** 2 * (f1 - f0))
                         / denom)
        fm = fc(xm)
        if abs(xm - x1) < tol:
            return OptimizeResult(xm, float(fm), None, None, k, True, fc.calls, 0,
                                  "parabolic")
        x0, f0, x1, f1, x2, f2 = x1, f1, xm, fm, x2, f2
    return OptimizeResult(xm, float(fc(xm)), None, None, max_iter, False, fc.calls,
                          0, "parabolic")


def brent_minimize(f, a: float, b: float, tol: float = 1e-10, max_iter: int = 200):
    """Brent's method: parabolic interpolation with a golden section fallback.

    The standard derivative-free 1-D minimizer.
    """
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    x = w = v = a + (1 - _GOLDEN) * (b - a)
    fx = fw = fv = fc(x)
    d = e = 0.0
    for k in range(1, max_iter + 1):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-14
        tol2 = 2 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            return OptimizeResult(x, float(fx), None, None, k, True, fc.calls, 0,
                                  "brent")
        use_golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            e_prev, e = e, d
            if abs(p) < abs(0.5 * q * e_prev) and q * (a - x) < p < q * (b - x):
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if xm > x else -tol1
                use_golden = False
        if use_golden:
            e = (b - x) if x < xm else (a - x)
            d = (1 - _GOLDEN) * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = fc(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return OptimizeResult(x, float(fx), None, None, max_iter, False, fc.calls, 0,
                          "brent")


def newton_minimize_1d(f, x0: float, df=None, d2f=None, tol: float = 1e-12,
                       max_iter: int = 100):
    """Newton's method applied to ``f'(x) = 0``, with a curvature safeguard."""
    fc = CountedFunction(f)
    x = float(x0)
    history = History([x])
    for k in range(1, max_iter + 1):
        g = df(x) if df is not None else numerical_derivative(fc, x, order=1)
        h = d2f(x) if d2f is not None else numerical_derivative(fc, x, order=2)
        if abs(g) < tol:
            return OptimizeResult(x, float(fc(x)), g, h, k, True, fc.calls, 0,
                                  "newton_1d", history)
        if h <= 0:
            h = abs(h) if h != 0 else 1.0  # force a descent step
        step = g / h
        x -= step
        history.append(x)
        if abs(step) < tol * max(1.0, abs(x)):
            return OptimizeResult(x, float(fc(x)), g, h, k, True, fc.calls, 0,
                                  "newton_1d", history)
    return OptimizeResult(x, float(fc(x)), None, None, max_iter, False, fc.calls,
                          0, "newton_1d", history)


def line_minimize(f, a=None, b=None, method: str = "brent", **kwargs):
    """Minimize a scalar function, bracketing automatically when needed."""
    if a is None or b is None:
        a, m, b = bracket_minimum(f, 0.0, 1.0)
    return {"brent": brent_minimize, "golden": golden_section,
            "ternary": ternary_search,
            "parabolic": parabolic_interpolation}[method](f, a, b, **kwargs)


# Apply a common context-local output policy to public iterative entry points.
for _name in ['newton_minimize_1d']:
    globals()[_name] = monitor(globals()[_name])
