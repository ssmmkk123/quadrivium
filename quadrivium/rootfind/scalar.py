"""Root finding for scalar equations ``f(x) = 0``.

Bracketing methods (guaranteed but linear), open methods (fast but local), and
the hybrids that combine both.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import BracketError, ConvergenceError
from ..core.types import RootResult
from ..core.utils import CountedFunction, numerical_derivative

__all__ = [
    "bisection",
    "false_position",
    "illinois",
    "pegasus",
    "ridders",
    "brent",
    "secant",
    "newton",
    "halley",
    "steffensen",
    "muller",
    "fixed_point",
    "aitken_accelerated",
    "inverse_quadratic",
    "chebyshev_method",
    "bracket_root",
    "find_all_roots",
    "itp",
]


def _check_bracket(f, a, b):
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return fa, fb, a
    if fb == 0.0:
        return fa, fb, b
    if np.sign(fa) == np.sign(fb):
        raise BracketError(
            f"f(a)={fa:.6g} and f(b)={fb:.6g} have the same sign: [{a}, {b}] "
            "does not bracket a root"
        )
    return fa, fb, None


def bisection(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200):
    """Interval halving. Always converges; gains exactly one bit per iteration."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    fa, fb, exact = _check_bracket(fc, a, b)
    if exact is not None:
        return RootResult(exact, 0.0, 0, True, fc.calls, "bisection", [], "exact root at endpoint")
    history = []
    c = a
    for k in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fcv = fc(c)
        history.append(c)
        if fcv == 0.0 or (b - a) / 2 < tol:
            return RootResult(c, fcv, k, True, fc.calls, "bisection", history, "converged")
        if np.sign(fcv) == np.sign(fa):
            a, fa = c, fcv
        else:
            b, fb = c, fcv
    return RootResult(c, fc(c), max_iter, False, fc.calls, "bisection", history,
                      "maximum iterations reached")


def false_position(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200):
    """Regula falsi: secant through the bracket endpoints."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    fa, fb, exact = _check_bracket(fc, a, b)
    if exact is not None:
        return RootResult(exact, 0.0, 0, True, fc.calls, "false_position", [], "exact root at endpoint")
    history = []
    c = a
    for k in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fcv = fc(c)
        history.append(c)
        if abs(fcv) < tol or abs(b - a) < tol:
            return RootResult(c, fcv, k, True, fc.calls, "false_position", history, "converged")
        if np.sign(fcv) == np.sign(fa):
            a, fa = c, fcv
        else:
            b, fb = c, fcv
    return RootResult(c, fc(c), max_iter, False, fc.calls, "false_position", history,
                      "maximum iterations reached")


def _modified_regula(f, a, b, tol, max_iter, rule, name):
    """Shared driver for the Illinois and Pegasus stalling fixes.

    Both are regula falsi with one change.  When the same endpoint is retained
    twice running -- the stall that drags plain false position down to linear
    convergence -- the retained endpoint's function value is scaled down, so
    the next secant leans back towards the other side.

    The scaling belongs on the retained step only.  Applying it on every step
    damps the good regula-falsi steps as well, and the iteration degenerates
    to linear convergence with ratio 1/2, no better than bisection.
    """
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    fa, fb, exact = _check_bracket(fc, a, b)
    if exact is not None:
        return RootResult(exact, 0.0, 0, True, fc.calls, name, [], "exact root at endpoint")
    history = []
    c = a
    for k in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fcv = fc(c)
        history.append(c)
        if abs(fcv) < tol or abs(b - a) < tol:
            return RootResult(c, fcv, k, True, fc.calls, name, history, "converged")
        if np.sign(fcv) != np.sign(fb):
            a, fa = b, fb           # the root moved to the other side: shift the bracket
        else:
            fa = rule(fa, fb, fcv)  # ``a`` retained again: damp its value
        b, fb = c, fcv
    return RootResult(c, fc(c), max_iter, False, fc.calls, name, history,
                      "maximum iterations reached")


def illinois(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200):
    """Illinois variant: halve the retained endpoint's value to stop stalling."""
    return _modified_regula(f, a, b, tol, max_iter,
                            lambda f_kept, f_other, f_new: f_kept * 0.5, "illinois")


def pegasus(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200):
    """Pegasus variant: scale by ``f_c/(f_c+f_new)``; faster than Illinois."""
    return _modified_regula(
        f, a, b, tol, max_iter,
        lambda f_kept, f_other, f_new: (
            f_kept * f_other / (f_other + f_new)
            if (f_other + f_new) != 0 else f_kept * 0.5),
        "pegasus",
    )


def ridders(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200):
    """Ridders' method: exponential correction, quadratic convergence, bracketed."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    fa, fb, exact = _check_bracket(fc, a, b)
    if exact is not None:
        return RootResult(exact, 0.0, 0, True, fc.calls, "ridders", [], "exact root at endpoint")
    history = []
    x = a
    for k in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fcv = fc(c)
        d = fcv * fcv - fa * fb
        if d <= 0:
            return RootResult(c, fcv, k, True, fc.calls, "ridders", history, "converged")
        x = c + (c - a) * np.sign(fa - fb) * fcv / np.sqrt(d)
        fx = fc(x)
        history.append(x)
        if abs(fx) < tol or abs(b - a) < tol:
            return RootResult(x, fx, k, True, fc.calls, "ridders", history, "converged")
        if np.sign(fcv) != np.sign(fx):
            a, fa, b, fb = c, fcv, x, fx
        elif np.sign(fa) != np.sign(fx):
            b, fb = x, fx
        else:
            a, fa = x, fx
    return RootResult(x, fc(x), max_iter, False, fc.calls, "ridders", history,
                      "maximum iterations reached")


def brent(f, a: float, b: float, tol: float = 1e-14, max_iter: int = 200):
    """Brent's method: inverse quadratic interpolation with a bisection fallback.

    The standard general-purpose bracketed solver -- superlinear in practice,
    never slower than bisection in the worst case.
    """
    fc_ = CountedFunction(f)
    a, b = float(a), float(b)
    fa, fb, exact = _check_bracket(fc_, a, b)
    if exact is not None:
        return RootResult(exact, 0.0, 0, True, fc_.calls, "brent", [], "exact root at endpoint")
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc_val = a, fa
    d = e = b - a
    history = []
    for k in range(1, max_iter + 1):
        if fb != 0 and np.sign(fb) == np.sign(fc_val):
            c, fc_val = a, fa
            d = e = b - a
        if abs(fc_val) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc_val = fb, fc_val, fb
        tol_act = 2 * np.finfo(float).eps * abs(b) + 0.5 * tol
        m = 0.5 * (c - b)
        history.append(b)
        if abs(m) <= tol_act or fb == 0.0:
            return RootResult(b, fb, k, True, fc_.calls, "brent", history, "converged")
        if abs(e) < tol_act or abs(fa) <= abs(fb):
            d = e = m  # bisection step
        else:
            s = fb / fa
            if a == c:
                p, q = 2 * m * s, 1 - s  # secant
            else:
                q_, r_ = fa / fc_val, fb / fc_val  # inverse quadratic
                p = s * (2 * m * q_ * (q_ - r_) - (b - a) * (r_ - 1))
                q = (q_ - 1) * (r_ - 1) * (s - 1)
            if p > 0:
                q = -q
            p = abs(p)
            if 2 * p < min(3 * m * q - abs(tol_act * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        a, fa = b, fb
        b = b + (d if abs(d) > tol_act else np.sign(m) * tol_act)
        fb = fc_(b)
    return RootResult(b, fb, max_iter, False, fc_.calls, "brent", history,
                      "maximum iterations reached")


def itp(f, a: float, b: float, tol: float = 1e-12, max_iter: int = 200,
        k1: float = 0.1, k2: float = 2.0, n0: int = 1):
    """ITP (Interpolate-Truncate-Project): superlinear with a bisection guarantee."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    fa, fb, exact = _check_bracket(fc, a, b)
    if exact is not None:
        return RootResult(exact, 0.0, 0, True, fc.calls, "itp", [], "exact root at endpoint")
    n_half = int(np.ceil(np.log2((b - a) / (2 * tol)))) if b - a > 2 * tol else 0
    n_max = n_half + n0
    history = []
    j = 0
    x = 0.5 * (a + b)
    while (b - a) > 2 * tol and j < max_iter:
        x_half = 0.5 * (a + b)
        r = max(tol * 2 ** (n_max - j) - 0.5 * (b - a), 0.0)
        delta = k1 * (b - a) ** k2
        x_f = (b * fa - a * fb) / (fa - fb)
        sigma = np.sign(x_half - x_f)
        x_t = x_f + sigma * delta if delta <= abs(x_half - x_f) else x_half
        x = x_t if abs(x_t - x_half) <= r else x_half - sigma * r
        fx = fc(x)
        history.append(x)
        if fx == 0.0:
            return RootResult(x, fx, j + 1, True, fc.calls, "itp", history, "converged")
        if np.sign(fx) == np.sign(fa):
            a, fa = x, fx
        else:
            b, fb = x, fx
        j += 1
    root = 0.5 * (a + b)
    return RootResult(root, fc(root), j, True, fc.calls, "itp", history, "converged")


def secant(f, x0: float, x1=None, tol: float = 1e-12, max_iter: int = 200):
    """Secant method: derivative-free, order ~1.618."""
    fc = CountedFunction(f)
    x0 = float(x0)
    x1 = float(x1) if x1 is not None else x0 + (1e-4 * max(abs(x0), 1.0))
    f0, f1 = fc(x0), fc(x1)
    history = [x0, x1]
    for k in range(1, max_iter + 1):
        if f1 == f0:
            return RootResult(x1, f1, k, abs(f1) < tol, fc.calls, "secant", history,
                              "zero denominator: f(x0) == f(x1)")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        f2 = fc(x2)
        if abs(x2 - x1) < tol * max(1.0, abs(x2)) or abs(f2) < tol:
            return RootResult(x2, f2, k, True, fc.calls, "secant", history, "converged")
        x0, f0, x1, f1 = x1, f1, x2, f2
    return RootResult(x1, f1, max_iter, False, fc.calls, "secant", history,
                      "maximum iterations reached")


def newton(f, x0: float, df=None, tol: float = 1e-12, max_iter: int = 200,
           damping: float = 1.0, multiplicity: int = 1):
    """Newton-Raphson, optionally damped and corrected for a known multiplicity.

    ``df=None`` falls back to a central difference derivative.
    """
    fc = CountedFunction(f)
    dfc = CountedFunction(df) if df is not None else None
    x = float(x0)
    history = [x]
    for k in range(1, max_iter + 1):
        fx = fc(x)
        if abs(fx) < tol:
            return RootResult(x, fx, k - 1, True, fc.calls, "newton", history, "converged")
        d = dfc(x) if dfc is not None else numerical_derivative(fc, x)
        if d == 0.0:
            return RootResult(x, fx, k, False, fc.calls, "newton", history,
                              "zero derivative encountered")
        step = damping * multiplicity * fx / d
        x_new = x - step
        history.append(x_new)
        if abs(step) < tol * max(1.0, abs(x_new)):
            return RootResult(x_new, fc(x_new), k, True, fc.calls, "newton", history, "converged")
        x = x_new
    return RootResult(x, fc(x), max_iter, False, fc.calls, "newton", history,
                      "maximum iterations reached")


def halley(f, x0: float, df=None, d2f=None, tol: float = 1e-12, max_iter: int = 100):
    """Halley's method: cubic convergence using the second derivative."""
    fc = CountedFunction(f)
    x = float(x0)
    history = [x]
    for k in range(1, max_iter + 1):
        fx = fc(x)
        if abs(fx) < tol:
            return RootResult(x, fx, k - 1, True, fc.calls, "halley", history, "converged")
        d1 = df(x) if df is not None else numerical_derivative(fc, x, order=1)
        d2 = d2f(x) if d2f is not None else numerical_derivative(fc, x, order=2)
        denom = 2 * d1 * d1 - fx * d2
        if abs(denom) < 1e-300:
            return RootResult(x, fx, k, False, fc.calls, "halley", history,
                              "zero denominator")
        x_new = x - 2 * fx * d1 / denom
        history.append(x_new)
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return RootResult(x_new, fc(x_new), k, True, fc.calls, "halley", history, "converged")
        x = x_new
    return RootResult(x, fc(x), max_iter, False, fc.calls, "halley", history,
                      "maximum iterations reached")


def chebyshev_method(f, x0: float, df=None, d2f=None, tol: float = 1e-12,
                     max_iter: int = 100):
    """Chebyshev's third-order method (Newton plus a curvature correction)."""
    fc = CountedFunction(f)
    x = float(x0)
    history = [x]
    for k in range(1, max_iter + 1):
        fx = fc(x)
        if abs(fx) < tol:
            return RootResult(x, fx, k - 1, True, fc.calls, "chebyshev", history, "converged")
        d1 = df(x) if df is not None else numerical_derivative(fc, x, order=1)
        d2 = d2f(x) if d2f is not None else numerical_derivative(fc, x, order=2)
        if d1 == 0.0:
            return RootResult(x, fx, k, False, fc.calls, "chebyshev", history, "zero derivative")
        u = fx / d1
        x_new = x - u - 0.5 * d2 * u * u / d1
        history.append(x_new)
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return RootResult(x_new, fc(x_new), k, True, fc.calls, "chebyshev", history, "converged")
        x = x_new
    return RootResult(x, fc(x), max_iter, False, fc.calls, "chebyshev", history,
                      "maximum iterations reached")


def steffensen(f, x0: float, tol: float = 1e-12, max_iter: int = 200):
    """Steffensen's method: Newton-like quadratic order without a derivative."""
    fc = CountedFunction(f)
    x = float(x0)
    history = [x]
    for k in range(1, max_iter + 1):
        fx = fc(x)
        if abs(fx) < tol:
            return RootResult(x, fx, k - 1, True, fc.calls, "steffensen", history, "converged")
        g = fc(x + fx) - fx
        if abs(g) < 1e-300:
            return RootResult(x, fx, k, False, fc.calls, "steffensen", history,
                              "zero denominator")
        x_new = x - fx * fx / g
        history.append(x_new)
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return RootResult(x_new, fc(x_new), k, True, fc.calls, "steffensen", history, "converged")
        x = x_new
    return RootResult(x, fc(x), max_iter, False, fc.calls, "steffensen", history,
                      "maximum iterations reached")


def muller(f, x0: float, x1=None, x2=None, tol: float = 1e-12, max_iter: int = 200):
    """Muller's method: parabolic interpolation; finds complex roots naturally."""
    fc = CountedFunction(f)
    x0 = complex(x0)
    x1 = complex(x1) if x1 is not None else x0 + 0.5
    x2 = complex(x2) if x2 is not None else x0 + 1.0
    history = []
    for k in range(1, max_iter + 1):
        f0, f1, f2 = fc(x0), fc(x1), fc(x2)
        h1, h2 = x1 - x0, x2 - x1
        if h1 == 0 or h2 == 0 or (h2 + h1) == 0:
            break
        d1, d2 = (f1 - f0) / h1, (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        disc = np.sqrt(complex(b * b - 4 * a * f2))
        denom = b + disc if abs(b + disc) > abs(b - disc) else b - disc
        if abs(denom) < 1e-300:
            break
        x3 = x2 - 2 * f2 / denom
        history.append(x3)
        if abs(x3 - x2) < tol * max(1.0, abs(x3)):
            root = x3.real if abs(x3.imag) < 1e-12 else x3
            return RootResult(root, fc(x3), k, True, fc.calls, "muller", history, "converged")
        x0, x1, x2 = x1, x2, x3
    root = x2.real if abs(x2.imag) < 1e-12 else x2
    return RootResult(root, fc(x2), max_iter, False, fc.calls, "muller", history,
                      "maximum iterations reached")


def inverse_quadratic(f, x0: float, x1: float, x2: float, tol: float = 1e-12,
                      max_iter: int = 200):
    """Inverse quadratic interpolation through three iterates."""
    fc = CountedFunction(f)
    x0, x1, x2 = float(x0), float(x1), float(x2)
    history = []
    for k in range(1, max_iter + 1):
        f0, f1, f2 = fc(x0), fc(x1), fc(x2)
        if len({f0, f1, f2}) < 3:
            break
        x3 = (x0 * f1 * f2 / ((f0 - f1) * (f0 - f2))
              + x1 * f0 * f2 / ((f1 - f0) * (f1 - f2))
              + x2 * f0 * f1 / ((f2 - f0) * (f2 - f1)))
        history.append(x3)
        if abs(x3 - x2) < tol * max(1.0, abs(x3)):
            return RootResult(x3, fc(x3), k, True, fc.calls, "inverse_quadratic",
                              history, "converged")
        x0, x1, x2 = x1, x2, x3
    return RootResult(x2, fc(x2), max_iter, False, fc.calls, "inverse_quadratic",
                      history, "maximum iterations reached")


def fixed_point(g, x0: float, tol: float = 1e-12, max_iter: int = 1000,
                relaxation: float = 1.0):
    """Fixed point iteration ``x <- g(x)``, optionally relaxed.

    Converges when ``|g'(x*)| < 1``; relaxation can create that condition.
    """
    gc = CountedFunction(g)
    x = float(x0)
    history = [x]
    for k in range(1, max_iter + 1):
        gx = gc(x)
        x_new = (1 - relaxation) * x + relaxation * gx
        history.append(x_new)
        if not np.isfinite(x_new):
            return RootResult(x, np.nan, k, False, gc.calls, "fixed_point", history,
                              "iteration diverged")
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return RootResult(x_new, x_new - gc(x_new), k, True, gc.calls,
                              "fixed_point", history, "converged")
        x = x_new
    return RootResult(x, x - gc(x), max_iter, False, gc.calls, "fixed_point", history,
                      "maximum iterations reached")


def aitken_accelerated(g, x0: float, tol: float = 1e-12, max_iter: int = 200):
    """Aitken's delta-squared acceleration of a fixed point iteration."""
    gc = CountedFunction(g)
    x = float(x0)
    history = [x]
    for k in range(1, max_iter + 1):
        x1 = gc(x)
        x2 = gc(x1)
        denom = x2 - 2 * x1 + x
        x_new = x2 if abs(denom) < 1e-300 else x - (x1 - x) ** 2 / denom
        history.append(x_new)
        if abs(x_new - x) < tol * max(1.0, abs(x_new)):
            return RootResult(x_new, x_new - gc(x_new), k, True, gc.calls,
                              "aitken", history, "converged")
        x = x_new
    return RootResult(x, x - gc(x), max_iter, False, gc.calls, "aitken", history,
                      "maximum iterations reached")


def bracket_root(f, a: float, b: float, factor: float = 1.6, max_iter: int = 60):
    """Expand ``[a, b]`` outward until it brackets a sign change."""
    a, b = float(a), float(b)
    if a == b:
        b = a + 1.0
    fa, fb = f(a), f(b)
    for _ in range(max_iter):
        if np.sign(fa) != np.sign(fb):
            return a, b
        if abs(fa) < abs(fb):
            a += factor * (a - b)
            fa = f(a)
        else:
            b += factor * (b - a)
            fb = f(b)
    raise BracketError("failed to bracket a root by outward expansion")


def find_all_roots(f, a: float, b: float, n: int = 200, tol: float = 1e-12,
                   method=brent):
    """Scan ``[a, b]`` on a grid and refine every sign change found.

    Roots separated by less than ``(b-a)/n`` may be missed; increase ``n`` for
    oscillatory functions.
    """
    xs = np.linspace(float(a), float(b), int(n) + 1)
    fs = np.array([f(x) for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        if fs[i] == 0.0:
            roots.append(xs[i])
        elif np.sign(fs[i]) != np.sign(fs[i + 1]):
            res = method(f, xs[i], xs[i + 1], tol=tol)
            if res.converged:
                roots.append(float(res.root))
    if fs[-1] == 0.0:
        roots.append(xs[-1])
    out = []
    for r in sorted(roots):
        if not out or abs(r - out[-1]) > 1e-8 * max(1.0, abs(r)):
            out.append(r)
    return np.array(out)
