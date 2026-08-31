"""Derivative-free optimization: search using function values alone.

The right tools when derivatives are unavailable, expensive, or the objective
is noisy or discontinuous.
"""

from __future__ import annotations

import numpy as np

from ..core.types import OptimizeResult
from ..core.utils import CountedFunction, as_vector

__all__ = [
    "line_minimize_1d",
    "nelder_mead",
    "powell",
    "hooke_jeeves",
    "coordinate_descent",
    "pattern_search",
    "compass_search",
    "cyclic_coordinate",
]


def line_minimize_1d(fun, tol: float = 1e-13, start: float = 0.0,
                     initial_step: float = 1.0):
    """Minimize a 1-D function, bracketing outward from ``start`` first.

    A fixed bracket is not safe: Brent's method only guarantees a minimum for a
    unimodal function, so on a multi-modal restriction it can settle in the
    wrong basin. Bracketing from the current point keeps the search local and
    descending.
    """
    from .scalar import bracket_minimum, brent_minimize
    from ..core.exceptions import BracketError

    try:
        a, _, c = bracket_minimum(fun, start, start + initial_step)
    except BracketError:
        a, c = start - initial_step, start + initial_step
    lo, hi = (a, c) if a < c else (c, a)
    return brent_minimize(fun, lo, hi, tol=tol)


def nelder_mead(f, x0, tol: float = 1e-12, max_iter: int = 5000, step: float = 0.1,
                alpha: float = 1.0, gamma: float = 2.0, rho: float = 0.5,
                sigma: float = 0.5, initial_simplex=None):
    """Nelder-Mead simplex: reflect, expand, contract, shrink.

    The most widely used derivative-free method; robust on low-dimensional
    problems but with no convergence guarantee in general.
    """
    fc = CountedFunction(lambda v: float(f(v)))
    x0 = as_vector(x0)
    n = x0.size
    if initial_simplex is not None:
        simplex = np.array(initial_simplex, dtype=float)
    else:
        simplex = np.vstack([x0] + [x0 + step * (abs(x0[i]) + 1.0) * np.eye(n)[i]
                                    for i in range(n)])
    fs = np.array([fc(p) for p in simplex])
    history = []
    for k in range(1, max_iter + 1):
        order = np.argsort(fs)
        simplex, fs = simplex[order], fs[order]
        history.append(simplex[0].copy())
        if np.max(np.abs(simplex[1:] - simplex[0])) < tol and (fs[-1] - fs[0]) < tol:
            return OptimizeResult(simplex[0], float(fs[0]), None, None, k, True,
                                  fc.calls, 0, "nelder_mead", history, "converged")
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])          # reflection
        fr = fc(xr)
        if fs[0] <= fr < fs[-2]:
            simplex[-1], fs[-1] = xr, fr
            continue
        if fr < fs[0]:                                            # expansion
            xe = centroid + gamma * (xr - centroid)
            fe = fc(xe)
            simplex[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        # contraction (outside if the reflection helped, inside otherwise)
        if fr < fs[-1]:
            xc = centroid + rho * (xr - centroid)
            fcv = fc(xc)
            if fcv <= fr:
                simplex[-1], fs[-1] = xc, fcv
                continue
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fcv = fc(xc)
            if fcv < fs[-1]:
                simplex[-1], fs[-1] = xc, fcv
                continue
        simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])  # shrink
        fs[1:] = np.array([fc(p) for p in simplex[1:]])
    order = np.argsort(fs)
    return OptimizeResult(simplex[order][0], float(fs[order][0]), None, None,
                          max_iter, False, fc.calls, 0, "nelder_mead", history,
                          "maximum iterations reached")


def powell(f, x0, tol: float = 1e-10, max_iter: int = 400, line_tol: float = 1e-13):
    """Powell's conjugate direction method.

    Minimizes along a set of directions and replaces one each cycle, building
    conjugate directions without derivatives.
    """
    fc = CountedFunction(lambda v: float(f(v)))
    x = as_vector(x0).copy()
    n = x.size
    directions = np.eye(n)
    fx = fc(x)
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        x_start = x.copy()
        f_start = fx
        biggest_drop = 0.0
        i_big = 0
        for i in range(n):
            d = directions[i]
            res = line_minimize_1d(lambda a: fc(x + a * d), tol=line_tol)
            drop = fx - res.fun
            if drop > biggest_drop:
                biggest_drop, i_big = drop, i
            x = x + res.x * d
            fx = res.fun
        history.append(x.copy())
        if 2.0 * (f_start - fx) <= tol * (abs(f_start) + abs(fx) + 1e-30):
            return OptimizeResult(x, float(fx), None, None, k, True, fc.calls, 0,
                                  "powell", history, "converged")
        # extrapolated point and Powell's replacement test
        x_ext = 2 * x - x_start
        f_ext = fc(x_ext)
        if f_ext < f_start:
            t = (2 * (f_start - 2 * fx + f_ext) * (f_start - fx - biggest_drop) ** 2
                 - biggest_drop * (f_start - f_ext) ** 2)
            if t < 0:
                new_dir = x - x_start
                res = line_minimize_1d(lambda a: fc(x + a * new_dir), tol=line_tol)
                x = x + res.x * new_dir
                fx = res.fun
                directions[i_big] = directions[n - 1]
                directions[n - 1] = new_dir / (np.linalg.norm(new_dir) + 1e-300)
    return OptimizeResult(x, float(fx), None, None, max_iter, False, fc.calls, 0,
                          "powell", history, "maximum iterations reached")


def hooke_jeeves(f, x0, step: float = 0.5, tol: float = 1e-12, max_iter: int = 10000,
                 shrink: float = 0.5):
    """Hooke-Jeeves pattern search: exploratory moves plus a pattern move."""
    fc = CountedFunction(lambda v: float(f(v)))
    x = as_vector(x0).copy()
    n = x.size
    fx = fc(x)
    h = step
    history = [x.copy()]

    def explore(base, fbase, h):
        y = base.copy()
        fy = fbase
        for i in range(n):
            for s in (h, -h):
                trial = y.copy()
                trial[i] += s
                ft = fc(trial)
                if ft < fy:
                    y, fy = trial, ft
                    break
        return y, fy

    for k in range(1, max_iter + 1):
        if h < tol:
            return OptimizeResult(x, float(fx), None, None, k, True, fc.calls, 0,
                                  "hooke_jeeves", history, "converged")
        y, fy = explore(x, fx, h)
        if fy < fx:
            while True:                              # pattern move
                x_new = 2 * y - x
                x, fx = y, fy
                history.append(x.copy())
                y2, fy2 = explore(x_new, fc(x_new), h)
                if fy2 >= fx:
                    break
                y, fy = y2, fy2
        else:
            h *= shrink
    return OptimizeResult(x, float(fx), None, None, max_iter, False, fc.calls, 0,
                          "hooke_jeeves", history, "maximum iterations reached")


def compass_search(f, x0, step: float = 0.5, tol: float = 1e-12,
                   max_iter: int = 20000, shrink: float = 0.5,
                   expand: float = 2.0):
    """Compass (coordinate) search: poll the ``2n`` axis directions.

    A generating set search with a convergence guarantee for smooth functions.
    """
    fc = CountedFunction(lambda v: float(f(v)))
    x = as_vector(x0).copy()
    n = x.size
    fx = fc(x)
    h = step
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        if h < tol:
            return OptimizeResult(x, float(fx), None, None, k, True, fc.calls, 0,
                                  "compass_search", history, "converged")
        improved = False
        for i in range(n):
            for s in (h, -h):
                trial = x.copy()
                trial[i] += s
                ft = fc(trial)
                if ft < fx:
                    x, fx = trial, ft
                    improved = True
                    break
            if improved:
                break
        if improved:
            h = min(h * expand, step)
            history.append(x.copy())
        else:
            h *= shrink
    return OptimizeResult(x, float(fx), None, None, max_iter, False, fc.calls, 0,
                          "compass_search", history, "maximum iterations reached")


def pattern_search(f, x0, **kwargs):
    """Generalized pattern search (alias of :func:`compass_search`)."""
    res = compass_search(f, x0, **kwargs)
    res.method = "pattern_search"
    return res


def coordinate_descent(f, x0, tol: float = 1e-10, max_iter: int = 1000,
                       bracket: float = 10.0):
    """Coordinate descent: exactly minimize one variable at a time.

    Excellent when the variables are weakly coupled and cheap per sweep, but it
    zigzags badly along a curved narrow valley -- Rosenbrock needs thousands of
    sweeps where Powell needs a handful. Prefer :func:`powell`, which builds
    conjugate directions from the same kind of line searches, when the
    variables interact strongly.
    """
    fc = CountedFunction(lambda v: float(f(v)))
    x = as_vector(x0).copy()
    n = x.size
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            def g(t, i=i):
                y = x.copy()
                y[i] = t
                return fc(y)

            res = line_minimize_1d(g, tol=1e-13, start=x[i],
                                   initial_step=min(bracket, 1.0))
            x[i] = res.x
        history.append(x.copy())
        if np.max(np.abs(x - x_old)) < tol:
            return OptimizeResult(x, float(fc(x)), None, None, k, True, fc.calls, 0,
                                  "coordinate_descent", history, "converged")
    return OptimizeResult(x, float(fc(x)), None, None, max_iter, False, fc.calls, 0,
                          "coordinate_descent", history, "maximum iterations reached")


def cyclic_coordinate(f, x0, **kwargs):
    """Alias of :func:`coordinate_descent`."""
    return coordinate_descent(f, x0, **kwargs)
