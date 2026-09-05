"""Line searches: choose a step length along a descent direction.

The Wolfe conditions are what make quasi-Newton updates provably stable, so
these routines underpin most of the gradient-based methods in this package.
"""

from __future__ import annotations

import numpy as np

from ..core.utils import as_vector

__all__ = [
    "backtracking",
    "armijo",
    "wolfe",
    "strong_wolfe",
    "exact_line_search",
    "goldstein",
]


def backtracking(f, x, direction, grad, alpha0: float = 1.0, rho: float = 0.5,
                 c1: float = 1e-4, max_iter: int = 60):
    """Backtracking until the Armijo sufficient decrease condition holds."""
    x, direction, grad = as_vector(x), as_vector(direction), as_vector(grad)
    f0 = float(f(x))
    slope = float(grad @ direction)
    alpha = alpha0
    for _ in range(max_iter):
        if float(f(x + alpha * direction)) <= f0 + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


def armijo(f, x, direction, grad, **kwargs):
    """Alias of :func:`backtracking` under the Armijo name."""
    return backtracking(f, x, direction, grad, **kwargs)


def goldstein(f, x, direction, grad, alpha0: float = 1.0, c: float = 0.25,
              max_iter: int = 60):
    """Goldstein conditions: bracket the step between two decrease lines."""
    x, direction, grad = as_vector(x), as_vector(direction), as_vector(grad)
    f0 = float(f(x))
    slope = float(grad @ direction)
    lo, hi = 0.0, np.inf
    alpha = alpha0
    for _ in range(max_iter):
        fa = float(f(x + alpha * direction))
        if fa > f0 + c * alpha * slope:
            hi = alpha
            alpha = 0.5 * (lo + hi)
        elif fa < f0 + (1 - c) * alpha * slope:
            lo = alpha
            alpha = 2 * alpha if np.isinf(hi) else 0.5 * (lo + hi)
        else:
            return alpha
    return alpha


def wolfe(f, grad_f, x, direction, alpha0: float = 1.0, c1: float = 1e-4,
          c2: float = 0.9, max_iter: int = 60):
    """Weak Wolfe conditions: Armijo decrease plus a curvature condition."""
    x, direction = as_vector(x), as_vector(direction)
    f0 = float(f(x))
    g0 = float(as_vector(grad_f(x)) @ direction)
    lo, hi = 0.0, np.inf
    alpha = alpha0
    for _ in range(max_iter):
        xa = x + alpha * direction
        if float(f(xa)) > f0 + c1 * alpha * g0:
            hi = alpha
            alpha = 0.5 * (lo + hi)
        elif float(as_vector(grad_f(xa)) @ direction) < c2 * g0:
            lo = alpha
            alpha = 2 * alpha if np.isinf(hi) else 0.5 * (lo + hi)
        else:
            return alpha
    return alpha


def strong_wolfe(f, grad_f, x, direction, alpha0: float = 1.0, c1: float = 1e-4,
                 c2: float = 0.9, alpha_max: float = 50.0, max_iter: int = 40,
                 phi0=None, dphi0=None):
    """Strong Wolfe line search with bracketing and an interpolating zoom.

    Guarantees ``|g(alpha)'d| <= c2 |g(0)'d|``, which keeps quasi-Newton
    curvature updates positive definite. The zoom phase uses safeguarded
    quadratic interpolation rather than bisection: for a quadratic objective
    that lands on the exact minimizer immediately, which is what preserves the
    conjugacy that nonlinear CG depends on.

    Every caller in the package already holds ``f(x)`` and ``grad f(x)`` from
    the step that chose ``direction``; pass them as ``phi0`` and ``dphi0`` to
    skip re-deriving them. That matters most when ``grad_f`` is a finite
    difference, where recomputing the gradient at ``x`` costs another ``2n``
    evaluations of ``f`` per line search.
    """
    x, direction = as_vector(x), as_vector(direction)
    if phi0 is None:
        phi0 = float(f(x))
    else:
        phi0 = float(phi0)
    if dphi0 is None:
        dphi0 = float(as_vector(grad_f(x)) @ direction)
    else:
        dphi0 = float(dphi0)
    if dphi0 >= 0:
        return 1e-8  # not a descent direction; take a negligible step

    def phi(a):
        return float(f(x + a * direction))

    def dphi(a):
        return float(as_vector(grad_f(x + a * direction)) @ direction)

    def zoom(lo, hi, phi_lo, dphi_lo, phi_hi):
        # phi_hi travels with the bracket: every value that lands on an
        # endpoint was evaluated when the endpoint was chosen, so the interval
        # never needs a second evaluation of its own endpoint.
        for _ in range(max_iter):
            dx = hi - lo
            if dx == 0.0:
                return lo
            # minimizer of the quadratic matching phi(lo), phi'(lo), phi(hi)
            denom = 2.0 * (phi_hi - phi_lo - dphi_lo * dx)
            a = lo - dphi_lo * dx * dx / denom if abs(denom) > 1e-300 else lo + 0.5 * dx
            lo_b, hi_b = (lo, hi) if lo < hi else (hi, lo)
            margin = 0.1 * abs(dx)
            if not np.isfinite(a) or not (lo_b + margin <= a <= hi_b - margin):
                a = lo + 0.5 * dx          # safeguard back to bisection
            pa = phi(a)
            if pa > phi0 + c1 * a * dphi0 or pa >= phi_lo:
                hi, phi_hi = a, pa
            else:
                da = dphi(a)
                if abs(da) <= -c2 * dphi0:
                    return a
                if da * (hi - lo) >= 0:
                    hi, phi_hi = lo, phi_lo
                lo, phi_lo, dphi_lo = a, pa, da
            if abs(hi - lo) < 1e-16 * max(1.0, abs(lo)):
                return lo
        return lo

    a_prev, phi_prev, dphi_prev = 0.0, phi0, dphi0
    a = min(alpha0, alpha_max)
    for i in range(max_iter):
        pa = phi(a)
        if pa > phi0 + c1 * a * dphi0 or (i > 0 and pa >= phi_prev):
            return zoom(a_prev, a, phi_prev, dphi_prev, pa)
        da = dphi(a)
        if abs(da) <= -c2 * dphi0:
            return a
        if da >= 0:
            return zoom(a, a_prev, pa, da, phi_prev)
        a_prev, phi_prev, dphi_prev = a, pa, da
        a = min(2 * a, alpha_max)
    return a


def exact_line_search(f, x, direction, bracket=(0.0, 2.0), tol: float = 1e-10):
    """Minimize ``f(x + alpha d)`` exactly in ``alpha`` by golden section."""
    from .scalar import golden_section

    x, direction = as_vector(x), as_vector(direction)
    res = golden_section(lambda a: float(f(x + a * direction)), bracket[0],
                         bracket[1], tol=tol)
    return float(res.x)
