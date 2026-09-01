"""Piecewise polynomial interpolation: splines in their several flavours.

Each constructor returns a :class:`PiecewisePolynomial` supporting evaluation,
differentiation, integration and root finding on the spline itself.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DimensionError, DomainError
from ..core.utils import as_vector
from ..linalg.direct import solve, thomas

__all__ = [
    "PiecewisePolynomial",
    "linear_spline",
    "quadratic_spline",
    "cubic_spline",
    "natural_cubic_spline",
    "clamped_cubic_spline",
    "not_a_knot_spline",
    "periodic_cubic_spline",
    "hermite_spline",
    "pchip",
    "akima_spline",
    "bspline_basis",
    "bspline",
    "bspline_interpolation",
    "catmull_rom",
    "cardinal_spline",
    "smoothing_spline",
    "tension_spline",
    "de_casteljau",
    "bezier",
    "bezier_derivative",
    "bezier_subdivide",
    "rational_bezier",
    "nurbs",
    "open_uniform_knots",
    "nurbs_circle",
]


class PiecewisePolynomial:
    """Piecewise polynomial on knots ``x`` with per-interval coefficients.

    ``coeffs[i]`` lists the coefficients of interval ``i`` in increasing powers
    of the local variable ``(t - x[i])``.
    """

    def __init__(self, x, coeffs, extrapolate: bool = True):
        self.x = as_vector(x)
        self.coeffs = [np.asarray(c, dtype=float) for c in coeffs]
        self.extrapolate = extrapolate
        if len(self.coeffs) != self.x.size - 1:
            raise DimensionError(
                f"{self.x.size} knots require {self.x.size - 1} coefficient sets, "
                f"got {len(self.coeffs)}"
            )

    def _interval(self, t):
        idx = np.searchsorted(self.x, t, side="right") - 1
        return np.clip(idx, 0, len(self.coeffs) - 1)

    def __call__(self, t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if not self.extrapolate and (np.any(t_arr < self.x[0]) or np.any(t_arr > self.x[-1])):
            raise DomainError(
                f"evaluation outside the spline domain [{self.x[0]}, {self.x[-1]}]"
            )
        idx = self._interval(t_arr)
        out = np.empty_like(t_arr)
        for k, (ti, i) in enumerate(zip(t_arr, idx)):
            dt = ti - self.x[i]
            c = self.coeffs[i]
            acc = 0.0
            for p in range(len(c) - 1, -1, -1):
                acc = acc * dt + c[p]
            out[k] = acc
        return out[0] if np.ndim(t) == 0 else out.reshape(np.shape(t))

    def derivative(self, order: int = 1):
        """Return the derivative spline (one polynomial degree lower)."""
        new = []
        for c in self.coeffs:
            d = c.copy()
            for _ in range(order):
                d = d[1:] * np.arange(1, len(d)) if len(d) > 1 else np.array([0.0])
            new.append(d if len(d) else np.array([0.0]))
        return PiecewisePolynomial(self.x, new, self.extrapolate)

    def antiderivative(self):
        """Return an antiderivative spline, continuous and zero at ``x[0]``."""
        new = []
        const = 0.0
        for i, c in enumerate(self.coeffs):
            ic = np.concatenate([[const], c / np.arange(1, len(c) + 1)])
            new.append(ic)
            h = self.x[i + 1] - self.x[i]
            acc = 0.0
            for p in range(len(ic) - 1, -1, -1):
                acc = acc * h + ic[p]
            const = acc
        return PiecewisePolynomial(self.x, new, self.extrapolate)

    def integrate(self, a=None, b=None) -> float:
        """Exact integral of the spline over ``[a, b]``."""
        a = self.x[0] if a is None else a
        b = self.x[-1] if b is None else b
        F = self.antiderivative()
        return float(F(b) - F(a))

    def roots(self):
        """Real roots of the spline inside its knot intervals."""
        out = []
        for i, c in enumerate(self.coeffs):
            if len(c) == 1:
                continue
            r = np.roots(c[::-1])
            for z in r:
                if abs(z.imag) < 1e-10:
                    t = self.x[i] + z.real
                    if self.x[i] - 1e-12 <= t <= self.x[i + 1] + 1e-12:
                        out.append(t)
        return np.unique(np.round(out, 12))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        deg = max(len(c) for c in self.coeffs) - 1 if self.coeffs else 0
        return (f"PiecewisePolynomial(degree={deg}, intervals={len(self.coeffs)}, "
                f"domain=[{self.x[0]:.4g}, {self.x[-1]:.4g}])")


def _check(x, y):
    x, y = as_vector(x), as_vector(y)
    if x.size != y.size:
        raise DimensionError(f"x has {x.size} points but y has {y.size}")
    if np.any(np.diff(x) <= 0):
        order = np.argsort(x)
        x, y = x[order], y[order]
        if np.any(np.diff(x) <= 0):
            raise ValueError("spline knots must be distinct")
    return x, y


def linear_spline(x, y):
    """Piecewise linear interpolation (the C^0 spline)."""
    x, y = _check(x, y)
    coeffs = [np.array([y[i], (y[i + 1] - y[i]) / (x[i + 1] - x[i])])
              for i in range(x.size - 1)]
    return PiecewisePolynomial(x, coeffs)


def quadratic_spline(x, y, slope0: float = 0.0):
    """C^1 quadratic spline; ``slope0`` sets the derivative at the left end."""
    x, y = _check(x, y)
    n = x.size - 1
    coeffs = []
    b = slope0
    for i in range(n):
        h = x[i + 1] - x[i]
        c = (y[i + 1] - y[i] - b * h) / h**2
        coeffs.append(np.array([y[i], b, c]))
        b = b + 2 * c * h
    return PiecewisePolynomial(x, coeffs)


def _cubic_from_moments(x, y, M):
    """Assemble cubic pieces from second derivatives ``M`` at the knots."""
    coeffs = []
    for i in range(x.size - 1):
        h = x[i + 1] - x[i]
        a = y[i]
        b = (y[i + 1] - y[i]) / h - h * (2 * M[i] + M[i + 1]) / 6.0
        c = M[i] / 2.0
        d = (M[i + 1] - M[i]) / (6.0 * h)
        coeffs.append(np.array([a, b, c, d]))
    return PiecewisePolynomial(x, coeffs)


def natural_cubic_spline(x, y):
    """Natural cubic spline: zero second derivative at both ends."""
    x, y = _check(x, y)
    n = x.size - 1
    if n < 1:
        raise DimensionError("need at least two points")
    if n == 1:
        return linear_spline(x, y)
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[0, 0] = A[n, n] = 1.0
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, rhs)
    return _cubic_from_moments(x, y, M)


def clamped_cubic_spline(x, y, dy0: float, dyn: float):
    """Clamped cubic spline: prescribed first derivatives at the ends."""
    x, y = _check(x, y)
    n = x.size - 1
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]
    rhs[0] = 6 * ((y[1] - y[0]) / h[0] - dy0)
    A[n, n - 1] = h[n - 1]
    A[n, n] = 2 * h[n - 1]
    rhs[n] = 6 * (dyn - (y[n] - y[n - 1]) / h[n - 1])
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, rhs)
    return _cubic_from_moments(x, y, M)


def not_a_knot_spline(x, y):
    """Not-a-knot cubic spline: third derivative continuous at the second and
    second-to-last knots. This is the default in most software."""
    x, y = _check(x, y)
    n = x.size - 1
    if n < 3:
        return natural_cubic_spline(x, y)
    h = np.diff(x)
    A = np.zeros((n + 1, n + 1))
    rhs = np.zeros(n + 1)
    A[0, 0] = h[1]
    A[0, 1] = -(h[0] + h[1])
    A[0, 2] = h[0]
    A[n, n - 2] = h[n - 1]
    A[n, n - 1] = -(h[n - 2] + h[n - 1])
    A[n, n] = h[n - 2]
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, rhs)
    return _cubic_from_moments(x, y, M)


def periodic_cubic_spline(x, y, tol: float = 1e-10):
    """Periodic cubic spline; requires ``y[0] == y[-1]``."""
    x, y = _check(x, y)
    if abs(y[0] - y[-1]) > tol:
        raise ValueError("periodic spline requires matching end values")
    n = x.size - 1
    h = np.diff(x)
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(n):
        im1, ip1 = (i - 1) % n, (i + 1) % n
        A[i, im1] = h[im1]
        A[i, i] = 2 * (h[im1] + h[i])
        A[i, ip1] = h[i]
        rhs[i] = 6 * ((y[ip1 if ip1 != 0 else 0] - y[i]) / h[i] if i < n else 0)
    for i in range(n):
        im1 = (i - 1) % n
        nxt = y[i + 1] if i + 1 <= n else y[1]
        rhs[i] = 6 * ((nxt - y[i]) / h[i] - (y[i] - y[i - 1 if i > 0 else n - 1]) / h[im1])
    Msub = np.linalg.solve(A, rhs)
    M = np.concatenate([Msub, [Msub[0]]])
    return _cubic_from_moments(x, y, M)


def cubic_spline(x, y, bc: str = "not-a-knot", **kwargs):
    """Cubic spline with a selectable boundary condition."""
    if bc == "natural":
        return natural_cubic_spline(x, y)
    if bc == "clamped":
        return clamped_cubic_spline(x, y, kwargs["dy0"], kwargs["dyn"])
    if bc == "periodic":
        return periodic_cubic_spline(x, y)
    if bc == "not-a-knot":
        return not_a_knot_spline(x, y)
    raise ValueError(f"unknown boundary condition {bc!r}")


def hermite_spline(x, y, dy):
    """Cubic Hermite spline from prescribed values and slopes."""
    x, y = _check(x, y)
    dy = as_vector(dy)
    coeffs = []
    for i in range(x.size - 1):
        h = x[i + 1] - x[i]
        a, b = y[i], dy[i]
        c = (3 * (y[i + 1] - y[i]) / h**2) - (2 * dy[i] + dy[i + 1]) / h
        d = (-2 * (y[i + 1] - y[i]) / h**3) + (dy[i] + dy[i + 1]) / h**2
        coeffs.append(np.array([a, b, c, d]))
    return PiecewisePolynomial(x, coeffs)


def pchip(x, y):
    """PCHIP: shape-preserving cubic Hermite, monotone where the data is."""
    x, y = _check(x, y)
    n = x.size
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    # one-sided three-point ends, limited to preserve monotonicity
    def end_slope(h0, h1, d0, d1):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > abs(3 * d0):
            return 3 * d0
        return s

    if n > 2:
        d[0] = end_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = end_slope(h[-1], h[-2], delta[-1], delta[-2])
    else:
        d[0] = d[-1] = delta[0]
    return hermite_spline(x, y, d)


def akima_spline(x, y):
    """Akima spline: local, and far less prone to overshoot than a cubic spline."""
    x, y = _check(x, y)
    n = x.size
    if n < 3:
        return linear_spline(x, y)
    m = np.diff(y) / np.diff(x)
    # Pad with two extrapolated slopes at each end, as Akima prescribes.
    mm = np.empty(n + 3)
    mm[2 : n + 1] = m
    mm[1] = 2 * mm[2] - mm[3]
    mm[0] = 2 * mm[1] - mm[2]
    mm[n + 1] = 2 * mm[n] - mm[n - 1]
    mm[n + 2] = 2 * mm[n + 1] - mm[n]
    d = np.zeros(n)
    for i in range(n):
        w1 = abs(mm[i + 3] - mm[i + 2])
        w2 = abs(mm[i + 1] - mm[i])
        if w1 + w2 < 1e-300:
            d[i] = 0.5 * (mm[i + 1] + mm[i + 2])
        else:
            d[i] = (w1 * mm[i + 1] + w2 * mm[i + 2]) / (w1 + w2)
    return hermite_spline(x, y, d)


def catmull_rom(x, y):
    """Catmull-Rom spline: centred-difference slopes."""
    x, y = _check(x, y)
    return cardinal_spline(x, y, tension=0.0)


def cardinal_spline(x, y, tension: float = 0.0):
    """Cardinal spline; ``tension=0`` reproduces Catmull-Rom, ``1`` gives linear."""
    x, y = _check(x, y)
    n = x.size
    d = np.zeros(n)
    c = 1.0 - tension
    for i in range(n):
        lo = max(i - 1, 0)
        hi = min(i + 1, n - 1)
        if hi > lo:
            d[i] = c * (y[hi] - y[lo]) / (x[hi] - x[lo])
    return hermite_spline(x, y, d)


def tension_spline(x, y, tension: float = 1.0):
    """Spline under tension: interpolates between a cubic spline and a polyline."""
    x, y = _check(x, y)
    base = natural_cubic_spline(x, y)
    lin = linear_spline(x, y)
    w = np.clip(tension, 0.0, 1.0)

    def s(t):
        return (1 - w) * base(t) + w * lin(t)

    return s


def bspline_basis(i: int, k: int, knots, t):
    """Cox-de Boor recursion for the ``i``-th B-spline basis of degree ``k``."""
    knots = as_vector(knots)
    t = np.asarray(t, dtype=float)
    if k == 0:
        left = knots[i] <= t
        right = t < knots[i + 1]
        # include the right endpoint in the final span
        if i + 2 == knots.size or knots[i + 1] == knots[-1]:
            right = t <= knots[i + 1]
        return np.where(left & right, 1.0, 0.0)
    out = np.zeros_like(t, dtype=float)
    d1 = knots[i + k] - knots[i]
    if d1 > 0:
        out = out + (t - knots[i]) / d1 * bspline_basis(i, k - 1, knots, t)
    d2 = knots[i + k + 1] - knots[i + 1]
    if d2 > 0:
        out = out + (knots[i + k + 1] - t) / d2 * bspline_basis(i + 1, k - 1, knots, t)
    return out


def bspline(control_points, knots, degree: int):
    """B-spline curve from control points and a knot vector."""
    P = np.asarray(control_points, dtype=float)
    knots = as_vector(knots)
    n = P.shape[0]

    def curve(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if P.ndim == 1:
            out = np.zeros_like(t_arr)
        else:
            out = np.zeros((t_arr.size, P.shape[1]))
        for i in range(n):
            B = bspline_basis(i, degree, knots, t_arr)
            out = out + (B * P[i] if P.ndim == 1 else B[:, None] * P[i])
        if np.ndim(t) == 0:
            return out[0]
        return out

    return curve


def bspline_interpolation(x, y, degree: int = 3):
    """Interpolating B-spline: solves for control points hitting every datum."""
    x, y = _check(x, y)
    n = x.size
    k = min(degree, n - 1)
    # clamped, averaged knot vector (de Boor's rule)
    interior = np.array([np.mean(x[i : i + k]) for i in range(1, n - k)]) if n - k > 1 else np.array([])
    knots = np.concatenate([np.repeat(x[0], k + 1), interior, np.repeat(x[-1], k + 1)])
    A = np.zeros((n, n))
    for r, t in enumerate(x):
        for i in range(n):
            A[r, i] = bspline_basis(i, k, knots, np.array([t]))[0]
    A[-1, -1] = 1.0 if np.all(A[-1] == 0) else A[-1, -1]
    P = np.linalg.solve(A, y)
    return bspline(P, knots, k)


def smoothing_spline(x, y, lam: float = 1.0, weights=None):
    """Cubic smoothing spline: balances fidelity against roughness ``lam``.

    Minimizes ``sum w_i (y_i - f(x_i))^2 + lam * integral f''^2``.
    """
    x, y = _check(x, y)
    n = x.size
    w = np.ones(n) if weights is None else as_vector(weights)
    h = np.diff(x)
    # second-difference operator and its Gram matrix (Reinsch formulation)
    D = np.zeros((n - 2, n))
    R = np.zeros((n - 2, n - 2))
    for i in range(n - 2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -(1.0 / h[i] + 1.0 / h[i + 1])
        D[i, i + 2] = 1.0 / h[i + 1]
        R[i, i] = (h[i] + h[i + 1]) / 3.0
        if i < n - 3:
            R[i, i + 1] = R[i + 1, i] = h[i + 1] / 6.0
    W = np.diag(1.0 / w)
    A = R + lam * (D @ W @ D.T)
    gamma = np.linalg.solve(A, D @ y)
    f = y - lam * W @ (D.T @ gamma)
    return natural_cubic_spline(x, f)


def de_casteljau(control_points, t):
    """Evaluate a Bezier curve by de Casteljau's algorithm.

    Repeated linear interpolation between control points.  Slower than
    expanding the Bernstein polynomials but numerically stable, because every
    intermediate value is a convex combination of the previous ones and so
    stays inside the control polygon -- the Bernstein form loses that with
    cancellation at high degree.
    """
    P = np.atleast_2d(np.asarray(control_points, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.empty((t.size, P.shape[1]))
    for k, tk in enumerate(t):
        Q = P.copy()
        for r in range(1, Q.shape[0]):
            Q = (1.0 - tk) * Q[:-1] + tk * Q[1:]
        out[k] = Q[0]
    return out[0] if out.shape[0] == 1 else out


def bezier(control_points):
    """Bezier curve through its control points; returns an evaluator on ``[0, 1]``."""
    P = np.atleast_2d(np.asarray(control_points, dtype=float))
    return lambda t: de_casteljau(P, t)


def bezier_derivative(control_points, order: int = 1):
    """Control points of the derivative curve.

    The derivative of a degree-``n`` Bezier curve is a degree-``n-1`` Bezier
    curve on the scaled differences ``n (P_{i+1} - P_i)`` -- so derivatives are
    exact and need no differencing.
    """
    P = np.atleast_2d(np.asarray(control_points, dtype=float))
    for _ in range(order):
        n = P.shape[0] - 1
        if n < 1:
            return np.zeros((1, P.shape[1]))
        P = n * (P[1:] - P[:-1])
    return P


def bezier_subdivide(control_points, t: float = 0.5):
    """Split a Bezier curve at ``t`` into two Bezier curves.

    The de Casteljau triangle's two outer edges are exactly the control points
    of the halves, which is why subdivision is free once the point is
    evaluated.  Returns ``(left, right)``.
    """
    P = np.atleast_2d(np.asarray(control_points, dtype=float))
    left, right = [P[0]], [P[-1]]
    Q = P.copy()
    while Q.shape[0] > 1:
        Q = (1.0 - t) * Q[:-1] + t * Q[1:]
        left.append(Q[0])
        right.append(Q[-1])
    return np.array(left), np.array(right[::-1])


def rational_bezier(control_points, weights):
    """Rational Bezier curve; unit weights reduce to the polynomial case.

    Rational curves represent conic sections exactly -- a circular arc is not
    a polynomial curve at any degree, which is the whole reason CAD uses
    rational forms.
    """
    P = np.atleast_2d(np.asarray(control_points, dtype=float))
    w = as_vector(weights)
    if w.size != P.shape[0]:
        raise DimensionError("one weight per control point required")
    Pw = np.hstack([P * w[:, None], w[:, None]])

    def curve(t):
        h = np.atleast_2d(de_casteljau(Pw, t))
        return (h[:, :-1] / h[:, -1:]) if h.shape[0] > 1 else (h[0, :-1] / h[0, -1])

    return curve


def nurbs(control_points, weights, knots, degree: int):
    """Non-uniform rational B-spline curve.

    Evaluated in homogeneous coordinates -- the control points are lifted by
    their weights, a plain B-spline is evaluated there, and the result is
    projected back by dividing through.  Doing the division at the end rather
    than blending rational basis functions directly is what keeps the
    evaluation as stable as the polynomial B-spline it is built on.
    """
    P = np.atleast_2d(np.asarray(control_points, dtype=float))
    w = as_vector(weights)
    knots = as_vector(knots)
    if w.size != P.shape[0]:
        raise DimensionError("one weight per control point required")
    if knots.size != P.shape[0] + degree + 1:
        raise DimensionError(
            f"a degree-{degree} NURBS with {P.shape[0]} control points needs "
            f"{P.shape[0] + degree + 1} knots, got {knots.size}"
        )
    Pw = np.hstack([P * w[:, None], w[:, None]])

    def curve(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty((t.size, P.shape[1]))
        for k, tk in enumerate(t):
            h = _bspline_point(Pw, knots, degree, tk)
            out[k] = h[:-1] / h[-1]
        return out[0] if out.shape[0] == 1 else out

    return curve


def _bspline_point(control, knots, degree, t):
    """One B-spline point by the Cox-de Boor recursion (de Boor's algorithm)."""
    n = control.shape[0]
    t = float(np.clip(t, knots[degree], knots[n]))
    # Locate the knot span containing t.
    span = degree
    for i in range(degree, n):
        if knots[i] <= t < knots[i + 1]:
            span = i
            break
    else:
        span = n - 1
    d = [control[span - degree + j].copy() for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = span - degree + j
            denom = knots[i + degree - r + 1] - knots[i]
            alpha = 0.0 if denom == 0 else (t - knots[i]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[degree]


def open_uniform_knots(n_control: int, degree: int):
    """Clamped (open uniform) knot vector: the curve meets its end control points."""
    m = n_control + degree + 1
    knots = np.zeros(m)
    interior = m - 2 * (degree + 1)
    knots[degree + 1: degree + 1 + interior] = np.arange(1, interior + 1) / (interior + 1.0)
    knots[degree + 1 + interior:] = 1.0
    return knots


def nurbs_circle(radius: float = 1.0, center=(0.0, 0.0)):
    """Exact unit circle as a quadratic NURBS (nine control points).

    A demonstration that rational splines capture conics exactly: the weights
    ``1/sqrt(2)`` at the corner control points are what bend the quadratic
    segments onto true circular arcs.  No polynomial spline can do this.
    """
    c = np.asarray(center, dtype=float)
    s = 1.0 / np.sqrt(2.0)
    P = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0],
                  [-1, -1], [0, -1], [1, -1], [1, 0]], dtype=float)
    w = np.array([1, s, 1, s, 1, s, 1, s, 1.0])
    knots = np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1.0])
    return nurbs(P * radius + c, w, knots, 2)
