"""Multidimensional quadrature on boxes, simplices and general regions."""

from __future__ import annotations

import numpy as np

from ..approx.orthopoly import gauss_legendre_nodes
from ..core.types import QuadratureResult
from ..core.utils import CountedFunction, as_vector

__all__ = [
    "double_integral",
    "triple_integral",
    "tensor_gauss",
    "nested_quadrature",
    "cubature_box",
    "triangle_quadrature",
    "tetrahedron_quadrature",
    "polar_integral",
    "spherical_integral",
    "monte_carlo_region",
    "smolyak_grid",
    "sparse_grid_quadrature",
]


def double_integral(f, ax, bx, ay, by, nx: int = 20, ny: int = 20):
    """Double integral over a rectangle or a ``y``-varying region.

    ``ay`` and ``by`` may be callables of ``x``, which handles non-rectangular
    regions bounded by two curves.
    """
    fc = CountedFunction(f)
    xs, wx = gauss_legendre_nodes(nx, ax, bx)
    total = 0.0
    for xi, wi in zip(xs, wx):
        lo = ay(xi) if callable(ay) else ay
        hi = by(xi) if callable(by) else by
        ys, wy = gauss_legendre_nodes(ny, lo, hi)
        total += wi * float(sum(wj * fc(xi, yj) for yj, wj in zip(ys, wy)))
    return QuadratureResult(float(total), None, fc.calls, nx * ny, True, "double_integral")


def triple_integral(f, ax, bx, ay, by, az, bz, nx: int = 10, ny: int = 10,
                    nz: int = 10):
    """Triple integral over a box or a nested variable region."""
    fc = CountedFunction(f)
    xs, wx = gauss_legendre_nodes(nx, ax, bx)
    total = 0.0
    for xi, wi in zip(xs, wx):
        ylo = ay(xi) if callable(ay) else ay
        yhi = by(xi) if callable(by) else by
        ys, wy = gauss_legendre_nodes(ny, ylo, yhi)
        for yj, wj in zip(ys, wy):
            zlo = az(xi, yj) if callable(az) else az
            zhi = bz(xi, yj) if callable(bz) else bz
            zs, wz = gauss_legendre_nodes(nz, zlo, zhi)
            total += wi * wj * float(sum(wk * fc(xi, yj, zk) for zk, wk in zip(zs, wz)))
    return QuadratureResult(float(total), None, fc.calls, nx * ny * nz, True,
                            "triple_integral")


def tensor_gauss(f, lows, highs, n=8):
    """Tensor-product Gauss-Legendre over a box in any dimension.

    Cost grows as ``n^d``, so this is practical up to about five dimensions;
    beyond that use :func:`~numethods.integrate.monte_carlo.quasi_monte_carlo`.
    """
    lows, highs = as_vector(lows), as_vector(highs)
    d = lows.size
    ns = [n] * d if np.isscalar(n) else list(n)
    grids = [gauss_legendre_nodes(ns[i], lows[i], highs[i]) for i in range(d)]
    fc = CountedFunction(f)
    total = 0.0
    for idx in np.ndindex(*[len(g[0]) for g in grids]):
        pt = np.array([grids[i][0][idx[i]] for i in range(d)])
        w = np.prod([grids[i][1][idx[i]] for i in range(d)])
        total += w * fc(pt)
    return QuadratureResult(float(total), None, fc.calls, int(np.prod(ns)), True,
                            "tensor_gauss")


def nested_quadrature(f, bounds, n: int = 20, rule=None):
    """Iterated one-dimensional quadrature over nested variable bounds.

    ``bounds`` is a list of ``(lo, hi)`` pairs where each entry may be a
    callable of the outer variables.
    """
    from .adaptive import adaptive_gauss_kronrod

    d = len(bounds)

    def integrate(level, fixed):
        lo, hi = bounds[level]
        lo = lo(*fixed) if callable(lo) else lo
        hi = hi(*fixed) if callable(hi) else hi
        if level == d - 1:
            return adaptive_gauss_kronrod(lambda t: f(*fixed, t), lo, hi, 1e-10).value
        xs, ws = gauss_legendre_nodes(n, lo, hi)
        return float(sum(w * integrate(level + 1, fixed + [x]) for x, w in zip(xs, ws)))

    return QuadratureResult(float(integrate(0, [])), None, 0, 0, True, "nested")


def cubature_box(f, lows, highs, n=8):
    """Alias of :func:`tensor_gauss` under the cubature name."""
    return tensor_gauss(f, lows, highs, n)


# Symmetric quadrature rules on the reference triangle, given in barycentric
# coordinates with weights summing to 1 (multiply by the triangle's area).
_TRI_RULES = {
    1: ([(1 / 3, 1 / 3, 1 / 3)], [1.0]),
    2: ([(2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)],
        [1 / 3, 1 / 3, 1 / 3]),
    3: ([(1 / 3, 1 / 3, 1 / 3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6)],
        [-27 / 48, 25 / 48, 25 / 48, 25 / 48]),
    4: ([(0.108103018168070, 0.445948490915965, 0.445948490915965),
         (0.445948490915965, 0.108103018168070, 0.445948490915965),
         (0.445948490915965, 0.445948490915965, 0.108103018168070),
         (0.816847572980459, 0.091576213509771, 0.091576213509771),
         (0.091576213509771, 0.816847572980459, 0.091576213509771),
         (0.091576213509771, 0.091576213509771, 0.816847572980459)],
        [0.223381589678011, 0.223381589678011, 0.223381589678011,
         0.109951743655322, 0.109951743655322, 0.109951743655322]),
}


def triangle_quadrature(f, vertices, degree: int = 4):
    """Symmetric quadrature over a triangle, exact to the given degree."""
    V = np.atleast_2d(np.asarray(vertices, dtype=float))
    pts, ws = _TRI_RULES[min(degree, 4)]
    area = 0.5 * abs((V[1, 0] - V[0, 0]) * (V[2, 1] - V[0, 1])
                     - (V[2, 0] - V[0, 0]) * (V[1, 1] - V[0, 1]))
    total = 0.0
    fc = CountedFunction(f)
    for (l1, l2, l3), w in zip(pts, ws):
        p = l1 * V[0] + l2 * V[1] + l3 * V[2]
        total += w * fc(p[0], p[1])
    return QuadratureResult(float(area * total), None, fc.calls, 1, True,
                            f"triangle_deg{degree}")


def tetrahedron_quadrature(f, vertices, degree: int = 2):
    """Quadrature over a tetrahedron in barycentric coordinates."""
    V = np.atleast_2d(np.asarray(vertices, dtype=float))
    if degree <= 1:
        pts = [(0.25, 0.25, 0.25, 0.25)]
        ws = [1.0]
    else:
        a, b = 0.585410196624969, 0.138196601125011
        pts = [(a, b, b, b), (b, a, b, b), (b, b, a, b), (b, b, b, a)]
        ws = [0.25] * 4
    M = np.array([V[1] - V[0], V[2] - V[0], V[3] - V[0]])
    vol = abs(np.linalg.det(M)) / 6.0
    fc = CountedFunction(f)
    total = 0.0
    for lam, w in zip(pts, ws):
        p = sum(l * V[i] for i, l in enumerate(lam))
        total += w * fc(p[0], p[1], p[2])
    return QuadratureResult(float(vol * total), None, fc.calls, 1, True,
                            f"tetrahedron_deg{degree}")


def polar_integral(f, r_min: float, r_max: float, theta_min: float = 0.0,
                   theta_max: float = 2 * np.pi, nr: int = 20, nt: int = 40):
    """Integrate over an annular sector, including the ``r`` Jacobian."""
    return double_integral(lambda r, th: f(r, th) * r, r_min, r_max,
                           theta_min, theta_max, nr, nt)


def spherical_integral(f, r_min: float, r_max: float, nr: int = 16, nt: int = 16,
                       np_: int = 32):
    """Integrate over a spherical shell with the ``r^2 sin(theta)`` Jacobian."""
    return triple_integral(lambda r, th, ph: f(r, th, ph) * r * r * np.sin(th),
                           r_min, r_max, 0.0, np.pi, 0.0, 2 * np.pi, nr, nt, np_)


def monte_carlo_region(f, indicator, lows, highs, n: int = 100000, rng=None):
    """Monte Carlo over an arbitrary region defined by an indicator function."""
    rng = np.random.default_rng(rng)
    lows, highs = as_vector(lows), as_vector(highs)
    d = lows.size
    pts = rng.uniform(lows, highs, size=(n, d))
    inside = np.array([bool(indicator(p)) for p in pts])
    vals = np.zeros(n)
    for i in np.flatnonzero(inside):
        vals[i] = f(pts[i])
    vol = float(np.prod(highs - lows))
    err = float(vol * np.std(vals, ddof=1) / np.sqrt(n))
    return QuadratureResult(vol * float(np.mean(vals)), err, n, 1, True,
                            "monte_carlo_region")


def smolyak_grid(dim: int, level: int, rule=None):
    """Smolyak sparse grid nodes and weights on ``[-1, 1]^dim``.

    A full tensor grid needs ``m^dim`` points; the sparse construction combines
    only those tensor products whose total level is bounded, cutting the count
    to roughly ``m (log m)^{dim-1}`` while keeping polynomial exactness up to
    the same total degree.  That is what makes moderate-dimensional quadrature
    (say 5-20 dimensions) feasible at all -- the tensor grid is hopeless there.

    Returns ``(nodes, weights)`` with ``nodes`` of shape ``(N, dim)``.
    """
    from ..approx.orthopoly import gauss_legendre_nodes
    from math import comb

    rule = (lambda m: gauss_legendre_nodes(m)) if rule is None else rule
    cache = {}

    def one_d(i):
        if i not in cache:
            m = 1 if i == 1 else 2 ** (i - 1) + 1     # nested growth
            cache[i] = rule(m)
        return cache[i]

    points = {}
    q_lo = max(level, dim)
    for q in range(q_lo, level + dim):
        coeff = ((-1.0) ** (level + dim - 1 - q)) * comb(dim - 1, level + dim - 1 - q)
        if coeff == 0.0:
            continue
        for idx in _compositions(q, dim):
            grids = [one_d(i) for i in idx]
            for combo in _tensor_indices([g[0].size for g in grids]):
                node = tuple(grids[d][0][combo[d]] for d in range(dim))
                w = coeff
                for d in range(dim):
                    w *= grids[d][1][combo[d]]
                points[node] = points.get(node, 0.0) + w
    nodes = np.array(list(points.keys()), dtype=float).reshape(-1, dim)
    weights = np.array(list(points.values()), dtype=float)
    return nodes, weights


def _compositions(total, parts):
    """All positive integer tuples of length ``parts`` summing to ``total``."""
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _tensor_indices(sizes):
    """Cartesian product of ``range(s)`` for each ``s`` in ``sizes``."""
    if not sizes:
        yield ()
        return
    for head in range(sizes[0]):
        for tail in _tensor_indices(sizes[1:]):
            yield (head,) + tail


def sparse_grid_quadrature(f, lows, highs, level: int = 4):
    """Integrate over a box with a Smolyak sparse grid.

    ``f`` takes a point vector.  Compare the node count against ``m^dim`` for
    the tensor rule of the same one-dimensional resolution to see the saving.
    """
    lows = np.atleast_1d(np.asarray(lows, dtype=float))
    highs = np.atleast_1d(np.asarray(highs, dtype=float))
    dim = lows.size
    nodes, weights = smolyak_grid(dim, level)
    mid = 0.5 * (highs + lows)
    half = 0.5 * (highs - lows)
    pts = mid + nodes * half
    fc = CountedFunction(f)
    vals = np.array([float(fc(p)) for p in pts])
    volume_scale = float(np.prod(half))
    return QuadratureResult(float(weights @ vals) * volume_scale, None, fc.calls,
                            nodes.shape[0], True, f"smolyak_level{level}")
