"""Interpolation in two or more dimensions."""

from __future__ import annotations

from .. import numeric as np

from ..core.exceptions import DimensionError
from ..core.utils import as_matrix, as_vector
from .spline import natural_cubic_spline, not_a_knot_spline

__all__ = [
    "bilinear",
    "bicubic",
    "nearest_neighbor",
    "regular_grid_interpolator",
    "tensor_product_spline",
    "shepard",
    "inverse_distance_weighting",
    "rbf_interpolation",
    "kriging",
    "barycentric_triangle",
    "triangular_interpolation",
    "trilinear",
]


def bilinear(x, y, Z):
    """Bilinear interpolation on a rectangular grid; ``Z`` has shape ``(len(x), len(y))``."""
    x, y = as_vector(x), as_vector(y)
    Z = np.asarray(Z, dtype=float)
    if Z.shape != (x.size, y.size):
        raise DimensionError(f"Z must have shape {(x.size, y.size)}, got {Z.shape}")

    def f(xi, yi):
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        yi = np.atleast_1d(np.asarray(yi, dtype=float))
        i = np.clip(np.searchsorted(x, xi) - 1, 0, x.size - 2)
        j = np.clip(np.searchsorted(y, yi) - 1, 0, y.size - 2)
        tx = (xi - x[i]) / (x[i + 1] - x[i])
        ty = (yi - y[j]) / (y[j + 1] - y[j])
        out = ((1 - tx) * (1 - ty) * Z[i, j] + tx * (1 - ty) * Z[i + 1, j]
               + (1 - tx) * ty * Z[i, j + 1] + tx * ty * Z[i + 1, j + 1])
        return out[0] if np.ndim(xi) == 0 or out.size == 1 else out

    return f


def bicubic(x, y, Z):
    """Bicubic interpolation by tensor-product cubic splines."""
    return tensor_product_spline(x, y, Z, spline=not_a_knot_spline)


def tensor_product_spline(x, y, Z, spline=natural_cubic_spline):
    """Tensor-product spline: interpolate along rows, then across the results."""
    x, y = as_vector(x), as_vector(y)
    Z = np.asarray(Z, dtype=float)
    row_splines = [spline(y, Z[i, :]) for i in range(x.size)]

    def f(xi, yi):
        xi_a = np.atleast_1d(np.asarray(xi, dtype=float))
        yi_a = np.atleast_1d(np.asarray(yi, dtype=float))
        out = np.empty((xi_a.size, yi_a.size))
        for k, yv in enumerate(yi_a):
            col = np.array([rs(yv) for rs in row_splines])
            out[:, k] = spline(x, col)(xi_a)
        if np.ndim(xi) == 0 and np.ndim(yi) == 0:
            return float(out[0, 0])
        if np.ndim(xi) == 0 or np.ndim(yi) == 0:
            return out.ravel()
        return out

    return f


def nearest_neighbor(points, values):
    """Nearest-neighbour interpolation for scattered data."""
    P = as_matrix(np.atleast_2d(points))
    v = as_vector(values)

    def f(q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        d = np.linalg.norm(q[:, None, :] - P[None, :, :], axis=2)
        idx = np.argmin(d, axis=1)
        return v[idx] if q.shape[0] > 1 else v[idx[0]]

    return f


def regular_grid_interpolator(grids, V, method: str = "linear"):
    """N-dimensional interpolation on a regular grid (linear or nearest)."""
    grids = [as_vector(g) for g in grids]
    V = np.asarray(V, dtype=float)
    ndim = len(grids)
    if V.shape != tuple(g.size for g in grids):
        raise DimensionError(f"V must have shape {tuple(g.size for g in grids)}")

    def f(q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        out = np.empty(q.shape[0])
        for r, pt in enumerate(q):
            if method == "nearest":
                idx = tuple(int(np.argmin(np.abs(g - pt[d]))) for d, g in enumerate(grids))
                out[r] = V[idx]
                continue
            lo = [int(np.clip(np.searchsorted(g, pt[d]) - 1, 0, g.size - 2))
                  for d, g in enumerate(grids)]
            t = [(pt[d] - grids[d][lo[d]]) / (grids[d][lo[d] + 1] - grids[d][lo[d]])
                 for d in range(ndim)]
            total = 0.0
            for corner in range(2**ndim):
                w = 1.0
                idx = []
                for d in range(ndim):
                    bit = (corner >> d) & 1
                    w *= t[d] if bit else (1 - t[d])
                    idx.append(lo[d] + bit)
                total += w * V[tuple(idx)]
            out[r] = total
        return out[0] if q.shape[0] == 1 else out

    return f


def trilinear(x, y, z, V):
    """Trilinear interpolation on a 3-D rectangular grid.

    The returned callable takes separate coordinates ``f(xi, yi, zi)``, matching
    :func:`bilinear`.  Use :func:`regular_grid_interpolator` directly if you
    would rather pass points as tuples.
    """
    grid = regular_grid_interpolator([x, y, z], V, "linear")

    def f(xi, yi, zi):
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        yi = np.atleast_1d(np.asarray(yi, dtype=float))
        zi = np.atleast_1d(np.asarray(zi, dtype=float))
        xi, yi, zi = np.broadcast_arrays(xi, yi, zi)
        out = np.array([grid((a, b, c)) for a, b, c in
                        zip(xi.ravel(), yi.ravel(), zi.ravel())])
        return float(out[0]) if out.size == 1 else out.reshape(xi.shape)

    return f


def shepard(points, values, power: float = 2.0, tol: float = 1e-12):
    """Shepard's method: global inverse-distance weighting."""
    return inverse_distance_weighting(points, values, power=power, tol=tol)


def inverse_distance_weighting(points, values, power: float = 2.0, k=None,
                               tol: float = 1e-12):
    """Inverse distance weighting; ``k`` restricts to the ``k`` nearest points."""
    P = np.atleast_2d(np.asarray(points, dtype=float))
    v = as_vector(values)

    def f(q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        out = np.empty(q.shape[0])
        for r, pt in enumerate(q):
            d = np.linalg.norm(P - pt, axis=1)
            hit = np.flatnonzero(d < tol)
            if hit.size:
                out[r] = v[hit[0]]
                continue
            order = np.argsort(d)[:k] if k is not None else np.arange(d.size)
            w = 1.0 / d[order] ** power
            out[r] = (w @ v[order]) / np.sum(w)
        return out[0] if q.shape[0] == 1 else out

    return f


_RBF_KERNELS = {
    "multiquadric": lambda r, e: np.sqrt(1.0 + (e * r) ** 2),
    "inverse_multiquadric": lambda r, e: 1.0 / np.sqrt(1.0 + (e * r) ** 2),
    "gaussian": lambda r, e: np.exp(-((e * r) ** 2)),
    "linear": lambda r, e: r,
    "cubic": lambda r, e: r**3,
    "quintic": lambda r, e: r**5,
    "thin_plate": lambda r, e: np.where(r > 0, r**2 * np.log(np.where(r > 0, r, 1.0)), 0.0),
}


def rbf_interpolation(points, values, kernel: str = "multiquadric",
                      epsilon: float = 1.0, smooth: float = 0.0):
    """Radial basis function interpolation for scattered data in any dimension.

    Available kernels: multiquadric, inverse_multiquadric, gaussian, linear,
    cubic, quintic, thin_plate.
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    v = as_vector(values)
    if kernel not in _RBF_KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; choose from {sorted(_RBF_KERNELS)}")
    phi = _RBF_KERNELS[kernel]
    n = P.shape[0]
    R = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    A = phi(R, epsilon) + smooth * np.eye(n)
    try:
        w = np.linalg.solve(A, v)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, v, rcond=None)[0]

    def f(q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        r = np.linalg.norm(q[:, None, :] - P[None, :, :], axis=2)
        out = phi(r, epsilon) @ w
        return out[0] if q.shape[0] == 1 else out

    f.weights = w
    return f


def kriging(points, values, sill: float = 1.0, range_: float = 1.0,
            nugget: float = 0.0, model: str = "spherical"):
    """Ordinary kriging with a spherical, exponential or gaussian variogram.

    Returns a callable giving ``(estimate, variance)``.
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    v = as_vector(values)
    n = P.shape[0]

    def gamma(h):
        h = np.asarray(h, dtype=float)
        if model == "spherical":
            out = np.where(h < range_,
                           nugget + sill * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
                           nugget + sill)
        elif model == "exponential":
            out = nugget + sill * (1.0 - np.exp(-3.0 * h / range_))
        elif model == "gaussian":
            out = nugget + sill * (1.0 - np.exp(-3.0 * (h / range_) ** 2))
        else:
            raise ValueError(f"unknown variogram model {model!r}")
        return np.where(h == 0, 0.0, out)

    H = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = gamma(H)
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    def f(q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        est = np.empty(q.shape[0])
        var = np.empty(q.shape[0])
        for r, pt in enumerate(q):
            d = np.linalg.norm(P - pt, axis=1)
            b = np.append(gamma(d), 1.0)
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b, rcond=None)[0]
            est[r] = w[:n] @ v
            var[r] = w[:n] @ gamma(d) + w[n]
        if q.shape[0] == 1:
            return float(est[0]), float(var[0])
        return est, var

    return f


def barycentric_triangle(p, a, b, c):
    """Barycentric coordinates of ``p`` with respect to triangle ``(a, b, c)``."""
    p, a, b, c = (as_vector(v) for v in (p, a, b, c))
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-300:
        raise ValueError("degenerate triangle")
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - beta - gamma, beta, gamma])


def triangular_interpolation(vertices, values):
    """Linear interpolation over a single triangle from its vertex values."""
    V = np.atleast_2d(np.asarray(vertices, dtype=float))
    vals = as_vector(values)

    def f(p):
        lam = barycentric_triangle(p, V[0], V[1], V[2])
        return float(lam @ vals)

    return f
