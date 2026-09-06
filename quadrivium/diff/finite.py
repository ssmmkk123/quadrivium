"""Numerical differentiation by finite differences and related techniques.

Includes arbitrary-order stencil generation (Fornberg's algorithm), Richardson
extrapolation, and the complex-step derivative, which is free of subtractive
cancellation.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.exceptions import DomainError
from ..core.utils import EPS, as_vector

__all__ = [
    "forward_difference",
    "backward_difference",
    "central_difference",
    "second_derivative",
    "third_derivative",
    "fourth_derivative",
    "five_point_stencil",
    "fornberg_weights",
    "finite_difference_weights",
    "differentiation_matrix",
    "richardson_extrapolation",
    "richardson_derivative",
    "complex_step_derivative",
    "optimal_step_size",
    "gradient_fd",
    "jacobian_fd",
    "hessian_fd",
    "differentiate_data",
    "savitzky_golay_derivative",
]


def forward_difference(f, x, h: float = 1e-6, order: int = 1):
    """Forward difference, accuracy ``O(h)``."""
    x = float(x)
    if order == 1:
        return (f(x + h) - f(x)) / h
    if order == 2:
        return (f(x + 2 * h) - 2 * f(x + h) + f(x)) / h**2
    if order == 3:
        return (f(x + 3 * h) - 3 * f(x + 2 * h) + 3 * f(x + h) - f(x)) / h**3
    raise ValueError("order must be 1, 2 or 3")


def backward_difference(f, x, h: float = 1e-6, order: int = 1):
    """Backward difference, accuracy ``O(h)``."""
    x = float(x)
    if order == 1:
        return (f(x) - f(x - h)) / h
    if order == 2:
        return (f(x) - 2 * f(x - h) + f(x - 2 * h)) / h**2
    if order == 3:
        return (f(x) - 3 * f(x - h) + 3 * f(x - 2 * h) - f(x - 3 * h)) / h**3
    raise ValueError("order must be 1, 2 or 3")


def central_difference(f, x, h: float = 1e-6, order: int = 1):
    """Central difference, accuracy ``O(h^2)``."""
    x = float(x)
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    if order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
    if order == 3:
        return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h**3)
    raise ValueError("order must be 1, 2 or 3")


def second_derivative(f, x, h: float = 1e-5):
    """Second derivative by the three-point central formula."""
    return central_difference(f, x, h, order=2)


def third_derivative(f, x, h: float = 1e-4):
    """Third derivative by the five-point central formula."""
    return central_difference(f, x, h, order=3)


def fourth_derivative(f, x, h: float = 1e-3):
    """Fourth derivative by the five-point central formula."""
    x = float(x)
    return (f(x + 2 * h) - 4 * f(x + h) + 6 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / h**4


def five_point_stencil(f, x, h: float = 1e-4):
    """Fourth-order accurate first derivative from five points."""
    x = float(x)
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def fornberg_weights(z: float, nodes, max_order: int = 1):
    """Fornberg's algorithm for finite difference weights.

    Returns an array ``W`` of shape ``(len(nodes), max_order+1)`` where
    ``W[:, m]`` are the weights approximating the ``m``-th derivative at ``z``
    from the values at ``nodes``. Handles arbitrary (even non-uniform) nodes.
    """
    x = as_vector(nodes)
    n = x.size - 1
    m = max_order
    if m > n:
        raise DomainError(f"{x.size} nodes cannot support derivative order {m}")
    c = np.zeros((n + 1, m + 1))
    c1, c4 = 1.0, x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def finite_difference_weights(order: int, accuracy: int = 2, kind: str = "central"):
    """Weights and offsets for a standard stencil.

    Returns ``(offsets, weights)`` such that
    ``f^(order)(x) ~ sum w_i f(x + offset_i h) / h**order``.
    """
    if kind == "central":
        p = 2 * ((order + 1) // 2) - 1 + accuracy
        half = p // 2
        offsets = np.arange(-half, half + 1)
    elif kind == "forward":
        offsets = np.arange(0, order + accuracy)
    elif kind == "backward":
        offsets = np.arange(-(order + accuracy) + 1, 1)
    else:
        raise ValueError("kind must be 'central', 'forward' or 'backward'")
    W = fornberg_weights(0.0, offsets.astype(float), order)
    return offsets, W[:, order]


def differentiation_matrix(x, order: int = 1, stencil: int = 3):
    """Dense differentiation matrix ``D`` with ``D @ f(x) ~ f^(order)(x)``.

    Uses a moving Fornberg stencil, so non-uniform grids are supported.
    """
    x = as_vector(x)
    n = x.size
    stencil = max(stencil, order + 1)
    D = np.zeros((n, n))
    half = stencil // 2
    for i in range(n):
        lo = int(np.clip(i - half, 0, max(n - stencil, 0)))
        hi = min(lo + stencil, n)
        lo = max(hi - stencil, 0)
        W = fornberg_weights(x[i], x[lo:hi], order)
        D[i, lo:hi] = W[:, order]
    return D


def richardson_extrapolation(F, h: float, levels: int = 5, factor: float = 2.0,
                             order: int = 2):
    """Richardson extrapolation of ``F(h)`` to ``h -> 0``.

    ``F`` must be a function of the step alone; ``order`` is the leading error
    exponent of the base formula. Returns ``(value, table)``.
    """
    T = np.zeros((levels, levels))
    for i in range(levels):
        T[i, 0] = F(h / factor**i)
        for j in range(1, i + 1):
            p = factor ** (order + j - 1)
            T[i, j] = (p * T[i, j - 1] - T[i - 1, j - 1]) / (p - 1.0)
    return T[levels - 1, levels - 1], T


def richardson_derivative(f, x, h: float = 0.1, levels: int = 5, order: int = 1):
    """Derivative by Richardson extrapolation of central differences.

    Reaches near machine precision without the step-size dilemma of a single
    finite difference.
    """
    base = lambda hh: central_difference(f, x, hh, order=order)
    value, table = richardson_extrapolation(base, h, levels, 2.0, 2)
    return value


def complex_step_derivative(f, x, h: float = 1e-20):
    """Complex-step derivative ``Im(f(x + ih))/h``.

    Exact to machine precision with no subtractive cancellation, but ``f`` must
    be analytic and implemented with complex-safe operations.
    """
    return float(np.imag(f(complex(x, h))) / h)


def optimal_step_size(order: int = 1, accuracy: int = 2, scale: float = 1.0):
    """Step size balancing truncation against round-off error."""
    total = order + accuracy
    return float(scale * EPS ** (1.0 / (total + 1)))


def gradient_fd(f, x, h=None, method: str = "central"):
    """Gradient of a scalar field by finite differences."""
    from ..core.utils import numerical_gradient

    if method == "central":
        return numerical_gradient(f, x, h)
    x = as_vector(x)
    g = np.empty_like(x)
    f0 = float(f(x))
    for j in range(x.size):
        hj = h if h is not None else optimal_step_size(1, 1, max(abs(x[j]), 1.0))
        xp = x.copy()
        if method == "forward":
            xp[j] += hj
            g[j] = (float(f(xp)) - f0) / hj
        else:
            xp[j] -= hj
            g[j] = (f0 - float(f(xp))) / hj
    return g


def jacobian_fd(F, x, h=None, method: str = "central"):
    """Jacobian of a vector field by finite differences."""
    from ..core.utils import numerical_jacobian

    return numerical_jacobian(F, x, h, method)


def hessian_fd(f, x, h=None):
    """Hessian by second-order finite differences."""
    from ..core.utils import numerical_hessian

    return numerical_hessian(f, x, h)


def differentiate_data(x, y, order: int = 1, stencil: int = 3):
    """Differentiate tabulated data using a moving finite difference stencil."""
    x, y = as_vector(x), as_vector(y)
    return differentiation_matrix(x, order, stencil) @ y


def savitzky_golay_derivative(y, window: int = 5, poly_order: int = 2,
                              order: int = 1, dx: float = 1.0):
    """Savitzky-Golay smoothing differentiator for noisy data.

    Fits a local polynomial by least squares and differentiates the fit, which
    suppresses the noise amplification of plain differencing.
    """
    y = as_vector(y)
    n = y.size
    if window % 2 == 0:
        window += 1
    if window > n:
        raise ValueError("window must not exceed the number of samples")
    half = window // 2
    from math import factorial

    idx = np.arange(-half, half + 1)
    A = np.vander(idx, poly_order + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    coef = pinv[order] * factorial(order) / dx**order
    padded = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
    out = np.array([coef @ padded[i : i + window] for i in range(n)])
    return out
