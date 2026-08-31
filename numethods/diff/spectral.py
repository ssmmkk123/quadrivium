"""Spectral differentiation: exponentially accurate derivatives.

For smooth functions these converge faster than any power of the grid spacing,
which is why they underpin spectral PDE solvers.
"""

from __future__ import annotations

import numpy as np

from ..core.utils import as_vector

__all__ = [
    "fourier_diff_matrix",
    "fourier_derivative",
    "chebyshev_diff_matrix",
    "chebyshev_points",
    "chebyshev_derivative",
    "spectral_derivative_fft",
    "chebyshev_coefficients",
    "chebyshev_evaluate",
    "clenshaw",
]


def fourier_diff_matrix(n: int, L: float = 2 * np.pi, order: int = 1):
    """Differentiation matrix for periodic data on ``n`` equispaced points.

    The matrix is circulant, built from its first column; the closed forms
    differ between even and odd ``n``, and both are handled.
    """
    h = 2 * np.pi / n
    k = np.arange(1, n)
    col = np.zeros(n)
    if order == 1:
        if n % 2 == 0:
            col[1:] = 0.5 * (-1.0) ** k / np.tan(k * h / 2.0)
        else:
            col[1:] = 0.5 * (-1.0) ** k / np.sin(k * h / 2.0)
    elif order == 2:
        if n % 2 == 0:
            col[0] = -np.pi**2 / (3 * h**2) - 1.0 / 6.0
            col[1:] = -((-1.0) ** k) / (2 * np.sin(k * h / 2.0) ** 2)
        else:
            col[0] = -np.pi**2 / (3 * h**2) + 1.0 / 12.0
            col[1:] = -((-1.0) ** k) * np.cos(k * h / 2.0) / (2 * np.sin(k * h / 2.0) ** 2)
    else:
        raise ValueError("order must be 1 or 2")
    D = _circulant(col)
    return D * (2 * np.pi / L) ** order


def _circulant(col):
    """Circulant matrix ``D[i, j] = col[(i - j) mod n]``."""
    n = col.size
    i = np.arange(n)
    return col[(i[:, None] - i[None, :]) % n]


def fourier_derivative(y, L: float = 2 * np.pi, order: int = 1):
    """Derivative of a periodic sampled function via the FFT."""
    y = as_vector(y)
    n = y.size
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    Y = np.fft.fft(y)
    if order % 2 == 1:
        Y[n // 2] = 0.0  # kill the unmatched Nyquist mode for odd orders
    return np.real(np.fft.ifft((1j * k) ** order * Y))


def spectral_derivative_fft(y, L: float = 2 * np.pi, order: int = 1):
    """Alias of :func:`fourier_derivative` under its common name."""
    return fourier_derivative(y, L, order)


def chebyshev_points(n: int, a: float = -1.0, b: float = 1.0):
    """Chebyshev-Gauss-Lobatto points ``cos(j pi / n)``, ``j = 0..n``.

    Returned in increasing order on ``[a, b]``.
    """
    x = np.cos(np.pi * np.arange(n + 1) / n)
    return 0.5 * (a + b) + 0.5 * (b - a) * x[::-1]


def chebyshev_diff_matrix(n: int, a: float = -1.0, b: float = 1.0):
    """Chebyshev differentiation matrix on ``n+1`` Lobatto points.

    Returns ``(D, x)`` with ``D @ f(x) ~ f'(x)`` to spectral accuracy.
    """
    if n == 0:
        return np.zeros((1, 1)), np.array([0.0])
    x_std = np.cos(np.pi * np.arange(n + 1) / n)  # descending, x_0 = 1
    c = np.ones(n + 1)
    c[0] = c[n] = 2.0
    c = c * (-1.0) ** np.arange(n + 1)
    X = np.tile(x_std.reshape(-1, 1), (1, n + 1))
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D = D - np.diag(D.sum(axis=1))
    # flip to ascending order and rescale to [a, b]
    P = np.arange(n, -1, -1)
    D = D[np.ix_(P, P)]
    x = 0.5 * (a + b) + 0.5 * (b - a) * x_std[::-1]
    return D * (2.0 / (b - a)), x


def chebyshev_derivative(f, n: int = 32, a: float = -1.0, b: float = 1.0):
    """Spectrally accurate derivative of ``f`` sampled at Lobatto points."""
    D, x = chebyshev_diff_matrix(n, a, b)
    return x, D @ np.array([f(xi) for xi in x])


def chebyshev_coefficients(f, n: int = 32, a: float = -1.0, b: float = 1.0):
    """Chebyshev series coefficients of ``f`` computed by the DCT (via FFT)."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    y = np.array([f(0.5 * (a + b) + 0.5 * (b - a) * xi) for xi in x])
    ext = np.concatenate([y, y[-2:0:-1]])
    c = np.real(np.fft.fft(ext))[: n + 1] / n
    c[0] /= 2.0
    c[n] /= 2.0
    return c


def clenshaw(coeffs, t):
    """Clenshaw recurrence: stable evaluation of a Chebyshev series."""
    c = as_vector(coeffs)
    t = np.asarray(t, dtype=float)
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for k in range(c.size - 1, 0, -1):
        b1, b2 = 2 * t * b1 - b2 + c[k], b1
    return t * b1 - b2 + c[0]


def chebyshev_evaluate(coeffs, t, a: float = -1.0, b: float = 1.0):
    """Evaluate a Chebyshev series on ``[a, b]`` using Clenshaw's algorithm."""
    s = (2.0 * np.asarray(t, dtype=float) - (a + b)) / (b - a)
    return clenshaw(coeffs, s)
