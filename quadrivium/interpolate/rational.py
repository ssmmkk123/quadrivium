"""Rational and trigonometric interpolation.

Rational forms handle poles and asymptotes that polynomials cannot represent;
trigonometric interpolation is the natural choice for periodic data.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.exceptions import DimensionError
from ..core.utils import as_vector

__all__ = [
    "thiele",
    "rational_interpolation",
    "bulirsch_stoer_rational",
    "floater_hormann",
    "trigonometric_interpolation",
    "fourier_interpolation",
    "continued_fraction_eval",
]


def thiele(x, y):
    """Thiele's continued fraction interpolation via inverse differences.

    Builds the table of inverse differences ``rho_k`` and evaluates the
    resulting continued fraction, whose partial denominators are
    ``a_k - a_{k-2}``.
    """
    x, y = as_vector(x), as_vector(y)
    n = x.size
    if n < 2:
        raise DimensionError("Thiele interpolation needs at least two points")
    rho = np.zeros((n, n))
    rho[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            denom = rho[i, j - 1] - rho[i + 1, j - 1]
            prev = rho[i + 1, j - 2] if j >= 2 else 0.0
            rho[i, j] = ((x[i] - x[i + j]) / denom + prev
                         if abs(denom) > 1e-300 else np.inf)
    a = rho[0].copy()

    def r(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t_arr)
        for m, ti in enumerate(t_arr):
            val = a[n - 1] - (a[n - 3] if n >= 3 else 0.0)
            for k in range(n - 2, 0, -1):
                base = a[k] - (a[k - 2] if k >= 2 else 0.0)
                val = base + (ti - x[k]) / val if abs(val) > 1e-300 else base
            out[m] = a[0] + (ti - x[0]) / val if abs(val) > 1e-300 else a[0]
        return out[0] if np.ndim(t) == 0 else out.reshape(np.shape(t))

    r.coefficients = a
    return r


def continued_fraction_eval(a, x_nodes, t):
    """Evaluate a Thiele continued fraction with coefficients ``a``."""
    a = as_vector(a)
    x_nodes = as_vector(x_nodes)
    n = a.size
    val = a[n - 1] - (a[n - 3] if n >= 3 else 0.0)
    for k in range(n - 2, 0, -1):
        val = a[k] - (a[k - 2] if k >= 2 else 0.0) + (t - x_nodes[k]) / val
    return a[0] + (t - x_nodes[0]) / val


def bulirsch_stoer_rational(x, y, t):
    """Bulirsch-Stoer rational extrapolation to ``t`` (the classic BS tableau)."""
    x, y = as_vector(x), as_vector(y)
    n = x.size
    # Column j holds T_{i,j-1}, the rational function through points i..i+j-1;
    # column 0 stays zero, standing for the T_{i,-1} = 0 of the recursion.
    R = np.zeros((n, n + 1))
    R[:, 1] = y
    for j in range(2, n + 1):
        for i in range(n - j + 1):
            num = R[i + 1, j - 1] - R[i, j - 1]
            if abs(num) < 1e-300:
                R[i, j] = R[i + 1, j - 1]
                continue
            back = R[i + 1, j - 1] - R[i + 1, j - 2]
            if abs(back) < 1e-300:
                R[i, j] = R[i + 1, j - 1]
                continue
            d1 = (t - x[i]) / (t - x[i + j - 1])
            denom = d1 * (1.0 - num / back) - 1.0
            R[i, j] = R[i + 1, j - 1] + (num / denom if abs(denom) > 1e-300
                                         else 0.0)
    return R[0, n]


def rational_interpolation(x, y, num_degree=None):
    """Linearized rational interpolation ``P(t)/Q(t)`` solved as a linear system.

    Enforces ``Q(x_i) y_i = P(x_i)`` with ``Q`` monic in its top coefficient.
    """
    x, y = as_vector(x), as_vector(y)
    n = x.size
    m = (n - 1) // 2 if num_degree is None else num_degree
    k = n - 1 - m
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(m + 1):
            A[i, j] = x[i] ** j
        for j in range(1, k + 1):
            A[i, m + j] = -y[i] * x[i] ** j
    coeff = np.linalg.solve(A, y)
    p = coeff[: m + 1]
    q = np.concatenate([[1.0], coeff[m + 1 :]])

    def r(t):
        t_arr = np.asarray(t, dtype=float)
        num = sum(p[j] * t_arr**j for j in range(m + 1))
        den = sum(q[j] * t_arr**j for j in range(k + 1))
        return num / den

    r.numerator = p
    r.denominator = q
    return r


def floater_hormann(x, y, d: int = 3):
    """Floater-Hormann barycentric rational interpolation.

    Has no poles on the real line for any blending degree ``d``, and is far more
    robust than polynomial interpolation on equispaced nodes.
    """
    x, y = as_vector(x), as_vector(y)
    n = x.size
    d = min(d, n - 1)
    w = np.zeros(n)
    for i in range(n):
        total = 0.0
        # w_i = sum_{k in J_i} (-1)^k prod_{j in {k..k+d}, j != i} 1/(x_i - x_j)
        for k0 in range(max(0, i - d), min(i, n - 1 - d) + 1):
            prod = 1.0
            for j in range(k0, k0 + d + 1):
                if j != i:
                    prod *= 1.0 / (x[i] - x[j])
            total += ((-1.0) ** k0) * prod
        w[i] = total

    def r(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t_arr)
        for k, ti in enumerate(t_arr):
            diff = ti - x
            hit = np.flatnonzero(diff == 0.0)
            if hit.size:
                out[k] = y[hit[0]]
                continue
            terms = w / diff
            out[k] = (terms @ y) / np.sum(terms)
        return out[0] if np.ndim(t) == 0 else out.reshape(np.shape(t))

    r.weights = w
    return r


def trigonometric_interpolation(x, y, period=None):
    """Trigonometric interpolation of periodic data on equispaced nodes."""
    x, y = as_vector(x), as_vector(y)
    n = x.size
    L = (x[-1] - x[0]) * n / (n - 1) if period is None else period
    x0 = x[0]

    def p(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros_like(t_arr)
        c = np.fft.fft(y) / n
        freqs = np.fft.fftfreq(n, d=1.0 / n)
        for k in range(n):
            out = out + np.real(c[k] * np.exp(2j * np.pi * freqs[k] * (t_arr - x0) / L))
        return out[0] if np.ndim(t) == 0 else out.reshape(np.shape(t))

    return p


def fourier_interpolation(y, factor: int = 2):
    """Band-limited resampling: zero-pad the spectrum to refine a periodic signal."""
    y = as_vector(y)
    n = y.size
    Y = np.fft.rfft(y)
    m = n * factor
    Y_pad = np.zeros(m // 2 + 1, dtype=complex)
    keep = min(len(Y), len(Y_pad))
    Y_pad[:keep] = Y[:keep]
    return np.fft.irfft(Y_pad, m) * factor
