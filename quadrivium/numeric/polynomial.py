"""Polynomial helpers, mirroring the ``numpy`` functions the library calls.

Coefficients follow NumPy's ``poly1d`` convention: highest power first.
"""

from __future__ import annotations

from .. import _qnp as _c

__all__ = ["polyval", "polyfit", "poly", "roots", "polymul", "polydiv",
           "polyadd", "polysub", "polyint", "polyder", "trim_zeros",
           "convolve", "chebyshev", "legendre", "leggauss"]


def polyval(p, x):
    """Horner evaluation, which is what NumPy does and is stabler than powers."""
    coeffs = _c.asarray(p)
    values = _c.asarray(x)
    dtype = _c.complex128 if (coeffs.dtype == _c.complex128 or
                              values.dtype == _c.complex128) else _c.float64
    total = _c.zeros(values.shape, dtype=dtype) if values.ndim else _c.zeros((), dtype=dtype)
    for i in range(coeffs.size):
        total = total * values + coeffs[i]
    return total


def polyfit(x, y, deg, rcond=None):
    """Least-squares polynomial fit through the Vandermonde system."""
    from . import vander
    xs = _c.asarray(x, dtype=float)
    ys = _c.asarray(y, dtype=float)
    A = vander(xs, deg + 1)
    solution, *_ = _c.linalg.lstsq(A, ys, rcond=rcond)
    return solution


def poly(seq_of_zeros):
    """Coefficients of the monic polynomial with the given roots."""
    zeros = _c.asarray(seq_of_zeros)
    if zeros.ndim == 2:
        zeros = _c.linalg.eigvals(zeros)
    coeffs = _c.array([1.0], dtype=zeros.dtype if zeros.dtype == _c.complex128 else _c.float64)
    for i in range(zeros.size):
        shifted = _c.concatenate([coeffs, _c.zeros(1, dtype=coeffs.dtype)])
        scaled = _c.concatenate([_c.zeros(1, dtype=coeffs.dtype), coeffs * zeros[i]])
        coeffs = shifted - scaled
    if coeffs.dtype == _c.complex128 and bool(_c.all(_c.imag(coeffs) == 0)):
        coeffs = _c.real(coeffs)
    return coeffs


def roots(p):
    """Roots as the eigenvalues of the companion matrix, as NumPy computes them."""
    coeffs = trim_zeros(_c.asarray(p, dtype=float), "f")
    while coeffs.size > 1 and float(coeffs[-1]) == 0.0:
        coeffs = coeffs[:-1]
    n = coeffs.size - 1
    if n < 1:
        return _c.zeros(0, dtype=_c.float64)
    companion = _c.zeros((n, n), dtype=_c.float64)
    companion[0, :] = -coeffs[1:] / coeffs[0]
    for i in range(1, n):
        companion[i, i - 1] = 1.0
    return _c.linalg.eigvals(companion)


def convolve(a, v, mode="full"):
    left = _c.asarray(a)
    right = _c.asarray(v)
    n, m = left.size, right.size
    dtype = _c.complex128 if _c.complex128 in (left.dtype, right.dtype) else _c.float64
    out = _c.zeros(n + m - 1, dtype=dtype)
    for i in range(n):
        out[i:i + m] += left[i] * right
    if mode == "full":
        return out
    if mode == "same":
        start = (m - 1) // 2
        return out[start:start + max(n, m)]
    return out[m - 1:n]


def polymul(a1, a2):
    return convolve(_c.asarray(a1, dtype=float), _c.asarray(a2, dtype=float))


def polyadd(a1, a2):
    left = _c.asarray(a1, dtype=float)
    right = _c.asarray(a2, dtype=float)
    n = max(left.size, right.size)
    out = _c.zeros(n, dtype=_c.float64)
    out[n - left.size:] += left
    out[n - right.size:] += right
    return out


def polysub(a1, a2):
    return polyadd(a1, -_c.asarray(a2, dtype=float))


def polydiv(u, v):
    """Synthetic division returning the quotient and remainder."""
    num = _c.asarray(u, dtype=float).astype(_c.float64, copy=True)
    den = _c.asarray(v, dtype=float)
    if den.size == 0 or float(den[0]) == 0.0:
        raise ZeroDivisionError("polydiv by a polynomial with a zero leading term")
    n = num.size - den.size + 1
    if n <= 0:
        return _c.zeros(1, dtype=_c.float64), num
    quotient = _c.zeros(n, dtype=_c.float64)
    for i in range(n):
        factor = float(num[i]) / float(den[0])
        quotient[i] = factor
        num[i:i + den.size] -= factor * den
    remainder = num[n:]
    return quotient, remainder


def polyint(p, m=1, k=0):
    coeffs = _c.asarray(p, dtype=float)
    for _ in range(m):
        n = coeffs.size
        out = _c.zeros(n + 1, dtype=_c.float64)
        for i in range(n):
            out[i] = float(coeffs[i]) / (n - i)
        out[n] = float(k) if not hasattr(k, "__len__") else float(k[0])
        coeffs = out
    return coeffs


def polyder(p, m=1):
    coeffs = _c.asarray(p, dtype=float)
    for _ in range(m):
        n = coeffs.size
        if n <= 1:
            return _c.zeros(1, dtype=_c.float64)
        out = _c.zeros(n - 1, dtype=_c.float64)
        for i in range(n - 1):
            out[i] = float(coeffs[i]) * (n - 1 - i)
        coeffs = out
    return coeffs


def trim_zeros(filt, trim="fb"):
    values = _c.asarray(filt)
    start, stop = 0, values.size
    if "f" in trim.lower():
        while start < stop and float(_c.absolute(values[start])) == 0.0:
            start += 1
    if "b" in trim.lower():
        while stop > start and float(_c.absolute(values[stop - 1])) == 0.0:
            stop -= 1
    return values[start:stop]


class _Chebyshev:
    """The one ``numpy.polynomial.chebyshev`` entry point the library uses."""

    @staticmethod
    def chebval(x, c):
        values = _c.asarray(x, dtype=float)
        coeffs = _c.asarray(c, dtype=float)
        n = coeffs.size
        if n == 1:
            return _c.full(values.shape, float(coeffs[0]), dtype=_c.float64)
        if n == 2:
            return coeffs[0] + coeffs[1] * values
        # Clenshaw recurrence, the same scheme NumPy uses.
        c0 = _c.full(values.shape, float(coeffs[n - 2]), dtype=_c.float64)
        c1 = _c.full(values.shape, float(coeffs[n - 1]), dtype=_c.float64)
        for i in range(3, n + 1):
            tmp = c0
            c0 = coeffs[n - i] - c1
            c1 = tmp + c1 * 2.0 * values
        return c0 + c1 * values


class _LegendreSeries:
    """The tiny slice of ``numpy.polynomial.legendre.Legendre`` used here."""

    def __init__(self, coefficients):
        self.coef = _c.asarray(coefficients, dtype=float)

    @classmethod
    def basis(cls, degree):
        coefficients = _c.zeros(degree + 1, dtype=_c.float64)
        coefficients[degree] = 1.0
        return cls(coefficients)

    def __call__(self, x):
        return legval(x, self.coef)


def legval(x, c):
    """Clenshaw recurrence for a Legendre series."""
    values = _c.asarray(x, dtype=float)
    coeffs = _c.asarray(c, dtype=float)
    n = coeffs.size
    if n == 1:
        return _c.full(values.shape, float(coeffs[0]), dtype=_c.float64)
    if n == 2:
        return coeffs[0] + coeffs[1] * values
    c0 = _c.full(values.shape, float(coeffs[n - 2]), dtype=_c.float64)
    c1 = _c.full(values.shape, float(coeffs[n - 1]), dtype=_c.float64)
    degree = n
    for i in range(3, n + 1):
        tmp = c0
        degree -= 1
        c0 = coeffs[n - i] - (c1 * (degree - 1)) / degree
        c1 = tmp + (c1 * values * (2 * degree - 1)) / degree
    return c0 + c1 * values


def leggauss(deg):
    """Gauss-Legendre nodes and weights on ``[-1, 1]``.

    Golub-Welsch: the nodes are the eigenvalues of the Jacobi matrix of the
    Legendre recurrence, and the weights come from the first component of each
    eigenvector.
    """
    if deg < 1:
        raise ValueError("deg must be a positive integer")
    if deg == 1:
        return _c.zeros(1, dtype=_c.float64), _c.full(1, 2.0, dtype=_c.float64)
    k = _c.arange(1, deg, dtype=float)
    off = k / _c.sqrt(4.0 * k * k - 1.0)
    jacobi = _c.zeros((deg, deg), dtype=_c.float64)
    for i in range(deg - 1):
        jacobi[i, i + 1] = float(off[i])
        jacobi[i + 1, i] = float(off[i])
    nodes, vectors = _c.linalg.eigh(jacobi)
    weights = 2.0 * vectors[0] ** 2
    return nodes, weights


class _Legendre:
    Legendre = _LegendreSeries
    legval = staticmethod(legval)
    leggauss = staticmethod(leggauss)


chebyshev = _Chebyshev()
legendre = _Legendre()
