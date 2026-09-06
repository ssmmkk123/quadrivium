"""Polynomial interpolation.

All constructors return callables that evaluate the interpolant, so they can be
passed straight to the quadrature and ODE routines.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.exceptions import DimensionError, DomainError
from ..core.utils import as_vector, unwrap_scalar

__all__ = [
    "lagrange",
    "lagrange_coefficients",
    "newton_divided_differences",
    "divided_difference_table",
    "newton_forward",
    "newton_backward",
    "neville",
    "barycentric",
    "barycentric_weights",
    "hermite",
    "chebyshev_nodes",
    "chebyshev_interpolation",
    "vandermonde_interpolation",
    "runge_demo_error",
    "interpolation_error_bound",
]


def _check_nodes(x, y):
    x, y = as_vector(x), as_vector(y)
    if x.size != y.size:
        raise DimensionError(f"x has {x.size} nodes but y has {y.size} values")
    if np.unique(x).size != x.size:
        raise ValueError("interpolation nodes must be distinct")
    return x, y


def lagrange(x, y):
    """Lagrange form of the interpolating polynomial.

    O(n^2) per evaluation; use :func:`barycentric` for repeated evaluation.
    """
    x, y = _check_nodes(x, y)
    n = x.size
    # The node differences do not depend on the evaluation point, so the table
    # is built once here instead of n times inside every basis function. The
    # inner loop divides by these rather than multiplying by stored
    # reciprocals: for closely spaced nodes the partial products reach the
    # edge of the double range, and there the two are not the same number.
    den = x[:, None] - x[None, :]

    def p(t):
        t = np.asarray(t, dtype=float)
        total = np.zeros(t.shape)
        # Three scratch arrays the size of the query, reused by every one of
        # the n^2 factors. The arithmetic is the textbook product; what the
        # buffers remove is the three temporaries each factor would otherwise
        # allocate, which is most of the cost at this size.
        term = np.empty(t.shape)
        shifted = np.empty(t.shape)
        for i in range(n):
            term.fill(y[i])
            for j in range(n):
                if j != i:
                    # Numerator and denominator stay interleaved: each factor
                    # is O(1), where accumulating the two products separately
                    # would underflow one and overflow the other long before
                    # their ratio left the double range.
                    np.subtract(t, x[j], out=shifted)
                    term *= shifted
                    term /= den[i, j]
            total += term
        return unwrap_scalar(total)

    return p


def lagrange_coefficients(x, y):
    """Monomial coefficients (highest degree first) of the Lagrange interpolant."""
    x, y = _check_nodes(x, y)
    n = x.size
    coeffs = np.zeros(n)
    for i in range(n):
        others = np.delete(x, i)
        basis = np.poly(others) if others.size else np.array([1.0])
        denom = np.prod(x[i] - others) if others.size else 1.0
        coeffs = coeffs + y[i] * basis / denom
    return coeffs


def divided_difference_table(x, y):
    """Full divided difference table; the top row holds the Newton coefficients."""
    x, y = _check_nodes(x, y)
    n = x.size
    table = np.zeros((n, n))
    table[:, 0] = y
    # Column j depends only on column j-1, and every entry within a column is
    # independent, so each column is one vector operation.
    for j in range(1, n):
        k = n - j
        table[:k, j] = ((table[1:k + 1, j - 1] - table[:k, j - 1])
                        / (x[j:j + k] - x[:k]))
    return table


def newton_divided_differences(x, y):
    """Newton's divided difference form; O(n) per evaluation after setup."""
    x, y = _check_nodes(x, y)
    coeffs = divided_difference_table(x, y)[0]

    def p(t):
        t = np.asarray(t, dtype=float)
        result = np.full_like(t, coeffs[-1], dtype=float)
        for k in range(len(coeffs) - 2, -1, -1):
            result = result * (t - x[k]) + coeffs[k]
        return unwrap_scalar(result)

    p.coefficients = coeffs
    return p


def newton_forward(x, y):
    """Newton's forward difference formula for equally spaced nodes."""
    x, y = _check_nodes(x, y)
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("newton_forward requires equally spaced nodes")
    n = x.size
    diff = y.copy()
    coeffs = [diff[0]]
    for k in range(1, n):
        diff = np.diff(diff)
        coeffs.append(diff[0])
    coeffs = np.array(coeffs)

    def p(t):
        t = np.asarray(t, dtype=float)
        s = (t - x[0]) / h
        total = np.full_like(t, coeffs[0], dtype=float)
        term = np.ones_like(t, dtype=float)
        for k in range(1, n):
            term = term * (s - (k - 1)) / k
            total = total + coeffs[k] * term
        return unwrap_scalar(total)

    return p


def newton_backward(x, y):
    """Newton's backward difference formula for equally spaced nodes."""
    x, y = _check_nodes(x, y)
    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h):
        raise ValueError("newton_backward requires equally spaced nodes")
    n = x.size
    diff = y.copy()
    coeffs = [diff[-1]]
    for k in range(1, n):
        diff = np.diff(diff)
        coeffs.append(diff[-1])
    coeffs = np.array(coeffs)

    def p(t):
        t = np.asarray(t, dtype=float)
        s = (t - x[-1]) / h
        total = np.full_like(t, coeffs[0], dtype=float)
        term = np.ones_like(t, dtype=float)
        for k in range(1, n):
            term = term * (s + (k - 1)) / k
            total = total + coeffs[k] * term
        return unwrap_scalar(total)

    return p


def neville(x, y, t):
    """Neville's tableau: evaluate the interpolant at ``t`` without coefficients.

    Returns ``(value, table)``; the table's diagonal shows convergence in degree.
    """
    x, y = _check_nodes(x, y)
    n = x.size
    Q = np.zeros((n, n))
    Q[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            Q[i, j] = ((t - x[i + j]) * Q[i, j - 1] + (x[i] - t) * Q[i + 1, j - 1]) / (x[i] - x[i + j])
    return Q[0, n - 1], Q


def barycentric_weights(x):
    """Barycentric weights ``w_j = 1 / prod_{k != j}(x_j - x_k)``.

    The differences are divided by the interval's logarithmic capacity before
    the product is taken. Barycentric interpolation is invariant under a common
    factor on the weights, so this changes no result -- but the raw product
    runs like ``capacity^n``, which reaches 1e117 for 400 Chebyshev nodes on
    ``[-1, 1]`` and overflows outright on ``[0, 1]``, where every weight comes
    back ``inf`` and every interpolated value ``nan``. Scaling makes the same
    problem behave the same way under any affine change of variable.
    """
    x = as_vector(x)
    n = x.size
    if n <= 1:
        return np.ones(n)
    span = float(x.max() - x.min())
    scale = 0.25 * span if span > 0.0 and np.isfinite(span) else 1.0
    w = np.empty(n)
    # Blocked so the (nodes x nodes) difference table never has to exist all
    # at once for a large node set.
    block = max(1, (1 << 18) // n)
    for start in range(0, n, block):
        stop = min(start + block, n)
        D = (x[start:stop, None] - x[None, :]) / scale
        # The k == j factor is omitted from the product, not set to zero.
        D[np.arange(stop - start), np.arange(start, stop)] = 1.0
        w[start:stop] = 1.0 / np.prod(D, axis=1)
    return w


def barycentric(x, y, weights=None):
    """Second-form barycentric interpolation: O(n) per point and stable."""
    x, y = _check_nodes(x, y)
    w = barycentric_weights(x) if weights is None else as_vector(weights)

    def p(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
        out = np.empty(t_arr.size)
        # All evaluation points share one formula, so they are handled as a
        # block: the two barycentric sums become a matrix-vector product and a
        # row sum. Blocking keeps the (points x nodes) table bounded no matter
        # how many points are asked for at once.
        block = max(1, (1 << 18) // max(x.size, 1))
        for start in range(0, t_arr.size, block):
            ts = t_arr[start:start + block]
            diff = ts[:, None] - x
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = w / diff
                chunk = (terms @ y) / terms.sum(axis=1)
            # A point sitting exactly on a node divides by zero and shows up
            # as a non-finite result; its limit is that node's value. Testing
            # the n results is O(points), where scanning the difference table
            # for exact zeros would be O(points x nodes).
            bad = np.flatnonzero(~np.isfinite(chunk))
            for i in bad:
                hit = np.flatnonzero(diff[i] == 0.0)
                chunk[i] = y[hit[0]] if hit.size else chunk[i]
            out[start:start + block] = chunk
        return out[0] if np.ndim(t) == 0 else out.reshape(np.shape(t))

    p.weights = w
    return p


def hermite(x, y, dy):
    """Hermite interpolation matching values and first derivatives.

    Produces the degree ``2n-1`` osculating polynomial.
    """
    x, y, dy = as_vector(x), as_vector(y), as_vector(dy)
    if not (x.size == y.size == dy.size):
        raise DimensionError("x, y and dy must have equal length")
    n = x.size
    z = np.repeat(x, 2)
    Q = np.zeros((2 * n, 2 * n))
    Q[:, 0] = np.repeat(y, 2)
    # First divided-difference column: a repeated node contributes the
    # derivative, a genuine pair the usual difference quotient.
    for i in range(n):
        Q[2 * i, 1] = dy[i]
        if i < n - 1:
            Q[2 * i + 1, 1] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    for j in range(2, 2 * n):
        for i in range(2 * n - j):
            Q[i, j] = (Q[i + 1, j - 1] - Q[i, j - 1]) / (z[i + j] - z[i])
    coeffs = Q[0, :].copy()

    def p(t):
        t = np.asarray(t, dtype=float)
        result = np.full_like(t, coeffs[-1], dtype=float)
        for k in range(2 * n - 2, -1, -1):
            result = result * (t - z[k]) + coeffs[k]
        return unwrap_scalar(result)

    p.coefficients = coeffs
    p.nodes = z
    return p


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0, kind: int = 1):
    """Chebyshev nodes on ``[a, b]``.

    ``kind=1`` gives the roots of ``T_n`` (open); ``kind=2`` the extrema
    (Chebyshev-Lobatto, including both endpoints).
    """
    if kind == 1:
        t = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
    elif kind == 2:
        t = np.cos(np.arange(n) * np.pi / (n - 1)) if n > 1 else np.array([0.0])
    else:
        raise ValueError("kind must be 1 (roots) or 2 (extrema)")
    return 0.5 * (a + b) + 0.5 * (b - a) * np.sort(t)


def chebyshev_interpolation(f, n: int, a: float = -1.0, b: float = 1.0, kind: int = 2):
    """Interpolate ``f`` at Chebyshev nodes -- near-optimal, no Runge phenomenon."""
    x = chebyshev_nodes(n, a, b, kind)
    y = np.array([f(xi) for xi in x])
    if kind == 2:
        # closed-form barycentric weights for Chebyshev-Lobatto points
        w = np.ones(n)
        w[1::2] = -1.0
        w[0] *= 0.5
        w[-1] *= 0.5
        return barycentric(x, y, w)
    return barycentric(x, y)


def vandermonde_interpolation(x, y):
    """Solve the Vandermonde system for the monomial coefficients.

    Included for completeness: the matrix is notoriously ill-conditioned, so
    prefer Newton or barycentric forms in practice.
    """
    x, y = _check_nodes(x, y)
    n = x.size
    V = np.vander(x, n)
    return np.linalg.solve(V, y)


def interpolation_error_bound(x, max_derivative: float, t=None):
    """Bound ``|f(t) - p(t)| <= max|f^(n)| / n! * prod|t - x_i|``."""
    x = as_vector(x)
    n = x.size
    from math import factorial

    def bound(t):
        t = np.asarray(t, dtype=float)
        prod = np.ones_like(t, dtype=float)
        for xi in x:
            prod = prod * np.abs(t - xi)
        return max_derivative / factorial(n) * prod

    return bound if t is None else bound(t)


def runge_demo_error(n: int, nodes: str = "equispaced"):
    """Max error interpolating the Runge function ``1/(1+25x^2)`` on ``[-1,1]``.

    Illustrates why equispaced nodes diverge as ``n`` grows while Chebyshev
    nodes converge.
    """
    f = lambda t: 1.0 / (1.0 + 25.0 * t**2)
    x = np.linspace(-1, 1, n) if nodes == "equispaced" else chebyshev_nodes(n, -1, 1, kind=2)
    p = barycentric(x, f(x))
    t = np.linspace(-1, 1, 2001)
    return float(np.max(np.abs(p(t) - f(t))))
