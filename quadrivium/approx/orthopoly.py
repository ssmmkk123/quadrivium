"""Classical orthogonal polynomials and Gauss quadrature nodes.

Each family is generated from its three-term recurrence; nodes and weights come
from the Golub-Welsch eigenvalue algorithm applied to the Jacobi matrix, which
is both elegant and numerically stable.
"""

from __future__ import annotations

import numpy as np

from ..core.utils import as_vector

__all__ = [
    "legendre",
    "legendre_coefficients",
    "chebyshev_t",
    "chebyshev_u",
    "hermite_physicists",
    "hermite_probabilists",
    "laguerre",
    "generalized_laguerre",
    "jacobi_polynomial",
    "gegenbauer",
    "recurrence_coefficients",
    "golub_welsch",
    "gauss_legendre_nodes",
    "gauss_chebyshev_nodes",
    "gauss_hermite_nodes",
    "gauss_laguerre_nodes",
    "gauss_jacobi_nodes",
    "gauss_lobatto_nodes",
    "gauss_radau_nodes",
    "orthogonal_series_fit",
]


def legendre(n: int, x):
    """Legendre polynomial ``P_n`` by the three-term recurrence."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(1, n):
        p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
    return p1


def legendre_coefficients(n: int):
    """Monomial coefficients of ``P_n`` (highest degree first)."""
    c0 = np.array([1.0])
    if n == 0:
        return c0
    c1 = np.array([1.0, 0.0])
    for k in range(1, n):
        c0, c1 = c1, (np.polysub(np.polymul([(2 * k + 1) / (k + 1), 0.0], c1),
                                 np.polymul([k / (k + 1)], c0)))
    return c1


def chebyshev_t(n: int, x):
    """Chebyshev polynomial of the first kind ``T_n``."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    t0, t1 = np.ones_like(x), x.copy()
    for _ in range(1, n):
        t0, t1 = t1, 2 * x * t1 - t0
    return t1


def chebyshev_u(n: int, x):
    """Chebyshev polynomial of the second kind ``U_n``."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    u0, u1 = np.ones_like(x), 2 * x
    for _ in range(1, n):
        u0, u1 = u1, 2 * x * u1 - u0
    return u1


def hermite_physicists(n: int, x):
    """Physicists' Hermite polynomial ``H_n`` (weight ``exp(-x^2)``)."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    h0, h1 = np.ones_like(x), 2 * x
    for k in range(1, n):
        h0, h1 = h1, 2 * x * h1 - 2 * k * h0
    return h1


def hermite_probabilists(n: int, x):
    """Probabilists' Hermite polynomial ``He_n`` (weight ``exp(-x^2/2)``)."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    h0, h1 = np.ones_like(x), x.copy()
    for k in range(1, n):
        h0, h1 = h1, x * h1 - k * h0
    return h1


def laguerre(n: int, x):
    """Laguerre polynomial ``L_n`` (weight ``exp(-x)`` on ``[0, inf)``)."""
    return generalized_laguerre(n, 0.0, x)


def generalized_laguerre(n: int, alpha: float, x):
    """Generalized Laguerre polynomial ``L_n^alpha``."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    l0 = np.ones_like(x)
    l1 = 1.0 + alpha - x
    for k in range(1, n):
        l0, l1 = l1, ((2 * k + 1 + alpha - x) * l1 - (k + alpha) * l0) / (k + 1)
    return l1


def jacobi_polynomial(n: int, alpha: float, beta: float, x):
    """Jacobi polynomial ``P_n^(alpha,beta)``; generalizes Legendre and Chebyshev."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    p0 = np.ones_like(x)
    p1 = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
    for k in range(1, n):
        c1 = 2 * (k + 1) * (k + alpha + beta + 1) * (2 * k + alpha + beta)
        c2 = (2 * k + alpha + beta + 1) * (alpha**2 - beta**2)
        c3 = (2 * k + alpha + beta) * (2 * k + alpha + beta + 1) * (2 * k + alpha + beta + 2)
        c4 = 2 * (k + alpha) * (k + beta) * (2 * k + alpha + beta + 2)
        p0, p1 = p1, ((c2 + c3 * x) * p1 - c4 * p0) / c1
    return p1


def gegenbauer(n: int, alpha: float, x):
    """Gegenbauer (ultraspherical) polynomial ``C_n^alpha``."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    c0, c1 = np.ones_like(x), 2 * alpha * x
    for k in range(1, n):
        c0, c1 = c1, (2 * (k + alpha) * x * c1 - (k + 2 * alpha - 1) * c0) / (k + 1)
    return c1


def recurrence_coefficients(kind: str, n: int, alpha: float = 0.0, beta: float = 0.0):
    """Monic three-term recurrence coefficients ``(a, b)`` and the weight mass.

    The recurrence is ``p_{k+1} = (x - a_k) p_k - b_k p_{k-1}``; ``b_0`` holds
    the integral of the weight function.
    """
    k = np.arange(n, dtype=float)
    if kind == "legendre":
        a = np.zeros(n)
        b = np.zeros(n)
        b[0] = 2.0
        if n > 1:
            b[1:] = 1.0 / (4.0 - 1.0 / k[1:] ** 2)
    elif kind == "chebyshev":
        a = np.zeros(n)
        b = np.full(n, 0.25)
        b[0] = np.pi
        if n > 1:
            b[1] = 0.5
    elif kind == "hermite":  # physicists', weight exp(-x^2)
        a = np.zeros(n)
        b = np.zeros(n)
        b[0] = np.sqrt(np.pi)
        if n > 1:
            b[1:] = k[1:] / 2.0
    elif kind == "laguerre":  # weight x^alpha exp(-x)
        from ..special.functions import gamma as _gamma

        a = 2.0 * k + alpha + 1.0
        b = np.zeros(n)
        b[0] = _gamma(alpha + 1.0)
        if n > 1:
            b[1:] = k[1:] * (k[1:] + alpha)
    elif kind == "jacobi":
        from ..special.functions import beta as _beta

        ab = alpha + beta
        a = np.zeros(n)
        b = np.zeros(n)
        for i in range(n):
            if i == 0:
                a[i] = (beta - alpha) / (ab + 2.0)
            else:
                num = beta**2 - alpha**2
                den = (2.0 * i + ab) * (2.0 * i + ab + 2.0)
                a[i] = num / den if den != 0 else 0.0
        b[0] = 2.0 ** (ab + 1.0) * _beta(alpha + 1.0, beta + 1.0)
        for i in range(1, n):
            if i == 1:
                b[i] = 4.0 * (alpha + 1.0) * (beta + 1.0) / ((ab + 2.0) ** 2 * (ab + 3.0))
            else:
                b[i] = (4.0 * i * (i + alpha) * (i + beta) * (i + ab)
                        / ((2.0 * i + ab) ** 2 * (2.0 * i + ab + 1.0) * (2.0 * i + ab - 1.0)))
    else:
        raise ValueError(f"unknown family {kind!r}")
    return a, b


def golub_welsch(a, b):
    """Gauss nodes and weights from the Jacobi matrix eigendecomposition.

    ``a`` and ``b`` are the monic recurrence coefficients; ``b[0]`` is the
    total mass of the weight function.
    """
    a, b = as_vector(a), as_vector(b)
    n = a.size
    if n == 1:
        return np.array([a[0]]), np.array([b[0]])
    J = np.diag(a) + np.diag(np.sqrt(b[1:n]), 1) + np.diag(np.sqrt(b[1:n]), -1)
    vals, vecs = np.linalg.eigh(J)
    w = b[0] * vecs[0, :] ** 2
    idx = np.argsort(vals)
    return vals[idx], w[idx]


def gauss_legendre_nodes(n: int, a: float = -1.0, b: float = 1.0):
    """Gauss-Legendre nodes and weights, exact for degree ``2n-1``."""
    x, w = golub_welsch(*recurrence_coefficients("legendre", n))
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    return xm, w * 0.5 * (b - a)


def gauss_chebyshev_nodes(n: int, kind: int = 1):
    """Gauss-Chebyshev nodes and weights (closed form)."""
    if kind == 1:
        x = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
        w = np.full(n, np.pi / n)
    else:
        j = np.arange(1, n + 1)
        x = np.cos(j * np.pi / (n + 1))
        w = np.pi / (n + 1) * np.sin(j * np.pi / (n + 1)) ** 2
    idx = np.argsort(x)
    return x[idx], w[idx]


def gauss_hermite_nodes(n: int):
    """Gauss-Hermite nodes and weights for ``exp(-x^2)`` on the real line."""
    return golub_welsch(*recurrence_coefficients("hermite", n))


def gauss_laguerre_nodes(n: int, alpha: float = 0.0):
    """Gauss-Laguerre nodes and weights for ``x^alpha exp(-x)`` on ``[0, inf)``."""
    return golub_welsch(*recurrence_coefficients("laguerre", n, alpha))


def gauss_jacobi_nodes(n: int, alpha: float = 0.0, beta: float = 0.0):
    """Gauss-Jacobi nodes and weights for ``(1-x)^alpha (1+x)^beta``."""
    return golub_welsch(*recurrence_coefficients("jacobi", n, alpha, beta))


def gauss_lobatto_nodes(n: int, a: float = -1.0, b: float = 1.0):
    """Gauss-Lobatto nodes and weights; both endpoints are included."""
    if n < 2:
        raise ValueError("Lobatto rules need at least 2 nodes")
    if n == 2:
        x = np.array([-1.0, 1.0])
        w = np.array([1.0, 1.0])
    else:
        # interior nodes are the roots of P'_{n-1}
        m = n - 2
        a_c, b_c = recurrence_coefficients("jacobi", m, 1.0, 1.0)
        xi, _ = golub_welsch(a_c, b_c)
        x = np.concatenate([[-1.0], xi, [1.0]])
        w = np.zeros(n)
        for i in range(n):
            w[i] = 2.0 / ((n - 1) * n * legendre(n - 1, np.array([x[i]]))[0] ** 2)
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    return xm, w * 0.5 * (b - a)


def gauss_radau_nodes(n: int, a: float = -1.0, b: float = 1.0):
    """Gauss-Radau nodes and weights; the left endpoint is included."""
    if n < 2:
        raise ValueError("Radau rules need at least 2 nodes")
    m = n - 1
    a_c, b_c = recurrence_coefficients("jacobi", m, 0.0, 1.0)
    xi, _ = golub_welsch(a_c, b_c)
    x = np.concatenate([[-1.0], xi])
    w = np.zeros(n)
    w[0] = 2.0 / n**2
    for i in range(1, n):
        w[i] = (1.0 / n**2) * (1.0 - x[i]) / legendre(n - 1, np.array([x[i]]))[0] ** 2
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    return xm, w * 0.5 * (b - a)


def orthogonal_series_fit(f, n: int, family: str = "legendre", a: float = -1.0,
                          b: float = 1.0):
    """Least squares projection of ``f`` onto an orthogonal polynomial basis.

    Coefficients are computed by Gauss quadrature, so the fit is the truncated
    orthogonal series -- the best ``L^2`` approximation of that degree.
    """
    if family == "legendre":
        x, w = gauss_legendre_nodes(n + 8, -1.0, 1.0)
        basis = legendre
        norms = np.array([2.0 / (2 * k + 1) for k in range(n + 1)])
    elif family == "chebyshev":
        x, w = gauss_chebyshev_nodes(n + 8)
        basis = chebyshev_t
        norms = np.array([np.pi if k == 0 else np.pi / 2 for k in range(n + 1)])
    else:
        raise ValueError("family must be 'legendre' or 'chebyshev'")
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    fx = np.array([f(ti) for ti in t])
    coeffs = np.array([(w * fx * basis(k, x)).sum() / norms[k] for k in range(n + 1)])

    def series(q):
        s = (2.0 * np.asarray(q, dtype=float) - (a + b)) / (b - a)
        return sum(coeffs[k] * basis(k, s) for k in range(n + 1))

    series.coefficients = coeffs
    return series
