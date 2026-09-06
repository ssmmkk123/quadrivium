"""Least squares fitting, Pade approximants, minimax and Fourier series."""

from __future__ import annotations

from .. import numeric as np

from ..core.utils import as_vector

__all__ = [
    "polyfit",
    "weighted_polyfit",
    "chebyshev_fit",
    "legendre_fit",
    "exponential_fit",
    "power_fit",
    "logarithmic_fit",
    "rational_fit",
    "pade",
    "pade_evaluate",
    "remez",
    "minimax_polynomial",
    "fourier_series",
    "fourier_coefficients",
    "trigonometric_fit",
    "spline_fit",
    "aaa",
    "chebyshev_economization",
]


def polyfit(x, y, degree: int = 1):
    """Least squares polynomial fit; returns coefficients highest degree first.

    Solved through QR on the Vandermonde matrix rather than the normal
    equations, which halves the loss of conditioning.
    """
    from ..linalg.lstsq import qr_least_squares

    x, y = as_vector(x), as_vector(y)
    A = np.vander(x, degree + 1)
    return qr_least_squares(A, y)


def weighted_polyfit(x, y, weights, degree: int = 1):
    """Weighted least squares polynomial fit."""
    from ..linalg.lstsq import weighted_least_squares

    x, y = as_vector(x), as_vector(y)
    return weighted_least_squares(np.vander(x, degree + 1), y, weights)


def chebyshev_fit(x, y, degree: int = 5, domain=None):
    """Least squares fit in the Chebyshev basis.

    Far better conditioned than the monomial basis at high degree.
    """
    from ..approx.orthopoly import chebyshev_t
    from ..linalg.lstsq import qr_least_squares

    x, y = as_vector(x), as_vector(y)
    a, b = (x.min(), x.max()) if domain is None else domain
    t = (2 * x - (a + b)) / (b - a)
    A = np.column_stack([chebyshev_t(k, t) for k in range(degree + 1)])
    coeffs = qr_least_squares(A, y)

    def f(q):
        s = (2 * np.asarray(q, dtype=float) - (a + b)) / (b - a)
        return sum(coeffs[k] * chebyshev_t(k, s) for k in range(degree + 1))

    f.coefficients = coeffs
    return f


def legendre_fit(x, y, degree: int = 5, domain=None):
    """Least squares fit in the Legendre basis."""
    from ..approx.orthopoly import legendre
    from ..linalg.lstsq import qr_least_squares

    x, y = as_vector(x), as_vector(y)
    a, b = (x.min(), x.max()) if domain is None else domain
    t = (2 * x - (a + b)) / (b - a)
    A = np.column_stack([legendre(k, t) for k in range(degree + 1)])
    coeffs = qr_least_squares(A, y)

    def f(q):
        s = (2 * np.asarray(q, dtype=float) - (a + b)) / (b - a)
        return sum(coeffs[k] * legendre(k, s) for k in range(degree + 1))

    f.coefficients = coeffs
    return f


def exponential_fit(x, y):
    """Fit ``y = a exp(b x)`` by linearizing in ``log y``.

    Requires positive ``y``; note that the linearization weights small values
    more heavily than a direct nonlinear fit would.
    """
    x, y = as_vector(x), as_vector(y)
    if np.any(y <= 0):
        raise ValueError("exponential_fit requires strictly positive y")
    c = polyfit(x, np.log(y), 1)
    return float(np.exp(c[1])), float(c[0])


def power_fit(x, y):
    """Fit ``y = a x^b`` by linearizing in log-log coordinates."""
    x, y = as_vector(x), as_vector(y)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power_fit requires strictly positive x and y")
    c = polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(c[1])), float(c[0])


def logarithmic_fit(x, y):
    """Fit ``y = a + b log x``."""
    x, y = as_vector(x), as_vector(y)
    c = polyfit(np.log(x), y, 1)
    return float(c[1]), float(c[0])


def rational_fit(x, y, m: int = 2, n: int = 2):
    """Linearized rational least squares fit ``P_m(x) / Q_n(x)`` with ``Q(0)=1``."""
    from ..linalg.lstsq import qr_least_squares

    x, y = as_vector(x), as_vector(y)
    A = np.column_stack([x**j for j in range(m + 1)]
                        + [-y * x**j for j in range(1, n + 1)])
    c = qr_least_squares(A, y)
    p = c[: m + 1]
    q = np.concatenate([[1.0], c[m + 1 :]])

    def f(t):
        t = np.asarray(t, dtype=float)
        return (sum(p[j] * t**j for j in range(m + 1))
                / sum(q[j] * t**j for j in range(n + 1)))

    f.numerator, f.denominator = p, q
    return f


def pade(coeffs, m: int, n: int):
    """Pade approximant ``[m/n]`` from Taylor coefficients (lowest order first).

    Often converges where the Taylor series diverges, because the poles of the
    rational form can represent nearby singularities.

    The Pade table can be *degenerate*: when a lower-order approximant already
    reproduces the series exactly (as ``[1/1]`` does for ``1/(1-x)``), the
    linear system for a higher-order one is singular. A minimum-norm solution
    is returned in that case, which still reproduces the series.
    """
    c = as_vector(coeffs)
    if c.size < m + n + 1:
        raise ValueError(f"need at least {m + n + 1} Taylor coefficients")
    # solve for the denominator from the linear system of the Pade conditions
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    for i in range(n):
        for j in range(n):
            idx = m + i - j
            A[i, j] = c[idx] if 0 <= idx < c.size else 0.0
        rhs[i] = -c[m + i + 1]
    if n > 0:
        try:
            b = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # degenerate block in the Pade table: take the minimum-norm solution
            b = np.linalg.lstsq(A, rhs, rcond=None)[0]
    else:
        b = np.zeros(0)
    q = np.concatenate([[1.0], b])
    p = np.zeros(m + 1)
    for i in range(m + 1):
        p[i] = c[i] + sum(q[j] * c[i - j] for j in range(1, min(i, n) + 1))
    return p, q


def pade_evaluate(p, q, x):
    """Evaluate a Pade approximant given its numerator and denominator."""
    x = np.asarray(x, dtype=float)
    num = sum(p[j] * x**j for j in range(len(p)))
    den = sum(q[j] * x**j for j in range(len(q)))
    return num / den


def remez(f, degree: int, a: float = -1.0, b: float = 1.0, max_iter: int = 60,
          tol: float = 1e-13):
    """Remez exchange algorithm for the minimax polynomial approximation.

    Iterates toward the equioscillation property that characterizes the best
    uniform approximation: the error attains its maximum with alternating sign
    at ``degree + 2`` points.
    """
    from ..interpolate.polynomial import chebyshev_nodes

    n = degree + 2
    x = chebyshev_nodes(n, a, b, kind=2)
    coeffs = np.zeros(degree + 1)
    E = 0.0
    for it in range(max_iter):
        # solve for the polynomial and the levelled error
        A = np.zeros((n, n))
        rhs = np.array([f(xi) for xi in x])
        for i in range(n):
            A[i, :degree + 1] = [x[i] ** j for j in range(degree + 1)]
            A[i, degree + 1] = (-1.0) ** i
        sol = np.linalg.solve(A, rhs)
        coeffs, E = sol[: degree + 1], sol[degree + 1]
        # find the new extrema of the error on a fine grid
        grid = np.linspace(a, b, 2000)
        err = np.array([f(t) for t in grid]) - np.array(
            [sum(coeffs[j] * t**j for j in range(degree + 1)) for t in grid])
        extrema = [0]
        for i in range(1, len(grid) - 1):
            if (err[i] - err[i - 1]) * (err[i + 1] - err[i]) < 0:
                extrema.append(i)
        extrema.append(len(grid) - 1)
        # keep the n largest alternating extrema
        extrema = sorted(extrema, key=lambda i: -abs(err[i]))[:n]
        x_new = np.sort(grid[sorted(extrema)])
        if len(x_new) < n:
            break
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    def approx(t):
        t = np.asarray(t, dtype=float)
        return sum(coeffs[j] * t**j for j in range(degree + 1))

    approx.coefficients = coeffs
    approx.error = abs(E)
    approx.nodes = x
    return approx


def minimax_polynomial(f, degree: int, a: float = -1.0, b: float = 1.0, **kwargs):
    """Best uniform (minimax) polynomial approximation via :func:`remez`."""
    return remez(f, degree, a, b, **kwargs)


def fourier_coefficients(f, n: int = 10, period: float = 2 * np.pi,
                         n_quad: int = 2000):
    """Fourier coefficients ``(a_0, a_k, b_k)`` of a periodic function."""
    L = period
    t = np.linspace(0, L, n_quad, endpoint=False)
    ft = np.array([f(ti) for ti in t])
    a0 = 2.0 * np.mean(ft)
    a = np.array([2.0 * np.mean(ft * np.cos(2 * np.pi * k * t / L))
                  for k in range(1, n + 1)])
    b = np.array([2.0 * np.mean(ft * np.sin(2 * np.pi * k * t / L))
                  for k in range(1, n + 1)])
    return a0, a, b


def fourier_series(f, n: int = 10, period: float = 2 * np.pi):
    """Truncated Fourier series of a periodic function."""
    L = period
    a0, a, b = fourier_coefficients(f, n, L)

    def s(t):
        t = np.asarray(t, dtype=float)
        out = np.full_like(t, a0 / 2.0)
        for k in range(1, n + 1):
            out = out + (a[k - 1] * np.cos(2 * np.pi * k * t / L)
                         + b[k - 1] * np.sin(2 * np.pi * k * t / L))
        return out

    s.a0, s.a, s.b = a0, a, b
    return s


def trigonometric_fit(x, y, n_harmonics: int = 3, period: float = 2 * np.pi):
    """Least squares fit of a truncated trigonometric series to scattered data."""
    from ..linalg.lstsq import qr_least_squares

    x, y = as_vector(x), as_vector(y)
    cols = [np.ones_like(x)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * k * x / period))
        cols.append(np.sin(2 * np.pi * k * x / period))
    A = np.column_stack(cols)
    c = qr_least_squares(A, y)

    def f(t):
        t = np.asarray(t, dtype=float)
        out = np.full_like(t, c[0])
        for k in range(1, n_harmonics + 1):
            out = (out + c[2 * k - 1] * np.cos(2 * np.pi * k * t / period)
                   + c[2 * k] * np.sin(2 * np.pi * k * t / period))
        return out

    f.coefficients = c
    return f


def spline_fit(x, y, smoothing: float = 1.0):
    """Smoothing spline fit to noisy data."""
    from ..interpolate.spline import smoothing_spline

    return smoothing_spline(x, y, lam=smoothing)


def aaa(f, points=None, tol: float = 1e-13, max_terms: int = 100,
        values=None):
    """AAA algorithm: near-best rational approximation in barycentric form.

    Greedily adds support points where the current approximation is worst, and
    solves a small least-squares problem for the barycentric weights at each
    step.  Two properties make it the modern default for rational fitting:
    the barycentric form is numerically stable where a ratio of explicit
    polynomials is not, and the greedy support-point choice sidesteps the
    nonlinear optimization that classical Pade and minimax fitting require.

    Handles poles, branch points and near-singular behaviour that polynomial
    approximation cannot touch.  Pass either a callable ``f`` with sample
    ``points``, or ``points`` together with ``values``.

    Returns ``(r, support, weights, fvals)`` where ``r`` is the evaluator.
    """
    if points is None:
        points = np.linspace(-1.0, 1.0, 1000)
    Z = np.asarray(points, dtype=float if not np.iscomplexobj(points) else complex)
    F = np.asarray(values if values is not None else [f(z) for z in Z])
    Z, F = Z.astype(F.dtype if np.iscomplexobj(F) else float), F.astype(
        complex if np.iscomplexobj(F) else float)
    m = Z.size
    J = list(range(m))            # indices not yet chosen as support points
    zj = np.array([], dtype=Z.dtype)
    fj = np.array([], dtype=F.dtype)
    wj = np.array([], dtype=F.dtype)
    C = np.empty((m, 0), dtype=F.dtype)
    R = np.full(m, np.mean(F))
    errors = []
    for _ in range(max_terms):
        j = int(np.argmax(np.abs(F - R)))
        if j not in J:
            break
        zj = np.append(zj, Z[j])
        fj = np.append(fj, F[j])
        J.remove(j)
        if not J:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            col = 1.0 / (Z - Z[j])
        col[j] = 0.0
        C = np.hstack([C, col[:, None]])
        # Loewner matrix: the weights are its smallest right singular vector,
        # which is what makes the interpolation conditions hold exactly at the
        # support points while minimizing the residual elsewhere.
        A = (F[J, None] * C[J, :] - C[J, :] * fj[None, :])
        if A.shape[0] < A.shape[1]:
            break
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        wj = Vh[-1, :].conj()
        num = C @ (wj * fj)
        den = C @ wj
        R = F.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            R[J] = num[J] / den[J]
        err = float(np.max(np.abs(F - R)))
        errors.append(err)
        if err <= tol * float(np.max(np.abs(F))):
            break

    zj_, wj_, fj_ = zj.copy(), wj.copy(), fj.copy()

    def r(x):
        """Barycentric evaluation, exact at the support points."""
        x = np.atleast_1d(np.asarray(x))
        out = np.zeros(x.shape, dtype=complex if np.iscomplexobj(fj_) else float)
        with np.errstate(divide="ignore", invalid="ignore"):
            D = x[..., None] - zj_
            # Support points would divide by zero; the limit there is f itself.
            exact = np.isclose(D, 0.0)
            safe = np.where(exact, 1.0, D)
            num = np.sum(wj_ * fj_ / safe, axis=-1)
            den = np.sum(wj_ / safe, axis=-1)
            out = num / den
        hit = np.any(exact, axis=-1)
        if np.any(hit):
            idx = np.argmax(exact[hit], axis=-1)
            out[hit] = fj_[idx]
        return out[0] if out.size == 1 and np.ndim(x) == 0 else out

    return r, zj_, wj_, fj_


def chebyshev_economization(coeffs, n: int, a: float = -1.0, b: float = 1.0):
    """Reduce a truncated power series to lower degree with minimal added error.

    Convert to the Chebyshev basis, drop the top coefficients, convert back.
    Because Chebyshev polynomials equioscillate, discarding the ``T_k`` term
    adds an error of exactly ``|c_k|`` spread evenly over the interval --
    against the strongly endpoint-weighted error of simply truncating the
    power series.  Returns the shortened power-basis coefficients.
    """
    c = np.asarray(coeffs, dtype=float)
    deg = c.size - 1
    if n >= deg:
        return c.copy()
    # Sample and fit in the Chebyshev basis on [a, b].
    k = np.arange(deg + 1)
    x = np.cos(np.pi * (k + 0.5) / (deg + 1))
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    y = np.polyval(c[::-1], t)
    cheb = np.zeros(deg + 1)
    for j in range(deg + 1):
        cheb[j] = (2.0 / (deg + 1)) * np.sum(y * np.cos(np.pi * j * (k + 0.5)
                                                        / (deg + 1)))
    cheb[0] *= 0.5
    cheb[n + 1:] = 0.0            # the economization step
    # Re-expand: evaluate the truncated Chebyshev series and refit in powers.
    tt = np.linspace(a, b, 4 * (n + 1) + 20)
    xx = (2 * tt - (a + b)) / (b - a)
    vals = np.polynomial.chebyshev.chebval(xx, cheb)
    return np.polyfit(tt, vals, n)
