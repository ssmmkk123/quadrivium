"""Special functions used throughout the library.

Implemented from scratch (Lanczos, continued fractions, series/asymptotic
switching) so the package depends on nothing beyond NumPy.
"""

from __future__ import annotations

import math

import numpy as np

from .. import _accel

# The array-valued functions below evaluate their scalar kernel through a
# Python loop, which is where nearly all of their runtime goes. Each one first
# offers the work to the compiled backend, which runs the identical
# approximation -- the same Lanczos table, the same reflection -- over the whole
# array at once. `_accel.kernel` returns None when no extension is loaded, and
# the loop underneath runs instead.

from ..core.exceptions import DomainError

__all__ = [
    "gamma",
    "log_gamma",
    "digamma",
    "beta",
    "log_beta",
    "factorial",
    "binomial",
    "erf",
    "erfc",
    "erfinv",
    "incomplete_gamma_lower",
    "incomplete_gamma_upper",
    "regularized_gamma_p",
    "regularized_gamma_q",
    "incomplete_beta",
    "bessel_j0",
    "bessel_j1",
    "bessel_jn",
    "bessel_y0",
    "bessel_y1",
    "bessel_yn",
    "bessel_i0",
    "bessel_i1",
    "bessel_in",
    "bessel_k0",
    "bessel_k1",
    "bessel_kn",
    "airy_ai",
    "airy_bi",
    "elliptic_k",
    "elliptic_e",
    "exponential_integral",
    "sine_integral",
    "cosine_integral",
    "zeta",
    "lambert_w",
    "dawson",
    "fresnel_s",
    "fresnel_c",
    "erfcx",
    "polygamma",
    "trigamma",
    "spherical_bessel_j",
    "spherical_bessel_y",
    "associated_legendre",
    "spherical_harmonic",
    "hyp1f1",
    "hyp2f1",
    "expint_n",
    "struve_h0",
    "logistic",
    "logit",
]

_LANCZOS_G = 7
_LANCZOS_COEF = np.array([
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
])


def log_gamma(x):
    """Log-gamma by the Lanczos approximation (accurate to ~15 digits)."""
    fast = _accel.kernel("log_gamma")
    if fast is not None:
        xa = np.asarray(x, dtype=float)
        # The kernel takes a flat array; ravel/reshape restores the caller's
        # shape, and a 0-d input keeps returning a plain float.
        out = fast(np.ascontiguousarray(xa).ravel()).reshape(xa.shape)
        return float(out) if xa.ndim == 0 else out
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x).astype(float)
    out = np.empty_like(x)
    for i, v in enumerate(x):
        if v < 0.5:
            # reflection formula
            out[i] = math.log(math.pi / abs(math.sin(math.pi * v))) - log_gamma(1.0 - v)
        else:
            z = v - 1.0
            a = _LANCZOS_COEF[0] + np.sum(_LANCZOS_COEF[1:] / (z + np.arange(1, 9)))
            t = z + _LANCZOS_G + 0.5
            out[i] = 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(a)
    return float(out[0]) if scalar else out


def gamma(x):
    """Gamma function for real arguments."""
    fast = _accel.kernel("gamma")
    if fast is not None:
        xa = np.asarray(x, dtype=float)
        # The kernel takes a flat array; ravel/reshape restores the caller's
        # shape, and a 0-d input keeps returning a plain float.
        out = fast(np.ascontiguousarray(xa).ravel()).reshape(xa.shape)
        return float(out) if xa.ndim == 0 else out
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    xa = np.atleast_1d(x).astype(float)
    out = np.empty_like(xa)
    for i, v in enumerate(xa):
        if v == np.floor(v) and v <= 0:
            out[i] = np.inf
        elif v < 0.5:
            out[i] = math.pi / (math.sin(math.pi * v) * gamma(1.0 - v))
        else:
            out[i] = math.exp(log_gamma(v))
    return float(out[0]) if scalar else out


def factorial(n):
    """Factorial via the gamma function (exact for small integers)."""
    n_arr = np.asarray(n)
    if n_arr.ndim == 0:
        return float(math.factorial(int(n))) if n == int(n) and n >= 0 else gamma(n + 1.0)
    return np.array([factorial(v) for v in n_arr.ravel()]).reshape(n_arr.shape)


def binomial(n, k):
    """Binomial coefficient, computed in log space for large arguments."""
    if n == int(n) and k == int(k) and 0 <= k <= n and n < 1000:
        return float(math.comb(int(n), int(k)))
    return math.exp(log_gamma(n + 1) - log_gamma(k + 1) - log_gamma(n - k + 1))


def beta(a, b):
    """Euler beta function ``B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)``."""
    return math.exp(log_beta(a, b))


def log_beta(a, b):
    """Logarithm of the beta function."""
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


def digamma(x):
    """Digamma ``psi(x) = d/dx log Gamma(x)`` by recurrence plus asymptotics."""
    x = float(x)
    result = 0.0
    # Recur upward until the asymptotic series is accurate to full precision.
    while x < 12.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    series = inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252
             - inv2 * (1.0 / 240 - inv2 * (1.0 / 132 - inv2 * 691.0 / 32760)))))
    return result + math.log(x) - 0.5 * inv - series


def erf(x):
    """Error function via its relation to the incomplete gamma function."""
    fast = _accel.kernel("erf")
    if fast is not None:
        xa = np.asarray(x, dtype=float)
        # The kernel takes a flat array; ravel/reshape restores the caller's
        # shape, and a 0-d input keeps returning a plain float.
        out = fast(np.ascontiguousarray(xa).ravel()).reshape(xa.shape)
        return float(out) if xa.ndim == 0 else out
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    xa = np.atleast_1d(x)
    out = np.array([math.erf(float(v)) for v in xa])
    return float(out[0]) if scalar else out


def erfc(x):
    """Complementary error function ``1 - erf(x)``, accurate in the tail."""
    fast = _accel.kernel("erfc")
    if fast is not None:
        xa = np.asarray(x, dtype=float)
        # The kernel takes a flat array; ravel/reshape restores the caller's
        # shape, and a 0-d input keeps returning a plain float.
        out = fast(np.ascontiguousarray(xa).ravel()).reshape(xa.shape)
        return float(out) if xa.ndim == 0 else out
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    xa = np.atleast_1d(x)
    out = np.array([math.erfc(float(v)) for v in xa])
    return float(out[0]) if scalar else out


def erfinv(y, tol: float = 1e-14, max_iter: int = 100):
    """Inverse error function by Newton iteration on a rational initial guess."""
    y = float(y)
    if abs(y) >= 1.0:
        if abs(y) == 1.0:
            return math.copysign(np.inf, y)
        raise DomainError("erfinv requires |y| < 1")
    # Winitzki's initial approximation
    a = 0.147
    ln1 = math.log(1 - y * y) if abs(y) < 1 else -np.inf
    t1 = 2 / (math.pi * a) + ln1 / 2
    x = math.copysign(math.sqrt(max(math.sqrt(t1 * t1 - ln1 / a) - t1, 0.0)), y)
    for _ in range(max_iter):
        err = math.erf(x) - y
        if abs(err) < tol:
            break
        x -= err / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
    return x


def regularized_gamma_p(a: float, x: float, tol: float = 1e-15, max_iter: int = 300):
    """Regularized lower incomplete gamma ``P(a, x)``.

    Uses the series for ``x < a+1`` and the continued fraction beyond, which is
    where each converges quickly.
    """
    if x < 0 or a <= 0:
        raise DomainError("require a > 0 and x >= 0")
    if x == 0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for n in range(1, max_iter):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * tol:
                break
        return total * math.exp(-x + a * math.log(x) - log_gamma(a))
    return 1.0 - regularized_gamma_q(a, x, tol, max_iter)


def regularized_gamma_q(a: float, x: float, tol: float = 1e-15, max_iter: int = 300):
    """Regularized upper incomplete gamma ``Q(a, x) = 1 - P(a, x)``."""
    if x < a + 1.0:
        return 1.0 - regularized_gamma_p(a, x, tol, max_iter)
    # Lentz's algorithm for the continued fraction
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h * math.exp(-x + a * math.log(x) - log_gamma(a))


def incomplete_gamma_lower(a: float, x: float):
    """Unregularized lower incomplete gamma ``gamma(a, x)``."""
    return regularized_gamma_p(a, x) * gamma(a)


def incomplete_gamma_upper(a: float, x: float):
    """Unregularized upper incomplete gamma ``Gamma(a, x)``."""
    return regularized_gamma_q(a, x) * gamma(a)


def incomplete_beta(a: float, b: float, x: float, tol: float = 1e-15,
                    max_iter: int = 300):
    """Regularized incomplete beta ``I_x(a, b)`` by continued fraction."""
    if x < 0.0 or x > 1.0:
        raise DomainError("require 0 <= x <= 1")
    if x == 0.0 or x == 1.0:
        return float(x)
    lbeta = log_beta(a, b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x, tol, max_iter) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x, tol, max_iter) / b


def _betacf(a, b, x, tol, max_iter):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h


# --------------------------------------------------------------------------
# Bessel functions (Abramowitz & Stegun polynomial approximations)
# --------------------------------------------------------------------------
# Below this the ascending series wins; above it the Hankel expansion does.
_J_SERIES_LIMIT = 18.0


def _j_series(nu: int, x: float, tol: float = 1e-17, max_terms: int = 60):
    """Ascending series for ``J_nu(x)``; near machine precision for |x| < 8."""
    half = 0.5 * x
    term = half**nu / math.factorial(nu)
    total = term
    q = -half * half
    for k in range(1, max_terms):
        term *= q / (k * (k + nu))
        total += term
        if abs(term) < tol * max(abs(total), 1e-30):
            break
    return total


def _hankel_pq(nu: int, x: float):
    """Leading Hankel asymptotic factors ``(P, Q)`` for large ``x``."""
    mu = 4.0 * nu * nu
    z = 8.0 * x
    # P ~ 1 - (mu-1)(mu-9)/(2!z^2) + (mu-1)(mu-9)(mu-25)(mu-49)/(4!z^4) - ...
    p1 = (mu - 1) * (mu - 9)
    p2 = p1 * (mu - 25) * (mu - 49)
    p3 = p2 * (mu - 81) * (mu - 121)
    P = 1.0 - p1 / (2 * z**2) + p2 / (24 * z**4) - p3 / (720 * z**6)
    q1 = mu - 1
    q2 = q1 * (mu - 9) * (mu - 25)
    q3 = q2 * (mu - 49) * (mu - 81)
    Q = q1 / z - q2 / (6 * z**3) + q3 / (120 * z**5)
    return P, Q


def bessel_j0(x):
    """Bessel function of the first kind, order 0.

    Machine precision for ``|x| < 10``, better than ``1e-10`` beyond, where the
    Hankel asymptotic expansion takes over.
    """
    x = float(x)
    ax = abs(x)
    if ax < _J_SERIES_LIMIT:
        return _j_series(0, ax)
    P, Q = _hankel_pq(0, ax)
    omega = ax - 0.25 * math.pi
    return math.sqrt(2.0 / (math.pi * ax)) * (P * math.cos(omega) - Q * math.sin(omega))


def bessel_j1(x):
    """Bessel function of the first kind, order 1."""
    x = float(x)
    ax = abs(x)
    if ax < _J_SERIES_LIMIT:
        ans = _j_series(1, ax)
    else:
        P, Q = _hankel_pq(1, ax)
        omega = ax - 0.75 * math.pi
        ans = math.sqrt(2.0 / (math.pi * ax)) * (P * math.cos(omega) - Q * math.sin(omega))
    return ans if x >= 0 else -ans


def bessel_y0(x):
    x = float(x)
    if x < 8.0:
        y = x * x
        p = (-2957821389.0 + y * (7062834065.0 + y * (-512359803.6
             + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733)))))
        q = (40076544269.0 + y * (745249964.8 + y * (7189466.438
             + y * (47447.26470 + y * (226.1030244 + y)))))
        return p / q + 0.636619772 * bessel_j0(x) * math.log(x)
    z = 8.0 / x
    y = z * z
    xx = x - 0.785398164
    p = (1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
         + y * (-0.2073370639e-5 + y * 0.2093887211e-6))))
    q = (-0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5
         + y * (0.7621095161e-6 + y * (-0.934945152e-7)))))
    return math.sqrt(0.636619772 / x) * (math.sin(xx) * p + z * math.cos(xx) * q)


def bessel_y1(x):
    x = float(x)
    if x < 8.0:
        y = x * x
        p = x * (-4900604943000.0 + y * (1275274390000.0 + y * (-51534381390.0
            + y * (734926455.1 + y * (-4237922.726 + y * 8511.937935)))))
        q = (24995805700000.0 + y * (424441966400.0 + y * (3733650367.0
             + y * (22459040.02 + y * (102042.605 + y * (354.9632885 + y))))))
        return p / q + 0.636619772 * (bessel_j1(x) * math.log(x) - 1.0 / x)
    z = 8.0 / x
    y = z * z
    xx = x - 2.356194491
    p = (1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
         + y * (0.2457520174e-5 + y * (-0.240337019e-6)))))
    q = (0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5
         + y * (-0.88228987e-6 + y * 0.105787412e-6))))
    return math.sqrt(0.636619772 / x) * (math.sin(xx) * p + z * math.cos(xx) * q)


def bessel_jn(n: int, x):
    """Bessel ``J_n`` by upward or downward recurrence as stability requires."""
    n = int(n)
    x = float(x)
    if n == 0:
        return bessel_j0(x)
    if n == 1:
        return bessel_j1(x)
    if n < 0:
        return (-1) ** n * bessel_jn(-n, x)
    if x == 0.0:
        return 0.0
    ax = abs(x)
    if ax > n:  # upward recurrence is stable here
        tox = 2.0 / ax
        bjm, bj = bessel_j0(ax), bessel_j1(ax)
        for j in range(1, n):
            bjm, bj = bj, j * tox * bj - bjm
        ans = bj
    else:  # Miller's downward recurrence
        tox = 2.0 / ax
        # Start the downward recurrence well above n; the constant controls
        # accuracy (Numerical Recipes uses 40, which leaves ~1e-7 here).
        m = 2 * ((n + int(math.sqrt(320.0 * n))) // 2)
        ans = 0.0
        jsum = 0
        total = 0.0
        bjp = 0.0
        bj = 1.0
        for j in range(m, 0, -1):
            bjm = j * tox * bj - bjp
            bjp = bj
            bj = bjm
            if abs(bj) > 1e10:
                bj *= 1e-10
                bjp *= 1e-10
                ans *= 1e-10
                total *= 1e-10
            if jsum:
                total += bj
            jsum = not jsum
            if j == n:
                ans = bjp
        total = 2.0 * total - bj
        ans /= total
    return ans if (x >= 0 or n % 2 == 0) else -ans


def bessel_yn(n: int, x):
    """Bessel ``Y_n`` by upward recurrence (stable for ``Y``)."""
    n = int(n)
    x = float(x)
    if n == 0:
        return bessel_y0(x)
    if n == 1:
        return bessel_y1(x)
    tox = 2.0 / x
    by, bym = bessel_y1(x), bessel_y0(x)
    for j in range(1, n):
        bym, by = by, j * tox * by - bym
    return by


def _bessel_i_series(nu: int, x: float) -> float:
    """Ascending series for ``I_nu(x)``.  Every term is positive, so there is
    no cancellation and the result is accurate to full precision -- only the
    term count grows with ``x``."""
    t = (x / 2.0) ** nu / math.factorial(nu)
    total = t
    for k in range(1, 600):
        t *= (x * x / 4.0) / (k * (k + nu))
        total += t
        if t < 1e-18 * total:
            break
    return total


def _bessel_i_asymptotic(nu: int, x: float) -> float:
    """Hankel asymptotic expansion for ``I_nu(x)``, used for large ``x``.

    The smallest term of the (divergent) series is of order ``e^{-2x}``
    relative to the sum, so beyond ``x = 15`` this reaches full double
    precision -- exactly where the ascending series starts needing many terms.
    """
    mu = 4.0 * nu * nu
    term = 1.0
    total = 1.0
    prev = math.inf
    for k in range(1, 40):
        term *= -(mu - (2 * k - 1) ** 2) / (k * 8.0 * x)
        if abs(term) > prev:      # the series has started to diverge
            break
        prev = abs(term)
        total += term
    return total * math.exp(x) / math.sqrt(2.0 * math.pi * x)


_I_ASYMPTOTIC_LIMIT = 15.0


def bessel_i0(x):
    """Modified Bessel function of the first kind, order 0."""
    ax = abs(float(x))
    if ax == 0.0:
        return 1.0
    if ax < _I_ASYMPTOTIC_LIMIT:
        return _bessel_i_series(0, ax)
    return _bessel_i_asymptotic(0, ax)


def bessel_i1(x):
    """Modified Bessel function of the first kind, order 1 (odd in ``x``)."""
    x = float(x)
    ax = abs(x)
    if ax == 0.0:
        return 0.0
    val = (_bessel_i_series(1, ax) if ax < _I_ASYMPTOTIC_LIMIT
           else _bessel_i_asymptotic(1, ax))
    return val if x >= 0.0 else -val


def bessel_in(n: int, x):
    """Modified Bessel ``I_n`` for integer order by Miller's recurrence.

    Forward recurrence on ``I`` is violently unstable -- it amplifies the
    ``K`` solution, which grows in the direction of increasing order.  Running
    the recurrence *backwards* from a high starting order damps that
    contaminant instead, and the result is fixed by normalizing against
    ``I_0``.
    """
    n = int(n)
    if n < 0:
        return bessel_in(-n, x)          # I_{-n} = I_n for integer n
    x = float(x)
    if n == 0:
        return bessel_i0(x)
    if n == 1:
        return bessel_i1(x)
    if x == 0.0:
        return 0.0
    ax = abs(x)
    # Start high enough that the seeded values are forgotten by the time the
    # recurrence reaches order n.  The margin has to grow with x as well as
    # with n: for x >> n the recurrence damps more slowly, and using sqrt(n)
    # alone loses three or four digits there.
    m = 2 * ((n + int(math.sqrt(80.0 * max(n, ax)))) // 2)
    prev, cur, ans = 0.0, 1.0, 0.0
    scale = 1.0
    for j in range(m, 0, -1):
        prev, cur = cur, prev + 2.0 * j / ax * cur
        if abs(cur) > 1e100:             # renormalize to avoid overflow
            cur *= 1e-100
            prev *= 1e-100
            ans *= 1e-100
            scale *= 1e-100
        if j == n:
            ans = prev
    ans *= bessel_i0(ax) / cur
    if x < 0.0 and n % 2:
        ans = -ans
    return ans


def _bessel_k_de(nu: int, x: float) -> float:
    """``K_nu(x)`` by the trapezoid rule on ``1/2 int e^{-x cosh t} cosh(nu t) dt``.

    The integrand decays doubly exponentially in ``t``, and for such functions
    the plain trapezoid rule on the whole real line converges geometrically in
    ``1/h`` -- so a few dozen terms give full precision.  The step is tightened
    as ``x`` grows because the peak at ``t = 0`` narrows like ``1/sqrt(x)``.
    """
    h = min(0.3, 4.0 * math.pi / (x + 45.0))
    span = math.acosh(760.0 / x) if x < 760.0 else 0.02
    n = int(math.ceil(span / h)) + 3
    t = np.arange(-n, n + 1) * h
    with np.errstate(over="ignore", under="ignore"):
        v = np.exp(-x * np.cosh(t)) * np.cosh(nu * t)
    return float(0.5 * h * np.sum(np.where(np.isfinite(v), v, 0.0)))


def bessel_k0(x):
    """Modified Bessel function of the second kind, order 0."""
    x = float(x)
    if x <= 0.0:
        raise DomainError("bessel_k0 requires x > 0")
    return _bessel_k_de(0, x)


def bessel_k1(x):
    """Modified Bessel function of the second kind, order 1."""
    x = float(x)
    if x <= 0.0:
        raise DomainError("bessel_k1 requires x > 0")
    return _bessel_k_de(1, x)


def bessel_kn(n: int, x):
    """Modified Bessel ``K_n`` for integer order.

    Unlike ``I``, forward recurrence is the *stable* direction for ``K``: it
    grows with order, so rounding error stays relatively small.
    """
    n = abs(int(n))
    x = float(x)
    if x <= 0.0:
        raise DomainError("bessel_kn requires x > 0")
    if n == 0:
        return bessel_k0(x)
    if n == 1:
        return bessel_k1(x)
    km1, k = bessel_k0(x), bessel_k1(x)
    for j in range(1, n):
        km1, k = k, km1 + 2.0 * j / x * k
    return k


def airy_ai(x, terms: int = 60):
    """Airy function ``Ai(x)`` from its Maclaurin series (moderate ``|x|``)."""
    x = float(x)
    c1 = 0.355028053887817239
    c2 = 0.258819403792806798
    f = g = 0.0
    term_f, term_g = 1.0, x
    for k in range(terms):
        f += term_f
        g += term_g
        term_f *= x**3 / ((3 * k + 2) * (3 * k + 3))
        term_g *= x**3 / ((3 * k + 3) * (3 * k + 4))
    return c1 * f - c2 * g


def airy_bi(x, terms: int = 60):
    """Airy function ``Bi(x)`` from its Maclaurin series (moderate ``|x|``)."""
    x = float(x)
    c1 = 0.355028053887817239
    c2 = 0.258819403792806798
    f = g = 0.0
    term_f, term_g = 1.0, x
    for k in range(terms):
        f += term_f
        g += term_g
        term_f *= x**3 / ((3 * k + 2) * (3 * k + 3))
        term_g *= x**3 / ((3 * k + 3) * (3 * k + 4))
    return math.sqrt(3.0) * (c1 * f + c2 * g)


def elliptic_k(m: float, tol: float = 1e-15):
    """Complete elliptic integral of the first kind by the AGM."""
    if m >= 1.0:
        raise DomainError("elliptic_k requires m < 1")
    a, b = 1.0, math.sqrt(1.0 - m)
    while abs(a - b) > tol:
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)


def elliptic_e(m: float, tol: float = 1e-15):
    """Complete elliptic integral of the second kind by the AGM."""
    if m > 1.0:
        raise DomainError("elliptic_e requires m <= 1")
    if m == 1.0:
        return 1.0
    a, b = 1.0, math.sqrt(1.0 - m)
    c = math.sqrt(max(m, 0.0))
    # E = K * (1 - sum_n 2^(n-1) c_n^2); the halving is applied at the end.
    total = c * c
    p = 1.0
    while abs(c) > tol:
        a, b, c = 0.5 * (a + b), math.sqrt(a * b), 0.5 * (a - b)
        p *= 2.0
        total += p * c * c
    return elliptic_k(m) * (1.0 - total / 2.0)


def exponential_integral(x: float, max_iter: int = 200, tol: float = 1e-15):
    """Exponential integral ``Ei(x)`` for ``x != 0``."""
    x = float(x)
    if x == 0.0:
        raise DomainError("Ei is singular at 0")
    euler = 0.5772156649015329
    if x < 0:
        # E1(-x) relation: Ei(x) = -E1(-x)
        return -_e1(-x, max_iter, tol)
    if x < 40:
        total = euler + math.log(abs(x))
        term = 1.0
        for k in range(1, max_iter):
            term *= x / k
            total += term / k
            if abs(term / k) < tol * abs(total):
                break
        return total
    total = 1.0
    term = 1.0
    for k in range(1, 30):
        term *= k / x
        total += term
    return math.exp(x) / x * total


def _e1(x, max_iter=200, tol=1e-15):
    """Exponential integral ``E_1(x)`` for ``x > 0``."""
    euler = 0.5772156649015329
    if x <= 1.0:
        total = -euler - math.log(x)
        term = 1.0
        for k in range(1, max_iter):
            term *= -x / k
            total -= term / k
            if abs(term / k) < tol:
                break
        return total
    # Lentz continued fraction
    tiny = 1e-300
    b = x + 1.0
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iter):
        a = -i * i
        b += 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        delta = c * d
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h * math.exp(-x)


def sine_integral(x: float, terms: int = 200, tol: float = 1e-16):
    """Sine integral ``Si(x)``."""
    x = float(x)
    total = 0.0
    term = x
    k = 0
    while abs(term) > tol * max(abs(total), 1e-30) and k < terms:
        total += term / (2 * k + 1)
        k += 1
        term *= -x * x / ((2 * k) * (2 * k + 1))
    return total


def cosine_integral(x: float, terms: int = 200, tol: float = 1e-16):
    """Cosine integral ``Ci(x)`` for ``x > 0``."""
    x = float(x)
    if x <= 0:
        raise DomainError("Ci requires x > 0")
    euler = 0.5772156649015329
    total = euler + math.log(x)
    term = 1.0
    k = 1
    while k < terms:
        term_k = (-1) ** k * x ** (2 * k) / (2 * k * math.factorial(2 * k))
        total += term_k
        if abs(term_k) < tol:
            break
        k += 1
    return total


def _zeta_borwein(s: float, n: int = 40) -> float:
    """Borwein's acceleration of the alternating zeta (eta) series.

    ``eta(s) = sum (-1)^{k-1} k^{-s} = (1 - 2^{1-s}) zeta(s)`` converges for
    every ``s > 0`` but far too slowly to use directly; weighting the partial
    sums by the Chebyshev-derived coefficients ``d_k`` below turns it into
    roughly ``n`` correct digits from ``n`` terms.
    """
    # d_k = n * sum_{i=0..k} (n+i-1)! 4^i / ((n-i)! (2i)!), built from the
    # term ratio t_i/t_{i-1} = 4 (n+i-1)(n-i+1) / (2i(2i-1)) so no factorial
    # is ever formed explicitly.
    d = np.zeros(n + 1)
    term = 1.0
    running = 1.0
    d[0] = float(n)
    for i in range(1, n + 1):
        term *= 4.0 * (n + i - 1) * (n - i + 1) / ((2 * i) * (2 * i - 1))
        running += term
        d[i] = n * running
    total = 0.0
    for k in range(n):
        total += ((-1.0) ** k) * (d[k] - d[n]) / (k + 1.0) ** s
    return -total / (d[n] * (1.0 - 2.0 ** (1.0 - s)))


def zeta(s: float, terms: int = 100000, tol: float = 1e-15):
    """Riemann zeta function, for any real ``s != 1``.

    ``s > 1`` uses Euler-Maclaurin acceleration directly.  ``s < 1`` is
    reflected through the functional equation
    ``zeta(s) = 2^s pi^{s-1} sin(pi s / 2) Gamma(1-s) zeta(1-s)``,
    which is how values like ``zeta(-1) = -1/12`` arise.  ``s = 1`` is the
    pole, and the negative even integers are the trivial zeros.
    """
    s = float(s)
    if s == 1.0:
        raise DomainError("zeta has a simple pole at s = 1")
    if 0.0 < s < 1.0:
        # The functional equation maps s to 1-s, which stays inside (0, 1) --
        # and is a fixed point at s = 1/2 -- so reflection cannot reach the
        # critical strip.  Borwein's alternating-series algorithm can.
        return _zeta_borwein(s)
    if s <= 0.0:
        if s == 0.0:
            return -0.5
        if s == int(s) and int(s) % 2 == 0:
            return 0.0                    # trivial zeros at -2, -4, -6, ...
        # Now 1 - s > 1, so the recursive call takes the Euler-Maclaurin path.
        return (2.0**s * math.pi ** (s - 1.0) * math.sin(0.5 * math.pi * s)
                * gamma(1.0 - s) * zeta(1.0 - s))
    n = 20
    total = sum(k ** (-s) for k in range(1, n))
    total += n ** (1 - s) / (s - 1) + 0.5 * n**-s
    # Euler-Maclaurin correction terms with Bernoulli numbers
    bern = [1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730, 7.0 / 6]
    term = s * n ** (-s - 1)
    fact = s
    for j, B in enumerate(bern, start=1):
        total += B / math.factorial(2 * j) * fact * n ** (-s - 2 * j + 1)
        fact *= (s + 2 * j - 1) * (s + 2 * j)
    return total


# --------------------------------------------------------------------------
# Additional special functions
# --------------------------------------------------------------------------
def lambert_w(x, branch: int = 0, tol: float = 1e-14, max_iter: int = 100):
    """Lambert W: the solution of ``W e^W = x``.

    ``branch=0`` is the principal branch (``W >= -1``), defined for
    ``x >= -1/e``; ``branch=-1`` is the lower real branch (``W <= -1``),
    defined on ``[-1/e, 0)``.  Solved by Halley's method, which converges
    cubically and handles the square-root behaviour near the branch point
    ``x = -1/e`` where Newton's method crawls.
    """
    x = float(x)
    inv_e = -1.0 / math.e
    if branch not in (0, -1):
        raise DomainError("only the real branches 0 and -1 are implemented")
    if x < inv_e:
        raise DomainError(f"lambert_w is undefined below -1/e for real x (got {x})")
    if x == inv_e:
        return -1.0
    if branch == 0:
        if x == 0.0:
            return 0.0
        w = math.log1p(x) if x < 3.0 else math.log(x) - math.log(math.log(x))
        if x < inv_e + 1e-3:                       # series about the branch point
            p = math.sqrt(2.0 * (math.e * x + 1.0))
            w = -1.0 + p - p * p / 3.0
    else:
        if x >= 0.0:
            raise DomainError("branch -1 requires -1/e <= x < 0")
        w = math.log(-x) - math.log(-math.log(-x)) if x > -1e-6 else -2.0
        if x < inv_e + 1e-3:
            p = -math.sqrt(2.0 * (math.e * x + 1.0))
            w = -1.0 + p - p * p / 3.0
    for _ in range(max_iter):
        ew = math.exp(w)
        f = w * ew - x
        if f == 0.0:
            return w
        # Halley: divides out the second-order term that slows plain Newton
        # to a crawl near the branch point.
        denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0)
        if denom == 0.0:
            break
        step = f / denom
        w -= step
        if abs(step) < tol * max(1.0, abs(w)):
            return w
    return w


def dawson(x, h: float = 0.2, terms: int = 60):
    """Dawson's function ``F(x) = e^{-x^2} int_0^x e^{t^2} dt``.

    Rybicki's method: the trapezoid rule applied to the integral's Fourier
    representation collapses to the sum below, whose error falls off like
    ``exp(-(pi/2h)^2)`` -- full double precision at ``h = 0.2`` with no
    cancellation anywhere.  The naive alternatives both fail: computing
    ``e^{-x^2}`` and the integral separately loses everything to overflow at
    large ``x``, while the Maclaurin series has terms of size ``e^{x^2}``
    that cancel down to an answer of order ``1/2x``.
    """
    x = float(x)
    ax = abs(x)
    if ax < 1e-8:                       # F(x) = x - 2x^3/3 + ...
        return x * (1.0 - 2.0 * x * x / 3.0)
    sign = 1.0 if x >= 0 else -1.0
    # Write x = n0*h + xp with n0 even, so the odd lattice point n becomes
    # n0 + k with k odd.  Re-indexing this way keeps the Gaussian centred near
    # xp (bounded by h, so it never underflows) while the denominator stays
    # n0 + k -- shifting the exponent without shifting the denominator would
    # evaluate a different function entirely.
    n0 = 2 * int(0.5 * ax / h + 0.5)
    xp = ax - n0 * h
    e1 = math.exp(2.0 * xp * h)
    e2 = e1 * e1
    e = e1
    total = 0.0
    for i in range(terms):
        k = 2 * i + 1
        gauss = math.exp(-((k * h) ** 2))
        if gauss == 0.0:
            break
        d1, d2 = n0 + k, n0 - k
        total += gauss * (e / d1 + 1.0 / (d2 * e) if d2 != 0 else gauss * e / d1)
        e *= e2
    return sign * math.exp(-xp * xp) * total / math.sqrt(math.pi)


def erfcx(x):
    """Scaled complementary error function ``e^{x^2} erfc(x)``.

    Stays finite and smooth for large positive ``x``, where ``erfc`` itself
    underflows to zero and ``e^{x^2}`` overflows -- the product is the only
    numerically usable form.
    """
    x = float(x)
    if x < 0.0:
        return 2.0 * math.exp(x * x) - erfcx(-x)
    if x < 5.0:
        return math.exp(x * x) * erfc(x)
    # erfcx(x) = (1/(x sqrt(pi))) * continued fraction, stable for large x.
    inv = 1.0 / (x * math.sqrt(math.pi))
    term = inv
    total = inv
    for n in range(1, 40):
        new = -term * (2 * n - 1) / (2.0 * x * x)
        if abs(new) > abs(term):
            break
        term = new
        total += term
    return total


def fresnel_s(x, tol: float = 1e-15):
    """Fresnel sine integral ``S(x) = int_0^x sin(pi t^2 / 2) dt``."""
    return _fresnel(x, tol)[0]


def fresnel_c(x, tol: float = 1e-15):
    """Fresnel cosine integral ``C(x) = int_0^x cos(pi t^2 / 2) dt``."""
    return _fresnel(x, tol)[1]


def _fresnel(x, tol: float = 1e-15):
    """Both Fresnel integrals as ``(S, C)``.

    The power series is exact for moderate ``x`` but its terms grow like
    ``(pi x^2/2)^n / n!`` before decaying, so it cancels away all precision
    once ``x`` is large.  Past ``x = 3`` the asymptotic form takes over,
    written through the auxiliary functions ``f`` and ``g``, which decay
    smoothly instead of oscillating.
    """
    x = float(x)
    ax = abs(x)
    sign = 1.0 if x >= 0 else -1.0
    if ax < 3.0:
        # S = sum (-1)^n (pi/2)^{2n+1} x^{4n+3} / ((2n+1)! (4n+3))
        # C = sum (-1)^n (pi/2)^{2n}   x^{4n+1} / ((2n)!   (4n+1))
        hp = 0.5 * math.pi
        x2, x4 = ax * ax, ax ** 4
        s = 0.0
        term = hp * ax ** 3 / 3.0
        n = 0
        while abs(term) > tol * max(abs(s), 1e-300) and n < 300:
            s += term
            n += 1
            term *= -(hp * hp) * x4 / ((2 * n) * (2 * n + 1))
            term *= (4 * n - 1) / (4 * n + 3)
        c = 0.0
        term = ax
        n = 0
        while abs(term) > tol * max(abs(c), 1e-300) and n < 300:
            c += term
            n += 1
            term *= -(hp * hp) * x4 / ((2 * n - 1) * (2 * n))
            term *= (4 * n - 3) / (4 * n + 1)
        return sign * s, sign * c
    if ax < 6.0:
        # Between the two expansions neither is good enough: the series has
        # already lost ~8 digits to cancellation and the asymptotic series
        # bottoms out at about the same size (both scale as exp(+-pi x^2/2),
        # so they cross while each is still around 1e-8).  Direct Gauss-
        # Legendre closes the gap -- the integrand is smooth with only a few
        # oscillations over this range, so a fixed high-order rule is exact.
        from ..integrate.gauss import gauss_legendre

        sv = float(gauss_legendre(lambda t: np.sin(0.5 * np.pi * t * t),
                                  0.0, ax, 120).value)
        cv = float(gauss_legendre(lambda t: np.cos(0.5 * np.pi * t * t),
                                  0.0, ax, 120).value)
        return sign * sv, sign * cv
    u = 0.5 * math.pi * ax * ax
    z = math.pi * ax * ax
    # f ~ 1/(pi x) sum (-1)^m (4m-1)!! / z^{2m},  g ~ 1/(pi^2 x^3) sum (-1)^m (4m+1)!! / z^{2m}
    f = _asymptotic_aux(z, -1) / (math.pi * ax)
    g = _asymptotic_aux(z, 1) / (math.pi * math.pi * ax ** 3)
    s = 0.5 - f * math.cos(u) - g * math.sin(u)
    c = 0.5 + f * math.sin(u) - g * math.cos(u)
    return sign * s, sign * c


def _asymptotic_aux(z: float, offset: int) -> float:
    """Sum ``(-1)^m (4m + offset)!! / z^{2m}``, truncated at its smallest term.

    The series is divergent, so it is cut where the terms stop shrinking --
    the point of optimal truncation, beyond which adding terms makes the
    answer worse.
    """
    term = 1.0
    total = 1.0
    prev = math.inf
    for m in range(1, 40):
        # (4m+offset)!!/(4m-4+offset)!! = (4m+offset)(4m-2+offset)
        term *= -(4.0 * m + offset) * (4.0 * m - 2.0 + offset) / (z * z)
        if abs(term) > prev:
            break
        prev = abs(term)
        total += term
    return total


def polygamma(n: int, x):
    """Polygamma ``psi^(n)(x)``, the ``n``-th derivative of ``log Gamma``.

    ``n = 0`` delegates to :func:`digamma`.  For ``n >= 1`` the recurrence
    ``psi^(n)(x) = psi^(n)(x+1) + (-1)^n n! / x^{n+1}`` pushes the argument
    into the range where the Hurwitz-zeta asymptotic series is accurate:
    ``psi^(n)(x) = (-1)^{n+1} n! zeta(n+1, x)``.
    """
    n = int(n)
    x = float(x)
    if n < 0:
        raise DomainError("polygamma requires n >= 0")
    if n == 0:
        return digamma(x)
    if x <= 0 and x == int(x):
        raise DomainError("polygamma has poles at the non-positive integers")
    shift = 0.0
    # psi^(n)(x+1) = psi^(n)(x) + (-1)^n n! x^{-(n+1)}, so stepping *up* in x
    # accumulates the term with the opposite sign.
    while x < 15.0:                     # recur up into the asymptotic regime
        shift -= ((-1.0) ** n) * math.factorial(n) / x ** (n + 1)
        x += 1.0
    # Hurwitz zeta by Euler-Maclaurin: zeta(s,x) = x^{1-s}/(s-1) + x^{-s}/2 + ...
    s = n + 1
    total = x ** (1.0 - s) / (s - 1.0) + 0.5 * x**-s
    bern = [1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730,
            7.0 / 6, -3617.0 / 510]
    fact = float(s)
    for j, B in enumerate(bern, start=1):
        total += B / math.factorial(2 * j) * fact * x ** (-s - 2 * j + 1)
        fact *= (s + 2 * j - 1) * (s + 2 * j)
    return ((-1.0) ** (n + 1)) * math.factorial(n) * total + shift


def trigamma(x):
    """Trigamma function ``psi'(x)``."""
    return polygamma(1, x)


def spherical_bessel_j(n: int, x):
    """Spherical Bessel ``j_n(x) = sqrt(pi/2x) J_{n+1/2}(x)``.

    Uses upward recurrence when ``n < x`` and downward (Miller) recurrence
    otherwise.  Upward recurrence is unstable in the regime ``n > x``, where
    ``j_n`` is decaying and any ``y_n`` contamination grows -- the same
    stability question as for the cylindrical functions.
    """
    n = int(n)
    x = float(x)
    if x == 0.0:
        return 1.0 if n == 0 else 0.0
    if n == 0:
        return math.sin(x) / x
    if n == 1:
        return math.sin(x) / (x * x) - math.cos(x) / x
    if n < x:                                       # upward is stable here
        jm1, j = math.sin(x) / x, math.sin(x) / (x * x) - math.cos(x) / x
        for k in range(1, n):
            jm1, j = j, (2 * k + 1) / x * j - jm1
        return j
    m = n + int(math.sqrt(80.0 * max(n, abs(x))))   # Miller start order
    jp1, jc = 0.0, 1.0
    ans = 0.0
    for k in range(m, 0, -1):
        # Entering the step, (jp1, jc) = (j_{k+1}, j_k); leaving it they are
        # (j_k, j_{k-1}).  So j_n appears in *jc* after the k = n+1 step.
        jp1, jc = jc, (2 * k + 1) / x * jc - jp1
        if abs(jc) > 1e100:
            jp1, jc, ans = jp1 * 1e-100, jc * 1e-100, ans * 1e-100
        if k == n + 1:
            ans = jc
    return ans * (math.sin(x) / x) / jc     # normalize against the true j_0


def spherical_bessel_y(n: int, x):
    """Spherical Bessel ``y_n(x) = sqrt(pi/2x) Y_{n+1/2}(x)``.

    Upward recurrence is stable for ``y`` at every order -- it is the growing
    solution, so relative error does not accumulate.
    """
    n = int(n)
    x = float(x)
    if x <= 0.0:
        raise DomainError("spherical_bessel_y requires x > 0")
    ym1 = -math.cos(x) / x
    if n == 0:
        return ym1
    y = -math.cos(x) / (x * x) - math.sin(x) / x
    for k in range(1, n):
        ym1, y = y, (2 * k + 1) / x * y - ym1
    return y


def associated_legendre(l: int, m: int, x):
    """Associated Legendre ``P_l^m(x)`` on ``[-1, 1]`` (Condon-Shortley phase).

    Built from the closed form for ``P_m^m`` and then recurred up in ``l``,
    which is the stable direction.
    """
    l, m = int(l), int(m)
    x = float(x)
    if m < 0 or m > l or abs(x) > 1.0:
        raise DomainError("associated_legendre requires 0 <= m <= l and |x| <= 1")
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max((1.0 - x) * (1.0 + x), 0.0))
        fact = 1.0
        for _ in range(m):
            pmm *= -fact * somx2      # the (-1)^m Condon-Shortley phase
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def spherical_harmonic(l: int, m: int, theta, phi):
    """Complex spherical harmonic ``Y_l^m(theta, phi)``.

    ``theta`` is the polar angle and ``phi`` the azimuth.  Negative ``m`` uses
    ``Y_l^{-m} = (-1)^m conj(Y_l^m)``.  Normalized so that the harmonics are
    orthonormal over the sphere.
    """
    l, m = int(l), int(m)
    if abs(m) > l:
        raise DomainError("spherical_harmonic requires |m| <= l")
    if m < 0:
        return ((-1.0) ** m) * np.conj(spherical_harmonic(l, -m, theta, phi))
    norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                     * math.factorial(l - m) / math.factorial(l + m))
    return norm * associated_legendre(l, m, math.cos(theta)) * np.exp(1j * m * phi)


def hyp1f1(a, b, z, tol: float = 1e-15, max_terms: int = 5000):
    """Confluent hypergeometric ``1F1(a; b; z)`` (Kummer's function).

    For ``z < 0`` the ascending series alternates, with terms as large as
    ``e^{|z|}`` summing to something of size ``e^{-|z|}`` -- so it loses
    ``2|z|`` nepers of precision and returns noise well before ``z = -20``.
    Kummer's transformation ``1F1(a;b;z) = e^z 1F1(b-a;b;-z)`` maps the
    argument to the positive axis, where (for ``b > a``) every term is
    positive and nothing cancels.  It is therefore applied for *all* negative
    ``z``, not just large ones.
    """
    a, b, z = float(a), float(b), float(z)
    if b <= 0 and b == int(b):
        raise DomainError("1F1 is undefined for non-positive integer b")
    if z < 0.0:
        return math.exp(z) * hyp1f1(b - a, b, -z, tol, max_terms)
    term = 1.0
    total = 1.0
    for n in range(max_terms):
        term *= (a + n) * z / ((b + n) * (n + 1))
        total += term
        if abs(term) < tol * max(abs(total), 1e-300):
            break
    return total


def hyp2f1(a, b, c, z, tol: float = 1e-15, max_terms: int = 20000):
    """Gauss hypergeometric ``2F1(a, b; c; z)`` for real ``|z| < 1``.

    The series converges only inside the unit disc.  For ``z`` in
    ``(-1, -0.5)`` the Pfaff transformation moves the argument into
    ``(0, 1/2)``, where convergence is fast; without it the series near
    ``z = -1`` needs impractically many terms.
    """
    a, b, c, z = float(a), float(b), float(c), float(z)
    if c <= 0 and c == int(c):
        raise DomainError("2F1 is undefined for non-positive integer c")
    if abs(z) >= 1.0:
        raise DomainError("this implementation requires |z| < 1")
    if z < -0.5:                        # Pfaff: 2F1(a,b;c;z) = (1-z)^-a 2F1(a,c-b;c;z/(z-1))
        return (1.0 - z) ** (-a) * hyp2f1(a, c - b, c, z / (z - 1.0), tol, max_terms)
    term = 1.0
    total = 1.0
    for n in range(max_terms):
        term *= (a + n) * (b + n) * z / ((c + n) * (n + 1))
        total += term
        if abs(term) < tol * max(abs(total), 1e-300):
            break
    return total


def expint_n(n: int, x, tol: float = 1e-14, max_iter: int = 200):
    """Generalized exponential integral ``E_n(x) = int_1^inf e^{-xt}/t^n dt``.

    A continued fraction for ``x > 1`` and a series for ``x <= 1``; the two
    regimes are exactly where each converges quickly.
    """
    n = int(n)
    x = float(x)
    if n < 0 or x < 0 or (x == 0 and n < 2):
        raise DomainError("expint_n requires n >= 0, x >= 0, and n >= 2 when x = 0")
    if n == 0:
        return math.exp(-x) / x
    if x == 0.0:
        return 1.0 / (n - 1)
    if x > 1.0:
        b = x + n
        c = 1e300
        d = 1.0 / b
        h = d
        for i in range(1, max_iter + 1):
            a = -i * (n - 1 + i)
            b += 2.0
            d = 1.0 / (a * d + b)
            c = b + a / c
            delta = c * d
            h *= delta
            if abs(delta - 1.0) < tol:
                break
        return h * math.exp(-x)
    total = (1.0 / (n - 1)) if n != 1 else (-math.log(x) - 0.5772156649015329)
    term = 1.0
    for i in range(1, max_iter + 1):
        term *= -x / i
        if i != n - 1:
            delta = -term / (i - (n - 1))
        else:
            psi = -0.5772156649015329
            for k in range(1, n):
                psi += 1.0 / k
            delta = term * (psi - math.log(x))
        total += delta
        if abs(delta) < tol * abs(total):
            break
    return total


def struve_h0(x, tol: float = 1e-14, max_terms: int = 300):
    """Struve function ``H_0(x)``: the particular solution of the driven Bessel equation."""
    x = float(x)
    term = 2.0 / math.pi
    total = term
    for k in range(1, max_terms):
        term *= -(x * x) / ((2 * k + 1) ** 2)
        total += term
        if abs(term) < tol * max(abs(total), 1e-300):
            break
    return total * x


def logistic(x):
    """Logistic sigmoid, written to avoid overflow for either sign of ``x``."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])                # e^x, never e^{-x}, when x is very negative
    out[~pos] = ex / (1.0 + ex)
    return out if out.ndim else float(out)


def logit(p):
    """Inverse of :func:`logistic`: ``log(p / (1 - p))``."""
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise DomainError("logit requires 0 < p < 1")
    return np.log(p) - np.log1p(-p)
