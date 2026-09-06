"""Special functions used throughout the library.

Implemented from scratch (Lanczos, continued fractions, series/asymptotic
switching) so the package depends on nothing beyond NumPy.

Every routine here is *array-native*: the series, recurrences and continued
fractions are advanced for the whole input at once, with a boolean mask
retiring elements as they converge, so an array of a million arguments costs
one pass of NumPy work per term rather than a million Python calls. Scalar
arguments still return Python floats, so the scalar spelling reads the same as
it always did.

Two conventions make the vectorized form safe:

* An iteration that stops "when the term is small" runs until *every* live
  element is small. Converged elements keep accumulating terms already below
  their own tolerance, which cannot move them.
* An asymptotic (divergent) series is truncated per element at its own
  smallest term by freezing that element's accumulator, never by a global
  break.
"""

from __future__ import annotations

import math

from .. import numeric as np

from .. import _accel

# Where a scalar kernel would still be the only option -- the Lanczos gamma,
# the error functions -- the compiled backend runs the identical approximation
# over the whole array at once. `_accel.kernel` returns None when no extension
# is loaded, and the NumPy expression underneath runs instead.

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

_EULER = 0.5772156649015329


# --------------------------------------------------------------------------
# Shape plumbing
# --------------------------------------------------------------------------
def _arr(x):
    """``(1-d float view, was-scalar, original shape)`` for one argument."""
    a = np.asarray(x, dtype=float)
    return np.atleast_1d(a), a.ndim == 0, a.shape


def _ret(out, scalar, shape):
    """Python scalar for scalar input, otherwise the caller's own shape."""
    return out.item() if scalar else np.reshape(out, shape)


def _bcast(*xs):
    """Broadcast several arguments to a common shape, as NumPy ufuncs do.

    The returned views are read-only; every kernel below allocates its own
    output rather than writing through them.
    """
    arrs = [np.asarray(v, dtype=float) for v in xs]
    scalar = all(a.ndim == 0 for a in arrs)
    shape = np.broadcast_shapes(*(a.shape for a in arrs))
    return ([np.atleast_1d(np.broadcast_to(a, shape)) for a in arrs],
            scalar, shape)


def _elementwise(name, x):
    """Run a compiled elementwise kernel, or return ``None`` if unavailable."""
    fast = _accel.kernel(name)
    if fast is None:
        return None
    xa = np.asarray(x, dtype=float)
    out = fast(np.ascontiguousarray(xa).ravel()).reshape(xa.shape)
    return float(out) if xa.ndim == 0 else out


# --------------------------------------------------------------------------
# Gamma and friends
# --------------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_COEF = np.array([
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
])


def _log_gamma_arr(x):
    """Lanczos log-gamma over an array, with reflection below ``x = 0.5``."""
    out = np.empty_like(x)
    low = x < 0.5
    if low.any():
        v = x[low]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[low] = (np.log(np.pi / np.abs(np.sin(np.pi * v)))
                        - _log_gamma_arr(1.0 - v))
    high = ~low
    if high.any():
        z = x[high] - 1.0
        # Horner over the eight Lanczos poles, one array temporary at a time
        # rather than the (n, 8) table the obvious expression would build.
        a = np.full(z.shape, _LANCZOS_COEF[0])
        for k in range(1, 9):
            a += _LANCZOS_COEF[k] / (z + k)
        t = z + _LANCZOS_G + 0.5
        out[high] = (0.5 * math.log(2 * math.pi) + (z + 0.5) * np.log(t)
                     - t + np.log(a))
    return out


def log_gamma(x):
    """Log-gamma by the Lanczos approximation (accurate to ~15 digits)."""
    fast = _elementwise("log_gamma", x)
    if fast is not None:
        return fast
    xa, scalar, shape = _arr(x)
    return _ret(_log_gamma_arr(xa), scalar, shape)


def _gamma_arr(x):
    out = np.empty_like(x)
    pole = (x <= 0.0) & (x == np.floor(x))
    out[pole] = np.inf
    low = ~pole & (x < 0.5)
    if low.any():
        v = x[low]
        out[low] = np.pi / (np.sin(np.pi * v) * _gamma_arr(1.0 - v))
    high = ~pole & ~low
    if high.any():
        # np.exp saturates to +inf past the double range; math.exp raises
        # there, which is a worse answer for one bad element of a big array.
        with np.errstate(over="ignore"):
            out[high] = np.exp(_log_gamma_arr(x[high]))
    return out


def gamma(x):
    """Gamma function for real arguments.

    Poles at the non-positive integers return ``+inf``, and arguments large
    enough to overflow the double range return ``+inf`` rather than raising.
    """
    fast = _elementwise("gamma", x)
    if fast is not None:
        return fast
    xa, scalar, shape = _arr(x)
    return _ret(_gamma_arr(xa), scalar, shape)


# 171! is the first factorial past the double range, so the exact table stops
# at 170 and everything above it overflows to +inf either way.
_FACTORIAL_TABLE = np.empty(171)
_f = 1.0
for _k in range(171):
    _FACTORIAL_TABLE[_k] = _f
    _f *= _k + 1
del _f, _k


def factorial(n):
    """Factorial via the gamma function (exact for small integers)."""
    na, scalar, shape = _arr(n)
    out = np.empty_like(na)
    exact = (na == np.floor(na)) & (na >= 0.0) & (na <= 170.0)
    if exact.any():
        out[exact] = _FACTORIAL_TABLE[na[exact].astype(np.intp)]
    rest = ~exact
    if rest.any():
        out[rest] = _gamma_arr(na[rest] + 1.0)
    return _ret(out, scalar, shape)


def binomial(n, k):
    """Binomial coefficient, computed in log space for large arguments."""
    (na, ka), scalar, shape = _bcast(n, k)
    with np.errstate(invalid="ignore", over="ignore"):
        out = np.exp(_log_gamma_arr(na + 1.0) - _log_gamma_arr(ka + 1.0)
                     - _log_gamma_arr(na - ka + 1.0))
    integral = ((na == np.floor(na)) & (ka == np.floor(ka))
                & (ka >= 0.0) & (ka <= na) & (na < 1000.0))
    # An integer-valued result is rounded back to that integer while the
    # log-space error is far below half a unit; the rest keep the exact
    # integer arithmetic the scalar routine used.
    snap = integral & (out < 1e12)
    out[snap] = np.round(out[snap])
    wide = integral & ~snap
    if wide.any():
        idx = np.flatnonzero(wide)
        out[idx] = [float(math.comb(int(na[i]), int(ka[i]))) for i in idx]
    return _ret(out, scalar, shape)


def beta(a, b):
    """Euler beta function ``B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b)``."""
    (aa, ba), scalar, shape = _bcast(a, b)
    with np.errstate(over="ignore"):
        out = np.exp(_log_beta_arr(aa, ba))
    return _ret(out, scalar, shape)


def _log_beta_arr(a, b):
    return _log_gamma_arr(a) + _log_gamma_arr(b) - _log_gamma_arr(a + b)


def log_beta(a, b):
    """Logarithm of the beta function."""
    (aa, ba), scalar, shape = _bcast(a, b)
    return _ret(_log_beta_arr(aa, ba), scalar, shape)


def _digamma_arr(x):
    """Digamma over an array; reflection below zero, recurrence above it."""
    out = np.empty_like(x)
    neg = x < 0.0
    if neg.any():
        # psi(x) = psi(1-x) - pi cot(pi x). Recurring up from a large negative
        # argument would instead need |x| additions to reach the asymptotic
        # range, so the reflection is what keeps the loop below bounded.
        v = x[neg]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[neg] = _digamma_arr(1.0 - v) - np.pi / np.tan(np.pi * v)
    pos = ~neg
    if pos.any():
        y = x[pos].astype(float, copy=True)
        acc = np.zeros_like(y)
        # At most twelve steps now that the argument is nonnegative.
        with np.errstate(divide="ignore"):
            while True:
                low = y < 12.0
                if not low.any():
                    break
                acc[low] -= 1.0 / y[low]
                y[low] += 1.0
        inv = 1.0 / y
        inv2 = inv * inv
        series = inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252
                 - inv2 * (1.0 / 240 - inv2 * (1.0 / 132 - inv2 * 691.0 / 32760)))))
        out[pos] = acc + np.log(y) - 0.5 * inv - series
    return out


def digamma(x):
    """Digamma ``psi(x) = d/dx log Gamma(x)`` by recurrence plus asymptotics."""
    xa, scalar, shape = _arr(x)
    return _ret(_digamma_arr(xa), scalar, shape)


def _erf_arr(x):
    """``erf`` over an array: the compiled kernel when present.

    NumPy has no ``erf`` ufunc and no rational approximation short of Cody's
    reaches the last bit, so the fallback is the correctly rounded scalar from
    the C library. Routines here that need ``erf`` internally -- ``erfinv``,
    ``erfcx`` -- go through this, so they are fully vectorized exactly when the
    backend is.
    """
    fast = _accel.kernel("erf")
    if fast is not None:
        return fast(np.ascontiguousarray(x).ravel()).reshape(x.shape)
    return np.array([math.erf(v) for v in x.ravel()],
                    dtype=float).reshape(x.shape)


def _erfc_arr(x):
    fast = _accel.kernel("erfc")
    if fast is not None:
        return fast(np.ascontiguousarray(x).ravel()).reshape(x.shape)
    return np.array([math.erfc(v) for v in x.ravel()],
                    dtype=float).reshape(x.shape)


def erf(x):
    """Error function, correctly rounded over the whole real line."""
    # Straight to the kernel when there is one: this is the hottest special
    # function in the package, and the reshape round trip the general path
    # takes is a measurable fraction of a large call.
    fast = _elementwise("erf", x)
    if fast is not None:
        return fast
    xa, scalar, shape = _arr(x)
    return _ret(_erf_arr(xa), scalar, shape)


def erfc(x):
    """Complementary error function ``1 - erf(x)``, accurate in the tail."""
    fast = _elementwise("erfc", x)
    if fast is not None:
        return fast
    xa, scalar, shape = _arr(x)
    return _ret(_erfc_arr(xa), scalar, shape)


def erfinv(y, tol: float = 1e-14, max_iter: int = 100):
    """Inverse error function by Newton iteration on a rational initial guess."""
    ya, scalar, shape = _arr(y)
    if np.any(np.abs(ya) > 1.0):
        raise DomainError("erfinv requires |y| < 1")
    out = np.empty_like(ya)
    edge = np.abs(ya) == 1.0
    out[edge] = np.copysign(np.inf, ya[edge])
    live = ~edge
    if live.any():
        v = ya[live]
        # Winitzki's initial approximation
        a = 0.147
        ln1 = np.log1p(-v * v)
        t1 = 2 / (math.pi * a) + ln1 / 2
        x = np.copysign(np.sqrt(np.maximum(np.sqrt(t1 * t1 - ln1 / a) - t1, 0.0)), v)
        active = np.ones(x.shape, dtype=bool)
        for _ in range(max_iter):
            err = _erf_arr(x) - v
            active &= np.abs(err) >= tol
            if not active.any():
                break
            step = err / (2.0 / math.sqrt(math.pi) * np.exp(-x * x))
            x = np.where(active, x - step, x)
        out[live] = x
    return _ret(out, scalar, shape)


# --------------------------------------------------------------------------
# Incomplete gamma and beta
# --------------------------------------------------------------------------
def _reg_gamma_p_arr(a, x, tol: float = 1e-15, max_iter: int = 300):
    """Series branch of ``P(a, x)``; the caller supplies only ``x < a + 1``."""
    term = 1.0 / a
    total = term.copy()
    live = np.ones(a.shape, dtype=bool)
    for n in range(1, max_iter):
        term = term * (x / (a + n))
        total = np.where(live, total + term, total)
        live &= np.abs(term) >= np.abs(total) * tol
        if not live.any():
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        return total * np.exp(-x + a * np.log(x) - _log_gamma_arr(a))


def _reg_gamma_q_arr(a, x, tol: float = 1e-15, max_iter: int = 300):
    """Lentz continued fraction for ``Q(a, x)``; needs ``x >= a + 1``."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = np.full(a.shape, 1.0 / tiny)
    d = 1.0 / b
    h = d.copy()
    live = np.ones(a.shape, dtype=bool)
    for i in range(1, max_iter):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = b + an / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = d * c
        h = np.where(live, h * delta, h)
        live &= np.abs(delta - 1.0) >= tol
        if not live.any():
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        return h * np.exp(-x + a * np.log(x) - _log_gamma_arr(a))


def _gamma_pq(a, x, tol, max_iter, want_p):
    """``P(a, x)`` or ``Q(a, x)``, each branch evaluated only where it holds."""
    out = np.empty(a.shape)
    zero = x == 0.0
    out[zero] = 0.0 if want_p else 1.0
    series = ~zero & (x < a + 1.0)
    if series.any():
        p = _reg_gamma_p_arr(a[series], x[series], tol, max_iter)
        out[series] = p if want_p else 1.0 - p
    frac = ~zero & ~series
    if frac.any():
        q = _reg_gamma_q_arr(a[frac], x[frac], tol, max_iter)
        out[frac] = 1.0 - q if want_p else q
    return out


def regularized_gamma_p(a, x, tol: float = 1e-15, max_iter: int = 300):
    """Regularized lower incomplete gamma ``P(a, x)``.

    Uses the series for ``x < a+1`` and the continued fraction beyond, which is
    where each converges quickly.
    """
    (aa, xa), scalar, shape = _bcast(a, x)
    if np.any(xa < 0) or np.any(aa <= 0):
        raise DomainError("require a > 0 and x >= 0")
    return _ret(_gamma_pq(aa, xa, tol, max_iter, True), scalar, shape)


def regularized_gamma_q(a, x, tol: float = 1e-15, max_iter: int = 300):
    """Regularized upper incomplete gamma ``Q(a, x) = 1 - P(a, x)``."""
    (aa, xa), scalar, shape = _bcast(a, x)
    if np.any(xa < 0) or np.any(aa <= 0):
        raise DomainError("require a > 0 and x >= 0")
    return _ret(_gamma_pq(aa, xa, tol, max_iter, False), scalar, shape)


def incomplete_gamma_lower(a, x):
    """Unregularized lower incomplete gamma ``gamma(a, x)``."""
    (aa, xa), scalar, shape = _bcast(a, x)
    if np.any(xa < 0) or np.any(aa <= 0):
        raise DomainError("require a > 0 and x >= 0")
    out = _gamma_pq(aa, xa, 1e-15, 300, True) * _gamma_arr(aa)
    return _ret(out, scalar, shape)


def incomplete_gamma_upper(a, x):
    """Unregularized upper incomplete gamma ``Gamma(a, x)``."""
    (aa, xa), scalar, shape = _bcast(a, x)
    if np.any(xa < 0) or np.any(aa <= 0):
        raise DomainError("require a > 0 and x >= 0")
    out = _gamma_pq(aa, xa, 1e-15, 300, False) * _gamma_arr(aa)
    return _ret(out, scalar, shape)


def incomplete_beta(a, b, x, tol: float = 1e-15, max_iter: int = 300):
    """Regularized incomplete beta ``I_x(a, b)`` by continued fraction."""
    (aa, ba, xa), scalar, shape = _bcast(a, b, x)
    if np.any(xa < 0.0) or np.any(xa > 1.0):
        raise DomainError("require 0 <= x <= 1")
    out = np.empty(aa.shape)
    edge = (xa == 0.0) | (xa == 1.0)
    out[edge] = xa[edge]
    live = ~edge
    if live.any():
        av, bv, xv = aa[live], ba[live], xa[live]
        lbeta = _log_beta_arr(av, bv)
        front = np.exp(av * np.log(xv) + bv * np.log1p(-xv) - lbeta)
        # The continued fraction converges quickly only on the side of the
        # distribution's mean; the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) moves
        # the other side across.
        near = xv < (av + 1.0) / (av + bv + 2.0)
        val = np.empty(av.shape)
        if near.any():
            val[near] = (front[near]
                         * _betacf(av[near], bv[near], xv[near], tol, max_iter)
                         / av[near])
        far = ~near
        if far.any():
            val[far] = 1.0 - (front[far]
                              * _betacf(bv[far], av[far], 1.0 - xv[far], tol, max_iter)
                              / bv[far])
        out[live] = val
    return _ret(out, scalar, shape)


def _betacf(a, b, x, tol, max_iter):
    """Lentz evaluation of the incomplete-beta continued fraction."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones(a.shape)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1.0 / d
    h = d.copy()
    live = np.ones(a.shape, dtype=bool)
    for m in range(1, max_iter):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        h = np.where(live, h * d * c, h)
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1.0 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = d * c
        h = np.where(live, h * delta, h)
        live &= np.abs(delta - 1.0) >= tol
        if not live.any():
            break
    return h


# --------------------------------------------------------------------------
# Bessel functions (Abramowitz & Stegun polynomial approximations)
# --------------------------------------------------------------------------
# Below this the ascending series wins; above it the Hankel expansion does.
_J_SERIES_LIMIT = 18.0


def _j_series(nu: int, x, tol: float = 1e-17, max_terms: int = 60):
    """Ascending series for ``J_nu(x)``; near machine precision for |x| < 8."""
    half = 0.5 * x
    term = half**nu / math.factorial(nu)
    total = np.array(term, dtype=float, copy=True)
    q = -half * half
    for k in range(1, max_terms):
        term = term * (q / (k * (k + nu)))
        total += term
        if np.all(np.abs(term) < tol * np.maximum(np.abs(total), 1e-30)):
            break
    return total


def _hankel_pq(nu: int, x):
    """Leading Hankel asymptotic factors ``(P, Q)`` for large ``x``."""
    mu = 4.0 * nu * nu
    z = 8.0 * x
    z2 = z * z
    # P ~ 1 - (mu-1)(mu-9)/(2!z^2) + (mu-1)(mu-9)(mu-25)(mu-49)/(4!z^4) - ...
    p1 = (mu - 1) * (mu - 9)
    p2 = p1 * (mu - 25) * (mu - 49)
    p3 = p2 * (mu - 81) * (mu - 121)
    P = 1.0 - p1 / (2 * z2) + p2 / (24 * z2 * z2) - p3 / (720 * z2 * z2 * z2)
    q1 = mu - 1
    q2 = q1 * (mu - 9) * (mu - 25)
    q3 = q2 * (mu - 49) * (mu - 81)
    Q = q1 / z - q2 / (6 * z * z2) + q3 / (120 * z * z2 * z2)
    return P, Q


def _j01_arr(nu: int, ax):
    """``J_0`` or ``J_1`` of a nonnegative array, by series then Hankel."""
    out = np.empty_like(ax)
    low = ax < _J_SERIES_LIMIT
    if low.any():
        out[low] = _j_series(nu, ax[low])
    high = ~low
    if high.any():
        v = ax[high]
        P, Q = _hankel_pq(nu, v)
        omega = v - (0.25 if nu == 0 else 0.75) * math.pi
        out[high] = np.sqrt(2.0 / (math.pi * v)) * (P * np.cos(omega)
                                                    - Q * np.sin(omega))
    return out


def bessel_j0(x):
    """Bessel function of the first kind, order 0.

    Machine precision for ``|x| < 10``, better than ``1e-10`` beyond, where the
    Hankel asymptotic expansion takes over.
    """
    xa, scalar, shape = _arr(x)
    return _ret(_j01_arr(0, np.abs(xa)), scalar, shape)


def bessel_j1(x):
    """Bessel function of the first kind, order 1."""
    xa, scalar, shape = _arr(x)
    out = _j01_arr(1, np.abs(xa))
    return _ret(np.where(xa >= 0, out, -out), scalar, shape)


def _y0_arr(x):
    out = np.empty_like(x)
    low = x < 8.0
    if low.any():
        v = x[low]
        y = v * v
        p = (-2957821389.0 + y * (7062834065.0 + y * (-512359803.6
             + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733)))))
        q = (40076544269.0 + y * (745249964.8 + y * (7189466.438
             + y * (47447.26470 + y * (226.1030244 + y)))))
        out[low] = p / q + 0.636619772 * _j01_arr(0, v) * np.log(v)
    high = ~low
    if high.any():
        v = x[high]
        z = 8.0 / v
        y = z * z
        xx = v - 0.785398164
        p = (1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
             + y * (-0.2073370639e-5 + y * 0.2093887211e-6))))
        q = (-0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5
             + y * (0.7621095161e-6 + y * (-0.934945152e-7)))))
        out[high] = np.sqrt(0.636619772 / v) * (np.sin(xx) * p + z * np.cos(xx) * q)
    return out


def _y1_arr(x):
    out = np.empty_like(x)
    low = x < 8.0
    if low.any():
        v = x[low]
        y = v * v
        p = v * (-4900604943000.0 + y * (1275274390000.0 + y * (-51534381390.0
                 + y * (734926455.1 + y * (-4237922.726 + y * 8511.937935)))))
        q = (24995805700000.0 + y * (424441966400.0 + y * (3733650367.0
             + y * (22459040.02 + y * (102042.605 + y * (354.9632885 + y))))))
        out[low] = p / q + 0.636619772 * (_j01_arr(1, v) * np.log(v) - 1.0 / v)
    high = ~low
    if high.any():
        v = x[high]
        z = 8.0 / v
        y = z * z
        xx = v - 2.356194491
        p = (1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
             + y * (0.2457520174e-5 + y * (-0.240337019e-6)))))
        q = (0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5
             + y * (-0.88228987e-6 + y * 0.105787412e-6))))
        out[high] = np.sqrt(0.636619772 / v) * (np.sin(xx) * p + z * np.cos(xx) * q)
    return out


def bessel_y0(x):
    """Bessel function of the second kind, order 0 (requires ``x > 0``)."""
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0.0):
        raise DomainError("bessel_y0 requires x > 0")
    return _ret(_y0_arr(xa), scalar, shape)


def bessel_y1(x):
    """Bessel function of the second kind, order 1 (requires ``x > 0``)."""
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0.0):
        raise DomainError("bessel_y1 requires x > 0")
    return _ret(_y1_arr(xa), scalar, shape)


def _jn_arr(n: int, x):
    """``J_n`` of an array: upward recurrence where stable, Miller's below."""
    ax = np.abs(x)
    out = np.zeros_like(ax)
    nz = ax != 0.0
    if nz.any():
        v = ax[nz]
        res = np.empty_like(v)
        up = v > n
        if up.any():
            w = v[up]
            tox = 2.0 / w
            bjm, bj = _j01_arr(0, w), _j01_arr(1, w)
            for j in range(1, n):
                bjm, bj = bj, j * tox * bj - bjm
            res[up] = bj
        down = ~up
        if down.any():
            w = v[down]
            tox = 2.0 / w
            # Start the downward recurrence well above n; the constant controls
            # accuracy (Numerical Recipes uses 40, which leaves ~1e-7 here).
            m = 2 * ((n + int(math.sqrt(320.0 * n))) // 2)
            ans = np.zeros_like(w)
            total = np.zeros_like(w)
            bjp = np.zeros_like(w)
            bj = np.ones_like(w)
            jsum = False
            for j in range(m, 0, -1):
                bjm = j * tox * bj - bjp
                bjp = bj
                bj = bjm
                big = np.abs(bj) > 1e10
                if big.any():
                    scale = np.where(big, 1e-10, 1.0)
                    bj = bj * scale
                    bjp = bjp * scale
                    ans = ans * scale
                    total = total * scale
                if jsum:
                    total += bj
                jsum = not jsum
                if j == n:
                    ans = bjp
            total = 2.0 * total - bj
            res[down] = ans / total
        out[nz] = res
    if n % 2:
        out = np.where(x >= 0, out, -out)
    return out


def bessel_jn(n: int, x):
    """Bessel ``J_n`` by upward or downward recurrence as stability requires."""
    n = int(n)
    xa, scalar, shape = _arr(x)
    if n == 0:
        return bessel_j0(x)
    if n == 1:
        return bessel_j1(x)
    if n < 0:
        return _ret((-1) ** n * _jn_arr(-n, xa), scalar, shape)
    return _ret(_jn_arr(n, xa), scalar, shape)


def bessel_yn(n: int, x):
    """Bessel ``Y_n`` by upward recurrence (stable for ``Y``)."""
    n = int(n)
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0.0):
        raise DomainError("bessel_yn requires x > 0")
    if n == 0:
        return _ret(_y0_arr(xa), scalar, shape)
    if n == 1:
        return _ret(_y1_arr(xa), scalar, shape)
    tox = 2.0 / xa
    by, bym = _y1_arr(xa), _y0_arr(xa)
    for j in range(1, n):
        bym, by = by, j * tox * by - bym
    return _ret(by, scalar, shape)


def _bessel_i_series(nu: int, x):
    """Ascending series for ``I_nu(x)``.  Every term is positive, so there is
    no cancellation and the result is accurate to full precision -- only the
    term count grows with ``x``."""
    t = (x / 2.0) ** nu / math.factorial(nu)
    total = np.array(t, dtype=float, copy=True)
    q = x * x / 4.0
    for k in range(1, 600):
        t = t * (q / (k * (k + nu)))
        total += t
        if np.all(t < 1e-18 * total):
            break
    return total


def _bessel_i_asymptotic(nu: int, x):
    """Hankel asymptotic expansion for ``I_nu(x)``, used for large ``x``.

    The smallest term of the (divergent) series is of order ``e^{-2x}``
    relative to the sum, so beyond ``x = 15`` this reaches full double
    precision -- exactly where the ascending series starts needing many terms.
    Each element is truncated at its own smallest term by freezing it, since
    the optimal truncation point moves with ``x``.
    """
    mu = 4.0 * nu * nu
    term = np.ones_like(x)
    total = np.ones_like(x)
    prev = np.full(x.shape, np.inf)
    live = np.ones(x.shape, dtype=bool)
    for k in range(1, 40):
        new = term * (-(mu - (2 * k - 1) ** 2) / (k * 8.0 * x))
        live &= np.abs(new) <= prev          # the series started to diverge
        if not live.any():
            break
        term = np.where(live, new, term)
        prev = np.where(live, np.abs(new), prev)
        total = np.where(live, total + term, total)
    return total * np.exp(x) / np.sqrt(2.0 * math.pi * x)


_I_ASYMPTOTIC_LIMIT = 15.0


def _i01_arr(nu: int, ax):
    out = np.empty_like(ax)
    zero = ax == 0.0
    out[zero] = 1.0 if nu == 0 else 0.0
    low = ~zero & (ax < _I_ASYMPTOTIC_LIMIT)
    if low.any():
        out[low] = _bessel_i_series(nu, ax[low])
    high = ~zero & ~low
    if high.any():
        out[high] = _bessel_i_asymptotic(nu, ax[high])
    return out


def bessel_i0(x):
    """Modified Bessel function of the first kind, order 0."""
    xa, scalar, shape = _arr(x)
    return _ret(_i01_arr(0, np.abs(xa)), scalar, shape)


def bessel_i1(x):
    """Modified Bessel function of the first kind, order 1 (odd in ``x``)."""
    xa, scalar, shape = _arr(x)
    out = _i01_arr(1, np.abs(xa))
    return _ret(np.where(xa >= 0.0, out, -out), scalar, shape)


def bessel_in(n: int, x):
    """Modified Bessel ``I_n`` for integer order by Miller's recurrence.

    Forward recurrence on ``I`` is violently unstable -- it amplifies the
    ``K`` solution, which grows in the direction of increasing order.  Running
    the recurrence *backwards* from a high starting order damps that
    contaminant instead, and the result is fixed by normalizing against
    ``I_0``.
    """
    n = abs(int(n))                      # I_{-n} = I_n for integer n
    xa, scalar, shape = _arr(x)
    if n == 0:
        return bessel_i0(x)
    if n == 1:
        return bessel_i1(x)
    ax = np.abs(xa)
    out = np.zeros_like(ax)
    nz = ax != 0.0
    if nz.any():
        v = ax[nz]
        # Start high enough that the seeded values are forgotten by the time
        # the recurrence reaches order n.  The margin has to grow with x as
        # well as with n: for x >> n the recurrence damps more slowly.  One
        # start order serves the whole array -- overshooting it only makes the
        # damping more complete.
        m = 2 * ((n + int(math.sqrt(80.0 * max(float(n), float(v.max()))))) // 2)
        prev = np.zeros_like(v)
        cur = np.ones_like(v)
        ans = np.zeros_like(v)
        tox = 2.0 / v
        for j in range(m, 0, -1):
            prev, cur = cur, prev + (j * tox) * cur
            big = np.abs(cur) > 1e100    # renormalize to avoid overflow
            if big.any():
                scale = np.where(big, 1e-100, 1.0)
                cur = cur * scale
                prev = prev * scale
                ans = ans * scale
            if j == n:
                ans = prev
        out[nz] = ans * (_i01_arr(0, v) / cur)
    if n % 2:
        out = np.where(xa < 0.0, -out, out)
    return _ret(out, scalar, shape)


def _bessel_k_de(nu: int, x):
    """``K_nu(x)`` by the trapezoid rule on ``1/2 int e^{-x cosh t} cosh(nu t) dt``.

    The integrand decays doubly exponentially in ``t``, and for such functions
    the plain trapezoid rule on the whole real line converges geometrically in
    ``1/h`` -- so a few dozen terms give full precision.  The step is tightened
    as ``x`` grows because the peak at ``t = 0`` narrows like ``1/sqrt(x)``.

    Both the step and the truncation point depend on ``x``, so the array is
    walked in blocks of *sorted* arguments: within a block the grid that serves
    the extreme element is nearly the grid every element wanted, and one
    ``(block, nodes)`` array evaluates them all. Sorting is what keeps that
    array small -- an unsorted block would need the finest step of its largest
    argument spread over the widest span of its smallest.
    """
    out = np.empty_like(x)
    order = np.argsort(x, kind="stable")
    block = 512
    for start in range(0, x.size, block):
        idx = order[start:start + block]
        xs = x[idx]
        lo, hi = float(xs[0]), float(xs[-1])
        h = min(0.3, 4.0 * math.pi / (hi + 45.0))
        span = math.acosh(760.0 / lo) if lo < 760.0 else 0.02
        n = int(math.ceil(span / h)) + 3
        t = np.arange(-n, n + 1) * h
        with np.errstate(over="ignore", under="ignore"):
            v = np.exp(-xs[:, None] * np.cosh(t)) * np.cosh(nu * t)
            out[idx] = 0.5 * h * np.sum(np.where(np.isfinite(v), v, 0.0), axis=1)
    return out


def bessel_k0(x):
    """Modified Bessel function of the second kind, order 0."""
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0.0):
        raise DomainError("bessel_k0 requires x > 0")
    return _ret(_bessel_k_de(0, xa), scalar, shape)


def bessel_k1(x):
    """Modified Bessel function of the second kind, order 1."""
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0.0):
        raise DomainError("bessel_k1 requires x > 0")
    return _ret(_bessel_k_de(1, xa), scalar, shape)


def bessel_kn(n: int, x):
    """Modified Bessel ``K_n`` for integer order.

    Unlike ``I``, forward recurrence is the *stable* direction for ``K``: it
    grows with order, so rounding error stays relatively small.
    """
    n = abs(int(n))
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0.0):
        raise DomainError("bessel_kn requires x > 0")
    if n == 0:
        return _ret(_bessel_k_de(0, xa), scalar, shape)
    if n == 1:
        return _ret(_bessel_k_de(1, xa), scalar, shape)
    km1, k = _bessel_k_de(0, xa), _bessel_k_de(1, xa)
    tox = 2.0 / xa
    for j in range(1, n):
        km1, k = k, km1 + (j * tox) * k
    return _ret(k, scalar, shape)


def _airy_series(x, terms: int):
    """The two Maclaurin sums ``f`` and ``g`` shared by ``Ai`` and ``Bi``."""
    f = np.zeros_like(x)
    g = np.zeros_like(x)
    term_f = np.ones_like(x)
    term_g = np.array(x, dtype=float, copy=True)
    x3 = x ** 3
    for k in range(terms):
        f += term_f
        g += term_g
        term_f = term_f * (x3 / ((3 * k + 2) * (3 * k + 3)))
        term_g = term_g * (x3 / ((3 * k + 3) * (3 * k + 4)))
    return f, g


_AIRY_C1 = 0.355028053887817239
_AIRY_C2 = 0.258819403792806798


def airy_ai(x, terms: int = 60):
    """Airy function ``Ai(x)`` from its Maclaurin series (moderate ``|x|``)."""
    xa, scalar, shape = _arr(x)
    f, g = _airy_series(xa, terms)
    return _ret(_AIRY_C1 * f - _AIRY_C2 * g, scalar, shape)


def airy_bi(x, terms: int = 60):
    """Airy function ``Bi(x)`` from its Maclaurin series (moderate ``|x|``)."""
    xa, scalar, shape = _arr(x)
    f, g = _airy_series(xa, terms)
    return _ret(math.sqrt(3.0) * (_AIRY_C1 * f + _AIRY_C2 * g), scalar, shape)


def _agm_k(m, tol):
    """AGM iteration for ``K(m)``; returns the converged arithmetic mean."""
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    for _ in range(64):
        if np.all(np.abs(a - b) <= tol):
            break
        a, b = 0.5 * (a + b), np.sqrt(a * b)
    return a


def elliptic_k(m, tol: float = 1e-15):
    """Complete elliptic integral of the first kind by the AGM."""
    ma, scalar, shape = _arr(m)
    if np.any(ma >= 1.0):
        raise DomainError("elliptic_k requires m < 1")
    return _ret(math.pi / (2.0 * _agm_k(ma, tol)), scalar, shape)


def elliptic_e(m, tol: float = 1e-15):
    """Complete elliptic integral of the second kind by the AGM."""
    ma, scalar, shape = _arr(m)
    if np.any(ma > 1.0):
        raise DomainError("elliptic_e requires m <= 1")
    out = np.empty_like(ma)
    one = ma == 1.0
    out[one] = 1.0
    live = ~one
    if live.any():
        v = ma[live]
        a = np.ones_like(v)
        b = np.sqrt(1.0 - v)
        c = np.sqrt(np.maximum(v, 0.0))
        # E = K * (1 - sum_n 2^(n-1) c_n^2); the halving is applied at the end.
        total = c * c
        p = np.ones_like(v)
        # The AGM converges after a different number of steps for each m, and
        # the weight p doubles every step, so an element that has converged
        # must be frozen rather than carried along: p * c^2 with a stale c and
        # a still-doubling p is not negligible.
        running = np.abs(c) > tol
        for _ in range(64):
            if not running.any():
                break
            an, bn, cn = 0.5 * (a + b), np.sqrt(a * b), 0.5 * (a - b)
            a = np.where(running, an, a)
            b = np.where(running, bn, b)
            c = np.where(running, cn, c)
            p = np.where(running, p * 2.0, p)
            total = np.where(running, total + p * c * c, total)
            running &= np.abs(c) > tol
        out[live] = (math.pi / (2.0 * _agm_k(v, tol))) * (1.0 - total / 2.0)
    return _ret(out, scalar, shape)


def _e1_arr(x, max_iter=200, tol=1e-15):
    """Exponential integral ``E_1(x)`` for ``x > 0``."""
    out = np.empty_like(x)
    small = x <= 1.0
    if small.any():
        v = x[small]
        total = -_EULER - np.log(v)
        term = np.ones_like(v)
        live = np.ones(v.shape, dtype=bool)
        for k in range(1, max_iter):
            term = term * (-v / k)
            total = np.where(live, total - term / k, total)
            live &= np.abs(term / k) >= tol
            if not live.any():
                break
        out[small] = total
    big = ~small
    if big.any():
        v = x[big]
        tiny = 1e-300
        b = v + 1.0
        c = np.full(v.shape, 1.0 / tiny)
        d = 1.0 / b
        h = d.copy()
        live = np.ones(v.shape, dtype=bool)
        for i in range(1, max_iter):
            a = -float(i * i)
            b = b + 2.0
            d = 1.0 / (a * d + b)
            c = b + a / c
            delta = c * d
            h = np.where(live, h * delta, h)
            live &= np.abs(delta - 1.0) >= tol
            if not live.any():
                break
        out[big] = h * np.exp(-v)
    return out


def exponential_integral(x, max_iter: int = 200, tol: float = 1e-15):
    """Exponential integral ``Ei(x)`` for ``x != 0``."""
    xa, scalar, shape = _arr(x)
    if np.any(xa == 0.0):
        raise DomainError("Ei is singular at 0")
    out = np.empty_like(xa)
    neg = xa < 0.0
    if neg.any():
        # E1(-x) relation: Ei(x) = -E1(-x)
        out[neg] = -_e1_arr(-xa[neg], max_iter, tol)
    ser = ~neg & (xa < 40.0)
    if ser.any():
        v = xa[ser]
        total = _EULER + np.log(np.abs(v))
        term = np.ones_like(v)
        live = np.ones(v.shape, dtype=bool)
        for k in range(1, max_iter):
            term = term * (v / k)
            total = np.where(live, total + term / k, total)
            live &= np.abs(term / k) >= tol * np.abs(total)
            if not live.any():
                break
        out[ser] = total
    asy = ~neg & ~ser
    if asy.any():
        v = xa[asy]
        total = np.ones_like(v)
        term = np.ones_like(v)
        for k in range(1, 30):
            term = term * (k / v)
            total += term
        out[asy] = np.exp(v) / v * total
    return _ret(out, scalar, shape)


# Above this the ascending series for Si and Ci has lost too much to
# cancellation -- its largest term is ~e^x/sqrt(x) while the answer is O(1) --
# and the continued fraction for E_1(ix) takes over. At x = 20 that is the
# difference between eight correct digits and fifteen.
_SICI_SERIES_LIMIT = 4.0


def _e1_imag_cf(x, max_iter: int = 400, tol: float = 1e-16):
    """``E_1(i x)`` for real ``x > 0`` by the same Lentz recurrence as ``E_1``.

    ``Si`` and ``Ci`` are its imaginary and real parts:
    ``E_1(ix) = -Ci(x) + i (Si(x) - pi/2)``. The recurrence is identical to the
    real one; only the arithmetic is complex.
    """
    tiny = 1e-300
    z = 1j * x
    b = z + 1.0
    c = np.full(x.shape, 1.0 / tiny, dtype=complex)
    d = 1.0 / b
    h = d.copy()
    live = np.ones(x.shape, dtype=bool)
    for i in range(1, max_iter):
        a = -float(i * i)
        b = b + 2.0
        d = 1.0 / (a * d + b)
        c = b + a / c
        delta = c * d
        h = np.where(live, h * delta, h)
        live &= np.abs(delta - 1.0) >= tol
        if not live.any():
            break
    return h * np.exp(-z)


def sine_integral(x, terms: int = 200, tol: float = 1e-16):
    """Sine integral ``Si(x)``.

    The ascending series is used while it is still free of cancellation, and
    the continued fraction for ``E_1(ix)`` beyond that.
    """
    xa, scalar, shape = _arr(x)
    ax = np.abs(xa)
    out = np.empty_like(ax)
    small = ax <= _SICI_SERIES_LIMIT
    if small.any():
        v = ax[small]
        total = np.zeros_like(v)
        term = v.copy()
        live = np.ones(v.shape, dtype=bool)
        for k in range(terms):
            live &= np.abs(term) > tol * np.maximum(np.abs(total), 1e-30)
            if not live.any():
                break
            total = np.where(live, total + term / (2 * k + 1), total)
            term = term * (-v * v / ((2 * (k + 1)) * (2 * (k + 1) + 1)))
        out[small] = total
    big = ~small
    if big.any():
        out[big] = 0.5 * math.pi + np.imag(_e1_imag_cf(ax[big]))
    return _ret(np.copysign(out, xa), scalar, shape)      # Si is odd


def cosine_integral(x, terms: int = 200, tol: float = 1e-16):
    """Cosine integral ``Ci(x)`` for ``x > 0``."""
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0):
        raise DomainError("Ci requires x > 0")
    out = np.empty_like(xa)
    small = xa <= _SICI_SERIES_LIMIT
    if small.any():
        v = xa[small]
        total = _EULER + np.log(v)
        # t_k = (-1)^k x^{2k} / (2k (2k)!): the power is advanced by its own
        # recurrence and the exact rational coefficient is folded in as a
        # float, which underflows to zero long before the loop runs out.
        x2k = np.ones_like(v)
        x2 = v * v
        fact = 1
        live = np.ones(v.shape, dtype=bool)
        for k in range(1, terms):
            fact *= (2 * k - 1) * (2 * k)
            coef = (-1.0) ** k / (2 * k * fact)
            if coef == 0.0:
                break
            x2k = x2k * x2
            term = coef * x2k
            total = np.where(live, total + term, total)
            live &= np.abs(term) >= tol
            if not live.any():
                break
        out[small] = total
    big = ~small
    if big.any():
        out[big] = -np.real(_e1_imag_cf(xa[big]))
    return _ret(out, scalar, shape)


def _zeta_borwein(s, n: int = 40):
    """Borwein's acceleration of the alternating zeta (eta) series.

    ``eta(s) = sum (-1)^{k-1} k^{-s} = (1 - 2^{1-s}) zeta(s)`` converges for
    every ``s > 0`` but far too slowly to use directly; weighting the partial
    sums by the Chebyshev-derived coefficients ``d_k`` below turns it into
    roughly ``n`` correct digits from ``n`` terms.
    """
    d = _borwein_weights(n)
    total = np.zeros_like(s)
    for k in range(n):
        total += ((-1.0) ** k) * (d[k] - d[n]) / (k + 1.0) ** s
    return -total / (d[n] * (1.0 - 2.0 ** (1.0 - s)))


_BORWEIN_CACHE: dict[int, np.ndarray] = {}


def _borwein_weights(n: int) -> np.ndarray:
    """``d_k`` for Borwein's algorithm; depends only on ``n``, so it is cached."""
    cached = _BORWEIN_CACHE.get(n)
    if cached is not None:
        return cached
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
    _BORWEIN_CACHE[n] = d
    return d


_ZETA_BERNOULLI = [1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66,
                   -691.0 / 2730, 7.0 / 6]


def _zeta_euler_maclaurin(s):
    """Euler-Maclaurin acceleration, valid for ``s > 1``."""
    n = 20
    total = np.zeros_like(s)
    for k in range(1, n):
        total += float(k) ** (-s)
    total += n ** (1 - s) / (s - 1) + 0.5 * float(n) ** -s
    fact = np.array(s, dtype=float, copy=True)
    for j, B in enumerate(_ZETA_BERNOULLI, start=1):
        total += B / math.factorial(2 * j) * fact * float(n) ** (-s - 2 * j + 1)
        fact = fact * ((s + 2 * j - 1) * (s + 2 * j))
    return total


def zeta(s, terms: int = 100000, tol: float = 1e-15):
    """Riemann zeta function, for any real ``s != 1``.

    ``s > 1`` uses Euler-Maclaurin acceleration directly.  ``s < 1`` is
    reflected through the functional equation
    ``zeta(s) = 2^s pi^{s-1} sin(pi s / 2) Gamma(1-s) zeta(1-s)``,
    which is how values like ``zeta(-1) = -1/12`` arise.  ``s = 1`` is the
    pole, and the negative even integers are the trivial zeros.
    """
    sa, scalar, shape = _arr(s)
    if np.any(sa == 1.0):
        raise DomainError("zeta has a simple pole at s = 1")
    out = np.empty_like(sa)
    # The functional equation maps s to 1-s, which stays inside (0, 1) -- and
    # is a fixed point at s = 1/2 -- so reflection cannot reach the critical
    # strip.  Borwein's alternating-series algorithm can.
    strip = (sa > 0.0) & (sa < 1.0)
    if strip.any():
        out[strip] = _zeta_borwein(sa[strip])
    high = sa > 1.0
    if high.any():
        out[high] = _zeta_euler_maclaurin(sa[high])
    low = ~strip & ~high
    if low.any():
        v = sa[low]
        res = np.empty_like(v)
        zero = v == 0.0
        res[zero] = -0.5
        trivial = ~zero & (v == np.floor(v)) & (np.floor(v) % 2 == 0)
        res[trivial] = 0.0               # trivial zeros at -2, -4, -6, ...
        rest = ~zero & ~trivial
        if rest.any():
            w = v[rest]
            # Now 1 - w > 1, so the reflected value comes from Euler-Maclaurin.
            res[rest] = (2.0**w * np.pi ** (w - 1.0) * np.sin(0.5 * np.pi * w)
                         * _gamma_arr(1.0 - w) * _zeta_euler_maclaurin(1.0 - w))
        out[low] = res
    return _ret(out, scalar, shape)


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
    xa, scalar, shape = _arr(x)
    inv_e = -1.0 / math.e
    if branch not in (0, -1):
        raise DomainError("only the real branches 0 and -1 are implemented")
    if np.any(xa < inv_e):
        bad = float(xa[xa < inv_e][0])
        raise DomainError(
            f"lambert_w is undefined below -1/e for real x (got {bad})")
    if branch == -1 and np.any(xa >= 0.0):
        raise DomainError("branch -1 requires -1/e <= x < 0")

    w = np.empty_like(xa)
    branch_pt = xa == inv_e
    w[branch_pt] = -1.0
    live = ~branch_pt
    if branch == 0:
        zero = live & (xa == 0.0)
        w[zero] = 0.0
        live = live & ~zero
    if not live.any():
        return _ret(w, scalar, shape)

    v = xa[live]
    if branch == 0:
        guess = np.where(v < 3.0, np.log1p(np.maximum(v, inv_e)),
                         np.log(np.maximum(v, 1.0))
                         - np.log(np.log(np.maximum(v, math.e))))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            guess = np.where(v > -1e-6,
                             np.log(-v) - np.log(-np.log(-np.minimum(v, -1e-300))),
                             -2.0)
    near = v < inv_e + 1e-3                       # series about the branch point
    if near.any():
        p = np.sqrt(2.0 * (math.e * v[near] + 1.0))
        if branch == -1:
            p = -p
        guess[near] = -1.0 + p - p * p / 3.0
    wv = guess

    active = np.ones(wv.shape, dtype=bool)
    for _ in range(max_iter):
        ew = np.exp(wv)
        f = wv * ew - v
        # Halley: divides out the second-order term that slows plain Newton
        # to a crawl near the branch point.
        denom = ew * (wv + 1.0) - (wv + 2.0) * f / (2.0 * wv + 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(denom == 0.0, 0.0, f / denom)
        active &= (f != 0.0) & (denom != 0.0)
        if not active.any():
            break
        wv = np.where(active, wv - step, wv)
        active &= np.abs(step) >= tol * np.maximum(1.0, np.abs(wv))
        if not active.any():
            break
    w[live] = wv
    return _ret(w, scalar, shape)


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
    xa, scalar, shape = _arr(x)
    ax = np.abs(xa)
    out = np.empty_like(ax)
    tiny = ax < 1e-8                    # F(x) = x - 2x^3/3 + ...
    out[tiny] = xa[tiny] * (1.0 - 2.0 * xa[tiny] * xa[tiny] / 3.0)
    live = ~tiny
    if live.any():
        v = ax[live]
        # Write x = n0*h + xp with n0 even, so the odd lattice point n becomes
        # n0 + k with k odd.  Re-indexing this way keeps the Gaussian centred
        # near xp (bounded by h, so it never underflows) while the denominator
        # stays n0 + k -- shifting the exponent without shifting the
        # denominator would evaluate a different function entirely.
        n0 = 2 * np.floor(0.5 * v / h + 0.5)
        xp = v - n0 * h
        e1 = np.exp(2.0 * xp * h)
        e2 = e1 * e1
        e = e1.copy()
        total = np.zeros_like(v)
        for i in range(terms):
            k = 2 * i + 1
            gauss = math.exp(-((k * h) ** 2))
            if gauss == 0.0:
                break
            # d2 = n0 - k is odd (n0 is even, k is odd) and so never zero.
            total += gauss * (e / (n0 + k) + 1.0 / ((n0 - k) * e))
            e *= e2
        out[live] = np.exp(-xp * xp) * total / math.sqrt(math.pi)
        out[live] = np.where(xa[live] >= 0, out[live], -out[live])
    return _ret(out, scalar, shape)


def _erfcx_arr(x):
    out = np.empty_like(x)
    neg = x < 0.0
    if neg.any():
        with np.errstate(over="ignore"):
            out[neg] = 2.0 * np.exp(x[neg] * x[neg]) - _erfcx_arr(-x[neg])
    mid = ~neg & (x < 5.0)
    if mid.any():
        v = x[mid]
        out[mid] = np.exp(v * v) * _erfc_arr(v)
    far = ~neg & ~mid
    if far.any():
        v = x[far]
        # erfcx(x) = (1/(x sqrt(pi))) * continued fraction, stable for large x.
        inv = 1.0 / (v * math.sqrt(math.pi))
        term = inv.copy()
        total = inv.copy()
        live = np.ones(v.shape, dtype=bool)
        # 2x^2 overflows past x ~ 1e154. That is the right answer here: the
        # correction terms are divided by it, so they vanish and the leading
        # 1/(x sqrt(pi)) stands alone, which is exactly the asymptotic limit.
        with np.errstate(over="ignore"):
            v2 = 2.0 * v * v
        for n in range(1, 40):
            new = -term * (2 * n - 1) / v2
            live &= np.abs(new) <= np.abs(term)
            if not live.any():
                break
            term = np.where(live, new, term)
            total = np.where(live, total + term, total)
        out[far] = total
    return out


def erfcx(x):
    """Scaled complementary error function ``e^{x^2} erfc(x)``.

    Stays finite and smooth for large positive ``x``, where ``erfc`` itself
    underflows to zero and ``e^{x^2}`` overflows -- the product is the only
    numerically usable form.
    """
    xa, scalar, shape = _arr(x)
    return _ret(_erfcx_arr(xa), scalar, shape)


def fresnel_s(x, tol: float = 1e-15):
    """Fresnel sine integral ``S(x) = int_0^x sin(pi t^2 / 2) dt``."""
    return _fresnel(x, tol)[0]


def fresnel_c(x, tol: float = 1e-15):
    """Fresnel cosine integral ``C(x) = int_0^x cos(pi t^2 / 2) dt``."""
    return _fresnel(x, tol)[1]


_FRESNEL_GL: tuple[np.ndarray, np.ndarray] | None = None


def _fresnel_gl_rule():
    """120-point Gauss-Legendre rule on ``[-1, 1]``, built once."""
    global _FRESNEL_GL
    if _FRESNEL_GL is None:
        from ..approx.orthopoly import gauss_legendre_nodes
        _FRESNEL_GL = gauss_legendre_nodes(120)
    return _FRESNEL_GL


def _fresnel(x, tol: float = 1e-15):
    """Both Fresnel integrals as ``(S, C)``.

    The power series is exact for moderate ``x`` but its terms grow like
    ``(pi x^2/2)^n / n!`` before decaying, so it cancels away all precision
    once ``x`` is large.  Past ``x = 3`` the asymptotic form takes over,
    written through the auxiliary functions ``f`` and ``g``, which decay
    smoothly instead of oscillating.
    """
    xa, scalar, shape = _arr(x)
    ax = np.abs(xa)
    s_out = np.empty_like(ax)
    c_out = np.empty_like(ax)

    small = ax < 3.0
    if small.any():
        v = ax[small]
        # S = sum (-1)^n (pi/2)^{2n+1} x^{4n+3} / ((2n+1)! (4n+3))
        # C = sum (-1)^n (pi/2)^{2n}   x^{4n+1} / ((2n)!   (4n+1))
        hp = 0.5 * math.pi
        x4 = v ** 4
        s = np.zeros_like(v)
        term = hp * v ** 3 / 3.0
        live = np.ones(v.shape, dtype=bool)
        for n in range(300):
            live &= np.abs(term) > tol * np.maximum(np.abs(s), 1e-300)
            if not live.any():
                break
            s = np.where(live, s + term, s)
            m = n + 1
            term = term * (-(hp * hp) * x4 / ((2 * m) * (2 * m + 1)))
            term = term * ((4 * m - 1) / (4 * m + 3))
        c = np.zeros_like(v)
        term = v.copy()
        live = np.ones(v.shape, dtype=bool)
        for n in range(300):
            live &= np.abs(term) > tol * np.maximum(np.abs(c), 1e-300)
            if not live.any():
                break
            c = np.where(live, c + term, c)
            m = n + 1
            term = term * (-(hp * hp) * x4 / ((2 * m - 1) * (2 * m)))
            term = term * ((4 * m - 3) / (4 * m + 1))
        s_out[small], c_out[small] = s, c

    mid = ~small & (ax < 6.0)
    if mid.any():
        # Between the two expansions neither is good enough: the series has
        # already lost ~8 digits to cancellation and the asymptotic series
        # bottoms out at about the same size (both scale as exp(+-pi x^2/2),
        # so they cross while each is still around 1e-8).  Direct Gauss-
        # Legendre closes the gap -- the integrand is smooth with only a few
        # oscillations over this range, so a fixed high-order rule is exact.
        nodes, weights = _fresnel_gl_rule()
        v = ax[mid]
        half = 0.5 * v
        # One (elements, 120) evaluation replaces two quadrature calls each.
        t = half[:, None] * (nodes + 1.0)
        arg = 0.5 * math.pi * t * t
        s_out[mid] = half * (np.sin(arg) @ weights)
        c_out[mid] = half * (np.cos(arg) @ weights)

    big = ~small & ~mid
    if big.any():
        v = ax[big]
        u = 0.5 * math.pi * v * v
        z = math.pi * v * v
        # f ~ 1/(pi x) sum (-1)^m (4m-1)!! / z^{2m},
        # g ~ 1/(pi^2 x^3) sum (-1)^m (4m+1)!! / z^{2m}
        f = _asymptotic_aux(z, -1) / (math.pi * v)
        g = _asymptotic_aux(z, 1) / (math.pi * math.pi * v ** 3)
        s_out[big] = 0.5 - f * np.cos(u) - g * np.sin(u)
        c_out[big] = 0.5 + f * np.sin(u) - g * np.cos(u)

    sign = np.where(xa >= 0, 1.0, -1.0)
    return (_ret(sign * s_out, scalar, shape),
            _ret(sign * c_out, scalar, shape))


def _asymptotic_aux(z, offset: int):
    """Sum ``(-1)^m (4m + offset)!! / z^{2m}``, truncated at its smallest term.

    The series is divergent, so it is cut where the terms stop shrinking --
    the point of optimal truncation, beyond which adding terms makes the
    answer worse.  Each element stops at its own optimum.
    """
    term = np.ones_like(z)
    total = np.ones_like(z)
    prev = np.full(z.shape, np.inf)
    live = np.ones(z.shape, dtype=bool)
    z2 = z * z
    for m in range(1, 40):
        # (4m+offset)!!/(4m-4+offset)!! = (4m+offset)(4m-2+offset)
        new = term * (-(4.0 * m + offset) * (4.0 * m - 2.0 + offset) / z2)
        live &= np.abs(new) <= prev
        if not live.any():
            break
        term = np.where(live, new, term)
        prev = np.where(live, np.abs(new), prev)
        total = np.where(live, total + term, total)
    return total


_POLYGAMMA_BERNOULLI = [1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66,
                        -691.0 / 2730, 7.0 / 6, -3617.0 / 510]


def polygamma(n: int, x):
    """Polygamma ``psi^(n)(x)``, the ``n``-th derivative of ``log Gamma``.

    ``n = 0`` delegates to :func:`digamma`.  For ``n >= 1`` the recurrence
    ``psi^(n)(x) = psi^(n)(x+1) + (-1)^n n! / x^{n+1}`` pushes the argument
    into the range where the Hurwitz-zeta asymptotic series is accurate:
    ``psi^(n)(x) = (-1)^{n+1} n! zeta(n+1, x)``.
    """
    n = int(n)
    if n < 0:
        raise DomainError("polygamma requires n >= 0")
    if n == 0:
        return digamma(x)
    xa, scalar, shape = _arr(x)
    if np.any((xa <= 0) & (xa == np.floor(xa))):
        raise DomainError("polygamma has poles at the non-positive integers")
    y = xa.astype(float, copy=True)
    shift = np.zeros_like(y)
    sign_n = (-1.0) ** n
    fact_n = float(math.factorial(n))
    # psi^(n)(x+1) = psi^(n)(x) + (-1)^n n! x^{-(n+1)}, so stepping *up* in x
    # accumulates the term with the opposite sign.
    for _ in range(4096):               # recur up into the asymptotic regime
        low = y < 15.0
        if not low.any():
            break
        shift[low] -= sign_n * fact_n / y[low] ** (n + 1)
        y[low] += 1.0
    # Hurwitz zeta by Euler-Maclaurin: zeta(s,x) = x^{1-s}/(s-1) + x^{-s}/2 + ...
    s = n + 1
    total = y ** (1.0 - s) / (s - 1.0) + 0.5 * y**-s
    fact = float(s)
    for j, B in enumerate(_POLYGAMMA_BERNOULLI, start=1):
        total += B / math.factorial(2 * j) * fact * y ** (-s - 2 * j + 1)
        fact *= (s + 2 * j - 1) * (s + 2 * j)
    out = ((-1.0) ** (n + 1)) * fact_n * total + shift
    return _ret(out, scalar, shape)


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
    xa, scalar, shape = _arr(x)
    out = np.empty_like(xa)
    zero = xa == 0.0
    out[zero] = 1.0 if n == 0 else 0.0
    live = ~zero
    if not live.any():
        return _ret(out, scalar, shape)
    v = xa[live]
    if n == 0:
        out[live] = np.sin(v) / v
        return _ret(out, scalar, shape)
    if n == 1:
        out[live] = np.sin(v) / (v * v) - np.cos(v) / v
        return _ret(out, scalar, shape)

    res = np.empty_like(v)
    up = n < v                                      # upward is stable here
    if up.any():
        w = v[up]
        jm1, j = np.sin(w) / w, np.sin(w) / (w * w) - np.cos(w) / w
        for k in range(1, n):
            jm1, j = j, (2 * k + 1) / w * j - jm1
        res[up] = j
    down = ~up
    if down.any():
        w = v[down]
        m = n + int(math.sqrt(80.0 * max(float(n), float(np.abs(w).max()))))
        jp1 = np.zeros_like(w)
        jc = np.ones_like(w)
        ans = np.zeros_like(w)
        for k in range(m, 0, -1):
            # Entering the step, (jp1, jc) = (j_{k+1}, j_k); leaving it they
            # are (j_k, j_{k-1}).  So j_n appears in *jc* after the k = n+1 step.
            jp1, jc = jc, (2 * k + 1) / w * jc - jp1
            big = np.abs(jc) > 1e100
            if big.any():
                scale = np.where(big, 1e-100, 1.0)
                jp1, jc, ans = jp1 * scale, jc * scale, ans * scale
            if k == n + 1:
                ans = jc
        res[down] = ans * (np.sin(w) / w) / jc   # normalize against true j_0
    out[live] = res
    return _ret(out, scalar, shape)


def spherical_bessel_y(n: int, x):
    """Spherical Bessel ``y_n(x) = sqrt(pi/2x) Y_{n+1/2}(x)``.

    Upward recurrence is stable for ``y`` at every order -- it is the growing
    solution, so relative error does not accumulate.
    """
    n = int(n)
    xa, scalar, shape = _arr(x)
    if np.any(xa <= 0.0):
        raise DomainError("spherical_bessel_y requires x > 0")
    ym1 = -np.cos(xa) / xa
    if n == 0:
        return _ret(ym1, scalar, shape)
    y = -np.cos(xa) / (xa * xa) - np.sin(xa) / xa
    for k in range(1, n):
        ym1, y = y, (2 * k + 1) / xa * y - ym1
    return _ret(y, scalar, shape)


def associated_legendre(l: int, m: int, x):
    """Associated Legendre ``P_l^m(x)`` on ``[-1, 1]`` (Condon-Shortley phase).

    Built from the closed form for ``P_m^m`` and then recurred up in ``l``,
    which is the stable direction.
    """
    l, m = int(l), int(m)
    xa, scalar, shape = _arr(x)
    if m < 0 or m > l or np.any(np.abs(xa) > 1.0):
        raise DomainError("associated_legendre requires 0 <= m <= l and |x| <= 1")
    pmm = np.ones_like(xa)
    if m > 0:
        somx2 = np.sqrt(np.maximum((1.0 - xa) * (1.0 + xa), 0.0))
        fact = 1.0
        for _ in range(m):
            pmm = pmm * (-fact * somx2)   # the (-1)^m Condon-Shortley phase
            fact += 2.0
    if l == m:
        return _ret(pmm, scalar, shape)
    pmmp1 = xa * (2 * m + 1) * pmm
    if l == m + 1:
        return _ret(pmmp1, scalar, shape)
    pll = np.zeros_like(xa)
    for ll in range(m + 2, l + 1):
        pll = (xa * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return _ret(pll, scalar, shape)


def spherical_harmonic(l: int, m: int, theta, phi):
    """Complex spherical harmonic ``Y_l^m(theta, phi)``.

    ``theta`` is the polar angle and ``phi`` the azimuth.  Negative ``m`` uses
    ``Y_l^{-m} = (-1)^m conj(Y_l^m)``.  Normalized so that the harmonics are
    orthonormal over the sphere.
    """
    l, m = int(l), int(m)
    if abs(m) > l:
        raise DomainError("spherical_harmonic requires |m| <= l")
    (th, ph), scalar, shape = _bcast(theta, phi)
    if m < 0:
        inner = spherical_harmonic(l, -m, th, ph)
        return _ret(((-1.0) ** m) * np.conj(np.atleast_1d(inner)), scalar, shape)
    norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                     * math.factorial(l - m) / math.factorial(l + m))
    plm = np.atleast_1d(associated_legendre(l, m, np.cos(th)))
    return _ret(norm * plm * np.exp(1j * m * ph), scalar, shape)


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
    (aa, ba, za), scalar, shape = _bcast(a, b, z)
    if np.any((ba <= 0) & (ba == np.floor(ba))):
        raise DomainError("1F1 is undefined for non-positive integer b")
    out = np.empty(aa.shape)
    neg = za < 0.0
    if neg.any():
        out[neg] = np.exp(za[neg]) * _1f1_series(ba[neg] - aa[neg], ba[neg],
                                                 -za[neg], tol, max_terms)
    pos = ~neg
    if pos.any():
        out[pos] = _1f1_series(aa[pos], ba[pos], za[pos], tol, max_terms)
    return _ret(out, scalar, shape)


def _1f1_series(a, b, z, tol, max_terms):
    term = np.ones(a.shape)
    total = np.ones(a.shape)
    live = np.ones(a.shape, dtype=bool)
    for n in range(max_terms):
        term = term * ((a + n) * z / ((b + n) * (n + 1)))
        total = np.where(live, total + term, total)
        live &= np.abs(term) >= tol * np.maximum(np.abs(total), 1e-300)
        if not live.any():
            break
    return total


def hyp2f1(a, b, c, z, tol: float = 1e-15, max_terms: int = 20000):
    """Gauss hypergeometric ``2F1(a, b; c; z)`` for real ``|z| < 1``.

    The series converges only inside the unit disc.  For ``z`` in
    ``(-1, -0.5)`` the Pfaff transformation moves the argument into
    ``(0, 1/2)``, where convergence is fast; without it the series near
    ``z = -1`` needs impractically many terms.
    """
    (aa, ba, ca, za), scalar, shape = _bcast(a, b, c, z)
    if np.any((ca <= 0) & (ca == np.floor(ca))):
        raise DomainError("2F1 is undefined for non-positive integer c")
    if np.any(np.abs(za) >= 1.0):
        raise DomainError("this implementation requires |z| < 1")
    out = np.empty(aa.shape)
    # Pfaff: 2F1(a,b;c;z) = (1-z)^-a 2F1(a,c-b;c;z/(z-1))
    pfaff = za < -0.5
    if pfaff.any():
        w = za[pfaff]
        out[pfaff] = (1.0 - w) ** (-aa[pfaff]) * _2f1_series(
            aa[pfaff], ca[pfaff] - ba[pfaff], ca[pfaff], w / (w - 1.0),
            tol, max_terms)
    rest = ~pfaff
    if rest.any():
        out[rest] = _2f1_series(aa[rest], ba[rest], ca[rest], za[rest],
                                tol, max_terms)
    return _ret(out, scalar, shape)


def _2f1_series(a, b, c, z, tol, max_terms):
    term = np.ones(a.shape)
    total = np.ones(a.shape)
    live = np.ones(a.shape, dtype=bool)
    for n in range(max_terms):
        term = term * ((a + n) * (b + n) * z / ((c + n) * (n + 1)))
        total = np.where(live, total + term, total)
        live &= np.abs(term) >= tol * np.maximum(np.abs(total), 1e-300)
        if not live.any():
            break
    return total


def expint_n(n: int, x, tol: float = 1e-14, max_iter: int = 200):
    """Generalized exponential integral ``E_n(x) = int_1^inf e^{-xt}/t^n dt``.

    A continued fraction for ``x > 1`` and a series for ``x <= 1``; the two
    regimes are exactly where each converges quickly.
    """
    n = int(n)
    xa, scalar, shape = _arr(x)
    if n < 0 or np.any(xa < 0) or (n < 2 and np.any(xa == 0)):
        raise DomainError("expint_n requires n >= 0, x >= 0, and n >= 2 when x = 0")
    if n == 0:
        return _ret(np.exp(-xa) / xa, scalar, shape)
    out = np.empty_like(xa)
    zero = xa == 0.0
    if zero.any():
        # n >= 2 here: the n < 2 case was rejected above, and 1/(n-1) must not
        # be evaluated for n == 1 even when no element selects it.
        out[zero] = 1.0 / (n - 1)
    frac = ~zero & (xa > 1.0)
    if frac.any():
        v = xa[frac]
        b = v + n
        c = np.full(v.shape, 1e300)
        d = 1.0 / b
        h = d.copy()
        live = np.ones(v.shape, dtype=bool)
        for i in range(1, max_iter + 1):
            a = -float(i * (n - 1 + i))
            b = b + 2.0
            d = 1.0 / (a * d + b)
            c = b + a / c
            delta = c * d
            h = np.where(live, h * delta, h)
            live &= np.abs(delta - 1.0) >= tol
            if not live.any():
                break
        out[frac] = h * np.exp(-v)
    ser = ~zero & ~frac
    if ser.any():
        w = xa[ser]
        if n != 1:
            total = np.full(w.shape, 1.0 / (n - 1))
        else:
            total = -np.log(w) - _EULER
        psi = -_EULER + sum(1.0 / k for k in range(1, n))
        term = np.ones_like(w)
        live = np.ones(w.shape, dtype=bool)
        for i in range(1, max_iter + 1):
            term = term * (-w / i)
            if i != n - 1:
                delta = -term / (i - (n - 1))
            else:
                delta = term * (psi - np.log(w))
            total = np.where(live, total + delta, total)
            live &= np.abs(delta) >= tol * np.abs(total)
            if not live.any():
                break
        out[ser] = total
    return _ret(out, scalar, shape)


def struve_h0(x, tol: float = 1e-14, max_terms: int = 300):
    """Struve function ``H_0(x)``: the particular solution of the driven Bessel equation."""
    xa, scalar, shape = _arr(x)
    term = np.full(xa.shape, 2.0 / math.pi)
    total = term.copy()
    x2 = xa * xa
    live = np.ones(xa.shape, dtype=bool)
    for k in range(1, max_terms):
        term = term * (-x2 / ((2 * k + 1) ** 2))
        total = np.where(live, total + term, total)
        live &= np.abs(term) >= tol * np.maximum(np.abs(total), 1e-300)
        if not live.any():
            break
    return _ret(total * xa, scalar, shape)


def logistic(x):
    """Logistic sigmoid, written to avoid overflow for either sign of ``x``."""
    xa, scalar, shape = _arr(x)
    out = np.empty_like(xa)
    pos = xa >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-xa[pos]))
    ex = np.exp(xa[~pos])               # e^x, never e^{-x}, when x is very negative
    out[~pos] = ex / (1.0 + ex)
    return _ret(out, scalar, shape)


def logit(p):
    """Inverse of :func:`logistic`: ``log(p / (1 - p))``."""
    pa, scalar, shape = _arr(p)
    if np.any((pa <= 0) | (pa >= 1)):
        raise DomainError("logit requires 0 < p < 1")
    return _ret(np.log(pa) - np.log1p(-pa), scalar, shape)
