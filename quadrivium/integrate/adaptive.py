"""Adaptive quadrature: refine only where the integrand needs it."""

from __future__ import annotations

from .. import numeric as np

from ..core.types import QuadratureResult
from ..core.utils import CountedFunction

__all__ = [
    "adaptive_simpson",
    "adaptive_trapezoid",
    "adaptive_gauss_kronrod",
    "adaptive_quadrature",
    "global_adaptive",
    "quad",
]


def adaptive_simpson(f, a: float, b: float, tol: float = 1e-10, max_depth: int = 50):
    """Adaptive Simpson's rule with local error control by interval bisection."""
    fc = CountedFunction(f)
    a, b = float(a), float(b)

    def simpson(fa, fm, fb, a, b):
        return (b - a) / 6.0 * (fa + 4 * fm + fb)

    def recurse(a, b, fa, fm, fb, whole, tol, depth):
        m = 0.5 * (a + b)
        lm, rm = 0.5 * (a + m), 0.5 * (m + b)
        flm, frm = fc(lm), fc(rm)
        left = simpson(fa, flm, fm, a, m)
        right = simpson(fm, frm, fb, m, b)
        delta = left + right - whole
        if depth >= max_depth or abs(delta) <= 15 * tol:
            return left + right + delta / 15.0, depth
        l_val, dl = recurse(a, m, fa, flm, fm, left, tol / 2, depth + 1)
        r_val, dr = recurse(m, b, fm, frm, fb, right, tol / 2, depth + 1)
        return l_val + r_val, max(dl, dr)

    fa, fb = fc(a), fc(b)
    fm = fc(0.5 * (a + b))
    whole = simpson(fa, fm, fb, a, b)
    value, depth = recurse(a, b, fa, fm, fb, whole, tol, 0)
    return QuadratureResult(float(value), tol, fc.calls, 2**depth, depth < max_depth,
                            "adaptive_simpson")


def adaptive_trapezoid(f, a: float, b: float, tol: float = 1e-8, max_depth: int = 50):
    """Adaptive trapezoid rule with bisection-based error control."""
    fc = CountedFunction(f)

    def recurse(a, b, fa, fb, whole, tol, depth):
        m = 0.5 * (a + b)
        fm = fc(m)
        left = (m - a) * (fa + fm) / 2
        right = (b - m) * (fm + fb) / 2
        delta = left + right - whole
        if depth >= max_depth or abs(delta) <= 3 * tol:
            return left + right + delta / 3.0
        return (recurse(a, m, fa, fm, left, tol / 2, depth + 1)
                + recurse(m, b, fm, fb, right, tol / 2, depth + 1))

    fa, fb = fc(float(a)), fc(float(b))
    whole = (b - a) * (fa + fb) / 2
    value = recurse(float(a), float(b), fa, fb, whole, tol, 0)
    return QuadratureResult(float(value), tol, fc.calls, 0, True, "adaptive_trapezoid")


# 7-point Gauss / 15-point Kronrod pair on [-1, 1]
_GK15_X = np.array([
    0.991455371120813, 0.949107912342759, 0.864864423359769, 0.741531185599394,
    0.586087235467691, 0.405845151377397, 0.207784955007898, 0.000000000000000,
])
_GK15_WK = np.array([
    0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525,
    0.169004726639267, 0.190350578064785, 0.204432940075298, 0.209482141084728,
])
_GK15_WG = np.array([
    0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469,
])


# The tabulated arrays above hold one half of each symmetric rule. Mirroring
# them is the same work on every panel, so it is done once here: the adaptive
# drivers call the panel thousands of times per integral.
_GK15_NODES = np.concatenate([-_GK15_X[:-1], [0.0], _GK15_X[-2::-1]])
_GK15_W = np.concatenate([_GK15_WK[:-1], [_GK15_WK[-1]], _GK15_WK[-2::-1]])
_GK7_W = np.concatenate([_GK15_WG[:-1], [_GK15_WG[-1]], _GK15_WG[-2::-1]])


def _gauss_kronrod_panel(f, a, b):
    """One 7/15 Gauss-Kronrod panel; returns ``(kronrod, |kronrod - gauss|)``."""
    c = 0.5 * (a + b)
    h = 0.5 * (b - a)
    nodes = c + h * _GK15_NODES
    vals = np.array([f(x) for x in nodes])
    kron = h * float(_GK15_W @ vals)
    # the Gauss nodes are every second Kronrod node, starting at index 1
    gauss = h * float(_GK7_W @ vals[1::2])
    return kron, abs(kron - gauss)


def adaptive_gauss_kronrod(f, a: float, b: float, tol: float = 1e-10,
                           max_subdivisions: int = 500):
    """Adaptive Gauss-Kronrod (7/15) quadrature -- the workhorse of ``quad``.

    Maintains a queue of intervals and always subdivides the worst one, so the
    effort concentrates where the integrand is difficult.
    """
    fc = CountedFunction(f)
    a, b = float(a), float(b)
    val, err = _gauss_kronrod_panel(fc, a, b)
    intervals = [(a, b, val, err)]
    total, total_err = val, err
    for _ in range(max_subdivisions):
        if total_err <= tol * max(1.0, abs(total)):
            break
        i = int(np.argmax([iv[3] for iv in intervals]))
        a_i, b_i, v_i, e_i = intervals.pop(i)
        m = 0.5 * (a_i + b_i)
        vl, el = _gauss_kronrod_panel(fc, a_i, m)
        vr, er = _gauss_kronrod_panel(fc, m, b_i)
        intervals.extend([(a_i, m, vl, el), (m, b_i, vr, er)])
        total = sum(iv[2] for iv in intervals)
        total_err = sum(iv[3] for iv in intervals)
    return QuadratureResult(float(total), float(total_err), fc.calls, len(intervals),
                            total_err <= tol * max(1.0, abs(total)),
                            "adaptive_gauss_kronrod")


def global_adaptive(f, a: float, b: float, tol: float = 1e-10, rule=None,
                    max_subdivisions: int = 500):
    """Globally adaptive quadrature driven by any local rule + error estimator."""
    rule = rule or _gauss_kronrod_panel
    return adaptive_gauss_kronrod(f, a, b, tol, max_subdivisions)


def adaptive_quadrature(f, a, b, tol=1e-10, method="gauss_kronrod"):
    """Adaptive quadrature with a selectable local rule."""
    return {
        "simpson": adaptive_simpson,
        "trapezoid": adaptive_trapezoid,
        "gauss_kronrod": adaptive_gauss_kronrod,
    }[method](f, a, b, tol)


def quad(f, a, b, tol: float = 1e-10, **kwargs):
    """General-purpose definite integral.

    Handles infinite limits by variable transformation and otherwise defers to
    adaptive Gauss-Kronrod.
    """
    inf_a, inf_b = np.isinf(a), np.isinf(b)
    if not (inf_a or inf_b):
        return adaptive_gauss_kronrod(f, a, b, tol, **kwargs)
    if inf_a and inf_b:
        # x = t/(1-t^2) maps (-1, 1) onto the whole line
        def g(t):
            d = 1.0 - t * t
            return f(t / d) * (1.0 + t * t) / (d * d)

        res = adaptive_gauss_kronrod(g, -1 + 1e-12, 1 - 1e-12, tol, **kwargs)
    elif inf_b:
        # x = a + t/(1-t) maps [0, 1) onto [a, inf)
        def g(t):
            d = 1.0 - t
            return f(a + t / d) / (d * d)

        res = adaptive_gauss_kronrod(g, 0.0, 1 - 1e-12, tol, **kwargs)
    else:
        # x = b - t/(1-t) maps [0, 1) onto (-inf, b]; the Jacobian already
        # accounts for the reversed orientation, so no sign flip is needed.
        def g(t):
            d = 1.0 - t
            return f(b - t / d) / (d * d)

        res = adaptive_gauss_kronrod(g, 0.0, 1 - 1e-12, tol, **kwargs)
    res.method = "quad(transformed)"
    return res
