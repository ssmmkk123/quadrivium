"""Gaussian quadrature drivers and the specialised high-accuracy rules.

Gauss rules attain degree ``2n-1`` with ``n`` nodes; Clenshaw-Curtis and
tanh-sinh trade a little of that optimality for nested points and for immunity
to endpoint singularities.
"""

from __future__ import annotations

import numpy as np

from ..approx.orthopoly import (
    gauss_chebyshev_nodes,
    gauss_hermite_nodes,
    gauss_jacobi_nodes,
    gauss_laguerre_nodes,
    gauss_legendre_nodes,
    gauss_lobatto_nodes,
    gauss_radau_nodes,
)
from ..core.exceptions import DomainError
from ..core.types import QuadratureResult
from ..core.utils import CountedFunction

__all__ = [
    "gauss_legendre",
    "gauss_chebyshev",
    "gauss_hermite",
    "gauss_laguerre",
    "gauss_jacobi",
    "gauss_lobatto",
    "gauss_radau",
    "gauss_kronrod",
    "composite_gauss",
    "clenshaw_curtis",
    "fejer",
    "tanh_sinh",
    "double_exponential",
    "singular_endpoint_quadrature",
    "filon",
    "cauchy_principal_value",
    "hadamard_finite_part",
]


def _apply(f, x, w, method):
    fc = CountedFunction(f)
    vals = np.array([fc(xi) for xi in x])
    return QuadratureResult(float(w @ vals), None, fc.calls, 1, True, method)


def gauss_legendre(f, a: float = -1.0, b: float = 1.0, n: int = 10):
    """Gauss-Legendre quadrature, exact for polynomials of degree ``2n-1``."""
    x, w = gauss_legendre_nodes(n, a, b)
    return _apply(f, x, w, f"gauss_legendre_{n}")


def gauss_chebyshev(f, n: int = 10, kind: int = 1):
    """Gauss-Chebyshev quadrature for the weight ``1/sqrt(1-x^2)`` on ``[-1,1]``.

    The returned value approximates ``int f(x) w(x) dx``, weight included.
    """
    x, w = gauss_chebyshev_nodes(n, kind)
    return _apply(f, x, w, f"gauss_chebyshev_{n}")


def gauss_hermite(f, n: int = 20):
    """Gauss-Hermite quadrature for ``int f(x) exp(-x^2) dx`` over the real line."""
    x, w = gauss_hermite_nodes(n)
    return _apply(f, x, w, f"gauss_hermite_{n}")


def gauss_laguerre(f, n: int = 20, alpha: float = 0.0):
    """Gauss-Laguerre quadrature for ``int_0^inf f(x) x^alpha exp(-x) dx``."""
    x, w = gauss_laguerre_nodes(n, alpha)
    return _apply(f, x, w, f"gauss_laguerre_{n}")


def gauss_jacobi(f, n: int = 10, alpha: float = 0.0, beta: float = 0.0):
    """Gauss-Jacobi quadrature for the weight ``(1-x)^alpha (1+x)^beta``."""
    x, w = gauss_jacobi_nodes(n, alpha, beta)
    return _apply(f, x, w, f"gauss_jacobi_{n}")


def gauss_lobatto(f, a: float = -1.0, b: float = 1.0, n: int = 10):
    """Gauss-Lobatto quadrature; includes both endpoints, degree ``2n-3``."""
    x, w = gauss_lobatto_nodes(n, a, b)
    return _apply(f, x, w, f"gauss_lobatto_{n}")


def gauss_radau(f, a: float = -1.0, b: float = 1.0, n: int = 10):
    """Gauss-Radau quadrature; includes the left endpoint, degree ``2n-2``."""
    x, w = gauss_radau_nodes(n, a, b)
    return _apply(f, x, w, f"gauss_radau_{n}")


def gauss_kronrod(f, a: float = -1.0, b: float = 1.0):
    """Single-panel 7/15 Gauss-Kronrod rule with its embedded error estimate."""
    from .adaptive import _gauss_kronrod_panel

    fc = CountedFunction(f)
    value, err = _gauss_kronrod_panel(fc, float(a), float(b))
    return QuadratureResult(float(value), float(err), fc.calls, 1, True, "gauss_kronrod_15")


def composite_gauss(f, a: float, b: float, panels: int = 10, n: int = 5):
    """Composite Gauss-Legendre: split into panels, apply an ``n``-point rule."""
    fc = CountedFunction(f)
    edges = np.linspace(float(a), float(b), panels + 1)
    total = 0.0
    for i in range(panels):
        x, w = gauss_legendre_nodes(n, edges[i], edges[i + 1])
        total += float(w @ np.array([fc(xi) for xi in x]))
    return QuadratureResult(float(total), None, fc.calls, panels, True,
                            f"composite_gauss_{panels}x{n}")


def clenshaw_curtis(f, a: float = -1.0, b: float = 1.0, n: int = 32):
    """Clenshaw-Curtis quadrature on Chebyshev points, weights via the FFT.

    Nearly as accurate as Gauss for smooth integrands, with nested nodes.
    """
    fc = CountedFunction(f)
    if n % 2:
        n += 1
    theta = np.pi * np.arange(n + 1) / n
    x_std = np.cos(theta)
    x = 0.5 * (a + b) + 0.5 * (b - a) * x_std
    vals = np.array([fc(xi) for xi in x])
    # exact Clenshaw-Curtis weights
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(1, n // 2):
        v -= 2.0 * np.cos(2 * k * theta[1:n]) / (4 * k * k - 1)
    v -= np.cos(n * theta[1:n]) / (n * n - 1)
    w[1:n] = 2.0 * v / n
    w[0] = w[n] = 1.0 / (n * n - 1)
    total = 0.5 * (b - a) * float(w @ vals)
    return QuadratureResult(float(total), None, fc.calls, 1, True, f"clenshaw_curtis_{n}")


def fejer(f, a: float = -1.0, b: float = 1.0, n: int = 32, rule: int = 1):
    """Fejer quadrature: the open cousin of Clenshaw-Curtis (no endpoints)."""
    fc = CountedFunction(f)
    if rule == 1:
        theta = np.pi * (2 * np.arange(n) + 1) / (2 * n)
        x_std = np.cos(theta)
        w = np.zeros(n)
        for j in range(n):
            s = sum(np.cos(2 * k * theta[j]) / (4 * k * k - 1)
                    for k in range(1, n // 2 + 1))
            w[j] = 2.0 / n * (1.0 - 2.0 * s)
    else:
        theta = np.pi * np.arange(1, n + 1) / (n + 1)
        x_std = np.cos(theta)
        w = np.zeros(n)
        for j in range(n):
            s = sum(np.sin((2 * k - 1) * theta[j]) / (2 * k - 1)
                    for k in range(1, (n + 1) // 2 + 1))
            w[j] = 4.0 * np.sin(theta[j]) / (n + 1) * s
    x = 0.5 * (a + b) + 0.5 * (b - a) * x_std
    vals = np.array([fc(xi) for xi in x])
    total = 0.5 * (b - a) * float(w @ vals)
    return QuadratureResult(float(total), None, fc.calls, 1, True, f"fejer{rule}_{n}")


def tanh_sinh(f, a: float = -1.0, b: float = 1.0, levels: int = 10,
              tol: float = 1e-14, f_offset=None):
    """Tanh-sinh (double exponential) quadrature.

    The substitution ``x = tanh(pi/2 sinh(t))`` sends the integrand and all its
    derivatives to zero at the endpoints, so endpoint singularities integrate
    cleanly and the trapezoid rule in ``t`` converges roughly exponentially.

    Two details matter for accuracy: the weights are formed from ``sech`` in a
    way that cannot overflow, and each abscissa is built as an offset from the
    nearer endpoint so that ``1 - |x|`` keeps its significant digits.

    Accuracy limit for singular integrands: an abscissa closer to an endpoint
    than ``eps * (b - a)`` rounds onto the endpoint itself, so the very tail of
    a singularity is lost -- about ``1e-8`` for an inverse-square-root. Pass
    ``f_offset(d, side)``, returning ``f(a + d)`` for ``side == -1`` and
    ``f(b - d)`` for ``side == +1``, to evaluate from the endpoint distance and
    recover full precision.
    """
    fc = CountedFunction(f)
    foc = CountedFunction(f_offset) if f_offset is not None else None
    a, b = float(a), float(b)
    c, r = 0.5 * (a + b), 0.5 * (b - a)
    # Hard cap: past this the weight underflows to zero in double precision.
    t_hard = 7.0

    def node(t):
        """Abscissa and weight for parameter ``t`` (overflow-free)."""
        u = 0.5 * np.pi * np.sinh(t)
        au = abs(u)
        e2 = np.exp(-2.0 * au)
        sech = 2.0 * np.exp(-au) / (1.0 + e2)
        w = 0.5 * np.pi * np.cosh(t) * sech * sech
        delta = 2.0 * e2 / (1.0 + e2)        # = 1 - tanh(|u|), accurate near 0
        x = b - r * delta if u >= 0 else a + r * delta
        return x, w

    def contribution(t):
        x, w = node(t)
        if w == 0.0:
            return 0.0
        if f_offset is not None:
            u = 0.5 * np.pi * np.sinh(t)
            e2 = np.exp(-2.0 * abs(u))
            d = r * (2.0 * e2 / (1.0 + e2))
            if d == 0.0:
                return 0.0
            v = foc(d, 1 if u >= 0 else -1)
        else:
            if not (a < x < b):
                return 0.0
            v = fc(x)
        return w * v if np.isfinite(v) else 0.0

    def accumulate(start, stride):
        """Sum symmetric pairs from ``t = start`` upward until negligible.

        The truncation point is found from the contributions themselves rather
        than fixed in advance, because a strong singularity such as ``x^-0.9``
        leaves the transformed integrand decaying far more slowly than usual.
        """
        total = 0.0
        t = start
        negligible = 0
        while t <= t_hard:
            c_t = contribution(t) + contribution(-t)
            total += c_t
            if abs(c_t) <= 1e-17 * max(abs(total), 1.0):
                negligible += 1
                if negligible >= 2:
                    break
            else:
                negligible = 0
            t += stride
        return total

    h = 1.0
    s = contribution(0.0) + accumulate(h, h)
    prev = h * s * r
    for level in range(1, levels + 1):
        h *= 0.5
        s += accumulate(h, 2 * h)
        total = h * s * r
        calls = fc.calls + (foc.calls if foc is not None else 0)
        if abs(total - prev) < tol * max(1.0, abs(total)):
            return QuadratureResult(float(total), abs(total - prev), calls,
                                    level, True, "tanh_sinh")
        prev = total
    calls = fc.calls + (foc.calls if foc is not None else 0)
    return QuadratureResult(float(prev), None, calls, levels, False, "tanh_sinh")


def double_exponential(f, a=-1.0, b=1.0, **kwargs):
    """Alias of :func:`tanh_sinh` under its other common name."""
    return tanh_sinh(f, a, b, **kwargs)


def singular_endpoint_quadrature(f, a: float, b: float, tol: float = 1e-12):
    """Integrate a function with integrable endpoint singularities.

    Uses tanh-sinh, which is the standard tool for ``1/sqrt(x)``-type behaviour.
    """
    return tanh_sinh(f, a, b, levels=10, tol=tol)


def filon(f, a: float, b: float, omega: float, n: int = 100, kind: str = "sin"):
    """Filon quadrature for oscillatory integrals of ``f(x) sin(wx)`` or ``cos(wx)``.

    Ordinary quadrature needs several points per oscillation, so its cost grows
    linearly in ``w`` and its accuracy collapses when ``w`` is large.  Filon's
    rule interpolates only the *slowly varying* ``f`` and integrates the
    oscillatory factor analytically, so accuracy actually *improves* with
    increasing ``w`` at fixed cost.

    ``n`` must be even (the rule pairs panels like Simpson's).
    """
    if n % 2:
        n += 1
    a, b, omega = float(a), float(b), float(omega)
    h = (b - a) / n
    theta = omega * h
    x = a + h * np.arange(n + 1)
    fc = CountedFunction(f)
    y = np.array([fc(xi) for xi in x], dtype=float)
    if abs(theta) < 1e-4:
        # Series forms: the closed expressions below are 0/0 as theta -> 0.
        alpha = 2.0 * theta**3 / 45.0 - 2.0 * theta**5 / 315.0
        beta = 2.0 / 3.0 + 2.0 * theta**2 / 15.0 - 4.0 * theta**4 / 105.0
        gamma = 4.0 / 3.0 - 2.0 * theta**2 / 15.0 + theta**4 / 210.0
    else:
        st, ct = np.sin(theta), np.cos(theta)
        alpha = (theta**2 + theta * st * ct - 2.0 * st**2) / theta**3
        beta = 2.0 * (theta * (1.0 + ct**2) - 2.0 * st * ct) / theta**3
        gamma = 4.0 * (st - theta * ct) / theta**3
    trig = np.sin if kind == "sin" else np.cos
    g = y * trig(omega * x)
    even = np.sum(g[0:n + 1:2]) - 0.5 * (g[0] + g[n])   # interior even nodes
    odd = np.sum(g[1:n:2])
    if kind == "sin":
        boundary = alpha * (y[0] * np.cos(omega * a) - y[n] * np.cos(omega * b))
    else:
        boundary = alpha * (y[n] * np.sin(omega * b) - y[0] * np.sin(omega * a))
    value = h * (boundary + beta * even + gamma * odd)
    return QuadratureResult(float(value), None, fc.calls, n, True, f"filon_{kind}")


def cauchy_principal_value(f, a: float, b: float, c: float, n: int = 200):
    """Cauchy principal value of ``int_a^b f(x)/(x - c) dx`` with ``a < c < b``.

    The integral diverges on each side of ``c`` and the divergences cancel only
    in the symmetric limit.  Subtracting ``f(c)`` first removes the pole --
    ``(f(x) - f(c))/(x - c)`` is bounded and integrable -- and the remaining
    ``f(c) * int dx/(x-c)`` is done analytically as ``f(c) ln|(b-c)/(c-a)|``.
    Attacking the original integrand numerically, however finely, cannot work:
    the answer depends on an exact cancellation of infinities.
    """
    a, b, c = float(a), float(b), float(c)
    if not a < c < b:
        raise DomainError("the pole must lie strictly inside the interval")
    fc = CountedFunction(f)
    fc_val = float(fc(c))

    def regular(x):
        d = x - c
        if abs(d) < 1e-8:
            # L'Hopital near the pole: the limit is f'(c).
            h = 1e-5
            return (float(fc(c + h)) - float(fc(c - h))) / (2 * h)
        return (float(fc(x)) - fc_val) / d

    left = gauss_legendre(regular, a, c, n).value
    right = gauss_legendre(regular, c, b, n).value
    analytic = fc_val * np.log(abs((b - c) / (c - a)))
    return QuadratureResult(float(left + right + analytic), None, fc.calls, 2,
                            True, "cauchy_principal_value")


def hadamard_finite_part(f, a: float, b: float, c: float, n: int = 200):
    """Hadamard finite part of ``int_a^b f(x)/(x-c)^2 dx``.

    A stronger singularity than the Cauchy case: even the principal value
    diverges, and what survives is the finite part after the divergent term is
    discarded.  Obtained here by subtracting the first two Taylor terms of
    ``f`` at ``c`` and integrating those analytically.
    """
    a, b, c = float(a), float(b), float(c)
    if not a < c < b:
        raise DomainError("the pole must lie strictly inside the interval")
    fc = CountedFunction(f)
    f0 = float(fc(c))
    h = 1e-5
    f1 = (float(fc(c + h)) - float(fc(c - h))) / (2 * h)

    def regular(x):
        d = x - c
        if abs(d) < 1e-6:
            return 0.0
        return (float(fc(x)) - f0 - f1 * d) / (d * d)

    reg = gauss_legendre(regular, a, c, n).value + gauss_legendre(regular, c, b, n).value
    # int (x-c)^-2 -> finite part -1/(b-c) - 1/(c-a); int (x-c)^-1 -> the CPV log.
    sing = f0 * (-1.0 / (b - c) - 1.0 / (c - a)) + f1 * np.log(abs((b - c) / (c - a)))
    return QuadratureResult(float(reg + sing), None, fc.calls, 2, True,
                            "hadamard_finite_part")
