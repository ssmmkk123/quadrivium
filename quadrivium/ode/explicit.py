"""Explicit one-step methods for initial value problems ``y' = f(t, y)``.

The Runge-Kutta family, from Euler up to the embedded adaptive pairs. All
solvers accept scalar or vector ``y`` and return an :class:`ODESolution`.
"""

from __future__ import annotations

import numpy as np

from .. import _accel

from ..core.exceptions import StepSizeError
from ..core.types import ODESolution
from ..core.utils import CountedFunction, as_vector

__all__ = [
    "euler",
    "heun",
    "midpoint_method",
    "ralston",
    "rk3",
    "rk4",
    "rk38",
    "rk_general",
    "rkf45",
    "cash_karp",
    "dormand_prince",
    "bogacki_shampine",
    "adaptive_rk",
    "solve_ivp",
    "BUTCHER_TABLEAUX",
]


def _prepare(f, y0):
    """Normalize the right-hand side to act on 1-D arrays."""
    y0 = as_vector(y0)
    scalar = np.ndim(np.asarray(y0)) == 0
    fc = CountedFunction(lambda t, y: as_vector(f(t, y[0] if y.size == 1 and scalar else y)))
    return fc, y0


def _fixed_step(f, t_span, y0, n, stepper, name):
    """Drive a fixed-step one-step method and record slopes for dense output.

    ``k1 = f(t, y)`` is evaluated here and handed to the stepper rather than
    recomputed inside it.  Every explicit Runge-Kutta method needs that value
    as its first stage, so this costs nothing and leaves the slope at each grid
    point available for cubic Hermite interpolation.
    """
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    y0 = as_vector(y0)
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.empty(n + 1)
    ys = np.empty((n + 1, y0.size))
    dys = np.empty((n + 1, y0.size))
    ts[0], ys[0] = t0, y0
    y = y0.copy()
    t = t0
    for i in range(n):
        k1 = fc(t, y)
        dys[i] = k1
        y = stepper(fc, t, y, h, k1)
        t = t0 + (i + 1) * h
        ts[i + 1], ys[i + 1] = t, y
    dys[n] = fc(t, y)
    return ODESolution(ts, ys, name, n, n, 0, fc.calls, True, "completed",
                       None, dys)


def euler(f, t_span, y0, n: int = 100):
    """Forward Euler, order 1. The simplest method and the least accurate."""
    return _fixed_step(f, t_span, y0, n,
                       lambda fc, t, y, h, k1: y + h * k1, "euler")


def heun(f, t_span, y0, n: int = 100):
    """Heun's method (explicit trapezoid), order 2."""
    def step(fc, t, y, h, k1):
        k2 = fc(t + h, y + h * k1)
        return y + h / 2 * (k1 + k2)

    return _fixed_step(f, t_span, y0, n, step, "heun")


def midpoint_method(f, t_span, y0, n: int = 100):
    """Explicit midpoint method, order 2."""
    def step(fc, t, y, h, k1):
        return y + h * fc(t + h / 2, y + h / 2 * k1)

    return _fixed_step(f, t_span, y0, n, step, "midpoint")


def ralston(f, t_span, y0, n: int = 100):
    """Ralston's method: the order-2 RK with minimal truncation error."""
    def step(fc, t, y, h, k1):
        k2 = fc(t + 2 * h / 3, y + 2 * h / 3 * k1)
        return y + h * (k1 + 3 * k2) / 4

    return _fixed_step(f, t_span, y0, n, step, "ralston")


def rk3(f, t_span, y0, n: int = 100):
    """Classical third-order Runge-Kutta."""
    def step(fc, t, y, h, k1):
        k2 = fc(t + h / 2, y + h / 2 * k1)
        k3 = fc(t + h, y - h * k1 + 2 * h * k2)
        return y + h * (k1 + 4 * k2 + k3) / 6

    return _fixed_step(f, t_span, y0, n, step, "rk3")


def rk4(f, t_span, y0, n: int = 100):
    """Classical fourth-order Runge-Kutta -- the default workhorse."""
    def step(fc, t, y, h, k1):
        k2 = fc(t + h / 2, y + h / 2 * k1)
        k3 = fc(t + h / 2, y + h / 2 * k2)
        k4 = fc(t + h, y + h * k3)
        return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return _fixed_step(f, t_span, y0, n, step, "rk4")


def rk38(f, t_span, y0, n: int = 100):
    """The 3/8-rule fourth-order Runge-Kutta."""
    def step(fc, t, y, h, k1):
        k2 = fc(t + h / 3, y + h / 3 * k1)
        k3 = fc(t + 2 * h / 3, y - h / 3 * k1 + h * k2)
        k4 = fc(t + h, y + h * (k1 - k2 + k3))
        return y + h * (k1 + 3 * k2 + 3 * k3 + k4) / 8

    return _fixed_step(f, t_span, y0, n, step, "rk38")


# Butcher tableaux for the embedded adaptive pairs: (c, A, b_high, b_low, order)
BUTCHER_TABLEAUX = {
    "rkf45": (
        np.array([0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2]),
        [
            [],
            [1 / 4],
            [3 / 32, 9 / 32],
            [1932 / 2197, -7200 / 2197, 7296 / 2197],
            [439 / 216, -8.0, 3680 / 513, -845 / 4104],
            [-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40],
        ],
        np.array([16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]),
        np.array([25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0]),
        5,
    ),
    "cash_karp": (
        np.array([0.0, 1 / 5, 3 / 10, 3 / 5, 1.0, 7 / 8]),
        [
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [3 / 10, -9 / 10, 6 / 5],
            [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
            [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
        ],
        np.array([37 / 378, 0.0, 250 / 621, 125 / 594, 0.0, 512 / 1771]),
        np.array([2825 / 27648, 0.0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4]),
        5,
    ),
    "dormand_prince": (
        np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]),
        [
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ],
        np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]),
        np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200,
                  187 / 2100, 1 / 40]),
        5,
    ),
    "bogacki_shampine": (
        np.array([0.0, 1 / 2, 3 / 4, 1.0]),
        [[], [1 / 2], [0.0, 3 / 4], [2 / 9, 1 / 3, 4 / 9]],
        np.array([2 / 9, 1 / 3, 4 / 9, 0.0]),
        np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8]),
        3,
    ),
}


def rk_general(f, t_span, y0, A, b, c, n: int = 100):
    """Run any explicit Runge-Kutta method from its Butcher tableau."""
    A = [np.asarray(row, dtype=float) for row in A]
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    s = len(b)

    def step(fc, t, y, h, k1):
        k = [k1] if c[0] == 0.0 and not len(A[0]) else []
        for i in range(len(k), s):
            yi = y.copy()
            for j in range(len(A[i])):
                yi = yi + h * A[i][j] * k[j]
            k.append(fc(t + c[i] * h, yi))
        return y + h * sum(b[i] * k[i] for i in range(s))

    return _fixed_step(f, t_span, y0, n, step, "rk_general")


def adaptive_rk(f, t_span, y0, tableau: str = "dormand_prince", rtol: float = 1e-8,
                atol: float = 1e-10, h0=None, max_step=np.inf, min_step: float = 1e-14,
                max_steps: int = 1000000, dense_output: bool = False):
    """Embedded Runge-Kutta with PI step size control.

    The two embedded solutions give a local error estimate at no extra cost;
    the step size is then adjusted to keep that estimate at the tolerance.
    """
    c, A, b_hi, b_lo, order = BUTCHER_TABLEAUX[tableau]
    A = [np.asarray(r, dtype=float) for r in A]
    s = len(b_hi)
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])

    fast = _accel.kernel("adaptive_rk")
    if fast is not None:
        # The stage assembly and the controller move to the compiled kernel;
        # ``f`` stays a Python callable and is called back per stage.
        a_flat = np.zeros((s, s))
        for i, row in enumerate(A):
            a_flat[i, : row.size] = row
        try:
            ts, ys, dys, accepted, rejected, calls = fast(
                fc, np.asarray(c, dtype=float), a_flat.ravel(),
                np.asarray(b_hi, dtype=float), np.asarray(b_lo, dtype=float),
                float(order), t0, tf, y, float(rtol), float(atol),
                None if h0 is None else float(h0), float(max_step),
                float(min_step), int(max_steps),
            )
        except RuntimeError as exc:
            if not str(exc).startswith("stepsize:"):
                raise
            raise StepSizeError(str(exc)[len("stepsize:"):]) from None
        interp = None
        if dense_output:
            from ..interpolate.spline import pchip

            splines = [pchip(ts, ys[:, j]) for j in range(ys.shape[1])]
            interp = lambda q: np.column_stack([sp(q) for sp in splines])
        return ODESolution(ts, ys, tableau, accepted + rejected, accepted,
                           rejected, calls, True, "completed", interp, dys)

    direction = 1.0 if tf >= t0 else -1.0
    t = t0
    h = (abs(tf - t0) / 100.0 if h0 is None else abs(h0)) * direction
    ts, ys, dys = [t], [y.copy()], []
    accepted = rejected = 0
    err_prev = 1.0
    for _ in range(max_steps):
        if (t - tf) * direction >= 0:
            break
        if abs(h) > abs(tf - t):
            h = tf - t
        k = []
        for i in range(s):
            yi = y.copy()
            for j in range(len(A[i])):
                if A[i][j] != 0.0:
                    yi = yi + h * A[i][j] * k[j]
            k.append(fc(t + c[i] * h, yi))
        y_hi = y + h * sum(b_hi[i] * k[i] for i in range(s))
        y_lo = y + h * sum(b_lo[i] * k[i] for i in range(s))
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_hi))
        err = float(np.sqrt(np.mean(((y_hi - y_lo) / scale) ** 2)))
        if err <= 1.0 or abs(h) <= min_step:
            # k[0] is f at the *start* of this step, so it is the slope
            # belonging to the point already in ``ys``.  Recording it costs
            # nothing and lifts dense output from linear to cubic Hermite.
            dys.append(k[0].copy())
            t = t + h
            y = y_hi
            ts.append(t)
            ys.append(y.copy())
            accepted += 1
            # PI controller: smoother than the pure elementary rule
            fac = 0.9 * err ** (-0.7 / order) * err_prev ** (0.4 / order) if err > 0 else 5.0
            err_prev = max(err, 1e-4)
        else:
            rejected += 1
            fac = 0.9 * err ** (-1.0 / order)
        h = h * min(5.0, max(0.2, fac))
        if abs(h) > max_step:
            h = max_step * direction
        if abs(h) < min_step:
            raise StepSizeError(
                f"step size underflow at t={t:.6g}: required h < {min_step:.2e}; "
                "the problem is likely stiff -- try an implicit solver"
            )
    dys.append(fc(t, y))  # slope at the final point (one extra evaluation)
    interp = None
    if dense_output:
        from ..interpolate.spline import pchip

        arr_t = np.array(ts)
        splines = [pchip(arr_t, np.array(ys)[:, j]) for j in range(y.size)]
        interp = lambda q: np.column_stack([sp(q) for sp in splines])
    return ODESolution(np.array(ts), np.array(ys), tableau, accepted + rejected,
                       accepted, rejected, fc.calls, True, "completed", interp,
                       np.array(dys))


def rkf45(f, t_span, y0, **kwargs):
    """Runge-Kutta-Fehlberg 4(5) with adaptive step size."""
    return adaptive_rk(f, t_span, y0, "rkf45", **kwargs)


def cash_karp(f, t_span, y0, **kwargs):
    """Cash-Karp 4(5) embedded pair."""
    return adaptive_rk(f, t_span, y0, "cash_karp", **kwargs)


def dormand_prince(f, t_span, y0, **kwargs):
    """Dormand-Prince 5(4) -- the method behind most ``ode45`` implementations."""
    return adaptive_rk(f, t_span, y0, "dormand_prince", **kwargs)


def bogacki_shampine(f, t_span, y0, **kwargs):
    """Bogacki-Shampine 3(2) embedded pair, good at loose tolerances."""
    return adaptive_rk(f, t_span, y0, "bogacki_shampine", **kwargs)


def solve_ivp(f, t_span, y0, method: str = "dormand_prince", **kwargs):
    """Solve an initial value problem with the requested method.

    Fixed-step methods take ``n``; adaptive ones take ``rtol``/``atol``.
    """
    fixed = {"euler": euler, "heun": heun, "midpoint": midpoint_method,
             "ralston": ralston, "rk3": rk3, "rk4": rk4, "rk38": rk38}
    if method in fixed:
        return fixed[method](f, t_span, y0, kwargs.pop("n", 100))
    if method in BUTCHER_TABLEAUX:
        return adaptive_rk(f, t_span, y0, method, **kwargs)
    from .implicit import backward_euler, bdf, implicit_midpoint, radau_iia, trapezoidal
    from .multistep import adams_bashforth, adams_moulton, predictor_corrector

    others = {
        "backward_euler": backward_euler, "trapezoidal": trapezoidal,
        "implicit_midpoint": implicit_midpoint, "radau": radau_iia, "bdf": bdf,
        "adams_bashforth": adams_bashforth, "adams_moulton": adams_moulton,
        "predictor_corrector": predictor_corrector,
    }
    if method in others:
        return others[method](f, t_span, y0, **kwargs)
    raise ValueError(f"unknown method {method!r}")
