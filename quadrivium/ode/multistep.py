"""Linear multistep methods.

These reuse past solution values instead of recomputing stages, so a ``k``-step
Adams method costs one right-hand side evaluation per step regardless of order.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.types import ODESolution
from ..core.utils import CountedFunction, as_vector

__all__ = [
    "adams_bashforth",
    "adams_moulton",
    "predictor_corrector",
    "abm4",
    "nystrom",
    "milne_simpson",
    "adams_coefficients",
    "variable_step_adams",
]

# Adams-Bashforth (explicit) coefficients, orders 1-6
_AB = {
    1: [1.0],
    2: [3 / 2, -1 / 2],
    3: [23 / 12, -16 / 12, 5 / 12],
    4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
    5: [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720],
    6: [4277 / 1440, -7923 / 1440, 9982 / 1440, -7298 / 1440, 2877 / 1440, -475 / 1440],
}

# Adams-Moulton (implicit) coefficients, orders 1-5
_AM = {
    1: [1.0],
    2: [1 / 2, 1 / 2],
    3: [5 / 12, 8 / 12, -1 / 12],
    4: [9 / 24, 19 / 24, -5 / 24, 1 / 24],
    5: [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720],
}


def adams_coefficients(order: int, kind: str = "bashforth"):
    """Coefficients of the Adams-Bashforth or Adams-Moulton formula."""
    table = _AB if kind == "bashforth" else _AM
    if order not in table:
        raise ValueError(f"order {order} not available for Adams-{kind}")
    return np.array(table[order])


def _bootstrap(f, t0, h, y0, k):
    """Generate the first ``k`` values with RK4, which is self-starting."""
    from .explicit import rk4

    if k <= 1:
        return [as_vector(y0)]
    sol = rk4(f, (t0, t0 + (k - 1) * h), y0, n=(k - 1) * 10)
    return [as_vector(sol(t0 + i * h)) for i in range(k)]


def adams_bashforth(f, t_span, y0, n: int = 100, order: int = 4):
    """Explicit Adams-Bashforth of order 1-6."""
    coef = adams_coefficients(order, "bashforth")
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    hist_y = _bootstrap(f, t0, h, y0, order)
    ys = np.empty((n + 1, hist_y[0].size))
    for i, v in enumerate(hist_y[: min(order, n + 1)]):
        ys[i] = v
    fs = [fc(ts[i], ys[i]) for i in range(min(order, n + 1))]
    for i in range(order - 1, n):
        y_new = ys[i] + h * sum(coef[j] * fs[-1 - j] for j in range(order))
        ys[i + 1] = y_new
        fs.append(fc(ts[i + 1], y_new))
        if len(fs) > order:
            fs.pop(0)
    return ODESolution(ts, ys, f"adams_bashforth{order}", n, n, 0, fc.calls, True,
                       "completed")


def adams_moulton(f, t_span, y0, n: int = 100, order: int = 4, tol: float = 1e-13,
                  max_iter: int = 50):
    """Implicit Adams-Moulton solved by fixed point iteration each step."""
    coef = adams_coefficients(order, "moulton")
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    k = max(order - 1, 1)
    hist = _bootstrap(f, t0, h, y0, k)
    ys = np.empty((n + 1, hist[0].size))
    for i, v in enumerate(hist[: min(k, n + 1)]):
        ys[i] = v
    fs = [fc(ts[i], ys[i]) for i in range(min(k, n + 1))]
    for i in range(k - 1, n):
        past = sum(coef[j] * fs[-j] for j in range(1, order)) if order > 1 else 0.0
        y_guess = ys[i] + h * fc(ts[i], ys[i])
        for _ in range(max_iter):
            y_new = ys[i] + h * (coef[0] * fc(ts[i + 1], y_guess) + past)
            if np.linalg.norm(y_new - y_guess, np.inf) < tol:
                y_guess = y_new
                break
            y_guess = y_new
        ys[i + 1] = y_guess
        fs.append(fc(ts[i + 1], y_guess))
        if len(fs) > order:
            fs.pop(0)
    return ODESolution(ts, ys, f"adams_moulton{order}", n, n, 0, fc.calls, True,
                       "completed")


def predictor_corrector(f, t_span, y0, n: int = 100, order: int = 4,
                        corrections: int = 1):
    """Adams-Bashforth predictor with Adams-Moulton corrector (PECE).

    Gets the accuracy of the implicit formula without solving a nonlinear
    system: the explicit predictor supplies the needed estimate.
    """
    ab = adams_coefficients(order, "bashforth")
    am = adams_coefficients(min(order, 5), "moulton")
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    hist = _bootstrap(f, t0, h, y0, order)
    ys = np.empty((n + 1, hist[0].size))
    for i, v in enumerate(hist[: min(order, n + 1)]):
        ys[i] = v
    fs = [fc(ts[i], ys[i]) for i in range(min(order, n + 1))]
    for i in range(order - 1, n):
        y_p = ys[i] + h * sum(ab[j] * fs[-1 - j] for j in range(order))
        y_c = y_p
        for _ in range(corrections):
            f_new = fc(ts[i + 1], y_c)
            past = sum(am[j] * fs[-j] for j in range(1, len(am)))
            y_c = ys[i] + h * (am[0] * f_new + past)
        ys[i + 1] = y_c
        fs.append(fc(ts[i + 1], y_c))
        if len(fs) > order:
            fs.pop(0)
    return ODESolution(ts, ys, f"predictor_corrector{order}", n, n, 0, fc.calls,
                       True, "completed")


def abm4(f, t_span, y0, n: int = 100):
    """Fourth-order Adams-Bashforth-Moulton predictor-corrector."""
    return predictor_corrector(f, t_span, y0, n, order=4)


def nystrom(f, t_span, y0, n: int = 100):
    """Nystrom's explicit midpoint multistep method (order 2)."""
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    hist = _bootstrap(f, t0, h, y0, 2)
    ys = np.empty((n + 1, hist[0].size))
    ys[0], ys[1] = hist[0], hist[1]
    for i in range(1, n):
        ys[i + 1] = ys[i - 1] + 2 * h * fc(ts[i], ys[i])
    return ODESolution(ts, ys, "nystrom", n, n, 0, fc.calls, True, "completed")


def milne_simpson(f, t_span, y0, n: int = 100):
    """Milne-Simpson predictor-corrector (order 4).

    Historically important, but only weakly stable: the parasitic root of its
    characteristic polynomial sits on the unit circle, so errors can oscillate.
    """
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    hist = _bootstrap(f, t0, h, y0, 4)
    ys = np.empty((n + 1, hist[0].size))
    for i in range(min(4, n + 1)):
        ys[i] = hist[i]
    fs = [fc(ts[i], ys[i]) for i in range(min(4, n + 1))]
    for i in range(3, n):
        y_p = ys[i - 3] + 4 * h / 3 * (2 * fs[-1] - fs[-2] + 2 * fs[-3])
        f_p = fc(ts[i + 1], y_p)
        y_c = ys[i - 1] + h / 3 * (f_p + 4 * fs[-1] + fs[-2])
        ys[i + 1] = y_c
        fs.append(fc(ts[i + 1], y_c))
        if len(fs) > 4:
            fs.pop(0)
    return ODESolution(ts, ys, "milne_simpson", n, n, 0, fc.calls, True, "completed")


def variable_step_adams(f, t_span, y0, rtol: float = 1e-8, atol: float = 1e-10,
                        order: int = 4, h0=None, max_steps: int = 200000):
    """Adams predictor-corrector with adaptive step size.

    The predictor-corrector difference supplies the local error estimate. The
    Adams coefficients assume a constant step, so the history is rebuilt with
    RK4 whenever the step actually changes -- and to keep that from happening
    every step, the size is only revised when the suggested factor leaves
    ``[0.5, 2]``.
    """
    from .explicit import rk4

    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    t0, tf = float(t_span[0]), float(t_span[1])
    y = as_vector(y0).copy()
    h = (tf - t0) / 100.0 if h0 is None else float(h0)
    ts, ys = [t0], [y.copy()]
    t = t0
    ab = adams_coefficients(order, "bashforth")
    am = adams_coefficients(min(order, 5), "moulton")
    hist_f = []            # f values at t, t-h, t-2h, ... (constant h)
    accepted = rejected = 0
    for _ in range(max_steps):
        if t >= tf - 1e-15 * max(1.0, abs(tf)):
            break
        h = min(h, tf - t)
        if len(hist_f) < order:
            # self-starting RK4 while the history is being (re)built
            y_new = rk4(f, (t, t + h), y, n=4).y[-1]
            t += h
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            hist_f.append(fc(t, y))
            accepted += 1
            continue
        y_p = y + h * sum(ab[j] * hist_f[-1 - j] for j in range(order))
        f_p = fc(t + h, y_p)
        past = sum(am[j] * hist_f[-j] for j in range(1, len(am)))
        y_new = y + h * (am[0] * f_p + past)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err = float(np.sqrt(np.mean(((y_new - y_p) / scale) ** 2)))
        if err <= 1.0:
            t += h
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            hist_f.append(fc(t, y))
            if len(hist_f) > order:
                hist_f.pop(0)
            accepted += 1
            fac = 0.9 * err ** (-1.0 / (order + 1)) if err > 0 else 4.0
            if fac > 2.0 or fac < 0.5:
                h *= min(4.0, max(0.2, fac))
                hist_f = []          # step changed: history is no longer valid
        else:
            rejected += 1
            h *= max(0.2, 0.9 * err ** (-1.0 / (order + 1)))
            hist_f = []
    return ODESolution(np.array(ts), np.array(ys), f"variable_adams{order}",
                       accepted + rejected, accepted, rejected, fc.calls, True,
                       "completed")
