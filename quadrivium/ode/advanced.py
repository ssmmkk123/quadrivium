"""Extrapolation methods, event location, second-order systems, DAEs and DDEs.

The parts of initial-value problem solving that sit outside the plain
``y' = f(t, y)`` picture: getting arbitrary order from a low-order scheme by
extrapolation, stopping exactly when a condition is met, exploiting the
structure of ``y'' = f(t, y, y')``, and handling algebraic constraints or
retarded arguments.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.exceptions import ConvergenceError, StepSizeError
from ..core.types import ODESolution
from ..core.utils import CountedFunction, as_vector, numerical_jacobian

__all__ = [
    "modified_midpoint",
    "bulirsch_stoer",
    "gragg_bulirsch_stoer",
    "richardson_ode",
    "solve_ivp_events",
    "find_events",
    "rk_nystrom",
    "runge_kutta_nystrom",
    "stormer_cowell",
    "dae_index1_bdf",
    "mass_matrix_ode",
    "dde_method_of_steps",
    "stiffness_ratio",
    "detect_stiffness",
]


def modified_midpoint(f, t, y, h, n: int, fc=None):
    """Gragg's modified midpoint rule: ``n`` substeps across one interval ``h``.

    The final smoothing step is what makes this special.  It cancels the
    oscillatory parasitic mode of the leapfrog recursion and leaves an error
    expansion in *even* powers of ``h`` alone -- so each extrapolation stage
    gains two orders instead of one, which is what makes Bulirsch-Stoer
    efficient.
    """
    fc = fc if fc is not None else (lambda tt, yy: as_vector(f(tt, yy)))
    y = as_vector(y).astype(float)
    hs = h / n
    y0 = y.copy()
    y1 = y0 + hs * fc(t, y0)
    for k in range(1, n):
        y0, y1 = y1, y0 + 2.0 * hs * fc(t + k * hs, y1)
    return 0.5 * (y1 + y0 + hs * fc(t + h, y1))


def gragg_bulirsch_stoer(f, t_span, y0, rtol: float = 1e-10, atol: float = 1e-12,
                         h0=None, max_order: int = 8, max_steps: int = 100000,
                         min_step: float = 1e-14):
    """Gragg-Bulirsch-Stoer: adaptive order *and* step size by extrapolation.

    Repeats the modified midpoint rule with more and more substeps and
    extrapolates the results to zero step size.  Because the error expansion
    has only even powers, the ``k``-th column of the tableau is accurate to
    order ``2k`` -- so on a smooth problem this reaches tolerances that would
    need absurdly small steps from a fixed-order method.  On a non-smooth one
    it is the wrong tool, since extrapolation assumes the expansion exists.
    """
    fc = CountedFunction(lambda tt, yy: as_vector(f(tt, yy)))
    y = as_vector(y0).astype(float)
    t0, tf = float(t_span[0]), float(t_span[1])
    direction = 1.0 if tf >= t0 else -1.0
    t = t0
    h = (abs(tf - t0) / 20.0 if h0 is None else abs(h0)) * direction
    steps = np.array([2 * (k + 1) for k in range(max_order + 1)])
    ts, ys, dys = [t], [y.copy()], []
    accepted = rejected = 0
    for _ in range(max_steps):
        if (t - tf) * direction >= 0:
            break
        if abs(h) > abs(tf - t):
            h = tf - t
        table = []
        accept = False
        for k in range(max_order + 1):
            table.append(modified_midpoint(f, t, y, h, int(steps[k]), fc))
            # Aitken-Neville extrapolation in h^2.
            for j in range(k - 1, -1, -1):
                ratio = (steps[k] / steps[j]) ** 2
                table[j] = table[j + 1] + (table[j + 1] - table[j]) / (ratio - 1.0)
            if k >= 1:
                scale = atol + rtol * np.maximum(np.abs(y), np.abs(table[0]))
                err = float(np.sqrt(np.mean(((table[0] - table[1]) / scale) ** 2)))
                if err <= 1.0:
                    accept = True
                    order = 2 * k
                    break
        if accept:
            dys.append(fc(t, y))
            t = t + h
            y = table[0]
            ts.append(t)
            ys.append(y.copy())
            accepted += 1
            fac = 0.94 * (1.0 / max(err, 1e-12)) ** (1.0 / max(order, 1))
            h = h * min(4.0, max(0.2, fac))
        else:
            rejected += 1
            h = 0.5 * h
        if abs(h) < min_step:
            raise StepSizeError(
                f"step size underflow at t={t:.6g}; extrapolation needs a "
                "smooth right-hand side -- try an implicit solver if stiff"
            )
    dys.append(fc(t, y))
    return ODESolution(np.array(ts), np.array(ys), "gragg_bulirsch_stoer",
                       accepted + rejected, accepted, rejected, fc.calls, True,
                       "completed", None, np.array(dys))


def bulirsch_stoer(f, t_span, y0, **kwargs):
    """Alias for :func:`gragg_bulirsch_stoer`."""
    return gragg_bulirsch_stoer(f, t_span, y0, **kwargs)


def richardson_ode(method, f, t_span, y0, order: int, levels: int = 4, n0: int = 16,
                   **kwargs):
    """Richardson-extrapolate any fixed-step solver to higher order.

    Runs ``method`` with ``n0, 2 n0, 4 n0, ...`` steps and eliminates the
    leading error terms.  Returns ``(value, table)``; the table's diagonal
    shows the order climbing one step per column.
    """
    vals = []
    for j in range(levels):
        sol = method(f, t_span, y0, n=n0 * 2**j, **kwargs)
        vals.append(np.asarray(sol.y[-1], dtype=float))
    table = [vals]
    for j in range(1, levels):
        row = []
        for i in range(levels - j):
            factor = 2.0 ** (order + j - 1)
            row.append((factor * table[j - 1][i + 1] - table[j - 1][i]) / (factor - 1.0))
        table.append(row)
    return table[-1][0], table


# --------------------------------------------------------------------------
# Event location
# --------------------------------------------------------------------------
def find_events(sol, event, tol: float = 1e-12, max_iter: int = 100,
                direction: int = 0):
    """Locate the roots of ``event(t, y)`` along an existing solution.

    Sign changes are found on the stored grid and then refined by bisection
    *against the solution's own interpolant*, so the located time is as
    accurate as the solver -- not as accurate as the output grid.

    ``direction`` selects which crossings count: ``+1`` increasing, ``-1``
    decreasing, ``0`` both.  Returns ``(t_events, y_events)``.
    """
    t = np.asarray(sol.t, dtype=float)
    g = np.array([float(event(t[i], sol.y[i])) for i in range(t.size)])
    t_ev, y_ev = [], []
    for i in range(t.size - 1):
        a, b = g[i], g[i + 1]
        if a == 0.0:
            t_ev.append(t[i])
            y_ev.append(np.asarray(sol.y[i]))
            continue
        if a * b > 0.0 or b == 0.0:
            continue
        if direction > 0 and a > b:
            continue
        if direction < 0 and a < b:
            continue
        lo, hi, glo = t[i], t[i + 1], a
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            gm = float(event(mid, sol(mid)))
            if gm == 0.0 or 0.5 * abs(hi - lo) < tol:
                break
            if glo * gm < 0.0:
                hi = mid
            else:
                lo, glo = mid, gm
        te = 0.5 * (lo + hi)
        t_ev.append(te)
        y_ev.append(np.asarray(sol(te)))
    return np.array(t_ev), (np.array(y_ev) if y_ev else np.empty((0, sol.y.shape[1])))


def solve_ivp_events(f, t_span, y0, events=None, terminal=None, solver=None,
                     **kwargs):
    """Integrate with event detection, optionally stopping at a terminal event.

    ``events`` is a callable or list of callables ``g(t, y)``; a root of any of
    them is an event.  ``terminal`` marks which of them stop the integration
    (a bool or list of bools).  The integration is *not* restarted at the
    event -- the solution is truncated there and the event time located on the
    dense output, which is exact to the solver's own accuracy.

    Returns ``(solution, t_events, y_events)``.
    """
    from .explicit import solve_ivp

    solver = solve_ivp if solver is None else solver
    if events is None:
        sol = solver(f, t_span, y0, **kwargs)
        return sol, [], []
    if callable(events):
        events = [events]
    terminal = [False] * len(events) if terminal is None else (
        [terminal] * len(events) if isinstance(terminal, bool) else list(terminal))
    sol = solver(f, t_span, y0, **kwargs)
    all_t, all_y = [], []
    stop_at = None
    for j, ev in enumerate(events):
        te, ye = find_events(sol, ev)
        all_t.append(te)
        all_y.append(ye)
        if terminal[j] and te.size:
            first = te[0]
            stop_at = first if stop_at is None else min(stop_at, first)
    if stop_at is not None:
        keep = np.asarray(sol.t) <= stop_at
        t_new = np.append(np.asarray(sol.t)[keep], stop_at)
        y_new = np.vstack([np.asarray(sol.y)[keep], np.atleast_2d(sol(stop_at))])
        d_new = None
        if sol.dydt is not None:
            d_new = np.vstack([np.asarray(sol.dydt)[keep],
                               np.atleast_2d(as_vector(f(stop_at, sol(stop_at))))])
        sol = ODESolution(t_new, y_new, sol.method + "+events", sol.n_steps,
                          sol.n_accepted, sol.n_rejected, sol.n_rhs_evals, True,
                          f"terminated at event t={stop_at:.10g}", None, d_new)
        all_t = [te[te <= stop_at + 1e-12] for te in all_t]
        all_y = [ye[: t.size] for t, ye in zip(all_t, all_y)]
    return sol, all_t, all_y


# --------------------------------------------------------------------------
# Second-order systems
# --------------------------------------------------------------------------
def rk_nystrom(f, t_span, y0, dy0, n: int = 100):
    """Runge-Kutta-Nystrom for ``y'' = f(t, y, y')``, order 4.

    Integrates the second-order equation directly rather than converting it to
    a first-order system of twice the size.  For the special case ``y'' = f(t, y)``
    -- no velocity dependence -- this needs three function evaluations per step
    where the converted RK4 needs four, and its position error constant is
    smaller.
    """
    fc = CountedFunction(lambda t, y, dy: as_vector(f(t, y, dy)))
    y = as_vector(y0).astype(float)
    dy = as_vector(dy0).astype(float)
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    Y = np.empty((n + 1, y.size))
    DY = np.empty((n + 1, y.size))
    Y[0], DY[0] = y, dy
    for i in range(n):
        t = ts[i]
        k1 = 0.5 * h * fc(t, y, dy)
        q = 0.5 * h * (dy + 0.5 * k1)
        k2 = 0.5 * h * fc(t + 0.5 * h, y + q, dy + k1)
        k3 = 0.5 * h * fc(t + 0.5 * h, y + q, dy + k2)
        r = h * (dy + k3)
        k4 = 0.5 * h * fc(t + h, y + r, dy + 2.0 * k3)
        y = y + h * (dy + (k1 + k2 + k3) / 3.0)
        dy = dy + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 3.0
        Y[i + 1], DY[i + 1] = y, dy
    sol = ODESolution(ts, Y, "rk_nystrom", n, n, 0, fc.calls, True, "completed",
                      None, DY)
    sol.velocity = DY
    return sol


def runge_kutta_nystrom(f, t_span, y0, dy0, n: int = 100):
    """Alias for :func:`rk_nystrom`."""
    return rk_nystrom(f, t_span, y0, dy0, n)


def stormer_cowell(f, t_span, y0, dy0, n: int = 100):
    """Stormer-Cowell multistep for ``y'' = f(t, y)`` (no velocity dependence).

    Uses the four-step coefficients ``(14, -5, 4, -1)/12``.  The familiar
    three-step form ``(13, -2, 1)/12`` leaves an ``h^3`` term in its Taylor
    expansion and is only *third* order globally, despite often being quoted
    as fourth; the four-step coefficients are chosen precisely so that term
    cancels.

    Acting on positions only makes it very cheap -- one force evaluation per
    step -- but also means the velocity it reports is a difference estimate
    rather than an integrated quantity.
    """
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    y0 = as_vector(y0).astype(float)
    dy0 = as_vector(dy0).astype(float)
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    Y = np.empty((n + 1, y0.size))
    Y[0] = y0
    # Bootstrap with Runge-Kutta-Nystrom rather than a low-order Taylor step.
    # A second-order start caps the *global* order of the whole run at three,
    # however accurate the multistep formula that follows it is.
    n_start = min(4, n)
    start = rk_nystrom(lambda t, yy, dyy: f(t, yy), (t0, t0 + n_start * h),
                       y0, dy0, n=n_start)
    Y[: n_start + 1] = np.asarray(start.y)
    accel = [fc(ts[k], Y[k]) for k in range(n_start + 1)]
    for i in range(n_start + 1, n + 1):
        a1, a2, a3, a4 = accel[-1], accel[-2], accel[-3], accel[-4]
        Y[i] = (2 * Y[i - 1] - Y[i - 2]
                + h * h * (14 * a1 - 5 * a2 + 4 * a3 - a4) / 12.0)
        accel.append(fc(ts[i], Y[i]))
    DY = np.gradient(Y, h, axis=0)
    sol = ODESolution(ts, Y, "stormer_cowell", n, n, 0, fc.calls, True,
                      "completed", None, DY)
    sol.velocity = DY
    return sol


# --------------------------------------------------------------------------
# Differential-algebraic and delay equations
# --------------------------------------------------------------------------
def dae_index1_bdf(f, g, t_span, y0, z0, n: int = 200, order: int = 2,
                   tol: float = 1e-12, max_iter: int = 60):
    """Semi-explicit index-1 DAE ``y' = f(t, y, z)``, ``0 = g(t, y, z)``.

    Index 1 means ``dg/dz`` is nonsingular, so the algebraic variables are
    locally determined by the differential ones.  BDF discretizes ``y'`` and
    the coupled nonlinear system is solved by Newton at each step -- the
    algebraic constraint is enforced *exactly* at every step rather than
    integrated, which is what distinguishes a DAE solver from applying an ODE
    solver to a stiff approximation.

    Returns an :class:`ODESolution` for ``y`` with the algebraic trajectory on
    its ``.z`` attribute.
    """
    BDF = {1: ([1.0], 1.0), 2: ([4 / 3, -1 / 3], 2 / 3),
           3: ([18 / 11, -9 / 11, 2 / 11], 6 / 11),
           4: ([48 / 25, -36 / 25, 16 / 25, -3 / 25], 12 / 25)}
    if order not in BDF:
        raise ValueError("order must be 1..4")
    alpha, beta = BDF[order]
    y = as_vector(y0).astype(float)
    z = as_vector(z0).astype(float)
    ny, nz = y.size, z.size
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    Y = np.empty((n + 1, ny))
    Z = np.empty((n + 1, nz))
    Y[0], Z[0] = y, z
    hist = [y.copy()]
    for i in range(n):
        tn = ts[i + 1]
        k = min(len(hist), order)
        a, b = BDF[k]
        past = sum(a[j] * hist[-1 - j] for j in range(k))

        def residual(w):
            yy, zz = w[:ny], w[ny:]
            return np.concatenate([yy - past - b * h * as_vector(f(tn, yy, zz)),
                                   as_vector(g(tn, yy, zz))])

        w = np.concatenate([y, z])
        for _ in range(max_iter):
            r = residual(w)
            if np.max(np.abs(r)) < tol:
                break
            J = numerical_jacobian(residual, w)
            try:
                step = np.linalg.solve(J, r)
            except np.linalg.LinAlgError as exc:
                raise ConvergenceError(
                    "DAE Newton step failed: the pencil is singular, which "
                    "usually means the problem is not index 1 (dg/dz singular)"
                ) from exc
            w = w - step
        else:
            raise ConvergenceError(f"DAE Newton did not converge at t={tn:.6g}")
        y, z = w[:ny], w[ny:]
        Y[i + 1], Z[i + 1] = y, z
        hist.append(y.copy())
        if len(hist) > order:
            hist.pop(0)
    sol = ODESolution(ts, Y, f"dae_index1_bdf{order}", n, n, 0, 0, True, "completed")
    sol.z = Z
    return sol


def mass_matrix_ode(M, f, t_span, y0, n: int = 200, theta: float = 0.5):
    """Solve ``M y' = f(t, y)`` for a constant (possibly singular) ``M``.

    A singular ``M`` makes this a DAE rather than an ODE, which is why the
    theta method is applied to the whole pencil ``M - theta h J`` instead of
    inverting ``M`` first: with ``M`` singular that inverse does not exist,
    but the pencil is still nonsingular for small ``h``.
    """
    M = np.atleast_2d(np.asarray(M, dtype=float))
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    y = as_vector(y0).astype(float)
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    Y = np.empty((n + 1, y.size))
    Y[0] = y
    for i in range(n):
        t, tn = ts[i], ts[i + 1]
        rhs_old = fc(t, y)

        def residual(w, tn=tn, y=y, rhs_old=rhs_old):
            return M @ (w - y) - h * (theta * fc(tn, w) + (1 - theta) * rhs_old)

        w = y.copy()
        for _ in range(60):
            r = residual(w)
            if np.max(np.abs(r)) < 1e-12:
                break
            J = numerical_jacobian(residual, w)
            w = w - np.linalg.solve(J, r)
        y = w
        Y[i + 1] = y
    return ODESolution(ts, Y, "mass_matrix_ode", n, n, 0, fc.calls, True, "completed")


def dde_method_of_steps(f, history, delays, t_span, n: int = 400, solver=None,
                        max_step=None, **kwargs):
    """Delay differential equations by the method of steps.

    ``f(t, y, ylags)`` receives the delayed states, ``history(t)`` supplies
    ``y`` for ``t <= t0``, and ``delays`` is the list of lags.  The interval is
    split at multiples of the smallest lag, and on each piece the equation is
    an ODE in terms of the already-computed past -- which is the whole idea of
    the method.

    Solutions of DDEs have derivative discontinuities that propagate from the
    initial point and smooth out one order per lag; stepping across those
    breakpoints rather than landing on them is what wrecks a naive
    integration, so they are used as segment boundaries here.

    Accuracy is capped by the *dense output* of the underlying solver, not by
    its tolerance: the delayed value is read from an interpolant, so a cubic
    Hermite interpolant limits the whole solution to O(h^4) no matter how
    tightly each segment is integrated.  ``max_step`` therefore defaults to
    one twentieth of the shortest lag, which keeps h small enough for that to
    stop mattering.
    """
    from .explicit import solve_ivp

    solver = solve_ivp if solver is None else solver
    delays = [float(d) for d in np.atleast_1d(delays)]
    if min(delays) <= 0:
        raise ValueError("delays must be positive")
    t0, tf = float(t_span[0]), float(t_span[1])
    tau = min(delays)
    kwargs.setdefault("max_step", tau / 20.0 if max_step is None else max_step)
    n_seg = max(1, int(np.ceil((tf - t0) / tau)))
    segments = [(t0 + k * tau, min(t0 + (k + 1) * tau, tf)) for k in range(n_seg)]
    all_t = [np.array([t0])]
    all_y = [np.atleast_2d(as_vector(history(t0)))]
    done = []          # finished segments, kept for their dense output

    def past(t):
        """Retarded state, read off the solver's own interpolant.

        Interpolating the stored grid linearly instead would cap the whole
        integration at second order however tightly each segment was
        solved: the delayed value feeds straight back into the right-hand
        side, so its interpolation error *is* the method error.
        """
        if t <= t0:
            return as_vector(history(t))
        for seg in done:
            if seg.t[0] <= t <= seg.t[-1]:
                return as_vector(seg(t))
        return as_vector(done[-1](done[-1].t[-1])) if done else all_y[0][0]

    y = as_vector(history(t0))
    for a, b in segments:
        if b <= a:
            continue
        rhs = lambda t, yy: as_vector(f(t, yy, [past(t - d) for d in delays]))
        sol = solver(rhs, (a, b), y, **kwargs)
        done.append(sol)
        all_t.append(np.asarray(sol.t)[1:])
        all_y.append(np.asarray(sol.y)[1:])
        y = np.asarray(sol.y)[-1]
    return ODESolution(np.concatenate(all_t), np.vstack(all_y), "dde_method_of_steps",
                       len(segments), len(segments), 0, 0, True, "completed")


def stiffness_ratio(jac, t, y):
    """Ratio of largest to smallest eigenvalue modulus of the Jacobian.

    A large ratio is the classic signature of stiffness: the fastest mode sets
    the step an explicit method may take, while the slowest sets how long the
    integration must run.
    """
    J = np.atleast_2d(np.asarray(jac(t, y), dtype=float))
    ev = np.abs(np.linalg.eigvals(J))
    ev = ev[ev > 1e-300]
    if ev.size == 0:
        return 1.0
    return float(np.max(ev) / np.min(ev))


def detect_stiffness(f, t, y, threshold: float = 1e3):
    """Estimate stiffness from a numerical Jacobian at one point.

    Returns ``(is_stiff, ratio)``.  Only a local indicator -- a problem can be
    stiff over part of its trajectory and not elsewhere -- but it answers the
    practical question of whether an explicit solver is about to crawl.
    """
    J = numerical_jacobian(lambda yy: as_vector(f(t, yy)), as_vector(y))
    ratio = stiffness_ratio(lambda *_: J, t, y)
    return bool(ratio > threshold), ratio
