"""Implicit methods for stiff initial value problems.

Stiff problems force explicit solvers into vanishingly small steps for
stability rather than accuracy; the A-stable methods here have no such limit.
Each stage solves a nonlinear system by (damped) Newton iteration.
"""

from __future__ import annotations

import numpy as np

from ..core.types import ODESolution
from ..core.utils import CountedFunction, as_vector, numerical_jacobian

__all__ = [
    "backward_euler",
    "trapezoidal",
    "crank_nicolson_ode",
    "implicit_midpoint",
    "theta_method",
    "gauss_legendre_irk",
    "radau_iia",
    "lobatto_iiic",
    "sdirk",
    "bdf",
    "rosenbrock",
    "esdirk",
    "tr_bdf2",
]


def _newton_solve(G, y_guess, jac=None, tol=1e-12, max_iter=50):
    """Damped Newton solve for one implicit stage."""
    y = as_vector(y_guess).copy()
    for _ in range(max_iter):
        r = as_vector(G(y))
        if np.linalg.norm(r, np.inf) < tol:
            return y
        J = jac(y) if jac is not None else numerical_jacobian(G, y)
        try:
            s = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            s = np.linalg.lstsq(J, -r, rcond=None)[0]
        lam = 1.0
        nr = np.linalg.norm(r)
        for _ in range(20):  # backtracking keeps the step from overshooting
            if np.linalg.norm(as_vector(G(y + lam * s))) < nr or lam < 1e-6:
                break
            lam *= 0.5
        y = y + lam * s
    return y


def _implicit_driver(f, t_span, y0, n, stage, name, jac=None):
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.empty(n + 1)
    ys = np.empty((n + 1, y.size))
    ts[0], ys[0] = t0, y
    t = t0
    for i in range(n):
        y = stage(fc, t, y, h, jac)
        t = t0 + (i + 1) * h
        ts[i + 1], ys[i + 1] = t, y
    return ODESolution(ts, ys, name, n, n, 0, fc.calls, True, "completed")


def backward_euler(f, t_span, y0, n: int = 100, jac=None):
    """Backward (implicit) Euler: order 1, A-stable and L-stable."""
    def stage(fc, t, y, h, jac):
        return _newton_solve(lambda z: z - y - h * fc(t + h, z), y + h * fc(t, y))

    return _implicit_driver(f, t_span, y0, n, stage, "backward_euler", jac)


def trapezoidal(f, t_span, y0, n: int = 100, jac=None):
    """Implicit trapezoid (Crank-Nicolson): order 2, A-stable."""
    def stage(fc, t, y, h, jac):
        fy = fc(t, y)
        return _newton_solve(lambda z: z - y - h / 2 * (fy + fc(t + h, z)),
                             y + h * fy)

    return _implicit_driver(f, t_span, y0, n, stage, "trapezoidal", jac)


def crank_nicolson_ode(f, t_span, y0, n: int = 100, **kw):
    """Alias of :func:`trapezoidal` under the Crank-Nicolson name."""
    return trapezoidal(f, t_span, y0, n, **kw)


def implicit_midpoint(f, t_span, y0, n: int = 100, jac=None):
    """Implicit midpoint rule: order 2, A-stable and symplectic."""
    def stage(fc, t, y, h, jac):
        def G(z):
            return z - y - h * fc(t + h / 2, 0.5 * (y + z))

        return _newton_solve(G, y + h * fc(t, y))

    return _implicit_driver(f, t_span, y0, n, stage, "implicit_midpoint", jac)


def theta_method(f, t_span, y0, n: int = 100, theta: float = 0.5, jac=None):
    """Theta method: Euler (0), trapezoid (1/2), backward Euler (1)."""
    def stage(fc, t, y, h, jac):
        fy = fc(t, y)
        if theta == 0.0:
            return y + h * fy
        return _newton_solve(
            lambda z: z - y - h * ((1 - theta) * fy + theta * fc(t + h, z)),
            y + h * fy)

    return _implicit_driver(f, t_span, y0, n, stage, f"theta_{theta}", jac)


def _irk_stage_solver(fc, t, y, h, A, b, c, tol=1e-12, max_iter=60):
    """Solve the coupled stage equations of a fully implicit RK method."""
    s = len(b)
    m = y.size
    K = np.tile(fc(t, y), (s, 1))
    # The stage combinations are the products A K and b K. Writing them as
    # Python sums over the stage index costs s temporaries per stage and runs
    # inside the Newton loop, which is the hot path of every fully implicit
    # step; as array products the whole stage coupling is one call.
    A_mat = np.asarray(A, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    t_stage = t + np.asarray(c, dtype=float) * h

    def residual(Kflat):
        K = Kflat.reshape(s, m)
        Y = y + h * (A_mat @ K)
        R = np.empty_like(K)
        for i in range(s):
            R[i] = K[i] - fc(t_stage[i], Y[i])
        return R.ravel()

    Kflat = _newton_solve(residual, K.ravel(), tol=tol, max_iter=max_iter)
    K = Kflat.reshape(s, m)
    return y + h * (b_vec @ K)


def gauss_legendre_irk(f, t_span, y0, n: int = 100, stages: int = 2, jac=None):
    """Gauss-Legendre implicit RK: order ``2s``, A-stable and symplectic.

    The highest order attainable for ``s`` stages.
    """
    if stages == 1:
        A = [[0.5]]
        b = [1.0]
        c = [0.5]
    elif stages == 2:
        r3 = np.sqrt(3.0)
        A = [[0.25, 0.25 - r3 / 6], [0.25 + r3 / 6, 0.25]]
        b = [0.5, 0.5]
        c = [0.5 - r3 / 6, 0.5 + r3 / 6]
    elif stages == 3:
        r15 = np.sqrt(15.0)
        A = [[5 / 36, 2 / 9 - r15 / 15, 5 / 36 - r15 / 30],
             [5 / 36 + r15 / 24, 2 / 9, 5 / 36 - r15 / 24],
             [5 / 36 + r15 / 30, 2 / 9 + r15 / 15, 5 / 36]]
        b = [5 / 18, 4 / 9, 5 / 18]
        c = [0.5 - r15 / 10, 0.5, 0.5 + r15 / 10]
    else:
        raise ValueError("stages must be 1, 2 or 3")

    def stage(fc, t, y, h, jac):
        return _irk_stage_solver(fc, t, y, h, A, b, c)

    return _implicit_driver(f, t_span, y0, n, stage, f"gauss_irk{stages}", jac)


def radau_iia(f, t_span, y0, n: int = 100, stages: int = 3, jac=None):
    """Radau IIA: order ``2s-1``, L-stable -- the standard choice for stiff problems."""
    if stages == 2:
        A = [[5 / 12, -1 / 12], [3 / 4, 1 / 4]]
        b = [3 / 4, 1 / 4]
        c = [1 / 3, 1.0]
    elif stages == 3:
        s6 = np.sqrt(6.0)
        A = [[(88 - 7 * s6) / 360, (296 - 169 * s6) / 1800, (-2 + 3 * s6) / 225],
             [(296 + 169 * s6) / 1800, (88 + 7 * s6) / 360, (-2 - 3 * s6) / 225],
             [(16 - s6) / 36, (16 + s6) / 36, 1 / 9]]
        b = [(16 - s6) / 36, (16 + s6) / 36, 1 / 9]
        c = [(4 - s6) / 10, (4 + s6) / 10, 1.0]
    else:
        raise ValueError("stages must be 2 or 3")

    def stage(fc, t, y, h, jac):
        return _irk_stage_solver(fc, t, y, h, A, b, c)

    return _implicit_driver(f, t_span, y0, n, stage, f"radau_iia{stages}", jac)


def lobatto_iiic(f, t_span, y0, n: int = 100, jac=None):
    """Lobatto IIIC (3 stages, order 4): L-stable, strongly damping."""
    A = [[1 / 6, -1 / 3, 1 / 6], [1 / 6, 5 / 12, -1 / 12], [1 / 6, 2 / 3, 1 / 6]]
    b = [1 / 6, 2 / 3, 1 / 6]
    c = [0.0, 0.5, 1.0]

    def stage(fc, t, y, h, jac):
        return _irk_stage_solver(fc, t, y, h, A, b, c)

    return _implicit_driver(f, t_span, y0, n, stage, "lobatto_iiic", jac)


def sdirk(f, t_span, y0, n: int = 100, jac=None):
    """Singly diagonally implicit RK (2 stages, order 3, A-stable).

    Each stage is solved on its own, so the cost is far below a fully implicit
    method of the same order.
    """
    gamma = (3.0 + np.sqrt(3.0)) / 6.0
    c = [gamma, 1.0 - gamma]
    b = [0.5, 0.5]
    A = [[gamma, 0.0], [1.0 - 2 * gamma, gamma]]

    def stage(fc, t, y, h, jac):
        K = []
        for i in range(2):
            base = y + h * sum(A[i][j] * K[j] for j in range(i))

            def G(k, base=base, i=i):
                return k - fc(t + c[i] * h, base + h * A[i][i] * k)

            K.append(_newton_solve(G, fc(t, y)))
        return y + h * (b[0] * K[0] + b[1] * K[1])

    return _implicit_driver(f, t_span, y0, n, stage, "sdirk", jac)


def esdirk(f, t_span, y0, n: int = 100, jac=None):
    """ESDIRK: explicit first stage, singly diagonally implicit, order 3.

    The order-3 conditions for three stages force ``gamma^2 - gamma + 1/6 = 0``;
    of its two roots only ``(1 + 1/sqrt(3))/2`` gives an A-stable method (the
    other amplifies). It is A-stable but not L-stable -- ``R(inf) ~ 0.73`` -- so
    for problems needing stiff decay damped hard, prefer :func:`radau_iia` or
    :func:`tr_bdf2`.
    """
    gamma = 0.5 * (1.0 + 1.0 / np.sqrt(3.0))
    a32 = (1.0 - 2.0 * gamma) / (4.0 * gamma)
    a31 = 1.0 - a32 - gamma
    c = [0.0, 2.0 * gamma, 1.0]
    A = [[0.0, 0.0, 0.0],
         [gamma, gamma, 0.0],
         [a31, a32, gamma]]
    b = A[2]

    def stage(fc, t, y, h, jac):
        K = [fc(t, y)]
        for i in range(1, 3):
            base = y + h * sum(A[i][j] * K[j] for j in range(i))

            def G(k, base=base, i=i):
                return k - fc(t + c[i] * h, base + h * A[i][i] * k)

            K.append(_newton_solve(G, K[-1]))
        return y + h * sum(b[i] * K[i] for i in range(3))

    return _implicit_driver(f, t_span, y0, n, stage, "esdirk3", jac)


def tr_bdf2(f, t_span, y0, n: int = 100, jac=None):
    """TR-BDF2: a trapezoid step followed by a BDF2 step.

    Order 2, L-stable, and stiffly accurate -- the scheme used in several
    circuit and multiphysics simulators for exactly those properties.
    """
    gamma = 2.0 - np.sqrt(2.0)
    r2 = np.sqrt(2.0) / 4.0
    c = [0.0, gamma, 1.0]
    A = [[0.0, 0.0, 0.0],
         [gamma / 2, gamma / 2, 0.0],
         [r2, r2, 1.0 - np.sqrt(2.0) / 2.0]]
    b = A[2]

    def stage(fc, t, y, h, jac):
        K = [fc(t, y)]
        for i in range(1, 3):
            base = y + h * sum(A[i][j] * K[j] for j in range(i))

            def G(k, base=base, i=i):
                return k - fc(t + c[i] * h, base + h * A[i][i] * k)

            K.append(_newton_solve(G, K[-1]))
        return y + h * sum(b[i] * K[i] for i in range(3))

    return _implicit_driver(f, t_span, y0, n, stage, "tr_bdf2", jac)


# BDF coefficients: alpha[k] for orders 1..6 with the leading beta
_BDF = {
    1: ([1.0, -1.0], 1.0),
    2: ([3 / 2, -2.0, 1 / 2], 1.0),
    3: ([11 / 6, -3.0, 3 / 2, -1 / 3], 1.0),
    4: ([25 / 12, -4.0, 3.0, -4 / 3, 1 / 4], 1.0),
    5: ([137 / 60, -5.0, 5.0, -10 / 3, 5 / 4, -1 / 5], 1.0),
    6: ([49 / 20, -6.0, 15 / 2, -20 / 3, 15 / 4, -6 / 5, 1 / 6], 1.0),
}


def bdf(f, t_span, y0, n: int = 100, order: int = 2, jac=None):
    """Backward differentiation formulas of order 1-6.

    BDF1-2 are A-stable; BDF3-6 are only stiffly stable, and BDF7+ is unstable
    (the second Dahlquist barrier), which is why the family stops at 6.
    """
    if not 1 <= order <= 6:
        raise ValueError("BDF is unstable beyond order 6")
    alpha, beta = _BDF[order]
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    y0 = as_vector(y0)
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    ys = np.empty((n + 1, y0.size))
    ys[0] = y0
    # bootstrap the history with a self-starting implicit method
    k = order
    if n >= k and k > 1:
        boot = radau_iia(f, (t0, t0 + (k - 1) * h), y0, n=(k - 1) * 4, stages=3)
        for i in range(1, k):
            ys[i] = boot(t0 + i * h)
    for i in range(k, n + 1):
        hist = sum(alpha[j] * ys[i - j] for j in range(1, k + 1))

        def G(z, i=i, hist=hist):
            return alpha[0] * z + hist - h * beta * fc(ts[i], z)

        ys[i] = _newton_solve(G, ys[i - 1] + h * fc(ts[i - 1], ys[i - 1]))
    return ODESolution(ts, ys, f"bdf{order}", n, n, 0, fc.calls, True, "completed")


def rosenbrock(f, t_span, y0, n: int = 100, jac=None):
    """Rosenbrock method (2-stage, order 2): linearly implicit.

    Uses the Jacobian directly instead of a Newton iteration, so each step
    costs exactly two linear solves.
    """
    gamma = 1.0 - np.sqrt(2.0) / 2.0
    fc = CountedFunction(lambda t, y: as_vector(f(t, y)))
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.empty(n + 1)
    ys = np.empty((n + 1, y.size))
    ts[0], ys[0] = t0, y
    I = np.eye(y.size)
    t = t0
    for i in range(n):
        J = jac(t, y) if jac is not None else numerical_jacobian(lambda z: fc(t, z), y)
        M = I - gamma * h * J
        k1 = np.linalg.solve(M, fc(t, y))
        k2 = np.linalg.solve(M, fc(t + h, y + h * k1) - 2 * k1)
        y = y + h * (1.5 * k1 + 0.5 * k2)
        t = t0 + (i + 1) * h
        ts[i + 1], ys[i + 1] = t, y
    return ODESolution(ts, ys, "rosenbrock", n, n, 0, fc.calls, True, "completed")
