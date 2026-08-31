"""Exponential integrators for semilinear problems ``y' = A y + g(t, y)``.

These treat the stiff linear part exactly through the matrix exponential, so
they stay stable at step sizes an explicit method could never take.
"""

from __future__ import annotations

import numpy as np

from ..core.types import ODESolution
from ..core.utils import CountedFunction, as_vector, check_square
from ..linalg.eigen import matrix_exponential

__all__ = [
    "phi_function",
    "exponential_euler",
    "etd_rk2",
    "etd_rk4",
    "exponential_rosenbrock",
    "magnus_second_order",
    "krylov_expm_multiply",
]


def phi_function(A, k: int = 1):
    """Matrix ``phi`` function ``phi_k(A)`` of the exponential integrators.

    ``phi_0(z) = e^z`` and ``phi_{k+1}(z) = (phi_k(z) - 1/k!)/z``. Computed from
    a single exponential of the augmented block matrix

    ``[[A, I, 0, ...], [0, 0, I, ...], ..., [0, ..., 0]]``

    whose top-right block is exactly ``phi_k(A)``. That formulation inherits the
    scaling-and-squaring accuracy of the exponential and avoids both the
    cancellation of the defining quotient near ``A = 0`` and the cancellation of
    a raw Taylor series for large ``||A||``.
    """
    A = check_square(A)
    n = A.shape[0]
    if k == 0:
        return matrix_exponential(A)
    m = n * (k + 1)
    M = np.zeros((m, m))
    M[:n, :n] = A
    for j in range(k):
        M[j * n : (j + 1) * n, (j + 1) * n : (j + 2) * n] = np.eye(n)
    E = matrix_exponential(M)
    return E[:n, k * n : (k + 1) * n]


def _phi_series(A, k, terms=60):
    """Direct Taylor series for ``phi_k`` (used when the recurrence is awkward)."""
    from math import factorial

    n = A.shape[0]
    P = np.eye(n) / factorial(k)
    term = np.eye(n)
    for j in range(1, terms):
        term = term @ A
        P = P + term / factorial(j + k)
    return P


def exponential_euler(A, g, t_span, y0, n: int = 100):
    """Exponential Euler: ``y_{n+1} = e^{hA} y_n + h phi_1(hA) g(t_n, y_n)``.

    Exact when ``g`` is zero, whatever the stiffness of ``A``.
    """
    A = check_square(A)
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    E = matrix_exponential(h * A)
    P1 = phi_function(h * A, 1)
    gc = CountedFunction(lambda t, v: as_vector(g(t, v)))
    ts = np.linspace(t0, tf, n + 1)
    ys = np.empty((n + 1, y.size))
    ys[0] = y
    for i in range(n):
        y = E @ y + h * (P1 @ gc(ts[i], y))
        ys[i + 1] = y
    return ODESolution(ts, ys, "exponential_euler", n, n, 0, gc.calls, True, "completed")


def etd_rk2(A, g, t_span, y0, n: int = 100):
    """Exponential time differencing with a second-order Runge-Kutta correction."""
    A = check_square(A)
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    E = matrix_exponential(h * A)
    P1 = phi_function(h * A, 1)
    P2 = phi_function(h * A, 2)
    gc = CountedFunction(lambda t, v: as_vector(g(t, v)))
    ts = np.linspace(t0, tf, n + 1)
    ys = np.empty((n + 1, y.size))
    ys[0] = y
    for i in range(n):
        gn = gc(ts[i], y)
        a = E @ y + h * (P1 @ gn)
        y = a + h * (P2 @ (gc(ts[i + 1], a) - gn))
        ys[i + 1] = y
    return ODESolution(ts, ys, "etd_rk2", n, n, 0, gc.calls, True, "completed")


def etd_rk4(A, g, t_span, y0, n: int = 100):
    """Cox-Matthews ETDRK4: fourth-order exponential time differencing."""
    A = check_square(A)
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    E = matrix_exponential(h * A)
    E2 = matrix_exponential(0.5 * h * A)
    hA = h * A
    p1h = phi_function(0.5 * hA, 1)
    # Cox-Matthews coefficients expressed through phi functions
    phi1 = phi_function(hA, 1)
    phi2 = phi_function(hA, 2)
    phi3 = phi_function(hA, 3)
    alpha = h * (phi1 - 3 * phi2 + 4 * phi3)
    beta = h * (2 * phi2 - 4 * phi3)
    gamma = h * (-phi2 + 4 * phi3)
    gc = CountedFunction(lambda t, v: as_vector(g(t, v)))
    ts = np.linspace(t0, tf, n + 1)
    ys = np.empty((n + 1, y.size))
    ys[0] = y
    for i in range(n):
        t = ts[i]
        Nu = gc(t, y)
        a = E2 @ y + 0.5 * h * (p1h @ Nu)
        Na = gc(t + h / 2, a)
        b = E2 @ y + 0.5 * h * (p1h @ Na)
        Nb = gc(t + h / 2, b)
        c = E2 @ a + 0.5 * h * (p1h @ (2 * Nb - Nu))
        Nc = gc(t + h, c)
        y = E @ y + alpha @ Nu + beta @ (Na + Nb) + gamma @ Nc
        ys[i + 1] = y
    return ODESolution(ts, ys, "etd_rk4", n, n, 0, gc.calls, True, "completed")


def exponential_rosenbrock(f, t_span, y0, n: int = 100, jac=None):
    """Exponential Rosenbrock-Euler: linearize at each step, then exponentiate.

    Order 2, exact for linear autonomous problems, and free of linear solves.
    For a non-autonomous right-hand side the ``h^2 phi_2(hJ) df/dt`` term is
    required as well -- without it the method drops to order 1 -- so it is
    included here, with ``df/dt`` taken by a central difference.
    """
    from ..core.utils import numerical_jacobian

    fc = CountedFunction(lambda t, v: as_vector(f(t, v)))
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    ys = np.empty((n + 1, y.size))
    ys[0] = y
    for i in range(n):
        t = ts[i]
        J = jac(t, y) if jac is not None else numerical_jacobian(lambda v: fc(t, v), y)
        P1 = phi_function(h * J, 1)
        dt = 1e-7 * max(1.0, abs(t))
        dfdt = (fc(t + dt, y) - fc(t - dt, y)) / (2 * dt)
        P2 = phi_function(h * J, 2)
        y = y + h * (P1 @ fc(t, y)) + h * h * (P2 @ dfdt)
        ys[i + 1] = y
    return ODESolution(ts, ys, "exponential_rosenbrock", n, n, 0, fc.calls, True,
                       "completed")


def magnus_second_order(A_of_t, t_span, y0, n: int = 100):
    """Second-order Magnus expansion for linear systems ``y' = A(t) y``.

    Preserves the Lie-group structure of the flow, so quantities such as
    orthogonality or determinant are retained exactly.
    """
    y = as_vector(y0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    ys = np.empty((n + 1, y.size))
    ys[0] = y
    for i in range(n):
        Omega = h * np.asarray(A_of_t(ts[i] + h / 2), dtype=float)
        y = matrix_exponential(Omega) @ y
        ys[i + 1] = y
    return ODESolution(ts, ys, "magnus2", n, n, 0, n, True, "completed")


def krylov_expm_multiply(A, v, t: float = 1.0, m: int = 30):
    """Compute ``exp(tA) v`` in a Krylov subspace, without forming the exponential.

    The standard approach when ``A`` is large and sparse.
    """
    from ..linalg.eigen import arnoldi

    A = check_square(A)
    v = as_vector(v)
    beta = np.linalg.norm(v)
    if beta == 0:
        return v.copy()
    m = min(m, A.shape[0])
    Q, H = arnoldi(A, m, v)
    Hm = H[:m, :m] if H.shape[0] > m else H
    k = Hm.shape[0]
    E = matrix_exponential(t * Hm)
    e1 = np.zeros(k)
    e1[0] = 1.0
    return beta * (Q[:, :k] @ (E @ e1))
