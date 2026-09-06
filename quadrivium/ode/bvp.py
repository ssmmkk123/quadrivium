"""Two-point boundary value problems.

Three classical strategies: shooting (reduce to an IVP plus a root find),
finite differences (discretize into an algebraic system), and collocation /
Galerkin (expand in basis functions).
"""

from __future__ import annotations

from .. import numeric as np

from ..core.exceptions import ConvergenceError
from ..core.types import ODESolution
from ..core.utils import as_vector, numerical_jacobian
from ..linalg.direct import solve, thomas
from ..rootfind.scalar import brent, secant
from ..rootfind.systems import damped_newton_system

__all__ = [
    "shooting",
    "multiple_shooting",
    "linear_shooting",
    "finite_difference_bvp",
    "nonlinear_fd_bvp",
    "collocation_bvp",
    "galerkin_bvp",
    "sturm_liouville",
]


def shooting(f, t_span, bc_left: float, bc_right: float, guess_range=(-10.0, 10.0),
             n: int = 200, tol: float = 1e-10, method="rk4"):
    """Single shooting for ``y'' = f(t, y, y')`` with Dirichlet conditions.

    Solves the initial slope that makes the right boundary condition hold; the
    inner IVP solves are done with a fixed-step Runge-Kutta method.
    """
    from .explicit import solve_ivp

    a, b = float(t_span[0]), float(t_span[1])

    def endpoint(s):
        sol = solve_ivp(lambda t, y: np.array([y[1], f(t, y[0], y[1])]),
                        (a, b), [bc_left, float(s)], method=method, n=n)
        return float(sol.y[-1, 0]) - bc_right

    lo, hi = guess_range
    flo, fhi = endpoint(lo), endpoint(hi)
    if np.sign(flo) != np.sign(fhi):
        res = brent(endpoint, lo, hi, tol=tol)
    else:
        res = secant(endpoint, 0.5 * (lo + hi), 0.5 * (lo + hi) + 1.0, tol=tol)
    if not res.converged:
        raise ConvergenceError("shooting failed to match the right boundary condition",
                               iterations=res.iterations, residual=res.f_root)
    sol = solve_ivp(lambda t, y: np.array([y[1], f(t, y[0], y[1])]),
                    (a, b), [bc_left, float(res.root)], method=method, n=n)
    sol.method = "shooting"
    return sol


def linear_shooting(p, q, r, t_span, bc_left: float, bc_right: float, n: int = 200):
    """Shooting for the linear problem ``y'' = p(t) y' + q(t) y + r(t)``.

    Exploits linearity: two IVP solves suffice, with no iteration at all.
    """
    from .explicit import rk4

    a, b = float(t_span[0]), float(t_span[1])
    sol1 = rk4(lambda t, y: np.array([y[1], p(t) * y[1] + q(t) * y[0] + r(t)]),
               (a, b), [bc_left, 0.0], n)
    sol2 = rk4(lambda t, y: np.array([y[1], p(t) * y[1] + q(t) * y[0]]),
               (a, b), [0.0, 1.0], n)
    denom = sol2.y[-1, 0]
    if abs(denom) < 1e-300:
        raise ConvergenceError("linear shooting is degenerate: the homogeneous "
                               "solution vanishes at the right endpoint")
    c = (bc_right - sol1.y[-1, 0]) / denom
    y = sol1.y + c * sol2.y
    return ODESolution(sol1.t, y, "linear_shooting", n, n, 0,
                       sol1.n_rhs_evals + sol2.n_rhs_evals, True, "completed")


def multiple_shooting(f, t_span, bc_left, bc_right, segments: int = 4, n: int = 100,
                      tol: float = 1e-10):
    """Multiple shooting: split the interval and match at the junctions.

    Far more stable than single shooting when the IVP grows rapidly, because no
    single trajectory is integrated over the whole interval.
    """
    from .explicit import rk4

    a, b = float(t_span[0]), float(t_span[1])
    edges = np.linspace(a, b, segments + 1)
    F = lambda t, y: np.array([y[1], f(t, y[0], y[1])])

    def residual(z):
        """Unknowns: the state at each segment start (except the fixed value)."""
        states = [np.array([bc_left, z[0]])]
        for k in range(1, segments):
            states.append(np.array([z[2 * k - 1], z[2 * k]]))
        res = []
        for k in range(segments):
            sol = rk4(F, (edges[k], edges[k + 1]), states[k], n // segments + 2)
            end = sol.y[-1]
            if k < segments - 1:
                res.extend(end - states[k + 1])
            else:
                res.append(end[0] - bc_right)
        return np.array(res)

    z0 = np.zeros(2 * segments - 1)
    slope = (bc_right - bc_left) / (b - a)
    z0[0] = slope
    for k in range(1, segments):
        z0[2 * k - 1] = bc_left + slope * (edges[k] - a)
        z0[2 * k] = slope
    sol_r = damped_newton_system(residual, z0, tol=tol, max_iter=200)
    z = as_vector(sol_r.root)
    states = [np.array([bc_left, z[0]])]
    for k in range(1, segments):
        states.append(np.array([z[2 * k - 1], z[2 * k]]))
    ts, ys = [], []
    for k in range(segments):
        sol = rk4(F, (edges[k], edges[k + 1]), states[k], n // segments + 2)
        ts.append(sol.t[:-1] if k < segments - 1 else sol.t)
        ys.append(sol.y[:-1] if k < segments - 1 else sol.y)
    return ODESolution(np.concatenate(ts), np.vstack(ys), "multiple_shooting",
                       segments, segments, 0, 0, sol_r.converged, sol_r.message)


def finite_difference_bvp(p, q, r, t_span, bc_left: float, bc_right: float,
                          n: int = 100, bc_type: str = "dirichlet",
                          bc_coeffs=None):
    """Finite differences for the linear BVP ``y'' = p(t)y' + q(t)y + r(t)``.

    Produces a tridiagonal system solved in ``O(n)`` by the Thomas algorithm.
    ``bc_type`` may be ``'dirichlet'`` or ``'neumann'``.
    """
    a, b = float(t_span[0]), float(t_span[1])
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    if bc_type == "dirichlet":
        m = n - 1
        ti = t[1:-1]
        lower = np.array([-1.0 - h / 2 * p(ti[i]) for i in range(1, m)])
        diag = np.array([2.0 + h * h * q(ti[i]) for i in range(m)])
        upper = np.array([-1.0 + h / 2 * p(ti[i]) for i in range(m - 1)])
        rhs = np.array([-h * h * r(ti[i]) for i in range(m)])
        rhs[0] += (1.0 + h / 2 * p(ti[0])) * bc_left
        rhs[-1] += (1.0 - h / 2 * p(ti[-1])) * bc_right
        # note the sign convention: the system solves -y'' + p y' + q y = -r
        diag = -diag
        lower = -lower
        upper = -upper
        rhs = -rhs
        y_in = thomas(lower, diag, upper, rhs)
        y = np.concatenate([[bc_left], y_in, [bc_right]])
    else:
        # Neumann at both ends via ghost points
        A = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)
        for i in range(1, n):
            A[i, i - 1] = 1.0 / h**2 + p(t[i]) / (2 * h)
            A[i, i] = -2.0 / h**2 - q(t[i])
            A[i, i + 1] = 1.0 / h**2 - p(t[i]) / (2 * h)
            rhs[i] = r(t[i])
        A[0, 0], A[0, 1] = -3 / (2 * h), 4 / (2 * h)
        A[0, 2] = -1 / (2 * h)
        rhs[0] = bc_left
        A[n, n], A[n, n - 1] = 3 / (2 * h), -4 / (2 * h)
        A[n, n - 2] = 1 / (2 * h)
        rhs[n] = bc_right
        y = np.linalg.solve(A, rhs)
    return ODESolution(t, y.reshape(-1, 1), "finite_difference_bvp", n, n, 0, 0,
                       True, "completed")


def nonlinear_fd_bvp(f, t_span, bc_left: float, bc_right: float, n: int = 100,
                     tol: float = 1e-12, max_iter: int = 100, y_guess=None):
    """Finite differences plus Newton for the nonlinear BVP ``y'' = f(t, y, y')``."""
    a, b = float(t_span[0]), float(t_span[1])
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    if y_guess is None:
        y_in = bc_left + (bc_right - bc_left) * (t[1:-1] - a) / (b - a)
    else:
        y_in = as_vector(y_guess)[1:-1].copy()

    def residual(u):
        full = np.concatenate([[bc_left], u, [bc_right]])
        yp = (full[2:] - full[:-2]) / (2 * h)
        ypp = (full[2:] - 2 * full[1:-1] + full[:-2]) / h**2
        return ypp - np.array([f(t[i + 1], full[i + 1], yp[i]) for i in range(len(u))])

    res = damped_newton_system(residual, y_in, tol=tol, max_iter=max_iter)
    y = np.concatenate([[bc_left], as_vector(res.root), [bc_right]])
    return ODESolution(t, y.reshape(-1, 1), "nonlinear_fd_bvp", res.iterations,
                       res.iterations, 0, res.function_calls, res.converged,
                       res.message)


def collocation_bvp(f, t_span, bc_left: float, bc_right: float, n: int = 12,
                    tol: float = 1e-12):
    """Spectral collocation for ``y'' = f(t, y, y')`` on Chebyshev points.

    Converges exponentially for smooth solutions.
    """
    from ..diff.spectral import chebyshev_diff_matrix

    a, b = float(t_span[0]), float(t_span[1])
    D, x = chebyshev_diff_matrix(n, a, b)
    D2 = D @ D

    def residual(u):
        full = u.copy()
        full[0], full[-1] = bc_left, bc_right
        yp = D @ full
        ypp = D2 @ full
        r = np.array([ypp[i] - f(x[i], full[i], yp[i]) for i in range(len(x))])
        r[0] = full[0] - bc_left
        r[-1] = full[-1] - bc_right
        return r

    u0 = bc_left + (bc_right - bc_left) * (x - a) / (b - a)
    res = damped_newton_system(residual, u0, tol=tol, max_iter=200)
    u = as_vector(res.root)
    u[0], u[-1] = bc_left, bc_right
    return ODESolution(x, u.reshape(-1, 1), "collocation_bvp", res.iterations,
                       res.iterations, 0, res.function_calls, res.converged,
                       res.message)


def galerkin_bvp(p, q, r, t_span, bc_left: float, bc_right: float, n: int = 20):
    """Galerkin finite elements for ``-(y')' + q y = -r`` with hat functions.

    Solves the weak form on a uniform mesh of linear elements.
    """
    a, b = float(t_span[0]), float(t_span[1])
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    # Weak form of -y'' + q y = -r against hat functions: stiffness 1/h terms
    # plus q times the consistent mass matrix (2h/3 on the diagonal, h/6 off).
    m = n - 1
    K = np.zeros((m, m))
    F = np.zeros(m)
    for i in range(m):
        qi = q(t[i + 1])
        K[i, i] = 2.0 / h + qi * 2.0 * h / 3.0
        if i > 0:
            K[i, i - 1] = -1.0 / h + qi * h / 6.0
        if i < m - 1:
            K[i, i + 1] = -1.0 / h + qi * h / 6.0
        F[i] = -r(t[i + 1]) * h
    # Dirichlet lift: move the known boundary values to the right-hand side,
    # including their mass coupling to the first and last interior nodes.
    F[0] += bc_left * (1.0 / h - q(t[1]) * h / 6.0)
    F[-1] += bc_right * (1.0 / h - q(t[n - 1]) * h / 6.0)
    u = np.linalg.solve(K, F)
    y = np.concatenate([[bc_left], u, [bc_right]])
    return ODESolution(t, y.reshape(-1, 1), "galerkin_bvp", n, n, 0, 0, True,
                       "completed")


def sturm_liouville(p, q, w, t_span, n: int = 100, n_eigen: int = 5):
    """Sturm-Liouville eigenproblem ``-(p y')' + q y = lambda w y``.

    Discretized by finite differences into a generalized symmetric eigenproblem
    with Dirichlet conditions at both ends.
    """
    a, b = float(t_span[0]), float(t_span[1])
    h = (b - a) / n
    t = np.linspace(a, b, n + 1)
    m = n - 1
    A = np.zeros((m, m))
    B = np.zeros((m, m))
    for i in range(m):
        ti = t[i + 1]
        p_plus = p(ti + h / 2)
        p_minus = p(ti - h / 2)
        A[i, i] = (p_plus + p_minus) / h**2 + q(ti)
        if i > 0:
            A[i, i - 1] = -p_minus / h**2
        if i < m - 1:
            A[i, i + 1] = -p_plus / h**2
        B[i, i] = w(ti)
    L = np.linalg.cholesky(B)
    C = np.linalg.solve(L, np.linalg.solve(L, A.T).T)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[:n_eigen]
    modes = np.linalg.solve(L.T, vecs[:, idx])
    full = np.zeros((n + 1, len(idx)))
    full[1:-1, :] = modes
    return vals[idx], t, full
