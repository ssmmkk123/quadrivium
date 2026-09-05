"""Parabolic PDEs: the heat / diffusion equation and its relatives.

The explicit scheme is cheap but conditionally stable (``r <= 1/2``); the
implicit and Crank-Nicolson schemes are unconditionally stable at the cost of a
tridiagonal solve per step.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DomainError
from ..core.types import PDESolution
from ..core.utils import as_vector
from ..linalg.direct import thomas

__all__ = [
    "heat_ftcs",
    "heat_btcs",
    "heat_crank_nicolson",
    "heat_theta",
    "heat_2d_adi",
    "method_of_lines",
    "diffusion_reaction",
    "advection_diffusion",
    "stability_ratio",
]


def stability_ratio(alpha: float, dt: float, dx: float) -> float:
    """Diffusion number ``r = alpha dt / dx^2``.

    The explicit FTCS scheme is stable only for ``r <= 1/2``.
    """
    return alpha * dt / dx**2


def _grid(x_span, nx, t_span, nt):
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    return x, t, x[1] - x[0], t[1] - t[0]


def heat_ftcs(u0, alpha: float, x_span, t_span, nx: int = 50, nt: int = 1000,
              bc=(0.0, 0.0), source=None, check_stability: bool = True):
    """Forward-time centred-space explicit scheme for ``u_t = alpha u_xx``.

    Conditionally stable: raises unless ``r = alpha dt/dx^2 <= 1/2``.
    """
    x, t, dx, dt = _grid(x_span, nx, t_span, nt)
    r = alpha * dt / dx**2
    if check_stability and r > 0.5:
        raise DomainError(
            f"FTCS is unstable for r = {r:.4f} > 1/2; use nt >= "
            f"{int(np.ceil(2 * alpha * (t[-1] - t[0]) / dx**2))} steps, or an "
            "implicit scheme such as heat_btcs / heat_crank_nicolson"
        )
    u = np.array([u0(xi) for xi in x], dtype=float) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, nx + 1))
    U[0] = u
    for k in range(nt):
        un = u.copy()
        u[1:-1] = un[1:-1] + r * (un[2:] - 2 * un[1:-1] + un[:-2])
        if source is not None:
            u[1:-1] += dt * np.array([source(xi, t[k]) for xi in x[1:-1]])
        u[0] = bc[0](t[k + 1]) if callable(bc[0]) else bc[0]
        u[-1] = bc[1](t[k + 1]) if callable(bc[1]) else bc[1]
        U[k + 1] = u
    return PDESolution(U, (x,), t, "heat_ftcs")


def heat_btcs(u0, alpha: float, x_span, t_span, nx: int = 50, nt: int = 100,
              bc=(0.0, 0.0), source=None):
    """Backward-time centred-space implicit scheme: unconditionally stable."""
    x, t, dx, dt = _grid(x_span, nx, t_span, nt)
    r = alpha * dt / dx**2
    u = np.array([u0(xi) for xi in x], dtype=float) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, nx + 1))
    U[0] = u
    m = nx - 1
    lower = np.full(m - 1, -r)
    diag = np.full(m, 1 + 2 * r)
    upper = np.full(m - 1, -r)
    for k in range(nt):
        rhs = u[1:-1].copy()
        if source is not None:
            rhs += dt * np.array([source(xi, t[k + 1]) for xi in x[1:-1]])
        bl = bc[0](t[k + 1]) if callable(bc[0]) else bc[0]
        br = bc[1](t[k + 1]) if callable(bc[1]) else bc[1]
        rhs[0] += r * bl
        rhs[-1] += r * br
        u[1:-1] = thomas(lower, diag, upper, rhs)
        u[0], u[-1] = bl, br
        U[k + 1] = u
    return PDESolution(U, (x,), t, "heat_btcs")


def heat_crank_nicolson(u0, alpha: float, x_span, t_span, nx: int = 50,
                        nt: int = 100, bc=(0.0, 0.0), source=None):
    """Crank-Nicolson: unconditionally stable and second order in both variables."""
    return heat_theta(u0, alpha, x_span, t_span, nx, nt, 0.5, bc, source)


def heat_theta(u0, alpha: float, x_span, t_span, nx: int = 50, nt: int = 100,
               theta: float = 0.5, bc=(0.0, 0.0), source=None):
    """Theta scheme: explicit (0), Crank-Nicolson (1/2), fully implicit (1)."""
    x, t, dx, dt = _grid(x_span, nx, t_span, nt)
    r = alpha * dt / dx**2
    u = np.array([u0(xi) for xi in x], dtype=float) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, nx + 1))
    U[0] = u
    m = nx - 1
    lower = np.full(m - 1, -theta * r)
    diag = np.full(m, 1 + 2 * theta * r)
    upper = np.full(m - 1, -theta * r)
    for k in range(nt):
        un = u[1:-1]
        rhs = un + (1 - theta) * r * (u[2:] - 2 * un + u[:-2])
        if source is not None:
            rhs = rhs + dt * np.array([source(xi, t[k] + theta * dt) for xi in x[1:-1]])
        bl = bc[0](t[k + 1]) if callable(bc[0]) else bc[0]
        br = bc[1](t[k + 1]) if callable(bc[1]) else bc[1]
        bl0 = bc[0](t[k]) if callable(bc[0]) else bc[0]
        br0 = bc[1](t[k]) if callable(bc[1]) else bc[1]
        rhs[0] += theta * r * bl + (1 - theta) * r * (bl0 - u[0])
        rhs[-1] += theta * r * br + (1 - theta) * r * (br0 - u[-1])
        u_new = thomas(lower, diag, upper, rhs) if theta > 0 else rhs
        u = np.concatenate([[bl], u_new, [br]])
        U[k + 1] = u
    return PDESolution(U, (x,), t, f"heat_theta_{theta}")


def heat_2d_adi(u0, alpha: float, x_span, y_span, t_span, nx: int = 40, ny: int = 40,
                nt: int = 100, bc: float = 0.0):
    """Peaceman-Rachford ADI for the 2-D heat equation.

    Splits each step into two half-steps, each a set of tridiagonal solves, so
    the cost stays ``O(N)`` while remaining unconditionally stable.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), ny + 1)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
    rx = alpha * dt / (2 * dx**2)
    ry = alpha * dt / (2 * dy**2)
    if callable(u0):
        U = np.array([[u0(xi, yj) for yj in y] for xi in x])
    else:
        U = np.array(u0, dtype=float)
    out = np.empty((nt + 1, nx + 1, ny + 1))
    out[0] = U
    lx = np.full(nx - 2, -rx)
    dxg = np.full(nx - 1, 1 + 2 * rx)
    ux = np.full(nx - 2, -rx)
    ly = np.full(ny - 2, -ry)
    dyg = np.full(ny - 1, 1 + 2 * ry)
    uy = np.full(ny - 2, -ry)
    # Every grid line in a half-step solves the *same* tridiagonal matrix, so
    # the lines go to the solver together: one factorization and one call per
    # half-step instead of one per line.
    for k in range(nt):
        half = U.copy()
        # implicit in x, explicit in y -- one column of `rhs` per interior j
        rhs = U[1:-1, 1:-1] + ry * (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2])
        rhs[0, :] += rx * bc
        rhs[-1, :] += rx * bc
        half[1:-1, 1:-1] = thomas(lx, dxg, ux, rhs)
        half[0, :] = half[-1, :] = bc
        half[:, 0] = half[:, -1] = bc
        # implicit in y, explicit in x -- transposed so each i is a column
        rhs = (half[1:-1, 1:-1]
               + rx * (half[2:, 1:-1] - 2 * half[1:-1, 1:-1] + half[:-2, 1:-1])).T
        rhs[0, :] += ry * bc
        rhs[-1, :] += ry * bc
        U[1:-1, 1:-1] = thomas(ly, dyg, uy, rhs).T
        U[0, :] = U[-1, :] = bc
        U[:, 0] = U[:, -1] = bc
        out[k + 1] = U
    return PDESolution(out, (x, y), t, "heat_2d_adi")


def method_of_lines(u0, rhs, x_span, t_span, nx: int = 50, solver="rk45",
                    bc=(0.0, 0.0), **kwargs):
    """Method of lines: discretize in space, then hand the ODE system to a solver.

    ``rhs(t, u, x, dx)`` returns ``du/dt`` at the interior nodes.
    """
    from ..ode.explicit import dormand_prince, rk4
    from ..ode.implicit import bdf, radau_iia

    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    dx = x[1] - x[0]
    u_init = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0)

    def f(t, u):
        full = u.copy()
        full[0] = bc[0](t) if callable(bc[0]) else bc[0]
        full[-1] = bc[1](t) if callable(bc[1]) else bc[1]
        du = rhs(t, full, x, dx)
        du[0] = du[-1] = 0.0
        return du

    solvers = {"rk45": dormand_prince, "rk4": rk4, "radau": radau_iia, "bdf": bdf}
    sol = solvers[solver](f, t_span, u_init, **kwargs)
    return PDESolution(sol.y, (x,), sol.t, f"method_of_lines_{solver}")


def diffusion_reaction(u0, alpha: float, reaction, x_span, t_span, nx: int = 50,
                       nt: int = 200, bc=(0.0, 0.0)):
    """Reaction-diffusion ``u_t = alpha u_xx + f(u)`` by IMEX splitting.

    Diffusion is treated implicitly (removing the stiff constraint) and the
    reaction term explicitly.
    """
    x, t, dx, dt = _grid(x_span, nx, t_span, nt)
    r = alpha * dt / dx**2
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, nx + 1))
    U[0] = u
    m = nx - 1
    lower = np.full(m - 1, -r)
    diag = np.full(m, 1 + 2 * r)
    upper = np.full(m - 1, -r)
    for k in range(nt):
        rhs = u[1:-1] + dt * np.array([reaction(v) for v in u[1:-1]])
        bl = bc[0](t[k + 1]) if callable(bc[0]) else bc[0]
        br = bc[1](t[k + 1]) if callable(bc[1]) else bc[1]
        rhs[0] += r * bl
        rhs[-1] += r * br
        u = np.concatenate([[bl], thomas(lower, diag, upper, rhs), [br]])
        U[k + 1] = u
    return PDESolution(U, (x,), t, "diffusion_reaction")


def advection_diffusion(u0, velocity: float, alpha: float, x_span, t_span,
                        nx: int = 100, nt: int = 500, bc=(0.0, 0.0),
                        upwind: bool = True):
    """Advection-diffusion ``u_t + v u_x = alpha u_xx``.

    Upwinding the advective term keeps the scheme monotone at high Peclet
    number, where central differencing oscillates.
    """
    x, t, dx, dt = _grid(x_span, nx, t_span, nt)
    r = alpha * dt / dx**2
    c = velocity * dt / dx
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, nx + 1))
    U[0] = u
    for k in range(nt):
        un = u.copy()
        diff = r * (un[2:] - 2 * un[1:-1] + un[:-2])
        if upwind:
            adv = (c * (un[1:-1] - un[:-2]) if velocity > 0
                   else c * (un[2:] - un[1:-1]))
        else:
            adv = c / 2 * (un[2:] - un[:-2])
        u[1:-1] = un[1:-1] - adv + diff
        u[0] = bc[0](t[k + 1]) if callable(bc[0]) else bc[0]
        u[-1] = bc[1](t[k + 1]) if callable(bc[1]) else bc[1]
        U[k + 1] = u
    return PDESolution(U, (x,), t, "advection_diffusion")
