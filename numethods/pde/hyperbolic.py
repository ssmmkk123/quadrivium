"""Hyperbolic PDEs: wave propagation and conservation laws.

Linear advection exposes the central tension of the field -- accuracy versus
monotonicity (Godunov's theorem) -- which is why the high-resolution schemes
here blend a low-order and a high-order flux through a limiter.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DomainError
from ..core.types import PDESolution
from ..core.utils import as_vector

__all__ = [
    "wave_explicit",
    "wave_implicit",
    "advection_upwind",
    "lax_friedrichs",
    "lax_wendroff",
    "beam_warming",
    "maccormack",
    "leapfrog_advection",
    "tvd_scheme",
    "flux_limiter",
    "godunov_burgers",
    "lax_friedrichs_burgers",
    "cfl_number",
]


def cfl_number(c: float, dt: float, dx: float) -> float:
    """Courant number ``C = c dt / dx``; explicit schemes need ``|C| <= 1``."""
    return c * dt / dx


def _setup(u0, x_span, nx, t_span, nt):
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    u = np.array([u0(xi) for xi in x], dtype=float) if callable(u0) else as_vector(u0).copy()
    return x, t, x[1] - x[0], t[1] - t[0], u


def wave_explicit(u0, v0, c: float, x_span, t_span, nx: int = 100, nt: int = 200,
                  bc=(0.0, 0.0), check_stability: bool = True):
    """Explicit central scheme for ``u_tt = c^2 u_xx``.

    Stable under the CFL condition ``c dt/dx <= 1``.
    """
    x, t, dx, dt, u = _setup(u0, x_span, nx, t_span, nt)
    lam = c * dt / dx
    if check_stability and lam > 1.0:
        raise DomainError(
            f"CFL condition violated: c dt/dx = {lam:.4f} > 1; use at least "
            f"{int(np.ceil(c * (t[-1] - t[0]) / dx))} time steps"
        )
    v = np.array([v0(xi) for xi in x]) if callable(v0) else as_vector(v0)
    U = np.empty((nt + 1, nx + 1))
    U[0] = u
    u_prev = u.copy()
    u_cur = u.copy()
    # first step uses the initial velocity (Taylor expansion in time)
    u_cur[1:-1] = (u[1:-1] + dt * v[1:-1]
                   + 0.5 * lam**2 * (u[2:] - 2 * u[1:-1] + u[:-2]))
    u_cur[0], u_cur[-1] = bc[0], bc[1]
    U[1] = u_cur
    for k in range(1, nt):
        u_new = np.empty_like(u_cur)
        u_new[1:-1] = (2 * u_cur[1:-1] - u_prev[1:-1]
                       + lam**2 * (u_cur[2:] - 2 * u_cur[1:-1] + u_cur[:-2]))
        u_new[0], u_new[-1] = bc[0], bc[1]
        u_prev, u_cur = u_cur, u_new
        U[k + 1] = u_cur
    return PDESolution(U, (x,), t, "wave_explicit")


def wave_implicit(u0, v0, c: float, x_span, t_span, nx: int = 100, nt: int = 200,
                  bc=(0.0, 0.0), theta: float = 0.25):
    """Newmark-style implicit wave scheme: unconditionally stable for ``theta >= 1/4``."""
    from ..linalg.direct import thomas

    x, t, dx, dt, u = _setup(u0, x_span, nx, t_span, nt)
    lam2 = (c * dt / dx) ** 2
    v = np.array([v0(xi) for xi in x]) if callable(v0) else as_vector(v0)
    U = np.empty((nt + 1, nx + 1))
    U[0] = u
    m = nx - 1
    lower = np.full(m - 1, -theta * lam2)
    diag = np.full(m, 1 + 2 * theta * lam2)
    upper = np.full(m - 1, -theta * lam2)
    u_prev = u.copy()
    u_cur = u.copy()
    u_cur[1:-1] = u[1:-1] + dt * v[1:-1] + 0.5 * lam2 * (u[2:] - 2 * u[1:-1] + u[:-2])
    u_cur[0], u_cur[-1] = bc[0], bc[1]
    U[1] = u_cur
    for k in range(1, nt):
        lap_cur = u_cur[2:] - 2 * u_cur[1:-1] + u_cur[:-2]
        lap_prev = u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]
        rhs = (2 * u_cur[1:-1] - u_prev[1:-1]
               + lam2 * ((1 - 2 * theta) * lap_cur + theta * lap_prev))
        u_new = np.empty_like(u_cur)
        u_new[1:-1] = thomas(lower, diag, upper, rhs)
        u_new[0], u_new[-1] = bc[0], bc[1]
        u_prev, u_cur = u_cur, u_new
        U[k + 1] = u_cur
    return PDESolution(U, (x,), t, "wave_implicit")


def _periodic_advection(u0, c, x_span, t_span, nx, nt, update, name):
    """Driver for linear advection schemes with periodic boundaries."""
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    nu = c * dt / dx
    u = np.array([u0(xi) for xi in x], dtype=float) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, u.size))
    U[0] = u
    for k in range(nt):
        u = update(u, nu)
        U[k + 1] = u
    return PDESolution(U, (x,), t, name)


def advection_upwind(u0, c: float, x_span, t_span, nx: int = 200, nt: int = 400):
    """First-order upwind: monotone but strongly diffusive."""
    def step(u, nu):
        if nu > 0:
            return u - nu * (u - np.roll(u, 1))
        return u - nu * (np.roll(u, -1) - u)

    return _periodic_advection(u0, c, x_span, t_span, nx, nt, step, "upwind")


def lax_friedrichs(u0, c: float, x_span, t_span, nx: int = 200, nt: int = 400):
    """Lax-Friedrichs: first order, stable, even more diffusive than upwind."""
    def step(u, nu):
        return 0.5 * (np.roll(u, -1) + np.roll(u, 1)) - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1))

    return _periodic_advection(u0, c, x_span, t_span, nx, nt, step, "lax_friedrichs")


def lax_wendroff(u0, c: float, x_span, t_span, nx: int = 200, nt: int = 400):
    """Lax-Wendroff: second order, but oscillates near discontinuities."""
    def step(u, nu):
        return (u - 0.5 * nu * (np.roll(u, -1) - np.roll(u, 1))
                + 0.5 * nu**2 * (np.roll(u, -1) - 2 * u + np.roll(u, 1)))

    return _periodic_advection(u0, c, x_span, t_span, nx, nt, step, "lax_wendroff")


def beam_warming(u0, c: float, x_span, t_span, nx: int = 200, nt: int = 400):
    """Beam-Warming: second order, one-sided (upwind-biased) stencil."""
    def step(u, nu):
        um1, um2 = np.roll(u, 1), np.roll(u, 2)
        return (u - nu / 2 * (3 * u - 4 * um1 + um2)
                + nu**2 / 2 * (u - 2 * um1 + um2))

    return _periodic_advection(u0, c, x_span, t_span, nx, nt, step, "beam_warming")


def maccormack(u0, c: float, x_span, t_span, nx: int = 200, nt: int = 400):
    """MacCormack predictor-corrector: second order, equivalent to Lax-Wendroff
    for linear advection but generalizing directly to nonlinear systems."""
    def step(u, nu):
        u_p = u - nu * (np.roll(u, -1) - u)
        return 0.5 * (u + u_p - nu * (u_p - np.roll(u_p, 1)))

    return _periodic_advection(u0, c, x_span, t_span, nx, nt, step, "maccormack")


def leapfrog_advection(u0, c: float, x_span, t_span, nx: int = 200, nt: int = 400):
    """Leapfrog: second order and non-dissipative, but needs two levels."""
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dx, dt = x[1] - x[0], t[1] - t[0]
    nu = c * dt / dx
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, u.size))
    U[0] = u
    u_prev = u
    u_cur = u - nu / 2 * (np.roll(u, -1) - np.roll(u, 1))  # start with Lax-Wendroff
    U[1] = u_cur
    for k in range(1, nt):
        u_new = u_prev - nu * (np.roll(u_cur, -1) - np.roll(u_cur, 1))
        u_prev, u_cur = u_cur, u_new
        U[k + 1] = u_cur
    return PDESolution(U, (x,), t, "leapfrog_advection")


def flux_limiter(r, kind: str = "van_leer"):
    """Flux limiter ``phi(r)`` blending high- and low-order fluxes.

    Available: minmod, superbee, van_leer, van_albada, mc, koren, ospre.

    Every limiter here returns 0 for ``r <= 0``.  A negative ``r`` means the
    solution has a local extremum in the stencil, and any nonzero ``phi`` there
    reintroduces the high-order flux that creates new extrema -- so this clamp
    is what makes the scheme TVD, not a cosmetic guard.  The result also lies
    in Sweby's region ``0 <= phi(r) <= min(2r, 2)``.
    """
    r = np.asarray(r, dtype=float)
    positive = r > 0.0
    if kind == "minmod":
        return np.maximum(0.0, np.minimum(1.0, r))
    if kind == "superbee":
        return np.maximum.reduce([np.zeros_like(r), np.minimum(2 * r, 1.0),
                                  np.minimum(r, 2.0)])
    if kind == "van_leer":
        return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-300)
    if kind == "van_albada":
        return np.where(positive, (r * r + r) / (r * r + 1.0), 0.0)
    if kind == "mc":
        return np.maximum(0.0, np.minimum.reduce([2 * r, 0.5 * (1 + r), 2 * np.ones_like(r)]))
    if kind == "koren":
        return np.maximum(0.0, np.minimum.reduce([2 * r, (1 + 2 * r) / 3.0,
                                                  2 * np.ones_like(r)]))
    if kind == "ospre":
        return np.where(positive, 1.5 * (r * r + r) / (r * r + r + 1.0), 0.0)
    raise ValueError(f"unknown limiter {kind!r}")


def tvd_scheme(u0, c: float, x_span, t_span, nx: int = 200, nt: int = 400,
               limiter: str = "van_leer"):
    """High-resolution TVD scheme: second-order accurate yet non-oscillatory.

    Blends the Lax-Wendroff flux with the upwind flux through a limiter, which
    is how Godunov's barrier is sidestepped (the scheme is nonlinear even for a
    linear equation).
    """
    def step(u, nu):
        # smoothness ratio at each interface, measured upwind
        du = np.roll(u, -1) - u
        eps = 1e-12
        if nu >= 0:
            r = (u - np.roll(u, 1)) / np.where(np.abs(du) < eps, eps, du)
        else:
            r = (np.roll(u, -2) - np.roll(u, -1)) / np.where(np.abs(du) < eps, eps, du)
        phi = flux_limiter(r, limiter)
        # low-order (upwind) flux plus limited antidiffusive correction
        if nu >= 0:
            flux = nu * u + 0.5 * nu * (1 - nu) * phi * du
        else:
            flux = nu * np.roll(u, -1) - 0.5 * nu * (1 + nu) * phi * du
        return u - (flux - np.roll(flux, 1))

    return _periodic_advection(u0, c, x_span, t_span, nx, nt, step, f"tvd_{limiter}")


def godunov_burgers(u0, x_span, t_span, nx: int = 200, nt: int = 400):
    """Godunov's method for the inviscid Burgers equation ``u_t + (u^2/2)_x = 0``.

    Solves the Riemann problem exactly at every interface, so shocks are
    captured at the right speed without spurious oscillation.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dx, dt = x[1] - x[0], t[1] - t[0]
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, u.size))
    U[0] = u

    def godunov_flux(ul, ur):
        """Exact Riemann solution flux for Burgers."""
        f = lambda v: 0.5 * v * v
        shock = ul > ur
        s = 0.5 * (ul + ur)
        out = np.where(
            shock,
            np.where(s > 0, f(ul), f(ur)),                       # shock
            np.where(ul > 0, f(ul), np.where(ur < 0, f(ur), 0.0)),  # rarefaction
        )
        return out

    for k in range(nt):
        F = godunov_flux(u, np.roll(u, -1))
        u = u - dt / dx * (F - np.roll(F, 1))
        U[k + 1] = u
    return PDESolution(U, (x,), t, "godunov_burgers")


def lax_friedrichs_burgers(u0, x_span, t_span, nx: int = 200, nt: int = 400):
    """Lax-Friedrichs for Burgers: robust and monotone, but smears shocks."""
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx, endpoint=False)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dx, dt = x[1] - x[0], t[1] - t[0]
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, u.size))
    U[0] = u
    for k in range(nt):
        f = 0.5 * u * u
        u = (0.5 * (np.roll(u, -1) + np.roll(u, 1))
             - 0.5 * dt / dx * (np.roll(f, -1) - np.roll(f, 1)))
        U[k + 1] = u
    return PDESolution(U, (x,), t, "lax_friedrichs_burgers")
