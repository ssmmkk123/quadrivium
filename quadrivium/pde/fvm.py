"""Finite volume methods for conservation laws.

The finite volume formulation updates cell averages through fluxes at the cell
faces, so mass is conserved to round-off by construction -- the property that
makes it the standard choice for shocks and for CFD.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.storage import (pde_solution as PDESolution, TimeGrid,
                            Trajectory, output_control, OutputRecorder)
from ..core.utils import as_vector

__all__ = [
    "fvm_1d_conservation",
    "fvm_diffusion",
    "riemann_solver_burgers",
    "riemann_solver_linear",
    "rusanov_flux",
    "hll_flux",
    "muscl_reconstruct",
    "fvm_muscl",
    "minmod",
]


def minmod(a, b):
    """Minmod limiter: zero when the arguments disagree in sign."""
    return np.where(a * b <= 0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


def riemann_solver_linear(ul, ur, c: float):
    """Exact Riemann flux for linear advection ``u_t + c u_x = 0``."""
    return c * (ul if c > 0 else ur)


def riemann_solver_burgers(ul, ur):
    """Exact Riemann flux for the Burgers equation."""
    f = lambda v: 0.5 * v * v
    s = 0.5 * (ul + ur)
    return np.where(ul > ur,
                    np.where(s > 0, f(ul), f(ur)),
                    np.where(ul > 0, f(ul), np.where(ur < 0, f(ur), 0.0)))


def rusanov_flux(ul, ur, flux, wave_speed):
    """Rusanov (local Lax-Friedrichs) flux: robust for any convex flux."""
    a = np.maximum(np.abs(wave_speed(ul)), np.abs(wave_speed(ur)))
    return 0.5 * (flux(ul) + flux(ur)) - 0.5 * a * (ur - ul)


def hll_flux(ul, ur, flux, wave_speed):
    """HLL approximate Riemann solver using two-wave speed estimates."""
    sl = np.minimum(wave_speed(ul), wave_speed(ur))
    sr = np.maximum(wave_speed(ul), wave_speed(ur))
    fl, fr = flux(ul), flux(ur)
    out = np.where(sl >= 0, fl,
                   np.where(sr <= 0, fr,
                            (sr * fl - sl * fr + sl * sr * (ur - ul)) / (sr - sl + 1e-300)))
    return out


def fvm_1d_conservation(u0, flux, wave_speed, x_span, t_span, nx: int = 200,
                        nt: int = 400, numerical_flux: str = "rusanov",
                        bc: str = "periodic"):
    """Finite volume solver for ``u_t + f(u)_x = 0`` (first order in space).

    ``numerical_flux`` selects ``'rusanov'``, ``'hll'`` or ``'lax_friedrichs'``.
    """
    x_edges = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])          # cell centres
    dx = x_edges[1] - x_edges[0]
    t = TimeGrid(t_span[0], t_span[1], nt + 1)
    dt = t[1] - t[0]
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = Trajectory(t)
    U[0] = u

    def extend(v):
        if bc == "periodic":
            return np.roll(v, -1), np.roll(v, 1)
        right = np.concatenate([v[1:], [v[-1]]])     # outflow / transmissive
        left = np.concatenate([[v[0]], v[:-1]])
        return right, left

    for k in range(nt):
        if U.recorder.stopped:
            break
        right, _ = extend(u)
        if numerical_flux == "rusanov":
            F = rusanov_flux(u, right, flux, wave_speed)
        elif numerical_flux == "hll":
            F = hll_flux(u, right, flux, wave_speed)
        else:
            F = 0.5 * (flux(u) + flux(right)) - 0.5 * dx / dt * (right - u)
        F_left = np.roll(F, 1) if bc == "periodic" else np.concatenate([[F[0]], F[:-1]])
        u = u - dt / dx * (F - F_left)
        U[k + 1] = u
    return PDESolution(U, (x,), t, f"fvm_{numerical_flux}")


def muscl_reconstruct(u, limiter="minmod"):
    """MUSCL reconstruction: limited linear states at the two cell faces.

    Returns ``(u_left_of_face, u_right_of_face)`` for the face to the right of
    each cell, giving second-order accuracy away from extrema.
    """
    du_back = u - np.roll(u, 1)
    du_fwd = np.roll(u, -1) - u
    if limiter == "minmod":
        slope = minmod(du_back, du_fwd)
    elif limiter == "mc":
        slope = minmod(0.5 * (du_back + du_fwd), 2 * minmod(du_back, du_fwd))
    elif limiter == "superbee":
        s1 = minmod(du_fwd, 2 * du_back)
        s2 = minmod(2 * du_fwd, du_back)
        slope = np.where(np.abs(s1) > np.abs(s2), s1, s2)
    else:
        slope = np.zeros_like(u)
    uL = u + 0.5 * slope                     # right face, from the left cell
    uR = np.roll(u - 0.5 * slope, -1)        # right face, from the right cell
    return uL, uR


def fvm_muscl(u0, flux, wave_speed, x_span, t_span, nx: int = 200, nt: int = 400,
              limiter: str = "minmod"):
    """Second-order MUSCL-Hancock finite volume scheme with SSP-RK2 in time.

    Combines limited reconstruction with a strong-stability-preserving time
    step, so it stays non-oscillatory while being second order in smooth regions.
    """
    x_edges = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = x_edges[1] - x_edges[0]
    t = TimeGrid(t_span[0], t_span[1], nt + 1)
    dt = t[1] - t[0]
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = Trajectory(t)
    U[0] = u

    def L(v):
        uL, uR = muscl_reconstruct(v, limiter)
        F = rusanov_flux(uL, uR, flux, wave_speed)
        return -(F - np.roll(F, 1)) / dx

    for k in range(nt):
        if U.recorder.stopped:
            break
        u1 = u + dt * L(u)                   # SSP-RK2 (Heun): both stages convex
        u = 0.5 * (u + u1 + dt * L(u1))
        U[k + 1] = u
    return PDESolution(U, (x,), t, f"fvm_muscl_{limiter}")


def fvm_diffusion(u0, alpha: float, x_span, t_span, nx: int = 100, nt: int = 500,
                  bc=(0.0, 0.0)):
    """Finite volume discretization of the diffusion equation.

    Fluxes at the faces are the centred gradients, which makes the scheme
    conservative even on a non-uniform mesh.
    """
    x_edges = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = x_edges[1] - x_edges[0]
    t = TimeGrid(t_span[0], t_span[1], nt + 1)
    dt = t[1] - t[0]
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = Trajectory(t)
    U[0] = u
    for k in range(nt):
        if U.recorder.stopped:
            break
        ghost_l = 2 * bc[0] - u[0]           # Dirichlet through a ghost cell
        ghost_r = 2 * bc[1] - u[-1]
        ext = np.concatenate([[ghost_l], u, [ghost_r]])
        flux = -alpha * (ext[1:] - ext[:-1]) / dx     # face fluxes
        u = u - dt / dx * (flux[1:] - flux[:-1])
        U[k + 1] = u
    return PDESolution(U, (x,), t, "fvm_diffusion")


# Share output policy through nested method-of-lines and wrapper calls.
for _name in __all__:
    if "t_span" in __import__("inspect").signature(globals()[_name]).parameters:
        globals()[_name] = output_control(globals()[_name])
del _name
