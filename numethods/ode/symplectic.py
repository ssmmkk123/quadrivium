"""Geometric integrators for Hamiltonian and separable systems.

Symplectic methods preserve the phase-space volume form, so the energy error
stays bounded over exponentially long times instead of drifting -- which is why
they dominate in celestial mechanics and molecular dynamics.
"""

from __future__ import annotations

import numpy as np

from ..core.types import ODESolution
from ..core.utils import CountedFunction, as_vector

__all__ = [
    "symplectic_euler",
    "velocity_verlet",
    "position_verlet",
    "leapfrog",
    "stormer_verlet",
    "ruth3",
    "yoshida4",
    "forest_ruth",
    "pefrl",
    "hamiltonian_flow",
    "energy_drift",
]


def _run_separable(dHdq, dHdp, t_span, q0, p0, n, coeffs, name):
    """Run a splitting method given its ``(c, d)`` coefficient pairs.

    ``coeffs`` is a list of ``(c_i, d_i)``: ``c_i`` advances position with the
    current momentum, ``d_i`` advances momentum with the force at the new
    position.
    """
    q = as_vector(q0).copy()
    p = as_vector(p0).copy()
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    qs = np.empty((n + 1, q.size))
    ps = np.empty((n + 1, p.size))
    qs[0], ps[0] = q, p
    calls = 0
    for i in range(n):
        for c, d in coeffs:
            if c != 0.0:
                q = q + c * h * dHdp(p)
            if d != 0.0:
                p = p - d * h * dHdq(q)
                calls += 1
        qs[i + 1], ps[i + 1] = q, p
    ys = np.hstack([qs, ps])
    return ODESolution(ts, ys, name, n, n, 0, calls, True, "completed")


def symplectic_euler(dHdq, dHdp, t_span, q0, p0, n: int = 1000):
    """Symplectic (semi-implicit) Euler: order 1, exactly symplectic."""
    return _run_separable(dHdq, dHdp, t_span, q0, p0, n, [(1.0, 1.0)],
                          "symplectic_euler")


def velocity_verlet(force, t_span, q0, v0, n: int = 1000, mass=1.0):
    """Velocity Verlet: order 2, symplectic and time-reversible.

    ``force(q)`` returns the force (that is, ``-dV/dq``).
    """
    q = as_vector(q0).copy()
    v = as_vector(v0).copy()
    m = np.asarray(mass, dtype=float)
    t0, tf = float(t_span[0]), float(t_span[1])
    h = (tf - t0) / n
    ts = np.linspace(t0, tf, n + 1)
    qs = np.empty((n + 1, q.size))
    vs = np.empty((n + 1, v.size))
    qs[0], vs[0] = q, v
    fc = CountedFunction(lambda x: as_vector(force(x)))
    a = fc(q) / m
    for i in range(n):
        q = q + h * v + 0.5 * h * h * a
        a_new = fc(q) / m
        v = v + 0.5 * h * (a + a_new)
        a = a_new
        qs[i + 1], vs[i + 1] = q, v
    return ODESolution(ts, np.hstack([qs, vs]), "velocity_verlet", n, n, 0,
                       fc.calls, True, "completed")


def position_verlet(dHdq, dHdp, t_span, q0, p0, n: int = 1000):
    """Position Verlet: the drift-kick-drift splitting, order 2."""
    return _run_separable(dHdq, dHdp, t_span, q0, p0, n,
                          [(0.5, 1.0), (0.5, 0.0)], "position_verlet")


def leapfrog(dHdq, dHdp, t_span, q0, p0, n: int = 1000):
    """Leapfrog (kick-drift-kick), order 2 and symplectic."""
    return _run_separable(dHdq, dHdp, t_span, q0, p0, n,
                          [(0.0, 0.5), (1.0, 0.5)], "leapfrog")


def stormer_verlet(force, t_span, q0, v0, n: int = 1000, mass=1.0):
    """Stormer-Verlet, in the velocity form."""
    res = velocity_verlet(force, t_span, q0, v0, n, mass)
    res.method = "stormer_verlet"
    return res


def ruth3(dHdq, dHdp, t_span, q0, p0, n: int = 1000):
    """Ruth's third-order symplectic integrator."""
    c = [1.0, -2.0 / 3.0, 2.0 / 3.0]
    d = [-1.0 / 24.0, 3.0 / 4.0, 7.0 / 24.0]
    return _run_separable(dHdq, dHdp, t_span, q0, p0, n, list(zip(c, d)), "ruth3")


def forest_ruth(dHdq, dHdp, t_span, q0, p0, n: int = 1000):
    """Forest-Ruth fourth-order symplectic integrator."""
    theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c = [theta / 2, (1 - theta) / 2, (1 - theta) / 2, theta / 2]
    d = [theta, 1 - 2 * theta, theta, 0.0]
    return _run_separable(dHdq, dHdp, t_span, q0, p0, n, list(zip(c, d)), "forest_ruth")


def yoshida4(dHdq, dHdp, t_span, q0, p0, n: int = 1000):
    """Yoshida's fourth-order method: three Verlet steps composed."""
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = -(2.0 ** (1.0 / 3.0)) * w1
    c = [w1 / 2, (w1 + w0) / 2, (w0 + w1) / 2, w1 / 2]
    d = [w1, w0, w1, 0.0]
    return _run_separable(dHdq, dHdp, t_span, q0, p0, n, list(zip(c, d)), "yoshida4")


def pefrl(dHdq, dHdp, t_span, q0, p0, n: int = 1000):
    """Position-extended Forest-Ruth-like: order 4 with a small error constant."""
    xi = 0.1786178958448091
    lam = -0.2123418310626054
    chi = -0.06626458266981849
    c = [xi, chi, 1 - 2 * (chi + xi), chi, xi]
    d = [0.5 * (1 - 2 * lam), lam, lam, 0.5 * (1 - 2 * lam), 0.0]
    return _run_separable(dHdq, dHdp, t_span, q0, p0, n, list(zip(c, d)), "pefrl")


def hamiltonian_flow(H_q, H_p, t_span, q0, p0, n: int = 1000, method: str = "yoshida4"):
    """Integrate Hamilton's equations with the chosen symplectic method."""
    methods = {
        "symplectic_euler": symplectic_euler, "leapfrog": leapfrog,
        "position_verlet": position_verlet, "ruth3": ruth3,
        "forest_ruth": forest_ruth, "yoshida4": yoshida4, "pefrl": pefrl,
    }
    return methods[method](H_q, H_p, t_span, q0, p0, n)


def energy_drift(solution, energy):
    """Relative energy drift along a trajectory: ``max |E(t) - E(0)| / |E(0)|``."""
    E = np.array([energy(y) for y in solution.y])
    E0 = E[0]
    return float(np.max(np.abs(E - E0)) / (abs(E0) if E0 != 0 else 1.0))
