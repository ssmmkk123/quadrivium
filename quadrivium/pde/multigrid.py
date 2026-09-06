"""Multigrid: the optimal-complexity solvers for elliptic problems.

Relaxation kills high-frequency error fast but stalls on smooth error; coarser
grids see that smooth error as oscillatory. Cycling between grids therefore
removes every frequency at the same rate, giving ``O(N)`` overall work.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.types import PDESolution

__all__ = [
    "restrict",
    "prolong",
    "smooth",
    "residual",
    "v_cycle",
    "w_cycle",
    "full_multigrid",
    "multigrid_solve",
]


def smooth(u, f, h: float, iterations: int = 2, omega: float = 1.0,
           method: str = "gauss_seidel"):
    """Relax ``lap u = f`` in place with weighted Jacobi or red-black Gauss-Seidel.

    ``omega = 2/3`` is the optimal damping for weighted Jacobi as a smoother.
    """
    u = u.copy()
    hh = h * h
    if method == "jacobi":
        for _ in range(iterations):
            u_old = u.copy()
            u[1:-1, 1:-1] = (1 - omega) * u_old[1:-1, 1:-1] + omega * 0.25 * (
                u_old[2:, 1:-1] + u_old[:-2, 1:-1] + u_old[1:-1, 2:] + u_old[1:-1, :-2]
                - hh * f[1:-1, 1:-1])
        return u
    # Red-black ordering: vectorized, and a better smoother than lexicographic.
    # Each colour is two strided sub-grids -- (odd, odd) and (even, even) rows
    # and columns for one colour, the mixed pair for the other -- so a
    # half-sweep is four slice reads and one slice write over exactly the
    # points being updated. Selecting the colour with a boolean mask instead
    # computes the stencil at every interior point and then gathers half of
    # them back out through random access.
    rows, cols = u.shape
    for _ in range(iterations):
        for color in (0, 1):
            for a in (0, 1):
                b = a ^ color
                r = slice(1 + a, rows - 1, 2)
                rm = slice(a, rows - 2, 2)
                rp = slice(2 + a, rows, 2)
                c = slice(1 + b, cols - 1, 2)
                cm = slice(b, cols - 2, 2)
                cp = slice(2 + b, cols, 2)
                nb = (u[rp, c] + u[rm, c] + u[r, cp] + u[r, cm] - hh * f[r, c])
                u[r, c] = (1 - omega) * u[r, c] + omega * 0.25 * nb
    return u


def residual(u, f, h: float):
    """Residual ``r = f - lap u`` on the interior (zero on the boundary)."""
    r = np.zeros_like(u)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
    ) / (h * h)
    return r


def restrict(r):
    """Full-weighting restriction to the next coarser grid."""
    n = (r.shape[0] - 1) // 2
    m = (r.shape[1] - 1) // 2
    out = np.zeros((n + 1, m + 1))
    f = r[2:-1:2, 2:-1:2]
    out[1:-1, 1:-1] = 0.25 * f + 0.125 * (
        r[1:-2:2, 2:-1:2] + r[3::2, 2:-1:2] + r[2:-1:2, 1:-2:2] + r[2:-1:2, 3::2]
    ) + 0.0625 * (
        r[1:-2:2, 1:-2:2] + r[3::2, 1:-2:2] + r[1:-2:2, 3::2] + r[3::2, 3::2]
    )
    return out


def prolong(e):
    """Bilinear interpolation to the next finer grid."""
    n = (e.shape[0] - 1) * 2
    m = (e.shape[1] - 1) * 2
    out = np.zeros((n + 1, m + 1))
    out[::2, ::2] = e                                    # coincident points
    out[1::2, ::2] = 0.5 * (e[:-1, :] + e[1:, :])        # midpoints in x
    out[::2, 1::2] = 0.5 * (e[:, :-1] + e[:, 1:])        # midpoints in y
    out[1::2, 1::2] = 0.25 * (e[:-1, :-1] + e[1:, :-1]
                              + e[:-1, 1:] + e[1:, 1:])  # cell centres
    return out


def _coarse_solve(u, f, h, iterations: int = 50):
    """Solve the coarsest problem by brute-force relaxation."""
    return smooth(u, f, h, iterations)


def v_cycle(u, f, h: float, nu1: int = 2, nu2: int = 2, level: int = 0,
            max_level: int = 20, omega: float = 1.0):
    """One multigrid V-cycle: smooth, restrict, recurse, prolong, smooth."""
    if u.shape[0] <= 4 or level >= max_level:
        return _coarse_solve(u, f, h)
    u = smooth(u, f, h, nu1, omega)              # pre-smoothing
    r = residual(u, f, h)
    rc = restrict(r)
    ec = np.zeros_like(rc)
    ec = v_cycle(ec, rc, 2 * h, nu1, nu2, level + 1, max_level, omega)
    u = u + prolong(ec)                          # coarse-grid correction
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    return smooth(u, f, h, nu2, omega)           # post-smoothing


def w_cycle(u, f, h: float, nu1: int = 2, nu2: int = 2, level: int = 0,
            max_level: int = 20, omega: float = 1.0):
    """W-cycle: two coarse-grid visits per level, more robust than a V-cycle."""
    if u.shape[0] <= 4 or level >= max_level:
        return _coarse_solve(u, f, h)
    u = smooth(u, f, h, nu1, omega)
    r = residual(u, f, h)
    rc = restrict(r)
    ec = np.zeros_like(rc)
    for _ in range(2):
        ec = w_cycle(ec, rc, 2 * h, nu1, nu2, level + 1, max_level, omega)
    u = u + prolong(ec)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    return smooth(u, f, h, nu2, omega)


def full_multigrid(f, h: float, nu1: int = 2, nu2: int = 2, cycles: int = 1):
    """Full multigrid (FMG): start on the coarsest grid and work upward.

    Delivers a solution accurate to discretization error in ``O(N)`` work, often
    from a single pass.
    """
    if f.shape[0] <= 4:
        return _coarse_solve(np.zeros_like(f), f, h)
    fc = restrict(f)
    uc = full_multigrid(fc, 2 * h, nu1, nu2, cycles)
    u = prolong(uc)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    for _ in range(cycles):
        u = v_cycle(u, f, h, nu1, nu2)
    return u


def multigrid_solve(f_func, x_span, y_span, n: int = 64, tol: float = 1e-10,
                    max_cycles: int = 50, cycle: str = "v", nu1: int = 2,
                    nu2: int = 2, bc: float = 0.0):
    """Solve ``lap u = f`` on a square grid by repeated multigrid cycles.

    ``n`` must be a power of two so the grid coarsens exactly.
    """
    if n & (n - 1) != 0:
        raise ValueError("n must be a power of two for the grid hierarchy")
    x = np.linspace(float(x_span[0]), float(x_span[1]), n + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), n + 1)
    h = x[1] - x[0]
    F = np.array([[f_func(xi, yj) for yj in y] for xi in x])
    u = np.zeros((n + 1, n + 1))
    cycler = v_cycle if cycle == "v" else w_cycle
    history = []
    r0 = np.linalg.norm(residual(u, F, h))
    for k in range(1, max_cycles + 1):
        u = cycler(u, F, h, nu1, nu2)
        rn = np.linalg.norm(residual(u, F, h))
        history.append(rn / (r0 if r0 > 0 else 1.0))
        if history[-1] < tol:
            return PDESolution(u, (x, y), None, f"multigrid_{cycle}", k, True, history)
    return PDESolution(u, (x, y), None, f"multigrid_{cycle}", max_cycles, False, history)
