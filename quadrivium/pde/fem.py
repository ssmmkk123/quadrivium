"""Finite element methods in one and two dimensions.

Assembles the weak form element by element, which is what lets the method
handle unstructured meshes and complicated geometry.
"""

from __future__ import annotations

import numpy as np

from ..core.types import PDESolution
from ..core.utils import as_vector

__all__ = [
    "fem_1d_linear",
    "fem_1d_quadratic",
    "fem_1d_mass_stiffness",
    "fem_2d_triangular",
    "unit_square_mesh",
    "assemble_1d",
    "fem_1d_time_dependent",
]


def fem_1d_mass_stiffness(nodes, order: int = 1):
    """Assemble the mass and stiffness matrices for 1-D Lagrange elements."""
    x = as_vector(nodes)
    n = x.size
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    for e in range(n - 1):
        h = x[e + 1] - x[e]
        ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        me = np.array([[2.0, 1.0], [1.0, 2.0]]) * h / 6.0
        idx = [e, e + 1]
        for a in range(2):
            for b in range(2):
                K[idx[a], idx[b]] += ke[a, b]
                M[idx[a], idx[b]] += me[a, b]
    return K, M


def assemble_1d(nodes, c_diff=1.0, c_react=0.0, source=None):
    """Assemble ``-(c u')' + r u = f`` on a 1-D mesh with linear elements."""
    x = as_vector(nodes)
    n = x.size
    A = np.zeros((n, n))
    b = np.zeros(n)
    for e in range(n - 1):
        h = x[e + 1] - x[e]
        xm = 0.5 * (x[e] + x[e + 1])
        c = c_diff(xm) if callable(c_diff) else c_diff
        r = c_react(xm) if callable(c_react) else c_react
        ke = c * np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        me = r * np.array([[2.0, 1.0], [1.0, 2.0]]) * h / 6.0
        idx = [e, e + 1]
        for a in range(2):
            for bb in range(2):
                A[idx[a], idx[bb]] += ke[a, bb] + me[a, bb]
        if source is not None:
            # two-point Gauss quadrature of f against the hat functions
            g = 1.0 / np.sqrt(3.0)
            for xi, w in [(-g, 1.0), (g, 1.0)]:
                xq = xm + 0.5 * h * xi
                N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
                b[idx] += w * 0.5 * h * source(xq) * N
    return A, b


def fem_1d_linear(source, x_span, bc=(0.0, 0.0), n: int = 50, c_diff=1.0,
                  c_react=0.0, nodes=None):
    """Solve ``-(c u')' + r u = f`` with linear elements and Dirichlet data.

    For the pure diffusion problem in one dimension the Galerkin solution is
    *nodally exact* -- the error vanishes at every node, and only the load
    quadrature limits accuracy. That superconvergence is why P1 can beat P2
    when compared at the nodes.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), n + 1) if nodes is None \
        else as_vector(nodes)
    A, b = assemble_1d(x, c_diff, c_react, source)
    # impose Dirichlet conditions by lifting the known values
    b -= A[:, 0] * bc[0] + A[:, -1] * bc[1]
    A_in = A[1:-1, 1:-1]
    u_in = np.linalg.solve(A_in, b[1:-1])
    u = np.concatenate([[bc[0]], u_in, [bc[1]]])
    return PDESolution(u, (x,), None, "fem_1d_linear")


def fem_1d_quadratic(source, x_span, bc=(0.0, 0.0), n: int = 25, c_diff=1.0,
                     c_react=0.0):
    """Solve with quadratic (P2) elements: one interior node per element."""
    n_nodes = 2 * n + 1
    x = np.linspace(float(x_span[0]), float(x_span[1]), n_nodes)
    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)
    gauss = [(-np.sqrt(3 / 5), 5 / 9), (0.0, 8 / 9), (np.sqrt(3 / 5), 5 / 9)]
    for e in range(n):
        idx = [2 * e, 2 * e + 1, 2 * e + 2]
        h = x[idx[2]] - x[idx[0]]
        xm = 0.5 * (x[idx[0]] + x[idx[2]])
        c = c_diff(xm) if callable(c_diff) else c_diff
        r = c_react(xm) if callable(c_react) else c_react
        for xi, w in gauss:
            # P2 shape functions and derivatives on the reference element
            N = np.array([0.5 * xi * (xi - 1), 1 - xi * xi, 0.5 * xi * (xi + 1)])
            dN = np.array([xi - 0.5, -2 * xi, xi + 0.5]) * (2.0 / h)
            jac = 0.5 * h
            for a in range(3):
                for bb in range(3):
                    A[idx[a], idx[bb]] += w * jac * (c * dN[a] * dN[bb] + r * N[a] * N[bb])
                if source is not None:
                    b[idx[a]] += w * jac * source(xm + 0.5 * h * xi) * N[a]
    b -= A[:, 0] * bc[0] + A[:, -1] * bc[1]
    u_in = np.linalg.solve(A[1:-1, 1:-1], b[1:-1])
    u = np.concatenate([[bc[0]], u_in, [bc[1]]])
    return PDESolution(u, (x,), None, "fem_1d_quadratic")


def unit_square_mesh(n: int = 8, x_span=(0.0, 1.0), y_span=(0.0, 1.0)):
    """Triangulate a rectangle into ``2 n^2`` right triangles.

    Returns ``(points, triangles, boundary_indices)``.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), n + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), n + 1)
    pts = np.array([[xi, yj] for j, yj in enumerate(y) for i, xi in enumerate(x)])
    tris = []
    for j in range(n):
        for i in range(n):
            k = j * (n + 1) + i
            tris.append([k, k + 1, k + n + 2])
            tris.append([k, k + n + 2, k + n + 1])
    tris = np.array(tris, dtype=int)
    on_bdry = np.zeros(len(pts), dtype=bool)
    tol = 1e-12
    on_bdry |= np.abs(pts[:, 0] - x[0]) < tol
    on_bdry |= np.abs(pts[:, 0] - x[-1]) < tol
    on_bdry |= np.abs(pts[:, 1] - y[0]) < tol
    on_bdry |= np.abs(pts[:, 1] - y[-1]) < tol
    return pts, tris, np.flatnonzero(on_bdry)


def fem_2d_triangular(source, points=None, triangles=None, boundary=None,
                      n: int = 8, bc=0.0, c_diff=1.0):
    """P1 finite elements on a triangular mesh for ``-c lap u = f``.

    Uses the closed-form gradients of the linear shape functions on each
    triangle, so element assembly needs no quadrature for the stiffness matrix.
    """
    if points is None:
        points, triangles, boundary = unit_square_mesh(n)
    P = np.asarray(points, dtype=float)
    T = np.asarray(triangles, dtype=int)
    npt = len(P)
    A = np.zeros((npt, npt))
    b = np.zeros(npt)
    # Every element contributes the same closed-form 3x3 stiffness matrix, so
    # all of them are formed at once and scattered in one pass. The per-element
    # determinant, two outer products and 3x3 scatter loop that this replaces
    # were the whole cost of assembly.
    p = P[T]                                       # (elements, 3, 2)
    # Twice the signed area, from the barycentric coordinate determinant.
    two_area = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    area = 0.5 * np.abs(two_area)
    live = area >= 1e-300
    if np.any(live):
        p, area, T_live = p[live], area[live], T[live]
        beta = np.stack([p[:, 1, 1] - p[:, 2, 1],
                         p[:, 2, 1] - p[:, 0, 1],
                         p[:, 0, 1] - p[:, 1, 1]], axis=1)
        gamma = np.stack([p[:, 2, 0] - p[:, 1, 0],
                          p[:, 0, 0] - p[:, 2, 0],
                          p[:, 1, 0] - p[:, 0, 0]], axis=1)
        centroid = p.mean(axis=1)
        if callable(c_diff):
            c = np.array([c_diff(cx, cy) for cx, cy in centroid])
        else:
            c = np.full(area.shape, float(c_diff))
        if callable(source):
            fval = np.asarray(source(centroid[:, 0], centroid[:, 1]), dtype=float)
            fval = np.broadcast_to(np.atleast_1d(fval), area.shape)
        else:
            fval = np.full(area.shape, float(source))
        ke = ((beta[:, :, None] * beta[:, None, :]
               + gamma[:, :, None] * gamma[:, None, :])
              * (c / (4.0 * area))[:, None, None])
        rows = np.repeat(T_live, 3, axis=1).ravel()
        cols = np.tile(T_live, (1, 3)).ravel()
        np.add.at(A, (rows, cols), ke.ravel())
        np.add.at(b, T_live.ravel(), np.repeat(fval * area / 3.0, 3))
    bdry = np.asarray(boundary, dtype=int)
    bc_fun = bc if callable(bc) else (lambda x, y: bc)
    u = np.zeros(npt)
    u[bdry] = np.array([bc_fun(P[k, 0], P[k, 1]) for k in bdry])
    interior = np.setdiff1d(np.arange(npt), bdry)
    rhs = b[interior] - A[np.ix_(interior, bdry)] @ u[bdry]
    u[interior] = np.linalg.solve(A[np.ix_(interior, interior)], rhs)
    return PDESolution(u, (P[:, 0], P[:, 1]), None, "fem_2d_triangular")


def fem_1d_time_dependent(u0, source, x_span, t_span, n: int = 50, nt: int = 100,
                          bc=(0.0, 0.0), c_diff=1.0, theta: float = 0.5):
    """Finite elements in space, theta method in time, for ``u_t = (c u')' + f``.

    ``source`` may be ``f(x)`` for a steady load or ``f(x, t)`` for a genuinely
    time-dependent one; the arity is detected once and the load vector is then
    reassembled each step.  The two loads are combined as
    ``theta*f(t_{k+1}) + (1-theta)*f(t_k)``, matching the time weighting of the
    operator -- using a single load would silently drop Crank-Nicolson
    (``theta=0.5``) back to first order in time.

    ``c_diff`` may be a constant or a callable ``c(x)``; a variable coefficient
    is integrated element by element rather than frozen at one sample point.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), n + 1)
    t = np.linspace(float(t_span[0]), float(t_span[1]), nt + 1)
    dt = t[1] - t[0]
    # assemble_1d handles a variable c(x) element-wise; the pure mass matrix
    # comes from the reaction term with unit coefficient.
    K, _ = assemble_1d(x, c_diff, 0.0)
    _, M = fem_1d_mass_stiffness(x)
    u = np.array([u0(xi) for xi in x]) if callable(u0) else as_vector(u0).copy()
    U = np.empty((nt + 1, n + 1))
    U[0] = u
    LHS = M + theta * dt * K
    RHS = M - (1 - theta) * dt * K
    time_dependent = source is not None and _accepts_time(source)

    def load(tk):
        if source is None:
            return np.zeros(n + 1)
        g = (lambda xx: source(xx, tk)) if time_dependent else source
        return assemble_1d(x, 0.0, 0.0, g)[1]

    f_old = load(t[0])
    idx = np.arange(1, n)
    A_in = LHS[np.ix_(idx, idx)]
    lu = np.linalg.inv(A_in) if n > 1 else None
    for k in range(nt):
        f_new = load(t[k + 1]) if time_dependent else f_old
        rhs = RHS @ u + dt * (theta * f_new + (1.0 - theta) * f_old)
        rhs_in = rhs[idx] - LHS[np.ix_(idx, [0, n])] @ np.array([bc[0], bc[1]])
        u_in = lu @ rhs_in
        u = np.concatenate([[bc[0]], u_in, [bc[1]]])
        U[k + 1] = u
        f_old = f_new
    return PDESolution(U, (x,), t, "fem_1d_time_dependent")


def _accepts_time(fn) -> bool:
    """True when ``fn`` takes a second (time) argument."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins and C callables
        return False
    required = [pr for pr in sig.parameters.values()
                if pr.kind in (pr.POSITIONAL_ONLY, pr.POSITIONAL_OR_KEYWORD)
                and pr.default is pr.empty]
    if any(pr.kind is pr.VAR_POSITIONAL for pr in sig.parameters.values()):
        return False
    return len(required) >= 2
