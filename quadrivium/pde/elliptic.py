"""Elliptic PDEs: Laplace and Poisson problems on rectangular grids."""

from __future__ import annotations

import numpy as np

from .. import _accel

from ..core.types import PDESolution
from ..core.utils import as_vector
from ..linalg.direct import block_tridiagonal_solve

__all__ = [
    "poisson_2d_direct",
    "poisson_2d_iterative",
    "laplace_2d",
    "poisson_1d",
    "poisson_9point",
    "poisson_neumann",
    "helmholtz_2d",
    "laplacian_matrix",
    "poisson_fft",
]


def laplacian_matrix(nx: int, ny: int = None, dx: float = 1.0, dy: float = None,
                     stencil: int = 5):
    """Sparse-pattern Laplacian on a grid of interior points (Dirichlet).

    Returns the dense matrix of the 5-point (or 9-point) stencil.
    """
    if ny is None:
        A = (np.diag(-2.0 * np.ones(nx)) + np.diag(np.ones(nx - 1), 1)
             + np.diag(np.ones(nx - 1), -1)) / dx**2
        return A
    dy = dx if dy is None else dy
    n = nx * ny
    A = np.zeros((n, n))
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if stencil == 5:
                A[k, k] = -2.0 / dx**2 - 2.0 / dy**2
                if i > 0:
                    A[k, k - 1] = 1.0 / dx**2
                if i < nx - 1:
                    A[k, k + 1] = 1.0 / dx**2
                if j > 0:
                    A[k, k - nx] = 1.0 / dy**2
                if j < ny - 1:
                    A[k, k + nx] = 1.0 / dy**2
            else:  # 9-point, fourth-order accurate on a uniform grid
                h2 = dx * dx
                A[k, k] = -20.0 / (6 * h2)
                for di, dj, w in [(-1, 0, 4), (1, 0, 4), (0, -1, 4), (0, 1, 4),
                                  (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)]:
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        A[k, jj * nx + ii] = w / (6 * h2)
    return A


def poisson_1d(f, x_span, bc=(0.0, 0.0), nx: int = 100):
    """Solve ``u'' = f(x)`` with Dirichlet conditions, by the Thomas algorithm."""
    from ..linalg.direct import thomas

    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    h = x[1] - x[0]
    m = nx - 1
    rhs = np.array([f(xi) * h * h for xi in x[1:-1]])
    rhs[0] -= bc[0]
    rhs[-1] -= bc[1]
    u_in = thomas(np.ones(m - 1), np.full(m, -2.0), np.ones(m - 1), rhs)
    u = np.concatenate([[bc[0]], u_in, [bc[1]]])
    return PDESolution(u, (x,), None, "poisson_1d")


def poisson_2d_direct(f, x_span, y_span, nx: int = 40, ny: int = 40, bc=0.0,
                      stencil: int = 5):
    """Solve ``u_xx + u_yy = f`` by forming and factorizing the linear system.

    ``stencil=5`` is the standard second-order scheme. ``stencil=9`` is the
    Mehrstellen scheme, which reaches fourth order only when the right-hand
    side carries the ``h^2/12 * laplacian(f)`` correction -- included here, and
    requiring a square grid (``dx == dy``).

    Exact up to round-off, but the ``O(N^3)`` factorization limits it to modest
    grids; use :func:`poisson_2d_iterative` or multigrid for larger ones.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    bc_fun = bc if callable(bc) else (lambda xx, yy: bc)
    mi, mj = nx - 1, ny - 1
    if stencil == 5:
        weights = [(0, 0, -2.0 / dx**2 - 2.0 / dy**2),
                   (-1, 0, 1.0 / dx**2), (1, 0, 1.0 / dx**2),
                   (0, -1, 1.0 / dy**2), (0, 1, 1.0 / dy**2)]
    elif stencil == 9:
        if abs(dx - dy) > 1e-12 * max(dx, dy):
            raise ValueError("the 9-point Mehrstellen stencil requires dx == dy")
        h2 = dx * dx
        weights = [(0, 0, -20.0 / (6 * h2))]
        weights += [(di, dj, 4.0 / (6 * h2)) for di, dj in
                    ((-1, 0), (1, 0), (0, -1), (0, 1))]
        weights += [(di, dj, 1.0 / (6 * h2)) for di, dj in
                    ((-1, -1), (1, -1), (-1, 1), (1, 1))]
    else:
        raise ValueError("stencil must be 5 or 9")

    def rhs_value(i, j):
        """Right-hand side at interior node (i, j), with the 9-point correction."""
        val = f(x[i], y[j])
        if stencil == 9:
            hf = dx
            lap_f = ((f(x[i] + hf, y[j]) - 2 * val + f(x[i] - hf, y[j])) / hf**2
                     + (f(x[i], y[j] + hf) - 2 * val + f(x[i], y[j] - hf)) / hf**2)
            val = val + hf * hf / 12.0 * lap_f
        return val

    # Ordering the unknowns with i fastest makes the system block-tridiagonal:
    # every stencil offset moves j by at most one, so a node in row j couples
    # only to rows j-1, j and j+1. Storing those blocks rather than the whole
    # matrix is O(N * nx) instead of O(N^2) -- 8 MB rather than 97 MB on a
    # 60x60 grid -- and the block solve is O(mj * mi^3) rather than O(N^3).
    B = [np.zeros((mi, mi)) for _ in range(mj)]
    A_sub = [np.zeros((mi, mi)) for _ in range(max(mj - 1, 0))]
    C_sup = [np.zeros((mi, mi)) for _ in range(max(mj - 1, 0))]
    b = np.zeros(mi * mj)
    for j in range(1, ny):
        jb = j - 1
        for i in range(1, nx):
            k = jb * mi + (i - 1)
            b[k] = rhs_value(i, j)
            for di, dj, w in weights:
                ii, jj = i + di, j + dj
                if 1 <= ii <= nx - 1 and 1 <= jj <= ny - 1:
                    block = B[jb] if dj == 0 else (A_sub[jb - 1] if dj < 0
                                                   else C_sup[jb])
                    block[i - 1, ii - 1] += w
                else:
                    b[k] -= w * bc_fun(x[ii], y[jj])   # known boundary value
    sol = block_tridiagonal_solve(
        A_sub, B, C_sup, [b[jb * mi : (jb + 1) * mi] for jb in range(mj)])
    U = np.zeros((nx + 1, ny + 1))
    for j in range(1, ny):
        for i in range(1, nx):
            U[i, j] = sol[(j - 1) * mi + (i - 1)]
    for i in range(nx + 1):
        U[i, 0] = bc_fun(x[i], y[0])
        U[i, -1] = bc_fun(x[i], y[-1])
    for j in range(ny + 1):
        U[0, j] = bc_fun(x[0], y[j])
        U[-1, j] = bc_fun(x[-1], y[j])
    return PDESolution(U, (x, y), None, f"poisson_2d_direct_{stencil}pt")


def poisson_2d_iterative(f, x_span, y_span, nx: int = 40, ny: int = 40, bc=0.0,
                         method: str = "sor", omega=None, tol: float = 1e-10,
                         max_iter: int = 20000):
    """Solve the Poisson equation by a stationary iteration on the grid.

    ``method`` is ``'jacobi'``, ``'gauss_seidel'``, ``'sor'`` or ``'cg'``.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    bc_fun = bc if callable(bc) else (lambda xx, yy: bc)
    U = np.zeros((nx + 1, ny + 1))
    for i in range(nx + 1):
        U[i, 0] = bc_fun(x[i], y[0])
        U[i, -1] = bc_fun(x[i], y[-1])
    for j in range(ny + 1):
        U[0, j] = bc_fun(x[0], y[j])
        U[-1, j] = bc_fun(x[-1], y[j])
    F = np.array([[f(xi, yj) for yj in y] for xi in x])
    if omega is None:
        # optimal SOR factor for the model problem on a rectangle
        rho = 0.5 * (np.cos(np.pi / nx) + np.cos(np.pi / ny))
        omega = 2.0 / (1.0 + np.sqrt(1.0 - rho**2))
    beta2 = dx**2 / dy**2
    denom = 2.0 * (1.0 + beta2)
    residuals = []
    if method == "cg":
        from ..linalg.iterative import conjugate_gradient

        A = -laplacian_matrix(nx - 1, ny - 1, dx, dy)
        b = np.zeros((nx - 1) * (ny - 1))
        for j in range(ny - 1):
            for i in range(nx - 1):
                k = j * (nx - 1) + i
                b[k] = -F[i + 1, j + 1]
                if i == 0:
                    b[k] += U[0, j + 1] / dx**2
                if i == nx - 2:
                    b[k] += U[-1, j + 1] / dx**2
                if j == 0:
                    b[k] += U[i + 1, 0] / dy**2
                if j == ny - 2:
                    b[k] += U[i + 1, -1] / dy**2
        res = conjugate_gradient(A, b, tol=tol)
        for j in range(ny - 1):
            for i in range(nx - 1):
                U[i + 1, j + 1] = res.x[j * (nx - 1) + i]
        return PDESolution(U, (x, y), None, "poisson_cg", res.iterations,
                           res.converged, res.residuals)
    if method != "jacobi":
        # Lexicographic Gauss-Seidel and SOR read values written earlier in the
        # same sweep, so neither can be expressed as an array update; the
        # reference below is a point-by-point loop. The compiled kernel runs
        # the whole iteration, not one sweep, because a converging solve is
        # tens of thousands of sweeps.
        w = 1.0 if method == "gauss_seidel" else omega
        fast = _accel.kernel("sor_poisson")
        if fast is not None:
            U = np.ascontiguousarray(U)
            it, conv, res = fast(U, np.ascontiguousarray(F), beta2, dx**2, w,
                                 tol, max_iter)
            return PDESolution(U, (x, y), None, f"poisson_{method}", it, conv,
                               list(res))
        for it in range(1, max_iter + 1):
            U_old = U.copy()
            for i in range(1, nx):
                for j in range(1, ny):
                    new = ((U[i + 1, j] + U[i - 1, j]
                            + beta2 * (U[i, j + 1] + U[i, j - 1])
                            - dx**2 * F[i, j]) / denom)
                    U[i, j] = (1 - w) * U[i, j] + w * new
            change = np.max(np.abs(U - U_old))
            residuals.append(change)
            if change < tol:
                return PDESolution(U, (x, y), None, f"poisson_{method}", it, True,
                                   residuals)
        return PDESolution(U, (x, y), None, f"poisson_{method}", max_iter, False,
                           residuals)
    for it in range(1, max_iter + 1):
        U_old = U.copy()
        U[1:-1, 1:-1] = ((U_old[2:, 1:-1] + U_old[:-2, 1:-1]
                          + beta2 * (U_old[1:-1, 2:] + U_old[1:-1, :-2])
                          - dx**2 * F[1:-1, 1:-1]) / denom)
        change = np.max(np.abs(U - U_old))
        residuals.append(change)
        if change < tol:
            return PDESolution(U, (x, y), None, f"poisson_{method}", it, True, residuals)
    return PDESolution(U, (x, y), None, f"poisson_{method}", max_iter, False, residuals)


def laplace_2d(x_span, y_span, nx: int = 40, ny: int = 40, bc=0.0, **kwargs):
    """Solve Laplace's equation ``u_xx + u_yy = 0``."""
    return poisson_2d_direct(lambda x, y: 0.0, x_span, y_span, nx, ny, bc, **kwargs)


def poisson_9point(f, x_span, y_span, nx: int = 40, ny: int = 40, bc=0.0):
    """Fourth-order accurate 9-point ("Mehrstellen") Poisson solver."""
    return poisson_2d_direct(f, x_span, y_span, nx, ny, bc, stencil=9)


def poisson_neumann(f, x_span, y_span, nx: int = 40, ny: int = 40,
                    tol: float = 1e-10, max_iter: int = 50000,
                    method: str = "dct", omega: float = None):
    """Poisson with homogeneous Neumann conditions on all four sides.

    The solution is unique only up to a constant, and exists only if the source
    integrates to zero, so ``f`` is projected onto that constraint and the
    result normalized to zero mean.

    ``method="dct"`` (default) diagonalizes the operator with a cosine
    transform and is exact up to round-off; ``method="sor"`` runs the
    relaxation sweep instead.  Both are second-order accurate: the boundary
    rows use the *ghost point* form of the condition, reflecting ``u`` about
    the edge so that the interior stencil applies unchanged at the boundary.
    Setting ``U[0] = U[1]`` instead -- a one-sided difference -- would satisfy
    ``u_n = 0`` only to first order and drag the whole solution down with it.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    F = np.array([[f(xi, yj) for yj in y] for xi in x])
    F = _project_compatible(F)   # enforce the solvability condition
    if method == "dct":
        U = _neumann_dct_solve(F, dx, dy)
        return PDESolution(U, (x, y), None, "poisson_neumann_dct", 1, True, [])
    if method != "sor":
        raise ValueError("method must be 'dct' or 'sor'")
    beta2 = dx**2 / dy**2
    denom = 2.0 * (1.0 + beta2)
    if omega is None:  # Young's estimate for the optimal SOR factor
        rho = 0.5 * (np.cos(np.pi / nx) + np.cos(np.pi / ny))
        omega = 2.0 / (1.0 + np.sqrt(max(1.0 - rho * rho, 1e-16)))
    U = np.zeros((nx + 1, ny + 1))
    I, J = np.indices(U.shape)
    colors = ((I + J) % 2 == 0, (I + J) % 2 == 1)

    def sweep_target(V):
        # 'reflect' padding is exactly the ghost-point condition: the layer
        # outside the left edge is a copy of the row one step inside it.
        P = np.pad(V, 1, mode="reflect")
        return ((P[2:, 1:-1] + P[:-2, 1:-1]
                 + beta2 * (P[1:-1, 2:] + P[1:-1, :-2])
                 - dx**2 * F) / denom)

    residuals = []
    for it in range(1, max_iter + 1):
        U_old = U.copy()
        # Red-black ordering: each half-sweep sees the other colour already
        # updated, which is what makes this Gauss-Seidel.  Over-relaxing a
        # *Jacobi* sweep with omega > 1 would diverge instead.
        for mask in colors:
            U = np.where(mask, U + omega * (sweep_target(U) - U), U)
        U = U - U.mean()         # pin the additive constant
        change = float(np.max(np.abs(U - U_old)))
        residuals.append(change)
        if change < tol:
            return PDESolution(U, (x, y), None, "poisson_neumann", it, True, residuals)
    return PDESolution(U, (x, y), None, "poisson_neumann", max_iter, False, residuals)


def _project_compatible(F: np.ndarray) -> np.ndarray:
    """Remove the mean of ``F`` under the trapezoid weights of the grid.

    The Neumann problem is solvable only when the source integrates to zero.
    The correct projection uses the same quadrature weights the discretization
    implies (half on edges, quarter at corners) -- subtracting a plain
    arithmetic mean leaves a residual constant that the solver cannot absorb.
    """
    wx = np.ones(F.shape[0])
    wx[0] = wx[-1] = 0.5
    wy = np.ones(F.shape[1])
    wy[0] = wy[-1] = 0.5
    W = np.outer(wx, wy)
    return F - np.sum(W * F) / np.sum(W)


def _neumann_dct_solve(F: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Direct Neumann-Poisson solve by the type-I discrete cosine transform.

    The eigenvectors of the ghost-point Neumann Laplacian on a node-centred
    grid are ``cos(pi k i / N)``, which is precisely the DCT-I basis, with
    eigenvalues ``-4/h^2 sin^2(pi k / 2N)``.  The constant mode has eigenvalue
    zero -- that is the undetermined additive constant -- so it is set to zero,
    which also normalizes the answer to zero mean.
    """
    from ..transforms.fourier import dct, idct

    m, n = F.shape
    Fh = np.apply_along_axis(lambda v: dct(v, 1), 0, F)
    Fh = np.apply_along_axis(lambda v: dct(v, 1), 1, Fh)
    lx = -4.0 / dx**2 * np.sin(np.pi * np.arange(m) / (2 * (m - 1))) ** 2
    ly = -4.0 / dy**2 * np.sin(np.pi * np.arange(n) / (2 * (n - 1))) ** 2
    lam = lx[:, None] + ly[None, :]
    lam[0, 0] = 1.0              # placeholder; the constant mode is pinned below
    Uh = Fh / lam
    Uh[0, 0] = 0.0               # zero mean
    U = np.apply_along_axis(lambda v: idct(v, 1), 0, Uh)
    return np.apply_along_axis(lambda v: idct(v, 1), 1, U)


def helmholtz_2d(f, k: float, x_span, y_span, nx: int = 40, ny: int = 40, bc=0.0):
    """Solve the Helmholtz equation ``u_xx + u_yy + k^2 u = f``."""
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    mi, mj = nx - 1, ny - 1
    A = laplacian_matrix(mi, mj, dx, dy) + k**2 * np.eye(mi * mj)
    bc_fun = bc if callable(bc) else (lambda xx, yy: bc)
    b = np.zeros(mi * mj)
    for j in range(mj):
        for i in range(mi):
            kk = j * mi + i
            b[kk] = f(x[i + 1], y[j + 1])
            if i == 0:
                b[kk] -= bc_fun(x[0], y[j + 1]) / dx**2
            if i == mi - 1:
                b[kk] -= bc_fun(x[-1], y[j + 1]) / dx**2
            if j == 0:
                b[kk] -= bc_fun(x[i + 1], y[0]) / dy**2
            if j == mj - 1:
                b[kk] -= bc_fun(x[i + 1], y[-1]) / dy**2
    sol = np.linalg.solve(A, b)
    U = np.zeros((nx + 1, ny + 1))
    for j in range(mj):
        for i in range(mi):
            U[i + 1, j + 1] = sol[j * mi + i]
    return PDESolution(U, (x, y), None, "helmholtz_2d")


def poisson_fft(f, x_span, y_span, nx: int = 64, ny: int = 64):
    """Fast Poisson solver with homogeneous Dirichlet data, via the sine transform.

    Diagonalizes the discrete Laplacian in ``O(N log N)``.
    """
    x = np.linspace(float(x_span[0]), float(x_span[1]), nx + 1)
    y = np.linspace(float(y_span[0]), float(y_span[1]), ny + 1)
    dx, dy = x[1] - x[0], y[1] - y[0]
    F = np.array([[f(xi, yj) for yj in y[1:-1]] for xi in x[1:-1]])
    m, n = F.shape
    # discrete sine transform via the FFT of an odd extension
    def dst2(A):
        ext = np.zeros((2 * (m + 1), 2 * (n + 1)))
        ext[1 : m + 1, 1 : n + 1] = A
        ext[m + 2 :, 1 : n + 1] = -A[::-1, :]
        ext[1 : m + 1, n + 2 :] = -A[:, ::-1]
        ext[m + 2 :, n + 2 :] = A[::-1, ::-1]
        T = np.fft.fft2(ext)
        return -np.real(T[1 : m + 1, 1 : n + 1]) / 4.0

    Fh = dst2(F)
    i = np.arange(1, m + 1)[:, None]
    j = np.arange(1, n + 1)[None, :]
    lam = (2 * (np.cos(np.pi * i / (m + 1)) - 1) / dx**2
           + 2 * (np.cos(np.pi * j / (n + 1)) - 1) / dy**2)
    Uh = Fh / lam
    U_in = dst2(Uh) * (4.0 / ((m + 1) * (n + 1)))
    U = np.zeros((nx + 1, ny + 1))
    U[1:-1, 1:-1] = U_in
    return PDESolution(U, (x, y), None, "poisson_fft")
