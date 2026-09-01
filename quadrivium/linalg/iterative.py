"""Iterative solvers for linear systems.

Two families: classical stationary splittings (Jacobi through SSOR) and Krylov
subspace methods (CG through GMRES). Every solver accepts either a dense array
or any object exposing ``@`` / ``matvec``, so the sparse types in
:mod:`quadrivium.linalg.sparse` work unchanged.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ConvergenceError, DimensionError
from ..core.types import IterationResult
from ..core.utils import as_vector, check_square
from .direct import forward_substitution, back_substitution

__all__ = [
    "jacobi_iteration",
    "gauss_seidel",
    "sor",
    "ssor",
    "richardson",
    "chebyshev_iteration",
    "steepest_descent",
    "conjugate_gradient",
    "preconditioned_cg",
    "minres",
    "gmres",
    "bicg",
    "bicgstab",
    "cgs",
    "cgnr",
    "lsqr",
    "jacobi_preconditioner",
    "ssor_preconditioner",
    "incomplete_cholesky",
    "ilu0",
    "optimal_sor_omega",
]


def _operator(A):
    """Return a matvec callable for a dense array or linear operator."""
    if callable(A) and not hasattr(A, "__matmul__"):
        return A
    if hasattr(A, "matvec"):
        return A.matvec
    A = np.asarray(A, dtype=float)
    return lambda v: A @ v


# --------------------------------------------------------------------------
# Stationary iterative methods
# --------------------------------------------------------------------------
def _stationary(A, b, x0, tol, max_iter, step, name):
    A = check_square(A)
    b = as_vector(b)
    n = A.shape[0]
    if b.size != n:
        raise DimensionError(f"A is {n}x{n} but b has length {b.size}")
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    bnorm = np.linalg.norm(b) or 1.0
    residuals = []
    # A divergent splitting overflows rather than raising, so errors are muted
    # here and detected through the non-finite residual test below.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(1, max_iter + 1):
            x = step(A, b, x)
            r = float(np.linalg.norm(b - A @ x) / bnorm)
            residuals.append(r)
            if not np.isfinite(r) or r > 1e12 * (residuals[0] if residuals else 1.0):
                return IterationResult(x, k, False, residuals, name,
                                       "iteration diverged: spectral radius of the "
                                       "iteration matrix is >= 1")
            if r < tol:
                return IterationResult(x, k, True, residuals, name, "converged")
    return IterationResult(x, max_iter, False, residuals, name,
                           "maximum iterations reached")


def jacobi_iteration(A, b, x0=None, tol: float = 1e-10, max_iter: int = 10000):
    """Jacobi iteration ``x <- D^-1 (b - (L+U) x)``.

    Converges for strictly diagonally dominant systems.
    """
    A = check_square(A)
    d = np.diag(A).copy()
    if np.any(d == 0.0):
        raise ZeroDivisionError("Jacobi requires a zero-free diagonal")
    R = A - np.diag(d)

    def step(A, b, x):
        return (b - R @ x) / d

    return _stationary(A, b, x0, tol, max_iter, step, "jacobi")


def gauss_seidel(A, b, x0=None, tol: float = 1e-10, max_iter: int = 10000):
    """Gauss-Seidel: forward substitution against the lower triangle."""
    A = check_square(A)
    L = np.tril(A)
    U = A - L

    def step(A, b, x):
        return forward_substitution(L, b - U @ x)

    return _stationary(A, b, x0, tol, max_iter, step, "gauss_seidel")


def sor(A, b, omega: float = 1.5, x0=None, tol: float = 1e-10, max_iter: int = 10000):
    """Successive over-relaxation; ``0 < omega < 2`` is required for convergence."""
    if not 0.0 < omega < 2.0:
        raise ValueError("SOR requires 0 < omega < 2")
    A = check_square(A)
    n = A.shape[0]
    d = np.diag(A)
    M = np.diag(d) / omega + np.tril(A, -1)
    N = M - A

    def step(A, b, x):
        return forward_substitution(M, b + N @ x)

    return _stationary(A, b, x0, tol, max_iter, step, "sor")


def ssor(A, b, omega: float = 1.5, x0=None, tol: float = 1e-10, max_iter: int = 10000):
    """Symmetric SOR: a forward sweep followed by a backward sweep."""
    A = check_square(A)
    d = np.diag(A)
    D = np.diag(d)
    L, U = np.tril(A, -1), np.triu(A, 1)
    M1 = D / omega + L
    M2 = D / omega + U

    def step(A, b, x):
        half = forward_substitution(M1, b - (U + (1 - 1 / omega) * D) @ x)
        return back_substitution(M2, b - (L + (1 - 1 / omega) * D) @ half)

    return _stationary(A, b, x0, tol, max_iter, step, "ssor")


def richardson(A, b, omega=None, x0=None, tol: float = 1e-10, max_iter: int = 10000):
    """Richardson iteration ``x <- x + omega r``.

    With ``omega=None`` uses the optimal ``2/(lmin+lmax)`` for SPD systems.
    """
    A = check_square(A)
    if omega is None:
        ev = np.linalg.eigvalsh((A + A.T) / 2)
        omega = 2.0 / (ev[0] + ev[-1])

    def step(A, b, x):
        return x + omega * (b - A @ x)

    return _stationary(A, b, x0, tol, max_iter, step, "richardson")


def chebyshev_iteration(A, b, lmin=None, lmax=None, x0=None, tol: float = 1e-10,
                        max_iter: int = 1000):
    """Chebyshev semi-iteration for SPD systems with a known spectral interval."""
    A = check_square(A)
    b = as_vector(b)
    n = A.shape[0]
    if lmin is None or lmax is None:
        ev = np.linalg.eigvalsh((A + A.T) / 2)
        lmin = ev[0] if lmin is None else lmin
        lmax = ev[-1] if lmax is None else lmax
    d = (lmax + lmin) / 2.0
    c = (lmax - lmin) / 2.0
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - A @ x
    bnorm = np.linalg.norm(b) or 1.0
    residuals = []
    p = np.zeros(n)
    alpha = 0.0
    for k in range(1, max_iter + 1):
        if k == 1:
            p = r.copy()
            alpha = 1.0 / d
        elif k == 2:
            beta = 0.5 * (c * alpha) ** 2
            alpha = 1.0 / (d - beta / alpha)
            p = r + beta * p
        else:
            beta = (c * alpha / 2.0) ** 2
            alpha = 1.0 / (d - beta / alpha)
            p = r + beta * p
        x = x + alpha * p
        r = b - A @ x
        res = np.linalg.norm(r) / bnorm
        residuals.append(res)
        if res < tol:
            return IterationResult(x, k, True, residuals, "chebyshev", "converged")
    return IterationResult(x, max_iter, False, residuals, "chebyshev",
                           "maximum iterations reached")


# --------------------------------------------------------------------------
# Krylov subspace methods
# --------------------------------------------------------------------------
def steepest_descent(A, b, x0=None, tol: float = 1e-10, max_iter: int = 10000):
    """Steepest descent for SPD systems (CG without conjugacy)."""
    A = check_square(A)
    b = as_vector(b)
    x = as_vector(x0).copy() if x0 is not None else np.zeros(A.shape[0])
    bnorm = np.linalg.norm(b) or 1.0
    residuals = []
    for k in range(1, max_iter + 1):
        r = b - A @ x
        res = np.linalg.norm(r) / bnorm
        residuals.append(res)
        if res < tol:
            return IterationResult(x, k, True, residuals, "steepest_descent", "converged")
        Ar = A @ r
        denom = r @ Ar
        if abs(denom) < 1e-300:
            break
        x = x + (r @ r) / denom * r
    return IterationResult(x, max_iter, False, residuals, "steepest_descent",
                           "maximum iterations reached")


def conjugate_gradient(A, b, x0=None, tol: float = 1e-10, max_iter=None):
    """Conjugate gradient for symmetric positive definite systems."""
    matvec = _operator(A)
    b = as_vector(b)
    n = b.size
    max_iter = max_iter if max_iter is not None else 10 * n
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - matvec(x)
    p = r.copy()
    rs = r @ r
    bnorm = np.linalg.norm(b) or 1.0
    residuals = [np.sqrt(rs) / bnorm]
    if residuals[0] < tol:
        return IterationResult(x, 0, True, residuals, "cg", "initial guess sufficed")
    for k in range(1, max_iter + 1):
        Ap = matvec(p)
        pAp = p @ Ap
        if pAp <= 0:
            return IterationResult(x, k, False, residuals, "cg",
                                   "non-positive curvature: matrix is not SPD")
        alpha = rs / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        residuals.append(np.sqrt(rs_new) / bnorm)
        if residuals[-1] < tol:
            return IterationResult(x, k, True, residuals, "cg", "converged")
        p = r + (rs_new / rs) * p
        rs = rs_new
    return IterationResult(x, max_iter, False, residuals, "cg",
                           "maximum iterations reached")


def preconditioned_cg(A, b, M=None, x0=None, tol: float = 1e-10, max_iter=None):
    """Preconditioned CG; ``M`` applies ``M^-1`` and defaults to Jacobi."""
    matvec = _operator(A)
    b = as_vector(b)
    n = b.size
    if M is None:
        d = np.diag(np.asarray(A, dtype=float))
        d = np.where(d == 0, 1.0, d)
        M = lambda v: v / d
    elif not callable(M):
        Mmat = np.asarray(M, dtype=float)
        M = lambda v: np.linalg.solve(Mmat, v)
    max_iter = max_iter if max_iter is not None else 10 * n
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - matvec(x)
    z = M(r)
    p = z.copy()
    rz = r @ z
    bnorm = np.linalg.norm(b) or 1.0
    residuals = [np.linalg.norm(r) / bnorm]
    for k in range(1, max_iter + 1):
        Ap = matvec(p)
        pAp = p @ Ap
        if abs(pAp) < 1e-300:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        residuals.append(np.linalg.norm(r) / bnorm)
        if residuals[-1] < tol:
            return IterationResult(x, k, True, residuals, "pcg", "converged")
        z = M(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return IterationResult(x, max_iter, False, residuals, "pcg",
                           "maximum iterations reached")


def minres(A, b, x0=None, tol: float = 1e-10, max_iter=None):
    """MINRES for symmetric (possibly indefinite) systems, via Lanczos + Givens."""
    matvec = _operator(A)
    b = as_vector(b)
    n = b.size
    max_iter = max_iter if max_iter is not None else 5 * n
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - matvec(x)
    beta = np.linalg.norm(r)
    bnorm = np.linalg.norm(b) or 1.0
    residuals = [beta / bnorm]
    if beta < tol * bnorm:
        return IterationResult(x, 0, True, residuals, "minres", "initial guess sufficed")
    v_prev = np.zeros(n)
    v = r / beta
    w, w_prev = np.zeros(n), np.zeros(n)
    eta = beta
    c_prev, c = 1.0, 1.0
    s_prev, s = 0.0, 0.0
    beta_prev = 0.0
    for k in range(1, max_iter + 1):
        Av = matvec(v)
        alpha = v @ Av
        v_next = Av - alpha * v - beta_prev * v_prev
        beta_next = np.linalg.norm(v_next)
        # apply the previous two rotations to the new Lanczos column
        d1 = c * alpha - c_prev * s * beta_prev
        d2 = np.sqrt(d1 * d1 + beta_next * beta_next)
        d0 = s * alpha + c_prev * c * beta_prev
        dm = s_prev * beta_prev
        c_new = d1 / d2 if d2 != 0 else 1.0
        s_new = beta_next / d2 if d2 != 0 else 0.0
        w_new = (v - d0 * w - dm * w_prev) / d2 if d2 != 0 else np.zeros(n)
        x = x + c_new * eta * w_new
        eta = -s_new * eta
        residuals.append(abs(eta) / bnorm)
        if residuals[-1] < tol:
            return IterationResult(x, k, True, residuals, "minres", "converged")
        if beta_next < 1e-14:
            return IterationResult(x, k, True, residuals, "minres", "Krylov space exhausted")
        v_prev, v = v, v_next / beta_next
        w_prev, w = w, w_new
        beta_prev = beta_next
        c_prev, c = c, c_new
        s_prev, s = s, s_new
    return IterationResult(x, max_iter, False, residuals, "minres",
                           "maximum iterations reached")


def gmres(A, b, x0=None, tol: float = 1e-10, restart=None, max_iter=None, M=None):
    """GMRES(m) with Givens rotations and optional left preconditioning."""
    matvec = _operator(A)
    b = as_vector(b)
    n = b.size
    m = restart if restart is not None else min(n, 50)
    max_iter = max_iter if max_iter is not None else 10 * n
    prec = M if callable(M) else (lambda v: v)
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    bnorm = np.linalg.norm(prec(b)) or 1.0
    residuals = []
    total = 0
    while total < max_iter:
        r = prec(b - matvec(x))
        beta = np.linalg.norm(r)
        residuals.append(beta / bnorm)
        if beta / bnorm < tol:
            return IterationResult(x, total, True, residuals, "gmres", "converged")
        V = np.zeros((n, m + 1))
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        V[:, 0] = r / beta
        k_used = 0
        for k in range(m):
            total += 1
            k_used = k + 1
            w = prec(matvec(V[:, k]))
            for i in range(k + 1):  # modified Gram-Schmidt
                H[i, k] = V[:, i] @ w
                w = w - H[i, k] * V[:, i]
            H[k + 1, k] = np.linalg.norm(w)
            if H[k + 1, k] > 1e-14:
                V[:, k + 1] = w / H[k + 1, k]
            for i in range(k):  # apply stored rotations
                t = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
                H[i + 1, k] = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
                H[i, k] = t
            denom = np.hypot(H[k, k], H[k + 1, k])
            if denom == 0.0:
                cs[k], sn[k] = 1.0, 0.0
            else:
                cs[k], sn[k] = H[k, k] / denom, H[k + 1, k] / denom
            H[k, k] = denom
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            residuals.append(abs(g[k + 1]) / bnorm)
            if abs(g[k + 1]) / bnorm < tol or total >= max_iter:
                break
        y = back_substitution(H[:k_used, :k_used], g[:k_used])
        x = x + V[:, :k_used] @ y
        if residuals[-1] < tol:
            return IterationResult(x, total, True, residuals, "gmres", "converged")
    return IterationResult(x, total, False, residuals, "gmres",
                           "maximum iterations reached")


def bicg(A, b, x0=None, tol: float = 1e-10, max_iter=None):
    """Biconjugate gradient for general nonsymmetric systems."""
    A = check_square(A)
    b = as_vector(b)
    n = b.size
    max_iter = max_iter if max_iter is not None else 10 * n
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - A @ x
    r_hat = r.copy()
    p, p_hat = r.copy(), r_hat.copy()
    bnorm = np.linalg.norm(b) or 1.0
    residuals = [np.linalg.norm(r) / bnorm]
    rho = r_hat @ r
    for k in range(1, max_iter + 1):
        if abs(rho) < 1e-300:
            break
        Ap = A @ p
        alpha = rho / (p_hat @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_hat = r_hat - alpha * (A.T @ p_hat)
        residuals.append(np.linalg.norm(r) / bnorm)
        if residuals[-1] < tol:
            return IterationResult(x, k, True, residuals, "bicg", "converged")
        rho_new = r_hat @ r
        beta = rho_new / rho
        p = r + beta * p
        p_hat = r_hat + beta * p_hat
        rho = rho_new
    return IterationResult(x, max_iter, False, residuals, "bicg",
                           "maximum iterations reached")


def bicgstab(A, b, x0=None, tol: float = 1e-10, max_iter=None, M=None):
    """BiCGSTAB: smoother convergence than BiCG, no transpose required."""
    matvec = _operator(A)
    b = as_vector(b)
    n = b.size
    max_iter = max_iter if max_iter is not None else 10 * n
    prec = M if callable(M) else (lambda v: v)
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - matvec(x)
    r0 = r.copy()
    bnorm = np.linalg.norm(b) or 1.0
    residuals = [np.linalg.norm(r) / bnorm]
    rho = alpha = omega = 1.0
    v = p = np.zeros(n)
    for k in range(1, max_iter + 1):
        rho_new = r0 @ r
        if abs(rho_new) < 1e-300 or abs(omega) < 1e-300:
            break
        beta = (rho_new / rho) * (alpha / omega)
        p = r + beta * (p - omega * v)
        p_hat = prec(p)
        v = matvec(p_hat)
        denom = r0 @ v
        if abs(denom) < 1e-300:
            break
        alpha = rho_new / denom
        s = r - alpha * v
        if np.linalg.norm(s) / bnorm < tol:
            x = x + alpha * p_hat
            residuals.append(np.linalg.norm(s) / bnorm)
            return IterationResult(x, k, True, residuals, "bicgstab", "converged")
        s_hat = prec(s)
        t = matvec(s_hat)
        tt = t @ t
        omega = (t @ s) / tt if tt > 1e-300 else 0.0
        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t
        rho = rho_new
        residuals.append(np.linalg.norm(r) / bnorm)
        if residuals[-1] < tol:
            return IterationResult(x, k, True, residuals, "bicgstab", "converged")
    return IterationResult(x, max_iter, False, residuals, "bicgstab",
                           "maximum iterations reached")


def cgs(A, b, x0=None, tol: float = 1e-10, max_iter=None):
    """Conjugate gradient squared: BiCG's polynomial applied twice."""
    A = check_square(A)
    b = as_vector(b)
    n = b.size
    max_iter = max_iter if max_iter is not None else 10 * n
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - A @ x
    r0 = r.copy()
    bnorm = np.linalg.norm(b) or 1.0
    residuals = [np.linalg.norm(r) / bnorm]
    rho = 1.0
    p = u = q = np.zeros(n)
    for k in range(1, max_iter + 1):
        rho_new = r0 @ r
        if abs(rho_new) < 1e-300:
            break
        if k == 1:
            u = r.copy()
            p = u.copy()
        else:
            beta = rho_new / rho
            u = r + beta * q
            p = u + beta * (q + beta * p)
        v = A @ p
        denom = r0 @ v
        if abs(denom) < 1e-300:
            break
        alpha = rho_new / denom
        q = u - alpha * v
        x = x + alpha * (u + q)
        r = r - alpha * (A @ (u + q))
        rho = rho_new
        residuals.append(np.linalg.norm(r) / bnorm)
        if residuals[-1] < tol:
            return IterationResult(x, k, True, residuals, "cgs", "converged")
    return IterationResult(x, max_iter, False, residuals, "cgs",
                           "maximum iterations reached")


def cgnr(A, b, x0=None, tol: float = 1e-10, max_iter=None):
    """CG on the normal equations ``A'A x = A'b`` (works for rectangular ``A``)."""
    A = np.asarray(A, dtype=float)
    b = as_vector(b)
    n = A.shape[1]
    max_iter = max_iter if max_iter is not None else 10 * n
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    r = b - A @ x
    z = A.T @ r
    p = z.copy()
    zz = z @ z
    bnorm = np.linalg.norm(A.T @ b) or 1.0
    residuals = [np.sqrt(zz) / bnorm]
    for k in range(1, max_iter + 1):
        w = A @ p
        ww = w @ w
        if ww < 1e-300:
            break
        alpha = zz / ww
        x = x + alpha * p
        r = r - alpha * w
        z = A.T @ r
        zz_new = z @ z
        residuals.append(np.sqrt(zz_new) / bnorm)
        if residuals[-1] < tol:
            return IterationResult(x, k, True, residuals, "cgnr", "converged")
        p = z + (zz_new / zz) * p
        zz = zz_new
    return IterationResult(x, max_iter, False, residuals, "cgnr",
                           "maximum iterations reached")


def lsqr(A, b, damp: float = 0.0, tol: float = 1e-12, max_iter=None):
    """LSQR for least squares ``min ||Ax - b||`` with optional Tikhonov damping."""
    A = np.asarray(A, dtype=float)
    b = as_vector(b)
    m, n = A.shape
    max_iter = max_iter if max_iter is not None else 4 * n
    x = np.zeros(n)
    beta = np.linalg.norm(b)
    u = b / beta if beta > 0 else b.copy()
    v = A.T @ u
    alpha = np.linalg.norm(v)
    v = v / alpha if alpha > 0 else v
    w = v.copy()
    phi_bar, rho_bar = beta, alpha
    residuals = []
    for k in range(1, max_iter + 1):
        u = A @ v - alpha * u
        beta = np.linalg.norm(u)
        if beta > 0:
            u = u / beta
        v = A.T @ u - beta * v
        alpha = np.linalg.norm(v)
        if alpha > 0:
            v = v / alpha
        rho_damped = np.sqrt(rho_bar**2 + beta**2 + damp**2)
        c = rho_bar / rho_damped
        s = beta / rho_damped
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar
        x = x + (phi / rho_damped) * w
        w = v - (theta / rho_damped) * w
        residuals.append(abs(phi_bar))
        if abs(phi_bar) < tol * (np.linalg.norm(b) or 1.0):
            return IterationResult(x, k, True, residuals, "lsqr", "converged")
    return IterationResult(x, max_iter, False, residuals, "lsqr",
                           "maximum iterations reached")


# --------------------------------------------------------------------------
# Preconditioners
# --------------------------------------------------------------------------
def jacobi_preconditioner(A):
    """Diagonal (Jacobi) preconditioner as a callable applying ``M^-1``."""
    d = np.diag(np.asarray(A, dtype=float)).copy()
    d = np.where(d == 0.0, 1.0, d)
    return lambda v: v / d


def ssor_preconditioner(A, omega: float = 1.0):
    """SSOR preconditioner as a callable applying ``M^-1``."""
    A = check_square(A)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    M1 = D / omega + L
    M2 = (D / omega + L.T)

    def apply(v):
        y = forward_substitution(M1, v)
        y = (omega / (2.0 - omega)) * (np.diag(A) * y)
        return back_substitution(M2, y)

    return apply


def incomplete_cholesky(A, drop_tol: float = 0.0):
    """Zero-fill incomplete Cholesky ``A ~ L L'`` respecting the sparsity of ``A``."""
    A = check_square(A)
    n = A.shape[0]
    L = np.zeros((n, n))
    mask = np.abs(A) > drop_tol
    for i in range(n):
        for j in range(i + 1):
            if not mask[i, j]:
                continue
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    s = abs(A[i, i]) + 1e-12  # shift to keep the factor real
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j] if L[j, j] != 0 else 0.0
    return L


def ilu0(A):
    """Zero-fill incomplete LU; returns ``(L, U)`` with the sparsity of ``A``."""
    A = check_square(A)
    n = A.shape[0]
    M = A.astype(float).copy()
    mask = A != 0.0
    for k in range(n - 1):
        if M[k, k] == 0.0:
            continue
        for i in range(k + 1, n):
            if not mask[i, k]:
                continue
            M[i, k] /= M[k, k]
            for j in range(k + 1, n):
                if mask[i, j]:
                    M[i, j] -= M[i, k] * M[k, j]
    L = np.tril(M, -1) + np.eye(n)
    U = np.triu(M)
    return L, U


def optimal_sor_omega(A) -> float:
    """Optimal SOR relaxation factor from the Jacobi spectral radius.

    Valid for consistently ordered matrices with a real Jacobi spectrum.
    """
    A = check_square(A)
    d = np.diag(A)
    J = np.eye(A.shape[0]) - A / d[:, None]
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))
    if rho >= 1.0:
        return 1.0
    return 2.0 / (1.0 + np.sqrt(1.0 - rho**2))
