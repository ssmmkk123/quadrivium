"""Least squares and regularization.

The four classical routes to ``min ||Ax-b||`` -- normal equations, QR, SVD and
the iterative Krylov methods -- plus the regularizers used when the problem is
rank deficient or ill-posed.
"""

from __future__ import annotations

import numpy as np

from .. import _accel
from ..core.exceptions import DimensionError, SingularMatrixError
from ..core.utils import as_matrix, as_vector
from .direct import back_substitution, cholesky_solve
from .iterative import lsqr

__all__ = [
    "normal_equations",
    "qr_least_squares",
    "svd_least_squares",
    "pseudoinverse",
    "ridge_regression",
    "tikhonov",
    "truncated_svd",
    "total_least_squares",
    "weighted_least_squares",
    "constrained_least_squares",
    "nonnegative_least_squares",
    "lsqr_least_squares",
    "residual_analysis",
]


def normal_equations(A, b) -> np.ndarray:
    """Solve ``A'A x = A'b`` via Cholesky.

    Fast but squares the condition number; prefer QR when ``A`` is ill-conditioned.
    """
    A, b = as_matrix(A), as_vector(b)
    return cholesky_solve(A.T @ A, A.T @ b)


def qr_least_squares(A, b) -> np.ndarray:
    """Least squares by Householder QR -- the numerically preferred default.

    Applies the reflectors directly to ``b``, using ``O(m*n)`` workspace for
    an ``m`` by ``n`` matrix instead of forming an ``m`` by ``m`` Q. Requires
    at least as many observations as columns and full column rank.
    """
    A, b = as_matrix(A), as_vector(b)
    m, n = A.shape
    if b.size != m:
        raise DimensionError(f"A has {m} rows but b has length {b.size}")
    if m < n:
        raise DimensionError("QR least squares requires at least as many rows as columns")
    fast = _accel.kernel("qr_least_squares")
    if fast is not None and n:
        try:
            return fast(A, b)
        except (ValueError, RuntimeError) as exc:
            err = _accel.translate_error(exc, SingularMatrixError, SingularMatrixError)
            if err is None:
                raise
            raise err from None
    R, rhs = A.copy(), b.copy()
    for k in range(min(m - 1, n)):
        v = R[k:, k].copy()
        normv = np.linalg.norm(v)
        if normv < 1e-300:
            continue
        v[0] += np.copysign(normv, v[0]) if v[0] != 0 else normv
        vn = np.linalg.norm(v)
        if vn < 1e-300:
            continue
        v /= vn
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        rhs[k:] -= (2.0 * (v @ rhs[k:])) * v
    return back_substitution(R[:n, :n], rhs[:n])


def svd_least_squares(A, b, rcond: float = 1e-15):
    """Minimum-norm least squares via the SVD; handles rank deficiency."""
    A, b = as_matrix(A), as_vector(b)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = rcond * (s[0] if s.size else 0.0)
    s_inv = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)
    return Vt.T @ (s_inv * (U.T @ b))


def pseudoinverse(A, rcond: float = 1e-15) -> np.ndarray:
    """Moore-Penrose pseudoinverse from the SVD."""
    A = as_matrix(A)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = rcond * (s[0] if s.size else 0.0)
    s_inv = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)
    return (Vt.T * s_inv) @ U.T


def ridge_regression(A, b, alpha: float = 1.0) -> np.ndarray:
    """L2-regularized solution ``(A'A + alpha I)^-1 A'b``."""
    A, b = as_matrix(A), as_vector(b)
    gram = A.T @ A
    gram.flat[::gram.shape[0] + 1] += alpha
    return np.linalg.solve(gram, A.T @ b)


def tikhonov(A, b, alpha: float = 1.0, L=None) -> np.ndarray:
    """General Tikhonov regularization ``min ||Ax-b||^2 + alpha ||L x||^2``."""
    A, b = as_matrix(A), as_vector(b)
    if L is None:
        return ridge_regression(A, b, alpha)
    L = as_matrix(L)
    return np.linalg.solve(A.T @ A + alpha * (L.T @ L), A.T @ b)


def truncated_svd(A, b, k: int) -> np.ndarray:
    """Least squares keeping only the ``k`` largest singular values."""
    A, b = as_matrix(A), as_vector(b)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, s.size)
    return Vt[:k].T @ ((U[:, :k].T @ b) / s[:k])


def total_least_squares(A, b):
    """Total least squares: errors in both ``A`` and ``b`` (orthogonal regression)."""
    A, b = as_matrix(A), as_vector(b)
    n = A.shape[1]
    C = np.hstack([A, b.reshape(-1, 1)])
    _, _, Vt = np.linalg.svd(C)
    v = Vt[-1]
    if abs(v[-1]) < 1e-300:
        raise np.linalg.LinAlgError("total least squares problem is degenerate")
    return -v[:n] / v[n]


def weighted_least_squares(A, b, weights) -> np.ndarray:
    """Weighted least squares ``min sum w_i (a_i'x - b_i)^2``."""
    A, b = as_matrix(A), as_vector(b)
    w = as_vector(weights)
    sw = np.sqrt(w)
    return qr_least_squares(A * sw[:, None], b * sw)


def constrained_least_squares(A, b, C, d):
    """Equality-constrained least squares: minimize ``||Ax-b||`` s.t. ``Cx = d``.

    Solved through the KKT system.
    """
    A, b = as_matrix(A), as_vector(b)
    C, d = as_matrix(C), as_vector(d)
    n, m = A.shape[1], C.shape[0]
    KKT = np.block([[A.T @ A, C.T], [C, np.zeros((m, m))]])
    rhs = np.concatenate([A.T @ b, d])
    return np.linalg.solve(KKT, rhs)[:n]


def nonnegative_least_squares(A, b, tol: float = 1e-10, max_iter: int = 300):
    """Lawson-Hanson active set algorithm for ``min ||Ax-b||`` with ``x >= 0``."""
    A, b = as_matrix(A), as_vector(b)
    m, n = A.shape
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    for _ in range(max_iter):
        if passive.all() or np.all(w[~passive] <= tol):
            break
        cand = np.where(~passive, w, -np.inf)
        j = int(np.argmax(cand))
        passive[j] = True
        for _ in range(max_iter):
            idx = np.where(passive)[0]
            s = np.zeros(n)
            s[idx] = np.linalg.lstsq(A[:, idx], b, rcond=None)[0]
            if np.all(s[idx] > tol):
                x = s
                break
            neg = idx[s[idx] <= tol]
            ratios = x[neg] / (x[neg] - s[neg])
            alpha = np.min(ratios)
            x = x + alpha * (s - x)
            passive[(x <= tol) & passive] = False
        w = A.T @ (b - A @ x)
    return np.maximum(x, 0.0)


def lsqr_least_squares(A, b, damp: float = 0.0, **kwargs):
    """Iterative least squares through LSQR (large or sparse problems)."""
    return lsqr(A, b, damp=damp, **kwargs).x


def residual_analysis(A, b, x):
    """Diagnostics for a fitted model: residual, R^2, RMSE and standard errors."""
    A, b, x = as_matrix(A), as_vector(b), as_vector(x)
    r = b - A @ x
    m, n = A.shape
    dof = max(m - n, 1)
    sigma2 = float(r @ r) / dof
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
        stderr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov, stderr = None, None
    return {
        "residual": r,
        "residual_norm": float(np.linalg.norm(r)),
        "rmse": float(np.sqrt(float(r @ r) / m)),
        "r_squared": float(1.0 - float(r @ r) / ss_tot) if ss_tot > 0 else np.nan,
        "sigma2": sigma2,
        "covariance": cov,
        "std_errors": stderr,
        "dof": dof,
    }
