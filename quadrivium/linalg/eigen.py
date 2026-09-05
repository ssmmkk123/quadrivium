"""Eigenvalue and singular value algorithms.

Covers the vector iterations (power / inverse / Rayleigh), the QR algorithm in
its unshifted, shifted and Francis double-shift forms, the Jacobi rotation
method, Krylov projections (Lanczos, Arnoldi), and SVD by one-sided Jacobi.
"""

from __future__ import annotations

import numpy as np

from .. import _accel

from ..core.exceptions import ConvergenceError, SingularMatrixError
from ..core.types import EigenResult
from ..core.utils import as_matrix, as_vector, check_square, is_symmetric
from .direct import hessenberg, householder_qr, plu_solve

__all__ = [
    "power_iteration",
    "inverse_power_iteration",
    "shifted_power_iteration",
    "rayleigh_quotient_iteration",
    "deflation_power",
    "qr_algorithm",
    "shifted_qr_algorithm",
    "francis_qr",
    "jacobi_eigen",
    "lanczos",
    "arnoldi",
    "sturm_sequence",
    "bisection_eigenvalues",
    "svd_jacobi",
    "svd_golub_kahan",
    "polar_decomposition",
    "schur",
    "schur_eigenvalues",
    "gershgorin_disks",
    "spectral_radius",
    "matrix_power",
    "matrix_exponential",
    "matrix_function",
]


def _safe_tangent(theta: float) -> float:
    """Tangent of the Jacobi rotation angle, guarded against overflow.

    The textbook form ``sign(t)/(|t| + sqrt(t^2+1))`` overflows for large
    ``theta``, where the root is accurately ``1/(2 theta)``.
    """
    if theta == 0.0:
        return 1.0
    if abs(theta) > 1e8:
        return 1.0 / (2.0 * theta)
    return np.sign(theta) / (abs(theta) + np.sqrt(theta * theta + 1.0))


# --------------------------------------------------------------------------
# Vector iterations
# --------------------------------------------------------------------------
def power_iteration(A, x0=None, tol: float = 1e-10, max_iter: int = 1000):
    """Dominant eigenpair by the power method."""
    A = check_square(A)
    n = A.shape[0]
    x = as_vector(x0) if x0 is not None else np.ones(n)
    x = x / np.linalg.norm(x)
    lam = 0.0
    for k in range(1, max_iter + 1):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny < 1e-300:
            return EigenResult(np.array([0.0]), x.reshape(-1, 1), k, True, "power")
        x_new = y / ny
        lam_new = float(x_new @ A @ x_new)
        if np.linalg.norm(x_new - x) < tol or abs(lam_new - lam) < tol:
            return EigenResult(np.array([lam_new]), x_new.reshape(-1, 1), k, True, "power")
        x, lam = x_new, lam_new
    return EigenResult(np.array([lam]), x.reshape(-1, 1), max_iter, False, "power")


def inverse_power_iteration(A, sigma: float = 0.0, x0=None, tol: float = 1e-10,
                            max_iter: int = 1000):
    """Eigenpair closest to the shift ``sigma`` by inverse iteration."""
    A = check_square(A)
    n = A.shape[0]
    M = A - sigma * np.eye(n)
    x = as_vector(x0) if x0 is not None else np.ones(n)
    x = x / np.linalg.norm(x)
    for k in range(1, max_iter + 1):
        try:
            y = plu_solve(M, x)
        except SingularMatrixError:
            return EigenResult(np.array([sigma]), x.reshape(-1, 1), k, True, "inverse_power")
        ny = np.linalg.norm(y)
        if ny < 1e-300:
            break
        x_new = y / ny
        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(x_new + x) < tol:
            lam = float(x_new @ A @ x_new)
            return EigenResult(np.array([lam]), x_new.reshape(-1, 1), k, True, "inverse_power")
        x = x_new
    lam = float(x @ A @ x)
    return EigenResult(np.array([lam]), x.reshape(-1, 1), max_iter, False, "inverse_power")


def shifted_power_iteration(A, sigma: float, **kwargs):
    """Power iteration on ``A - sigma I``, undoing the shift at the end."""
    A = check_square(A)
    res = power_iteration(A - sigma * np.eye(A.shape[0]), **kwargs)
    return EigenResult(res.eigenvalues + sigma, res.eigenvectors, res.iterations,
                       res.converged, "shifted_power")


def rayleigh_quotient_iteration(A, x0=None, tol: float = 1e-12, max_iter: int = 100):
    """Rayleigh quotient iteration: cubic convergence for symmetric ``A``."""
    A = check_square(A)
    n = A.shape[0]
    x = as_vector(x0) if x0 is not None else np.ones(n)
    x = x / np.linalg.norm(x)
    lam = float(x @ A @ x)
    for k in range(1, max_iter + 1):
        try:
            y = plu_solve(A - lam * np.eye(n), x)
        except (SingularMatrixError, np.linalg.LinAlgError):
            return EigenResult(np.array([lam]), x.reshape(-1, 1), k, True, "rayleigh")
        ny = np.linalg.norm(y)
        if ny < 1e-300 or not np.isfinite(ny):
            return EigenResult(np.array([lam]), x.reshape(-1, 1), k, True, "rayleigh")
        x = y / ny
        lam_new = float(x @ A @ x)
        if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
            return EigenResult(np.array([lam_new]), x.reshape(-1, 1), k, True, "rayleigh")
        lam = lam_new
    return EigenResult(np.array([lam]), x.reshape(-1, 1), max_iter, False, "rayleigh")


def deflation_power(A, k=None, tol: float = 1e-10, max_iter: int = 1000):
    """Leading ``k`` eigenpairs of a symmetric matrix by Hotelling deflation."""
    A = check_square(A)
    n = A.shape[0]
    k = n if k is None else min(k, n)
    B = A.astype(float).copy()
    vals, vecs = [], []
    for _ in range(k):
        res = power_iteration(B, tol=tol, max_iter=max_iter)
        lam = float(res.eigenvalues[0])
        v = res.eigenvectors[:, 0]
        vals.append(lam)
        vecs.append(v)
        B = B - lam * np.outer(v, v)
    return EigenResult(np.array(vals), np.array(vecs).T, k, True, "deflation")


# --------------------------------------------------------------------------
# QR algorithm
# --------------------------------------------------------------------------
def _hessenberg_qr_sweep(H, V=None):
    """One unshifted QR step on an upper Hessenberg ``H``, in place.

    Overwrites ``H`` with ``R Q`` for ``H = Q R``, using the ``n-1`` Givens
    rotations that clear the subdiagonal. Hessenberg form is preserved by the
    step, so the whole iteration stays ``O(n^2)`` rather than the ``O(n^3)`` a
    dense factorization would cost. When ``V`` is given it is post-multiplied
    by the same ``Q``.
    """
    n = H.shape[0]
    cs = np.empty(n - 1)
    sn = np.empty(n - 1)
    for k in range(n - 1):
        a, b = H[k, k], H[k + 1, k]
        r = np.hypot(a, b)
        c, sg = (1.0, 0.0) if r == 0.0 else (a / r, b / r)
        cs[k], sn[k] = c, sg
        row0 = H[k, k:].copy()
        row1 = H[k + 1, k:]
        H[k, k:] = c * row0 + sg * row1
        H[k + 1, k:] = c * row1 - sg * row0
    for k in range(n - 1):
        c, sg = cs[k], sn[k]
        # R is upper triangular, so the rotation on columns (k, k+1) only
        # reaches row k+1 -- which is exactly the entry that restores the
        # Hessenberg subdiagonal.
        col0 = H[: k + 2, k].copy()
        col1 = H[: k + 2, k + 1]
        H[: k + 2, k] = c * col0 + sg * col1
        H[: k + 2, k + 1] = c * col1 - sg * col0
        if V is not None:
            v0 = V[:, k].copy()
            v1 = V[:, k + 1]
            V[:, k] = c * v0 + sg * v1
            V[:, k + 1] = c * v1 - sg * v0


def qr_algorithm(A, tol: float = 1e-12, max_iter: int = 5000, compute_vectors: bool = False):
    """Unshifted QR iteration ``A_{k+1} = R_k Q_k``.

    The iteration is preceded by a Householder reduction to upper Hessenberg
    form. That is an orthogonal similarity, so it changes no eigenvalue, and
    Hessenberg form is invariant under a QR step -- which drops the cost of a
    sweep from ``O(n^3)`` to ``O(n^2)`` and lets the convergence test read the
    subdiagonal alone instead of the whole lower triangle.

    Being unshifted, this still converges only linearly, at a rate set by the
    ratios of successive eigenvalue magnitudes, and does not converge at all
    for a matrix with complex-conjugate pairs. :func:`shifted_qr_algorithm` and
    :func:`francis_qr` exist for those cases.
    """
    A = check_square(A)
    n = A.shape[0]
    if n == 0:
        return EigenResult(np.zeros(0), np.zeros((0, 0)) if compute_vectors else None,
                           0, True, "qr")
    if compute_vectors:
        Ak, V = hessenberg(A, compute_q=True)
        Ak = np.ascontiguousarray(Ak)
    else:
        Ak = np.ascontiguousarray(hessenberg(A, compute_q=False))
        V = None
    if n == 1:
        return EigenResult(np.diag(Ak).copy(), V if compute_vectors else None,
                           0, True, "qr")

    # A subdiagonal entry is negligible when it is small next to the diagonal
    # entries it sits between -- the same deflation test
    # :func:`shifted_qr_algorithm` and :func:`francis_qr` use below. Comparing
    # the subdiagonal against an unscaled ``tol`` instead cannot succeed at
    # all once ``||A||`` is large, because rounding holds a converged
    # subdiagonal near eps*||A||: the iteration then spends its entire budget
    # and reports failure on an answer it found in the first few sweeps.
    fast = _accel.kernel("hessenberg_qr_iterate")
    if fast is not None:
        iters, converged = fast(Ak, V, tol, max_iter)
        return EigenResult(np.diag(Ak).copy(), V if compute_vectors else None,
                           iters, converged, "qr")

    sub = np.arange(n - 1)
    for k in range(1, max_iter + 1):
        _hessenberg_qr_sweep(Ak, V)
        diag = np.abs(np.diag(Ak))
        if np.all(np.abs(Ak[sub + 1, sub])
                  <= tol * (diag[:-1] + diag[1:] + 1e-300)):
            return EigenResult(np.diag(Ak).copy(), V if compute_vectors else None,
                               k, True, "qr")
    return EigenResult(np.diag(Ak).copy(), V if compute_vectors else None,
                       max_iter, False, "qr")


def shifted_qr_algorithm(A, tol: float = 1e-12, max_iter: int = 5000):
    """Hessenberg QR with Wilkinson shifts and deflation (real spectra)."""
    A = check_square(A)
    n = A.shape[0]
    H = hessenberg(A, compute_q=False)
    eigs = np.zeros(n)
    m = n
    iters = 0
    while m > 0 and iters < max_iter:
        if m == 1:
            eigs[0] = H[0, 0]
            break
        # deflate on a negligible sub-diagonal entry
        small = abs(H[m - 1, m - 2]) <= tol * (abs(H[m - 1, m - 1]) + abs(H[m - 2, m - 2]) + 1e-300)
        if small:
            eigs[m - 1] = H[m - 1, m - 1]
            H = H[: m - 1, : m - 1]
            m -= 1
            continue
        a, b = H[m - 2, m - 2], H[m - 2, m - 1]
        c, d = H[m - 1, m - 2], H[m - 1, m - 1]
        delta = (a - d) / 2.0
        disc = delta * delta + b * c
        if disc >= 0:
            sq = np.sqrt(disc)
            mu = d + delta - np.sign(delta if delta != 0 else 1.0) * sq
        else:
            mu = d  # complex pair: fall back to the Rayleigh shift
        Q, R = householder_qr(H - mu * np.eye(m), reduced=False)
        H = R @ Q + mu * np.eye(m)
        iters += 1
    if m == 2:
        eigs[:2] = np.linalg.eigvals(H[:2, :2]).real
    return EigenResult(np.sort(eigs), None, iters, iters < max_iter, "shifted_qr")


def francis_qr(A, tol: float = 1e-12, max_iter: int = 1000):
    """Francis double-shift QR: real Schur form, handling complex pairs.

    Returns the (possibly complex) eigenvalues read off the quasi-triangular
    real Schur form.
    """
    A = check_square(A)
    n = A.shape[0]
    H = hessenberg(A, compute_q=False).astype(float)
    eigs = np.zeros(n, dtype=complex)
    hi = n - 1
    it = 0
    while hi >= 0 and it < max_iter:
        it += 1
        if hi == 0:
            eigs[0] = H[0, 0]
            break
        # look for a negligible sub-diagonal
        lo = hi
        while lo > 0 and abs(H[lo, lo - 1]) > tol * (abs(H[lo, lo]) + abs(H[lo - 1, lo - 1]) + 1e-300):
            lo -= 1
        if lo == hi:
            eigs[hi] = H[hi, hi]
            hi -= 1
            continue
        if lo == hi - 1:
            block = H[hi - 1 : hi + 1, hi - 1 : hi + 1]
            eigs[hi - 1 : hi + 1] = np.linalg.eigvals(block)
            hi -= 2
            continue
        # implicit double shift on the active block
        M = H[lo : hi + 1, lo : hi + 1]
        m = M.shape[0]
        s = M[m - 2, m - 2] + M[m - 1, m - 1]
        t = M[m - 2, m - 2] * M[m - 1, m - 1] - M[m - 2, m - 1] * M[m - 1, m - 2]
        X = M @ M - s * M + t * np.eye(m)
        Q, _ = householder_qr(X, reduced=False)
        M = Q.T @ M @ Q
        M = np.triu(M, -1)
        H[lo : hi + 1, lo : hi + 1] = M
    if hi >= 0 and it >= max_iter:
        return EigenResult(np.linalg.eigvals(A), None, it, False, "francis_qr")
    if np.allclose(eigs.imag, 0.0):
        eigs = eigs.real
    return EigenResult(eigs, None, it, True, "francis_qr")


def jacobi_eigen(A, tol: float = 1e-12, max_sweeps: int = 100):
    """Cyclic Jacobi rotations for symmetric matrices (very accurate)."""
    A = check_square(A)
    if not is_symmetric(A, tol=1e-8):
        raise ValueError("Jacobi eigenvalue method requires a symmetric matrix")
    n = A.shape[0]
    # ``tol`` is an absolute bound on the off-diagonal mass, tightened to a
    # relative one for a matrix that is small to begin with: below unit norm,
    # an absolute 1e-12 would stop while the off-diagonal was still a sizeable
    # fraction of the matrix. Above unit norm the absolute bound stands, and
    # the stagnation test below is what ends the iteration once rounding
    # holds ``off`` near eps*||A||_F -- an absolute ``tol`` is unreachable
    # there, and testing it alone spends every sweep and then reports failure
    # on an answer found in the first few.
    threshold = tol * min(1.0, float(np.linalg.norm(A)) or 1.0)
    fast = _accel.kernel("jacobi_eigen")
    if fast is not None and n:
        vals, vecs, sweep, conv = fast(np.ascontiguousarray(A, dtype=float),
                                       threshold, max_sweeps)
        idx = np.argsort(vals)
        return EigenResult(vals[idx].copy(), vecs[:, idx], sweep, conv, "jacobi")
    D = A.astype(float).copy()
    V = np.eye(n)
    # Each rotation moves mass from the off-diagonal to the diagonal without
    # changing the Frobenius norm, so ``off`` decreases monotonically until
    # rounding stops it, near eps*||A||_F. A sweep that fails to reduce it has
    # reached that floor and the decomposition is as converged as double
    # precision allows.
    prev_off = np.inf
    for sweep in range(1, max_sweeps + 1):
        off = np.sqrt(2.0 * np.sum(np.tril(D, -1) ** 2))
        if off < threshold or off >= prev_off:
            break
        prev_off = off
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(D[p, q]) < 1e-300:
                    continue
                theta = (D[q, q] - D[p, p]) / (2.0 * D[p, q])
                t = _safe_tangent(theta)
                c = 1.0 / np.sqrt(t * t + 1.0)
                s = t * c
                J = np.eye(n)
                J[p, p] = J[q, q] = c
                J[p, q], J[q, p] = s, -s
                D = J.T @ D @ J
                V = V @ J
    idx = np.argsort(np.diag(D))
    return EigenResult(np.diag(D)[idx].copy(), V[:, idx], sweep, True, "jacobi")


# --------------------------------------------------------------------------
# Krylov subspace projections
# --------------------------------------------------------------------------
def lanczos(A, k=None, v0=None, reorthogonalize: bool = True):
    """Lanczos tridiagonalization of a symmetric matrix.

    Returns ``(alpha, beta, Q)`` with ``T = tridiag(beta, alpha, beta)``.
    """
    A = check_square(A)
    n = A.shape[0]
    k = n if k is None else min(k, n)
    q = as_vector(v0) if v0 is not None else np.ones(n)
    q = q / np.linalg.norm(q)
    Q = np.zeros((n, k))
    alpha = np.zeros(k)
    beta = np.zeros(max(k - 1, 0))
    q_prev = np.zeros(n)
    b_prev = 0.0
    for j in range(k):
        Q[:, j] = q
        w = A @ q - b_prev * q_prev
        alpha[j] = q @ w
        w = w - alpha[j] * q
        if reorthogonalize:
            w -= Q[:, : j + 1] @ (Q[:, : j + 1].T @ w)
        b = np.linalg.norm(w)
        if j < k - 1:
            beta[j] = b
            if b < 1e-14:
                Q = Q[:, : j + 1]
                return alpha[: j + 1], beta[:j], Q
            q_prev, q, b_prev = q, w / b, b
    return alpha, beta, Q


def arnoldi(A, k=None, v0=None):
    """Arnoldi iteration for general matrices; returns ``(Q, H)``.

    ``H`` is ``(k+1, k)`` upper Hessenberg with ``A Q_k = Q_{k+1} H``.
    """
    A = check_square(A)
    n = A.shape[0]
    k = n if k is None else min(k, n)
    q = as_vector(v0) if v0 is not None else np.ones(n)
    Q = np.zeros((n, k + 1))
    H = np.zeros((k + 1, k))
    Q[:, 0] = q / np.linalg.norm(q)
    for j in range(k):
        w = A @ Q[:, j]
        for i in range(j + 1):
            H[i, j] = Q[:, i] @ w
            w = w - H[i, j] * Q[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-14:
            return Q[:, : j + 1], H[: j + 1, : j + 1]
        Q[:, j + 1] = w / H[j + 1, j]
    return Q, H


def sturm_sequence(alpha, beta, x: float) -> int:
    """Number of eigenvalues of a symmetric tridiagonal matrix below ``x``."""
    alpha = as_vector(alpha)
    beta = as_vector(beta)
    n = alpha.size
    count = 0
    d = alpha[0] - x
    if d < 0:
        count += 1
    for i in range(1, n):
        if d == 0.0:
            d = 1e-300
        d = alpha[i] - x - beta[i - 1] ** 2 / d
        if d < 0:
            count += 1
    return count


def bisection_eigenvalues(alpha, beta, tol: float = 1e-12):
    """All eigenvalues of a symmetric tridiagonal matrix via Sturm bisection."""
    alpha = as_vector(alpha)
    beta = as_vector(beta)
    n = alpha.size
    r = np.max(np.abs(alpha)) + 2 * (np.max(np.abs(beta)) if beta.size else 0.0) + 1.0
    eigs = np.zeros(n)
    for i in range(n):
        lo, hi = -r, r
        while hi - lo > tol * max(1.0, abs(lo) + abs(hi)):
            mid = 0.5 * (lo + hi)
            if sturm_sequence(alpha, beta, mid) > i:
                hi = mid
            else:
                lo = mid
        eigs[i] = 0.5 * (lo + hi)
    return eigs


# --------------------------------------------------------------------------
# Singular values and related decompositions
# --------------------------------------------------------------------------
def svd_jacobi(A, tol: float = 1e-13, max_sweeps: int = 60):
    """One-sided Jacobi SVD: ``A = U S V'`` with high relative accuracy."""
    A = as_matrix(A)
    m, n = A.shape
    transposed = m < n
    W = A.T.copy() if transposed else A.astype(float).copy()
    m, n = W.shape
    fast = _accel.kernel("svd_jacobi")
    if fast is not None and m and n:
        W, V, _sweeps = fast(np.ascontiguousarray(W), tol, max_sweeps)
        return _svd_assemble(W, V, m, n, transposed)
    V = np.eye(n)
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                a = W[:, p] @ W[:, p]
                b = W[:, q] @ W[:, q]
                c = W[:, p] @ W[:, q]
                if abs(c) < tol * np.sqrt(a * b + 1e-300):
                    continue
                off = max(off, abs(c) / np.sqrt(a * b + 1e-300))
                zeta = (b - a) / (2.0 * c)
                t = _safe_tangent(zeta)
                cs = 1.0 / np.sqrt(1.0 + t * t)
                sn = cs * t
                Wp = cs * W[:, p] - sn * W[:, q]
                Wq = sn * W[:, p] + cs * W[:, q]
                W[:, p], W[:, q] = Wp, Wq
                Vp = cs * V[:, p] - sn * V[:, q]
                Vq = sn * V[:, p] + cs * V[:, q]
                V[:, p], V[:, q] = Vp, Vq
        if off == 0.0:
            break
    return _svd_assemble(W, V, m, n, transposed)


def _svd_assemble(W, V, m, n, transposed):
    """Read the singular values off the rotated columns and order the factors.

    One-sided Jacobi leaves ``W = U S``, so the singular values are the column
    norms and ``U`` is ``W`` with those divided out. A zero column has no
    determined direction; it is left at zero rather than dividing by it.
    """
    s = np.linalg.norm(W, axis=0)
    idx = np.argsort(-s)
    s, V, W = s[idx], V[:, idx], W[:, idx]
    nonzero = s > 1e-300
    U = np.zeros((m, n))
    np.divide(W, s, out=U, where=nonzero)
    if transposed:
        return V, s, U.T
    return U, s, V.T


def svd_golub_kahan(A):
    """SVD through bidiagonalization plus a symmetric eigen-solve on ``B'B``."""
    A = as_matrix(A)
    m, n = A.shape
    AtA = A.T @ A
    res = jacobi_eigen(AtA)
    vals = np.clip(res.eigenvalues, 0.0, None)[::-1]
    V = res.eigenvectors[:, ::-1]
    s = np.sqrt(vals)
    U = np.zeros((m, min(m, n)))
    for j in range(min(m, n)):
        if s[j] > 1e-300:
            U[:, j] = A @ V[:, j] / s[j]
    return U, s[: min(m, n)], V.T


def polar_decomposition(A, side: str = "right"):
    """Polar decomposition of a (possibly rectangular) matrix.

    ``side="right"`` returns ``(R, P)`` with ``A = R @ P``; ``side="left"``
    returns ``(P, R)`` with ``A = P @ R``.  ``R`` has orthonormal columns
    (rows) and ``P`` is symmetric positive semidefinite.

    The reduced SVD is used throughout, so tall, square and wide matrices are
    all handled.  ``P`` is positive *definite* only when ``A`` has full rank;
    a rank-deficient ``A`` gives a singular ``P``, which is the correct answer
    rather than an error.
    """
    A = as_matrix(A)
    # full_matrices=False is essential: with the full SVD, U is m-by-m and Vt
    # is n-by-n, so U @ Vt is not even conformable when m != n.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    R = U @ Vt
    if side == "right":
        return R, Vt.T @ (s[:, None] * Vt)
    if side == "left":
        return U @ (s[:, None] * U.T), R
    raise ValueError("side must be 'right' or 'left'")


def _wilkinson_shift(T, m: int) -> float:
    """Wilkinson shift from the trailing 2-by-2 block ending at index ``m``."""
    a, b = T[m - 1, m - 1], T[m - 1, m]
    c, d = T[m, m - 1], T[m, m]
    delta = 0.5 * (a - d)
    disc = delta * delta + b * c
    if disc < 0.0:  # complex pair: fall back to the Rayleigh shift
        return float(d)
    sgn = 1.0 if delta >= 0.0 else -1.0
    denom = delta + sgn * np.sqrt(disc)
    return float(d) if denom == 0.0 else float(d - b * c / denom)


def schur(A, tol: float = 1e-12, max_iter: int = 2000):
    """Real Schur form ``A = Q T Q'`` by shifted QR with deflation.

    ``T`` is quasi-triangular: 1-by-1 blocks hold real eigenvalues and 2-by-2
    blocks hold complex-conjugate pairs.  A real matrix with complex spectrum
    has no triangular real Schur form, so the convergence test must accept
    2-by-2 blocks -- testing ``norm(tril(T, -1)) < tol`` instead would never
    pass and would burn every iteration.

    Returns ``(Q, T)``.  Use :func:`schur_eigenvalues` to read the spectrum
    off the diagonal blocks.
    """
    A = check_square(A)
    n = A.shape[0]
    T, Q = hessenberg(A)
    if n == 1:
        return Q, T
    m = n - 1  # active block is T[:m+1, :m+1]
    iters = 0
    while m > 0 and iters < max_iter:
        # Deflate any negligible subdiagonal at the bottom of the active block.
        scale = abs(T[m - 1, m - 1]) + abs(T[m, m])
        if abs(T[m, m - 1]) <= tol * max(scale, 1.0):
            T[m, m - 1] = 0.0
            m -= 1
            continue
        if m >= 2:
            scale = abs(T[m - 2, m - 2]) + abs(T[m - 1, m - 1])
            if abs(T[m - 1, m - 2]) <= tol * max(scale, 1.0):
                T[m - 1, m - 2] = 0.0  # a converged 2-by-2 block
                m -= 2
                continue
        elif m == 1:
            break  # trailing 2-by-2 with a complex pair: leave it be
        k = m + 1
        a, b = T[m - 1, m - 1], T[m - 1, m]
        c, d = T[m, m - 1], T[m, m]
        delta = 0.5 * (a - d)
        if delta * delta + b * c < 0.0:
            # Complex-conjugate shift pair.  A single real shift stagnates
            # here, so use the real double-shift polynomial
            # (T - mu I)(T - conj(mu) I) = T^2 - tr*T + det*I, whose factors
            # are complex but whose product is real.
            tr, det = a + d, a * d - b * c
            Tk = T[:k, :k]
            M = Tk @ Tk - tr * Tk + det * np.eye(k)
        else:
            mu = _wilkinson_shift(T, m)
            M = T[:k, :k] - mu * np.eye(k)
        Qk, _ = householder_qr(M, reduced=False)
        T[:k, :k] = Qk.T @ T[:k, :k] @ Qk
        if k < n:  # carry the rotation through the off-diagonal coupling
            T[:k, k:] = Qk.T @ T[:k, k:]
        Q[:, :k] = Q[:, :k] @ Qk
        iters += 1
    T[np.tril_indices(n, -2)] = 0.0
    _standardize_2x2_blocks(T, Q, tol)
    return Q, T


def _standardize_2x2_blocks(T, Q, tol: float) -> None:
    """Triangularize any 2-by-2 diagonal block whose eigenvalues are real.

    Deflation can accept a 2-by-2 block before its own subdiagonal has
    converged.  When that block turns out to have a real pair, one Givens
    rotation splits it, so a matrix with a purely real spectrum comes back
    genuinely triangular instead of carrying a spurious coupling.
    """
    n = T.shape[0]
    i = 0
    while i < n - 1:
        c_sub = T[i + 1, i]
        if c_sub == 0.0:
            i += 1
            continue
        a, b, d = T[i, i], T[i, i + 1], T[i + 1, i + 1]
        delta = 0.5 * (a - d)
        disc = delta * delta + b * c_sub
        if disc <= 0.0:
            i += 2  # genuine complex pair
            continue
        sgn = 1.0 if delta >= 0.0 else -1.0
        lam = d + delta - sgn * np.sqrt(disc)  # eigenvalue nearest d
        # Rotate the eigenvector of (a - lam, b; c, d - lam) onto e1.
        x, y = b, lam - a
        if abs(x) + abs(y) < tol * (abs(a) + abs(d) + 1.0):
            x, y = lam - d, c_sub
        r = np.hypot(x, y)
        if r == 0.0:
            i += 2
            continue
        cs, sn = x / r, y / r
        G = np.array([[cs, -sn], [sn, cs]])
        T[i:i + 2, :] = G.T @ T[i:i + 2, :]
        T[:, i:i + 2] = T[:, i:i + 2] @ G
        Q[:, i:i + 2] = Q[:, i:i + 2] @ G
        T[i + 1, i] = 0.0
        i += 1


def schur_eigenvalues(T, tol: float = 1e-13) -> np.ndarray:
    """Eigenvalues read from the diagonal blocks of a real Schur form.

    2-by-2 blocks are found by a relative test on the subdiagonal, so a form
    produced elsewhere -- carrying round-off rather than exact zeros -- is read
    correctly rather than as a string of spurious complex pairs.
    """
    T = check_square(T)
    n = T.shape[0]
    vals = np.zeros(n, dtype=complex)
    scale = float(np.max(np.abs(T))) if n else 1.0
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > tol * max(
                abs(T[i, i]) + abs(T[i + 1, i + 1]), scale):
            a, b, c, d = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
            tr, det = a + d, a * d - b * c
            disc = complex(0.25 * tr * tr - det)
            root = np.sqrt(disc)
            vals[i] = 0.5 * tr + root
            vals[i + 1] = 0.5 * tr - root
            i += 2
        else:
            vals[i] = T[i, i]
            i += 1
    return vals


def gershgorin_disks(A):
    """Gershgorin disks as ``(centers, radii)``; the spectrum lies in their union."""
    A = check_square(A)
    centers = np.diag(A).copy()
    radii = np.sum(np.abs(A), axis=1) - np.abs(centers)
    return centers, radii


def spectral_radius(A) -> float:
    """Largest eigenvalue modulus."""
    return float(np.max(np.abs(np.linalg.eigvals(check_square(A)))))


def matrix_power(A, p: int) -> np.ndarray:
    """Integer matrix power by binary exponentiation."""
    A = check_square(A)
    n = A.shape[0]
    if p < 0:
        return matrix_power(np.linalg.inv(A), -p)
    result = np.eye(n)
    base = A.astype(float).copy()
    while p:
        if p & 1:
            result = result @ base
        base = base @ base
        p >>= 1
    return result


def matrix_exponential(A, order: int = 6) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring with a Pade approximant."""
    A = check_square(A)
    n = A.shape[0]
    nrm = np.max(np.sum(np.abs(A), axis=1))
    s = max(0, int(np.ceil(np.log2(nrm))) + 1) if nrm > 0.5 else 0
    As = A / (2.0**s)
    X = np.eye(n)
    N = np.eye(n)
    D = np.eye(n)
    c = 1.0
    for k in range(1, order + 1):
        c = c * (order - k + 1) / (k * (2 * order - k + 1))
        X = As @ X
        N = N + c * X
        D = D + ((-1) ** k) * c * X
    E = np.linalg.solve(D, N)
    for _ in range(s):
        E = E @ E
    return E


def matrix_function(A, f):
    """Apply a scalar function to a diagonalizable matrix via its eigendecomposition."""
    A = check_square(A)
    if is_symmetric(A):
        w, V = np.linalg.eigh(A)
        return (V * f(w)) @ V.T
    w, V = np.linalg.eig(A)
    out = (V * f(w)) @ np.linalg.inv(V)
    return out.real if np.allclose(out.imag, 0.0) else out
