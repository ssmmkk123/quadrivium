"""Direct methods for dense linear systems.

Gaussian elimination in its several pivoting variants, the classical
factorizations (LU, Cholesky, LDL', QR), and the specialised band solvers.
All factorizations are written explicitly rather than delegated to LAPACK so
the algorithms themselves are inspectable.
"""

from __future__ import annotations

import numpy as np

from .. import _accel

# Several routines here offer their work to the compiled backend first. The
# Python implementation underneath is the definition of what the kernel must
# compute, stays under test, and runs whenever no extension is loaded --
# see `quadrivium.accel` and `tests/test_accel.py`.
from ..core.exceptions import DimensionError, SingularMatrixError
from ..core.utils import as_matrix, as_vector, check_square, is_symmetric

__all__ = [
    "forward_substitution",
    "back_substitution",
    "gauss_elimination",
    "gauss_jordan",
    "lu_decomposition",
    "lu_solve",
    "plu_decomposition",
    "plu_solve",
    "lu_complete_pivot",
    "cholesky",
    "cholesky_solve",
    "ldl_decomposition",
    "ldl_solve",
    "crout",
    "doolittle",
    "gram_schmidt_qr",
    "modified_gram_schmidt_qr",
    "householder_qr",
    "givens_qr",
    "qr_solve",
    "hessenberg",
    "bidiagonalize",
    "thomas",
    "banded_solve",
    "block_tridiagonal_solve",
    "solve",
    "inverse",
    "determinant",
    "rank",
    "nullspace",
    "sherman_morrison",
    "woodbury",
]


# --------------------------------------------------------------------------
# Triangular solves
# --------------------------------------------------------------------------
def forward_substitution(L, b, unit_diagonal: bool = False) -> np.ndarray:
    """Solve ``L x = b`` for lower-triangular ``L``."""
    L = check_square(L)
    b = as_vector(b)
    n = L.shape[0]
    if b.size != n:
        raise DimensionError(f"L is {n}x{n} but b has length {b.size}")
    fast = _accel.kernel("forward_substitution")
    if fast is not None and n:
        try:
            return fast(np.ascontiguousarray(L, dtype=float), np.ascontiguousarray(b, dtype=float), unit_diagonal)
        except Exception as exc:  # noqa: BLE001
            err = _accel.translate_error(exc, SingularMatrixError, SingularMatrixError)
            if err is None:
                raise
            raise err from None
    x = np.zeros(n)
    for i in range(n):
        s = b[i] - L[i, :i] @ x[:i]
        if unit_diagonal:
            x[i] = s
        else:
            if L[i, i] == 0.0:
                raise SingularMatrixError(f"zero diagonal entry at row {i}")
            x[i] = s / L[i, i]
    return x


def back_substitution(U, b, unit_diagonal: bool = False) -> np.ndarray:
    """Solve ``U x = b`` for upper-triangular ``U``."""
    U = check_square(U)
    b = as_vector(b)
    n = U.shape[0]
    if b.size != n:
        raise DimensionError(f"U is {n}x{n} but b has length {b.size}")
    fast = _accel.kernel("back_substitution")
    if fast is not None and n:
        # ``cholesky_solve`` passes ``L.T``, which is a transposed view and so
        # F-contiguous. Handing the kernel the underlying C-contiguous buffer
        # with a transpose flag avoids copying the whole matrix for what is
        # only an O(n^2) solve.
        mat, transposed = U, False
        if not U.flags.c_contiguous and U.flags.f_contiguous:
            mat, transposed = U.T, True
        try:
            return fast(np.ascontiguousarray(mat, dtype=float),
                        np.ascontiguousarray(b, dtype=float),
                        unit_diagonal, transposed)
        except Exception as exc:  # noqa: BLE001
            err = _accel.translate_error(exc, SingularMatrixError, SingularMatrixError)
            if err is None:
                raise
            raise err from None
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i] - U[i, i + 1 :] @ x[i + 1 :]
        if unit_diagonal:
            x[i] = s
        else:
            if U[i, i] == 0.0:
                raise SingularMatrixError(f"zero diagonal entry at row {i}")
            x[i] = s / U[i, i]
    return x


# --------------------------------------------------------------------------
# Gaussian elimination
# --------------------------------------------------------------------------
def gauss_elimination(A, b, pivoting: str = "partial"):
    """Solve ``A x = b`` by Gaussian elimination.

    ``pivoting`` selects ``'none'``, ``'partial'``, ``'scaled'`` (scaled partial
    pivoting) or ``'complete'`` (full pivoting, which also permutes columns).
    """
    A = check_square(A).copy()
    b = as_vector(b).copy()
    n = A.shape[0]
    col_order = np.arange(n)
    scale = np.max(np.abs(A), axis=1) if pivoting == "scaled" else None

    for k in range(n - 1):
        if pivoting == "partial":
            p = k + int(np.argmax(np.abs(A[k:, k])))
        elif pivoting == "scaled":
            ratios = np.abs(A[k:, k]) / np.where(scale[k:] == 0, 1.0, scale[k:])
            p = k + int(np.argmax(ratios))
        elif pivoting == "complete":
            sub = np.abs(A[k:, k:])
            p_rel, q_rel = np.unravel_index(int(np.argmax(sub)), sub.shape)
            p, q = k + p_rel, k + q_rel
            if q != k:
                A[:, [k, q]] = A[:, [q, k]]
                col_order[[k, q]] = col_order[[q, k]]
        elif pivoting == "none":
            p = k
        else:
            raise ValueError(f"unknown pivoting strategy {pivoting!r}")

        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
            if scale is not None:
                scale[[k, p]] = scale[[p, k]]
        if A[k, k] == 0.0:
            raise SingularMatrixError("matrix is singular to working precision")
        m = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k:] -= np.outer(m, A[k, k:])
        b[k + 1 :] -= m * b[k]
        A[k + 1 :, k] = 0.0

    if A[n - 1, n - 1] == 0.0:
        raise SingularMatrixError("matrix is singular to working precision")
    x = back_substitution(A, b)
    if pivoting == "complete":
        out = np.empty(n)
        out[col_order] = x
        return out
    return x


def gauss_jordan(A, b=None):
    """Gauss-Jordan elimination to reduced row echelon form.

    With ``b`` given returns the solution vector; without it returns ``A^-1``.
    """
    A = check_square(A)
    n = A.shape[0]
    rhs = np.eye(n) if b is None else as_vector(b).reshape(n, 1)
    M = np.hstack([A.astype(float), rhs])
    for k in range(n):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if abs(M[p, k]) < 1e-300:
            raise SingularMatrixError("matrix is singular to working precision")
        if p != k:
            M[[k, p]] = M[[p, k]]
        M[k] /= M[k, k]
        # Eliminate column k from every other row at once. Row by row this is
        # the same arithmetic -- a row with a zero in column k subtracts zero
        # either way -- but as one rank-1 update it is a single BLAS call
        # instead of n Python iterations.
        col = M[:, k].copy()
        col[k] = 0.0
        M -= np.outer(col, M[k])
    return M[:, n:] if b is None else M[:, n]


# --------------------------------------------------------------------------
# LU family
# --------------------------------------------------------------------------
def lu_decomposition(A):
    """Unpivoted ``A = L U`` (Doolittle: unit diagonal on ``L``)."""
    A = check_square(A)
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    for k in range(n - 1):
        if U[k, k] == 0.0:
            raise SingularMatrixError(f"zero pivot at step {k}; use plu_decomposition")
        L[k + 1 :, k] = U[k + 1 :, k] / U[k, k]
        U[k + 1 :, k:] -= np.outer(L[k + 1 :, k], U[k, k:])
        U[k + 1 :, k] = 0.0
    return L, U


def plu_decomposition(A):
    """Partially pivoted ``P A = L U``; returns ``(P, L, U)``."""
    A = check_square(A)
    n = A.shape[0]
    fast = _accel.kernel("plu")
    if fast is not None and n:
        perm, LU = fast(np.ascontiguousarray(A, dtype=float))
        L = np.tril(LU, -1) + np.eye(n)
        U = np.triu(LU)
        return np.eye(n)[perm], L, U
    U = A.astype(float).copy()
    L = np.eye(n)
    perm = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p], k:] = U[[p, k], k:]
            L[[k, p], :k] = L[[p, k], :k]
            perm[[k, p]] = perm[[p, k]]
        if U[k, k] == 0.0:
            continue
        L[k + 1 :, k] = U[k + 1 :, k] / U[k, k]
        U[k + 1 :, k:] -= np.outer(L[k + 1 :, k], U[k, k:])
        U[k + 1 :, k] = 0.0
    P = np.eye(n)[perm]
    return P, L, U


def lu_complete_pivot(A):
    """Complete pivoting ``P A Q = L U``; returns ``(P, L, U, Q)``."""
    A = check_square(A)
    n = A.shape[0]
    U = A.astype(float).copy()
    L = np.eye(n)
    rperm, cperm = np.arange(n), np.arange(n)
    for k in range(n - 1):
        sub = np.abs(U[k:, k:])
        i_rel, j_rel = np.unravel_index(int(np.argmax(sub)), sub.shape)
        i, j = k + i_rel, k + j_rel
        if i != k:
            U[[k, i], :] = U[[i, k], :]
            L[[k, i], :k] = L[[i, k], :k]
            rperm[[k, i]] = rperm[[i, k]]
        if j != k:
            U[:, [k, j]] = U[:, [j, k]]
            cperm[[k, j]] = cperm[[j, k]]
        if U[k, k] == 0.0:
            continue
        L[k + 1 :, k] = U[k + 1 :, k] / U[k, k]
        U[k + 1 :, k:] -= np.outer(L[k + 1 :, k], U[k, k:])
        U[k + 1 :, k] = 0.0
    P = np.eye(n)[rperm]
    Q = np.eye(n)[:, cperm]
    return P, L, U, Q


def lu_solve(L, U, b, P=None) -> np.ndarray:
    """Solve using a precomputed LU (optionally PLU) factorization."""
    b = as_vector(b)
    rhs = P @ b if P is not None else b
    y = forward_substitution(L, rhs, unit_diagonal=True)
    return back_substitution(U, y)


def plu_solve(A, b) -> np.ndarray:
    """Factor with partial pivoting and solve in one call."""
    P, L, U = plu_decomposition(A)
    if np.any(np.abs(np.diag(U)) < 1e-300):
        raise SingularMatrixError("matrix is singular to working precision")
    return lu_solve(L, U, b, P)


def doolittle(A):
    """Doolittle factorization (unit diagonal on ``L``) computed by inner products."""
    A = check_square(A)
    n = A.shape[0]
    L, U = np.eye(n), np.zeros((n, n))
    for i in range(n):
        # Both inner loops are inner products against the already-computed
        # leading block, so each becomes one matrix-vector product.
        U[i, i:] = A[i, i:] - L[i, :i] @ U[:i, i:]
        if i + 1 < n:
            if U[i, i] == 0.0:
                raise SingularMatrixError(f"zero pivot at step {i}")
            L[i + 1:, i] = (A[i + 1:, i] - L[i + 1:, :i] @ U[:i, i]) / U[i, i]
    return L, U


def crout(A):
    """Crout factorization (unit diagonal on ``U``)."""
    A = check_square(A)
    n = A.shape[0]
    L, U = np.zeros((n, n)), np.eye(n)
    for j in range(n):
        L[j:, j] = A[j:, j] - L[j:, :j] @ U[:j, j]
        if L[j, j] == 0.0:
            raise SingularMatrixError(f"zero pivot at step {j}")
        if j + 1 < n:
            U[j, j + 1:] = (A[j, j + 1:] - L[j, :j] @ U[:j, j + 1:]) / L[j, j]
    return L, U


# --------------------------------------------------------------------------
# Symmetric factorizations
# --------------------------------------------------------------------------
def cholesky(A, lower: bool = True) -> np.ndarray:
    """Cholesky factor of a symmetric positive definite matrix."""
    A = check_square(A)
    n = A.shape[0]
    fast = _accel.kernel("cholesky")
    if fast is not None and n:
        try:
            L = fast(np.ascontiguousarray(A, dtype=float))
        except Exception as exc:  # noqa: BLE001 - re-raised as our own type below
            err = _accel.translate_error(exc, SingularMatrixError, SingularMatrixError)
            if err is None:
                raise
            raise err from None
        return L if lower else L.T
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0.0:
                    raise SingularMatrixError(
                        "matrix is not positive definite "
                        f"(non-positive pivot {s:.3e} at index {i})"
                    )
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L if lower else L.T


def cholesky_solve(A, b) -> np.ndarray:
    """Solve an SPD system via Cholesky."""
    L = cholesky(A)
    y = forward_substitution(L, b)
    return back_substitution(L.T, y)


def ldl_decomposition(A):
    """``A = L D L'`` for symmetric (possibly indefinite) ``A``; returns ``(L, d)``."""
    A = check_square(A)
    if not is_symmetric(A, tol=1e-10):
        raise ValueError("LDL' requires a symmetric matrix")
    n = A.shape[0]
    L = np.eye(n)
    d = np.zeros(n)
    for j in range(n):
        # w = L[j, :j] * d[:j] appears in both the pivot and every entry of
        # the column below it, so it is formed once and the column becomes a
        # single matrix-vector product.
        w = L[j, :j] * d[:j]
        d[j] = A[j, j] - L[j, :j] @ w
        if d[j] == 0.0:
            raise SingularMatrixError(f"zero pivot at step {j}")
        if j + 1 < n:
            L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ w) / d[j]
    return L, d


def ldl_solve(A, b) -> np.ndarray:
    """Solve a symmetric system via the ``L D L'`` factorization."""
    L, d = ldl_decomposition(A)
    y = forward_substitution(L, b, unit_diagonal=True)
    z = y / d
    return back_substitution(L.T, z, unit_diagonal=True)


# --------------------------------------------------------------------------
# Orthogonal factorizations
# --------------------------------------------------------------------------
def gram_schmidt_qr(A):
    """Classical Gram-Schmidt ``A = Q R`` (numerically the weakest variant)."""
    A = as_matrix(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        # Classical Gram-Schmidt projects against the *original* column, so
        # every coefficient is known before any subtraction: the projection is
        # one matrix-vector product and the subtraction one more.
        R[:j, j] = Q[:, :j].T @ A[:, j]
        v = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j] if R[j, j] > 1e-300 else 0.0
    return Q, R


def modified_gram_schmidt_qr(A):
    """Modified Gram-Schmidt: same result, far better orthogonality."""
    A = as_matrix(A)
    m, n = A.shape
    V = A.astype(float).copy()
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j] if R[j, j] > 1e-300 else 0.0
        # The remaining columns are orthogonalized against q_j independently of
        # one another, so the whole trailing block is one rank-1 update. This
        # is still MGS -- the update uses the *current* V, re-read at each j.
        if j + 1 < n:
            R[j, j + 1:] = Q[:, j] @ V[:, j + 1:]
            V[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
    return Q, R


def householder_qr(A, reduced: bool = True):
    """Householder reflections ``A = Q R`` (backward stable)."""
    A = as_matrix(A)
    m, n = A.shape
    fast = _accel.kernel("householder_qr")
    if fast is not None and m and n:
        Q, R = fast(np.ascontiguousarray(A, dtype=float), True)
        if reduced and m > n:
            return Q[:, :n].copy(), R[:n, :].copy()
        return Q, R
    R = A.astype(float).copy()
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if normx < 1e-300:
            continue
        v = x.copy()
        v[0] += np.sign(x[0]) * normx if x[0] != 0 else normx
        vn = np.linalg.norm(v)
        if vn < 1e-300:
            continue
        v /= vn
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)
    if reduced and m > n:
        return Q[:, :n], R[:n, :]
    return Q, R


def givens_rotation(a: float, b: float):
    """Return ``(c, s)`` zeroing ``b`` in ``[a, b]`` (numerically safe form)."""
    if b == 0.0:
        return 1.0, 0.0
    if abs(b) > abs(a):
        tau = -a / b
        s = 1.0 / np.sqrt(1.0 + tau * tau)
        return s * tau, s
    tau = -b / a
    c = 1.0 / np.sqrt(1.0 + tau * tau)
    return c, c * tau


def givens_qr(A):
    """QR by Givens rotations; ideal for sparse or nearly-triangular matrices."""
    A = as_matrix(A)
    m, n = A.shape
    fast = _accel.kernel("givens_qr")
    if fast is not None and m and n:
        return fast(np.ascontiguousarray(A, dtype=float))
    R = A.astype(float).copy()
    Q = np.eye(m)
    for j in range(n):
        for i in range(m - 1, j, -1):
            if R[i, j] == 0.0:
                continue
            c, s = givens_rotation(R[i - 1, j], R[i, j])
            G = np.array([[c, -s], [s, c]])
            R[[i - 1, i], j:] = G @ R[[i - 1, i], j:]
            Q[:, [i - 1, i]] = Q[:, [i - 1, i]] @ G.T
    return Q, R


def qr_solve(A, b, method: str = "householder") -> np.ndarray:
    """Least-squares / square solve through a QR factorization."""
    A = as_matrix(A)
    b = as_vector(b)
    factor = {
        "householder": householder_qr,
        "givens": givens_qr,
        "gram_schmidt": gram_schmidt_qr,
        "mgs": modified_gram_schmidt_qr,
    }[method]
    Q, R = factor(A)
    n = A.shape[1]
    return back_substitution(R[:n, :n], (Q.T @ b)[:n])


def hessenberg(A, compute_q: bool = True):
    """Reduce ``A`` to upper Hessenberg form by Householder similarity."""
    A = check_square(A)
    n = A.shape[0]
    H = A.astype(float).copy()
    Q = np.eye(n)
    for k in range(n - 2):
        x = H[k + 1 :, k]
        normx = np.linalg.norm(x)
        if normx < 1e-300:
            continue
        v = x.copy()
        v[0] += np.sign(x[0]) * normx if x[0] != 0 else normx
        vn = np.linalg.norm(v)
        if vn < 1e-300:
            continue
        v /= vn
        H[k + 1 :, :] -= 2.0 * np.outer(v, v @ H[k + 1 :, :])
        H[:, k + 1 :] -= 2.0 * np.outer(H[:, k + 1 :] @ v, v)
        if compute_q:
            Q[:, k + 1 :] -= 2.0 * np.outer(Q[:, k + 1 :] @ v, v)
    H[np.tril_indices(n, -2)] = 0.0
    return (H, Q) if compute_q else H


def bidiagonalize(A):
    """Golub-Kahan bidiagonalization ``A = U B V'`` with ``B`` upper bidiagonal."""
    A = as_matrix(A)
    m, n = A.shape
    B = A.astype(float).copy()
    U, V = np.eye(m), np.eye(n)
    for k in range(min(m, n)):
        x = B[k:, k]
        if np.linalg.norm(x[1:]) > 1e-300:
            v = x.copy()
            v[0] += np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else np.linalg.norm(x)
            v /= np.linalg.norm(v)
            B[k:, k:] -= 2.0 * np.outer(v, v @ B[k:, k:])
            U[:, k:] -= 2.0 * np.outer(U[:, k:] @ v, v)
        if k < n - 2:
            y = B[k, k + 1 :]
            if np.linalg.norm(y[1:]) > 1e-300:
                w = y.copy()
                w[0] += np.sign(y[0]) * np.linalg.norm(y) if y[0] != 0 else np.linalg.norm(y)
                w /= np.linalg.norm(w)
                B[:, k + 1 :] -= 2.0 * np.outer(B[:, k + 1 :] @ w, w)
                V[:, k + 1 :] -= 2.0 * np.outer(V[:, k + 1 :] @ w, w)
    return U, B, V


# --------------------------------------------------------------------------
# Structured systems
# --------------------------------------------------------------------------
def thomas(a, b, c, d) -> np.ndarray:
    """Thomas algorithm for tridiagonal systems.

    ``a`` sub-diagonal (length n-1 or n with a[0] ignored), ``b`` diagonal,
    ``c`` super-diagonal, ``d`` right-hand side. Runs in O(n).

    ``d`` may also be a 2-D array holding one right-hand side per *column*, in
    which case the returned array has the same shape. The elimination
    coefficients depend only on the matrix, so a whole block of systems costs
    barely more than one -- which is what alternating-direction and
    line-relaxation schemes need, where the same tridiagonal matrix is solved
    once per grid line.
    """
    if np.ndim(d) == 2:
        return _thomas_many(a, b, c, d)
    b = as_vector(b)
    d = as_vector(d)
    n = b.size
    a = as_vector(a)
    c = as_vector(c)
    a = a[-(n - 1) :] if a.size >= n else a
    c = c[: n - 1] if c.size >= n else c

    fast = _accel.kernel("thomas")
    if fast is not None and n:
        try:
            return fast(a, b, c, d)
        except RuntimeError as exc:
            raise (_accel.translate_error(exc, SingularMatrixError, SingularMatrixError)
                   or exc) from None

    b = b.copy()
    d = d.copy()
    for i in range(1, n):
        if b[i - 1] == 0.0:
            raise SingularMatrixError("zero pivot in Thomas algorithm")
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    if b[n - 1] == 0.0:
        raise SingularMatrixError("zero pivot in Thomas algorithm")
    x = np.zeros(n)
    x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def _thomas_many(a, b, c, D):
    """Solve one tridiagonal system against every column of ``D``."""
    b = as_vector(b)
    n = b.size
    a = as_vector(a)
    c = as_vector(c)
    a = a[-(n - 1):] if a.size >= n else a
    c = c[: n - 1] if c.size >= n else c
    D = np.asarray(D, dtype=float)
    if D.shape[0] != n:
        raise DimensionError("right-hand side must have one row per diagonal entry")
    if not n or not D.shape[1]:
        return np.zeros(D.shape)

    fast = _accel.kernel("thomas_batch")
    if fast is not None:
        # The kernel solves in place, so it gets its own buffer: an
        # `ascontiguousarray` of an already-contiguous input is the caller's
        # array, and writing through it would destroy their right-hand side.
        out = np.array(D, dtype=float, copy=True, order="C")
        try:
            fast(a, b, c, out)
        except RuntimeError as exc:
            raise (_accel.translate_error(exc, SingularMatrixError, SingularMatrixError)
                   or exc) from None
        return out

    # Same recurrence as the single-system routine, with the right-hand side
    # step applied to every column at once.
    diag = b.astype(float, copy=True)
    X = np.array(D, dtype=float, copy=True)
    for i in range(1, n):
        if diag[i - 1] == 0.0:
            raise SingularMatrixError("zero pivot in Thomas algorithm")
        m = a[i - 1] / diag[i - 1]
        diag[i] -= m * c[i - 1]
        X[i] -= m * X[i - 1]
    if diag[n - 1] == 0.0:
        raise SingularMatrixError("zero pivot in Thomas algorithm")
    X[n - 1] /= diag[n - 1]
    for i in range(n - 2, -1, -1):
        X[i] = (X[i] - c[i] * X[i + 1]) / diag[i]
    return X


def banded_solve(A, b, kl: int, ku: int) -> np.ndarray:
    """Banded Gaussian elimination with partial pivoting.

    ``kl``/``ku`` are the lower/upper bandwidths of the dense matrix ``A``.
    """
    A = check_square(A).copy()
    b = as_vector(b).copy()
    n = A.shape[0]
    fill = ku + kl
    for k in range(n - 1):
        rows = min(k + kl + 1, n)
        p = k + int(np.argmax(np.abs(A[k:rows, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        if A[k, k] == 0.0:
            raise SingularMatrixError("singular banded matrix")
        cols = min(k + fill + 1, n)
        m = A[k + 1 : rows, k] / A[k, k]
        A[k + 1 : rows, k:cols] -= np.outer(m, A[k, k:cols])
        b[k + 1 : rows] -= m * b[k]
        A[k + 1 : rows, k] = 0.0
    return back_substitution(A, b)


def block_tridiagonal_solve(A_blocks, B_blocks, C_blocks, d_blocks):
    """Block Thomas algorithm.

    ``B_blocks`` are the diagonal blocks, ``A_blocks`` the sub-diagonal blocks
    and ``C_blocks`` the super-diagonal blocks.
    """
    n = len(B_blocks)
    B = [np.array(b, dtype=float) for b in B_blocks]
    d = [np.atleast_1d(np.array(v, dtype=float)) for v in d_blocks]
    C = [np.array(c, dtype=float) for c in C_blocks]
    A = [np.array(a, dtype=float) for a in A_blocks]
    Cp = [None] * n
    dp = [None] * n
    Cp[0] = np.linalg.solve(B[0], C[0]) if n > 1 else None
    dp[0] = np.linalg.solve(B[0], d[0])
    for i in range(1, n):
        M = B[i] - A[i - 1] @ Cp[i - 1] if Cp[i - 1] is not None else B[i]
        if i < n - 1:
            Cp[i] = np.linalg.solve(M, C[i])
        dp[i] = np.linalg.solve(M, d[i] - A[i - 1] @ dp[i - 1])
    x = [None] * n
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - Cp[i] @ x[i + 1]
    return np.concatenate(x)


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------
def solve(A, b, method: str = "auto") -> np.ndarray:
    """Solve ``A x = b``, choosing a factorization automatically by default.

    ``auto`` uses Cholesky for SPD matrices, the Thomas algorithm for
    tridiagonal ones, and pivoted LU otherwise.
    """
    A = check_square(A)
    if method == "auto":
        n = A.shape[0]
        # A tridiagonal matrix has at most 3n-2 nonzeros, and counting them
        # costs one streaming pass. The exact test below allocates three n x n
        # temporaries, so it is worth guarding for the dense case.
        if (n > 2 and np.count_nonzero(A) <= 3 * n - 2
                and np.count_nonzero(A - np.triu(np.tril(A, 1), -1)) == 0):
            return thomas(np.diag(A, -1), np.diag(A), np.diag(A, 1), b)
        if is_symmetric(A):
            try:
                return cholesky_solve(A, b)
            except SingularMatrixError:
                pass
        return plu_solve(A, b)
    return {
        "lu": plu_solve,
        "plu": plu_solve,
        "gauss": gauss_elimination,
        "gauss_jordan": gauss_jordan,
        "cholesky": cholesky_solve,
        "ldl": ldl_solve,
        "qr": qr_solve,
    }[method](A, b)


def inverse(A) -> np.ndarray:
    """Matrix inverse via Gauss-Jordan elimination."""
    return gauss_jordan(A)


def determinant(A) -> float:
    """Determinant from the pivoted LU factorization."""
    A = check_square(A)
    P, L, U = plu_decomposition(A)
    sign = np.linalg.det(P)
    return float(sign * np.prod(np.diag(U)))


def rank(A, tol=None) -> int:
    """Numerical rank from the singular values."""
    A = as_matrix(A)
    s = np.linalg.svd(A, compute_uv=False)
    if tol is None:
        tol = max(A.shape) * (s[0] if s.size else 0.0) * np.finfo(float).eps
    return int(np.sum(s > tol))


def nullspace(A, tol=None) -> np.ndarray:
    """Orthonormal basis for the null space, from the SVD."""
    A = as_matrix(A)
    U, s, Vt = np.linalg.svd(A)
    if tol is None:
        tol = max(A.shape) * (s[0] if s.size else 0.0) * np.finfo(float).eps
    ns = Vt[np.sum(s > tol) :]
    return ns.T


def sherman_morrison(Ainv, u, v) -> np.ndarray:
    """Inverse of the rank-one update ``A + u v'`` given ``A^-1``."""
    Ainv = check_square(Ainv)
    u, v = as_vector(u), as_vector(v)
    denom = 1.0 + v @ Ainv @ u
    if abs(denom) < 1e-300:
        raise SingularMatrixError("Sherman-Morrison update is singular")
    return Ainv - np.outer(Ainv @ u, v @ Ainv) / denom


def woodbury(Ainv, U, Cinv, V) -> np.ndarray:
    """Inverse of ``A + U C V`` given ``A^-1`` and ``C^-1`` (Woodbury identity)."""
    Ainv = check_square(Ainv)
    U, V = as_matrix(U), as_matrix(V)
    inner = Cinv + V @ Ainv @ U
    return Ainv - Ainv @ U @ np.linalg.solve(inner, V @ Ainv)
