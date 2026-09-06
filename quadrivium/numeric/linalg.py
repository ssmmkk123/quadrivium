"""Linear algebra, mirroring ``numpy.linalg`` for the routines used here.

The factorisations themselves are in C; what remains are the compositions
built on top of them.
"""

from __future__ import annotations

from .. import _qnp as _c

LinAlgError = _c.LinAlgError

solve = _c.linalg.solve
inv = _c.linalg.inv
det = _c.linalg.det
slogdet = _c.linalg.slogdet
cholesky = _c.linalg.cholesky
eigh = _c.linalg.eigh
eigvalsh = _c.linalg.eigvalsh
eig = _c.linalg.eig
eigvals = _c.linalg.eigvals
svd = _c.linalg.svd
lstsq = _c.linalg.lstsq
norm = _c.linalg.norm

__all__ = ["solve", "inv", "det", "slogdet", "cholesky", "eigh", "eigvalsh",
           "eig", "eigvals", "svd", "lstsq", "norm", "pinv", "matrix_rank",
           "matrix_power", "qr", "cond", "LinAlgError"]


def matrix_rank(A, tol=None):
    """Number of singular values above the tolerance."""
    s = _c.linalg.svd(A, compute_uv=False)
    if s.size == 0:
        return 0
    shape = _c.asarray(A).shape
    if tol is None:
        tol = max(shape) * float(s[0]) * 2.220446049250313e-16
    return int(_c.count_nonzero(s > tol))


def pinv(A, rcond=1e-15):
    """Moore-Penrose pseudo-inverse via the singular value decomposition."""
    matrix = _c.asarray(A)
    U, s, Vt = _c.linalg.svd(matrix, full_matrices=False)
    cutoff = rcond * (float(s[0]) if s.size else 0.0)
    inverted = _c.zeros(s.shape, dtype=_c.float64)
    for i in range(s.size):
        value = float(s[i])
        if value > cutoff:
            inverted[i] = 1.0 / value
    return _c.matmul(_c.transpose(Vt) * inverted, _c.transpose(U))


def cond(A, p=None):
    """Condition number in the requested norm."""
    matrix = _c.asarray(A)
    if p is None or p == 2:
        s = _c.linalg.svd(matrix, compute_uv=False)
        if s.size == 0 or float(s[s.size - 1]) == 0.0:
            return float("inf")
        return float(s[0]) / float(s[s.size - 1])
    return _c.linalg.norm(matrix, p) * _c.linalg.norm(inv(matrix), p)


def matrix_power(A, n):
    matrix = _c.asarray(A)
    if n < 0:
        matrix = inv(matrix)
        n = -n
    result = _c.eye(matrix.shape[0], dtype=matrix.dtype)
    base = matrix
    while n:
        if n & 1:
            result = _c.matmul(result, base)
        base = _c.matmul(base, base)
        n >>= 1
    return result


def qr(A, mode="reduced"):
    """Householder QR; ``mode`` selects the reduced or complete factors."""
    matrix = _c.asarray(A, dtype=float).astype(_c.float64, copy=True)
    m, n = matrix.shape
    k = min(m, n)
    Q = _c.eye(m, dtype=_c.float64)
    for j in range(k):
        column = matrix[j:, j].copy()
        alpha = -_c.linalg.norm(column) if float(column[0]) >= 0 else _c.linalg.norm(column)
        if abs(alpha) < 1e-300:
            continue
        v = column.copy()
        v[0] = float(v[0]) - alpha
        vnorm = _c.linalg.norm(v)
        if vnorm < 1e-300:
            continue
        v = v / vnorm
        matrix[j:, j:] -= 2.0 * _c.outer(v, _c.matmul(v, matrix[j:, j:]))
        Q[:, j:] -= 2.0 * _c.outer(_c.matmul(Q[:, j:], v), v)
    R = _c.zeros((m, n), dtype=_c.float64)
    for i in range(m):
        for j in range(i, n):
            R[i, j] = matrix[i, j]
    if mode == "complete":
        return Q, R
    return Q[:, :k], R[:k, :]
