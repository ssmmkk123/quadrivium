"""Linear algebra, mirroring ``numpy.linalg`` for the routines used here.

The factorisations themselves are in C; what remains are the compositions
built on top of them.
"""

from __future__ import annotations

from .. import _qnp as _c

LinAlgError = _c.LinAlgError

lu_factor = _c.linalg.lu_factor
lu_solve = _c.linalg.lu_solve
_solve = _c.linalg.solve
inv = _c.linalg.inv
det = _c.linalg.det
slogdet = _c.linalg.slogdet
_cholesky = _c.linalg.cholesky
_eigh = _c.linalg.eigh
_eigvalsh = _c.linalg.eigvalsh
eig = _c.linalg.eig
eigvals = _c.linalg.eigvals
_svd = _c.linalg.svd
lstsq = _c.linalg.lstsq
norm = _c.linalg.norm

__all__ = ["solve", "inv", "det", "slogdet", "cholesky", "eigh", "eigvalsh",
           "eig", "eigvals", "svd", "lstsq", "norm", "pinv", "matrix_rank",
           "matrix_power", "qr", "cond", "LinAlgError"]


def matrix_rank(A, tol=None):
    """Number of singular values above the tolerance."""
    s = svd(A, compute_uv=False)
    if s.size == 0:
        return 0
    shape = _c.asarray(A).shape
    if tol is None:
        tol = max(shape) * float(s[0]) * 2.220446049250313e-16
    return int(_c.count_nonzero(s > tol))


def pinv(A, rcond=1e-15):
    """Moore-Penrose pseudo-inverse via the singular value decomposition."""
    matrix = _c.asarray(A)
    U, s, Vt = svd(matrix, full_matrices=False)
    cutoff = rcond * (float(s[0]) if s.size else 0.0)
    inverted = _c.zeros(s.shape, dtype=_c.float64)
    for i in range(s.size):
        value = float(s[i])
        if value > cutoff:
            inverted[i] = 1.0 / value
    return _c.matmul(_c.conjugate(_c.transpose(Vt)) * inverted, _c.conjugate(_c.transpose(U)))


def cond(A, p=None):
    """Condition number in the requested norm."""
    matrix = _c.asarray(A)
    if p is None or p == 2:
        s = svd(matrix, compute_uv=False)
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


def _factor_dtype(matrix):
    if matrix.dtype in (_c.float32, _c.complex64, _c.float64, _c.complex128):
        return matrix.dtype
    return _c.float64


def _map_factors(matrix, function, shapes, dtypes):
    from ._batch import indices
    batch = matrix.shape[:-2]
    if not batch:
        result = function(matrix)
        values = result if isinstance(result, tuple) else (result,)
        values = tuple(x.astype(dt, copy=False) for x, dt in zip(values, dtypes))
    else:
        values = tuple(_c.empty(batch + sh, dtype=dt) for sh, dt in zip(shapes, dtypes))
        for index in indices(batch):
            result = function(matrix[index])
            factors = result if isinstance(result, tuple) else (result,)
            for dest, value in zip(values, factors):
                dest[index] = value
    return values[0] if len(values) == 1 else values


def _matrix(a, square=False):
    a = _c.asarray(a)
    if a.ndim < 2 or (square and a.shape[-1] != a.shape[-2]):
        raise LinAlgError("expected matrices with square final dimensions" if square else "expected matrices with at least two dimensions")
    return a


def solve(a, b, *, vector=None):
    """Solve broadcast stacks of systems using native pivoted LU.

    A one-dimensional ``b`` is a shared vector. Higher-dimensional ``b`` has
    matrix core dimensions ``(n, nrhs)``; pass ``vector=True`` for stacks of
    vectors with core dimension ``(n,)``. Broadcasting never copies ``a``.
    """
    from . import result_type
    from ._batch import indices, broadcast_shape, batch_index
    a, b = _matrix(a, square=True), _c.asarray(b)
    vector = b.ndim == 1 if vector is None else bool(vector)
    core = 1 if vector else 2
    if b.ndim < core or b.shape[-core] != a.shape[-1]:
        raise LinAlgError("incompatible right-hand side dimensions")
    batch = broadcast_shape(a.shape[:-2], b.shape[:-core])
    dt = result_type(_factor_dtype(a), _factor_dtype(b))
    if not batch:
        return _solve(a, b).astype(dt, copy=False)
    out = _c.empty(batch + b.shape[-core:], dtype=dt)
    for index in indices(batch):
        ai, bi = batch_index(index, a.shape[:-2]), batch_index(index, b.shape[:-core])
        out[index] = _solve(a[ai] if ai else a, b[bi] if bi else b)
    return out


def cholesky(a):
    """Lower Cholesky factors of broadcast real or complex Hermitian matrices."""
    a = _matrix(a, square=True)
    return _map_factors(a, _cholesky, (a.shape[-2:],), (_factor_dtype(a),))


def eigh(a):
    """Sorted eigenpairs of real symmetric or complex Hermitian matrix stacks.

    The lower triangle defines each matrix and imaginary diagonal entries
    are ignored, matching the existing real solver's lower-triangle policy.
    """
    a = _matrix(a, square=True)
    dt = _factor_dtype(a)
    real = _c.float32 if dt in (_c.float32, _c.complex64) else _c.float64
    return _map_factors(a, _eigh, ((a.shape[-1],), a.shape[-2:]), (real, dt))


def eigvalsh(a):
    a = _matrix(a, square=True)
    dt = _c.float32 if a.dtype in (_c.float32, _c.complex64) else _c.float64
    return _map_factors(a, _eigvalsh, ((a.shape[-1],),), (dt,))


def svd(a, full_matrices=True, compute_uv=True):
    """Native real/complex SVD of matrix stacks, with bounded per-matrix scratch."""
    a = _matrix(a)
    m, n = a.shape[-2:]
    k = min(m, n)
    dt = _factor_dtype(a)
    real = _c.float32 if dt in (_c.float32, _c.complex64) else _c.float64
    fn = lambda x: _svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
    if not compute_uv:
        return _map_factors(a, fn, ((k,),), (real,))
    shapes = ((m, m if full_matrices else k), (k,), (n if full_matrices else k, n))
    return _map_factors(a, fn, shapes, (dt, real, dt))


def _qr_one(a, mode):
    m, n = a.shape
    k = min(m, n)
    if a.dtype.kind != "c":
        q, r = _c._accel.householder_qr(a, want_q=mode != "r", reduced=mode != "complete")
        return r[:k] if mode == "r" else (q, r if mode == "complete" else r[:k])
    # Compact reflectors: reduced mode needs O(m*n), never an m-by-m Q.
    r = a.astype(_c.complex128, copy=True)
    reflectors = []
    for j in range(k):
        v = r[j:, j].copy()
        normv = _c.linalg.norm(v)
        if normv == 0:
            reflectors.append(None)
            continue
        alpha = -normv * (complex(v[0]) / abs(complex(v[0])) if v[0] else 1)
        v[0] -= alpha
        v /= _c.linalg.norm(v)
        r[j:, j:] -= 2 * _c.outer(v, _c.matmul(_c.conjugate(v), r[j:, j:]))
        r[j, j] = alpha
        r[j+1:, j] = 0
        reflectors.append(v)
    if mode == "r":
        return r[:k]
    qcols = m if mode == "complete" else k
    q = _c.eye(m, qcols, dtype=_c.complex128)
    for j in range(k - 1, -1, -1):
        v = reflectors[j]
        if v is not None:
            q[j:] -= 2 * _c.outer(v, _c.matmul(_c.conjugate(v), q[j:]))
    return q, r if mode == "complete" else r[:k]


def qr(a, mode="reduced"):
    """Householder QR for real/complex matrix stacks; reduced/complete/r modes."""
    if mode not in ("reduced", "complete", "r"):
        raise ValueError("mode must be 'reduced', 'complete', or 'r'")
    a = _matrix(a)
    m, n = a.shape[-2:]
    k = min(m, n)
    rows = m if mode == "complete" else k
    dt = _factor_dtype(a)
    shapes = ((rows, n),) if mode == "r" else ((m, rows), (rows, n))
    return _map_factors(a, lambda x: _qr_one(x, mode), shapes, (dt,) * len(shapes))


__all__ += ["lu_factor", "lu_solve"]
