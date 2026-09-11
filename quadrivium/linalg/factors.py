"""Reusable dense factorizations; factor once, solve many right-hand sides."""
from __future__ import annotations

import math

from .. import numeric as np
from ..core.exceptions import DimensionError, SingularMatrixError

__all__ = ["LUFactor", "CholeskyFactor", "QRFactor", "lu_factor", "cholesky_factor", "qr_factor"]


def _matrix(A, square=False):
    A = np.asarray(A)
    if A.ndim != 2 or (square and A.shape[0] != A.shape[1]):
        raise DimensionError("expected a square matrix" if square else "expected a matrix")
    dtype = complex if np.iscomplexobj(A) else float
    A = np.array(A, dtype=dtype, copy=True)
    if not np.all(np.isfinite(A)):
        raise ValueError("factorization requires finite entries")
    return A


def _rhs(b, m, dtype):
    b = np.asarray(b)
    if b.ndim not in (1, 2) or b.shape[0] != m:
        raise DimensionError("right-hand side must have shape (m,) or (m, nrhs)")
    vector = b.ndim == 1
    work = np.array(b, dtype=np.result_type(b.dtype, dtype), copy=True)
    return work.reshape((m, 1)) if vector else work, vector


def _finish(x, vector, out):
    x = x[:, 0] if vector else x
    if out is not None:
        if not isinstance(out, np.ndarray) or out.shape != x.shape:
            raise DimensionError("out must be an array matching the solution shape")
        if out.dtype.kind not in "fc" or (x.dtype.kind == "c" and out.dtype.kind != "c"):
            raise TypeError("out dtype cannot represent the solution")
        out[...] = x
        return out
    return x


def _triangular(A, x, lower, unit=False, adjoint=False):
    n = A.shape[0]
    for i in (range(n) if lower else range(n - 1, -1, -1)):
        part = slice(None, i) if lower else slice(i + 1, None)
        row = np.conjugate(A[part, i]) if adjoint else A[i, part]
        if row.size:
            x[i] -= row @ x[part]
        if not unit:
            d = np.conjugate(A[i, i]) if adjoint else A[i, i]
            if d == 0:
                raise SingularMatrixError(f"zero factor diagonal at {i}")
            x[i] /= d
    return x


class LUFactor:
    """Partial-pivoted LU stored in one matrix and an O(n) permutation.

    ``solve`` uses only an RHS-sized temporary and is safe for concurrent calls.
    ``out`` allows the caller to retain its own reusable output storage.
    """
    def __init__(self, A):
        native = getattr(np.linalg, "lu_factor", None)
        if native is not None:
            source = np.asarray(A)
            if source.ndim != 2 or source.shape[0] != source.shape[1]:
                raise DimensionError("expected a square matrix")
            if not np.all(np.isfinite(source)):
                raise ValueError("factorization requires finite entries")
            self._native = native(source)
            self.packed, steps = self._native
            n = source.shape[0]
            self.shape, self.dtype = self.packed.shape, self.packed.dtype
            self.pivots = np.arange(n, dtype=int)
            self._parity = 1
            for k in range(n):
                j = int(steps[k])
                if j != k:
                    self.pivots[[k, j]] = self.pivots[[j, k]]
                    self._parity *= -1
            return
        self._native = None
        lu = _matrix(A, square=True)
        n = lu.shape[0]
        self.shape, self.dtype = lu.shape, lu.dtype
        self.pivots = np.arange(n, dtype=int)
        self._parity = 1
        for k in range(n):
            pivot = k + int(np.argmax(np.abs(lu[k:, k])))
            if lu[pivot, k] == 0:
                raise SingularMatrixError(f"singular matrix at pivot {k}")
            if pivot != k:
                lu[[k, pivot]] = lu[[pivot, k]]
                self.pivots[[k, pivot]] = self.pivots[[pivot, k]]
                self._parity *= -1
            lu[k + 1:, k] /= lu[k, k]
            # Small rank-one tiles bound scratch independently of n squared.
            for j in range(k + 1, n, 64):
                lu[k + 1:, j:j + 64] -= lu[k + 1:, k, None] * lu[k, j:j + 64]
        self.packed = lu

    def solve(self, b, out=None, trans="N"):
        """Solve A x=b, A.T x=b (``trans='T'``), or A.H x=b (``'H'``)."""
        if trans == "N" and self._native is not None:
            x = np.linalg.lu_solve(self._native, b)
            return _finish(x[:, None] if x.ndim == 1 else x, x.ndim == 1, out)
        x, vector = _rhs(b, self.shape[0], self.dtype)
        if trans == "N":
            x = x[self.pivots]
            _triangular(self.packed, x, True, unit=True)
            _triangular(self.packed, x, False)
        elif trans in ("T", "H"):
            conjugate = trans == "H"
            if not conjugate:
                x = np.conjugate(x)
            _triangular(self.packed, x, True, adjoint=True)
            _triangular(self.packed, x, False, unit=True, adjoint=True)
            result = np.empty_like(x)
            result[self.pivots] = x
            x = result if conjugate else np.conjugate(result)
        else:
            raise ValueError("trans must be 'N', 'T', or 'H'")
        return _finish(x, vector, out)

    def slogdet(self):
        diag = np.diag(self.packed)
        if not diag.size:
            return 1.0, 0.0
        magnitudes = np.abs(diag)
        return self._parity * np.prod(diag / magnitudes), float(np.sum(np.log(magnitudes)))

    def determinant(self):
        sign, logabs = self.slogdet()
        return sign * math.exp(logabs)

    __call__ = solve


class CholeskyFactor:
    """Reusable lower Cholesky factor of a positive-definite Hermitian matrix."""
    def __init__(self, A):
        A = _matrix(A, square=True)
        if not np.allclose(A, np.conjugate(A.T), rtol=1e-12, atol=1e-14):
            raise ValueError("Cholesky requires a Hermitian matrix")
        self.shape, self.dtype = A.shape, A.dtype
        self.lower = np.linalg.cholesky(A)

    def solve(self, b, out=None):
        x, vector = _rhs(b, self.shape[0], self.dtype)
        _triangular(self.lower, x, True)
        _triangular(self.lower, x, False, adjoint=True)
        return _finish(x, vector, out)

    def logdet(self):
        return 2.0 * float(np.sum(np.log(np.real(np.diag(self.lower)))))

    __call__ = solve


class QRFactor:
    """Packed Householder QR for full-column-rank least squares (m >= n).

    Reflector tails occupy the strict lower triangle; no dense Q is retained.
    Storage is O(m*n) and each solve needs O(m*nrhs) working memory.
    """
    def __init__(self, A):
        A = _matrix(A)
        m, n = A.shape
        if m < n:
            raise DimensionError("QRFactor requires m >= n")
        self.shape, self.dtype = A.shape, A.dtype
        self.tau = np.zeros(n)
        threshold = float(np.linalg.norm(A)) * np.finfo(float).eps * max(m, n)
        for k in range(n):
            v = A[k:, k].copy()
            norm = float(np.linalg.norm(v))
            if norm <= threshold:
                raise SingularMatrixError("matrix is numerically rank deficient")
            phase = v[0] / abs(v[0]) if abs(v[0]) else 1.0
            alpha = -phase * norm
            v[0] -= alpha
            v /= v[0]
            tau = 2.0 / float(np.sum(np.abs(v) ** 2))
            for j in range(k + 1, n, 64):
                block = A[k:, j:j + 64]
                block -= tau * v[:, None] * (np.conjugate(v) @ block)
            A[k, k] = alpha
            A[k + 1:, k] = v[1:]
            self.tau[k] = tau
        self.packed = A

    def solve(self, b, out=None):
        x, vector = _rhs(b, self.shape[0], self.dtype)
        n = self.shape[1]
        for k in range(n):
            v = self.packed[k:, k].copy()
            v[0] = 1
            x[k:] -= self.tau[k] * v[:, None] * (np.conjugate(v) @ x[k:])
        x = x[:n].copy()
        _triangular(self.packed[:n], x, False)
        return _finish(x, vector, out)

    __call__ = solve


def lu_factor(A):
    """Factor A once for repeated linear solves; see :class:`LUFactor`."""
    return LUFactor(A)


def cholesky_factor(A):
    """Factor a Hermitian positive-definite A for repeated solves."""
    return CholeskyFactor(A)


def qr_factor(A):
    """Factor a full-column-rank A for repeated least-squares solves."""
    return QRFactor(A)
