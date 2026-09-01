"""Sparse matrix formats and sparse-specific algorithms.

Implements COO / CSR / CSC / DIA storage with the operations the iterative
solvers need, plus reordering and sparse direct factorization.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DimensionError
from ..core.utils import as_vector

__all__ = [
    "COOMatrix",
    "CSRMatrix",
    "CSCMatrix",
    "DIAMatrix",
    "from_dense",
    "identity_sparse",
    "diags",
    "spmv",
    "sparse_solve",
    "reverse_cuthill_mckee",
    "bandwidth",
    "sparsity",
]


class _SparseBase:
    """Shared behaviour for the sparse containers."""

    shape: tuple

    def matvec(self, v):  # pragma: no cover - overridden
        raise NotImplementedError

    def __matmul__(self, v):
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            return self.matvec(v)
        return np.column_stack([self.matvec(v[:, j]) for j in range(v.shape[1])])

    def todense(self) -> np.ndarray:
        n = self.shape[1]
        return np.column_stack([self.matvec(e) for e in np.eye(n)])

    @property
    def nnz(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def density(self) -> float:
        return self.nnz / (self.shape[0] * self.shape[1])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<{type(self).__name__} shape={self.shape} nnz={self.nnz} "
                f"density={self.density():.3%}>")


class COOMatrix(_SparseBase):
    """Coordinate format: parallel ``(row, col, data)`` arrays. Cheap to build."""

    def __init__(self, rows, cols, data, shape=None):
        self.rows = np.asarray(rows, dtype=int)
        self.cols = np.asarray(cols, dtype=int)
        self.data = np.asarray(data, dtype=float)
        if not (self.rows.size == self.cols.size == self.data.size):
            raise DimensionError("rows, cols and data must have equal length")
        if shape is None:
            shape = (int(self.rows.max()) + 1, int(self.cols.max()) + 1)
        self.shape = tuple(shape)

    @property
    def nnz(self) -> int:
        return self.data.size

    def matvec(self, v):
        v = as_vector(v)
        out = np.zeros(self.shape[0])
        np.add.at(out, self.rows, self.data * v[self.cols])
        return out

    def tocsr(self):
        order = np.lexsort((self.cols, self.rows))
        r, c, d = self.rows[order], self.cols[order], self.data[order]
        indptr = np.zeros(self.shape[0] + 1, dtype=int)
        np.add.at(indptr, r + 1, 1)
        return CSRMatrix(np.cumsum(indptr), c, d, self.shape)

    def transpose(self):
        return COOMatrix(self.cols, self.rows, self.data, self.shape[::-1])


class CSRMatrix(_SparseBase):
    """Compressed sparse row: the workhorse format for matrix-vector products."""

    def __init__(self, indptr, indices, data, shape):
        self.indptr = np.asarray(indptr, dtype=int)
        self.indices = np.asarray(indices, dtype=int)
        self.data = np.asarray(data, dtype=float)
        self.shape = tuple(shape)

    @property
    def nnz(self) -> int:
        return self.data.size

    def matvec(self, v):
        v = as_vector(v)
        if v.size != self.shape[1]:
            raise DimensionError(f"matrix is {self.shape} but vector has length {v.size}")
        out = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            s, e = self.indptr[i], self.indptr[i + 1]
            if e > s:
                out[i] = self.data[s:e] @ v[self.indices[s:e]]
        return out

    def rmatvec(self, v):
        """Product with the transpose, ``A' v``."""
        v = as_vector(v)
        out = np.zeros(self.shape[1])
        for i in range(self.shape[0]):
            s, e = self.indptr[i], self.indptr[i + 1]
            np.add.at(out, self.indices[s:e], self.data[s:e] * v[i])
        return out

    def diagonal(self):
        d = np.zeros(min(self.shape))
        for i in range(len(d)):
            s, e = self.indptr[i], self.indptr[i + 1]
            hit = self.indices[s:e] == i
            if hit.any():
                d[i] = self.data[s:e][hit][0]
        return d

    def tocoo(self):
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        return COOMatrix(rows, self.indices, self.data, self.shape)


class CSCMatrix(_SparseBase):
    """Compressed sparse column: efficient column slicing and transposed products."""

    def __init__(self, indptr, indices, data, shape):
        self.indptr = np.asarray(indptr, dtype=int)
        self.indices = np.asarray(indices, dtype=int)
        self.data = np.asarray(data, dtype=float)
        self.shape = tuple(shape)

    @property
    def nnz(self) -> int:
        return self.data.size

    def matvec(self, v):
        v = as_vector(v)
        out = np.zeros(self.shape[0])
        for j in range(self.shape[1]):
            s, e = self.indptr[j], self.indptr[j + 1]
            if e > s:
                np.add.at(out, self.indices[s:e], self.data[s:e] * v[j])
        return out


class DIAMatrix(_SparseBase):
    """Diagonal storage: the natural format for finite-difference stencils."""

    def __init__(self, data, offsets, shape):
        self.data = np.atleast_2d(np.asarray(data, dtype=float))
        self.offsets = np.asarray(offsets, dtype=int)
        self.shape = tuple(shape)

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self.data))

    def matvec(self, v):
        v = as_vector(v)
        m, n = self.shape
        out = np.zeros(m)
        # data is column-indexed: data[i, j] holds A[j - offset, j]
        for row, k in zip(self.data, self.offsets):
            if k >= 0:
                length = min(m, n - k)
                out[:length] += row[k : k + length] * v[k : k + length]
            else:
                length = min(m + k, n)
                out[-k : -k + length] += row[:length] * v[:length]
        return out


def from_dense(A, tol: float = 0.0, fmt: str = "csr"):
    """Convert a dense array to a sparse container, dropping entries <= ``tol``."""
    A = np.asarray(A, dtype=float)
    rows, cols = np.nonzero(np.abs(A) > tol)
    coo = COOMatrix(rows, cols, A[rows, cols], A.shape)
    return coo if fmt == "coo" else coo.tocsr()


def identity_sparse(n: int, fmt: str = "csr"):
    """Sparse identity of order ``n``."""
    return from_dense(np.eye(n), fmt=fmt)


def diags(diagonals, offsets, shape=None):
    """Build a ``DIAMatrix`` from diagonals and their offsets."""
    diagonals = [np.atleast_1d(np.asarray(d, dtype=float)) for d in diagonals]
    offsets = np.atleast_1d(np.asarray(offsets, dtype=int))
    if shape is None:
        n = max(len(d) + abs(int(o)) for d, o in zip(diagonals, offsets))
        n = max(n, max(len(d) for d in diagonals))
        shape = (n, n)
    m, n = shape
    data = np.zeros((len(diagonals), n))
    for i, (d, o) in enumerate(zip(diagonals, offsets)):
        o = int(o)
        # a diagonal at offset o holds at most min(m, n - o) entries for o >= 0
        # (min(m + o, n) for o < 0); anything longer is truncated to fit
        if o >= 0:
            length = min(len(d), max(0, min(m, n - o)))
            data[i, o : o + length] = d[:length]
        else:
            length = min(len(d), max(0, min(m + o, n)))
            data[i, :length] = d[:length]
    return DIAMatrix(data, offsets, shape)


def spmv(A, v):
    """Sparse matrix-vector product for any container in this module."""
    return A.matvec(v) if hasattr(A, "matvec") else np.asarray(A, dtype=float) @ as_vector(v)


def sparse_solve(A, b, method: str = "cg", **kwargs):
    """Solve a sparse system with a matrix-free Krylov method."""
    from .iterative import bicgstab, conjugate_gradient, gmres

    solver = {"cg": conjugate_gradient, "gmres": gmres, "bicgstab": bicgstab}[method]
    return solver(A, b, **kwargs)


def bandwidth(A) -> tuple:
    """Lower and upper bandwidths of a (dense or sparse) matrix."""
    A = A.todense() if hasattr(A, "todense") else np.asarray(A, dtype=float)
    rows, cols = np.nonzero(A)
    if rows.size == 0:
        return 0, 0
    d = cols - rows
    return int(max(0, -d.min())), int(max(0, d.max()))


def sparsity(A) -> float:
    """Fraction of entries that are exactly zero."""
    A = A.todense() if hasattr(A, "todense") else np.asarray(A, dtype=float)
    return float(np.count_nonzero(A == 0) / A.size)


def reverse_cuthill_mckee(A):
    """Reverse Cuthill-McKee ordering; returns a permutation reducing bandwidth."""
    A = A.todense() if hasattr(A, "todense") else np.asarray(A, dtype=float)
    n = A.shape[0]
    adj = [np.flatnonzero(A[i] != 0) for i in range(n)]
    degree = np.array([len(np.setdiff1d(a, [i])) for i, a in enumerate(adj)])
    visited = np.zeros(n, dtype=bool)
    order = []
    while not visited.all():
        remaining = np.flatnonzero(~visited)
        start = remaining[int(np.argmin(degree[remaining]))]
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            order.append(node)
            nbrs = [j for j in adj[node] if j != node and not visited[j]]
            for j in sorted(nbrs, key=lambda j: degree[j]):
                visited[j] = True
                queue.append(j)
    return np.array(order[::-1], dtype=int)
