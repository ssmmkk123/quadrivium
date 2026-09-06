"""Sparse matrix formats and sparse-specific algorithms.

Implements COO / CSR / CSC / DIA storage with the operations the iterative
solvers need, plus reordering and sparse direct factorization.
"""

from __future__ import annotations

from collections import deque
from operator import index

from .. import numeric as np

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


def _shape(shape):
    try:
        m, n = shape
        m, n = index(m), index(n)
    except (TypeError, ValueError) as exc:
        raise DimensionError("shape must contain two nonnegative integers") from exc
    if m < 0 or n < 0:
        raise DimensionError("shape must contain two nonnegative integers")
    return m, n


def _integer_vector(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise DimensionError(f"{name} must be one-dimensional")
    if raw.size and raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain integer indices")
    try:
        with np.errstate(invalid="raise", over="raise"):
            result = raw.astype(np.intp, copy=False)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name} must contain representable integer indices") from exc
    if not np.array_equal(raw, result):
        raise ValueError(f"{name} must contain representable integer indices")
    return result


def _data_vector(data):
    result = np.asarray(data, dtype=float)
    if result.ndim != 1:
        raise DimensionError("data must be one-dimensional")
    return result


def _compressed(indptr, indices, data, shape, axis):
    shape = _shape(shape)
    indptr = _integer_vector(indptr, "indptr")
    indices = _integer_vector(indices, "indices")
    data = _data_vector(data)
    if indptr.size != shape[axis] + 1:
        raise DimensionError("indptr length must equal the compressed dimension plus one")
    if indices.size != data.size:
        raise DimensionError("indices and data must have equal length")
    if (indptr[0] != 0 or indptr[-1] != data.size
            or np.any(indptr[1:] < indptr[:-1])):
        raise ValueError("indptr must start at zero, be nondecreasing and end at nnz")
    if np.any(indices < 0) or np.any(indices >= shape[1 - axis]):
        raise ValueError("sparse indices are outside the matrix shape")
    return indptr, indices, data, shape


def _vector(v, size, shape):
    v = as_vector(v)
    if v.size != size:
        raise DimensionError(f"matrix is {shape} but vector has length {v.size}")
    return v


def _segment_sum(indptr, values):
    """Sum compressed segments, including leading/interior/trailing empty ones."""
    out = np.zeros(indptr.size - 1)
    nonempty = indptr[1:] > indptr[:-1]
    if values.size:
        out[nonempty] = np.add.reduceat(values, indptr[:-1][nonempty])
    return out


class _SparseBase:
    """Shared behaviour for the sparse containers."""

    shape: tuple

    def matvec(self, v):  # pragma: no cover - overridden
        raise NotImplementedError

    def __matmul__(self, v):
        v = np.asarray(v, dtype=float)
        if v.ndim == 1:
            return self.matvec(v)
        if v.ndim != 2 or v.shape[0] != self.shape[1]:
            raise DimensionError(f"cannot multiply matrix {self.shape} by array {v.shape}")
        out = np.empty((self.shape[0], v.shape[1]))
        for j in range(v.shape[1]):
            out[:, j] = self.matvec(v[:, j])
        return out

    def todense(self) -> np.ndarray:
        out = np.zeros(self.shape)
        rows, cols, data = _coordinates(self)
        np.add.at(out, (rows, cols), data)
        return out

    @property
    def nnz(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def density(self) -> float:
        size = self.shape[0] * self.shape[1]
        return self.nnz / size if size else 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<{type(self).__name__} shape={self.shape} nnz={self.nnz} "
                f"density={self.density():.3%}>")


class COOMatrix(_SparseBase):
    """Coordinate format: parallel ``(row, col, data)`` arrays. Cheap to build."""

    def __init__(self, rows, cols, data, shape=None):
        self.rows = _integer_vector(rows, "rows")
        self.cols = _integer_vector(cols, "cols")
        self.data = _data_vector(data)
        if not (self.rows.size == self.cols.size == self.data.size):
            raise DimensionError("rows, cols and data must have equal length")
        if shape is None:
            shape = ((int(self.rows.max()) + 1, int(self.cols.max()) + 1)
                     if self.rows.size else (0, 0))
        self.shape = _shape(shape)
        if (np.any(self.rows < 0) or np.any(self.rows >= self.shape[0])
                or np.any(self.cols < 0) or np.any(self.cols >= self.shape[1])):
            raise ValueError("sparse coordinates are outside the matrix shape")

    @property
    def nnz(self) -> int:
        return self.data.size

    def matvec(self, v):
        v = _vector(v, self.shape[1], self.shape)
        return np.bincount(self.rows, weights=self.data * v[self.cols],
                           minlength=self.shape[0]).astype(float, copy=False)

    def tocsr(self):
        order = np.lexsort((self.cols, self.rows))
        r, c, d = self.rows[order], self.cols[order], self.data[order]
        indptr = np.empty(self.shape[0] + 1, dtype=np.intp)
        indptr[0] = 0
        np.cumsum(np.bincount(r, minlength=self.shape[0]), out=indptr[1:])
        return CSRMatrix(indptr, c, d, self.shape)

    def transpose(self):
        return COOMatrix(self.cols, self.rows, self.data, self.shape[::-1])


class CSRMatrix(_SparseBase):
    """Compressed sparse row: the workhorse format for matrix-vector products."""

    def __init__(self, indptr, indices, data, shape):
        self.indptr, self.indices, self.data, self.shape = _compressed(
            indptr, indices, data, shape, axis=0)

    @property
    def nnz(self) -> int:
        return self.data.size

    def matvec(self, v):
        v = _vector(v, self.shape[1], self.shape)
        return _segment_sum(self.indptr, self.data * v[self.indices])

    def rmatvec(self, v):
        """Product with the transpose, ``A' v``."""
        v = _vector(v, self.shape[0], self.shape)
        values = np.repeat(v, np.diff(self.indptr)) * self.data
        return np.bincount(self.indices, weights=values,
                           minlength=self.shape[1]).astype(float, copy=False)

    def diagonal(self):
        d = np.zeros(min(self.shape))
        for i in range(len(d)):
            s, e = self.indptr[i], self.indptr[i + 1]
            hit = self.indices[s:e] == i
            if hit.any():
                d[i] = np.sum(self.data[s:e][hit])
        return d

    def tocoo(self):
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        return COOMatrix(rows, self.indices, self.data, self.shape)


class CSCMatrix(_SparseBase):
    """Compressed sparse column: efficient column slicing and transposed products."""

    def __init__(self, indptr, indices, data, shape):
        self.indptr, self.indices, self.data, self.shape = _compressed(
            indptr, indices, data, shape, axis=1)

    @property
    def nnz(self) -> int:
        return self.data.size

    def matvec(self, v):
        v = _vector(v, self.shape[1], self.shape)
        values = np.repeat(v, np.diff(self.indptr)) * self.data
        return np.bincount(self.indices, weights=values,
                           minlength=self.shape[0]).astype(float, copy=False)

    def rmatvec(self, v):
        """Product with the transpose, ``A' v``."""
        v = _vector(v, self.shape[0], self.shape)
        return _segment_sum(self.indptr, self.data * v[self.indices])


class DIAMatrix(_SparseBase):
    """Diagonal storage: the natural format for finite-difference stencils."""

    def __init__(self, data, offsets, shape):
        self.data = np.atleast_2d(np.asarray(data, dtype=float))
        self.offsets = _integer_vector(offsets, "offsets")
        self.shape = _shape(shape)
        if self.data.ndim != 2 or self.data.shape[0] != self.offsets.size:
            raise DimensionError("DIA data must have one row per offset")

    def _diagonals(self):
        m, n = self.shape
        # data is column-indexed: data[i, j] holds A[j - offset, j].
        for row, offset in zip(self.data, self.offsets):
            k = int(offset)
            start, stop = max(0, k), min(m + k, n, row.size)
            if stop > start:
                yield k, start, stop, row[start:stop]

    @property
    def nnz(self) -> int:
        return sum(int(np.count_nonzero(d)) for _, _, _, d in self._diagonals())

    def matvec(self, v):
        v = _vector(v, self.shape[1], self.shape)
        out = np.zeros(self.shape[0])
        for k, start, stop, data in self._diagonals():
            out[start - k:stop - k] += data * v[start:stop]
        return out


def from_dense(A, tol: float = 0.0, fmt: str = "csr"):
    """Convert a dense array to a sparse container, dropping entries <= ``tol``."""
    if fmt not in ("coo", "csr", "csc"):
        raise ValueError("fmt must be 'coo', 'csr', or 'csc'")
    if tol < 0 or np.isnan(tol):
        raise ValueError("tol must be nonnegative")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise DimensionError("expected a two-dimensional dense matrix")
    rows, cols = np.nonzero(np.abs(A) > tol)
    coo = COOMatrix(rows, cols, A[rows, cols], A.shape)
    if fmt == "coo":
        return coo
    if fmt == "csc":
        transposed = coo.transpose().tocsr()
        return CSCMatrix(transposed.indptr, transposed.indices, transposed.data, A.shape)
    return coo.tocsr()


def identity_sparse(n: int, fmt: str = "csr"):
    """Sparse identity of order ``n``."""
    shape = _shape((n, n))
    if fmt not in ("coo", "csr", "csc"):
        raise ValueError("fmt must be 'coo', 'csr', or 'csc'")
    indices = np.arange(n, dtype=np.intp)
    data = np.ones(n)
    if fmt == "coo":
        return COOMatrix(indices, indices, data, shape)
    cls = CSRMatrix if fmt == "csr" else CSCMatrix
    return cls(np.arange(n + 1, dtype=np.intp), indices, data, shape)


def diags(diagonals, offsets, shape=None):
    """Build a ``DIAMatrix`` from diagonals and their offsets."""
    diagonals = [np.atleast_1d(np.asarray(d, dtype=float)) for d in diagonals]
    offsets = _integer_vector(np.atleast_1d(offsets), "offsets")
    if len(diagonals) != offsets.size:
        raise DimensionError("provide one diagonal per offset")
    if any(d.ndim != 1 for d in diagonals):
        raise DimensionError("diagonals must be one-dimensional")
    if shape is None:
        n = max((len(d) + abs(int(o)) for d, o in zip(diagonals, offsets)), default=0)
        shape = (n, n)
    m, n = shape = _shape(shape)
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


def _coordinates(A):
    """Return stored entries with O(nnz) temporary storage; preserve duplicates."""
    if isinstance(A, COOMatrix):
        return A.rows, A.cols, A.data
    if isinstance(A, CSRMatrix):
        rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
        return rows, A.indices, A.data
    if isinstance(A, CSCMatrix):
        cols = np.repeat(np.arange(A.shape[1]), np.diff(A.indptr))
        return A.indices, cols, A.data
    if isinstance(A, DIAMatrix):
        rows, cols, values = [], [], []
        for k, start, stop, data in A._diagonals():
            c = np.arange(start, stop)
            rows.append(c - k)
            cols.append(c)
            values.append(data)
        if values:
            return np.concatenate(rows), np.concatenate(cols), np.concatenate(values)
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0)
    raise TypeError(f"unsupported sparse format {type(A).__name__}")


def _nonzero_coordinates(A):
    """Actual nonzero positions, after summing duplicate sparse coordinates."""
    if not isinstance(A, _SparseBase):
        A = A.todense() if hasattr(A, "todense") else A
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise DimensionError("expected a two-dimensional matrix")
        return (*np.nonzero(A), A.shape)
    rows, cols, data = _coordinates(A)
    if not data.size:
        return rows, cols, A.shape
    order = np.lexsort((cols, rows))
    rows, cols, data = rows[order], cols[order], data[order]
    first = np.empty(data.size, dtype=bool)
    first[0] = True
    first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    starts = np.flatnonzero(first)
    # Explicit zeros and duplicate entries that cancel must not become edges.
    if starts.size == data.size:
        nonzero = data != 0
    else:
        summed = np.zeros(starts.size)
        # Match todense's accumulation order even with ill-scaled duplicates.
        np.add.at(summed, np.cumsum(first) - 1, data)
        nonzero = summed != 0
    starts = starts[nonzero]
    return rows[starts], cols[starts], A.shape


def bandwidth(A) -> tuple:
    """Lower and upper bandwidths of a (dense or sparse) matrix."""
    rows, cols, _ = _nonzero_coordinates(A)
    if rows.size == 0:
        return 0, 0
    d = cols - rows
    return int(max(0, -d.min())), int(max(0, d.max()))


def sparsity(A) -> float:
    """Fraction of entries that are exactly zero (one for an empty matrix)."""
    rows, _, shape = _nonzero_coordinates(A)
    size = shape[0] * shape[1]
    return float((size - rows.size) / size) if size else 1.0


def reverse_cuthill_mckee(A):
    """Reverse Cuthill-McKee ordering; returns a permutation reducing bandwidth."""
    rows, cols, shape = _nonzero_coordinates(A)
    if shape[0] != shape[1]:
        raise DimensionError("reverse Cuthill-McKee requires a square matrix")
    n = shape[0]
    off_diagonal = rows != cols
    rows, cols = rows[off_diagonal], cols[off_diagonal]
    degree = np.bincount(rows, minlength=n)
    indptr = np.empty(n + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    visited = np.zeros(n, dtype=bool)
    order = []
    # One sorted seed list avoids rescanning every vertex for each component.
    for start in np.argsort(degree, kind="stable"):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            order.append(node)
            nbrs = cols[indptr[node]:indptr[node + 1]]
            nbrs = nbrs[~visited[nbrs]]
            for j in sorted(nbrs, key=lambda j: degree[j]):
                visited[j] = True
                queue.append(j)
    return np.array(order[::-1], dtype=int)
