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

from .. import _qnp as _core

# Optional compiled kernels; the array formulations below stay as the fallback.
_csr_matvec = getattr(_core.linalg, "csr_matvec", None)
_csr_validate = getattr(_core.linalg, "csr_validate", None)

# Bound vectorized diagonal lookups by both row count and stored entries.
_DIAGONAL_TILE_ROWS = 8192
_DIAGONAL_TILE_ELEMENTS = 131072

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
    "spmm",
    "sparse_triangular_solve",
    "ilu_preconditioner",
    "ichol_preconditioner",
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
    if raw.dtype == np.intp:
        # Already the index type: the cast below is a no-op and the round-trip
        # comparison would only be checking the array against itself.
        return raw
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
    # Three array comparisons here each allocate a full-length boolean; the
    # compiled check makes the same decisions in a single allocation-free pass.
    if _csr_validate is not None:
        _csr_validate(indptr, indices, shape[1 - axis])
    else:
        if (indptr[0] != 0 or indptr[-1] != data.size
                or np.any(indptr[1:] < indptr[:-1])):
            raise ValueError("indptr must start at zero, be nondecreasing and end at nnz")
        if np.any(indices < 0) or np.any(indices >= shape[1 - axis]):
            raise ValueError("sparse indices are outside the matrix shape")
    return indptr, indices, data, shape


def _vector(v, size, shape):
    v = np.asarray(v)
    if v.ndim != 1:
        raise DimensionError("expected a one-dimensional vector")
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

    @classmethod
    def _from_parts(cls, indptr, indices, data, shape):
        """Adopt arrays this module built itself, skipping revalidation.

        Only for structures whose validity follows from how they were
        constructed; anything reaching the public constructor is still checked.
        """
        self = cls.__new__(cls)
        self.indptr, self.indices, self.data, self.shape = indptr, indices, data, shape
        return self

    def __matmul__(self, v):
        if isinstance(v, _SparseBase):
            return _sparse_product(self, v)
        v = np.asarray(v)
        if v.ndim == 1:
            return self.matvec(v)
        if v.ndim != 2 or v.shape[0] != self.shape[1]:
            raise DimensionError(f"cannot multiply matrix {self.shape} by array {v.shape}")
        out = np.empty((self.shape[0], v.shape[1]), dtype=np.result_type(self.dtype, v.dtype))
        for j in range(v.shape[1]):
            out[:, j] = self.matvec(v[:, j])
        return out

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    @property
    def H(self):
        # Sparse storage is currently real-valued.
        return self.transpose()

    def transpose(self):
        if isinstance(self, CSRMatrix):
            return CSCMatrix(self.indptr, self.indices, self.data, self.shape[::-1])
        if isinstance(self, CSCMatrix):
            return CSRMatrix(self.indptr, self.indices, self.data, self.shape[::-1])
        rows, cols, data = _coordinates(self)
        return COOMatrix(cols, rows, data, self.shape[::-1])

    def rmatvec(self, v):
        return self.transpose().matvec(v)

    def matmat(self, v):
        return self @ v

    def __rmatmul__(self, v):
        v = np.asarray(v)
        return (self.T @ v.T).T

    def tocsr(self):
        if isinstance(self, CSRMatrix):
            return self
        rows, cols, data = _coordinates(self)
        return COOMatrix(rows, cols, data, self.shape).tocsr()

    def tocsc(self):
        if isinstance(self, CSCMatrix):
            return self
        transposed = self.transpose().tocsr()
        return CSCMatrix(transposed.indptr, transposed.indices, transposed.data, self.shape)

    def diagonal(self):
        return self.tocsr().diagonal()

    def sum_duplicates(self):
        """Return sorted canonical CSR storage, combining duplicate entries."""
        return _rows_to_csr(_row_dicts(self), self.shape)

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
        if np.iscomplexobj(v):
            return self.matvec(v.real) + 1j * self.matvec(v.imag)
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
        if np.iscomplexobj(v):
            return self.matvec(v.real) + 1j * self.matvec(v.imag)
        # The compiled kernel walks the rows once; the array formulation below
        # first materialises the gather and the product, two nnz-sized
        # temporaries that dominate both the time and the memory at scale.
        if (_csr_matvec is not None and self.data.dtype == np.float64
                and v.dtype == np.float64):
            return _csr_matvec(self.indptr, self.indices, self.data, v)
        return _segment_sum(self.indptr, self.data * v[self.indices])

    def rmatvec(self, v):
        """Product with the transpose, ``A' v``."""
        v = _vector(v, self.shape[0], self.shape)
        if np.iscomplexobj(v):
            return self.rmatvec(v.real) + 1j * self.rmatvec(v.imag)
        values = np.repeat(v, np.diff(self.indptr)) * self.data
        return np.bincount(self.indices, weights=values,
                           minlength=self.shape[1]).astype(float, copy=False)

    def diagonal(self):
        d = np.zeros(min(self.shape))
        start = 0
        while start < d.size:
            stop = min(start + _DIAGONAL_TILE_ROWS, d.size)
            s = int(self.indptr[start])
            if self.indptr[stop] - s > _DIAGONAL_TILE_ELEMENTS:
                stop = max(start + 1, start + int(np.searchsorted(
                    self.indptr[start:stop + 1], s + _DIAGONAL_TILE_ELEMENTS,
                    side="right")) - 1)
            e = int(self.indptr[stop])
            if stop == start + 1:
                # A single very long row needs no expanded row-index array.
                hit = self.indices[s:e] == start
                if hit.any():
                    d[start] = np.sum(self.data[s:e][hit])
            elif e > s:
                columns = self.indices[s:e]
                candidate = (columns >= start) & (columns < stop)
                positions = np.flatnonzero(candidate) + s
                rows = columns[candidate]
                # Column r is diagonal precisely when its storage position
                # lies in row r. This avoids expanding every row index.
                hit = ((self.indptr[rows] <= positions)
                       & (positions < self.indptr[rows + 1]))
                rows = rows[hit]
                d[rows] = self.data[positions[hit]] + 0.0
                # Usually a row has at most one diagonal entry. Duplicates
                # retain the original pairwise sum, including cancellation.
                repeated = rows[1:][rows[1:] == rows[:-1]]
                for row in np.unique(repeated):
                    rs, re = self.indptr[row], self.indptr[row + 1]
                    d[row] = np.sum(self.data[rs:re][self.indices[rs:re] == row])
            start = stop
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
        if np.iscomplexobj(v):
            return self.matvec(v.real) + 1j * self.matvec(v.imag)
        values = np.repeat(v, np.diff(self.indptr)) * self.data
        return np.bincount(self.indices, weights=values,
                           minlength=self.shape[0]).astype(float, copy=False)

    def rmatvec(self, v):
        """Product with the transpose, ``A' v``."""
        v = _vector(v, self.shape[0], self.shape)
        if np.iscomplexobj(v):
            return self.rmatvec(v.real) + 1j * self.rmatvec(v.imag)
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
        if np.iscomplexobj(v):
            return self.matvec(v.real) + 1j * self.matvec(v.imag)
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
    # A diagonal built from aranges is valid by construction, so re-scanning
    # three million-entry arrays to confirm it would be pure overhead.
    return cls._from_parts(np.arange(n + 1, dtype=np.intp), indices, data, shape)


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


def _row_dicts(A):
    """Canonical row dictionaries, O(nnz) storage, with duplicate cancellation."""
    A = A.tocsr()
    rows = []
    for i in range(A.shape[0]):
        row = {}
        for p in range(int(A.indptr[i]), int(A.indptr[i + 1])):
            j = int(A.indices[p])
            row[j] = row.get(j, 0.0) + float(A.data[p])
        rows.append({j: value for j, value in row.items() if value != 0.0})
    return rows


def _rows_to_csr(rows, shape):
    indptr, indices, data = [0], [], []
    for row in rows:
        for j in sorted(row):
            if row[j] != 0:
                indices.append(j)
                data.append(row[j])
        indptr.append(len(data))
    return CSRMatrix(indptr, indices, data, shape)


def _sparse_product(A, B):
    if A.shape[1] != B.shape[0]:
        raise DimensionError("sparse product has incompatible shapes")
    A, B = A.tocsr(), B.tocsr()
    indptr, indices, data = [0], [], []
    for i in range(A.shape[0]):
        row = {}
        for p in range(int(A.indptr[i]), int(A.indptr[i + 1])):
            k, a = int(A.indices[p]), float(A.data[p])
            for q in range(int(B.indptr[k]), int(B.indptr[k + 1])):
                j = int(B.indices[q])
                row[j] = row.get(j, 0.0) + a * float(B.data[q])
        for j in sorted(row):
            if row[j] != 0:
                indices.append(j)
                data.append(row[j])
        indptr.append(len(data))
    return CSRMatrix(indptr, indices, data, (A.shape[0], B.shape[1]))


def spmm(A, B):
    """Sparse/sparse or sparse/dense matrix product without dense conversion.

    Sparse products return canonical CSR and need storage proportional to the
    result plus one accumulator dictionary for the current output row.
    """
    if isinstance(A, _SparseBase):
        return A @ B
    if isinstance(B, _SparseBase):
        return B.__rmatmul__(A)
    return np.asarray(A) @ np.asarray(B)


def sparse_triangular_solve(A, b, lower=True, unit_diagonal=False, out=None):
    """Solve a CSR/CSC triangular system, including multiple right-hand sides."""
    if not isinstance(A, _SparseBase) or A.shape[0] != A.shape[1]:
        raise DimensionError("expected a square sparse matrix")
    A = A.tocsr()
    b = np.asarray(b, dtype=float)
    if b.ndim not in (1, 2) or b.shape[0] != A.shape[0]:
        raise DimensionError("right-hand side has incompatible shape")
    x = b.copy()
    for i in (range(A.shape[0]) if lower else range(A.shape[0] - 1, -1, -1)):
        diagonal = 1.0 if unit_diagonal else 0.0
        for p in range(int(A.indptr[i]), int(A.indptr[i + 1])):
            j, value = int(A.indices[p]), A.data[p]
            if j == i and not unit_diagonal:
                diagonal += value
            elif (j < i if lower else j > i):
                x[i] -= value * x[j]
            elif j != i and value != 0:
                raise ValueError("matrix contains entries outside the requested triangle")
        if diagonal == 0:
            raise np.linalg.LinAlgError(f"zero sparse triangular pivot at row {i}")
        x[i] /= diagonal
    if out is not None:
        if not isinstance(out, np.ndarray) or out.shape != x.shape:
            raise DimensionError("out must match the solution shape")
        out[...] = x
        return out
    return x


def _sparse_ilu0(A):
    if A.shape[0] != A.shape[1]:
        raise DimensionError("ILU requires a square matrix")
    if not np.all(np.isfinite(A.data)):
        raise ValueError("ILU requires finite entries")
    rows = _row_dicts(A)
    lower, upper = [], []
    for i, row in enumerate(rows):
        for j in sorted(k for k in row if k < i):
            pivot = rows[j].get(j, 0.0)
            if pivot == 0:
                raise np.linalg.LinAlgError(f"zero ILU pivot at row {j}")
            multiplier = row[j] / pivot
            row[j] = multiplier
            for k, value in rows[j].items():
                if k > j and k in row:
                    row[k] -= multiplier * value
        if row.get(i, 0.0) == 0:
            raise np.linalg.LinAlgError(f"zero ILU pivot at row {i}")
        lower.append({**{j: v for j, v in row.items() if j < i}, i: 1.0})
        upper.append({j: v for j, v in row.items() if j >= i})
    return _rows_to_csr(lower, A.shape), _rows_to_csr(upper, A.shape)


def _sparse_ichol(A, drop_tol=0.0):
    if A.shape[0] != A.shape[1]:
        raise DimensionError("incomplete Cholesky requires a square matrix")
    if not np.isfinite(drop_tol) or drop_tol < 0:
        raise ValueError("drop_tol must be finite and nonnegative")
    rows = _row_dicts(A)
    for i, row in enumerate(rows):
        for j, value in row.items():
            partner = rows[j].get(i, 0.0)
            if not np.isfinite(value) or abs(value - partner) > 1e-12 * max(1.0, abs(value), abs(partner)):
                raise ValueError("incomplete Cholesky requires a finite symmetric matrix")
    lower = []
    for i, row in enumerate(rows):
        li = {}
        for j in sorted(k for k in row if k < i and abs(row[k]) > drop_tol):
            lj = lower[j]
            correction = sum(v * lj.get(k, 0.0) for k, v in li.items() if k < j)
            li[j] = (row[j] - correction) / lj[j]
        d = row.get(i, 0.0) - sum(v * v for v in li.values())
        if d <= 0 or not np.isfinite(d):
            raise np.linalg.LinAlgError(f"non-positive incomplete Cholesky pivot at row {i}")
        li[i] = float(np.sqrt(d))
        lower.append(li)
    return _rows_to_csr(lower, A.shape)


class _SparsePreconditioner:
    def __init__(self, L, U, unit_lower=False):
        self.L, self.U, self.shape = L, U, L.shape
        self.unit_lower = unit_lower

    def solve(self, b, out=None):
        y = sparse_triangular_solve(self.L, b, unit_diagonal=self.unit_lower)
        return sparse_triangular_solve(self.U, y, lower=False, out=out)

    __call__ = solve


def ilu_preconditioner(A):
    """Factor sparse A with ILU(0), returning a reusable inverse application."""
    if not isinstance(A, _SparseBase):
        A = from_dense(A)
    L, U = _sparse_ilu0(A)
    return _SparsePreconditioner(L, U, unit_lower=True)


def ichol_preconditioner(A, drop_tol=0.0):
    """Factor sparse SPD A with IC(0), returning a reusable inverse application."""
    if not isinstance(A, _SparseBase):
        A = from_dense(A)
    L = _sparse_ichol(A, drop_tol)
    return _SparsePreconditioner(L, L.T)
