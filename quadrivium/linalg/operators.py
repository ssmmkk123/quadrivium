"""Composable matrix-free linear maps with explicit adjoint products."""
from __future__ import annotations

from operator import index

from .. import numeric as np
from ..core.exceptions import DimensionError

__all__ = ["LinearOperator", "aslinearoperator"]


class LinearOperator:
    """A linear map that stores functions instead of matrix entries.

    ``rmatvec`` applies the conjugate transpose, as required by least-squares
    solvers. ``T`` is the ordinary transpose and ``H`` the adjoint. Products,
    sums and scalar multiples remain lazy. A supplied ``matmat`` can batch
    columns; otherwise only one column-sized temporary is needed at a time.
    """
    __array_priority__ = 10000

    def __init__(self, shape, matvec, rmatvec=None, matmat=None, dtype=float):
        try:
            m, n = (index(s) for s in shape)
        except (TypeError, ValueError) as exc:
            raise DimensionError("operator shape must contain two integers") from exc
        if min(m, n) < 0:
            raise DimensionError("operator dimensions must be nonnegative")
        if not callable(matvec):
            raise TypeError("matvec must be callable")
        if rmatvec is not None and not callable(rmatvec):
            raise TypeError("rmatvec must be callable")
        if matmat is not None and not callable(matmat):
            raise TypeError("matmat must be callable")
        self.shape = (m, n)
        self.dtype = np.dtype(dtype)
        self._matvec, self._rmatvec, self._matmat = matvec, rmatvec, matmat

    def matvec(self, x):
        x = np.asarray(x)
        if x.shape != (self.shape[1],):
            raise DimensionError(f"expected vector shape {(self.shape[1],)}, got {x.shape}")
        y = np.asarray(self._matvec(x))
        if y.shape != (self.shape[0],):
            raise DimensionError("matvec returned an incompatible shape")
        return y

    def rmatvec(self, x):
        if self._rmatvec is None:
            raise NotImplementedError("this operator has no adjoint product (rmatvec)")
        x = np.asarray(x)
        if x.shape != (self.shape[0],):
            raise DimensionError("adjoint input has incompatible shape")
        y = np.asarray(self._rmatvec(x))
        if y.shape != (self.shape[1],):
            raise DimensionError("rmatvec returned an incompatible shape")
        return y

    def matmat(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] != self.shape[1]:
            raise DimensionError("matrix input has incompatible shape")
        if self._matmat is not None:
            y = np.asarray(self._matmat(x))
            if y.shape != (self.shape[0], x.shape[1]):
                raise DimensionError("matmat returned an incompatible shape")
            return y
        y = np.empty((self.shape[0], x.shape[1]), dtype=np.result_type(self.dtype, x.dtype))
        for j in range(x.shape[1]):
            y[:, j] = self.matvec(x[:, j])
        return y

    def __call__(self, x):
        return self @ x

    def __matmul__(self, other):
        if isinstance(other, LinearOperator) or hasattr(other, "matvec"):
            other = aslinearoperator(other)
            if self.shape[1] != other.shape[0]:
                raise DimensionError("operator product has incompatible shapes")
            adj = (lambda x: other.rmatvec(self.rmatvec(x))) if (
                self._rmatvec is not None and other._rmatvec is not None) else None
            return LinearOperator((self.shape[0], other.shape[1]),
                                  lambda x: self.matvec(other.matvec(x)), adj,
                                  lambda x: self.matmat(other.matmat(x)),
                                  np.result_type(self.dtype, other.dtype))
        x = np.asarray(other)
        return self.matvec(x) if x.ndim == 1 else self.matmat(x)

    @property
    def H(self):
        return LinearOperator(self.shape[::-1], self.rmatvec, self.matvec, dtype=self.dtype)

    @property
    def T(self):
        return LinearOperator(self.shape[::-1],
                              lambda x: np.conjugate(self.rmatvec(np.conjugate(x))),
                              lambda x: np.conjugate(self.matvec(np.conjugate(x))),
                              dtype=self.dtype)

    def transpose(self):
        return self.T

    def adjoint(self):
        return self.H

    def __mul__(self, scalar):
        if np.asarray(scalar).ndim != 0:
            return NotImplemented
        adj = None if self._rmatvec is None else lambda x: np.conjugate(scalar) * self.rmatvec(x)
        return LinearOperator(self.shape, lambda x: scalar * self.matvec(x), adj,
                              lambda x: scalar * self.matmat(x),
                              np.result_type(self.dtype, np.asarray(scalar).dtype))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        other = aslinearoperator(other)
        if self.shape != other.shape:
            raise DimensionError("operator sum has incompatible shapes")
        adj = (lambda x: self.rmatvec(x) + other.rmatvec(x)) if (
            self._rmatvec is not None and other._rmatvec is not None) else None
        return LinearOperator(self.shape, lambda x: self.matvec(x) + other.matvec(x),
                              adj, dtype=np.result_type(self.dtype, other.dtype))

    def __sub__(self, other):
        return self + (-aslinearoperator(other))

    def __repr__(self):
        return f"LinearOperator(shape={self.shape}, dtype={self.dtype})"


def aslinearoperator(A):
    """Wrap a dense array or sparse/matrix-free object without densifying it."""
    if isinstance(A, LinearOperator):
        return A
    if hasattr(A, "matvec"):
        return LinearOperator(A.shape, A.matvec, getattr(A, "rmatvec", None),
                              getattr(A, "matmat", None), getattr(A, "dtype", float))
    A = np.asarray(A)
    if A.ndim != 2:
        raise DimensionError("expected a matrix or LinearOperator")
    return LinearOperator(A.shape, lambda x: A @ x,
                          lambda x: np.conjugate(A.T) @ x,
                          lambda x: A @ x, A.dtype)
