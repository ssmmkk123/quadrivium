"""Numerical utilities: norms, machine constants, and numerical derivatives.

These helpers are used pervasively, so they are written to be allocation-light
and to accept both scalar and vector callables.
"""

from __future__ import annotations

from typing import Callable

from .. import numeric as np

from .exceptions import DimensionError

__all__ = [
    "EPS",
    "SQRT_EPS",
    "machine_epsilon",
    "unit_roundoff",
    "norm",
    "matrix_norm",
    "condition_number",
    "relative_error",
    "absolute_error",
    "as_vector",
    "as_matrix",
    "check_square",
    "is_symmetric",
    "is_positive_definite",
    "is_diagonally_dominant",
    "numerical_derivative",
    "numerical_jacobian",
    "numerical_gradient",
    "numerical_hessian",
    "wrap_scalar_function",
    "CountedFunction",
]

EPS = float(np.finfo(float).eps)
SQRT_EPS = float(np.sqrt(EPS))

# Bound once: `as_vector` compares against it on every call.
_F64 = np.dtype(np.float64)

# Distinguishes "no argument given" from a legitimate ``None`` argument.
_UNSET = object()


def machine_epsilon(dtype=float) -> float:
    """Smallest ``e`` with ``1 + e != 1`` in the given floating type."""
    one = np.array(1.0, dtype=dtype)
    eps = np.array(1.0, dtype=dtype)
    two = np.array(2.0, dtype=dtype)
    while one + eps / two != one:
        eps = eps / two
    return float(eps)


def unit_roundoff(dtype=float) -> float:
    """Unit roundoff ``u = eps / 2`` for round-to-nearest arithmetic."""
    return machine_epsilon(dtype) / 2.0


def norm(x, p=2) -> float:
    """Vector ``p``-norm; ``p`` may be 1, 2, any positive float, or ``inf``."""
    x = np.asarray(x, dtype=float).ravel()
    if p == np.inf or p == "inf":
        return float(np.max(np.abs(x))) if x.size else 0.0
    if p == 1:
        return float(np.sum(np.abs(x)))
    if p == 2:
        return float(np.sqrt(np.dot(x, x)))
    if p <= 0:
        raise ValueError("p must be positive or inf")
    return float(np.sum(np.abs(x) ** p) ** (1.0 / p))


def matrix_norm(A, p="fro") -> float:
    """Matrix norm: ``1`` (max column sum), ``inf`` (max row sum), ``2``
    (spectral), or ``'fro'`` (Frobenius)."""
    A = np.asarray(A, dtype=float)
    if p == "fro":
        return float(np.sqrt(np.sum(A * A)))
    if p == 1:
        return float(np.max(np.sum(np.abs(A), axis=0)))
    if p == np.inf or p == "inf":
        return float(np.max(np.sum(np.abs(A), axis=1)))
    if p == 2:
        return float(np.sqrt(np.max(np.abs(np.linalg.eigvalsh(A.T @ A)))))
    raise ValueError(f"unsupported matrix norm {p!r}")


def condition_number(A, p=2) -> float:
    """Condition number ``||A|| * ||A^-1||`` in the requested norm."""
    A = check_square(A)
    if p == 2:
        s = np.linalg.svd(A, compute_uv=False)
        return float(np.inf) if s[-1] == 0 else float(s[0] / s[-1])
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float(np.inf)
    return matrix_norm(A, p) * matrix_norm(Ainv, p)


def absolute_error(approx, exact) -> float:
    """``||approx - exact||_inf``."""
    return norm(np.asarray(approx, dtype=float) - np.asarray(exact, dtype=float), np.inf)


def relative_error(approx, exact) -> float:
    """Absolute error scaled by ``||exact||``; falls back to absolute if zero."""
    denom = norm(exact, np.inf)
    err = absolute_error(approx, exact)
    return err / denom if denom > 0 else err


def unwrap_scalar(value):
    """Return a 0-d array's single element, and anything else unchanged.

    Element-wise operations already yield Python scalars for 0-d operands, but
    array constructors still produce 0-d arrays, so a routine that promises a
    scalar for scalar input finishes through this.
    """
    return value[()] if getattr(value, "ndim", None) == 0 else value


def as_vector(x) -> np.ndarray:
    """Coerce to a 1-D float array (scalars become length-1 arrays).

    This runs in the inner loop of every iterative solver in the package, so
    the overwhelmingly common case -- an argument that is already a contiguous
    1-D ``float64`` array -- returns immediately. The general path is the
    original coercion; both return a view when one is possible and a copy when
    the input cannot be viewed as contiguous 1-D, so aliasing is unchanged.
    """
    if (x.__class__ is np.ndarray and x.ndim == 1 and x.dtype == _F64
            and x.flags.c_contiguous):
        return x
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel()


def as_matrix(A) -> np.ndarray:
    """Coerce to a 2-D float array."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise DimensionError(f"expected a 2-D array, got shape {A.shape}")
    return A


def check_square(A) -> np.ndarray:
    """Coerce to a 2-D float array and require it to be square."""
    A = as_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise DimensionError(f"expected a square matrix, got shape {A.shape}")
    return A


def is_symmetric(A, tol: float = 1e-12) -> bool:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    from .. import _accel

    fast = _accel.kernel("is_symmetric")
    if fast is not None and A.shape[0]:
        # Same predicate as ``np.allclose(A, A.T, atol=tol)`` -- including
        # NumPy's default rtol of 1e-5 -- but streamed in one pass with an
        # early exit, rather than materialising |A - A.T|.
        return bool(fast(np.ascontiguousarray(A), tol, 1e-5))
    return bool(np.allclose(A, A.T, atol=tol))


def is_positive_definite(A, tol: float = 0.0) -> bool:
    """True when ``A`` is symmetric with a successful Cholesky factorization."""
    if not is_symmetric(A):
        return False
    try:
        np.linalg.cholesky(np.asarray(A, dtype=float))
    except np.linalg.LinAlgError:
        return False
    return True if tol == 0.0 else bool(np.min(np.linalg.eigvalsh(A)) > tol)


def is_diagonally_dominant(A, strict: bool = True) -> bool:
    """Row diagonal dominance test (sufficient for Jacobi/Gauss-Seidel)."""
    A = check_square(A)
    diag = np.abs(np.diag(A))
    off = np.sum(np.abs(A), axis=1) - diag
    return bool(np.all(diag > off)) if strict else bool(np.all(diag >= off))


class CountedFunction:
    """Callable wrapper that tallies evaluations.

    Useful for reporting ``function_calls`` without threading counters through
    every algorithm.
    """

    __slots__ = ("f", "calls")

    def __init__(self, f: Callable):
        self.f = f
        self.calls = 0

    def __call__(self, x=_UNSET, *args, **kwargs):
        # Single-argument calls dominate -- every objective and every integrand
        # -- and skipping the re-packing there is worth the extra branch. The
        # sentinel keeps keyword-only and zero-argument calls working, since
        # this class is part of the public API.
        self.calls += 1
        if args or kwargs or x is _UNSET:
            if x is _UNSET:
                return self.f(*args, **kwargs)
            return self.f(x, *args, **kwargs)
        return self.f(x)

    def reset(self) -> None:
        self.calls = 0


def wrap_scalar_function(f: Callable) -> Callable:
    """Return a version of ``f`` that maps arrays elementwise.

    Tries the vectorized call first and falls back to a list comprehension, so
    plain ``math``-style functions work anywhere the library accepts a callable.
    """

    def wrapped(x):
        x_arr = np.asarray(x, dtype=float)
        try:
            out = np.asarray(f(x_arr), dtype=float)
            if out.shape == x_arr.shape:
                return out
        except Exception:
            pass
        if x_arr.ndim == 0:
            return float(f(float(x_arr)))
        return np.array([float(f(float(v))) for v in x_arr.ravel()]).reshape(x_arr.shape)

    return wrapped


def _step(x, h=None, order=2):
    if h is not None:
        return h
    scale = max(abs(float(x)), 1.0)
    return (EPS ** (1.0 / (order + 1))) * scale


def numerical_derivative(f, x, h=None, order=1, accuracy=2):
    """Finite-difference derivative of a scalar function.

    ``order`` is the derivative order (1-4) and ``accuracy`` the formal order of
    accuracy (2 or 4) of the central stencil used.
    """
    x = float(x)
    h = _step(x, h, order + accuracy)
    if order == 1:
        if accuracy == 2:
            return (f(x + h) - f(x - h)) / (2 * h)
        return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)
    if order == 2:
        if accuracy == 2:
            return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
        return (
            -f(x - 2 * h) + 16 * f(x - h) - 30 * f(x) + 16 * f(x + h) - f(x + 2 * h)
        ) / (12 * h**2)
    if order == 3:
        return (
            -f(x - 2 * h) + 2 * f(x - h) - 2 * f(x + h) + f(x + 2 * h)
        ) / (2 * h**3)
    if order == 4:
        return (
            f(x - 2 * h) - 4 * f(x - h) + 6 * f(x) - 4 * f(x + h) + f(x + 2 * h)
        ) / h**4
    raise ValueError("order must be 1, 2, 3 or 4")


def numerical_jacobian(F, x, h=None, method="central"):
    """Jacobian of a vector field ``F: R^n -> R^m`` at ``x``."""
    x = as_vector(x)
    n = x.size
    f0 = as_vector(F(x))
    m = f0.size
    J = np.empty((m, n))
    for j in range(n):
        hj = _step(x[j], h, 2)
        if method == "forward":
            xp = x.copy()
            xp[j] += hj
            J[:, j] = (as_vector(F(xp)) - f0) / hj
        elif method == "backward":
            xm = x.copy()
            xm[j] -= hj
            J[:, j] = (f0 - as_vector(F(xm))) / hj
        else:
            xp, xm = x.copy(), x.copy()
            xp[j] += hj
            xm[j] -= hj
            J[:, j] = (as_vector(F(xp)) - as_vector(F(xm))) / (2 * hj)
    return J


def numerical_gradient(f, x, h=None):
    """Gradient of a scalar field via central differences."""
    x = as_vector(x)
    g = np.empty_like(x)
    for j in range(x.size):
        hj = _step(x[j], h, 2)
        xp, xm = x.copy(), x.copy()
        xp[j] += hj
        xm[j] -= hj
        g[j] = (f(xp) - f(xm)) / (2 * hj)
    return g


def numerical_hessian(f, x, h=None):
    """Symmetric Hessian via second-order central differences."""
    x = as_vector(x)
    n = x.size
    hs = np.array([_step(x[j], h, 4) for j in range(n)])
    f0 = float(f(x))
    H = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += hs[i]
                xm[i] -= hs[i]
                H[i, i] = (float(f(xp)) - 2 * f0 + float(f(xm))) / hs[i] ** 2
            else:
                xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
                xpp[i] += hs[i]; xpp[j] += hs[j]
                xpm[i] += hs[i]; xpm[j] -= hs[j]
                xmp[i] -= hs[i]; xmp[j] += hs[j]
                xmm[i] -= hs[i]; xmm[j] -= hs[j]
                H[i, j] = H[j, i] = (
                    float(f(xpp)) - float(f(xpm)) - float(f(xmp)) + float(f(xmm))
                ) / (4 * hs[i] * hs[j])
    return H
