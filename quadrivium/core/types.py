"""Lightweight result containers returned by the solvers.

Every container behaves like a small immutable record with a readable ``repr``
so results stay legible at an interactive prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

__all__ = [
    "RootResult",
    "IterationResult",
    "QuadratureResult",
    "ODESolution",
    "OptimizeResult",
    "EigenResult",
    "PDESolution",
]


@dataclass
class RootResult:
    """Outcome of a root-finding method."""

    root: Any
    f_root: Any = None
    iterations: int = 0
    converged: bool = False
    function_calls: int = 0
    method: str = ""
    history: list = field(default_factory=list)
    message: str = ""

    @property
    def x(self):
        return self.root

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RootResult(root={_fmt(self.root)}, converged={self.converged}, "
            f"iterations={self.iterations}, method={self.method!r})"
        )


@dataclass
class IterationResult:
    """Outcome of an iterative linear solver."""

    x: Any
    iterations: int = 0
    converged: bool = False
    residuals: list = field(default_factory=list)
    method: str = ""
    message: str = ""

    @property
    def residual(self):
        return self.residuals[-1] if self.residuals else None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"IterationResult(converged={self.converged}, iterations={self.iterations}, "
            f"residual={_fmt(self.residual)}, method={self.method!r})"
        )


@dataclass
class QuadratureResult:
    """Outcome of a quadrature rule."""

    value: float
    error_estimate: Optional[float] = None
    function_calls: int = 0
    subintervals: int = 0
    converged: bool = True
    method: str = ""

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"QuadratureResult(value={_fmt(self.value)}, "
            f"error_estimate={_fmt(self.error_estimate)}, method={self.method!r})"
        )


@dataclass
class ODESolution:
    """Outcome of an initial-value problem integration.

    ``t`` has shape ``(n,)`` and ``y`` has shape ``(n, dim)``.
    """

    t: np.ndarray
    y: np.ndarray
    method: str = ""
    n_steps: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    n_rhs_evals: int = 0
    success: bool = True
    message: str = ""
    interpolant: Optional[Callable] = None
    dydt: Optional[np.ndarray] = None

    def __call__(self, t_query):
        """Evaluate the solution at ``t_query`` (scalar or array).

        Accuracy depends on what the solver recorded.  With ``dydt`` -- the
        right-hand side at each stored point, which every Runge-Kutta method
        already computes as its first stage -- this is cubic Hermite
        interpolation, whose O(h^4) error matches a 4th/5th-order integrator.
        Without it the fallback is linear interpolation, accurate only to
        O(h^2), which would otherwise dominate the solver's own error.
        """
        tq = np.atleast_1d(np.asarray(t_query, dtype=float))
        scalar = np.ndim(t_query) == 0
        if self.interpolant is not None:
            out = self.interpolant(tq)
            out = np.asarray(out)
            return out[0] if scalar else out
        if self.dydt is not None and len(self.t) > 1:
            out = self._hermite(tq)
        else:
            out = np.empty((tq.size, self.y.shape[1]))
            for j in range(self.y.shape[1]):
                out[:, j] = np.interp(tq, self.t, self.y[:, j])
        return out[0] if scalar else out

    def _hermite(self, tq: np.ndarray) -> np.ndarray:
        """Piecewise cubic Hermite through the stored states and slopes."""
        t, y, d = self.t, self.y, np.asarray(self.dydt)
        ascending = t[-1] >= t[0]
        key = t if ascending else t[::-1]
        idx = np.clip(np.searchsorted(key, tq) - 1, 0, key.size - 2)
        if not ascending:  # map back to the original (descending) ordering
            idx = t.size - 2 - idx
        h = t[idx + 1] - t[idx]
        u = ((tq - t[idx]) / h)[:, None]
        h = h[:, None]
        u2, u3 = u * u, u * u * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        return (h00 * y[idx] + h10 * h * d[idx]
                + h01 * y[idx + 1] + h11 * h * d[idx + 1])

    @property
    def y_final(self):
        return self.y[-1]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ODESolution(method={self.method!r}, steps={self.n_steps}, "
            f"t=[{_fmt(self.t[0])}, {_fmt(self.t[-1])}], dim={self.y.shape[1]})"
        )


@dataclass
class OptimizeResult:
    """Outcome of an optimization run."""

    x: Any
    fun: float = np.nan
    jac: Any = None
    hess: Any = None
    iterations: int = 0
    converged: bool = False
    function_calls: int = 0
    gradient_calls: int = 0
    method: str = ""
    history: list = field(default_factory=list)
    message: str = ""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OptimizeResult(fun={_fmt(self.fun)}, converged={self.converged}, "
            f"iterations={self.iterations}, method={self.method!r})"
        )


@dataclass
class EigenResult:
    """Eigenvalues (and optionally eigenvectors) of a matrix."""

    eigenvalues: np.ndarray
    eigenvectors: Optional[np.ndarray] = None
    iterations: int = 0
    converged: bool = True
    method: str = ""

    def __iter__(self):
        yield self.eigenvalues
        yield self.eigenvectors

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EigenResult(n={np.size(self.eigenvalues)}, converged={self.converged}, "
            f"method={self.method!r})"
        )


@dataclass
class PDESolution:
    """Outcome of a PDE solve on a structured grid."""

    u: np.ndarray
    grids: Sequence[np.ndarray] = ()
    t: Optional[np.ndarray] = None
    method: str = ""
    iterations: int = 0
    converged: bool = True
    residuals: list = field(default_factory=list)

    @property
    def x(self):
        return self.grids[0] if self.grids else None

    @property
    def y(self):
        return self.grids[1] if len(self.grids) > 1 else None

    @property
    def final(self):
        """Last time level for time-dependent problems, else the field itself."""
        return self.u[-1] if self.t is not None else self.u

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PDESolution(shape={np.shape(self.u)}, method={self.method!r})"


def _fmt(v):
    if v is None:
        return "None"
    try:
        if np.isscalar(v) or np.ndim(v) == 0:
            return f"{float(v):.6g}"
    except (TypeError, ValueError):
        pass
    return f"array{np.shape(v)}"
