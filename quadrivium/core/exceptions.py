"""Exception hierarchy shared by every solver in :mod:`numethods`."""

from __future__ import annotations

__all__ = [
    "NumethodsError",
    "ConvergenceError",
    "SingularMatrixError",
    "DimensionError",
    "DomainError",
    "StepSizeError",
    "BracketError",
]


class NumethodsError(Exception):
    """Base class for all library errors."""


class ConvergenceError(NumethodsError):
    """An iterative method failed to reach the requested tolerance.

    The partial state is attached so callers can inspect / restart.
    """

    def __init__(self, message, iterations=None, residual=None, best=None):
        super().__init__(message)
        self.iterations = iterations
        self.residual = residual
        self.best = best


class SingularMatrixError(NumethodsError):
    """Matrix is singular (or numerically so) for the requested operation."""


class DimensionError(NumethodsError, ValueError):
    """Array shapes are incompatible."""


class DomainError(NumethodsError, ValueError):
    """Argument outside the domain of validity of the method."""


class StepSizeError(NumethodsError):
    """Adaptive step size underflowed the minimum allowed value."""


class BracketError(NumethodsError, ValueError):
    """A bracketing method was given an interval that does not bracket a root."""
