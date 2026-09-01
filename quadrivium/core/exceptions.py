"""Exception hierarchy shared by every solver in :mod:`quadrivium`."""

from __future__ import annotations

__all__ = [
    "QuadriviumError",
    "ConvergenceError",
    "SingularMatrixError",
    "DimensionError",
    "DomainError",
    "StepSizeError",
    "BracketError",
]


class QuadriviumError(Exception):
    """Base class for all library errors."""


class ConvergenceError(QuadriviumError):
    """An iterative method failed to reach the requested tolerance.

    The partial state is attached so callers can inspect / restart.
    """

    def __init__(self, message, iterations=None, residual=None, best=None):
        super().__init__(message)
        self.iterations = iterations
        self.residual = residual
        self.best = best


class SingularMatrixError(QuadriviumError):
    """Matrix is singular (or numerically so) for the requested operation."""


class DimensionError(QuadriviumError, ValueError):
    """Array shapes are incompatible."""


class DomainError(QuadriviumError, ValueError):
    """Argument outside the domain of validity of the method."""


class StepSizeError(QuadriviumError):
    """Adaptive step size underflowed the minimum allowed value."""


class BracketError(QuadriviumError, ValueError):
    """A bracketing method was given an interval that does not bracket a root."""
