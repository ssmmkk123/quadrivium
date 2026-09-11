"""Shared infrastructure: result records, exceptions, and numerical utilities."""

from .exceptions import (
    BracketError,
    ConvergenceError,
    DimensionError,
    DomainError,
    QuadriviumError,
    SingularMatrixError,
    StepSizeError,
)
from .types import (
    EigenResult,
    IterationResult,
    ODESolution,
    OptimizeResult,
    PDESolution,
    QuadratureResult,
    RootResult,
)
from .utils import (
    EPS,
    SQRT_EPS,
    CountedFunction,
    absolute_error,
    as_matrix,
    as_vector,
    check_square,
    condition_number,
    is_diagonally_dominant,
    is_positive_definite,
    is_symmetric,
    machine_epsilon,
    matrix_norm,
    norm,
    numerical_derivative,
    numerical_gradient,
    numerical_hessian,
    numerical_jacobian,
    relative_error,
    unit_roundoff,
    wrap_scalar_function,
)

__all__ = [n for n in dir() if not n.startswith("_")]

from .storage import OutputRecorder, SolverCheckpoint, resume_ode, resume_pde
__all__ += ["OutputRecorder", "SolverCheckpoint", "resume_ode", "resume_pde"]
