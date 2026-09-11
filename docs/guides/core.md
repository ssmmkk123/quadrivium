# Shared numerical conventions

The `quadrivium.core` module defines the result records, error types, shape
conversions, and numerical utilities used throughout Quadrivium. Understanding
these conventions makes it easier to compare methods and to decide whether a
computed answer is accurate enough for the problem.

Most names are also available from `quadrivium`. Use explicit submodule imports
when the name alone is ambiguous, especially for derivatives and norms.

```pycon
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> from quadrivium.core import norm, absolute_error, relative_error
>>> norm([3.0, -4.0])
5.0

```

This guide explains how to interpret outputs. The [core reference](../api/core.md)
provides signatures and field definitions; [getting started](../getting-started.md)
shows the same conventions in complete calculations.

## Read the result before extracting the answer

A result record keeps the mathematical answer together with the information
needed to assess it. The records are mutable dataclasses. They are neither
immutable certificates nor interchangeable arrays.

| Record | Answer | Diagnostics to inspect |
| --- | --- | --- |
| `RootResult` | `root`, also available as `x` | `f_root`, `converged`, `iterations`, `function_calls`, `message` |
| `IterationResult` | `x` | `converged`, `residuals`, `iterations`, `message` |
| `QuadratureResult` | `value` | `error_estimate`, `converged`, `function_calls`, `subintervals` |
| `ODESolution` | `t`, `y`, `y_final` | `success`, `message`, step and right-hand-side counts |
| `OptimizeResult` | `x`, `fun` | `jac`, `converged`, `iterations`, call counts, `message` |
| `EigenResult` | `eigenvalues`, optional `eigenvectors` | `converged`, `iterations`, `method` |
| `PDESolution` | `u`, `grids`, optional `t`, `final` | `converged`, `iterations`, `residuals` |

Direct linear solves and many approximation constructors return arrays or
callables instead. Sampled-data integration can return a plain number. The
method's documented return type takes precedence over a family-level convention.

```pycon
>>> result = qd.brent(lambda x: x*x - 2.0, 0.0, 2.0)
>>> result.converged and abs(result.f_root) < 1e-10
True
>>> result.x == result.root
True
>>> result.function_calls > 0
True

```

A convergence flag means that the method met its implemented stopping rule.
That rule may use a residual, a change in iterates, a bracket width, or an
embedded error estimate. It does not establish that the model is correct, the
solution is unique, or an independent physical error target is satisfied.

### Convenient conversions and their limits

`float(quadrature_result)` extracts a scalar integral. It discards the error
estimate and cannot represent a vector or genuinely complex integral.
`EigenResult` unpacks as `(eigenvalues, eigenvectors)`; eigenvectors can be
`None` when they were not computed.

```pycon
>>> integral = qd.quad(lambda x: x*x, 0.0, 1.0)
>>> abs(float(integral) - 1.0/3.0) < 1e-12
True
>>> values, vectors = qd.jacobi_eigen(np.eye(2))
>>> values.shape, vectors.shape
((2,), (2, 2))

```

`IterationResult.residual` is the last stored residual or `None` if the history
is empty. Residual normalization varies by solver. For a comparison across
methods, recompute the same residual expression for every returned answer.

For ODE results, `y` has shape `(number_of_saved_times, state_dimension)`.
A scalar query `solution(t)` returns a state vector; an array of query times
returns one row per time. Evaluation uses a solver-provided interpolant when
available, cubic Hermite interpolation when saved slopes are present, and
linear interpolation otherwise. Query within the solved interval and remember
that interpolation introduces its own error.

For time-dependent PDE results, `final` selects the last computed field.
For stationary problems, it returns the field directly. With restricted output
storage, `y_final` and `final` can represent the endpoint even when that endpoint
was not among the explicitly requested saved times.

## Distinguish residual, forward error, and conditioning

A **residual** measures how well a computed answer satisfies the equation.
For a linear solve it is `A @ x - b`. A **forward error** compares that answer
with the true answer. A small residual can coexist with a large forward error
when the problem is sensitive to perturbations.

`absolute_error(approx, exact)` uses the infinity norm of the difference.
`relative_error` divides it by the infinity norm of `exact`; when the exact
value is zero, it returns the absolute error instead.

```pycon
>>> absolute_error([1.01, 2.0], [1.0, 2.0]) < 0.011
True
>>> relative_error([1.01, 2.0], [1.0, 2.0]) < 0.006
True
>>> relative_error(0.25, 0.0)
0.25

```

This zero-reference convention avoids division by zero. It also means the
quantity changes units at a zero reference. For application-level acceptance,
a mixed test such as `error <= atol + rtol * reference_scale` is often clearer.
Choose the scale from the physical or mathematical quantity being checked.

### Vector and matrix norms

`norm(x, p=2)` flattens its input and converts it to real floating-point data.
It supports positive `p`, plus `np.inf` or `"inf"`. Values below one are
accepted, although the resulting expression is mathematically a quasi-norm.

`matrix_norm(A, p="fro")` has a different default and interpretation:

| Choice | Meaning |
| --- | --- |
| `p=1` | Largest absolute column sum |
| `p=np.inf` | Largest absolute row sum |
| `p=2` | Largest singular value, evaluated through `A.T @ A` |
| `p="fro"` | Square root of the sum of squared entries |

The spectral norm helper forms a Gram matrix, so it can lose accuracy on badly
scaled inputs. Use `numeric.linalg.norm` when you need the array layer's complex
or batched behavior. The real-valued core utilities do not preserve complex
inputs as Hermitian calculations.

```pycon
>>> from quadrivium.core import matrix_norm, condition_number
>>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
>>> matrix_norm(A, 1), matrix_norm(A, np.inf)
(5.0, 5.0)
>>> 1.0 < condition_number(A) < 2.0
True

```

`condition_number` requires a square matrix. Its default 2-norm calculation
uses singular values; other choices explicitly form an inverse. Infinity
signals singularity in the detected cases. A finite result is an estimate of
sensitivity, not a guarantee of useful digits. See [linear algebra](linalg.md)
for residual checks, regularization, and condition estimation.

## Floating-point precision and stable expressions

`EPS` is double-precision machine epsilon, and `SQRT_EPS` is its square root.
`machine_epsilon(dtype)` computes the spacing at one for the chosen floating
type. `unit_roundoff(dtype)` is half that spacing under round-to-nearest.
These quantities describe arithmetic resolution, not measurement uncertainty.

```pycon
>>> from quadrivium.core import EPS, SQRT_EPS, machine_epsilon, unit_roundoff
>>> EPS == float(np.finfo(float).eps)
True
>>> unit_roundoff() == EPS / 2
True
>>> SQRT_EPS == 2.0**-26
True

```

Subtraction of nearby values can erase significant digits. An algebraically
equivalent expression can therefore be substantially more accurate. For
example, use `np.expm1(x)` for `exp(x)-1` near zero and `np.log1p(x)` for
`log(1+x)` near zero. Increasing iteration counts cannot recover information
that the function evaluation already lost.

<figure markdown="span">
  ![Error in a cancellation-prone exponential expression compared with a stable evaluation](../assets/figures/core-roundoff-budget.svg#only-light)
  ![Error in a cancellation-prone exponential expression compared with a stable evaluation](../assets/figures/core-roundoff-budget-dark.svg#only-dark)
  <figcaption>For h from 10⁻¹⁶ to 10⁻¹, exp(h)−1 and expm1(h) are compared with a 60-digit decimal reference. The relative-error panel reveals cancellation in the direct subtraction even when its absolute error looks small.</figcaption>
</figure>

## Shape and property checks

`as_vector` converts scalar input to length one and flattens higher-dimensional
input. It can share storage with an already suitable array. It does **not**
validate that the original object represented one mathematical vector.
`as_matrix` requires exactly two dimensions; `check_square` additionally requires
equal side lengths. These helpers convert to real double precision.

```pycon
>>> from quadrivium.core import as_vector, as_matrix, check_square
>>> as_vector([[1, 2], [3, 4]]).tolist()
[1.0, 2.0, 3.0, 4.0]
>>> as_matrix([[1, 2]]).shape
(1, 2)
>>> check_square([[1, 0], [0, 1]]).shape
(2, 2)

```

Use `.copy()` when an independent working array is required. Validate the
original `ndim` before calling `as_vector` if accidentally flattening a batch
would change your problem.

`is_symmetric` compares a real matrix with its transpose using a tolerance.
Its `tol` is an absolute tolerance used together with the array comparison's
relative tolerance. `is_positive_definite` checks symmetry and attempts
Cholesky; positive `tol` also imposes a smallest-eigenvalue threshold.
`is_diagonally_dominant` checks rows and is strict by default.

```pycon
>>> from quadrivium.core import is_symmetric, is_positive_definite, is_diagonally_dominant
>>> is_symmetric(A), is_positive_definite(A), is_diagonally_dominant(A)
(True, True, True)

```

These tests help choose algorithms, but they do not establish every hypothesis
of a convergence theorem. In particular, approximate symmetry and strict
positive definiteness should be assessed relative to your data's scale.

## Derivatives and call accounting

`numerical_derivative` handles scalar derivative orders one through four.
`numerical_gradient`, `numerical_jacobian`, and `numerical_hessian` use finite
differences. A Jacobian for `F: R^n -> R^m` has shape `(m, n)`; a scalar-field
Hessian has shape `(n, n)`. Function values must be finite near the requested
point, including the perturbed points.

```pycon
>>> from quadrivium.core import numerical_gradient, numerical_jacobian
>>> f = lambda x: x[0]**2 + 3*x[1]**2
>>> np.allclose(numerical_gradient(f, [1.0, 2.0]), [2.0, 12.0])
True
>>> numerical_jacobian(lambda x: [x[0] + x[1], x[0] - x[1]], [1, 2]).shape
(2, 2)

```

The default perturbation is scaled to the coordinate. Supplying `h` overrides
that choice. Check sensitivity to step size when variables differ greatly in
scale or the function is noisy. The [differentiation guide](diff.md) explains
finite differences, automatic differentiation, and spectral methods.

`CountedFunction` counts calls to a callable, including calls that raise. It
accepts positional and keyword arguments, exposes `.calls`, and supports
`.reset()`. A vectorized call counts once even if it evaluates many entries.

```pycon
>>> from quadrivium.core import CountedFunction, wrap_scalar_function
>>> counted = CountedFunction(lambda x: x*x)
>>> counted(3), counted.calls
(9, 1)
>>> counted.reset()
>>> counted.calls
0
>>> import math
>>> vector_sine = wrap_scalar_function(math.sin)
>>> np.allclose(vector_sine([0.0, math.pi/2]), [0.0, 1.0])
True

```

`wrap_scalar_function` maps a scalar function elementwise over arrays. It first
tries an array call and falls back to scalar calls if needed; it is not an
adapter that turns a scalar objective into a general multivariate objective.

## Errors and output storage

`QuadriviumError` is the base for `BracketError`, `DimensionError`,
`DomainError`, `SingularMatrixError`, `StepSizeError`, and `ConvergenceError`.
Not every error raised by the package belongs to this hierarchy: ordinary
`ValueError`, `TypeError`, and the array layer's `LinAlgError` are also used.
Catch the documented failure cases around a specific operation instead of
assuming one base class catches every possible exception.

An iterative method may return a partial answer with a false convergence flag.
Other routines raise when no useful result can be produced. `ConvergenceError`
can carry `iterations`, `residual`, and `best` to preserve diagnostics.

Time integrators supporting common output controls can retain selected times,
every nth state, or only the final state. `OutputRecorder` implements this
storage policy; it does not set solver accuracy. `SolverCheckpoint` stores
numerical restart state as JSON and deliberately excludes model callables.
Use `resume_ode` or `resume_pde` with the model supplied again. Restart fidelity
is method-dependent; see the [ODE guide](ode.md), [PDE guide](pde.md), and
[workflows](workflows.md) before treating a restart as an identical continuation.
