# Getting started

This tour introduces the conventions you need to use Quadrivium reliably:
imports, arrays, callbacks, result records, tolerances, failure handling, and
validation. Each calculation is small enough to check against an independent
answer. Install the package first using the [installation guide](installation.md).

## Imports and arrays

```pycon
>>> import math
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> qd.brent is qd.rootfind.brent
True
>>> values = np.array([1.0, 2.0, 3.0])
>>> isinstance(values, np.ndarray), values.shape
(True, (3,))

```

`np` is an alias for `quadrivium.numeric`, the package's own C-backed array
module. It is not NumPy. Familiar names such as `array`, `linspace`, `sin`, and
`linalg.norm` cover many common operations, but compatibility has boundaries.
The [array guide](guides/numeric.md) describes dtypes, broadcasting, views,
buffers, batching, and file support.

Top-level `qd` contains frequently used methods. Import specialized methods
from their subpackage; the [API index](api/index.md) lists exports. The array
engine's `np.linalg` primitives and the algorithms in `qd.linalg` have different
interfaces and return contracts.

## Find a root

Suppose you need the positive solution of `x² = 2`. Define the residual
`f(x) = x² - 2` and choose a bracket whose endpoint values have opposite signs.
For a continuous function, that sign change establishes that a root exists
inside the interval. It does not establish uniqueness.

```pycon
>>> f = lambda x: x*x - 2
>>> root = qd.brent(f, 0, 2, tol=1e-12)
>>> round(root.root, 10), root.converged
(1.4142135624, True)
>>> abs(f(root.root)) < 1e-10
True
>>> root.function_calls > 0
True

```

The residual check answers a different question from the stopping flag: it
measures how well the computed point satisfies the equation. A small residual
can still coexist with a large root error when the function is very flat.
See [root finding](guides/rootfind.md) for derivative methods and systems.

## Integrate a function

For a smooth scalar integral on a finite interval, `quad` is a convenient
starting point. The return value includes the estimate and diagnostic fields.

```pycon
>>> integral = qd.quad(lambda x: x*x, 0, 1)
>>> round(float(integral.value), 10), integral.converged
(0.3333333333, True)
>>> abs(float(integral) - 1/3) < 1e-12
True

```

`float(integral)` is shorthand for a scalar result's value. An error estimate
is method-dependent and is not a certified bound. Sampled-data rules may
return a plain scalar instead. Vector quadrature has its own norm and result
contract. Choose a rule based on smoothness, singularities, oscillation, and
whether you can evaluate the function: see [integration](guides/integrate.md).

## Integrate a dynamical system

An initial-value problem specifies a derivative, a time interval, and an initial
state. The callback receives a scalar time and a one-dimensional state array;
it must return a derivative with the corresponding dimension.

```pycon
>>> rhs = lambda t, y: -2*y
>>> solution = qd.solve_ivp(rhs, (0, 1), [1.0], rtol=1e-9, atol=1e-11)
>>> solution.success
True
>>> solution.y.shape == (len(solution.t), 1)
True
>>> abs(float(solution.y_final[0]) - math.exp(-2)) < 1e-8
True
>>> query = np.linspace(0, 1, 5)
>>> sampled = solution(query)
>>> sampled.shape
(5, 1)
>>> float(np.max(np.abs(sampled[:, 0] - np.exp(-2*query)))) < 1e-7
True

```

Time runs along the first axis of `solution.y`; components run along the
second. Calling the solution interpolates its output. Interpolation accuracy
depends on the method and recorded data, and can differ from endpoint accuracy.
Query within the computed interval. With sparse output or `final_only=True`,
you have deliberately retained less information for later interpolation.

<figure markdown="span">
  ![Adaptive exponential decay trajectory and interpolation error against an analytic reference](assets/figures/start-decay-validation.svg#only-light)
  ![Adaptive exponential decay trajectory and interpolation error against an analytic reference](assets/figures/start-decay-validation-dark.svg#only-dark)
  <figcaption>This separate experiment solves y′ = −y, y(0) = 1 over [0, 4], with rtol = 10⁻⁶ and atol = 10⁻⁹. The left panel compares the adaptive trajectory with exp(−t); the right checks interpolation error at 301 points. The tolerance scale is a local control scale, not a guaranteed bound on the displayed global error.</figcaption>
</figure>

For a long integration, choose output before solving:

```pycon
>>> final = qd.solve_ivp(rhs, (0, 1), [1.0], final_only=True)
>>> final.y.shape
(1, 1)
>>> abs(float(final.y_final[0]) - math.exp(-2)) < 1e-7
True

```

[ODEs](guides/ode.md) explains fixed-step, adaptive, and stiff methods.
[Scientific workflows](guides/workflows.md) covers output controls,
checkpoints, and sensitivities.

## Solve a linear system

Write the model as `A @ x = b`. Solve it directly rather than explicitly
forming an inverse; then recompute the residual with the original inputs.

```pycon
>>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
>>> b = np.array([1.0, 2.0])
>>> x = qd.solve(A, b)
>>> residual = float(np.linalg.norm(A @ x - b))
>>> residual < 1e-12
True

```

A small residual shows that `x` solves a nearby linear system. It does not
establish that the solution is insensitive to measurement error. Conditioning,
rank, scaling, and matrix structure determine which solver is appropriate.
For repeated right-hand sides, use a [reusable factor](guides/linalg.md).

## Minimize a cost

Define a scalar objective of a one-dimensional parameter vector. Supplying an
analytic gradient avoids finite-difference evaluation costs and step-size error.

```pycon
>>> objective = lambda x: (x[0] - 2)**2 + 3*(x[1] + 1)**2
>>> gradient = lambda x: np.array([2*(x[0] - 2), 6*(x[1] + 1)])
>>> fit = qd.minimize(objective, [0.0, 0.0], method="bfgs", grad_f=gradient)
>>> fit.converged, round(float(fit.fun), 10)
(True, 0.0)
>>> float(np.linalg.norm(gradient(fit.x))) < 1e-6
True

```

This objective is a convex quadratic with a known minimizer. On a nonconvex
problem, convergence from one initial guess does not prove global optimality.
An objective-stagnation stopping condition also differs from a gradient-norm
condition. Check the returned message, scale your parameters, and verify the
constraints: see [optimization](guides/optimize.md).

## Result records

Some routines return values directly. Many solvers return dataclasses whose
fields describe the answer and the work performed. The records are mutable;
array fields are not inherently read-only snapshots for downstream code.

| Record | Main output | Evidence to inspect |
| --- | --- | --- |
| `RootResult` | `root`, alias `x` | `f_root`, `converged`, `iterations`, `function_calls`, `message` |
| `IterationResult` | `x` | `converged`, `residuals`, `iterations`, `message` |
| `QuadratureResult` | `value` | `error_estimate`, `converged`, `function_calls`, `subintervals` |
| `ODESolution` | `t`, `y`, `y_final` | `success`, `message`, `n_steps`, accepted/rejected steps, RHS calls |
| `OptimizeResult` | `x`, `fun` | `converged`, `jac`, work counters, `message` |
| `EigenResult` | `eigenvalues`, `eigenvectors` | `converged`, `iterations`; verify `A @ v - λ*v` |
| `PDESolution` | `u`, `grids`, sometimes `t` | Method-specific layout, `converged`, `residuals` |

Fields not populated by a method can retain defaults. Histories may be disabled
or downsampled; their length is not a universal iteration counter. In an ODE
result, `y_final` refers to the final computed state even when `save_at` omits
the integration endpoint. Read the [core guide](guides/core.md) for details.

## Tolerances and budgets

Tolerances only make sense together with a scale and a stopping rule.

| Control | Typical meaning | What it does not imply |
| --- | --- | --- |
| Root or iterative `tol` | A method-specific residual, step, or bracket test | Uniform relative error in the answer |
| ODE `rtol`, `atol` | Scale local error by relative and absolute state magnitudes | A strict bound on global or interpolated error |
| Fixed-step `n` | Discretize the interval into a requested number of steps | Automatic accuracy or stability control |
| Optimizer `ftol` | Small relative change in objective, where supported | A small gradient or a global optimum |
| `max_iter`, `max_steps` | Bound algorithmic work | A guarantee of convergence within that budget |

Near zero, an absolute tolerance matters because relative error is ill-defined.
For components with different physical scales, rescaling or supported
component-wise tolerances can prevent one variable from dominating the test.
Making a tolerance tiny can increase cost without improving meaningful digits.

For a method of order `p` in a smooth asymptotic regime, halving the step
should reduce the error by roughly `2**p`. Repeat a solve on a finer grid and
compare an actual quantity of interest. [Design and validation](design.md)
explains what refinement can and cannot establish.

## How failure is reported

A result may report failure, or a routine may raise an exception. Do not assume
that every invalid input raises a `QuadriviumError`: numeric primitives and
several workflow APIs also use standard Python and linear-algebra exceptions.

```pycon
>>> failed = qd.newton(lambda x: x*x + 1, 1.0, lambda x: 2*x, max_iter=20)
>>> failed.converged, failed.message
(False, 'zero derivative encountered')
>>> from quadrivium.core import BracketError
>>> try:
...     qd.bisection(lambda x: x*x + 1, 0, 1)
... except BracketError:
...     print("Choose a valid bracket or a different problem formulation.")
Choose a valid bracket or a different problem formulation.

```

Check the method's status and finite output, then validate the numerical answer.
If a solve fails, inspect scaling, assumptions, initial guesses, step sizes,
and limits before increasing the iteration budget. [Known limitations](limitations.md)
distinguishes mathematical restrictions from implementation boundaries.

## Randomness and reproduction

Most stochastic numerical APIs accept `rng=` as a seed or Quadrivium generator.
Historical generator classes have separate `seed` interfaces. Repeatedly
passing the same integer restarts the same sequence; pass a generator to
advance a continuing stream across calls.

```pycon
>>> a = qd.monte_carlo(lambda x: x*x, 0, 1, n=1000, rng=42)
>>> b = qd.monte_carlo(lambda x: x*x, 0, 1, n=1000, rng=42)
>>> float(a) == float(b)
True

```

A repeated seed demonstrates reproducibility, not statistical independence.
Use independent streams for replications, and interpret Monte Carlo uncertainty
with repeated experiments or appropriate diagnostics. See [stochastic methods](guides/stochastic.md).

## Performance

The C engine handles array operations, and selected algorithms dispatch to
native kernels. Other methods use Python orchestration. Work can be dominated
by factorization, memory traffic, callback evaluations, or stored output,
depending on the problem.

Measure representative inputs, including their shapes and dtypes. Reuse factors,
exploit sparse operators, and record only needed output. Timing a tiny callback
says little about a simulation whose callback solves another model. The graphs
in this documentation primarily compare numerical error and algorithmic work;
they are not universal speed benchmarks.

Inspect the active backend with `qd.accel.show_config()`. The context manager
`qd.accel.disabled()` selects Python reference methods while retaining the
required C array engine. See [design](design.md) for the architecture.

## Continue by task

Choose a guide from the [home page](index.md), try the [example scripts](examples.md),
or build a [scientific workflow](guides/workflows.md). Keep the
[API reference](api/index.md) open for exact signatures and defaults.
