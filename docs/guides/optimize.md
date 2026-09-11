# Optimization and fitting

Optimization asks for parameters that make an objective small. A useful result
needs more than a small reported value: the parameters must obey constraints,
the objective must represent the intended problem, and the stopping criterion
must be appropriate for its scale. Quadrivium exposes both complete optimizers
and components such as line searches, projections, and proximal operators.

Use the [optimization reference](../api/optimize.md) for full signatures.
This guide explains which problem structure each interface uses, how to read
its result, and how to check an answer independently.

## Start with a smooth scalar objective

```pycon
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> objective = lambda x: (x[0] - 2.0)**2 + 4.0 * (x[1] + 1.0)**2
>>> gradient = lambda x: np.array([2 * (x[0] - 2), 8 * (x[1] + 1)])
>>> result = qd.minimize(objective, [0.0, 0.0], grad_f=gradient)
>>> result.converged, np.allclose(result.x, [2.0, -1.0], atol=1e-7)
(True, True)
>>> float(result.fun) < 1e-12
True

```

`f(x)` receives a one-dimensional parameter vector and returns a scalar.
`grad_f(x)` returns a vector of the same length. The default dispatcher method
is BFGS. If a gradient is omitted, methods that require it generally estimate
it with finite differences, which costs extra objective evaluations and can
be unreliable for a noisy or discontinuous objective.

Most iterative minimizers return `OptimizeResult`:

| Attribute | Interpretation |
| --- | --- |
| `x` | Returned parameter vector, or scalar for one-dimensional search |
| `fun` | Objective value at the returned parameters |
| `converged`, `message` | Whether a method's stopping test passed and why it stopped |
| `iterations` | Outer iterations reported by the method |
| `function_calls`, `gradient_calls` | Evaluation counters where recorded |
| `jac`, `hess` | Derivative or curvature information where supplied |
| `history` | Recorded iterates or method-specific progress values |

There is no universal `success` field for these results: use `converged`.
A true flag means the method's stopping criterion was met. It is not proof of
a global optimum or evidence that the underlying model is correct.

## Let the problem structure choose the method

| Structure | Starting method | Why |
| --- | --- | --- |
| One scalar parameter in an interval | `brent_minimize`, `golden_section` | Bracket reduction needs no gradient |
| Smooth objective, moderate dimension | `bfgs` | Learns curvature from successive gradients |
| Many smooth parameters | `lbfgs` | Stores a limited number of curvature pairs |
| Simple parameter bounds | `lbfgsb` | Projects onto a box and handles active variables |
| Hessian-vector products available | `newton_cg` | Avoids forming a dense Hessian |
| Nonconvex quadratic models | `trust_region`, `modified_newton` | Controls steps when curvature is indefinite |
| Residual vector from data | `least_squares`, `curve_fit` | Uses least-squares structure and robust losses |
| Objective values only | `nelder_mead`, `powell`, `compass_search` | No derivative callback required |
| Multiple attraction basins | `differential_evolution`, `basin_hopping`, `cma_es` | Broader search with a finite evaluation budget |
| Nonlinear equalities/inequalities | `sqp`, `augmented_lagrangian` | Models constraints explicitly |
| Smooth loss plus a simple nonsmooth penalty | `proximal_gradient`, `fista`, `admm` | Uses a proximal operator for the penalty |
| Linear objective and linear constraints | `linprog` | Exploits the linear program directly |

Call specialized functions directly when their arguments convey the structure
more clearly. `minimize` is a dispatcher with a defined method table; it does
not accept every exported optimizer name. In particular, call `lbfgsb`,
`newton_cg`, `least_squares`, and `dual_annealing_lite` directly.
For population methods dispatched through `minimize`, its `x0` position is
used as a bounds sequence; direct calls avoid that overloaded meaning.

## Scale variables and verify derivatives

Suppose one parameter is naturally around `1e-6` and another around `1e3`.
A unit step has very different meaning in the two directions. Define scaled
variables `x = center + scale*z`, optimize over `z`, and transform derivatives
by the chain rule. For `g(z)=f(center+scale*z)`, the gradient is
`scale*grad_f(x)` for a diagonal scale vector.

A simple derivative check compares a directional finite difference with the
analytic directional derivative:

```pycon
>>> point = np.array([0.3, -0.2])
>>> direction = np.array([0.4, -0.7])
>>> h = 1e-5
>>> difference = (objective(point + h*direction) - objective(point - h*direction)) / (2*h)
>>> abs(float(difference - gradient(point) @ direction)) < 1e-8
True

```

Try several points and step sizes. Agreement for one direction at one point
can miss an indexing or sign error. Finite differences themselves lose accuracy
at excessively small steps, and stochastic objectives need a controlled random
stream or a different validation strategy.

<figure markdown="span">
  ![Gradient descent and BFGS on an anisotropic quadratic, with objective histories](../assets/figures/optimize-scaling-paths.svg#only-light)
  ![Gradient descent and BFGS on an anisotropic quadratic, with objective histories](../assets/figures/optimize-scaling-paths-dark.svg#only-dark)
  <figcaption>A narrow quadratic valley gives very different curvature along its axes. Paths explain the direction of progress, while objective histories quantify it. The comparison illustrates the role of curvature and scaling; iteration counts alone do not account for differing evaluation cost.</figcaption>
</figure>

## Bracket a one-dimensional minimum

```pycon
>>> from quadrivium.optimize import brent_minimize, golden_section
>>> one = brent_minimize(lambda x: (x - 2.0)**2 + 1.0, 0.0, 5.0)
>>> round(float(one.x), 6), round(float(one.fun), 6)
(2.0, 1.0)

```

Golden-section and Fibonacci searches repeatedly shrink an interval containing
a minimum under a unimodality assumption. Brent's method also attempts
parabolic interpolation when it is useful. A minimum bracket is not a
sign-change bracket for a root: minimizing `f` and solving `f=0` are different
problems. If the interval contains several local minima, interval reduction
does not guarantee that the lowest one will be selected.

`bracket_minimum` searches for a bracket from initial points.
`line_minimize_1d` performs outward bracketing and scalar minimization for a
function of a step length. Boundaries that are physically meaningful should be
represented deliberately; do not rely on a search happening to stay inside a
domain where the objective is defined.

## Curvature, line searches, and memory

BFGS updates an inverse-Hessian approximation using parameter and gradient
differences. Its dense curvature storage grows quadratically with parameter
count. L-BFGS stores only the most recent `m` pairs, reducing curvature storage
to approximately `O(m*n)`.

A line search chooses the distance along a proposed descent direction.
`backtracking` enforces sufficient decrease; `strong_wolfe` also tests
curvature. These safeguards improve the reliability of quasi-Newton updates,
but their assumptions still require meaningful objective and gradient values.
`exact_line_search` solves a bounded scalar subproblem numerically; “exact”
in its name does not mean an analytic or zero-error answer.

`newton_method` uses a Hessian, while `newton_cg` can use
`hess_vec(x, v)` products. Trust-region methods compare actual improvement
with the improvement predicted by a local model. A poor ratio leads to a
smaller region. The dogleg subproblem suits positive-definite models;
Steihaug conjugate gradients can stop on negative curvature or the boundary.

For deterministic smooth objectives, adaptive learning-rate rules such as
Adam are alternatives, not automatic replacements for curvature methods.
Their learning-rate choices and stopping behavior still need validation.
A function name does not add minibatching or a stochastic training pipeline.

## Bounds and general constraints

Bound interfaces differ. `lbfgsb` takes one `(lower, upper)` pair per
parameter; `None` can mark an unbounded side. Robust `least_squares` instead
takes `(lower_vector, upper_vector)`, with scalar values broadcast if desired.

```pycon
>>> from quadrivium.optimize import lbfgsb
>>> bounded = lbfgsb(lambda x: (x[0] - 3)**2, [0.0],
...                   grad_f=lambda x: 2 * (x - 3), bounds=[(0.0, 1.0)])
>>> np.allclose(bounded.x, [1.0], atol=1e-8)
True

```

A constrained optimum can have a nonzero raw gradient because the improving
direction points outside the feasible set. Check projected gradients or KKT
conditions in that case.

Nonlinear equality callbacks use `eq(x)=0`; inequality callbacks use
`ineq(x)<=0`. Return vectors to express several constraints.

```pycon
>>> from quadrivium.optimize import sqp
>>> constrained = sqp(lambda x: float(x @ x), [2.0, -1.0],
...     eq=lambda x: np.array([x[0] + x[1] - 1.0]), tol=1e-10)
>>> np.allclose(constrained.x, [0.5, 0.5], atol=1e-7)
True
>>> abs(float(np.sum(constrained.x)) - 1.0) < 1e-8
True

```

Penalty methods discourage violation with a growing penalty; very large
penalties can worsen conditioning. Barrier methods require a strictly feasible
interior start. Augmented Lagrangian methods also estimate multipliers.
`projected_gradient` is useful when a projection is inexpensive;
`project_box`, `project_simplex`, and `project_ball` supply common projections.
Check feasibility separately from the objective and stationarity.

## Fit a residual vector

`least_squares` accepts `fun(parameters)` returning a one-dimensional residual
vector of constant length. For `m` observations and `n` parameters, its Jacobian
has shape `(m, n)`. With the default linear loss the objective is
`0.5*sum(residual**2)`.

```pycon
>>> from quadrivium.optimize import least_squares
>>> times = np.linspace(0, 2, 21)
>>> observed = 2.5 * times + 0.4
>>> residual = lambda p: p[0] * times + p[1] - observed
>>> fit = least_squares(residual, [1.0, 0.0], bounds=([0.0, -1.0], [4.0, 1.0]))
>>> fit.converged, np.allclose(fit.x, [2.5, 0.4], atol=1e-6)
(True, True)
>>> fit.residuals.shape, fit.active_mask.tolist()
((21,), [0, 0])

```

Besides standard result fields, this solver provides `residuals`,
`optimality`, `active_mask`, and `jacobian`. `active_mask` uses `-1` for a
lower bound, `1` for an upper bound, and `0` for a free coordinate.
`jac` in the generic result is the objective gradient; `jacobian` is the
residual Jacobian as an operator. These are different mathematical objects.

Use `ftol`, `xtol`, and `gtol` for changes in cost, changes in parameters, and
projected gradient. `max_nfev` limits residual evaluations, including numerical
Jacobian work. A tiny parameter update can reflect poor scaling rather than a
well-determined fit, so inspect residuals and parameter sensitivity as well.

### Robust losses and sparse Jacobians

Available losses are `linear`, `huber`, `soft_l1`, and `cauchy`.
`f_scale` sets the residual scale at which robust downweighting becomes
important. If observations have known standard deviations, standardize the
residuals first so the scale has an interpretable meaning.

```pycon
>>> contaminated = np.array([1.0, 1.0, 1.0, 1.0, 20.0])
>>> linear = least_squares(lambda p: p[0] - contaminated, [0.0])
>>> robust = least_squares(lambda p: p[0] - contaminated, [0.0],
...                        loss="soft_l1", f_scale=0.2)
>>> abs(float(robust.x[0]) - 1.0) < abs(float(linear.x[0]) - 1.0)
True

```

Robust loss changes the objective; it does not identify which measurements
are erroneous. Compare fitted residuals and the scientific meaning of the
outlying observations before accepting that change.

`jac` can be `"2-point"`, `"3-point"`, or a callable returning a dense matrix,
sparse matrix, or `LinearOperator` with both forward and adjoint products.
The solver applies normal-equation products without assembling a dense normal
matrix. `jac_sparsity` groups finite-difference columns that do not affect the
same residual rows. Its structural zeros must be correct: an omitted nonzero
can produce a wrong derivative. `x_scale="jac"` estimates parameter scales
from Jacobian column norms; explicit positive scales are also accepted.

## Fit a named model

`curve_fit` wraps residual construction around `model(xdata, *parameters)`.
Its `jac` callback follows the package convention **`jac(parameters)`**, not
`jac(xdata, *parameters)`.

```pycon
>>> from quadrivium.optimize import curve_fit
>>> model = lambda t, amplitude, rate: amplitude * np.exp(-rate * t)
>>> values = model(times, 2.0, 0.7)
>>> curve = curve_fit(model, times, values, [1.0, 1.0])
>>> np.allclose(curve.x, [2.0, 0.7], atol=1e-6)
True

```

Positive `sigma` values weight residuals by their reciprocal. For ordinary
linear loss, the wrapper estimates `covariance` and `std_errors` from the
local Jacobian unless `compute_covariance=False`. `absolute_sigma=True`
treats supplied standard deviations as absolute rather than rescaling by the
residual variance. Singular curvature can leave covariance unavailable.

For robust losses, the current implementation returns `covariance=None`
and `std_errors=None`. Even an available local covariance is an approximation
whose usefulness depends on identifiability, model adequacy, and noise
assumptions. Bounds, parameter degeneracy, and strong nonlinearity can make a
symmetric local uncertainty summary misleading.

## Global search, regularization, and linear programs

Population methods such as `differential_evolution` take a finite bounds box
and `rng=` for reproducibility. Repeat with independent seeds and compare
objective values and feasibility. Population stagnation is a useful stopping
signal, not a certificate that the global minimum has been found.
`basin_hopping` combines perturbations with local minimization; `cma_es`
adapts covariance and can have substantial quadratic storage cost.

For `f(x)+g(x)`, proximal methods handle a smooth term with a gradient and a
simple nonsmooth term with a proximal map. `lasso` minimizes
`0.5*||A*x-b||**2 + lam*||x||_1`; `ridge` uses an L2 penalty and returns a
coefficient vector directly. L1 encourages exact zero coefficients, whereas
L2 shrinks coefficients continuously. Standardize columns before comparing
penalties across variables with different units.

```pycon
>>> from quadrivium.optimize import soft_threshold
>>> soft_threshold(np.array([-2.0, -0.2, 0.0, 1.5]), 0.5).tolist()
[-1.5, -0.0, 0.0, 1.0]
>>> lp = qd.linprog([-3.0, -2.0], A_ub=[[1.0, 1.0], [1.0, 3.0]],
...                 b_ub=[4.0, 6.0])
>>> np.allclose(lp.x, [4.0, 0.0]), round(float(lp.fun), 6)
(True, -12.0)

```

The LP interface minimizes `c @ x` with nonnegative variables. Negate `c` to
express maximization, as above. Unrestricted-sign variables need an explicit
reformulation. `A_ub*x <= b_ub` and `A_eq*x = b_eq` encode linear constraints;
`assignment_problem` handles the specialized matching problem.

## Histories and independent checks

Use `store_history=False` when only the result matters, or `history_stride=k`
when a coarser iteration trace is sufficient. `callback(x)` receives a private
copy; returning `True` or raising `StopIteration` stops the optimizer. History
contents and evaluation counters vary across method families, so inspect the
chosen method before plotting a generic result.

Before using fitted parameters, recompute the objective, check constraints,
validate derivatives, and perturb the starting point. Compare models at a
common evaluation budget when expensive simulation calls dominate. For a
simulation-based objective, tighten the simulator tolerances to confirm that
the optimizer is not fitting numerical error.

See [differentiation](diff.md) for gradients, [linear algebra](linalg.md) for
least-squares systems, and [scientific workflows](workflows.md) for parameter
recovery through an ODE solve.
