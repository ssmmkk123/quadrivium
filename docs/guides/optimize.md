# Optimization

```python
from quadrivium.optimize import lbfgs, curve_fit, differential_evolution
import quadrivium as qd          # qd.minimize, qd.bfgs, qd.linprog, ...
```

95 routines: one-dimensional searches, line searches, gradient and quasi-Newton
methods, trust regions, derivative-free search, global optimizers, constrained
methods, proximal splitting, and linear programming. Full signatures are in the
[`optimize` reference](../api/optimize.md).

## The default path

```pycon
>>> import numpy as np
>>> import quadrivium as qd
>>> rosen = lambda v: (1 - v[0])**2 + 100*(v[1] - v[0]**2)**2
>>> res = qd.minimize(rosen, [-1.2, 1.0], method="bfgs")
>>> res.converged, [round(float(v), 6) for v in res.x], round(res.fun, 12)
(True, [1.0, 1.0], 0.0)

```

`minimize` dispatches on `method=`, passing the rest through. Gradient-based:
`bfgs`, `lbfgs`, `dfp`, `sr1`, `newton`, `modified_newton`, `cg_fr`, `cg_pr`,
`cg_hs`, `gradient_descent`, `momentum`, `nesterov`, `adam`, `adagrad`,
`rmsprop`, `barzilai_borwein`, `trust_region`. Derivative-free: `nelder_mead`,
`powell`, `hooke_jeeves`, `compass_search`, `coordinate_descent`. Global
(these take `bounds` in place of `x0`): `differential_evolution`,
`particle_swarm`, `simulated_annealing`, `genetic_algorithm`, `basin_hopping`,
`cma_es`, `dual_annealing_lite`.

Supplying an exact gradient always helps, and the result records how much:

```pycon
>>> grad = lambda v: np.array([-2*(1 - v[0]) - 400*v[0]*(v[1] - v[0]**2),
...                            200*(v[1] - v[0]**2)])
>>> fd = qd.minimize(rosen, [-1.2, 1.0], method="bfgs")
>>> exact = qd.minimize(rosen, [-1.2, 1.0], method="bfgs", grad_f=grad)
>>> exact.function_calls < fd.function_calls / 3
True

```

## Choosing a method

| Situation | Method |
| --- | --- |
| smooth, few variables, gradient available | `bfgs` |
| smooth, many variables (memory matters) | `lbfgs`, `newton_cg` |
| smooth with simple bounds | `lbfgsb` |
| Hessian available and cheap | `newton_method`, `trust_region` |
| Hessian available but indefinite | `modified_newton`, `trust_region` with Steihaug-CG |
| nonsmooth or noisy objective | `nelder_mead`, `powell`, `compass_search` |
| sum of squares (data fitting) | `levenberg_marquardt`, `gauss_newton`, `curve_fit` |
| many local minima | `differential_evolution`, `cma_es`, `basin_hopping` |
| stochastic objective, huge dimension | `adam`, `rmsprop`, `momentum` |
| equality or inequality constraints | `sqp`, `augmented_lagrangian`, `penalty_method` |
| constraint is a simple set to project onto | `projected_gradient` |
| objective is smooth + a nonsmooth penalty | `fista`, `admm`, `lasso` |
| linear objective and constraints | `linprog`, `simplex`, `interior_point_lp` |
| assignment or matching | `assignment_problem` |
| one variable, bracketed | `brent_minimize`, `golden_section` |

## One dimension

```pycon
>>> from quadrivium.optimize import brent_minimize, golden_section
>>> r = brent_minimize(lambda x: (x - 2)**2 + 1, 0, 5)
>>> round(float(r.x), 7), round(float(r.fun), 12)
(2.0, 1.0)

```

`golden_section` and `fibonacci_search` reduce the bracket by a fixed factor
each step and need no derivative or smoothness — they only need unimodality.
`parabolic_interpolation` converges much faster on a smooth function but can
fail; `brent_minimize` combines the two, which is why it is the default.
`bracket_minimum` finds an initial bracket.

## Line searches

Every gradient method must decide how far to move along a direction. The line
searches are exposed separately, so a method can be assembled from parts:

| Function | Condition enforced |
| --- | --- |
| `backtracking`, `armijo` | sufficient decrease |
| `goldstein` | sufficient decrease, two-sided |
| `wolfe` | decrease and curvature |
| `strong_wolfe` | decrease and \|curvature\|, with interpolating zoom |
| `exact_line_search` | exact minimum along the ray |

The strong Wolfe conditions are what `bfgs` and `lbfgs` use, because the
curvature condition is what keeps the quasi-Newton update positive definite.

## Quasi-Newton methods

BFGS builds an approximate inverse Hessian from successive gradients. It
converges superlinearly without ever forming or factorizing a Hessian:

```pycon
>>> from quadrivium.optimize import bfgs, lbfgs, sr1, dfp
>>> for method in (bfgs, dfp, sr1, lbfgs):
...     r = method(rosen, [-1.2, 1.0], grad, tol=1e-10)
...     print(f"{method.__name__:6s} converged={r.converged}  f < 1e-18: {r.fun < 1e-18}")
bfgs   converged=True  f < 1e-18: True
dfp    converged=True  f < 1e-18: True
sr1    converged=True  f < 1e-18: True
lbfgs  converged=True  f < 1e-18: True

```

`lbfgs` keeps only the last `m` update pairs, so its memory is `O(mn)` rather
than `O(n²)` — the difference between feasible and impossible in high
dimensions. `newton_cg` (truncated Newton) goes further: it never forms the
Hessian at all, solving each Newton system approximately with CG and
terminating on a forcing sequence.

```pycon
>>> from quadrivium.optimize import newton_cg
>>> quad = lambda v: float(v @ (np.arange(1, 51) * v))       # 50 variables
>>> qgrad = lambda v: 2 * np.arange(1, 51) * v
>>> r = newton_cg(quad, np.ones(50), qgrad, tol=1e-10)
>>> r.converged, float(np.max(np.abs(r.x))) < 1e-6
(True, True)

```

## Trust regions

A line search picks a direction then a distance; a trust region picks a radius
then the best step inside it. That is the more robust order when the model may
be a poor fit — a nonconvex region, an indefinite Hessian:

```pycon
>>> tr = qd.trust_region(rosen, [-1.2, 1.0], grad, tol=1e-10)
>>> tr.converged, [round(float(v), 6) for v in tr.x]
(True, [1.0, 1.0])

```

The subproblem solver is selectable: `cauchy_point` (steepest descent to the
boundary — cheap, and genuinely only linearly convergent), `dogleg` (needs a
positive definite model), `steihaug_cg` (handles indefiniteness and scales to
large problems).

## Least squares and curve fitting

For a sum of squared residuals, the Gauss-Newton approximation to the Hessian
costs nothing extra. Levenberg-Marquardt interpolates between Gauss-Newton and
gradient descent, which is what makes it reliable far from the solution:

```pycon
>>> from quadrivium.optimize import curve_fit
>>> t = np.linspace(0, 4, 40)
>>> y = 2.5 * np.exp(-1.3 * t) + 0.5
>>> model = lambda x, a, b, c: a * np.exp(-b * x) + c      # model(x, *params)
>>> fit = curve_fit(model, t, y, [1.0, 1.0, 0.0])
>>> [round(float(v), 6) for v in fit.x]
[2.5, 1.3, 0.5]

```

## Derivative-free methods

When the objective is noisy, discontinuous, or a simulation you cannot
differentiate, these use only function values:

```pycon
>>> from quadrivium.optimize import nelder_mead, powell
>>> nm_res = nelder_mead(rosen, [-1.2, 1.0], tol=1e-12)
>>> nm_res.converged, round(float(nm_res.fun), 10)
(True, 0.0)

```

`nelder_mead` reflects and contracts a simplex; `powell` does successive line
searches along conjugate directions; `hooke_jeeves` and `compass_search` are
pattern searches with convergence guarantees on smooth functions.

## Global optimization

A local method finds the nearest minimum. When there are many, the search must
be global — and no global method can promise the true optimum in finite time,
so what you choose is a sampling strategy:

```pycon
>>> rastrigin = lambda v: 20 + sum(x**2 - 10*np.cos(2*np.pi*x) for x in v)
>>> de = qd.differential_evolution(rastrigin, [(-5.12, 5.12)] * 2, rng=0, tol=1e-10)
>>> float(de.fun) < 1e-8
True

```

`cma_es` adapts a full covariance matrix and is the strongest general choice
on ill-conditioned nonconvex problems; `particle_swarm` and
`genetic_algorithm` are population methods; `simulated_annealing` and
`basin_hopping` accept worse points to escape local minima;
`dual_annealing_lite` alternates annealing with local refinement. All take
`rng=` for reproducibility.

## Constrained optimization

```pycon
>>> from quadrivium.optimize import sqp
>>> # minimize x² + y² subject to x + y = 1  →  (0.5, 0.5)
>>> con = sqp(lambda v: v[0]**2 + v[1]**2, [2.0, -1.0],
...           eq=lambda v: np.array([v[0] + v[1] - 1.0]), tol=1e-12)
>>> [round(float(v), 8) for v in con.x]
[0.5, 0.5]

```

| Method | Approach |
| --- | --- |
| `penalty_method` | add a growing penalty for violation; simple, ill-conditioned at the end |
| `barrier_method` | stay strictly feasible, push the barrier down |
| `augmented_lagrangian` | penalty plus multiplier estimates; avoids the ill-conditioning |
| `sqp` | solve a quadratic model with linearized constraints each step |
| `projected_gradient` | step, then project back onto the feasible set |
| `active_set_qp`, `solve_qp` | quadratic objective, linear constraints |

`project_box`, `project_simplex`, and `project_ball` are the projections;
`kkt_residual` measures how close a point is to satisfying the KKT conditions,
which is the honest way to check a constrained answer.

## Proximal and sparse methods

For `f(x) + g(x)` with `f` smooth and `g` nonsmooth but simple, proximal
gradient methods take a gradient step on `f` and a proximal step on `g`.
FISTA's momentum improves the rate from `O(1/k)` to `O(1/k²)`:

```pycon
>>> from quadrivium.optimize import lasso
>>> rng = np.random.default_rng(0)
>>> A = rng.standard_normal((60, 30))
>>> x_true = np.zeros(30); x_true[[3, 11, 25]] = [2.0, -3.0, 1.5]
>>> b = A @ x_true
>>> x_hat = lasso(A, b, lam=0.05)
>>> int(np.sum(np.abs(x_hat.x) > 1e-6)) <= 8      # sparse, as asked
True

```

`ista`, `fista`, `proximal_gradient`, `admm`, `admm_lasso`, and
`douglas_rachford` are the general splitting methods; `soft_threshold`,
`prox_l1`, `prox_l2`, `prox_box`, and `prox_nonneg` are the proximal
operators; `ridge` and `elastic_net` complete the regularized regression set.

## Linear programming

```pycon
>>> # maximize 3x + 2y subject to x + y ≤ 4, x + 3y ≤ 6, x, y ≥ 0
>>> res = qd.linprog([-3.0, -2.0], A_ub=[[1.0, 1.0], [1.0, 3.0]], b_ub=[4.0, 6.0])
>>> [round(float(v), 8) for v in res.x], round(float(res.fun), 8)
([4.0, 0.0], -12.0)

```

`simplex` is the two-phase method with Bland's rule available for degenerate
pivots; `big_m_simplex` and `two_phase_simplex` handle the initial feasible
basis differently; `interior_point_lp` is a primal-dual path-following method,
which is the one that scales. `assignment_problem` solves the rectangular
assignment problem by the Hungarian algorithm in `O(n³)`.

## Pitfalls

- **`converged=True` means a stationary point, not a global minimum.** For a
  nonconvex objective, restart from several points or use a global method.
- **A finite-difference gradient limits your accuracy to about `√ε`.** Asking
  for `tol=1e-12` without an exact gradient will usually stall.
- **Scaling matters more than the method.** Variables differing by orders of
  magnitude will defeat any of these; rescale so a unit step means something
  comparable in each direction.
- **Nelder-Mead can converge to a non-stationary point.** It has no
  convergence theory in more than one dimension; verify with a gradient check.
- **Global methods have no stopping criterion worth trusting.** `max_iter` is
  the real budget; `tol` only detects that the population has collapsed.
- **Penalty methods become ill-conditioned as the penalty grows.** Use
  `augmented_lagrangian` when the constraints must be satisfied tightly.

## See also

- [`optimize` API reference](../api/optimize.md) — every signature.
- [Differentiation guide](diff.md) — exact gradients and Hessians by AD.
- [Root finding guide](rootfind.md) — stationarity is `∇f(x) = 0`.
- [Linear algebra guide](linalg.md) — the least squares family.
- `examples/04_optimization.py` — a runnable tour.
