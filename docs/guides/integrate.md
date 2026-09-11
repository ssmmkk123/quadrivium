# Numerical integration

Numerical integration estimates an accumulated quantity from a callable or
sampled values. The method should match the information available: a smooth
function, an endpoint singularity, a known oscillatory factor, a periodic
signal, or a high-dimensional sampling problem.

Most callable quadrature routines return `QuadratureResult`, with `.value`,
`.error_estimate`, `.function_calls`, `.subintervals`, and `.converged`.
Sampled-data helpers return a scalar or array instead. Keep the result record
until accuracy has been assessed; converting to `float` extracts only a scalar
value. The [integration reference](../api/integrate.md) documents each routine.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import integrate as qi
>>> result = qi.quad(lambda x: x*x, 0.0, 1.0)
>>> result.converged and abs(result.value - 1.0/3.0) < 1e-12
True
>>> result.function_calls > 0
True

```

## Match the rule to the integrand

| Situation | Starting point | Important consideration |
| --- | --- | --- |
| General scalar callable on a finite interval | `quad` | Adaptive estimates can miss unsampled features |
| Known smooth callable and fixed node budget | `gauss_legendre` | No embedded error estimate |
| Smooth periodic function over a full period | `trapezoid_rule` | Endpoint periodicity is essential |
| Locally difficult but finite values | `adaptive_gauss_kronrod` | Split known trouble locations |
| Array- or complex-valued callable | `quad_vec` | Output shape must stay fixed |
| Integrable endpoint singularity | `tanh_sinh` | Evaluate only within the mathematical domain |
| Infinite interval | `quad`, or an appropriate weighted Gauss rule | Tail behavior still needs checking |
| Known rapidly oscillatory sine/cosine factor | `filon` | Pass the slow amplitude, not the full product |
| Tabulated data | `trapezoid_data`, `simpson_data` | Missing detail between samples cannot be recovered |
| Low-dimensional box | `tensor_gauss`, `double_integral` | Tensor node count grows exponentially with dimension |
| Smooth moderate-dimensional box | `sparse_grid_quadrature` | Benefit depends on smoothness and mixed interactions |
| Sampling-based high-dimensional estimate | `monte_carlo_nd`, `quasi_monte_carlo` | Statistical or sampling error needs independent assessment |

A narrow peak, discontinuity, or singularity should influence the setup before
you tighten a tolerance. No adaptive rule can refine a feature that all its
initial sample points fail to reveal.

## General-purpose scalar quadrature

`quad(f, a, b, tol=...)` uses globally adaptive Gauss-Kronrod 7/15 quadrature.
Each panel compares a seven-point Gauss estimate with a fifteen-point Kronrod
estimate; the panel with the largest estimated error is subdivided next.

For a finite scalar integral, the stopping target is
`estimated_error <= tol * max(1, abs(value))`. This combines absolute control
for small values and relative control for values larger than one. It is not
an independent pair of `epsabs` and `epsrel` parameters.

```pycon
>>> smooth = qi.quad(np.exp, 0.0, 1.0, tol=1e-11)
>>> smooth.converged and abs(float(smooth) - (np.e - 1.0)) < 1e-11
True
>>> smooth.error_estimate <= 1e-11 * max(1.0, abs(smooth.value))
True

```

`max_subdivisions` limits refinement. A result that exhausts this budget can
have `converged=False`; inspect the error estimate and increase the budget or
reformulate the problem. Very small requested tolerances may lie below the
integrand's evaluation accuracy or the arithmetic's useful resolution.

### Split known difficulty explicitly

For an interior kink or a known narrow feature, divide the interval at useful
locations and integrate the parts. Keep both error estimates and add them
conservatively when combining results.

```pycon
>>> left = qi.quad(lambda x: abs(x - 0.3), 0.0, 0.3)
>>> right = qi.quad(lambda x: abs(x - 0.3), 0.3, 1.0)
>>> abs(left.value + right.value - 0.29) < 1e-12
True
>>> left.converged and right.converged
True

```

The scalar `quad` interface does not accept an explicit `points` breakpoint
list. `quad_vec` does. When adding pieces, remember that applying the same
relative tolerance separately does not necessarily reproduce the tolerance
rule for the final sum, especially under cancellation.

`global_adaptive` currently delegates to the same Gauss-Kronrod implementation.
Its `rule` argument is not used to change the local rule. Use a named supported
method through `adaptive_quadrature` to select Simpson, trapezoid, or
Gauss-Kronrod behavior.

## Fixed composite rules and work comparisons

For composite rules, `n` generally specifies subintervals rather than function
calls. Trapezoid uses both endpoints; Simpson uses pairs of subintervals and
adjusts an odd count upward. Simpson 3/8 and Boole rules similarly need panel
counts compatible with their grouping.

```pycon
>>> trapezoid = qi.trapezoid_rule(np.exp, 0, 1, n=32)
>>> simpson = qi.simpson_rule(np.exp, 0, 1, n=32)
>>> abs(simpson.value - (np.e - 1)) < abs(trapezoid.value - (np.e - 1))
True

```

For sufficiently smooth functions before roundoff dominates, composite
trapezoid has second-order error, Simpson fourth-order error, and Boole
sixth-order error. These statements assume the required derivatives exist.
Refining across a kink can produce a different rate.

<figure markdown="span">
  ![Absolute integration error against actual integrand evaluations for trapezoid, Simpson, and Gauss-Legendre rules](../assets/figures/integrate-accuracy-budget.svg#only-light)
  ![Absolute integration error against actual integrand evaluations for trapezoid, Simpson, and Gauss-Legendre rules](../assets/figures/integrate-accuracy-budget-dark.svg#only-dark)
  <figcaption>For exp(x) on [0,1], the rules achieve different accuracy for the same evaluation budget. This smooth example favors high-order quadrature; singular or unresolved integrands can change the ranking. Exact-zero measured errors are shown at the labelled plotting floor.</figcaption>
</figure>

Compare actual callable evaluations, since node-count parameters differ among
methods and derivative-based corrections have additional cost. Fixed rules
usually return no error estimate; their default `converged=True` means the
rule was evaluated, not that a requested error tolerance was verified.

`romberg` applies Richardson extrapolation to successive trapezoid refinements.
It is useful for smooth integrands when the expected even-power error expansion
holds. `romberg_table` exposes the extrapolation table, while
`corrected_trapezoid` and `euler_maclaurin` use endpoint derivative corrections.
High-degree single-panel Newton-Cotes rules can have large alternating
weights; composite low-degree or Gaussian rules are often better conditioned.

## Gaussian quadrature and included weights

An n-point Gauss-Legendre rule is exact for polynomials through degree `2*n-1`
in exact arithmetic. Floating-point evaluation and node construction introduce
roundoff, so verify using a reasonable numerical tolerance.

```pycon
>>> polynomial = qi.gauss_legendre(lambda x: x**7, 0, 1, n=4)
>>> abs(float(polynomial) - 1.0/8.0) < 1e-12
True

```

Other Gaussian families include a weight in the integral. Pass the unweighted
factor `f`; multiplying by the weight again computes a different quantity.

| Rule | Domain | Weight included by the rule |
| --- | --- | --- |
| `gauss_legendre` | `[a,b]` | 1 |
| `gauss_hermite` | Real line | `exp(-x*x)` |
| `gauss_laguerre` | `[0,infinity)` | `x**alpha * exp(-x)` |
| `gauss_jacobi` | `[-1,1]` | `(1-x)**alpha * (1+x)**beta` |
| `gauss_chebyshev(kind=1)` | `[-1,1]` | `1/sqrt(1-x*x)` |
| `gauss_chebyshev(kind=2)` | `[-1,1]` | `sqrt(1-x*x)` |

```pycon
>>> hermite = qi.gauss_hermite(lambda x: x*x, n=12)
>>> abs(float(hermite) - np.sqrt(np.pi)/2) < 1e-10
True
>>> laguerre = qi.gauss_laguerre(lambda x: x**3, n=12)
>>> abs(float(laguerre) - 6.0) < 1e-10
True

```

Lobatto includes both endpoints and Radau includes one. `composite_gauss`
applies a fixed rule on multiple panels. Clenshaw-Curtis and Fejér use
Chebyshev-related nodes; some node sets are nested under suitable refinement,
but separate calls here should not be assumed to cache previous evaluations.
The [approximation module](approx.md) exposes node and weight construction.

## Array and complex integrands

`quad_vec` shares quadrature nodes across all output components and preserves
the integrand's shape. Outputs must remain finite and have a fixed shape on
every evaluation. Real and complex arrays are supported. In the current numeric backend, a
complex scalar return can fail during panel assembly; return a length-one
complex array and extract its integrated component as shown below.

```pycon
>>> vector = qi.quad_vec(lambda x: np.array([x, x*x]), 0.0, 1.0,
...                      epsabs=1e-11, epsrel=1e-10)
>>> vector.converged and np.allclose(vector.value, [0.5, 1.0/3.0])
True
>>> vector.value.shape, vector.error_estimate.shape
((2,), (2,))
>>> complex_result = qi.quad_vec(lambda x: np.array([np.exp(1j*x)]), 0, np.pi)
>>> abs(complex_result.value[0] - 2j) < 1e-10
True

```

With `norm="max"`, each component is checked against
`epsabs + epsrel*abs(component_integral)`. `epsabs` may be an array broadcasting
to the integrand shape. With `norm="2"`, Euclidean norms control the aggregate
error. Scale components or choose componentwise absolute tolerances when their
units and magnitudes differ.

`limit` bounds the retained panels, and `points` supplies interior breakpoints
for finite limits. Breakpoints must fit the panel budget. The returned error
estimate retains the component shape even when aggregate norm control is used.
Do not call `float(result)` on a vector or genuinely complex result.

## Infinite intervals and singularities

Scalar `quad` transforms infinite limits to finite coordinates and applies
adaptive quadrature there. For example, a half-line uses a rational map of a
finite interval. This helps evaluate decaying tails but is not a guarantee
that every improper integral exists or that a slowly decaying tail is resolved.

```pycon
>>> gaussian = qi.quad(lambda x: np.exp(-x*x), -np.inf, np.inf)
>>> gaussian.converged and abs(float(gaussian) - np.sqrt(np.pi)) < 1e-9
True

```

Check tails by an independent transformation, analytic bound, or split into
finite and tail contributions. A very large finite endpoint merely truncates
the problem; it does not reproduce an infinite-domain calculation automatically.

`tanh_sinh` clusters evaluations near finite endpoints through a double
exponential transformation. It is often effective for integrable endpoint
singularities such as `1/sqrt(x)` and `log(x)`.

```pycon
>>> endpoint = qi.tanh_sinh(lambda x: 1/np.sqrt(x), 0.0, 1.0)
>>> abs(float(endpoint) - 2.0) < 1e-9
True

```

An interior pole requires specifying what quantity is intended. A Cauchy
principal value is different from an ordinary improper integral.
`cauchy_principal_value(f, a, b, c)` interprets the integrand as `f(x)/(x-c)`;
pass the smooth numerator. `hadamard_finite_part` uses the corresponding
second-order pole convention.

```pycon
>>> principal = qi.cauchy_principal_value(lambda x: 1.0, -1.0, 1.0, c=0.0)
>>> abs(float(principal)) < 1e-12
True

```

Neither method turns an otherwise divergent ordinary integral into a convergent
one. State the regularized mathematical definition with the reported result.

## Oscillatory integrals

`filon(f, a, b, omega, kind)` separates a smooth amplitude from a known sine or
cosine factor. Supply `f(x)` alone; `kind="sin"` means the full integrand is
`f(x)*sin(omega*x)`. The method integrates the oscillatory factor analytically
within its amplitude approximation.

```pycon
>>> omega = 100.0
>>> oscillatory = qi.filon(lambda x: 1.0, 0.0, 1.0, omega=omega, kind="sin")
>>> abs(float(oscillatory) - (1 - np.cos(omega))/omega) < 1e-10
True

```

A fixed amplitude grid can remain effective as frequency increases when the
amplitude stays smooth. It does not resolve an independently oscillating,
sharp, or singular amplitude for free. Cancellation can make a relative target
unhelpful when the integral is near zero; choose an absolute scale too.

## Integrating sampled data

`trapezoid_data(x,y)` accepts nonuniform coordinates and returns a float.
`simpson_data` integrates local quadratic fits over pairs of intervals and
uses trapezoid on a final unmatched interval. `cumulative_trapezoid` returns
one cumulative value per coordinate, including the supplied initial value.

```pycon
>>> coordinates = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
>>> samples = coordinates**2
>>> abs(qi.simpson_data(coordinates, samples) - 1.0/3.0) < 1e-12
True
>>> cumulative = qi.cumulative_trapezoid(coordinates, samples, initial=0.0)
>>> cumulative.shape, float(cumulative[0])
((5,), 0.0)

```

Keep coordinates ordered in the intended integration direction and values
aligned with them. Duplicated coordinates are unsuitable for the quadratic
formula because interval ratios divide by spacing. Uneven sampling can produce
negative quadratic weights and sensitivity to noise. A rule's formal order
cannot compensate for an unmeasured narrow peak between samples.

## Monte Carlo and several dimensions

Monte Carlo estimates an integral from random samples. For independent samples
with finite variance, standard error typically decreases as `1/sqrt(n)`.
One extra decimal digit can therefore require about one hundred times as many
samples. Set `rng` for reproducibility, and inspect variation across independent
seeds rather than selecting a favorable run.

```pycon
>>> sampled = qi.monte_carlo(lambda x: x*x, 0.0, 1.0, n=10000, rng=0)
>>> sampled.error_estimate > 0
True
>>> abs(sampled.value - 1.0/3.0) < 5*sampled.error_estimate
True

```

Here `.error_estimate` is a statistical standard-error estimate, not a
Gauss-Kronrod discrepancy or a guaranteed bound. Heavy-tailed samples and
importance weights can invalidate a simple finite-variance interpretation.
Stratification, control variates, importance sampling, and antithetic variates
reduce variance when their assumptions match the integrand.

Quasi-Monte Carlo uses structured sample sets. `quasi_monte_carlo` supports
Halton, Sobol, and Latin-hypercube choices and returns no error estimate.
Halton and Sobol choices are deterministic here; the Latin-hypercube option
uses random sampling without an exposed seed on this wrapper. Test increasing
sample counts and different constructions rather than reading
`converged=True` as an accuracy certificate.

`double_integral` accepts `f(x,y)` and inner bounds that may depend on `x`.
`triple_integral` extends the idea to three variables. These are fixed-order
nested Gaussian rules, not adaptive SciPy-style wrappers.

```pycon
>>> triangle = qi.double_integral(lambda x, y: 1.0, 0, 1, 0, lambda x: 1-x)
>>> abs(float(triangle) - 0.5) < 1e-12
True

```

`tensor_gauss` accepts a point-vector callable on a box and costs the product
of per-axis node counts. Sparse grids reduce that tensor cost for suitable
smooth low-interaction integrands, while Monte Carlo is less directly
penalized by dimension. There is no universal dimension at which one wins:
compare accuracy and evaluations on the actual problem.

## Interpreting diagnostics honestly

Error-estimate semantics differ by routine. Gauss-Kronrod reports a sum of
embedded discrepancies; Monte Carlo reports standard error; fixed rules may
report `None`. The current adaptive Simpson and trapezoid implementations
store the requested tolerance as their estimate rather than an independently
accumulated final error. Adaptive trapezoid also does not flag depth-budget
exhaustion in its convergence field. Use a refinement comparison when relying
on those methods.

Report the integral, method, domain, tolerance or sampling budget, convergence
status, and the meaning of the error estimate. For a critical quantity, compare
an independent rule, a transformed calculation, or an analytic identity.
See [core conventions](core.md) for result interpretation and
[stochastic methods](stochastic.md) for sampling tools.
