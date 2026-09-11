# Known limitations

Quadrivium exposes many methods, but their domains and guarantees differ.
This page collects boundaries that matter across tasks. Read the relevant
guide and API entry before treating a familiar method name as a promise of
another library's behavior.

## Arrays and precision

The array engine supports `bool`, `int64`, `float32`, `float64`, `complex64`,
and `complex128`. Higher-level algorithms frequently convert to real double
precision using shared helpers. Compact or complex storage support in
`quadrivium.numeric` does not imply that every solver preserves that dtype or
accepts complex values.

There is no general arbitrary-precision, interval-arithmetic, GPU, or distributed
array backend. Buffer interoperability is useful, but Quadrivium is not a
complete implementation of the NumPy API. Check the [array guide](guides/numeric.md)
for supported operations and the distinction between views and copies.

Double precision has finite relative resolution, but a universal absolute
accuracy floor such as `1e-16` is misleading: scale, conditioning, cancellation,
and underflow all matter. Zero error on one special input does not establish
that accuracy everywhere.

## Results and error estimates

| Reported quantity | What it tells you | Additional check |
| --- | --- | --- |
| `converged=True` | A method-specific stopping condition was met | Residual, gradient, constraints, or refinement |
| ODE `success=True` | The integrator completed or terminated as reported | Read `message`; compare achieved output and requested endpoint |
| `error_estimate` | An estimator associated with a particular rule | Independent reference or refinement; inspect difficult features |
| Small linear residual | A nearby equation is solved accurately | Conditioning and input uncertainty |
| Small objective change | The cost stopped changing significantly | Stationarity and parameter sensitivity |
| MCMC diagnostic near its target | No problem detected by that diagnostic | Multiple chains, trace behavior, tail exploration |

Not every routine populates every record field. Some fixed rules return
scalars; some diagnostics are absent when histories are disabled. A generic
pipeline must branch on the actual return contract rather than assume all
methods share one status field.

Malformed input may raise package-specific exceptions, standard `ValueError`
or `TypeError`, or numeric linear-algebra exceptions. Nonconvergence is often
returned as data, but this is not universal.

## Linear algebra

A dense factorization still requires storage proportional to the matrix size.
Reduced QR and reusable factors reduce unnecessary work; they do not make a
general dense problem sparse. General sparse direct LU with symbolic fill
analysis is outside the current method set. Matrix-free solvers also require
appropriate operator and preconditioner behavior.

Normal equations can square a least-squares design matrix's condition number.
Classical Gram–Schmidt can lose orthogonality on nearly dependent columns.
For sensitive problems, compare QR or SVD approaches and inspect rank and
residuals. A successful solve cannot recover information lost in noisy inputs.

See [linear algebra](guides/linalg.md) for the exact dispatcher choices,
complex factor interfaces, and sparse assumptions.

## Differentiation, interpolation, and approximation

A forward difference often balances `O(h)` truncation against `O(epsilon/h)`
roundoff; a central difference often balances `O(h**2)` against the same
roundoff term. Under those assumptions, useful step scales are roughly
`sqrt(epsilon)` and `epsilon**(1/3)` after accounting for the function's scale.
The central difference's best attainable error scales roughly as
`epsilon**(2/3)`, not `epsilon**(1/3)`. Noise can dominate both estimates.

Automatic differentiation differentiates supported executed operations in
floating-point arithmetic. It is not arbitrary-precision differentiation and
does not automatically handle every array operation, external function,
branching behavior, or nondifferentiable point. Complex-step differentiation
requires an analytic extension and code that preserves the imaginary part.

Runge-type oscillation is a risk for high-degree interpolation at equally
spaced nodes, not a theorem that every such interpolant diverges. Chebyshev
nodes, local splines, or shape-preserving interpolants offer different tradeoffs.
Adaptive approximation uses sampled error checks and can miss narrow,
unsampled features. Validate on additional points.

## Quadrature

An adaptive rule can miss a narrow peak or discontinuity if its sample points
never encounter it. Split at known difficult locations and compare refinements.
Endpoint singularities, infinite intervals, oscillations, and principal values
need appropriate transformations or specialized rules.

A method may stop because its subdivision budget is exhausted. Some legacy
routines have less informative status reporting; the [integration guide](guides/integrate.md)
documents specific cases. An error estimate is not a proof of a global bound.
Monte Carlo errors have sampling uncertainty; one seed is not a convergence study.

## Differential equations

Adaptive ODE tolerances control a local error estimate. Global error, event
location, and interpolated output have separate accuracy considerations. A
stiff method's implicit solve introduces another convergence problem. Fixed-step
methods still need step refinement and a stability check.

Output controls reduce stored data; they do not guarantee that arbitrary later
interpolation remains accurate. A numerical checkpoint stores solver state,
not model code or the entire execution environment. Compatibility and exact
restart behavior are method-specific.

PDE methods require the documented boundary conditions and grid layout.
Explicit diffusion and advection have stability restrictions. An implicit
method can be stable while severely underresolving the solution. High-order
linear advection schemes can oscillate near jumps; limiters and WENO address
that tradeoff under their own assumptions.

See [ODEs](guides/ode.md), [PDEs](guides/pde.md), and
[scientific workflows](guides/workflows.md) before combining time, mesh,
nonlinear, and output tolerances.

## Optimization and statistical inference

A local optimizer can converge to a local minimum or stall in a flat region.
A population's collapse does not certify a global optimum. Nelder–Mead can
stagnate or converge to a nonstationary point on some problems. Verify a
stationarity measure when available, and examine multiple starts where needed.

Parameter scaling affects finite differences, line searches, and trust regions.
A small residual norm does not establish parameter identifiability. Covariance
estimates rely on assumptions about the local model and observation errors;
robust losses do not automatically supply a valid ordinary least-squares
covariance. The robust `curve_fit` path currently returns no covariance or
standard-error estimate.

Monte Carlo's `N**(-1/2)` standard-error rate assumes the relevant variance is
finite and sampling assumptions hold. Correlated samples reduce effective
information. Rank-based MCMC diagnostics help detect problems but do not prove
convergence or discovery of every mode.

## Stochastic equations and special functions

Euler–Maruyama and Milstein solve Itô SDEs under their documented noise
assumptions. Stochastic Heun uses the Stratonovich interpretation. With
multiplicative noise, these interpretations generally describe different
processes unless the drift is corrected. Strong convergence compares paths
under coupled Brownian increments; independent paths do not measure it.

Sobol directions in higher dimensions are generated rather than drawn from a
full optimized external direction table. Randomized QMC uncertainty comes from
independent scrambles, not from treating points in one scramble as independent.

Special functions have poles, overflow regions, and domain restrictions. Bessel
orders in the ordinary integer-order routines are integers. Near a zero, use
absolute error; near a pole, relative conditioning can deteriorate. Scaled and
tail functions can avoid cancellation, but their domains must still be checked.

## Runtime and concurrency

Performance depends on method, size, dtype, layout, callbacks, and retained
output. Native kernels use one computational thread per call; selected long
callback-free loops release the GIL. This does not mean every routine executes
concurrently or that sharing mutable storage is safe. Independent calls should
not mutate the same array memory.

The Python reference mode still requires the C array engine. For comparisons,
report the backend and workload rather than infer performance from the
language used in one layer.

## Report unexpected behavior

An incorrect value, wrong order, crash, or misleading status outside a stated
limitation is useful to report. Provide a minimal example, version or commit,
backend, input scales, method options, actual result, and independent expected
behavior. The [contributing guide](contributing.md) explains how to turn that
case into a regression.
