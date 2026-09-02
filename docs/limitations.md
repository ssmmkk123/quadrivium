# Known limitations

Everything here is a real property of a method or of this implementation,
documented rather than papered over. None of it is a bug; where a limitation
could be removed, the alternative is named.

## Library-wide

**Speed.** The algorithms are written to be read, which means Python-level
loops wherever the method is a loop. Dense factorizations are competitive up
to a few hundred rows; ODE and PDE time stepping runs roughly 10–100× slower
than a compiled integrator. See [Performance](getting-started.md#performance).
For large sparse solves or inner loops that run millions of times, use a
compiled library.

**Precision.** Everything is double precision. There is no extended-precision
or interval arithmetic, so no method here can return a certified error bound —
only an estimate.

**Threads.** Nothing is parallelized. NumPy's own BLAS may use threads for the
matrix products underneath, but no algorithm here is threaded or vectorized
across independent problems.

**Sparse support is basic.** COO, CSR, CSC, and DIA storage with matrix-vector
products, reverse Cuthill-McKee reordering, and the Krylov solvers. There is
no sparse direct factorization, no fill-reducing ordering beyond RCM, and no
graph partitioning.

## Specific methods

**`cauchy_point` trust region converges linearly.** It minimises the model
along the steepest-descent direction only, ignoring the Newton direction
entirely. It is enough for global convergence and nothing more; use `dogleg`
or `steihaug_cg` for a superlinear rate.

**`stochastic_heun` solves the Stratonovich equation.** It converges to the
Stratonovich solution, which differs from the Itô one by the drift correction
`½ b b′`. For multiplicative noise it and `euler_maruyama` are solving
genuinely different equations. This is a property of the scheme, not a defect;
use `milstein` for the Itô solution at strong order 1.

**`stormer_cowell` is third order in its familiar form.** The widely quoted
three-step coefficients `(13, −2, 1)/12` leave an `h³` term in the Taylor
expansion. The implementation uses the four-step coefficients
`(14, −5, 4, −1)/12`, chosen so that term cancels. It also reports velocity as
a difference estimate rather than an integrated quantity, because it acts on
positions only.

**Unpreconditioned `newton_krylov` scales with the mesh.** The number of
Krylov iterations grows as a PDE mesh is refined — a property of the operator,
not of the implementation. That is what `precond=` is for: on the `n = 1000`
Bratu problem it takes the cost from 102,101 residual evaluations to 21.

**`euler_maruyama` has strong order 1/2, not 1.** The Itô-Taylor expansion
contains a `b b′(ΔW² − Δt)/2` term the method omits. `milstein` keeps it.

**Bessel functions take integer orders only.** Half-integer orders are the
spherical Bessel functions, which have their own routines
(`spherical_bessel_j`, `spherical_bessel_y`). There is no arbitrary real
order.

**`zeta` loses accuracy near `s = 1`.** That is a pole; values approaching it
degrade, and `zeta(1.0)` is undefined.

**Classical Gram-Schmidt is here to be compared against, not used.** It loses
orthogonality on nearly dependent columns, as it should. Use `householder_qr`.

**`normal_equations` squares the condition number.** It is provided because
it is the formula everyone learns; `qr_least_squares` is the one to use.

**Newton-Cotes rules above degree 8 have negative weights.** The
implementation will build them, and they will lose digits to cancellation. Use
a composite low-order rule or a Gauss rule.

**Global optimizers have no reliable stopping criterion.** `tol` detects that
the population has collapsed, which is not the same as having found the global
optimum. `max_iter` is the real budget.

**`nelder_mead` has no convergence theory above one dimension.** It can
converge to a non-stationary point. Verify with a gradient check when one is
available.

## Numerical facts that look like limitations

These are properties of the mathematics, not of the code:

- **Finite differences cannot beat about `√ε` accuracy** for a first
  derivative, `∛ε` for a central one. Use automatic differentiation
  ([`quadrivium.diff`](guides/diff.md)) for exact derivatives.
- **Monte Carlo converges as `n^{−1/2}`** in every dimension. One more digit
  costs a hundred times the work.
- **Second-order schemes oscillate at discontinuities.** Godunov's theorem: no
  linear scheme above first order can be monotone. Use a limiter or WENO.
- **A truncated Fourier series overshoots a jump by about 9%** however many
  terms it has. That is Gibbs' phenomenon.
- **Polynomial interpolation on equally spaced points diverges** as the degree
  grows (Runge). Use Chebyshev nodes or a spline.
- **Explicit time stepping has a stability limit.** Exceeding it diverges;
  it does not merely lose accuracy.

<figure markdown="span">
  ![The best each technique can do, measured](assets/figures/limitations-accuracy-floors.svg#only-light)
  ![The best each technique can do, measured](assets/figures/limitations-accuracy-floors-dark.svg#only-dark)
  <figcaption>None of these floors is an implementation defect. A forward difference cannot beat √ε whatever step it takes; Monte Carlo buys a digit for a hundred times the samples; automatic differentiation has no floor of its own because it never subtracts nearby numbers.</figcaption>
</figure>

## Reporting something else

If a method is wrong rather than limited — the wrong answer, the wrong
convergence order, a failure that is not reported — that is a bug. Please
[open an issue](https://github.com/ssmmkk123/quadrivium/issues) with the
smallest case that shows it, and what the answer should be.
