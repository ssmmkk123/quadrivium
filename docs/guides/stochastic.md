# Stochastic methods and statistical computation

Randomized computation has two separate accuracy questions: whether the
algorithm approximates the intended model, and whether enough random samples
have been collected. A reproducible seed helps repeat an experiment, but does
not answer either question. Quadrivium provides random streams, distribution
objects, Monte Carlo tools, MCMC diagnostics, and stochastic differential
equation solvers to make those checks explicit.

See the [stochastic API](../api/stochastic.md) for complete signatures and
[Monte Carlo integration](integrate.md) for quadrature-specific methods.

## Create reproducible random streams

Most randomized interfaces accept `rng=` as an integer seed, a
`quadrivium.numeric.random.Generator`, or `None`. Passing the same integer to
two calls starts two copies of the same stream. Passing a generator continues
its state across calls.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import stochastic as st
>>> first = st.box_muller(n=8, rng=17)
>>> second = st.box_muller(n=8, rng=17)
>>> np.array_equal(first, second)
True
>>> generator = np.random.default_rng(17)
>>> saved = generator.state
>>> draws = generator.standard_normal(5)
>>> generator.state = saved
>>> np.array_equal(draws, generator.standard_normal(5))
True

```

Save generator state when resuming a random calculation, together with the
algorithm options and package version. A seed alone does not encode how many
values have already been consumed. Reproducibility also depends on the order
and shape of random calls and on any changes in the implementation.

For separate experiments or workers, `spawn_rngs` derives deterministic child
streams by index. Assign one child per task so results do not depend on the
order in which workers finish.

```pycon
>>> workers = st.spawn_rngs(42, 3)
>>> values = workers[2].random(4)
>>> np.array_equal(values, st.spawn_rngs(42, 3)[2].random(4))
True

```

The classical `LCG`, `ParkMiller`, `XorShift`, `MersenneTwister`, and
`middle_square` interfaces expose generator algorithms for comparison.
`spectral_test` can expose short lattice relations in successive outputs.
Use the standard `rng=` workflow for ordinary simulations instead of choosing
a historical generator merely because its one-dimensional histogram looks
uniform. None of these simulation interfaces is a cryptographic API.

## Use distribution objects for probabilities and tails

`Normal`, `Uniform`, `Exponential`, `Gamma`, `Beta`, `StudentT`, `ChiSquare`,
`Poisson`, and `Binomial` have scalar parameters and support array observations.
The parameter conventions are explicit: for example `Exponential(scale=...)`
uses a scale, and `Gamma(shape, scale=...)` uses shape and scale.

| Method | Meaning |
| --- | --- |
| `pdf`, `logpdf` | Density and log density; discrete classes also expose these conventions |
| `pmf`, `logpmf` | Probability mass and log mass for discrete distributions |
| `cdf(x)` | Probability at or below x |
| `sf(x)` | Upper-tail probability directly |
| `logcdf`, `logsf` | Log probabilities without first forming a tiny probability |
| `ppf(q)` | Lower-tail quantile |
| `isf(q)` | Inverse survival function |
| `rvs(size=..., rng=...)` | Random samples |

```pycon
>>> normal = st.Normal(loc=0.0, scale=1.0)
>>> round(normal.ppf(0.975), 6)
1.959964
>>> np.allclose(normal.cdf([-1.0, 0.0, 1.0]) + normal.sf([-1.0, 0.0, 1.0]), 1.0)
True
>>> normal.sf(40.0) == 0.0, normal.logsf(40.0) < -800.0
(True, True)
>>> np.array_equal(normal.rvs(size=5, rng=2), normal.rvs(size=5, rng=2))
True

```

Compute a small upper tail with `sf`, rather than subtracting a CDF close to
one from one. Use `logsf` when the probability itself is too small for float64.
The logarithm can remain representable even when `sf` underflows to zero.
`isf` similarly avoids first computing `1-q` for a tiny upper-tail probability.

A discrete quantile is the first supported integer whose CDF reaches the
requested probability. A continuous quantile round trip is checked by equality
to numerical tolerance; a discrete one is checked by inequalities.

```pycon
>>> poisson = st.Poisson(mu=3.0)
>>> k = poisson.ppf(0.8)
>>> poisson.cdf(k - 1) < 0.8 <= poisson.cdf(k)
True
>>> round(float(st.Binomial(5, 0.5).pmf(2)), 6)
0.3125

```

Parameter and quantile validation rejects invalid values. Distribution support
still matters: a density outside support can correctly be zero and its log
density negative infinity. Those values are not automatically numerical errors.

## Choose a sampling algorithm

| Available information | Interface | Main requirement |
| --- | --- | --- |
| Inverse CDF | `inverse_transform` | Correct inverse over the unit interval |
| Normal variates | `box_muller`, `marsaglia_polar` | Valid location and scale |
| Target and proposal densities | `rejection_sampling` | Envelope `M*proposal_pdf >= target_pdf` |
| Log-concave target | `adaptive_rejection` | Log concavity and a useful initial support |
| Discrete weights | `sample_discrete`, `alias_table` | Valid probabilities |
| Mean and covariance | `multivariate_normal` | Compatible shapes and suitable covariance |
| Unnormalized density | MCMC methods | Exploration and mixing diagnostics |

A rejection sampler's envelope is a mathematical condition, not a speed knob.
An underestimated `M` can invalidate the resulting distribution. Increasing it
usually lowers acceptance. Adaptive rejection builds an envelope using log
concavity; use a different method for a multimodal or non-log-concave target.

```pycon
>>> sample = st.box_muller(n=5000, rng=3)
>>> summary = st.describe(sample)
>>> abs(summary["mean"]) < 0.08, abs(summary["std"] - 1.0) < 0.08
(True, True)

```

A rough moment check is useful during development. It cannot establish that a
sampler handles tails, dependence, or a multivariate geometry correctly.
Check known distributional identities and repeated seeded experiments too.

## Calibrate Monte Carlo error

For independent observations with finite variance, the standard error of a
sample mean is estimated by `sample_std/sqrt(n)`. It measures sampling
variability, not discretization error or model bias. Roughly four times the
independent samples are needed to halve that error.

```pycon
>>> uniform = np.random.default_rng(9).random(4000)
>>> values = np.exp(uniform)
>>> estimate = float(np.mean(values))
>>> standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
>>> abs(estimate - float(np.e - 1)) < 5 * standard_error
True

```

The broad bound in this example checks one seeded calculation. An error
estimator is better assessed across repeated independent runs: compare its
reported scale with the observed spread of estimates around an exact answer.
A single unusually accurate run is weak evidence about estimator quality.

<figure markdown="span">
  ![Observed Monte Carlo integration error compared with reported standard error](../assets/figures/stochastic-error-calibration.svg#only-light)
  ![Observed Monte Carlo integration error compared with reported standard error](../assets/figures/stochastic-error-calibration-dark.svg#only-dark)
  <figcaption>Independent seeded estimates of the integral of exp(x) on the unit interval provide an observable error distribution. The graph compares empirical root-mean-square error with the reported standard-error scale as the sample count grows. Agreement assesses calibration for this experiment; it is not a deterministic bound on every run.</figcaption>
</figure>

## Stateful Sobol and randomized QMC

Low-discrepancy sequences cover a box more evenly than independent random
points in many useful settings. Their benefit depends on the integrand's
smoothness, dimension, and effective structure. Deterministic net points are
not independent random observations, so their pointwise sample variance is
not an ordinary Monte Carlo uncertainty estimate.

```pycon
>>> engine = st.Sobol(2, scramble=True, seed=8)
>>> points = engine.random_base2(5)
>>> points.shape, engine.num_generated
((32, 2), 32)
>>> state = engine.state
>>> continued = st.Sobol.from_state(state)
>>> np.array_equal(engine.random(4), continued.random(4))
True

```

`random_base2(m)` adds `2**m` points and checks that the **cumulative** count
remains a power of two. `random(n)` allows arbitrary counts, giving up that
balance condition. The engine supports reset, fast-forward, JSON-compatible
state, scrambling, and child engines; dimensions range from 1 to 1024.
The higher-dimensional directions use generated primitive-polynomial tables,
not optimized Joe–Kuo direction tables. Do not assume identical points or
performance to a different Sobol implementation.

`randomized_qmc` averages independent scrambled replicates and reports the
standard error across their estimates. Each replicate contains `2**m` points;
`replicates` must be at least two. `batch_size` bounds point storage without
changing the intended integral.

```pycon
>>> integral = st.randomized_qmc(lambda x: x[0] * x[1], [0, 0], [1, 1],
...                              m=7, replicates=4, seed=5, batch_size=32)
>>> abs(float(integral.value) - 0.25) < 0.01
True
>>> integral.function_calls
512

```

The callback receives a scalar in one dimension and a point vector in higher
dimensions. `error_estimate` is a sampling estimate across replicates; it is
not a certified quadrature bound. Increase both resolution and replication
when validating a new problem.

## Markov chain Monte Carlo

MCMC uses dependent draws to explore a distribution available through a log
density. A normalizing constant is unnecessary because it cancels in the
acceptance ratio.

```pycon
>>> target = lambda x: -0.5 * float(x @ x)
>>> chain = st.random_walk_metropolis(target, [0.0], step=1.5,
...                                   n=1000, burn=200, thin=2, rng=4)
>>> chain.shape
(500, 1)
>>> 0.0 < float(chain.acceptance) < 1.0
True

```

`Chain` is an array subclass with diagnostic attributes. Its axes are
`(retained_draws, parameters)`. Here `n` counts post-burn iterations; `thin=2`
keeps every other one, so it returns 500 draws. Thinning reduces storage, but
does not create more information from a fixed number of transitions.

For custom Metropolis-Hastings proposals, `proposal(x, rng)` returns a
candidate. `log_proposal_ratio(x, y)` is
`log(q(x|y)) - log(q(y|x))`; omit it only for a symmetric proposal.
`gibbs_sampler` takes conditional sampling callbacks. `hamiltonian_mc` needs a
gradient of the **log target**, together with step size and leapfrog count.
`nuts_lite` randomizes HMC trajectory length; it does not implement the full
recursive no-U-turn algorithm. `slice_sampler` has a width and step-out budget,
so it still has numerical settings to assess. Parallel tempering can help
explore separated modes, but requires checking exchange and exploration.

### Diagnose multiple chains

Use chains with different initial states and independent streams.
`mcmc_diagnostics` expects `(chains, draws, ...)`, with at least two chains
and four draws per chain; remaining axes represent parameter dimensions.
It returns rank-normalized split R-hat, bulk ESS, tail ESS, and mean MCSE.

```pycon
>>> independent = np.random.default_rng(0).standard_normal((4, 300, 2))
>>> diagnostics = st.mcmc_diagnostics(independent)
>>> sorted(diagnostics)
['ess_bulk', 'ess_tail', 'mcse_mean', 'rhat']
>>> diagnostics["rhat"].shape
(2,)
>>> bool(np.all(diagnostics["rhat"] < 1.1))
True

```

This example checks the shape contract using independent normal observations;
it is not a substitute for assessing actual sampler chains. R-hat near one
checks agreement among split chains. ESS estimates the information loss due
to dependence, and tail ESS focuses on quantile regions that can mix worse
than the center. Acceptance rate alone cannot prove adequate exploration.

Constant or degenerate chains can produce undefined diagnostics. The routines
report NaN or infinity where appropriate rather than treating zero empirical
spread as perfect certainty. Inspect traces, cross-chain agreement, and
sensitivity to warm-up and tuning. A warm-up length is an experimental choice,
not a universal fixed fraction that guarantees stationarity.

## Statistics and resampling

`describe` reports a descriptive summary. `variance` defaults to `ddof=1`,
whereas array-level `np.var` defaults to population normalization. Use matching
normalizations when comparing implementations.

```pycon
>>> data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
>>> summary = st.describe(data)
>>> summary["mean"], summary["median"], round(summary["variance"], 6)
(5.0, 4.5, 4.571429)
>>> location, variance = st.welford_mean_var(1e9 + np.arange(4.0))
>>> round(float(variance), 6)
1.666667

```

Welford's update avoids cancellation in `mean(x*x)-mean(x)**2` when values
have a large offset and small spread. `covariance_matrix` treats rows as
observations and columns as variables. Regression and PCA build on these
conventions; center or scale variables according to their units and the
interpretation needed from the model.

`bootstrap` resamples observations with replacement and returns a dictionary
including the percentile interval under `"ci"`. `jackknife` leaves out each
observation in turn. These procedures inherit assumptions about the sampling
unit; independent row resampling is not automatically appropriate for a time
series or clustered measurements.

```pycon
>>> boot = st.bootstrap(np.arange(20.0), np.mean, n_resamples=500, rng=0)
>>> lower, upper = boot["ci"]
>>> lower < 9.5 < upper
True

```

Hypothesis-test functions return numerical summaries, not a decision about
scientific importance. Check the test's assumptions, sample construction, and
which null distribution is being used before interpreting a p-value.

## Stochastic differential equations

The SDE interfaces use drift and diffusion callbacks of **`(x, t)`**, whereas
ordinary ODE callbacks use `(t, y)`. They approximate
`dX = a(X,t)*dt + b(X,t)*dW` on a uniform increasing time grid.
The returned `ODESolution` stores `(n+1, state_dimension)` under full retention.

```pycon
>>> drift = lambda x, t: 0.2 * x
>>> diffusion = lambda x, t: 0.3 * x
>>> trajectory = st.milstein(drift, diffusion, (0, 1), [1.0],
...                           n=200, rng=6, db=lambda x, t: np.full_like(x, 0.3))
>>> trajectory.y.shape
(201, 1)
>>> bool(np.all(np.isfinite(trajectory.y)))
True

```

| Method | Main interpretation and limitation |
| --- | --- |
| Euler-Maruyama | Itô; typical strong order 1/2 and weak order 1 under regularity assumptions |
| Milstein | Itô correction; scalar/diagonal noise form, not general Lévy-area simulation |
| Implicit Milstein | Drift-implicit update; assess its inner iteration accuracy |
| Stochastic Heun | Stratonovich interpretation |
| Stochastic RK | Derivative-free order-one scheme in the documented noise setting |
| `srk_strong_1_5` | Higher-order method restricted to additive noise |
| Tamed Euler | Modified drift for superlinear-growth problems |

The implemented diffusion multiplication is componentwise. A full matrix-valued
noise model needs a formulation supported by the chosen method; it is not
silently handled by passing an arbitrary diffusion matrix.
For multiplicative noise, Itô and Stratonovich equations differ by a drift
correction. Switching methods without changing the model interpretation can
change the equation being solved.

Strong error compares trajectories driven by the **same** Brownian path.
Generate fine increments and sum adjacent increments for a coarser grid;
reusing a seed with a different `n` does not establish that coupling by itself.
Methods accepting `dW` allow explicit control, with increment shape
`(n, state_dimension)`. Weak error compares expectations and needs enough
independent paths to separate sampling noise from time-discretization bias.

## Exact transitions, jumps, and retained output

`geometric_brownian_motion` and `ornstein_uhlenbeck` provide exact transition
sampling with `exact=True`. Exact transitions remove grid discretization error
at sampled times, but do not produce the entire continuous path between them.
`cox_ingersoll_ross` uses an approximate positivity-handling discretization;
it has no `exact=True` option. General Milstein or Euler updates do not
universally preserve positivity merely because the continuous model does.

`gillespie_ssa` advances reaction events from propensities and stoichiometry;
`tau_leaping` approximates several firings per step. Check stoichiometric axes,
nonnegative population behavior, and step refinement for a reaction model.
Tau-leaping keeps its fixed step grid, which can finish past the requested
endpoint. `poisson_process` returns event times, not a regularly sampled count
trajectory.

SDE and reaction solvers support `save_at`, `save_every`, `final_only`, and
callbacks. Diffusion `save_at` values use interpolation of simulated states;
they do not draw a conditional Brownian bridge. Jump outputs use
right-continuous sampling to preserve discrete populations. `y_final` records
the actual final state even when selected output omits it. SDE restart
checkpoints are not supported; Brownian/Poisson path utilities keep their full
output. See [scientific workflows](workflows.md) for shared output contracts.
