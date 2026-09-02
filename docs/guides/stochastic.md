# Stochastic methods

```python
from quadrivium.stochastic import milstein, hamiltonian_mc, bootstrap
import quadrivium as qd          # qd.metropolis_hastings, qd.euler_maruyama, ...
```

75 routines: pseudorandom and low-discrepancy generators, sampling algorithms,
MCMC with diagnostics, descriptive and inferential statistics, and SDE
integrators. Full signatures are in the
[`stochastic` reference](../api/stochastic.md).

Every routine that consumes randomness takes `rng=`, which is passed to
`np.random.default_rng`: an integer seed, a `Generator`, or `None` for fresh
entropy. Seeding makes a run exactly reproducible.

## Generators

The classical generators are here to be studied, with their defects intact.
RANDU-style linear congruential generators fail the spectral test — their
triples lie on a small number of planes, which is invisible in one dimension
and fatal in three:

```pycon
>>> import numpy as np
>>> from quadrivium.stochastic import LCG, MersenneTwister, spectral_test
>>> lcg = LCG(seed=1)
>>> round(float(lcg.random()), 12)
0.513870078139

```

`ParkMiller`, `XorShift`, and `MersenneTwister` (a full MT19937) complete the
set, alongside `middle_square` — von Neumann's method, included because
watching it collapse into a short cycle is the clearest argument for the
theory that followed it.

<figure markdown="span">
  ![RANDU's triples lie on a lattice; a modern generator's do not](../assets/figures/stochastic-spectral-test.svg#only-light)
  ![RANDU's triples lie on a lattice; a modern generator's do not](../assets/figures/stochastic-spectral-test-dark.svg#only-dark)
  <figcaption>Successive triples from each generator, restricted to a thin slab of the unit cube so the structure is visible edge on. `spectral_test` finds the lattice on its own and reports the coefficients (9, −6, 1) — RANDU satisfies x[i+2] = 6x[i+1] − 9x[i] exactly.</figcaption>
</figure>

For quasi-random points, `halton`, `sobol`, `van_der_corput`, and
`latin_hypercube_sample` produce low-discrepancy sequences that cover a space
more evenly than random points, which is what makes quasi-Monte Carlo
converge faster.

<figure markdown="span">
  ![Pseudorandom points against Halton and Sobol points](../assets/figures/stochastic-low-discrepancy.svg#only-light)
  ![Pseudorandom points against Halton and Sobol points](../assets/figures/stochastic-low-discrepancy-dark.svg#only-dark)
  <figcaption>Random points clump, and a clump is a region the integrand is not sampled in. Low-discrepancy sequences fill the square by construction, which is what makes quasi-Monte Carlo converge faster than n<sup>−1/2</sup> on a smooth integrand.</figcaption>
</figure>

**Use these for study, not as your source of randomness.** For real work pass
`rng=` and let NumPy's PCG64 generate.

## Sampling from a distribution

| Method | Function | Needs |
| --- | --- | --- |
| inverse transform | `inverse_transform` | the inverse CDF |
| Box-Muller / polar | `box_muller`, `marsaglia_polar` | nothing (normals) |
| rejection | `rejection_sampling` | a proposal and a bound `M` |
| adaptive rejection | `adaptive_rejection` | a log-concave log-density |
| ratio of uniforms | `ratio_of_uniforms` | a bounded density |
| alias table | `alias_table`, `sample_discrete` | discrete probabilities; `O(1)` per draw |
| multivariate normal | `multivariate_normal` | mean and covariance |

```pycon
>>> from quadrivium.stochastic import box_muller, describe
>>> z = box_muller(n=20_000, rng=0)
>>> stats = describe(z)
>>> abs(stats["mean"]) < 0.05 and abs(stats["std"] - 1.0) < 0.05
True

```

Adaptive rejection sampling (Gilks and Wild) builds its own envelope from the
concavity of the log-density and tightens it as it goes, so it needs no tuning
constant — the reason it is the standard choice inside Gibbs samplers.

## MCMC

When you can evaluate a density only up to a constant, sample it with a Markov
chain:

```pycon
>>> from quadrivium.stochastic import random_walk_metropolis, acceptance_rate
>>> log_target = lambda x: -0.5 * float(x[0]**2)          # standard normal
>>> chain = random_walk_metropolis(log_target, [0.0], step=2.0, n=20_000,
...                                burn=2000, rng=0)
>>> samples = chain[:, 0]              # Chain is an ndarray subclass
>>> abs(float(np.mean(samples))) < 0.1, abs(float(np.std(samples)) - 1.0) < 0.1
(True, True)
>>> 0.2 < acceptance_rate(chain) < 0.7
True

The draws *are* the returned object — `Chain` subclasses `numpy.ndarray` and
carries the sampler's diagnostics as attributes, so it indexes, slices, and
plots like any array:

>>> chain.shape, round(float(chain.acceptance), 3)
((20000, 1), 0.499)

```

| Sampler | Function | Suited to |
| --- | --- | --- |
| Metropolis-Hastings | `metropolis_hastings` | any proposal you supply |
| random-walk Metropolis | `random_walk_metropolis` | the default starting point |
| Gibbs | `gibbs_sampler` | conditionals available in closed form |
| Hamiltonian Monte Carlo | `hamiltonian_mc` | gradients available, high dimension |
| NUTS-lite | `nuts_lite` | HMC without choosing a path length |
| slice sampling | `slice_sampler` | no tuning at all |
| parallel tempering | `parallel_tempering` | multimodal targets |

HMC uses the gradient to propose distant states that are still accepted, which
is why it beats a random walk badly as dimension grows:

```pycon
>>> from quadrivium.stochastic import hamiltonian_mc, effective_sample_size
>>> grad = lambda x: -x
>>> hmc = hamiltonian_mc(lambda x: -0.5*float(x @ x), grad, np.zeros(1),
...                      step=0.3, n_leapfrog=10, n=4000, burn=500, rng=0)
>>> rw = random_walk_metropolis(lambda x: -0.5*float(x @ x), np.zeros(1),
...                             step=1.0, n=4000, burn=500, rng=0)
>>> float(effective_sample_size(hmc)[0]) > float(effective_sample_size(rw)[0])
True

```

**A chain must be diagnosed, not trusted.** `effective_sample_size` says how
many independent draws the chain is worth, `autocorrelation_time` how long it
takes to forget where it was, `gelman_rubin` compares several chains for
agreement (`R̂` near 1), and `acceptance_rate` catches a proposal that is far
too wide or too narrow.

<figure markdown="span">
  ![Random-walk Metropolis against Hamiltonian Monte Carlo](../assets/figures/stochastic-mcmc-diagnostics.svg#only-light)
  ![Random-walk Metropolis against Hamiltonian Monte Carlo](../assets/figures/stochastic-mcmc-diagnostics-dark.svg#only-dark)
  <figcaption>The same ten-dimensional target and the same number of draws. The random walk explores in steps that undo each other, so six thousand draws are worth two hundred independent ones; HMC uses the gradient to propose distant states that are still accepted.</figcaption>
</figure>

## Statistics

```pycon
>>> from quadrivium.stochastic import describe, welford_mean_var
>>> data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
>>> stats = describe(data)
>>> stats["mean"], stats["median"], round(stats["variance"], 6)
(5.0, 4.5, 4.571429)

```

`welford_mean_var` uses Welford's online algorithm rather than
`E[x²] − E[x]²`, which loses all precision when the mean is large compared to
the spread:

```pycon
>>> big = 1e9 + np.array([1.0, 2.0, 3.0, 4.0])
>>> mean_w, var_w = welford_mean_var(big)
>>> abs(float(var_w) - float(np.var(big, ddof=1))) < 1e-9        # exact
True
>>> naive = float(np.mean(big**2) - np.mean(big)**2)             # population form
>>> abs(naive - 1.25) > 1e-3                                     # and wrong
True

```

Regression and multivariate: `linear_regression`, `polynomial_regression`,
`logistic_regression` (Newton-IRLS), `pca`, `covariance_matrix`,
`correlation_matrix`, `kernel_density`, `histogram_density`.

Resampling and testing: `bootstrap` (percentile confidence intervals and a
bias estimate), `jackknife`, `permutation_test`, `t_test`, `chi_square_test`,
`ks_test`, `anova_one_way`, `confidence_interval`.

<figure markdown="span">
  ![The bootstrap distribution of a median, and the interval it gives](../assets/figures/stochastic-bootstrap.svg#only-light)
  ![The bootstrap distribution of a median, and the interval it gives](../assets/figures/stochastic-bootstrap-dark.svg#only-dark)
  <figcaption>Resampling the data with replacement four thousand times gives the sampling distribution of any statistic — here a median, which has no convenient closed form — and the 2.5% and 97.5% quantiles of that distribution are the interval.</figcaption>
</figure>

```pycon
>>> from quadrivium.stochastic import bootstrap
>>> boot = bootstrap(np.arange(20.0), np.mean, n_resamples=2000, rng=0)
>>> lo, hi = boot["ci"]
>>> lo < 9.5 < hi                                 # the sample mean is inside
True

```

## Stochastic differential equations

`dX = a(X, t) dt + b(X, t) dW`. The drift and diffusion are callables of
`(x, t)`; the result is an `ODESolution` like any other trajectory:

```pycon
>>> from quadrivium.stochastic import euler_maruyama, milstein
>>> drift = lambda x, t: 1.5 * x
>>> diffusion = lambda x, t: 0.4 * x
>>> path = milstein(drift, diffusion, (0, 1), [1.0], n=1000, rng=0)
>>> path.y.shape, bool(np.all(path.y > 0))        # GBM stays positive
((1001, 1), True)

```

| Solver | Strong order | Note |
| --- | --- | --- |
| `euler_maruyama` | 0.5 | the direct analogue of forward Euler |
| `milstein` | 1.0 | adds the `b b′` correction term |
| `implicit_milstein` | 1.0 | stable for stiff drift |
| `stochastic_heun` | 1.0 | converges to the **Stratonovich** solution |
| `stochastic_rk` | 1.0 | derivative-free Milstein |

| `srk_strong_1_5` | 1.5 | additive noise only |
| `tamed_euler` | 0.5 | for superlinearly growing drift, where Euler diverges |

<figure markdown="span">
  ![Geometric Brownian motion: paths, and the distribution they sample](../assets/figures/stochastic-sde-paths.svg#only-light)
  ![Geometric Brownian motion: paths, and the distribution they sample](../assets/figures/stochastic-sde-paths-dark.svg#only-dark)
  <figcaption>Each path is one realisation; the average over paths follows the deterministic exponential. The right panel is the answer to the question an SDE solver is usually asked — the law of the solution at a fixed time — against the exact lognormal density.</figcaption>
</figure>

Strong order 1/2 for Euler-Maruyama is not a weakness of the implementation:
the Itô-Taylor expansion has a `b b′(ΔW² − Δt)/2` term that the method omits.
Milstein keeps it. `strong_error` and `weak_error` measure both orders
empirically against an exact solution.

<figure markdown="span">
  ![Measured strong convergence of Euler-Maruyama and Milstein](../assets/figures/stochastic-strong-order.svg#only-light)
  ![Measured strong convergence of Euler-Maruyama and Milstein](../assets/figures/stochastic-strong-order-dark.svg#only-dark)
  <figcaption>Both solvers are driven by the same Brownian increments as the exact solution, which is what makes this a pathwise (strong) comparison rather than a comparison of distributions. The measured slopes bracket the theoretical 1/2 and 1 within the Monte Carlo noise of 600 paths.</figcaption>
</figure>

`brownian_path` and `brownian_bridge` generate the driving noise;
`geometric_brownian_motion`, `ornstein_uhlenbeck`, and `cox_ingersoll_ross`
have exact samplers (`exact=True`) that step at any size without
discretization error. For jump processes, `poisson_process`, `gillespie_ssa`
(exact stochastic simulation of a reaction network), and `tau_leaping` (its
approximate, faster cousin).

## Pitfalls

- **Stochastic Heun solves the Stratonovich equation, not the Itô one.** For
  multiplicative noise they have different solutions. This is a property of
  the scheme, not an error.
- **A chain that has converged in appearance may not have.** Check `R̂` across
  several chains started far apart, not just one trace plot.
- **Burn-in is not optional, and its length is not knowable in advance.**
- **`euler_maruyama` diverges for superlinear drift.** Use `tamed_euler`.
- **Monte Carlo error falls as `1/√n`.** Four times the samples for two times
  the accuracy — see the [integration guide](integrate.md).
- **The historical generators are not safe for simulation.** Use `rng=`.

## See also

- [`stochastic` API reference](../api/stochastic.md) — every signature.
- [Integration guide](integrate.md) — Monte Carlo quadrature and variance
  reduction.
- [ODE guide](ode.md) — the deterministic integrators these extend.
- `examples/06_extended_methods.py` — SDEs, wavelets, matrix equations, WENO.
