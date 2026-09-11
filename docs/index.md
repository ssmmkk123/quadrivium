# Quadrivium documentation

Quadrivium implements numerical methods for scientific computing, from a
bracketed root to a parameter-fitting workflow driven by differential equations.
Its Python source exposes the methods, and its bundled C array engine and
numerical kernels handle native computation. No runtime package dependencies
are required.

These pages describe the **source checkout used to build this site**. The
[changelog](changelog.md) separates unreleased work from published releases.
If an example uses an API absent from your installed release, use the
[source installation](installation.md#from-source) or that release's source
and documentation together.

## Start with a calculation

```pycon
>>> import quadrivium as qd
>>> result = qd.brent(lambda x: x*x - 2, 0, 2)
>>> round(result.root, 8), result.converged
(1.41421356, True)
>>> abs(result.f_root) < 1e-10
True

```

Here Brent's method finds a zero within a sign-changing interval. The record
contains the root, residual, status, and work counters. That pattern—calculate,
inspect, validate—runs through the documentation, although return types and
stopping criteria differ by routine.

## Choose a learning path

| If you want to… | Read in this order |
| --- | --- |
| Run your first calculation | [Installation](installation.md), [getting started](getting-started.md), a task guide below |
| Understand the numerics | Task guide, its experiment, [design and validation](design.md) |
| Build a multi-step model | [Array layer](guides/numeric.md), [scientific workflows](guides/workflows.md), [limitations](limitations.md) |
| Look up a call or default | [API reference](api/index.md), then the linked guide for assumptions |
| Diagnose an unexpected answer | [FAQ](faq.md), [limitations](limitations.md), the guide's validation discussion |
| Improve the project | [Contributing](contributing.md), [figure methodology](figures.md), [release process](release-process.md) |

## Numerical methods by problem

| Problem | Guide | Typical starting point |
| --- | --- | --- |
| Arrays, broadcasting, buffers, numeric files | [The array layer](guides/numeric.md) | `from quadrivium import numeric as np` |
| Results, errors, norms, output recording | [Core conventions](guides/core.md) | Read the result's status and diagnostics |
| Linear systems and decompositions | [Linear algebra](guides/linalg.md) | `solve`, `lu_factor`, `conjugate_gradient` |
| Equations and fixed points | [Root finding](guides/rootfind.md) | `brent`, `newton_system` |
| Values between nodes | [Interpolation](guides/interpolate.md) | `pchip`, `cubic_spline`, `barycentric` |
| Fitting and function compression | [Approximation](guides/approx.md) | `chebyshev_fit`, `chebfun` |
| Derivatives | [Differentiation](guides/diff.md) | Finite differences or supported automatic differentiation |
| Integrals | [Integration](guides/integrate.md) | `quad`, `quad_vec`, composite rules |
| Time-dependent systems | [Ordinary differential equations](guides/ode.md) | `solve_ivp`, `bdf_adaptive`, `radau_adaptive` |
| Spatial equations | [Partial differential equations](guides/pde.md) | Heat, Poisson, advection, FEM, spectral methods |
| Parameter estimation and minimization | [Optimization](guides/optimize.md) | `minimize`, `least_squares` |
| Signals and frequency analysis | [Transforms](guides/transforms.md) | `fft`, `stft`, stateful filters |
| Randomness and inference | [Stochastic methods](guides/stochastic.md) | Seeded sampling, SDEs, MCMC diagnostics |
| Mathematical function evaluation | [Special functions](guides/special.md) | Gamma, error functions, Bessel families |

The [scientific workflow guide](guides/workflows.md) connects these areas:
fit a model, reuse expensive work, choose stored output, and examine sensitivity.
The [example scripts](examples.md) provide longer executable tours.

## Conventions that matter

Documentation imports use `qd` for `quadrivium`. The alias `np` refers to
`quadrivium.numeric`, not NumPy. Outputs are Quadrivium arrays unless the
routine explicitly returns a scalar, tuple, mapping, or result record.

A success flag means the selected method met its implementation's stopping
condition. It does not prove that the model is appropriate or that every digit
is accurate. Different methods expose different evidence: residuals, estimated
integration error, gradient norms, work counters, or refinement behavior.

No single method is best for every input. The guides explain the structures a
method needs, what can go wrong, and how to check the output independently.
Graphs show concrete experiments using this package, accompanied by the
assumptions needed to interpret them. See [how to reproduce them](figures.md).

## Documentation and source

The hand-written guides explain decisions and scientific meaning. The reference
is generated from exports, signatures, docstrings, and public class methods.
Runnable `pycon` examples are checked by the documentation tests; site builds
check page links and anchors. Numerical plots are generated separately and
committed, so a normal site build does not rerun the experiments.

For help with an installation or calculation, start with the [FAQ](faq.md).
For development procedures, see [Contributing](contributing.md).
