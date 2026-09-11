# Quadrivium

[![PyPI](https://img.shields.io/pypi/v/quadrivium.svg)](https://pypi.org/project/quadrivium/)
[![CI](https://github.com/ssmmkk123/quadrivium/actions/workflows/ci.yml/badge.svg)](https://github.com/ssmmkk123/quadrivium/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Quadrivium is a numerical computing library with readable implementations of
linear algebra, calculus, differential equations, optimization, transforms,
and stochastic methods. It includes its own C array engine and numerical
kernels, with **no runtime package dependencies**.

Use it to study an algorithm, compare methods on the same problem, or build a
scientific workflow whose residuals, tolerances, and intermediate results you
can inspect. Python implementations and C kernels live in the same repository.
The array engine is required even when numerical acceleration is disabled.

**[Read the documentation](https://ssmmkk123.github.io/quadrivium/)** ·
[Installation](docs/installation.md) · [Getting started](docs/getting-started.md) ·
[Scientific workflows](docs/guides/workflows.md) · [API reference](docs/api/index.md)

## Install

Python 3.9 or newer is required. Installing a matching binary wheel needs no
compiler; building from source needs a C compiler and Python development headers.

```bash
python -m pip install quadrivium
```

The documentation in this checkout describes the working tree, including
[unreleased changes](CHANGELOG.md). For these APIs, install this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
See [installation and troubleshooting](docs/installation.md) for details.

## Solve a problem and check the answer

Examples use `qd` for the package and `np` for **Quadrivium's own array module**.
The alias does not import NumPy. Array results have type
`quadrivium.numeric.ndarray`; compatible external buffers can be converted at
the boundary.

```pycon
>>> import math
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> root = qd.brent(lambda x: x*x - 2, 0, 2)
>>> root.converged and abs(root.root**2 - 2) < 1e-10
True
>>> integral = qd.quad(lambda x: math.exp(-x*x), -np.inf, np.inf)
>>> abs(float(integral) - math.sqrt(math.pi)) < 1e-10
True
>>> solution = qd.solve_ivp(lambda t, y: -2*y, (0, 1), [1.0], rtol=1e-9)
>>> solution.success and abs(float(solution.y_final[0]) - math.exp(-2)) < 1e-8
True

```

A solver's success flag reports its stopping condition. It does not certify
the model or guarantee a global error bound. Check a residual, an analytic
solution, a conservation law, or a refinement study as appropriate.

```pycon
>>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
>>> b = np.array([1.0, 2.0])
>>> x = qd.solve(A, b)
>>> float(np.linalg.norm(A @ x - b)) < 1e-12
True
>>> objective = lambda x: (x[0] - 2)**2 + 3*(x[1] + 1)**2
>>> gradient = lambda x: np.array([2*(x[0] - 2), 6*(x[1] + 1)])
>>> fit = qd.minimize(objective, [0.0, 0.0], method="bfgs", grad_f=gradient)
>>> fit.converged and float(np.linalg.norm(fit.x - [2.0, -1.0])) < 1e-6
True

```

## Find the right method

| Your task | Start here | What to check |
| --- | --- | --- |
| Work with arrays, batches, files, and buffers | [Array layer](docs/guides/numeric.md) | Shape, dtype, views, and supported operations |
| Understand results and output storage | [Core](docs/guides/core.md) | Status fields and retained history |
| Solve systems, factor matrices, compute eigenvalues | [Linear algebra](docs/guides/linalg.md) | Residual, conditioning, structure, rank |
| Solve scalar or nonlinear equations | [Root finding](docs/guides/rootfind.md) | Bracket assumptions and final residual |
| Evaluate between measured points | [Interpolation](docs/guides/interpolate.md) | Node placement, overshoot, extrapolation |
| Fit or compress a function | [Approximation](docs/guides/approx.md) | Error away from the fitting points |
| Compute derivatives or sensitivities | [Differentiation](docs/guides/diff.md) | Step size and supported differentiated operations |
| Integrate functions or sampled data | [Integration](docs/guides/integrate.md) | Error estimate, singularities, evaluation cost |
| Integrate a dynamical system | [ODEs](docs/guides/ode.md) | Stiffness, local tolerance, stored output |
| Solve a spatial model | [PDEs](docs/guides/pde.md) | Boundary conditions, stability, mesh refinement |
| Fit parameters or minimize a cost | [Optimization](docs/guides/optimize.md) | Scaling, stationarity, constraints |
| Analyze or filter signals | [Transforms](docs/guides/transforms.md) | Sampling rate, normalization, boundaries |
| Simulate uncertainty or sample a distribution | [Stochastic methods](docs/guides/stochastic.md) | Seeds, uncertainty, effective sample size |
| Evaluate mathematical functions | [Special functions](docs/guides/special.md) | Domains, poles, tails, identities |

Each guide connects method choice to runnable examples and a numerical
experiment. The [API reference](docs/api/index.md) lists complete signatures,
source documentation, and class methods; subpackages expose more methods than
the curated top-level namespace.

## Build a complete scientific workflow

The [workflow guide](docs/guides/workflows.md) explains how to combine:

- reusable factors and matrix-free operators for repeated solves;
- bounded output, callbacks, and numerical checkpoints for long simulations;
- adaptive stiff ODEs, sensitivities, and robust nonlinear least squares;
- array differentiation, adaptive approximation, and vector quadrature;
- streaming filters, distributions, randomized quasi-Monte Carlo, and diagnostics.

The [limitations](docs/limitations.md) describe the boundaries of these APIs.
In particular, support for complex or batched arrays in the numeric engine does
not imply that every higher-level method accepts them.

## Understand and reproduce the graphs

The documentation's graphs are generated experiments, with explicit problems,
units or dimensionless axes, reference solutions, and interpretation beside
the figure. They compare error and work or expose a numerical failure mode.
They are not hardware performance promises.

```bash
python tools/gen_docs.py
python tools/gen_figures.py
python -m pytest -q tests/test_docs.py
python -m mkdocs build --strict
```

The SVGs are checked in for both light and dark themes. Matplotlib is needed
only to regenerate them. See [figure methodology](docs/figures.md) for the
catalogue, reproduction commands, and visual review process.

## Develop and validate

```bash
python -m pytest -q
QUADRIVIUM_NO_ACCEL=1 python -m pytest -q
python tools/gen_docs.py --check
python tools/gen_figures.py --check
```

The second test command selects Python implementations of accelerated methods;
it still uses the C array engine. Test dependencies include NumPy as an
independent reference. Tests cover numerical identities, convergence, array
ownership, native safety, output storage, and documentation.

See [Contributing](CONTRIBUTING.md), [Support](SUPPORT.md), and
[Security](SECURITY.md). Quadrivium is distributed under the [MIT license](LICENSE).
