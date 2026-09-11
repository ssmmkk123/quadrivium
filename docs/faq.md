# Frequently asked questions

## What is Quadrivium for?

It provides inspectable numerical methods and scientific workflows: solve an
equation, examine the iterations, compare discretizations, and connect a model
to fitting or uncertainty analysis. Choose based on the method's documented
contract and your own accuracy and scale requirements. The [home page](index.md)
organizes the guides by problem.

## Does it depend on NumPy or SciPy?

No runtime package dependency is declared. The package's C extension supplies
arrays and numerical kernels. NumPy is used as an independent test reference,
and Matplotlib is optional for regenerating documentation figures.

## Why do examples import something called np?

The convention is `from quadrivium import numeric as np`. The alias is local
to the example. Results are Quadrivium arrays, not NumPy arrays. Many familiar
operations exist, but [the array guide](guides/numeric.md) describes compatibility
boundaries. Use explicit conversion at external-library boundaries.

## Why is _qnp missing after cloning the repository?

The required C extension has not been built for that interpreter. From the
checkout, run `python -m pip install -e .`. If the build fails, inspect the
compiler error. See [installation troubleshooting](installation.md#troubleshoot-an-installation).

## Why is an API in these docs absent from my installation?

The site and repository documentation describe the checkout they were built
from, which may contain unreleased APIs. Inspect `qd.__file__` and
`qd.__version__`, then compare with the [changelog](changelog.md). The version
string alone does not distinguish an unreleased working tree from its base
release. Use matching source and documentation.

## Can I disable the compiled backend?

You can select Python reference implementations of accelerated methods with
`qd.accel.disabled()` or `QUADRIVIUM_NO_ACCEL=1`. The C array engine remains
required. `qd.accel.show_config()` reports backend information.

## Why does a small residual not guarantee a good answer?

It indicates that the computed answer nearly satisfies the supplied equation.
Ill-conditioning can amplify a small data or arithmetic perturbation into a
large solution change. Inspect both residuals and sensitivity. See
[linear algebra](guides/linalg.md) and [design and validation](design.md).

## What should I do with converged=False?

Read the message and inspect finite output and available histories. Confirm
the problem's assumptions, input scale, initial guess or bracket, and stopping
budget. Some methods instead raise an exception; there is no universal
failure-return policy across every API. The [getting-started guide](getting-started.md#how-failure-is-reported)
shows both cases.

## Why did tighter tolerance stop improving my result?

You may have reached rounding limits, noisy callbacks, poor conditioning,
interpolation error, or a different source of discretization error. ODE
`tolerance` is not a global error certificate. Refine one source of error at a
time and compare a quantity of interest with an independent reference.

## Does solve_ivp return the same array layout as other libraries?

Its `y` has shape `(stored_times, state_dimension)`. Querying a vector of times
returns `(query_times, state_dimension)`. Use `y_final` for the final computed
state when selected output times omit the endpoint. See [ODEs](guides/ode.md).

## How can I avoid retaining a huge trajectory?

Use supported `save_at`, `save_every`, or `final_only` controls before solving.
A callback can observe copied states as integration advances. Retaining less
output also reduces information available for later interpolation. The
[workflow guide](guides/workflows.md) explains memory and checkpoint contracts.

## Must I supply derivatives?

Many methods provide finite-difference fallbacks, but not every derivative-based
routine does. Check the exact signature. Supplying derivatives can reduce
function calls and finite-difference error. [Automatic differentiation](guides/diff.md)
works only through its supported operations; it is not a universal wrapper for
arbitrary Python or external functions.

## How do I make random calculations reproducible?

Pass a documented integer seed or Quadrivium generator. An integer restarts the
stream each time; a generator advances it across calls. Record the environment,
source revision, method, and options as well as the seed.

```pycon
>>> import quadrivium as qd
>>> first = qd.monte_carlo(lambda x: x*x, 0, 1, n=1000, rng=42)
>>> again = qd.monte_carlo(lambda x: x*x, 0, 1, n=1000, rng=42)
>>> float(first) == float(again)
True

```

Reproducibility is not independence. For uncertainty estimates, use appropriate
independent streams or randomized replications and [diagnostics](guides/stochastic.md).

## How should I interpret the graphs?

Read the problem, axes, reference, and caption together. A convergence curve
shows one experiment, not a guarantee for every input. A work count differs
from elapsed time. The new figures use actual package calculations and
analytic references or explicit statistical experiments; the
[figure methodology](figures.md) explains regeneration and review.

## Can I use a special function at a pole or in an extreme tail?

Only according to its documented behavior. Some functions return infinities;
others reject arguments. Subtracting a CDF from one can lose tail accuracy;
use a supported survival function directly. Near zeros, compare absolute error.
See [special functions](guides/special.md) and [stochastic distributions](guides/stochastic.md).

## How do I cite an experiment using the package?

Include Quadrivium, the repository URL, the release or commit, and the numerical
method and options used. For example, a reproducibility note can give the
source revision, Python version, backend, dtype, tolerance, and random seed.
Do not identify an unreleased working tree only by its base version number.

## Where should I report a problem?

Use the repository's issue tracker for a minimal reproducible numerical or
installation problem. Include the expected answer and how it was established.
For private security or conduct reports, follow the root repository's
`SECURITY.md` or `CODE_OF_CONDUCT.md`. See [Contributing](contributing.md) for
development and documentation checks.
