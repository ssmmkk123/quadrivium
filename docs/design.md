# Design and validation

Quadrivium puts numerical methods and their diagnostics in one inspectable
codebase. The design combines a native array engine, Python implementations of
algorithms, selected native kernels, and small result records. This page
explains how those layers fit together and how to establish that a calculation
is useful.

## Architecture

| Layer | Source | Responsibility |
| --- | --- | --- |
| Array engine | `csrc/`, `quadrivium/numeric/` | Storage, indexing, broadcasting, reductions, array linear algebra, transforms, random generation |
| Numerical kernels | `csrc/accel*.c`, `quadrivium/_accel/` | Native implementations of selected algorithmic work and backend dispatch |
| Method subpackages | `quadrivium/linalg/`, `ode/`, and peers | Public algorithms, callbacks, orchestration, diagnostics |
| Shared contracts | `quadrivium/core/` | Result records, helpers, selected exceptions, output storage and checkpoints |
| Documentation tools | `tools/gen_docs.py`, `tools/gen_figures.py` | Reference generation and reproducible numerical experiments |

The package does not delegate to an installed NumPy, SciPy, BLAS, or LAPACK
runtime. NumPy is an optional independent test reference; Matplotlib is a
figure-generation dependency. Some algorithms call the package's own
`numeric.linalg` primitives. Readability does not require every method to avoid
all lower-level kernels.

Importing the package requires its C array extension. Disabling acceleration
changes selected method dispatch; it does not produce a compiler-free package.

```pycon
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> with qd.accel.disabled():
...     result = qd.brent(lambda x: x*x - 2, 0, 2)
>>> result.converged and abs(result.root**2 - 2) < 1e-10
True

```

## Separate the model from the numerical method

A numerical solver receives a mathematical problem expressed in arrays and
callbacks. The solver cannot determine whether the equation describes your
experiment, whether units are consistent, or whether fitted parameters are
identifiable. Validation therefore has several levels:

1. **Model:** equations, units, domains, boundary conditions, and assumptions.
2. **Discretization:** step size, grid, basis, quadrature rule, or sample count.
3. **Algebra:** residuals and convergence of the discrete problem's solver.
4. **Arithmetic:** precision, conditioning, finite values, and cancellation.
5. **Interpretation:** uncertainty and sensitivity of the reported quantity.

Converging an algebraic solver to a tiny residual cannot repair an underresolved
mesh. Increasing sample count cannot remove bias from an incorrect stochastic
model. These distinctions determine what to measure in a test or graph.

## Independent references

A strong numerical test uses information that is independent of the particular
implementation being checked.

| Reference | Example | What it detects |
| --- | --- | --- |
| Closed-form answer | `y'=-y`, `y(0)=1`, so `y(t)=exp(-t)` | Value errors and convergence behavior |
| Manufactured solution | Choose a smooth field, derive its source and boundary values | Discretization, boundary, and sign mistakes |
| Reconstruction identity | `P @ A = L @ U` | Permutations, factors, and ordering errors |
| Structural identity | Orthogonality, Parseval, conserved mass | Violations hidden by a few value checks |
| Independent implementation | Compare the C array engine with NumPy in tests | Indexing, layout, reduction, and dtype errors |
| Refinement law | RK4 endpoint error decreases approximately as `h**4` | Missing terms and order reduction |

An identity can be insufficient by itself. A forward transform and inverse
might share a normalization error that cancels in a round trip. Combine
round-trip checks with a known spectrum or energy identity. Two implementations
can share the same conceptual error; an analytic reference adds a different
kind of evidence.

## Measure convergence rather than assume it

For smooth problems in the asymptotic regime, a method's leading error often
has the form `E(h) ≈ C*h**p`. With a known exact answer, estimate the observed
order from `log(E(h)/E(h/2))/log(2)`.

```pycon
>>> import math
>>> exact = math.exp(-1)
>>> def endpoint_error(n):
...     sol = qd.rk4(lambda t, y: -y, (0, 1), [1.0], n=n)
...     return abs(float(sol.y_final[0]) - exact)
>>> ratio = endpoint_error(20) / endpoint_error(40)
>>> 15 < ratio < 17
True

```

This verifies approximately fourth-order endpoint convergence on one smooth
problem. It does not establish the same rate for discontinuous forcing,
stiff order reduction, or a lower-order interpolant. Very coarse grids can lie
outside the asymptotic regime; very fine grids can reach a roundoff floor.

<figure markdown="span">
  ![Measured error under refinement compared with reference convergence slopes](assets/figures/validation-refinement-orders.svg#only-light)
  ![Measured error under refinement compared with reference convergence slopes](assets/figures/validation-refinement-orders-dark.svg#only-dark)
  <figcaption>Refinement experiments compare computed errors with reference slopes on problems with known answers. Read the range over which a slope holds, rather than treating a single fitted order as a universal property of all inputs. The figure is generated separately from the site build.</figcaption>
</figure>

Without an exact solution, compare successive refinements and monitor the
quantity actually used downstream. Richardson estimates rely on a dominant
error term and consistent refinements; disagreement is a reason to investigate,
not a reason to discard the inconvenient run.

## Residuals and conditioning

A residual measures how closely a candidate satisfies the supplied equation.
For `A @ x = b`, a useful scale-aware measure is
`||A @ x - b|| / (||A||*||x|| + ||b||)` when the denominator is nonzero.
This is a backward-error indicator. Forward solution error also depends on the
conditioning of `A`.

```pycon
>>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
>>> b = np.array([1.0, 2.0])
>>> x = qd.solve(A, b)
>>> relative_residual = float(np.linalg.norm(A @ x-b)) / (float(np.linalg.norm(A))*float(np.linalg.norm(x)) + float(np.linalg.norm(b)))
>>> relative_residual < 1e-14
True

```

Use consistent norms when applying a condition-number bound. Near a singular
matrix, small data perturbations can move the solution substantially even if
the solve has a tiny residual. The [linear algebra guide](guides/linalg.md)
shows both measurements in its experiment.

## Ownership and reproducibility

Numerical correctness includes storage behavior. Views can share memory;
explicit copies isolate it. Native code must handle strides, aliasing, empty
arrays, shape overflow, and buffer lifetimes without corrupting data. Output
recording and callback ownership need checks in addition to endpoint accuracy.

For a reproducible experiment, record the source revision, Python and plotting
versions, backend, dtype, inputs, tolerances, stopping budgets, and seeds.
Keep the problem and reference calculation in the figure source. Use a fixed
plotting environment for byte-for-byte SVG checks; differences in renderer
versions can change files without changing numerical conclusions.

## Performance evidence

Performance claims need a workload and an environment. Useful measurements
include callback evaluations, iteration counts, factorization reuse, stored
states, and peak memory as well as elapsed time. When comparing algorithms,
compare work at a common achieved accuracy where possible.

A warm cache, array conversion, thread settings, or a trivial callback can
change a timing result. The current documentation therefore uses explicit
numerical experiments rather than carrying historical speedup tables forward
as promises about a different backend or machine.

## Documentation as part of validation

`tests/test_docs.py` executes hand-written `pycon` examples, verifies generated
reference freshness, and checks figure references and catalogue consistency.
The strict MkDocs build validates navigation, links, and anchors. Figure
regeneration is a separate numerical run; a regular site build uses the
committed SVGs.

The [figure methodology](figures.md) explains the graph catalogue and how to
review regenerated plots. [Contributing](contributing.md) connects these checks
to the development workflow, and [limitations](limitations.md) records where
additional validation is needed.
