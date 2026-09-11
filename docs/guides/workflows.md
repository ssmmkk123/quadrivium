# Scientific workflows

Numerical methods become most useful when composed: an optimizer calls an ODE
solver, many right-hand sides share a factorization, or a long signal arrives
in chunks. At those boundaries, output shapes, retained state, tolerances, and
reproducibility matter as much as the individual algorithm.

This guide develops connected examples and explains what information must pass
between stages. The dedicated guides provide deeper method selection and the
API pages give complete signatures.

## Plan the numerical contract

Before composing algorithms, identify the output that the next stage actually
needs. An endpoint calculation does not need every time step; a terminal
objective gradient does not always need a full sensitivity tensor; a repeated
linear solve does not need to refactor the same matrix.

| Workflow need | Useful interface | Main check |
| --- | --- | --- |
| Endpoint of a long simulation | `final_only=True`, `y_final`, `final` | Actual endpoint and completion status |
| Values at observation times | `save_at` | Interpolation accuracy and ordered times |
| Continue an interrupted calculation | Checkpoint plus model/options | Compatible restart state |
| Many solves with the same matrix | Reusable factor object | Residual for each right-hand side |
| Large sparse or implicit matrix | `LinearOperator`, sparse matrices | Adjoint correctness and iterative convergence |
| Differentiate simulation outputs | Forward sensitivities or adjoint | Independent perturbation check |
| Data with possible outliers | Robust `least_squares` | Residual interpretation and scale |
| Parallel randomized calculations | Indexed child RNG streams | Stable assignment of streams to tasks |
| Continuous signal processing | Stateful filter/resampler | Chunk invariance and boundary conventions |

Use `quadrivium.numeric` for the examples. Explicit dtype choices affect array
storage, while high-level solvers may promote their working states to double
precision. Compact input storage is not a promise that all intermediate arrays
or solver calculations use the same compact dtype.

## Recover a parameter through an ODE solve

Suppose a decay model is `y'=-k*y`, `y(0)=1`, and the observation is its value
at time two. In this controlled example the observation is generated from
`k=1.7`, so there is an exact answer against which to validate the workflow.

```pycon
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> observed = float(np.exp(-1.7 * 2.0))
>>> def endpoint(rate, tolerance=1e-9):
...     solution = qd.solve_ivp(lambda t, y: -rate*y, (0, 2), [1.0],
...                             rtol=tolerance, atol=tolerance*1e-3,
...                             final_only=True)
...     if not solution.success:
...         raise RuntimeError(solution.message)
...     return float(solution.y_final[0])
>>> recovered = qd.brent(lambda rate: endpoint(rate) - observed, 0.2, 3.0, tol=1e-10)
>>> recovered.converged, abs(float(recovered.root) - 1.7) < 1e-7
(True, True)

```

The inner solver produces one scalar endpoint. The outer root solver adjusts
`rate` until the endpoint residual changes from positive to negative and is
then refined inside the bracket. The outer result's `function_calls` counts
endpoint evaluations, not the total right-hand side evaluations performed
inside them.

Two numerical errors coexist: root-search error and ODE integration error.
An outer tolerance much smaller than the inner simulation error only solves
the approximate model more precisely. It does not recover the exact parameter
more accurately.

```pycon
>>> coarse = qd.brent(lambda rate: endpoint(rate, 1e-3) - observed, 0.2, 3.0, tol=1e-10)
>>> abs(float(recovered.root) - 1.7) < abs(float(coarse.root) - 1.7)
True

```

<figure markdown="span">
  ![Decay parameter recovered from an endpoint observation, with error versus inner solver tolerance](../assets/figures/workflow-parameter-recovery.svg#only-light)
  ![Decay parameter recovered from an endpoint observation, with error versus inner solver tolerance](../assets/figures/workflow-parameter-recovery-dark.svg#only-dark)
  <figcaption>A single exact observation y(2)=exp(−3.4) identifies k=1.7 in this decay model. Brent's outer root tolerance is held tight while the inner IVP tolerance changes. The fitted trajectory can look accurate on the left while the parameter-error plot still exposes the cost of a loose inner solve. This experiment measures numerical error, without observational noise.</figcaption>
</figure>

With noisy observations at several times, formulate a residual vector and use
`least_squares` instead of forcing an exact match to every measurement.
With several parameters, identifiability matters: different parameter vectors
may predict almost identical data. Refining solver tolerances cannot resolve
that lack of information.

## Store observations independently of internal steps

Output control belongs in the integration call, rather than in a slice taken
after creating a full trajectory.

```pycon
>>> rhs = lambda t, y: -y
>>> chosen = qd.solve_ivp(rhs, (0, 1), [1.0], save_at=[0.2, 0.6], rtol=1e-9)
>>> chosen.t.tolist()
[0.2, 0.6]
>>> chosen.y.shape
(2, 1)
>>> chosen.checkpoint.t
1.0
>>> abs(float(chosen.y_final[0]) - float(np.exp(-1))) < 1e-8
True

```

The last stored sample is at time `0.6`; the actual integrated endpoint is at
`1.0`. Use `y_final` for that endpoint, and `PDESolution.final` for a PDE field.
`y[-1]` or `u[-1]` means the last **retained** sample and can represent a
different time.

| Output policy | Stored states | Suitable use |
| --- | --- | --- |
| Default | Accepted states/full method grid | Diagnosis and later interpolation |
| `save_every=k` | Initial, every kth state, endpoint | Coarse trajectory monitoring |
| `save_at=[...]` | Specified observation times | Comparing model and data |
| `final_only=True` | Actual last state only | Parameter sweeps and endpoint statistics |

`save_at` must be finite, strictly ordered in the integration direction, and
inside `t_span`. It cannot be empty, combined with `final_only`, or combined
with a nondefault `save_every`. A reverse-time ODE run uses decreasing output
times. SDE time spans must be increasing.

Requested times are interpolated as required. They do not tell an adaptive
integrator to resolve every physical event occurring between them. Later
interpolation also uses the retained information: saving two distant points
and then calling the result at many intermediate times cannot reconstruct
discarded fine detail.

### Monitor without retaining every state

```pycon
>>> seen = []
>>> def monitor(t, y):
...     seen.append(float(t))
...     return t >= 0.5
>>> stopped = qd.rk4(rhs, (0, 1), [1.0], n=20,
...                  callback=monitor, final_only=True)
>>> stopped.success, stopped.checkpoint.t, len(seen)
(False, 0.5, 11)

```

The callback sees the initial state and subsequent accepted states, as an
independent copy. Returning `True` stops at that state. A callback stop is
reported separately from successful completion of the requested interval;
inspect the status even when early stopping was intentional.

For optimization and iterative root finding, use `callback(x)`,
`store_history=False`, or `history_stride=k`. Iterative linear solvers preserve
the latest residual even when full residual history is disabled. If a callback
itself appends every state to a Python list, that user-created history can
still grow without bound.

## Save and resume numerical state

A `SolverCheckpoint` contains time, state, method, and method-specific numerical
metadata. JSON serialization stores those values; it does not serialize model
functions, closures, or executable Python objects.

```pycon
>>> from quadrivium.core import SolverCheckpoint, resume_ode, resume_pde
>>> partial = qd.solve_ivp(rhs, (0, 0.4), [1.0], rtol=1e-9, final_only=True)
>>> encoded = partial.checkpoint.to_json()
>>> restored = SolverCheckpoint.from_json(encoded)
>>> continued = resume_ode(rhs, restored, 1.0, rtol=1e-9, final_only=True)
>>> continued.success, abs(float(continued.y_final[0]) - float(np.exp(-1))) < 1e-8
(True, True)

```

Supply the model again, together with tolerances and other options that define
the continued problem. Store those options beside the checkpoint if the run
must be reproducible outside the current Python session.

Adaptive BDF and Radau restore their step/order/history metadata. Many other
multistep methods restart using a startup procedure from the saved state,
which is not a bit-identical continuation of previous internal history.
Split Hamiltonian or exponential models can use a custom restart adapter with
signature `solver(f, t_span, y0, **options)`. Delay restarts restore a recorded
history window and may need the original history function for earlier times.

For a PDE, resupply spatial and model options explicitly:

```pycon
>>> from quadrivium.pde import heat_btcs
>>> initial = lambda x: np.sin(np.pi*x)
>>> first_half = heat_btcs(initial, 0.1, (0, 1), (0, 0.1), nx=10, nt=50,
...                        final_only=True)
>>> second_half = resume_pde(heat_btcs, first_half.checkpoint, 0.2,
...                          alpha=0.1, x_span=(0, 1), nx=10, final_only=True)
>>> whole = heat_btcs(initial, 0.1, (0, 1), (0, 0.2), nx=10, nt=100,
...                   final_only=True)
>>> np.allclose(second_half.final, whole.final, atol=1e-12)
True

```

Wave and leapfrog PDE checkpoints carry two time levels. Their recurrence
restart requires the same time step and an interval containing an integer
number of those steps. SDE restart checkpoints are not supported. RNG state
alone is not a generic replacement for missing solver recurrence state.

## Reuse a factorization or expose an operator

If a matrix stays fixed while right-hand sides change, factor once and solve
repeatedly. `lu_factor`, `cholesky_factor`, and `qr_factor` provide reusable
objects. Their solve methods accept vector or matrix right-hand sides and an
optional output buffer.

```pycon
>>> matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
>>> factor = qd.lu_factor(matrix)
>>> right_sides = np.array([[4.0, 1.0], [3.0, 0.0]])
>>> solutions = factor.solve(right_sides)
>>> np.allclose(matrix @ solutions, right_sides, atol=1e-12)
True
>>> np.allclose(factor.solve([4.0, 3.0]), [1.0, 1.0])
True

```

LU supports ordinary, transpose, and adjoint solves using `trans="N"`, `"T"`,
and `"H"`, together with determinant and signed-log-determinant methods.
Cholesky requires positive-definite Hermitian input. Compact QR targets
full-column-rank least squares and stores reflectors instead of a full square
Q. Rebuild a factor when the matrix changes; it represents a fixed matrix,
not a live symbolic relationship to the original array.

For a matrix defined by its action, use `LinearOperator`:

```pycon
>>> diagonal = np.array([2.0, 3.0, 5.0])
>>> operator = qd.LinearOperator((3, 3), lambda v: diagonal*v,
...                               rmatvec=lambda v: diagonal*v)
>>> np.allclose(operator @ np.ones(3), diagonal)
True
>>> np.allclose(operator.H @ np.ones(3), diagonal)
True

```

`rmatvec` means the **conjugate transpose** product. It is needed by
rectangular least-squares solvers. `.T` and `.H` differ for complex operators.
A useful check is the adjoint identity between inner products for independently
chosen test vectors. CG, GMRES, CGNR, and LSQR accept appropriate operators;
their symmetry, definiteness, and adjoint requirements remain in force.

Sparse matrices and their incomplete-factorization/SSOR preconditioners keep
sparse storage on the supported paths. That does not eliminate conditioning
or fill-in concerns. Monitor residuals and iteration counts, and distinguish
the memory in the matrix representation from memory used by Krylov history.

## Compact, batched, and file-backed arrays

`float32` uses four bytes per real value; `float64` uses eight. `complex64`
and `complex128` use eight and sixteen bytes per complex value. Select dtype
according to the accuracy and memory needs of the data, then verify where a
higher-level solver promotes precision.

```pycon
>>> compact = np.ones((100, 4), dtype=np.float32)
>>> compact.nbytes
1600
>>> batch = np.array([[[2., 0.], [0., 3.]], [[4., 0.], [0., 5.]]])
>>> answers = np.linalg.solve(batch, np.ones(2))
>>> answers.shape
(2, 2)
>>> np.allclose(batch @ answers[..., None], np.ones((2, 2, 1)))
True

```

`matmul` broadcasts leading batch dimensions. Batched factorization and solve
interfaces have explicit matrix/vector axis conventions; see the
[numeric guide](numeric.md) before combining higher-dimensional right-hand
sides. Complex Cholesky and Hermitian eigenvalue problems require conjugate
symmetry, not ordinary transpose symmetry.

Use `save`/`load` for numeric `.npy` data and `open_memmap` for OS-backed array
storage. The following example writes to a temporary directory and cleans it
up after verification:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> with tempfile.TemporaryDirectory() as directory:
...     path = Path(directory) / "states.npy"
...     mapped = np.open_memmap(path, mode="w+", dtype=np.float32, shape=(3, 2))
...     mapped[:] = np.arange(6).reshape(3, 2)
...     np.flush(mapped)
...     loaded = np.load(path)
...     print(loaded.shape, loaded.dtype.name, float(loaded[2, 1]))
...     del mapped, loaded
(3, 2) float32 5.0

```

Modes `r`, `r+`, `w+`, and `c` mean read-only, read/write, create, and
copy-on-write. Flush writable mappings when persistence matters. A live view
keeps the underlying mapping pinned. `frombuffer` shares compatible aligned
contiguous storage without copying and respects the buffer owner's lifetime
and writability. Numeric file loading does not execute pickle payloads.

## Differentiate a simulation or an array expression

Forward sensitivities evolve derivatives alongside the state. For
`y'=p*y`, the exact endpoint derivative with respect to p is `t*y(t)`:

```pycon
>>> sensitive = qd.solve_ivp_sensitivities(
...     lambda t, y, p: p[0]*y, (0, 2), [1.0], [0.3],
...     jac_y=lambda t, y, p: np.array([[p[0]]]),
...     jac_p=lambda t, y, p: y[:, None], final_only=True, rtol=1e-9)
>>> sensitive.sensitivities.shape
(1, 1, 1)
>>> abs(float(sensitive.sensitivity_final[0, 0]) - float(2*np.exp(0.6))) < 1e-7
True

```

The sensitivity axes are `(retained_times, states, parameters)`.
`initial=True` also tracks derivatives with respect to initial state values;
`initial_sensitivity` handles an initial condition that depends on parameters.
Use `sensitivity_final` when `save_at` omits the endpoint. The separate
`augmented_checkpoint` contains the whole variational system and is not a
state-only IVP checkpoint.

An adjoint can be preferable for a scalar terminal objective with many
parameters. `adjoint_sensitivity` stores checkpoints and replays one segment
at a time. It requires a deterministic right-hand side, and its memory depends
on both checkpoint count and the longest replayed segment. These are
continuous sensitivities; adaptive step-selection decisions are not being
differentiated. Check gradients with finite perturbations at several solver
tolerances before using them in optimization.

For ordinary differentiable array expressions, use `Tensor`,
`array_value_and_grad`, `jvp`, or `vjp`. Products can avoid materializing a
dense Jacobian:

```pycon
>>> value, gradient = qd.array_value_and_grad(lambda x: np.sum(x*x), [1., 2., 3.])
>>> value, gradient.tolist()
(14.0, [2.0, 4.0, 6.0])
>>> _, product = qd.jvp(lambda x: x*x, [1., 2.], [3., 4.])
>>> product.tolist()
[6.0, 16.0]

```

Tensor values are read-only. Mutation and complex differentiation are not
supported. A Python callback or custom numerical solver is not automatically
differentiable merely because its inputs came from a Tensor.

## Compose fitting, randomness, and streaming signals

For observations with different scales, construct standardized residuals.
`least_squares` accepts bounds, robust losses, parameter scaling, numerical
Jacobian sparsity, and operators with forward/adjoint products. It applies
normal-equation products without constructing a dense normal matrix.
`curve_fit` builds residuals from a named model; covariance is unavailable
for robust losses in the current implementation. See [optimization](optimize.md)
for the output contract and interpretation.

Keep random streams separate from adaptive solver callbacks when randomness
belongs to the model. A solver may evaluate an ODE right-hand side repeatedly
at trial states, including rejected steps; consuming fresh randomness there
changes the model with the evaluation schedule. Use the dedicated
[SDE interfaces](stochastic.md#stochastic-differential-equations) for supported
noise models, or explicitly define the deterministic forcing realization.

For Monte Carlo workers, derive indexed child streams with `spawn_rngs`.
For randomized QMC, estimate uncertainty across independent scrambled
replicates instead of applying independent-sample formulas to net points.
Store generator or Sobol state with the sample count and experiment parameters.

Streaming signal processors follow the same state-management principle.
Reuse `FIRFilter`, `IIRFilter`, or `SOSFilter` across chunks; save their state
along with the coefficients. `PolyphaseResampler` also retains overlap and
counters and reports causal delay. `stft` stores synthesis metadata;
`istft` needs that metadata to reconstruct boundaries and partial frames.
See [transforms](transforms.md) for executable chunk and reconstruction checks.

## Adaptive representations and continuation

When a scalar function will be queried repeatedly, `chebfun` builds a
piecewise Chebyshev approximation and exposes evaluation, derivatives,
integrals, and roots. It adapts polynomial degree and interval subdivision
under explicit budgets.

```pycon
>>> approximation = qd.chebfun(lambda x: np.exp(x), (0, 1))
>>> approximation.converged
True
>>> abs(approximation.integrate() - float(np.e - 1)) < 1e-9
True
>>> combined = qd.quad_vec(lambda x: np.array([x, x*x]), 0, 1)
>>> np.allclose(combined.value, [0.5, 1/3], atol=1e-10)
True

```

`quad_vec` shares adaptive nodes across vector or complex integrands. Choose
componentwise or Euclidean error control according to the output's meaning;
`limit` bounds retained panels. Error estimates from these adaptive methods
are sample-based, not proofs that arbitrary unseen structure is absent.

For spatial geometry, `Delaunay` and `LinearNDInterpolator` expose planar
triangulation and point-location behavior; outside-hull choices must be made
explicit. `RBFInterpolator(neighbors=k)` uses bounded local systems with a
k-d tree. `adaptive_fem` estimates local error and refines triangles within
an element budget. Approximation error, spatial discretization error, and
linear-solve error should be checked separately.

`pseudo_arclength` follows equilibrium branches through folds by solving a
bordered predictor-corrector system. It can report stability using Jacobian
eigenvalues when the residual is a dynamical-system right-hand side.
Reported fold or stability-change locations are interpolated candidates,
not certified bifurcation points. Refine the continuation step and inspect
residuals before drawing conclusions from a branch diagram.

## Make results reviewable

For a repeatable calculation, retain the model version, parameters, units,
solver options, dtype, random-stream assignment, and output policy together
with the numerical result. Include the stopping status and a meaningful
validation quantity such as an independent residual, conservation check,
refinement comparison, or sampling standard error.

Budget the complete workflow: inner solves, derivative evaluations, retained
histories, sparse factorizations, and replay work can dominate an outer
algorithm's apparent iteration count. A useful benchmark compares the final
quantity's error at a measured cost and records what was held fixed.
