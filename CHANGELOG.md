# Changelog

This file records user-visible additions, behavior changes, and notable fixes.
`Unreleased` describes the current source tree; it does not describe a new
published package version. Released entries retain the installation and
behavior details that applied at the time of release.

## [Unreleased]

### Added

- **A native numeric namespace.** `quadrivium.numeric` provides strided
  multidimensional arrays, broadcasting, basic and advanced indexing, views,
  reductions, dense linear algebra, FFTs, sorting, searching, polynomial
  helpers, and random sampling. Supported storage types are `bool`, `int64`,
  `float32`, `float64`, `complex64`, and `complex128`. Compatible external
  arrays can exchange data through the buffer protocol.
- **Array persistence and random state.** Numeric `.npy` save/load, memory
  mapping with retained buffer lifetimes, and serializable random-generator
  state support reproducible workflows without object serialization.
- **Batched and complex dense calculations.** Supported array-layer operations
  accept batches and complex data. High-level solves and selected
  factorizations expose those capabilities, and reusable LU, Cholesky, and QR
  factor objects support repeated right-hand sides and caller-owned output
  arrays. Thomas elimination accepts a matrix of right-hand-side columns.
- **Sparse and matrix-free workflows.** Linear operators, sparse arithmetic,
  sparse triangular solves, incomplete-factorization preconditioners, and
  validated compressed storage allow supported algorithms to avoid dense
  intermediates.
- **Controlled solver output.** Supported ODE, PDE, and SDE routines can retain
  selected times, every nth state, or only the final state. Root finders,
  optimizers, and iterative linear solvers offer history retention and callback
  controls on supported methods. JSON ODE/PDE checkpoints store numerical
  state; restarting requires the model to be supplied again.
- **Stiff integration and sensitivities.** Adaptive BDF and Radau methods,
  structured Jacobian solve paths, forward sensitivities, and checkpointed
  adjoint calculations extend the time-integration interfaces.
- **Bounded nonlinear least squares.** Robust losses, bounds, parameter
  scaling, and sparse finite-difference coloring support residual models with
  more realistic constraints and evaluation costs.
- **Array automatic differentiation.** `Tensor`, array gradients, Jacobian-vector
  products, and vector-Jacobian products differentiate supported real array
  operations without assembling a full Jacobian for directional products.
- **Stateful signal processing.** Filter state, polyphase resampling, and
  reconstructable short-time Fourier transforms support block processing and
  analysis followed by synthesis.
- **Probability and simulation tools.** Distribution objects and tail
  functions, randomized Sobol integration, and rank-based MCMC diagnostics
  extend the statistical interfaces.
- **Adaptive representations and geometry.** Piecewise adaptive Chebyshev
  approximation, shared-node vector quadrature, sparse adaptive FEM, planar
  triangulation, local radial basis interpolation, and pseudo-arclength
  continuation provide additional ways to resolve difficult functions and
  solution branches.
- **Array special functions.** Public special-function interfaces support
  array evaluation as documented while preserving scalar returns for scalar
  inputs. Series, recurrences, continued fractions, and asymptotic branches
  retire converged array elements individually where applicable.
- **Backend controls.** `quadrivium.accel` exposes availability, active backend,
  version, kernel inventory, configuration, and a `disabled()` context manager.
  `QUADRIVIUM_NO_ACCEL=1` selects reference algorithm implementations; it does
  not remove the required native numeric array layer.
- **Validation infrastructure.** Differential numeric-backend tests,
  accelerated/reference comparisons, vectorization regressions, isolated
  timing and memory checks, documentation doctests, generated-reference
  checks, and figure-catalog checks cover the expanded implementation.

### Changed

- **The runtime array dependency changed from NumPy to the project's C core.**
  Library code uses `quadrivium.numeric`, compiled as `quadrivium._qnp` from
  `csrc/`. The current source distribution therefore requires a working native
  build or a compatible built wheel. NumPy remains useful as an optional
  validation and interoperability dependency; arbitrary NumPy APIs and dtypes
  are not implied to be supported.
- **C replaces the intermediate Rust backend.** Acceleration kernels build
  with the C array core. Rust sources, Cargo dependencies, and the optional
  Rust build path have been removed. The active compiled backend reports
  `"c"`. Earlier unreleased Rust installation instructions and environment
  switches no longer describe the current build.
- **Numeric compatibility is tested at the supported-operation level.**
  Pairwise reductions and seeded PCG64 sampling follow the compatible numeric
  conventions. Symmetric eigendecomposition chooses a deterministic
  eigenvector sign, which can change seeded algorithms that use eigenvectors
  as a sampling basis, such as CMA-ES. Integer `reciprocal` promotes to floating
  point instead of applying integer division.
- **Reusable memory and bounded work replace avoidable allocations.** Native
  kernels borrow input buffers where safe and return owned working arrays.
  Reduced QR constructs a thin Q, compact least squares applies reflectors
  directly to right-hand sides, FFT plans use a bounded cache, and supported
  adaptive integration paths retain only requested output.
- **Structured algorithms replace dense formulations.** Natural, clamped, and
  not-a-knot cubic splines solve tridiagonal systems. Direct two-dimensional
  Poisson-family solves use block-tridiagonal structure. QR eigenvalue
  iteration first reduces to Hessenberg form, and Jacobi rotations update
  only affected rows and columns.
- **Dense updates use their mathematical structure.** Dense BFGS and Broyden
  inverse-Hessian updates use quadratic-cost rank updates, retaining an
  alternative expression for extreme arithmetic. Dense factorizations use
  blocked matrix products; implicit Runge-Kutta stage assembly uses array
  products.
- **Repeated work is shared within a calculation.** Grid/scattered
  interpolation uses bounded evaluation batches, kriging reuses factorizations,
  KDE uses bounded tiles, and piecewise-polynomial evaluation applies Horner's
  method across query arrays. Special-purpose paths reduce repeated conversion
  and argument packing in `as_vector` and `CountedFunction`.
- **Transforms and reductions use more compact implementations.** DCT-II uses
  a shorter FFT extension and returns owned real coefficients. Additional
  cosine/sine transforms use FFT formulations; FFT stages, spectral windows,
  and short wavelet filters reduce temporary work. Segmented and cumulative
  reductions avoid per-element Python allocations, sparse diagonal extraction
  uses bounded tiles, and sample summaries share sorting work.
- **Optimization reuses known objective information.** `strong_wolfe` accepts
  optional `phi0` and `dphi0` and carries evaluated bracket values through its
  search. BFGS, L-BFGS, DFP, SR1, and the Broyden-class interface accept `ftol`
  for objective-change stopping where applicable.
- **Additional numerical loops run in compiled kernels.** These include
  Givens QR, Jacobi SVD, tridiagonal systems with multiple right-hand sides,
  Hessenberg QR iteration, selected PDE sweeps, and adaptive Runge-Kutta stage
  and step-control calculations. Reference implementations remain available
  for supported backend comparisons.
- **Documentation has been rewritten throughout.** Guides explain method
  selection, shapes, outputs, tolerances, validation, and implementation
  limitations with executable examples. The generated API reference includes
  full signatures, complete docstrings, and public class methods. Seventeen
  reproducible numerical experiments replace the previous collection of 83
  plots and its diagrams, with paired light/dark figures, clearer axes,
  reference quantities, and explanatory captions. Figure source and recorded
  measurements make the comparisons reproducible.
- **Obsolete unreleased benchmark narratives have been consolidated.** Earlier
  measurements mixed superseded Rust, NumPy, and C configurations and different
  workloads. They are no longer presented as current performance guarantees.
  Measure the checked-out version, active backend, input sizes, and machine
  configuration for a meaningful comparison.

### Fixed

- **Native memory safety and integer edge cases.** Basic and advanced indexing
  reject excessive dimensionality before writing array headers. Array byte
  counts are checked for overflow before allocation and view export. Minimum
  signed-integer division/remainder by minus one uses defined wrapping rather
  than trapping. Zero-dimensional conversions avoid invalid null-pointer
  comparisons, and cumulative integer operations use defined wrapping.
- **Aliased writes.** Assignment and in-place unary/binary operations snapshot
  overlapping sources when required, correcting reversed-slice assignments,
  overlapping updates, and aliased `out=` operations. Disjoint strided views
  avoid unnecessary copies even when their address ranges overlap.
- **External buffer ownership and writability.** Arrays and views keep buffer
  exports alive for their lifetime. Read-only exporters cannot become writable
  through array flags. Native kernels preserve supported Fortran, strided,
  and unaligned inputs; in-place kernels enforce their alignment and
  contiguity requirements.
- **Empty and extreme-scale numeric behavior.** Empty means return NaN rather
  than a plausible zero. Euclidean norms handle extreme scales and initialize
  empty-axis outputs. Cumulative reductions preserve the first signed zero
  and complex infinity; negative segment boundaries agree between contiguous
  and strided paths.
- **Input validation.** Fractional advanced indices and repeated reduction
  axes are rejected. Random distributions validate parameters before drawing,
  including invalid scales and rates. Numeric FFTs reject zero-length
  transform axes, and native kernels reject malformed dimensions before
  entering numerical loops.
- **FFT lengths, signs, and phases.** Real inverse FFTs resize the retained
  half-spectrum before constructing its Hermitian counterpart for a requested
  output length. Mixed-radix inverse signs and resampling Nyquist treatment
  are corrected. Bluestein phases reduce squared indices modulo the period
  before floating conversion, improving long DCT-I accuracy.
- **Signal-processing edge cases.** Singleton KDE bandwidth handling and empty
  spectrogram outputs are corrected.
- **Special-function regimes and overflow.** Airy, Struve, Bessel Y,
  polygamma, sine-integral, and cosine-integral calculations use revised
  series, quadrature, continued-fraction, or asymptotic regimes to avoid
  severe cancellation and invalid large-argument behavior. Bessel Y overflow
  preserves its sign, gamma/factorial overflow returns positive infinity,
  and `expint_n(1, x)` avoids a spurious division by zero.
- **Scale-aware eigenvalue termination.** Jacobi eigenvalue iteration detects
  its rounding floor, and unshifted QR uses relative subdiagonal deflation.
  Large or rescaled matrices no longer exhaust iteration budgets solely
  because an unattainable absolute off-diagonal threshold was used.
- **Consistent partial eigenpairs.** LOBPCG projects onto its final search
  block before returning, keeping eigenvalues and eigenvectors from the same
  approximation and avoiding placeholder infinity when a basis collapses.
- **Barycentric weight scaling.** Node differences are scaled before weight
  products so an affine change of interpolation interval does not by itself
  cause overflow and NaN results for large node sets.
- **Input preservation in tridiagonal solves.** A two-dimensional Thomas
  right-hand side is copied before the in-place native solve, so caller-owned
  contiguous data are not overwritten.
- **Adaptive time-integration termination.** Runge-Kutta respects its first-step
  limit, rejects nonfinite or unsatisfiable steps, supports backward and
  zero-length dense output, and reports trajectories truncated by `max_steps`
  as unsuccessful. NaN Poisson updates no longer report convergence.
- **Optimization stopping.** BFGS, L-BFGS, DFP, and SR1 can terminate on small
  objective changes when finite-difference gradient noise prevents a useful
  gradient threshold. Gradient descent, Adam, RMSProp, AdaGrad, momentum, and
  Nesterov methods stop on nonfinite updates with an informative failure
  message instead of filling their remaining histories with NaNs.
- **Lifetime and concurrency.** Solver results release completed callback
  closures. Array differentiation retains immutable forward values and
  indexing keys. Backend contexts are isolated across threads and asyncio
  tasks, and cached FFT plans support concurrent use.
- **Sampling on singular covariances.** The eigendecomposition fallback in
  `multivariate_normal` suppresses eigenvalues at the decomposition's rounding
  floor instead of converting numerical noise into spurious sampled variance.
- **Packaging and regression collection.** Distribution builds exclude stale
  native binaries where inappropriate. CI uses pytest so parametrized
  regressions are collected, and documentation examples and supported
  accelerated/reference paths are checked explicitly.

## [1.1.0] - 2026-09-01

The first release published to PyPI introduced the Quadrivium name, packaging,
and documentation site. At this release, the library contained 836 public
functions and classes across 13 subpackages and depended on NumPy. Its
pure-Python wheel and installation requirements belong to this historical
release; they do not describe the current unreleased C-based source tree.

### Added

- The `quadrivium` distribution on PyPI, initially shipped as a pure-Python
  `py3-none-any` wheel requiring no compiler.
- A documentation site with installation and getting-started material,
  subpackage method-selection guides, a generated API reference, design and
  limitation discussions, and release instructions.
- `docs` and `dev` extras, expanded package metadata, project URLs, and a build
  version read from `quadrivium.__version__`.
- Tagged-release automation to build and check distributions, publish through
  PyPI OIDC Trusted Publishing, and attach artifacts and build attestations to
  GitHub Releases.
- Regression tests for the fixes below, executable documentation examples,
  and generated-reference consistency checks.

### Changed

- The development name `numethods` became `quadrivium` before the first public
  release. The distribution is installed as `quadrivium`, the Python import
  is `quadrivium`, and the base exception is `QuadriviumError`. Documentation
  uses the `qd` alias to distinguish the package from the `quad` integrator.

### Fixed

- Illinois and Pegasus false-position variants damp the retained endpoint
  only when it persists as required by the algorithm. Previously, damping
  every iteration reduced their convergence rate and made their trajectories
  effectively identical. Returned roots are unchanged; the iteration counts
  needed to reach them are reduced.
- Documented `rng=` integer seeds are normalized through `default_rng` in
  randomized range finding, randomized SVD/eigendecomposition, Nyström
  approximation, subspace iteration, LOBPCG, and SDE/simulation routines.
  Previously, passing an integer directly could raise `AttributeError` even
  though passing a generator worked.

Earlier development versions were not published to a package index.

[Unreleased]: https://github.com/ssmmkk123/quadrivium/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ssmmkk123/quadrivium/releases/tag/v1.1.0
