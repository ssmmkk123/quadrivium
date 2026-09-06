# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
a patch release fixes behaviour without changing the API, a minor release adds
methods or optional arguments, and a major release is required for anything
that could break an existing call.

## [Unreleased]

### Changed

- **NumPy is no longer a dependency: the array layer is now this project's own,
  written in C.** `quadrivium.numeric` replaces it -- the same names, the same
  signatures, the same semantics for the subset the package used, compiled from
  the sources in `csrc/` into `quadrivium._qnp`. It provides strided
  N-dimensional arrays over `bool`, `int64`, `float64` and `complex128`, with
  broadcasting, views, the whole indexing grammar (basic slicing, boolean
  masks, integer arrays, `ix_`, `add.at`), reductions, dense linear algebra
  (`solve`, `inv`, `det`, `slogdet`, `cholesky`, `eig`, `eigh`, `svd`, `lstsq`,
  `pinv`, `norm`), the FFT family, sorting and searching, polynomials, and a
  PCG64 generator. Library code that read `import numpy as np` now reads
  `from .. import numeric as np` and is otherwise untouched.

  Compatibility was the point, so it is tested as such: 367 differential tests
  compare the backend against NumPy operation by operation, exactly where an
  exact answer exists. Sums use NumPy's pairwise scheme and its block size, so
  a sum agrees bit for bit rather than to a tolerance. `random.default_rng`
  reproduces NumPy's SeedSequence, PCG64, ziggurat tables and Lemire bounded
  integers, so a seeded run yields the identical stream -- every published
  number in the documentation is unchanged. Array `repr` and `str` match NumPy
  character for character. Arrays export the buffer protocol, so callers who
  have NumPy can still hand NumPy arrays in and read results back out.

  Two differences are deliberate. `linalg.eigh` fixes the sign of each
  eigenvector so the largest-magnitude entry is positive, which LAPACK does not
  promise and which varies between BLAS builds; the decomposition is therefore
  reproducible across machines, but a routine that consumes eigenvectors as a
  random basis -- `cma_es` -- follows a different trajectory for a given seed
  than it did against LAPACK. And `reciprocal` on an integer array promotes to
  float rather than performing integer division.

  The package is faster and smaller for the change. Measured with
  `tools/bench_scalability.py`, one process per case and threads pinned to one:
  the Gaussian density estimate 32.3 -> 16.6 ms, Welch PSD 24.7 -> 14.4 ms, FFT
  0.44 -> 0.33 ms, sparse identity 0.015 -> 0.014 ms, and least squares and the
  CSR matrix-vector product unchanged. Peak resident memory falls from 36-43
  MiB to 23-29 MiB in every case, most of it the NumPy import that no longer
  happens. The element-wise kernels carry a vectorised `exp` accurate to one
  ulp of libm, the matrix product is a packed kernel with an AVX2/FMA
  micro-kernel running at roughly 46 GFLOPS single-threaded, and indexing has
  direct paths for `a[mask]` and `a[indices]` in both directions. Nothing is
  retained between operations: a buffer returns to the allocator when its array
  dies, which `test_temporaries_do_not_accumulate` checks by watching peak RSS
  across four hundred iterations of large temporaries.

- **The compiled Rust kernels now cross the boundary through the buffer
  protocol.** They previously took NumPy arrays through the `numpy` crate;
  they now accept anything exporting a PEP 3118 buffer and allocate their
  results through the array core. The kernels themselves are unchanged. Because
  the buffer protocol only entered CPython's limited API in 3.11, the extension
  is built against the running interpreter rather than the stable ABI -- which
  costs nothing, since the array core makes every wheel interpreter-specific
  anyway.

- **`multivariate_normal` drops eigenvalues at the decomposition's rounding
  floor** when falling back to an eigendecomposition for a singular covariance.
  `sqrt(1e-16)` is `1e-8`, large enough to appear in the samples of an exactly
  singular covariance; those eigenvalues are noise and their square roots are
  not.

### Fixed

- **`jacobi_eigen` and `qr_algorithm` could not converge on a matrix of any
  appreciable size.** Both compared the off-diagonal mass against `tol`
  outright, and rounding holds that mass near `eps*||A||`: for `||A||` of a
  few hundred the default `1e-12` is unreachable, so the iteration spent its
  whole budget and returned `converged=False` on an answer it had found in the
  first few sweeps. `jacobi_eigen` on a 120x120 SPD matrix ran all 100 sweeps
  in 94 ms and reported failure; it now converges in 14 sweeps and 19 ms with
  a relative residual of 1.4e-15. Jacobi's `tol` is now an absolute bound,
  tightened to a relative one below unit norm, plus a stagnation test -- its
  off-diagonal mass falls monotonically, so a sweep that fails to reduce it has
  reached the floor rounding imposes. `qr_algorithm` uses the relative
  per-subdiagonal deflation test its siblings `shifted_qr_algorithm` and
  `francis_qr` already used, which makes it scale-invariant: the same symmetric
  problem now takes the same 1607 iterations to the same accuracy whether it is
  scaled by `1e-4`, `1`, or `1e4`.
- **`barycentric` returned `nan` for node sets that were merely placed
  differently.** The weights are a product of `n` node differences, which runs
  like `capacity^n`: 400 Chebyshev nodes give weights of `1e117` on `[-1, 1]`
  and overflow to `inf` on `[0, 1]`, so the same interpolation problem worked
  or failed depending on an affine change of variable. The differences are now
  divided by the interval's logarithmic capacity first. Barycentric
  interpolation is invariant under a common factor on the weights, so no result
  changes -- but 400-node interpolation now reaches 1e-15 on `[-1, 1]`,
  `[0, 1]` and `[0, 0.01]` alike.
- **`lobpcg` paired eigenvalues from one iteration with eigenvectors from the
  next** when it ran out of iterations, and returned the `inf` placeholder if
  the search basis collapsed on the first pass. It now projects onto the final
  block before returning. On a 60x60 problem stopped early, the residual of the
  returned pair falls from 1.7e-2 to 1.7e-7.
- **`sine_integral` and `cosine_integral` lost up to ten digits at large
  arguments.** The ascending series has terms of size `e^x/sqrt(x)` summing to
  an `O(1)` answer, so `Ci(20)` was accurate only to 6.7e-10 and `Si(20)` to
  3.6e-10. Past `x = 4` both now use the continued fraction for `E_1(ix)`,
  whose real and imaginary parts they are: 3e-16 there, and in fewer iterations
  than the series needed.
- `gamma` and `factorial` return `+inf` past the double range instead of
  raising `OverflowError`, so one out-of-range element no longer fails a whole
  array. `expint_n(1, x)` no longer evaluates `1/(n-1)`.
- `linalg.thomas` copies a 2-D right-hand side before solving; the compiled
  kernel works in place, and an already-contiguous input would otherwise be
  overwritten in the caller's hands.

- Native kernels preserve Fortran, strided, and unaligned input layouts and
  reject malformed dimensions before entering numerical loops. In-place
  kernels require aligned C-contiguous buffers. NaN Poisson updates no
  longer report convergence.
- Adaptive Runge-Kutta respects the first-step limit, rejects nonfinite or
  unsatisfiable steps, supports backward/zero-length dense output, and marks
  trajectories truncated by `max_steps` as unsuccessful.
- Backend contexts are isolated between threads and asyncio tasks.
- Mixed-radix inverse FFT signs, resampling Nyquist handling, singleton KDE
  bandwidths, and empty spectrogram outputs are corrected.
- Optional builds exclude stale native binaries from portable wheels and
  honor Cargo-reported artifact paths, including custom target directories.
- CI uses pytest to collect parametrized regressions and explicitly runs the
  documentation doctests on both backends.
- **`bfgs`, `lbfgs`, `dfp` and `sr1` reported `converged=False` at the
  minimizer.** Their only stopping test was `‖grad f‖ < tol`, defaulting to
  `1e-10`. A central-difference gradient — the default when the caller supplies
  none — resolves the gradient to about `eps^(2/3)` times the scale of `f`, so
  that threshold sits below the noise floor and cannot be reached. On a
  6-dimensional Rosenbrock, L-BFGS reached `f = 1.3e-16` within 40 iterations
  and then ran the remaining 960 without moving, spending 114 619 function
  evaluations to return `converged=False`. All four now also stop when a full
  step fails to change the objective by a relative `ftol` (new argument,
  default `1e-12`), reporting `"converged: objective change below ftol"`. Same
  minimizer, 41 iterations, 1 107 evaluations.
- **`gradient_descent`, `adam`, `rmsprop`, `adagrad`, `momentum` and
  `nesterov` ran to `max_iter` after overflowing.** Once a step sent the
  iterate to NaN, every later iteration was NaN arithmetic appended to
  `history`, and the result said "maximum iterations reached" — indistinguishable
  from a run that nearly converged. They now stop at the overflow, report
  `converged=False` with a message naming the step size, and return the
  history up to that point. A diverging `gradient_descent` on Rosenbrock takes
  7 iterations and 0.4 ms rather than 10 000 and 513 ms, and its peak
  allocation drops from 1.6 MB to 3.4 KB.

### Added

- **Every function in `quadrivium.special` now evaluates arrays.** Forty-three
  of the fifty-one accepted only scalars -- `bessel_j0([1.0, 2.0])` raised
  `TypeError` -- so array work meant a Python loop at the call site. Each
  series, continued fraction, recurrence and AGM iteration now advances the
  whole input at once, with a mask retiring elements as they converge and each
  asymptotic series truncated per element at its own smallest term. Scalar
  arguments still return Python scalars, and the values are unchanged: all 51
  functions were replayed over their domains against the previous
  implementation and agree to 1e-12 relative or better. Against the Python loop
  a caller previously had to write, at 5 000-20 000 points: `fresnel_c` 734x,
  `airy_ai` 54x, `associated_legendre` 40x, `erfcx` and `bessel_i0` 24x,
  `dawson` 22x, `bessel_j0` 21x, `beta` and `struve_h0` 20x, `elliptic_k` 18x,
  `bessel_k0` and `hyp1f1` 15x, `elliptic_e` 14x, `hyp2f1` and `polygamma` 11x,
  `lambert_w` 10x, `digamma` and `zeta` 8-9x, `erfinv` and `sine_integral` 5x.
- `linalg.thomas` accepts a 2-D right-hand side, solving one tridiagonal matrix
  against every column. The elimination coefficients depend only on the matrix,
  so a block of systems costs barely more than one -- which is what
  line-relaxation and alternating-direction schemes need. Backed by a new
  `thomas_batch` kernel; `pde.heat_2d_adi` was making one boundary crossing per
  grid line and now makes one per half-step.
- **Three compiled kernels**: `svd_jacobi` (one-sided Jacobi SVD, 53x at
  n = 80), `givens_qr` (62x at n = 120) and `thomas_batch`. The first two are
  sequences of `O(n^2)` plane rotations driven from Python -- seven NumPy calls
  per rotation on `n^2/2` rotations per sweep, each doing a few microseconds of
  arithmetic. Both match their Python twins to machine precision.
- `tests/test_vectorization.py`: 29 tests and 163 subtests covering
  array-native evaluation for every special function, scale invariance of the
  eigenvalue iterations and of barycentric interpolation, the batched
  tridiagonal solve including its aliasing contract, and exact agreement
  between the blocked low-discrepancy sequences and the scalar recurrences they
  replaced.

- Compact Householder least squares in Python and Rust, applying reflectors
  to the right-hand side with `O(m*n)` workspace instead of forming full Q.
- Sparse operations avoid dense intermediates; compressed products use
  vectorized reductions and validate their storage structure.
- Bounded KDE tiles, vectorized FFT stages, reusable spectral windows, and
  short wavelet filters reduce interpreter overhead and temporary storage.
- `tools/bench_scalability.py` measures isolated-process timing, tracked
  allocations, and peak RSS. See `PERFORMANCE.md` for measured results and
  memory tradeoffs. The Rust crate drops its unused duplicate ndarray
  dependency and declares its dependencies' actual Rust 1.83 minimum.
- **Five compiled kernels that own a whole iteration** rather than one step,
  because the boundary crossing costs more than the arithmetic when a solve is
  thousands of `O(n^2)` sweeps:
  `hessenberg_qr_iterate` (unshifted QR on a Hessenberg matrix, 102× at
  n = 40), `sor_poisson` (lexicographic Gauss-Seidel/SOR, 114×),
  `thomas` (tridiagonal solve, 64× at n = 50 000), `lid_driven_cavity`
  (16×), and the `qr_algorithm` eigenvector accumulation. Each is
  bit-identical to its Python twin, including iteration counts and the
  exceptions raised, and `tests/test_accel.py` pins that.
- `optimize.bfgs`, `dfp`, `sr1`, `broyden_class` and `lbfgs` take `ftol`.
- **A packed SIMD `gemm` (`rust/src/gemm.rs`).** Hand-written AVX2/FMA
  micro-kernel holding a 6x8 tile of `C` in registers, cache blocking over all
  three dimensions, packed panels, and rayon across row bands, with a runtime
  feature check and a portable scalar fallback. Roughly 46 GFLOPS
  single-threaded, about 82% of this core's peak. `gemm_nt_acc` computes
  `A B'` without materialising the transpose.
- **Optional compiled backend.** A Rust extension (`quadrivium._quadrivium_rs`)
  now provides compiled versions of the kernels that dominate runtime. It is
  built automatically when a Rust toolchain is available and is entirely
  optional: without it every routine falls back to the pure-Python
  implementation, which stays in the tree and stays tested.
- `quadrivium.accel` for backend introspection and control — `available()`,
  `backend()`, `version()`, `kernels()`, `show_config()`, and a `disabled()`
  context manager that forces the reference implementation. The environment
  variable `QUADRIVIUM_NO_ACCEL=1` does the same process-wide, and
  `QUADRIVIUM_NO_RUST=1` skips the compiled build at install time.
- `tests/test_accel.py`: 31 tests and 100 subtests requiring the two backends to
  agree to floating-point noise and to raise the same exceptions, over sizes
  that straddle the kernels' 64-wide blocking thresholds. The kernels that run
  a whole iteration are held to a stricter contract: they drive their own
  convergence tests, so the iteration count and the converged flag have to
  match as well, not just the fields.

### Changed

- **The cubic splines solve a tridiagonal system instead of a dense one.**
  `natural_cubic_spline`, `clamped_cubic_spline` and `not_a_knot_spline` each
  built a dense `(n+1) x (n+1)` matrix and called `np.linalg.solve` on it —
  `O(n^2)` memory and an `O(n^3)` factorization for a system whose every row
  touches three columns. They now store three bands and run the Thomas
  algorithm. At n = 3 200 that is 365 ms and 80.6 MB down to 0.81 ms and
  1.1 MB; a 12 800-point spline wanted 1.3 GB and now takes 3 MB. The
  not-a-knot rows reach outside the band, so its two end moments are
  eliminated into the interior system rather than the band being widened —
  clearing them against the neighbouring row instead would divide by
  `h[1] - h[0]`, which vanishes on a uniform grid. Values agree with the dense
  solve to 6.5e-15 relative across uniform, random, clustered and geometric
  knot spacings.
- **`pde.poisson_2d_direct` assembles block-tridiagonal blocks, not a dense
  matrix.** With the unknowns ordered `i`-fastest, every stencil offset moves
  `j` by at most one, so a node couples only to the row above and below: the
  system is block-tridiagonal with `ny-1` blocks of size `nx-1`. It was being
  stored as one dense `N x N` matrix and handed to `np.linalg.solve` -- 97 MB
  and an `O(N^3)` factorization on a 60x60 grid. It now stores the blocks and
  uses `linalg.block_tridiagonal_solve`: 233.78 ms and 94.8 MB down to 10.74 ms
  and 11.3 MB, agreeing with the dense solve to 1.8e-16 on both the 5- and
  9-point stencils. `laplace_2d`, `poisson_9point` and `helmholtz_2d` inherit
  it. A 90x90 grid, which wanted about 480 MB, now takes 39 MB.
- **`PiecewisePolynomial.__call__` evaluates the whole query at once.** It ran
  one Python iteration per query point, with an inner Horner loop inside that;
  it now gathers one coefficient column at a time and runs Horner over the
  entire query. 28× at 100 000 points. A hand-built ragged coefficient list
  still takes the per-point path.
- **`linalg.qr_algorithm` reduces to Hessenberg form first.** That is an
  orthogonal similarity, so no eigenvalue moves, and Hessenberg form survives a
  QR step — which drops a sweep from `O(n^3)` to `O(n^2)` and lets the
  convergence test read the subdiagonal instead of the whole lower triangle.
  With the compiled kernel, 1 707 ms to 16.8 ms at n = 40. Accuracy improves
  where it converges (n = 20: 2.7e-12 to 4.8e-14); where unshifted QR does not
  converge it still says so, and the docstring now points at
  `shifted_qr_algorithm` and `francis_qr`.
- **The fully implicit Runge-Kutta stage coupling is an array product.**
  `radau_iia`, `gauss_legendre_irk` and `lobatto_iiic` wrote `A K` and `b K` as
  Python sums over the stage index, inside the Newton loop, allocating `s`
  temporaries per stage per residual evaluation. 2.2× on `radau_iia`.
- **`strong_wolfe` no longer re-derives what the caller already knows.** Every
  caller in the package holds `f(x)` and `grad f(x)` from the step that chose
  the direction; they are now passed in as `phi0` and `dphi0` (new optional
  arguments), which with a finite-difference gradient saves `2n + 1` evaluations
  of `f` per line search. The zoom phase also carries `phi(hi)` with the
  bracket rather than re-evaluating an endpoint it had already measured.
  Together: `bfgs` 2.1×, `newton_cg` and `nonlinear_cg` about 12% fewer
  evaluations even with an analytic gradient.
- **`core.utils.as_vector` and `CountedFunction.__call__` have fast paths.**
  Between them they ran 6.5 million times over the test suite. `as_vector`
  returns immediately for an argument that is already a contiguous 1-D
  `float64` array (3.2× on that path, and identical aliasing either way);
  `CountedFunction` skips re-packing single-argument calls.
- `integrate.adaptive_gauss_kronrod` mirrors its half-tabulated node and weight
  arrays once at import instead of on every panel: 2.4× on `quad`.
- `pde.smooth` builds its two red-black masks once per call rather than a fresh
  `np.indices` per half-sweep.
- `stochastic.hamiltonian_mc` carries the current state's log-density between
  iterations instead of recomputing it — one fewer call to the user's
  `log_target` per sample, and the chain is bit-identical.
- `linalg.thomas` no longer allocates an `n`-vector it never reads.
- `ode.adaptive_rk` — and so `solve_ivp`, `dormand_prince`, `rkf45`,
  `cash_karp` and `bogacki_shampine` — runs its stage assembly, error estimate
  and PI step controller in the compiled kernel, calling back into Python for
  the right-hand side. The port is faithful rather than merely equivalent: both
  backends take bit-identical steps, down to the step count, the rejection
  count and the number of right-hand-side evaluations. Measured 4.5–5.9×.
- `linalg.cholesky`, `linalg.plu_decomposition`, `linalg.householder_qr`,
  `linalg.forward_substitution`, `linalg.back_substitution`,
  `linalg.jacobi_eigen`, `transforms.fft`, `transforms.ifft`, `special.gamma`,
  `special.log_gamma`, `special.erf` and `special.erfc` dispatch to the compiled
  kernel when one is present. Results, signatures and exception types are
  unchanged.
- `linalg.jacobi_eigen` is asymptotically faster even before the language
  change: the compiled rotation updates only the two rows and columns it
  touches, where the Python implementation formed a dense rotation matrix and
  multiplied it through — `O(n^3)` per sweep rather than `O(n^5)`. Measured
  381× at n = 80.

### Performance

Thirteen routines whose inner loop ran in Python now do that work in NumPy or
in a compiled kernel. Single-threaded, best of twenty runs, same machine:

| Routine | Workload | Before | After | |
|---|---|---:|---:|---:|
| `linalg.ilu0` | 200x200 | 691 ms | 6.2 ms | 111x |
| `linalg.givens_qr` | 120x120 | 52.6 ms | 0.84 ms | 62x |
| `linalg.ldl_decomposition` | 200x200 | 43.5 ms | 0.72 ms | 61x |
| `linalg.svd_jacobi` | 80x80 | 201 ms | 3.8 ms | 53x |
| `linalg.doolittle`, `crout` | 200x200 | 34 ms | 0.94 ms | 36x |
| `linalg.gauss_jordan` | 150x150 | 13.0 ms | 1.6 ms | 8.2x |
| `stochastic.ks_test` | 20 000 samples | 29.3 ms | 0.23 ms | 127x |
| `stochastic.sobol`, `halton` | 4 096 x 8 | 11.2 ms | 0.69 ms | 16x |
| `transforms.dct` III/IV, `dst` II, `idct` | 2 048 | 37 ms | 0.09 ms | 395x |
| `transforms.dwt2` | 256x256, db4 | 17.7 ms | 2.0 ms | 8.9x |
| `interpolate.bezier` | 40 points, 2 000 samples | 90.1 ms | 5.4 ms | 17x |
| `interpolate.barycentric` | 400 nodes, 5 000 points | 28.4 ms | 4.6 ms | 6.2x |
| `pde.heat_2d_adi` | 48x48, 100 steps | 39.5 ms | 3.8 ms | 10.5x |
| `pde.fem_2d_triangular` | n = 32, assembly only | 22.9 ms | 1.25 ms | 18x |

`dct` types III and IV, `dst` type II and `idct` were transcriptions of their
defining sums: an `O(n^2)` double loop with the outer one in Python, growing
quadratically to 37 ms at n = 2 048 and about 600 ms at n = 8 192. Each is now
one length-`2n` inverse FFT with a twiddle on one or both ends.

Two more follow from the convergence fixes above rather than from any change to
the arithmetic: `jacobi_eigen` 5.0x at n = 120 (14 sweeps rather than 100), and
`svd_golub_kahan` 6.2x through it.

Across a 187-case sweep of the public API the aggregate is 1.60x. Routines
dominated by a user callback -- the ODE and SDE integrators, the optimizers --
are unchanged, because their cost is the caller's function rather than the
framework around it.


The dense factorizations are now built on that `gemm`: Cholesky is right-looking
with two levels of blocking, LU is blocked with recursive panels, and QR uses
the compact WY representation so each panel's trailing update is one `gemm`
rather than `nb` rank-1 updates. Against NumPy's own kernels (multithreaded
OpenBLAS 0.3.31, 12 cores) `householder_qr` and `matmul` are within about 1.2x,
`cholesky` and `solve` within about 1.5-1.9x at n = 1600, and the power-of-two
FFT within about 1.5-2.2x of pocketfft. `jacobi_eigen` remains ~46x off
`numpy.linalg.eigh`, which is close to the ratio of the two algorithms' work:
cyclic Jacobi costs roughly 6-10 n^3 against a tridiagonal reduction's 4n^3/3.

Against the pure-Python fallback:

Measured against the same routine with the backend switched off: `special.gamma`
1225× at 100 000 points, `special.log_gamma` 619×, `linalg.jacobi_eigen` 381× at
n = 80, `transforms.fft` 115× at n = 1024 and 94× at n = 10 000, `linalg.cholesky` 62× at n = 100,
`linalg.solve` 19× at n = 200, `ode.solve_ivp` 4.5–5.9×. Routines whose inner loop was already NumPy — and
so already BLAS — gain a small constant factor instead; `linalg.householder_qr`
is 3.7× at n = 200. The full test suite runs in 27 s rather than 59 s.

### Notes

Against NumPy's own kernels rather than against the Python fallback, the
compiled backend closes most of the gap but does not match native code: the
dense factorizations run 4–15× slower than LAPACK, and the power-of-two FFT is
within 2× of pocketfft while composite lengths are ~24× off. Accuracy is not
the tradeoff — the FFT matches `numpy.fft` to 3e-15 relative at every length
tested. Memory behaves better than speed: the kernels allocate their working
space once, so the compiled path adds no peak RSS over a warm-up call where the
Python path needs a further 0.9-7.7 MB of scratch, and allocates 100-35000x
fewer objects.

### Notes

Kernels were wired in only where they measurably win. `qr_algorithm` was
implemented in Rust, measured at 1.02–1.50× against a NumPy `R @ Q` backed by a
tuned BLAS, and removed again; it kept its Python loop and reached 1.33–2.08×
through the accelerated `householder_qr` underneath. *(Superseded: the ceiling
there was the algorithm, not the language. Reducing to Hessenberg form first
makes a sweep `O(n^2)` instead of `O(n^3)`, and a compiled kernel that owns the
whole iteration then reaches 102× — see `hessenberg_qr_iterate` below.)* The
scalar-only special functions (`digamma`, `erfcx`, the Bessel family) have no
Python loop to eliminate and were left alone.

### Added

- **83 generated figures across the documentation site**, one to seven per
  page: convergence histories, stability regions, sparsity patterns, node
  distributions, spectra, shocks, sampling diagnostics, and the rest. Every one
  of them is drawn by `tools/gen_figures.py` from data the library computes —
  the residual histories are real residual histories, the convergence orders
  are measured by refining a discretization and fitting the slope, the
  stability regions are found by taking one step of the method being described.
  Each figure is rendered twice, once for each colour scheme, and checked in,
  so building the site needs neither Matplotlib nor the minutes the
  computations take.
- **Decision diagrams** on the guides that have to answer "which method?" —
  `solve`'s dispatch, the scalar root finders, the interpolation and
  approximation families, quadrature rules, stiff and non-stiff integrators,
  the three PDE types, and the optimizers — plus the result-record and
  exception hierarchies on the core guide, and the subpackage import graph on
  the home page.
- **`tools/gen_figures.py`** and the `figures` extra
  (`pip install -e ".[figures]"`, Matplotlib only). `--check` reports stale
  figures, `--only REGEX` re-renders a subset, and `--png DIR` writes copies
  for inspection.
- **Six tests covering the figures**: both colour variants exist for every
  figure, every figure a page references exists, every figure that exists is
  shown on the page it was registered for, and the catalogue matches the
  directory.

### Changed

- The test suite is 385 tests, up from 370.

## [1.1.0] - 2026-09-01

The first release published to PyPI, under a new name. The library itself — 836
public functions and classes across 13 subpackages, depending only on NumPy —
was complete before this release; what is new here is the name, the packaging,
the documentation site, and two fixes found while writing that documentation.

### Changed

- **The project is now Quadrivium**, developed until now as `numethods`. The
  quadrivium was the medieval curriculum of the four mathematical arts —
  arithmetic, geometry, music, astronomy — and it abbreviates to *quad*. Three
  things changed with it, all before any public release, so no installed code
  is affected:
  - the distribution is `quadrivium` (`pip install quadrivium`),
  - the import is `import quadrivium` (`import quadrivium as qd` throughout the
    documentation, since `quad` is also the general-purpose integrator),
  - the base exception is `QuadriviumError`, and the rest of the hierarchy is
    unchanged.

### Added

- **Published to PyPI** as `quadrivium`: `pip install quadrivium`. The
  distribution is a pure-Python `py3-none-any` wheel, so no compiler and no
  platform-specific build is involved.
- **Documentation site** at <https://ssmmkk123.github.io/quadrivium/>: a
  getting-started guide covering the conventions shared by every routine, one
  narrative guide per subpackage on choosing between methods that solve the
  same problem, a complete API reference generated from the package itself, and
  pages on design, known limitations, and the release process. Every example on
  every page is a doctest executed by the test suite.
- **`docs` and `dev` extras** (`pip install "quadrivium[dev]"`) for building the
  documentation and cutting a release.
- **Project metadata** for PyPI: homepage, documentation, changelog and issue
  URLs, an expanded classifier and keyword set, and a single source of truth
  for the version, read from `quadrivium.__version__` at build time.
- **Release automation**: a tagged push builds the distributions, checks them
  with `twine`, publishes to PyPI through OIDC Trusted Publishing with no
  stored credentials, and attaches the artifacts and their build attestations
  to a GitHub Release.
- Tests covering the two fixes below, and tests that keep the documentation
  from drifting: every documentation example is executed, and the generated API
  reference is compared against the installed package.

### Fixed

- **`illinois` and `pegasus` now converge superlinearly, as they are meant to.**
  The retained endpoint's function value was being damped on *every* iteration
  rather than only when the same endpoint was kept twice running. That damped
  the good regula-falsi steps too, and both methods degenerated to linear
  convergence with ratio 1/2 — no better than bisection — while Pegasus's
  scaling collapsed onto Illinois's halving, making the two functions return
  identical results. On `x³ − 2x − 5` over `[1, 3]` to `1e-12`, both now take 8
  iterations rather than 41. Roots returned are unchanged; only the number of
  iterations needed to reach them is.
- **`rng=` now accepts an integer seed everywhere it is documented to.**
  `randomized_range_finder`, `randomized_svd`, `randomized_eigh`,
  `nystrom_approximation`, `subspace_iteration`, `lobpcg`, and every SDE
  routine (`brownian_path`, `euler_maruyama`, `milstein`, `gillespie_ssa`,
  `tau_leaping` and the rest) passed the argument straight to a generator
  method, so a seed raised `AttributeError` while `None` and a `Generator`
  worked. They now normalise through `np.random.default_rng`, matching the rest
  of the library.

Earlier versions were developed in the repository and never published to a
package index, so this is the first version installable with `pip`.

[Unreleased]: https://github.com/ssmmkk123/quadrivium/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ssmmkk123/quadrivium/releases/tag/v1.1.0
