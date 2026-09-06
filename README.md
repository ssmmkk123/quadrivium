# Quadrivium

[![PyPI](https://img.shields.io/pypi/v/quadrivium.svg)](https://pypi.org/project/quadrivium/)
[![Python versions](https://img.shields.io/pypi/pyversions/quadrivium.svg)](https://pypi.org/project/quadrivium/)
[![CI](https://github.com/ssmmkk123/quadrivium/actions/workflows/ci.yml/badge.svg)](https://github.com/ssmmkk123/quadrivium/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue.svg)](https://ssmmkk123.github.io/quadrivium/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive, from-scratch library of the numerical methods used in
scientific computing. Every algorithm is written out explicitly — LU
factorization loops over its pivots, the FFT does its own bit reversal, the
Hungarian algorithm walks its own augmenting paths — so the method itself is
readable rather than hidden behind a compiled call.

**836 public functions and classes across 13 subpackages.
No runtime dependencies: the arrays are its own, written in C.**

*The quadrivium was the medieval curriculum of the four mathematical arts —
arithmetic, geometry, music, astronomy — the complete education in number.
It abbreviates to **quad**.*

📖 **[Documentation](https://ssmmkk123.github.io/quadrivium/)** ·
[Getting started](https://ssmmkk123.github.io/quadrivium/getting-started/) ·
[Guides](https://ssmmkk123.github.io/quadrivium/guides/linalg/) ·
[API reference](https://ssmmkk123.github.io/quadrivium/api/) ·
[Changelog](CHANGELOG.md)

## Installation

```bash
pip install quadrivium
```

Python 3.9 or newer, a C compiler if you build from source, and nothing else.

The package computes on its own array type, `quadrivium.numeric`, compiled from
the C sources in `csrc/`. It is a drop-in replacement for the subset of the
NumPy API this library used to depend on -- the same names, the same
signatures, the same semantics -- so code that says

```python
from quadrivium import numeric as np
```

reads exactly as it did before. Arrays are strided and N-dimensional over
`bool`, `int64`, `float64` and `complex128`, with broadcasting, views, the full
indexing grammar, dense linear algebra, transforms and a PCG64 generator. They
export the buffer protocol, so if you already have NumPy you can pass NumPy
arrays straight in and read the results back with `numpy.asarray`.

Wheels also bundle an optional Rust extension of accelerated kernels (see
[Performance](#performance)). That one is genuinely optional: if no wheel
matches your platform, `pip` builds from source, and if a Rust toolchain is not
present the install still succeeds and those routines run their readable Python
implementations. Set `QUADRIVIUM_NO_RUST=1` to skip it deliberately.

For an editable development installation with the test and documentation
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Quick start

The documentation imports the package as `qd` rather than `quad`, so that the
library's own general-purpose integrator stays legible as `qd.quad(...)`.

```python
from quadrivium import numeric as np
import quadrivium as qd

qd.brent(lambda x: x**3 - 2*x - 5, 1, 3).root      # 2.0945514815423265
qd.quad(lambda x: np.exp(-x*x), -np.inf, np.inf)   # sqrt(pi), to 5e-15
qd.solve_ivp(lambda t, y: -2*y, (0, 1), [1.0]).y[-1, 0]
qd.minimize(rosenbrock, [-1.2, 1.0], method="bfgs")

qd.milstein(drift, diffusion, (0, 1), [1.0])       # SDEs, strong order 1
qd.wavedec(signal, "db4", level=5)                 # wavelets
qd.sylvester(A, B, C)                              # A X + X B = C in O(n^3)
qd.gragg_bulirsch_stoer(f, (0, 10), y0)            # extrapolation
qd.newton_krylov(F, x0, precond=M)                 # Jacobian-free Newton
qd.aaa(f, points)                                  # rational approximation
```

Results come back as small records — `RootResult`, `QuadratureResult`,
`ODESolution`, `OptimizeResult`, `EigenResult`, `PDESolution`, `IterationResult` —
carrying the answer together with iteration counts, residual histories,
convergence flags and a message explaining what happened:

```pycon
>>> import quadrivium as qd
>>> result = qd.brent(lambda x: x**3 - 2*x - 5, 1, 3)
>>> round(result.root, 12), result.converged, result.method
(2.094551481542, True, 'brent')

```

Stochastic routines take `rng=` — an integer seed or a generator —
so every run is reproducible.

## What is included

| Area | Methods |
|---|---|
| **`linalg`** | Gaussian elimination (4 pivoting strategies), LU/PLU/complete-pivot, Doolittle, Crout, Cholesky, LDL', QR (Gram-Schmidt, modified GS, Householder, Givens), Hessenberg, bidiagonalization, Thomas, banded and block-tridiagonal solvers, Sherman-Morrison, Woodbury · power/inverse/shifted/Rayleigh iteration, deflation, QR algorithm (plain, shifted, Francis double-shift), Jacobi eigenvalue, Lanczos, Arnoldi, Sturm bisection, SVD (one-sided Jacobi, Golub-Kahan), polar, Schur, Gershgorin, matrix exponential and functions · Jacobi, Gauss-Seidel, SOR, SSOR, Richardson, Chebyshev, steepest descent, CG, PCG, MINRES, GMRES(m), BiCG, BiCGSTAB, CGS, CGNR, LSQR, ILU(0), incomplete Cholesky, SSOR preconditioning · normal equations, QR/SVD least squares, pseudoinverse, ridge, Tikhonov, TSVD, total least squares, NNLS, equality-constrained LS · COO/CSR/CSC/DIA sparse formats, reverse Cuthill-McKee · **matrix functions**: `sqrtm` (scaled Denman-Beavers), `logm` (inverse scaling and squaring), matrix sign · **matrix equations**: Sylvester and Lyapunov by Bartels-Stewart, discrete Lyapunov by doubling, Riccati by Newton-Kleinman · Hager 1-norm condition estimation · **randomized**: range finder, randomized SVD/eigh, Nyström, interpolative and CUR decompositions · subspace iteration, LOBPCG, symmetric-definite generalized eigenproblem, QZ |
| **`rootfind`** | Bisection, false position, Illinois, Pegasus, Ridders, Brent, ITP, secant, Newton (damped, multiplicity-corrected), Halley, Chebyshev, Steffensen, Muller, inverse quadratic, fixed point, Aitken · Newton and damped Newton for systems, Broyden good/bad, Wolfe-Bittner secant, nonlinear Gauss-Seidel, continuation/homotopy, dogleg · Horner, synthetic division, deflation, Durand-Kerner, Aberth-Ehrlich, Bairstow, Laguerre, companion matrix, Sturm sequences, root bounds · **Anderson acceleration** of fixed-point iterations, **Jacobian-free Newton-Krylov** with restarted matrix-free GMRES and optional preconditioning |
| **`interpolate`** | Lagrange, Newton divided differences, forward/backward differences, Neville, barycentric, Hermite, Chebyshev, Vandermonde · linear/quadratic/cubic splines (natural, clamped, not-a-knot, periodic), Hermite, PCHIP, Akima, B-splines, Catmull-Rom, cardinal, tension, smoothing splines · Thiele continued fractions, Bulirsch-Stoer, Floater-Hormann, trigonometric, band-limited resampling · bilinear, bicubic, tensor-product, N-D regular grid, Shepard/IDW, RBF (7 kernels), kriging, barycentric triangles · **Bézier** curves (de Casteljau, exact derivative curves, subdivision), rational Bézier, **NURBS** with de Boor evaluation |
| **`approx`** | Legendre, Chebyshev T/U, Hermite (both conventions), Laguerre, Jacobi, Gegenbauer; Golub-Welsch nodes; Gauss-Legendre/Chebyshev/Hermite/Laguerre/Jacobi/Lobatto/Radau · polynomial, weighted, Chebyshev-basis and Legendre-basis fitting, exponential/power/logarithmic/rational fits, Padé approximants, Remez exchange (minimax), Fourier series and trigonometric fitting · **AAA** near-best rational approximation in barycentric form, Chebyshev economization |
| **`diff`** | Forward/backward/central differences to 4th order, five-point stencils, **Fornberg** arbitrary-order weights, differentiation matrices on non-uniform grids, Richardson extrapolation, complex-step, Savitzky-Golay · **automatic differentiation**: forward-mode `Dual`, reverse-mode `Variable` tape, `HyperDual` for exact Hessians · Fourier and Chebyshev spectral differentiation, Clenshaw evaluation |
| **`integrate`** | Riemann, midpoint, trapezoid, Simpson 1/3 and 3/8, Boole, arbitrary-degree Newton-Cotes, Romberg, Euler-Maclaurin · adaptive Simpson/trapezoid/Gauss-Kronrod, `quad` with infinite-limit transformations · Gauss-Legendre/Chebyshev/Hermite/Laguerre/Jacobi/Lobatto/Radau/Kronrod, Clenshaw-Curtis, Fejér, **tanh-sinh** (with an endpoint-offset form for full precision on singularities) · Monte Carlo, stratified, importance, control variates, antithetic, quasi-MC (Halton/Sobol/LHS), VEGAS, hit-or-miss · double/triple integrals with variable bounds, tensor-product cubature, triangle and tetrahedron rules, polar and spherical · **Filon** quadrature for oscillatory integrands, **Cauchy principal values** and Hadamard finite parts, **Smolyak sparse grids** |
| **`ode`** | Euler, Heun, midpoint, Ralston, RK3, RK4, RK-3/8, arbitrary Butcher tableaux; embedded adaptive RKF45, Cash-Karp, Dormand-Prince, Bogacki-Shampine with PI step control and dense output · backward Euler, trapezoidal, implicit midpoint, θ-method, Gauss-Legendre IRK, Radau IIA, Lobatto IIIC, SDIRK, ESDIRK, TR-BDF2, BDF1-6, Rosenbrock · Adams-Bashforth 1-6, Adams-Moulton 1-5, predictor-corrector, Nyström, Milne-Simpson, variable-step Adams · symplectic Euler, leapfrog, velocity/position Verlet, Ruth3, Forest-Ruth, Yoshida4, PEFRL · exponential Euler, ETD-RK2/RK4, exponential Rosenbrock, Magnus, Krylov `expm` action · shooting, multiple shooting, linear shooting, finite-difference BVP, collocation, Galerkin, Sturm-Liouville · **Gragg-Bulirsch-Stoer** extrapolation with adaptive order, modified midpoint, Richardson extrapolation of any fixed-step method · **event location** on the dense output, with direction filtering and terminal events · **Runge-Kutta-Nyström** and Störmer-Cowell for second-order systems · **index-1 DAEs** by BDF, singular mass matrices, **delay equations** by the method of steps · stiffness detection |
| **`pde`** | Heat FTCS/BTCS/Crank-Nicolson/θ, 2-D ADI, method of lines, reaction-diffusion, advection-diffusion · wave (explicit and implicit), upwind, Lax-Friedrichs, Lax-Wendroff, Beam-Warming, MacCormack, leapfrog, TVD with 7 flux limiters, Godunov for Burgers · Poisson/Laplace/Helmholtz (5-point and 4th-order 9-point Mehrstellen), Jacobi/Gauss-Seidel/SOR/CG iterations, FFT fast solver, Neumann problems · **multigrid** V-cycle, W-cycle, full multigrid · 1-D P1/P2 and 2-D triangular finite elements · finite volume with Rusanov/HLL/exact Riemann fluxes and MUSCL reconstruction · Fourier and Chebyshev spectral solvers, Kuramoto-Sivashinsky · **WENO3/WENO5** reconstruction with **SSP-RK2/RK3** time stepping · incompressible **Navier-Stokes** by Chorin projection, vorticity-streamfunction pseudo-spectral solver, lid-driven cavity |
| **`optimize`** | Golden section, Fibonacci, ternary, parabolic, Brent, 1-D Newton · Armijo, Goldstein, Wolfe, strong Wolfe (interpolating zoom), exact line search · gradient descent, momentum, Nesterov, AdaGrad, RMSProp, Adam, nonlinear CG (FR/PR/HS), Barzilai-Borwein · Newton, modified Newton, BFGS, DFP, SR1, Broyden class, L-BFGS · trust region with Cauchy/dogleg/Steihaug-CG subproblems, Gauss-Newton, Levenberg-Marquardt, `curve_fit` · Nelder-Mead, Powell, Hooke-Jeeves, compass/pattern search, coordinate descent · simulated annealing, particle swarm, differential evolution, genetic algorithm, basin hopping, **CMA-ES**, dual annealing · penalty, log-barrier, augmented Lagrangian, projected gradient, SQP, active-set QP, KKT residuals · ISTA, FISTA, ADMM, Douglas-Rachford, LASSO, ridge, elastic net · simplex (two-phase, Big-M, Bland's rule), primal-dual interior point, Hungarian assignment · **truncated Newton (Newton-CG)** with forcing sequences and negative-curvature handling, **L-BFGS-B** with bound constraints |
| **`transforms`** | Direct DFT, radix-2 Cooley-Tukey, Bluestein chirp-z, mixed-radix, real FFT, 2-D FFT, DCT-I/II/III/IV, DST-I/II, Hartley · convolution (direct and FFT), correlation, autocorrelation, Wiener deconvolution, periodogram, Welch, spectrogram, Hilbert transform, resampling, low-pass filtering, **18 window functions** (Kaiser, flat-top, Blackman-Harris, Nuttall, Bohman, Parzen, …) in symmetric and periodic forms · **wavelets**: DWT/IDWT for 11 orthogonal families (Haar, Daubechies, Symlet, Coiflet), multi-level, 2-D, stationary (shift-invariant) transform, universal-threshold denoising, continuous wavelet transform with Morlet and Ricker, Goertzel |
| **`stochastic`** | LCG, Park-Miller, xorshift, **MT19937**, middle-square, van der Corput, Halton, Sobol, Latin hypercube, dual-lattice spectral test · inverse transform, Box-Muller, Marsaglia polar, rejection, **adaptive rejection (Gilks-Wild)**, ratio of uniforms, Walker alias table, multivariate normal · Metropolis-Hastings, random-walk Metropolis, Gibbs, **HMC**, NUTS-lite, slice sampling, parallel tempering, ESS, autocorrelation time, Gelman-Rubin · descriptive statistics, Welford, linear/polynomial/logistic regression, PCA, KDE, bootstrap, jackknife, permutation tests, t/χ²/KS/ANOVA · **SDE solvers**: Euler-Maruyama, Milstein (explicit and drift-implicit), stochastic Heun and Runge-Kutta, order-1.5 additive-noise Taylor, tamed Euler · Brownian paths and bridges, exact GBM/Ornstein-Uhlenbeck samplers, CIR with full truncation · **Gillespie SSA** and tau-leaping for reaction networks |
| **`special`** | Gamma, log-gamma, digamma, **polygamma**, beta, incomplete gamma and beta, erf/erfc/erfinv, **erfcx**, **Dawson**, **Fresnel** integrals, Bessel J/Y/I/K to arbitrary integer order, **spherical Bessel**, Airy, complete elliptic integrals, exponential integrals E_n, sine/cosine integrals, **Riemann zeta on the whole real line** (Euler-Maclaurin, Borwein, functional equation), **Lambert W** (both real branches), **confluent and Gauss hypergeometric**, **associated Legendre and spherical harmonics**, Struve H0 |

## Documentation

The [documentation site](https://ssmmkk123.github.io/quadrivium/) carries the
material this README only summarizes:

- **[Getting started](https://ssmmkk123.github.io/quadrivium/getting-started/)** —
  the conventions every routine shares: calling patterns, the seven result
  records, tolerances, how failure is reported, reproducible randomness, and an
  honest account of performance.
- **[Guides](https://ssmmkk123.github.io/quadrivium/guides/linalg/)** — one per
  subpackage, each opening with a table and a decision diagram that map a
  situation to a method, then working through the choice with runnable
  examples and 83 figures.
- **[API reference](https://ssmmkk123.github.io/quadrivium/api/)** — all 836
  public names with real signatures, generated from the package itself.
- **[Design and validation](https://ssmmkk123.github.io/quadrivium/design/)** and
  **[known limitations](https://ssmmkk123.github.io/quadrivium/limitations/)**.

Every example on those pages is a doctest executed by the test suite, and the
API reference is regenerated and compared against the package, so neither can
drift from the code. The figures are generated the same way, by
`tools/gen_figures.py`, which runs the method being illustrated and plots what
it returns — a stability region is measured by taking one step of the method,
a convergence order by refining the grid, a shock by capturing it.

## Performance

The latest [performance and reliability report](PERFORMANCE.md) records
reproducible timing and memory measurements for compact least squares,
sparse operations, density estimation, and transforms.

The algorithms are written out in Python so they can be read. That makes the
hot ones slow: a Cholesky factorization that loops over its own pivots in the
interpreter is two orders of magnitude off a compiled one. So the kernels that
dominate runtime also exist as compiled Rust, and the Python routine calls
whichever backend is present.

```python
>>> import quadrivium as qd
>>> print(qd.accel.show_config())  # doctest: +SKIP
quadrivium acceleration: rust (extension 1.2.0)
  compiled kernels (20): adaptive_rk, back_substitution, cholesky, erf, erfc,
  fft, forward_substitution, gamma, hessenberg_qr_iterate, householder_qr,
  ifft, is_symmetric, jacobi_eigen, lid_driven_cavity, log_gamma, matmul,
  plu, qr_least_squares, sor_poisson, thomas
```

Measured on this machine (n is the problem size; "python" is the same routine
with the compiled backend switched off):

| routine | size | python | rust | speedup |
|---|---|---|---|---|
| `special.gamma` | 100 000 | 442.10 ms | 0.367 ms | **1203×** |
| `special.log_gamma` | 100 000 | 277.77 ms | 0.238 ms | **1165×** |
| `transforms.fft` | 10 000 | 130.51 ms | 0.279 ms | **468×** |
| `linalg.jacobi_eigen` | 80 | 1508.29 ms | 3.882 ms | **389×** |
| `transforms.fft` | 1 024 | 1.38 ms | 0.012 ms | **115×** |
| `linalg.cholesky` | 200 | 17.10 ms | 0.275 ms | **62×** |
| `linalg.solve` | 200 | 17.88 ms | 0.320 ms | **56×** |
| `special.erfc` | 100 000 | 10.15 ms | 0.313 ms | **32×** |
| `linalg.householder_qr` | 400 | 94.25 ms | 7.153 ms | **13×** |
| `linalg.plu_decomposition` | 400 | 34.47 ms | 3.148 ms | **11×** |
| `ode.solve_ivp` (Lorenz) | 3 | 128.66 ms | 29.73 ms | **4.3×** |

The kernels above replace a loop over array *elements*. A second group replaces
a loop over *iterations* — the sweep, the step, the relaxation — where the
whole solve moves across the boundary once instead of thousands of times:

| routine | size | python | rust | speedup |
|---|---|---|---|---|
| `pde.poisson_2d_iterative` (Gauss-Seidel) | 30 | 783.94 ms | 6.897 ms | **114×** |
| `linalg.qr_algorithm` | 40 | 1226.99 ms | 12.062 ms | **102×** |
| `pde.poisson_2d_iterative` (SOR) | 40 | 142.36 ms | 1.522 ms | **94×** |
| `linalg.thomas` | 50 000 | 33.11 ms | 0.520 ms | **64×** |
| `pde.heat_crank_nicolson` | 800 x 400 | 224.95 ms | 6.148 ms | **37×** |
| `pde.lid_driven_cavity` | 21 | 238.19 ms | 14.554 ms | **16×** |

`thomas` carries the implicit PDE solvers with it, since they all reduce to a
tridiagonal solve per step: `heat_btcs` 42×, `heat_crank_nicolson` 29×,
`wave_implicit` 23×, `heat_2d_adi` 9.6×.

Reproduce with `python tools/bench_accel.py`.

### Speedups that are not the backend

Some of the cost was the algorithm rather than the language, and those wins
apply with the extension switched off as well.

| routine | size | before | after | |
|---|---|---|---|---|
| `interpolate.cubic_spline` | 3 200 | 365 ms, 80.6 MB | 0.81 ms, 1.1 MB | **453×**, 75× less memory |
| `interpolate.natural_cubic_spline` | 2 000 | 113.98 ms, 31.6 MB | 0.53 ms, 0.6 MB | **217×**, 53× less memory |
| `pde.poisson_2d_direct` | 60 x 60 | 233.78 ms, 94.8 MB | 10.74 ms, 11.3 MB | **22×**, 8× less memory |
| `PiecewisePolynomial.__call__` | 100 000 points | 56.47 ms | 2.02 ms | **28×** |
| `ode.radau_iia` | 100 steps | 42.8 ms | 19.7 ms | **2.2×** |
| `optimize.lbfgs` (Rosenbrock, no gradient) | 6 | 114 619 evaluations | 1 107 | **104× fewer** |

The pattern in most of these is a matrix that was denser than the problem. The
splines were building a dense `(n+1)²` matrix and factorizing it for a system
that is tridiagonal; they now solve it in `O(n)` time and memory, which is what
takes a 12 800-point spline from unusable to 14 ms. `poisson_2d_direct` was
doing the same to a system that is block-tridiagonal. `PiecewisePolynomial` was
evaluating one query point per Python iteration, and `radau_iia` wrote its
stage coupling `A K` as a Python sum over the stage index. L-BFGS was
recomputing `f` and `grad f` at a point it had just left — and then failing to
notice it had converged, which is the next section.

### How this compares to NumPy's own kernels

The table above is quadrivium against itself. The harder question is how it
compares to a library that has been binding LAPACK for thirty years. NumPy here
is linked against a multithreaded OpenBLAS 0.3.31 on a 12-core machine.

| operation | size | quadrivium | numpy (OpenBLAS / pocketfft) | |
|---|---|---|---|---|
| `householder_qr` | 1600 | 231.3 ms | 198.9 ms | 1.16× slower |
| `matmul` | 1600 | 41.3 ms | 35.1 ms | 1.18× slower |
| `householder_qr` | 800 | 32.0 ms | 26.9 ms | 1.19× slower |
| `cholesky` | 1600 | 32.1 ms | 21.1 ms | 1.52× slower |
| `fft` n=1024 (power of two) | | 0.017 ms | 0.011 ms | 1.54× slower |
| `matmul` | 800 | 6.64 ms | 4.14 ms | 1.60× slower |
| `cholesky` | 800 | 9.27 ms | 4.95 ms | 1.87× slower |
| `solve` | 1600 | 38.1 ms | 19.6 ms | 1.94× slower |
| `fft` n=262144 (power of two) | | 7.16 ms | 3.23 ms | 2.22× slower |
| `cholesky` | 400 | 1.99 ms | 0.71 ms | 2.82× slower |
| `fft` n=100000 (composite) | | 3.91 ms | 0.92 ms | 4.24× slower |
| `jacobi_eigen` | 160 | 65.4 ms | 1.42 ms | 46× slower |

Reproduce with `python tools/bench_accel.py --native`. Warm the BLAS thread pool
before trusting any single row: OpenBLAS pays its pool startup on first use, and
an unwarmed run reports NumPy as slower than it is.

**Where the remaining gaps are, and why.**

*Dense factorizations* run between parity and about 3× off LAPACK. They are
built on a packed `gemm` in `rust/src/gemm.rs` with a hand-written AVX2/FMA
micro-kernel holding a 6×8 tile of `C` in registers, cache blocking over all
three dimensions, and rayon across row bands. Single-threaded it reaches about
46 GFLOPS, roughly 82% of this core's peak, so the kernel itself is close to
the machine; the shortfall against OpenBLAS is thread scaling and the fixed cost
of packing at small `n`. Everything else follows from it: Cholesky is
right-looking with two levels of blocking, LU is blocked with recursive panels,
and QR uses the compact WY representation so the trailing update is a `gemm`
rather than `nb` rank-1 updates.

*The FFT* is within about 2× of pocketfft on power-of-two lengths and about 4×
on composite ones, with agreement to 3 × 10⁻¹⁵ relative at every length tested.
Composite lengths recurse with a shared twiddle table and closed-form radix
2/3/4/5 butterflies; what is still missing against pocketfft is an iterative
formulation with better locality than a strided recursion.

*`jacobi_eigen` is the one large gap, and it is algorithmic rather than an
implementation defect.* Cyclic Jacobi does roughly `6-10 n^3` of work; LAPACK's
`eigh` reduces to tridiagonal form and costs about `4n^3/3` plus a cheap QL
sweep. The library offers Jacobi because it is the more accurate method on
graded matrices, and the ratio to `eigh` is close to the ratio of the two
algorithms' work. Rewriting the rotation to exploit symmetry was tried and
measured no faster -- the cost there is strided column access, not multiply
count -- so the literal form was kept.

**If you want the fastest possible dense linear algebra, call LAPACK.** What
this library offers is the algorithm written out where you can read it, now
running within a small multiple of the tuned version rather than a hundred.

### Switching backends

Both implementations stay in the tree and are tested against each other, so the
readable version is never stale:

```python
from quadrivium import accel

accel.available()          # True when the compiled backend is active
accel.backend()            # 'rust' or 'python'

with accel.disabled():     # force the reference implementation
    L = qd.cholesky(A)     # ... same answer, more slowly
```

`QUADRIVIUM_NO_ACCEL=1` in the environment does the same thing process-wide.
`tests/test_accel.py` runs both paths over sizes that straddle the kernels'
blocking thresholds and requires them to agree to floating-point noise, and to
raise the same exceptions on singular and indefinite input.

### Memory

The compiled kernels allocate their working space once and reuse it. The Python
implementations allocate inside their loops — `np.outer(v, v @ R[k:, k:])`
builds a fresh matrix on every QR step; the Jacobi sweep built a whole n × n
rotation matrix per rotation. Peak resident memory over three runs, and the
Python heap traffic underneath it:

| case | peak RSS growth: rust / python | Python heap allocated: rust / python |
|---|---|---|
| `cholesky` n=1000 | 0 KB / 0 KB | 480 B / 8.0 MB |
| `householder_qr` n=1000 | 0 KB / 7.7 MB | 696 B / 24.1 MB |
| `fft` n=262144 | 0 KB / 2.2 MB | 288 B / 12.6 MB |
| `solve_ivp` (van der Pol) | 0 KB / 0.9 MB | 4.6 KB / 320 KB |

The result arrays are the same size either way, so this is not a smaller
footprint for the answer — it is the disappearance of the transient working set.
The compiled path never grew peak RSS beyond what the warm-up call had already
reached; the Python path needed another 0.9–7.7 MB of scratch on top, and
allocated between 100× and 35 000× more objects to get there.

The larger memory wins came from choosing a smaller problem to solve. A cubic
spline through *n* points assembles a system that couples each knot only to its
two neighbours; building that as a dense `(n+1)²` matrix costs `O(n²)` memory
for an `O(n)` problem, and at 12 800 points it wanted 1.3 GB. Storing the three
bands instead brings it to 3 MB, and a 50 000-point spline — previously out of
reach on any ordinary machine — builds in 47 ms.

### Reporting convergence

A solver that finds the answer and reports failure is worse than a slow one,
because nothing downstream can tell the difference. Two cases were fixed:

- **The quasi-Newton family** (`bfgs`, `lbfgs`, `dfp`, `sr1`) tested only
  `‖grad f‖ < tol`, with `tol = 1e-10` by default. When the gradient is a
  central difference — which it is whenever the caller does not supply one —
  the gradient is only resolved to about `eps^(2/3)`, so that test sits *below*
  the noise floor and can never be met. L-BFGS on a 6-dimensional Rosenbrock
  sat on the minimizer for all 1 000 iterations, spent 114 619 function
  evaluations, and returned `converged=False` at `f = 1.3e-16`. The methods now
  also stop when a full step no longer changes the objective to relative
  precision `ftol` (new argument, default `1e-12`): 41 iterations, 1 107
  evaluations, `converged=True`, the same minimizer.
- **`gradient_descent` and the adaptive first-order methods** (`adam`,
  `rmsprop`, `adagrad`, `momentum`, `nesterov`) ran to `max_iter` after
  overflowing to NaN, appending a NaN iterate to `history` each time, and
  reported "maximum iterations reached" — which reads like a tolerance nearly
  met rather than a step size that has to be reduced. They now stop at the
  overflow and say so, naming the step size.

## Design

**Every claim is tested against something independent** — an analytic solution,
an exact identity, a convergence rate, or a reference implementation. The suite
checks properties, not just values:

- convergence *orders* (RK4 improves 16× per halving, Boole 64×, BDF6 64×)
- exactness where theory demands it (Gauss rules to degree 2n−1, P1 finite
  elements nodally exact in 1-D, spectral methods to machine precision)
- structural identities (Bessel Wronskians, Parseval, symplectic energy drift,
  mass conservation, partition of unity, KKT conditions)
- known failure modes: FTCS blows up above r = 1/2, Lax-Wendroff oscillates at
  a discontinuity while TVD limiters do not, RANDU's triples are caught lying
  on 15 planes, explicit RK4 goes unstable on a stiff problem where Radau IIA
  does not.
- published benchmarks: the lid-driven cavity reproduces Ghia, Ghia & Shin
  (1982) to within 1% at Re = 100; randomized SVD attains the Eckart-Young
  optimum to four digits; Gillespie's algorithm reproduces the exact binomial
  law of a death process (χ² = 16.5 on 18 bins).

Some checks pin a method down by a property no table of constants could:
Daubechies-N wavelets must annihilate polynomials of degree below N, and do;
the Neumann Poisson solver must be second order, and is only because its
boundary uses ghost points; every flux limiter must lie in Sweby's TVD region,
and they do — two of them did not until the sign was fixed.

**Failures are reported, not hidden.** A diverging iteration returns
`converged=False` with an explanation rather than raising on overflow; a line
search that stalls at floating-point precision says so; FTCS refuses an
unstable step size and tells you how many steps you need.

**Numerical care is explicit.** Where the naive formula is wrong the code says
why: Welford's algorithm instead of `E[x²]−E[x]²`; the augmented-matrix φ
functions instead of a cancelling Taylor series (φ₁(−40) comes out exact where
the series returns −1.7 × 10⁷); overflow-free `sech` weights in tanh-sinh;
Bland's rule on degenerate simplex pivots; a positive-definiteness check before
dogleg trusts a Newton step; Kummer's transformation applied to *every* negative
argument of ₁F₁, because the alternating series loses 2|z| nepers before it
loses none; Rybicki's method for Dawson's function, where both obvious routes
overflow; the projection method's FFT inverting the symbol of the exact
difference operators it is paired with, so `div u` comes out at 10⁻¹⁶ rather
than 10⁻⁵.

**Where a limitation is real, it is documented rather than papered over.** The
Cauchy-point trust region genuinely converges only linearly; stochastic Heun
converges to the Stratonovich solution and so does *not* converge to the Itô
one; Störmer-Cowell's familiar three-step coefficients are third order despite
being widely quoted as fourth; unpreconditioned Newton-Krylov needs more Krylov
steps as a PDE mesh is refined — which is why `precond=` exists, and why it
takes n = 1000 Bratu from 102,101 residual evaluations to 21. The full list is
in [known limitations](https://ssmmkk123.github.io/quadrivium/limitations/).

## Examples

```bash
python examples/01_linear_algebra.py     # factorizations, eigenvalues, Krylov
python examples/02_calculus.py           # differentiation and quadrature
python examples/03_differential_equations.py
python examples/04_optimization.py
python examples/05_pde_and_transforms.py
python examples/06_extended_methods.py   # SDEs, wavelets, matrix equations, WENO
```

Each script prints the numbers that justify what it demonstrates — convergence
ratios, residual norms, iteration counts — rather than plotting anything.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development workflow and the validation a numerical change needs: a
convergence order, an exactness result, a conservation law, or an identity —
not a value the code itself produced. Please report security issues according
to [SECURITY.md](SECURITY.md).

Maintainers cutting a release should follow
[the release process](https://ssmmkk123.github.io/quadrivium/release-process/).

## License

Quadrivium is available under the [MIT License](LICENSE).
