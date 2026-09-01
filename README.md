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

**836 public functions and classes across 13 subpackages. 370 tests, all passing.
Depends only on NumPy.**

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

Python 3.9 or newer, NumPy 1.20 or newer, and nothing else. The distribution is
a pure-Python wheel, so there is no compiler and no platform-specific build
involved.

For an editable development installation with the test and documentation
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m unittest discover -s tests
```

## Quick start

The documentation imports the package as `qd` rather than `quad`, so that the
library's own general-purpose integrator stays legible as `qd.quad(...)`.

```python
import numpy as np
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

Stochastic routines take `rng=` — an integer seed or a `numpy.random.Generator` —
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
  subpackage, each opening with a table that maps a situation to a method, then
  working through the choice with runnable examples.
- **[API reference](https://ssmmkk123.github.io/quadrivium/api/)** — all 836
  public names with real signatures, generated from the package itself.
- **[Design and validation](https://ssmmkk123.github.io/quadrivium/design/)** and
  **[known limitations](https://ssmmkk123.github.io/quadrivium/limitations/)**.

Every example on those pages is a doctest executed by the test suite, and the
API reference is regenerated and compared against the package, so neither can
drift from the code.

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
