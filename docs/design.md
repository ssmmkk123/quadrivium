# Design and validation

The library rests on four commitments. They are what distinguish it from a
thin wrapper around a compiled library, and they are why its tests look the
way they do.

## 1. The algorithm stays visible

Every method is written out at the level a textbook states it. LU
factorization loops over its pivots, the FFT does its own bit reversal, the
Hungarian algorithm walks its own augmenting paths, Bartels-Stewart reduces to
Schur form and back-substitutes block by block.

`quadrivium.numeric` supplies array storage and the primitives underneath — a
matrix product, an element-wise exponential — but never the method itself.
There is no call to `numeric.linalg.solve` inside `quadrivium.linalg.solve`,
and no call to `numeric.fft` inside `quadrivium.transforms.fft`.

That array layer is the package's own, written in C under `csrc/`: strided
N-dimensional arrays over four dtypes, broadcasting element-wise operations,
NumPy's indexing grammar, its pairwise summation, dense factorisations,
transforms and a PCG64 generator. It replaced a NumPy dependency and keeps
NumPy's semantics deliberately, down to reproducing its random streams bit for
bit, so results carry over unchanged.

The consequence is that reading the source is a way to learn the method, and
modifying it is a way to test a claim about it. The cost is speed; see
[Performance](getting-started.md#performance).

## 2. Every claim is tested against something independent

341 tests of the numerics, all passing, and none of them merely records what
the code currently returns. Each checks the answer against something the code cannot
have got wrong the same way:

**Analytic solutions.** Heat diffusion against `e^{−απ²t} sin(πx)`, the
harmonic oscillator against its closed form, Poisson against a manufactured
solution.

**Convergence orders.** Halving the step must divide the error by the right
factor — RK4 by 16, Boole's rule by 64, BDF6 by 64, cubic splines by 16. An
implementation that is subtly one order low passes a value check and fails
this one.

<figure markdown="span">
  ![Convergence orders measured across the library](assets/figures/design-measured-orders.svg#only-light)
  ![Convergence orders measured across the library](assets/figures/design-measured-orders-dark.svg#only-dark)
  <figcaption>Every bar pair is one method's promised order beside the order measured by refining the discretization and fitting the slope. These are computed when this page is built, by the same code the test suite runs on every commit.</figcaption>
</figure>

```pycon
>>> from quadrivium import numeric as np
>>> import quadrivium as qd
>>> exact = float(np.exp(-1.0))
>>> e1 = abs(float(qd.rk4(lambda t, y: -y, (0, 1), [1.0], n=20).y[-1, 0]) - exact)
>>> e2 = abs(float(qd.rk4(lambda t, y: -y, (0, 1), [1.0], n=40).y[-1, 0]) - exact)
>>> round(e1 / e2)                       # fourth order: 2⁴
16

```

**Exactness where theory demands it.** An `n`-point Gauss rule integrates
every polynomial of degree `2n − 1` exactly. P1 finite elements are nodally
exact for the 1-D Poisson problem. Spectral differentiation of a smooth
periodic function is exact to machine precision. These are not approximate
checks with a generous tolerance — they hold to `1e-15`.

**Structural identities.** Bessel Wronskians, Parseval's theorem, the
partition of unity for B-splines, `P + Q = 1` for the regularized incomplete
gamma functions, symplectic energy conservation, mass conservation in
conservation laws, the KKT conditions at a constrained optimum.

**Known failure modes, reproduced deliberately.** FTCS blows up above
`r = 1/2`. Lax-Wendroff oscillates at a discontinuity while a TVD limiter does
not. RANDU's triples are caught lying on 15 planes by the spectral test.
Explicit RK4 goes unstable on a stiff problem where Radau IIA does not. A
method that cannot fail in the way theory says it should is not the method it
claims to be.

**Published benchmarks.** The lid-driven cavity reproduces Ghia, Ghia and Shin
(1982) to within 1% at Re = 100. Randomized SVD attains the Eckart-Young
optimum to four digits. Gillespie's algorithm reproduces the exact binomial
law of a death process (χ² = 16.5 on 18 bins).

Some checks pin a method down by a property that no table of constants could.
Daubechies-N wavelets must annihilate polynomials of degree below N, and do.
The Neumann Poisson solver must be second order, and is only because its
boundary uses ghost points. Every flux limiter must lie in Sweby's TVD region,
and they do — two of them did not until a sign was fixed.

## 3. Failures are reported, not hidden

A diverging iteration returns `converged=False` with an explanation rather
than raising on overflow. A line search that stalls at floating-point
precision says so. FTCS refuses an unstable step size and states how many
steps would be needed instead.

```pycon
>>> r = qd.rootfind.fixed_point(lambda x: 2*x, 1.0, max_iter=50)
>>> r.converged, r.message
(False, 'maximum iterations reached')

```

The rule is that an exception means *this input cannot be worked with*, while
a result record with `converged=False` means *this method did not get there*.
The second is information, and throwing it away as an exception would destroy
the partial answer and the history that explain why.

## 4. Numerical care is explicit

Where the naive formula is wrong, the code uses the stable one and says why in
a comment or docstring:

- **Welford's algorithm** for variance, instead of `E[x²] − E[x]²`, which
  cancels catastrophically when the mean is large relative to the spread.
- **Augmented-matrix φ functions** for exponential integrators, instead of a
  cancelling Taylor series: `φ₁(−40)` comes out exact where the series returns
  `−1.7 × 10⁷`.
- **Overflow-free `sech` weights** in tanh-sinh quadrature, where the direct
  expression overflows before the weights underflow.
- **Bland's rule** on degenerate simplex pivots, which is what prevents
  cycling.
- **A positive-definiteness check** before the dogleg step trusts a Newton
  direction.
- **Kummer's transformation** applied to *every* negative argument of `₁F₁`,
  because the alternating series loses about `2|z|` nepers before it converges.
- **Rybicki's method** for Dawson's function, where both obvious routes
  overflow.
- **The projection method's FFT** inverting the symbol of the exact difference
  operators it is paired with, so `div u` comes out at `10⁻¹⁶` rather than
  `10⁻⁵`.

## 5. Real limitations are documented

Where a method is genuinely weaker than its reputation, the docstring says so
rather than leaving you to find out. The Cauchy-point trust region converges
only linearly. Stochastic Heun converges to the Stratonovich solution and
therefore does *not* converge to the Itô one. Störmer-Cowell's familiar
three-step coefficients are third order despite being widely quoted as fourth.
Unpreconditioned Newton-Krylov needs more Krylov steps as a mesh is refined.

These are collected in [Known limitations](limitations.md).

## The figures are generated too

Every figure on this site is drawn by `tools/gen_figures.py` from data the
library computes: the residual histories are real residual histories, the
convergence orders are measured, the shock was captured by the solver being
described. A figure is registered next to the page it belongs to, rendered
once for each colour scheme, and checked in — so building the site needs
neither Matplotlib nor the minutes it takes to solve every problem shown.

```bash
python -m pip install -e ".[figures]"   # Matplotlib, only for regenerating
python tools/gen_figures.py             # rewrite every figure
python tools/gen_figures.py --check     # verify the checked-in ones
```

The test suite checks that every figure a page references exists and that
every figure that exists is referenced by the page it was registered for, so
a renamed method cannot leave a stale picture behind.

## Testing your own changes

The same standard applies to contributions. A new method needs evidence beyond
one expected value: a convergence order, an exactness property, a conservation
law, an identity, or agreement with an independent analytic solution. See
[Contributing](contributing.md).

```bash
python -m pytest -q      # numerics, backend, memory and documentation checks
python -m pytest                          # same suite under pytest
```

The documentation is tested too: every example on these pages is a doctest run
by `tests/test_docs.py`, and the API reference is regenerated from the package
and compared, so neither can drift from the code.
