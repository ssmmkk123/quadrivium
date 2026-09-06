# Approximation

```python
from quadrivium.approx import remez, pade, aaa, gauss_legendre_nodes
import quadrivium as qd          # qd.polyfit, qd.chebyshev_fit, qd.legendre, ...
```

38 routines in two groups: orthogonal polynomials with their Gauss quadrature
nodes, and fitting — least squares, minimax, rational, and Fourier. Full
signatures are in the [`approx` reference](../api/approx.md).

Approximation differs from [interpolation](interpolate.md) in what it promises:
an interpolant passes through every point, an approximation minimises an error
measure over the whole interval. Which measure — least squares, uniform,
weighted — is the choice this subpackage is organised around.

## Fitting data

```pycon
>>> from quadrivium import numeric as np
>>> import quadrivium as qd
>>> x = np.array([0.0, 1.0, 2.0, 3.0])
>>> y = x**2 + x + 1
>>> [round(float(c), 10) for c in qd.polyfit(x, y, degree=2)]
[1.0, 1.0, 1.0]

```

Coefficients come back highest degree first, so `numeric.polyval` evaluates them.

| Data or goal | Function |
| --- | --- |
| polynomial least squares | `polyfit` |
| known per-point variances | `weighted_polyfit` |
| high degree, needs conditioning | `chebyshev_fit`, `legendre_fit` |
| `y = a·e^{bx}` | `exponential_fit` |
| `y = a·x^b` | `power_fit` |
| `y = a + b·ln x` | `logarithmic_fit` |
| a ratio of polynomials | `rational_fit`, `aaa` |
| periodic data | `trigonometric_fit`, `fourier_series` |
| smooth curve through noisy data | `spline_fit` |

```mermaid
flowchart TD
    A["a function, or data"] --> B{"what is being minimised?"}
    B -- "average error" --> C{"data or function?"}
    C -- "noisy data" --> D["polyfit<br/>weighted_polyfit"]
    C -- "a function" --> E["chebyshev_fit<br/>orthogonal_series_fit"]
    B -- "worst error" --> F["remez<br/>chebyshev_economization"]
    B -- "a Taylor series at a point" --> G["pade"]
    B -- "poles or steep gradients" --> H["aaa, rational_fit"]
    B -- "periodic structure" --> I["fourier_series<br/>trigonometric_fit"]
```

Above degree six or so, fitting in the monomial basis loses digits to
conditioning — the Vandermonde matrix of a fine grid has a condition number
that grows exponentially in the degree. Fitting in an orthogonal basis does
not:

```pycon
>>> t = np.linspace(-1, 1, 200)
>>> vals = np.exp(t)
>>> cheb = qd.chebyshev_fit(t, vals, degree=12)
>>> float(np.max(np.abs(cheb(t) - vals))) < 1e-12
True

```

## Orthogonal polynomials

Six families, each evaluated by its stable three-term recurrence rather than by
an explicit formula:

```pycon
>>> from quadrivium.approx import legendre, chebyshev_t, hermite_physicists, laguerre
>>> round(float(legendre(3, 0.5)), 12)             # P₃(x) = (5x³ - 3x)/2
-0.4375
>>> round(float(chebyshev_t(5, np.cos(0.7))), 12) == round(float(np.cos(5 * 0.7)), 12)
True

```

<figure markdown="span">
  ![Legendre and Chebyshev polynomials of the first six degrees](../assets/figures/approx-orthogonal-families.svg#only-light)
  ![Legendre and Chebyshev polynomials of the first six degrees](../assets/figures/approx-orthogonal-families-dark.svg#only-dark)
  <figcaption>Both families are evaluated by their three-term recurrence, not by an explicit formula: the recurrence is stable where the formula is not. Legendre polynomials are orthogonal against weight 1, Chebyshev against 1/√(1−x²), which is why their extrema crowd differently.</figcaption>
</figure>

Orthogonality is the property they are for, and it holds numerically:

```pycon
>>> from quadrivium.approx import gauss_legendre_nodes
>>> nodes, weights = gauss_legendre_nodes(20)
>>> inner = float(np.sum(weights * legendre(3, nodes) * legendre(5, nodes)))
>>> abs(inner) < 1e-14                             # ⟨P₃, P₅⟩ = 0
True
>>> norm = float(np.sum(weights * legendre(3, nodes)**2))
>>> abs(norm - 2/7) < 1e-14                        # ⟨P₃, P₃⟩ = 2/(2n+1)
True

```

`recurrence_coefficients` gives the three-term coefficients for any family, and
`golub_welsch` turns them into quadrature nodes and weights by solving a
symmetric tridiagonal eigenproblem — the algorithm behind every
`gauss_*_nodes` function here:

| Nodes | Weight function | Interval |
| --- | --- | --- |
| `gauss_legendre_nodes` | `1` | `[a, b]` |
| `gauss_chebyshev_nodes` | `1/√(1−x²)` | `[−1, 1]` |
| `gauss_hermite_nodes` | `e^{−x²}` | `(−∞, ∞)` |
| `gauss_laguerre_nodes` | `x^α e^{−x}` | `[0, ∞)` |
| `gauss_jacobi_nodes` | `(1−x)^α(1+x)^β` | `[−1, 1]` |
| `gauss_lobatto_nodes` | `1`, endpoints included | `[a, b]` |
| `gauss_radau_nodes` | `1`, one endpoint included | `[a, b]` |

<figure markdown="span">
  ![Node positions and weights for five families of quadrature nodes](../assets/figures/approx-gauss-nodes.svg#only-light)
  ![Node positions and weights for five families of quadrature nodes](../assets/figures/approx-gauss-nodes-dark.svg#only-dark)
  <figcaption>Marker area is the weight the node carries. Every Gauss family clusters its nodes near the ends of the interval, which is the same clustering that cures Runge's phenomenon, and none of them is equally spaced.</figcaption>
</figure>

`orthogonal_series_fit` expands a function in any of these families directly.

## Minimax approximation

Least squares minimises the average error; the minimax (uniform) polynomial
minimises the worst error, which is what you want when the approximation
carries a guarantee. The Remez exchange algorithm computes it, and the answer
is characterised by equioscillation — the error attains its maximum magnitude
with alternating sign at `degree + 2` points:

```pycon
>>> from quadrivium.approx import remez
>>> best = remez(np.exp, 4, -1, 1)
>>> grid = np.linspace(-1, 1, 2001)
>>> err = best(grid) - np.exp(grid)
>>> minimax_error = float(np.max(np.abs(err)))
>>> ls = qd.polyfit(grid, np.exp(grid), degree=4)
>>> ls_error = float(np.max(np.abs(np.polyval(ls, grid) - np.exp(grid))))
>>> minimax_error < ls_error          # smaller worst-case, by construction
True

```

<figure markdown="span">
  ![The minimax error equioscillates; the least squares error does not](../assets/figures/approx-remez-equioscillation.svg#only-light)
  ![The minimax error equioscillates; the least squares error does not](../assets/figures/approx-remez-equioscillation-dark.svg#only-dark)
  <figcaption>Approximating exp on [−1, 1] by a degree-4 polynomial. The Remez error touches its maximum magnitude, with alternating sign, at degree + 2 points — the property that characterises the minimax polynomial and the one the exchange algorithm drives towards.</figcaption>
</figure>

`chebyshev_economization` is the cheap approximation to the same idea: expand
in Chebyshev polynomials, drop the highest terms, and the error added is
exactly the size of the dropped coefficients, spread evenly.

## Rational approximation

A rational function approximates far better than a polynomial of the same total
degree when the target has poles or steep gradients. Padé matches a Taylor
series term for term:

```pycon
>>> from quadrivium.approx import pade, pade_evaluate
>>> taylor = [1.0, 1.0, 0.5, 1/6, 1/24]            # e^x to x⁴
>>> num, den = pade(taylor, 2, 2)
>>> at = 0.5
>>> pade_err = abs(float(pade_evaluate(num, den, at)) - float(np.exp(at)))
>>> taylor_err = abs(float(np.polyval(taylor[::-1], at)) - float(np.exp(at)))
>>> pade_err < taylor_err / 3          # same coefficients, better answer
True

```

<figure markdown="span">
  ![A Pade approximant against the Taylor polynomial it was built from](../assets/figures/approx-pade.svg#only-light)
  ![A Pade approximant against the Taylor polynomial it was built from](../assets/figures/approx-pade-dark.svg#only-dark)
  <figcaption>Both use exactly the same five Taylor coefficients of exp. Writing them as a ratio rather than a sum extends the useful range by an order of magnitude in the error, at no extra information cost.</figcaption>
</figure>

Padé is a local approximation, built from derivatives at one point. For a
global one from sampled values, AAA is the modern method: it places its support
points greedily where the error is worst and keeps everything in barycentric
form, which stays stable where an explicit ratio of polynomials would not:

```pycon
>>> from quadrivium.approx import aaa
>>> z = np.linspace(-1.2, 1.2, 400)
>>> r, support, values, weights = aaa(np.tan, z, tol=1e-12)
>>> float(np.max(np.abs(r(z) - np.tan(z)))) < 1e-10
True
>>> len(support) < 25                              # few terms for that accuracy
True

```

<figure markdown="span">
  ![Rational approximation against polynomial approximation of tan](../assets/figures/approx-rational-vs-polynomial.svg#only-light)
  ![Rational approximation against polynomial approximation of tan](../assets/figures/approx-rational-vs-polynomial-dark.svg#only-dark)
  <figcaption>tan has poles just outside the interval, and a polynomial has to spend its degree imitating them. AAA places poles where the function has them and reaches 1e-12 with sixteen coefficients, where the polynomial is still at 1e-4 with twenty-seven.</figcaption>
</figure>

## Fourier approximation

```pycon
>>> from quadrivium.approx import fourier_series
>>> square = lambda t: np.sign(np.sin(t))
>>> approx = fourier_series(square, n=25)
>>> abs(float(approx(1.0)) - 1.0) < 0.1            # away from the jump
True

```

The Gibbs phenomenon is real and does not go away with more terms — the
overshoot at a jump converges to about 9% of the jump height however many
harmonics you add. `trigonometric_fit` fits harmonics to sampled data, and
`fourier_coefficients` returns the coefficients themselves.

<figure markdown="span">
  ![Gibbs' phenomenon: the overshoot at a jump does not shrink](../assets/figures/approx-gibbs.svg#only-light)
  ![Gibbs' phenomenon: the overshoot at a jump does not shrink](../assets/figures/approx-gibbs-dark.svg#only-dark)
  <figcaption>Adding harmonics narrows the ringing but does not lower it: the first overshoot converges to 8.95% of the jump height. It is a property of the truncated series, not of the arithmetic.</figcaption>
</figure>

## Pitfalls

- **Least squares in the monomial basis is ill-conditioned.** Use
  `chebyshev_fit` or `legendre_fit` above degree six.
- **Minimax is not always what you want.** It is the right criterion when a
  worst-case bound matters; least squares is right when the data is noisy.
- **Padé approximants can have spurious poles** inside the region of interest,
  from near-cancellation in the coefficients. Check the denominator's roots
  (`qd.polynomial_roots`), or use `aaa`, which is built to avoid them.
- **A truncated Fourier series overshoots at jumps.** No number of terms fixes
  it; filter the coefficients or accept the ringing.
- **Extrapolation is not approximation.** Every method here is fitted on an
  interval and says nothing outside it.

## See also

- [`approx` API reference](../api/approx.md) — every signature.
- [Interpolation guide](interpolate.md) — passing through the points exactly.
- [Integration guide](integrate.md) — the Gauss rules these nodes drive.
- [Transforms guide](transforms.md) — the FFT route to Fourier coefficients.
