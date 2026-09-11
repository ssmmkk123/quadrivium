# Interpolation: values between samples

Interpolation constructs a function passing through supplied data. It answers
what a chosen model predicts between observations; matching the observations
exactly does not establish that the prediction is accurate. The distribution
of nodes, smoothness of the underlying function, and constraints such as
monotonicity matter more than simply choosing a high polynomial degree.

Quadrivium provides global polynomials, local splines, rational interpolants,
regular-grid methods, scattered-data methods, and parametric curves. Most
constructors return a callable. Splines commonly return `PiecewisePolynomial`,
which also supports derivatives, integrals, and roots. See the
[interpolation reference](../api/interpolate.md) for individual return types.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import interpolate as ip
>>> x = np.array([0.0, 1.0, 2.0, 3.0])
>>> y = np.array([0.0, 1.0, 1.5, 2.0])
>>> interpolant = ip.pchip(x, y)
>>> np.allclose(interpolant(x), y)
True
>>> 1.0 < float(interpolant(1.5)) < 1.5
True

```

## Choose what the curve must preserve

| Data and requirement | Starting point | Important limitation |
| --- | --- | --- |
| Piecewise linear behavior is sufficient | `linear_spline` | Derivative jumps at knots |
| Smooth data, smooth curvature desired | `cubic_spline` | Can overshoot between samples |
| Monotone data must remain monotone | `pchip` | Usually only continuously differentiable |
| Function and derivatives known at knots | `hermite_spline` | Supplied derivatives determine local shape |
| Local response to irregular samples | `akima_spline` | Not a general monotonicity guarantee |
| Smooth callable, freely chosen nodes | `chebyshev_interpolation` | Accuracy still depends on smoothness |
| Fixed nodes, global polynomial needed | `barycentric` | Stable evaluation cannot fix poor nodes |
| Oscillatory behavior with known period | Trigonometric/Fourier interpolation | Period and sampling must be consistent |
| Cartesian product grid | `regular_grid_interpolator` | Array axes must match grid order |
| Scattered planar samples | `LinearNDInterpolator` | Explicit convex-hull policy is needed |
| Scattered samples in several dimensions | `RBFInterpolator` | Kernel, scaling, and neighbors affect conditioning |
| Noisy observations | Smoothing or [approximation](approx.md) | Exact interpolation also fits noise |

Use a low-complexity method first and compare it on withheld or newly sampled
points. Extrapolation beyond the sampled domain is a different problem and
requires additional assumptions about the underlying function.

## Prepare one-dimensional data

Use matching finite coordinate and value vectors. Polynomial interpolation
requires distinct abscissae. Most spline constructors sort coordinates and
values together and reject repeated knots. Supplying already sorted data
makes the intended ordering explicit and avoids ambiguity with additional
arrays such as weights or derivative samples.

At least two points are needed for a line segment; more specialized methods
may require additional knots. Duplicate measurements at the same coordinate
should be aggregated or treated as a fitting problem rather than silently
assigned different interpolated values.

```pycon
>>> order = np.argsort([2.0, 0.0, 1.0])
>>> sorted_x = np.array([2.0, 0.0, 1.0])[order]
>>> sorted_y = np.array([4.0, 0.0, 1.0])[order]
>>> sorted_x.tolist(), sorted_y.tolist()
([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])

```

Coordinates with very different magnitudes can degrade polynomial or radial
basis conditioning. Map a one-dimensional fitting interval to a moderate
range such as `[-1, 1]`; for multidimensional distances, nondimensionalize
coordinates using physically meaningful scales.

## Global polynomial interpolation

`lagrange`, `newton_divided_differences`, and `barycentric` represent the same
unique polynomial for distinct nodes and the same values, up to numerical
error. They differ in representation and evaluation cost. Barycentric form is
a practical default because it avoids explicitly expanding a polynomial into
monomial coefficients.

```pycon
>>> nodes = np.array([-1.0, 0.0, 1.0])
>>> values = nodes**2
>>> polynomial = ip.barycentric(nodes, values)
>>> np.allclose(polynomial([-0.5, 0.25]), [0.25, 0.0625])
True
>>> newton_form = ip.newton_divided_differences(nodes, values)
>>> np.allclose(newton_form([-0.5, 0.25]), polynomial([-0.5, 0.25]))
True

```

`lagrange_coefficients` and `vandermonde_interpolation` produce explicit
monomial coefficients. This is useful for algebraic manipulation but can be
poorly conditioned at high degree. `neville` evaluates through a recursive
table at specified query points; forward and backward Newton forms are
intended for equally spaced nodes. Hermite polynomial interpolation also
matches supplied derivatives.

### Node placement can dominate algorithm choice

A high-degree polynomial on equally spaced nodes can oscillate near the
endpoints even when the data come from a smooth function. This is the Runge
phenomenon. Replacing the evaluator alone does not fix it: the polynomial
itself can be a poor approximation.

Chebyshev nodes cluster toward the endpoints. In this submodule,
`chebyshev_nodes(n, ...)` returns **n nodes** in increasing order.
`kind=1` gives roots of a Chebyshev polynomial and excludes the endpoints;
`kind=2` includes the endpoints when `n>1`.

```pycon
>>> runge = lambda t: 1.0 / (1.0 + 25.0*t*t)
>>> chebyshev = ip.chebyshev_interpolation(runge, n=25, a=-1, b=1)
>>> query = np.linspace(-1, 1, 201)
>>> float(np.max(np.abs(chebyshev(query) - runge(query)))) < 0.01
True

```

<figure markdown="span">
  ![Off-node polynomial interpolation errors using equally spaced and Chebyshev nodes](../assets/figures/interpolate-node-choice.svg#only-light)
  ![Off-node polynomial interpolation errors using equally spaced and Chebyshev nodes](../assets/figures/interpolate-node-choice-dark.svg#only-dark)
  <figcaption>Barycentric interpolants of 1/(1+25x²) use 5 to 25 equally spaced or Chebyshev nodes. Maximum errors are measured on 2,001 independent query points; the 17-node profile shows where the discrepancy occurs. Matching the interpolation nodes cannot reveal this off-node error.</figcaption>
</figure>

Chebyshev nodes improve polynomial approximation for many smooth functions;
they do not make a discontinuity smooth or provide a universal error bound.
If sampling is under your control and a fixed degree is inconvenient, consider
[`approx.chebfun`](approx.md#adaptive-piecewise-chebyshev-approximation).

## Cubic splines and boundary conditions

A cubic spline uses one cubic polynomial per interval and imposes continuity
conditions at interior knots. It avoids the high global degree of a polynomial
through all samples. Boundary conditions determine the remaining endpoint
behavior and can materially affect the first and last intervals.

| Constructor or `bc` | Endpoint assumption |
| --- | --- |
| `natural_cubic_spline`, `bc="natural"` | Second derivative is zero at both endpoints |
| `clamped_cubic_spline`, `bc="clamped"` | First derivatives `dy0` and `dyn` are supplied |
| `not_a_knot_spline`, default `bc="not-a-knot"` | First two and last two pieces share their cubic behavior |
| `periodic_cubic_spline`, `bc="periodic"` | Values and derivatives match across the period boundary |

Natural does not mean universally best: zero endpoint curvature is an
assumption. If endpoint slopes are known from physics or an analytic model,
clamped conditions use that information directly.

```pycon
>>> knots = np.array([0.0, 0.5, 1.0])
>>> cubic = ip.clamped_cubic_spline(knots, knots**3, dy0=0.0, dyn=3.0)
>>> np.allclose(cubic([0.25, 0.75]), [0.25**3, 0.75**3])
True
>>> np.allclose(cubic.derivative()([0.25, 0.75]), [3*0.25**2, 3*0.75**2])
True

```

For periodic splines, include both endpoint coordinates and matching endpoint
values. This differs from FFT sampling, where the duplicated endpoint is
excluded. `periodic_cubic_spline` checks endpoint agreement with a tolerance.

## Shape-preserving interpolation and smoothing

PCHIP chooses slopes to preserve monotonicity of monotone data on each
interval. It is useful for cumulative quantities, tabulated material
properties, and other data where overshoot would be misleading.

```pycon
>>> monotone = ip.pchip(x, y)
>>> dense = np.linspace(x[0], x[-1], 101)
>>> bool(np.all(np.diff(monotone(dense)) >= -1e-12))
True

```

A continuously differentiable PCHIP curve need not have a continuous second
derivative. A cubic spline offers smoother curvature but does not guarantee
monotonicity or positivity between knots. Akima interpolation reduces some
oscillatory behavior through local slope estimates, but it is not a universal
shape-preserving replacement.

If values are noisy, exact interpolation preserves that noise. A smoothing
spline balances weighted squared residuals against integrated squared
curvature. `smoothing_spline(x, y, lam=..., weights=...)` uses positive weights
and a nonnegative smoothing parameter; a larger `lam` penalizes roughness
more strongly. Select smoothing from measurement uncertainty or validation,
not from visual smoothness alone.

The smoothing implementation assembles dense matrices, so it is not a
constant-memory method for arbitrarily large datasets. Sort all inputs
consistently before passing weights. The [approximation guide](approx.md)
explains fitting and held-out error checks.

## Work with `PiecewisePolynomial`

Spline objects evaluate scalars or arrays and expose their knots as `.x`.
`coeffs[i]` contains coefficients in **ascending powers of the local variable**
`t - x[i]`. This is different from the descending global coefficients returned
by ordinary polynomial-fitting routines.

`derivative(order)` returns another piecewise polynomial.
`antiderivative()` returns a continuous antiderivative whose value is zero at
the first knot. `integrate(a, b)` computes the integral of the represented
spline, and `roots()` searches its individual pieces inside knot intervals.

```pycon
>>> line = ip.linear_spline([0.0, 1.0, 2.0], [-1.0, 1.0, 3.0])
>>> np.allclose(line.derivative()([0.25, 1.25]), [2.0, 2.0])
True
>>> round(line.integrate(0.0, 2.0), 12)
2.0
>>> line.roots().tolist()
[0.5]

```

An integral computed exactly from the piecewise coefficients still has the
modeling error of the spline. Likewise, a root of the interpolant need not be
an equally accurate root of the original function. Re-evaluate the original
model at important derived values.

Splines extrapolate with their first or last polynomial by default. Disable
that behavior when values outside the knots should be rejected:

```pycon
>>> line.extrapolate = False
>>> from quadrivium.core import DomainError
>>> try:
...     line(-0.1)
... except DomainError:
...     print("Query is outside the sampled interval.")
Query is outside the sampled interval.

```

`roots()` does not describe an entire interval of zeros as an infinite root
set. Treat identically zero pieces and near-multiple roots as special cases
when interpreting its output.

## Rational and periodic interpolants

Rational interpolation represents a ratio of polynomials or a barycentric
rational form. Floater-Hormann uses local degree `d` to build a rational
interpolant on ordered real nodes; Thiele uses reciprocal differences.
A rational model can describe behavior that needs a high-degree polynomial,
but denominator zeros and ill-conditioned data can produce poles.

```pycon
>>> rational = ip.floater_hormann(x, y, d=2)
>>> np.allclose(rational(x), y)
True

```

`trigonometric_interpolation` fits periodic data using a specified or inferred
period. `fourier_interpolation` resamples a uniformly sampled periodic signal
by Fourier padding. Verify the implied period and endpoint convention before
using either method; a mismatch creates an artificial discontinuity. More
samples cannot recover frequencies already aliased in the original data.

## Regular grids and scattered data

For `regular_grid_interpolator(grids, V)`, the value array has shape
`tuple(len(grid) for grid in grids)`. Axis zero corresponds to the first grid,
axis one to the second, and so on. Queries are coordinate rows with one column
per dimension. Use strictly increasing finite axes and at least two points
per axis for linear interpolation.

```pycon
>>> gx = np.array([0.0, 1.0, 2.0])
>>> gy = np.array([0.0, 1.0])
>>> V = gx[:, None] + 2*gy[None, :]
>>> grid_model = ip.regular_grid_interpolator([gx, gy], V)
>>> np.allclose(grid_model([[0.5, 0.25], [1.5, 0.5]]), [1.0, 2.5])
True

```

Linear regular-grid interpolation extends the boundary cell outside the grid;
nearest-neighbor mode selects the nearest grid location. Neither automatically
provides a physical extrapolation model. Validate query bounds yourself.
The `bilinear`, `bicubic`, and `trilinear` helpers provide related lower-dimensional
interfaces with their own documented argument layouts.

`LinearNDInterpolator` and `Delaunay` currently operate on **planar** points,
despite the general-looking class name. The interpolator uses triangles inside
the convex hull and offers `outside="fill"`, `"nearest"`, or `"raise"`.
The default fills outside queries with `NaN`.

```pycon
>>> points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
>>> planar = ip.LinearNDInterpolator(points, [0.0, 1.0, 2.0], outside="raise")
>>> abs(float(planar([0.25, 0.25])) - 0.75) < 1e-12
True

```

`RBFInterpolator` supports scattered coordinates with a configurable kernel,
shape parameter `epsilon`, smoothing, and polynomial reproduction.
`neighbors=k` restricts each local fit and uses a bounded cache; `None` builds
a global fit. Local neighbor changes can create derivative discontinuities.
Neighbor sets must contain enough geometrically independent points for the
polynomial basis. `KDTree` exposes neighbor queries separately.

## Parametric curves and next steps

Bézier, B-spline, rational Bézier, and NURBS routines represent geometric curves.
Control points guide shape; they are not generally observations the curve must
interpolate. De Casteljau evaluation and subdivision support Bézier curves;
NURBS add rational weights and knot vectors. Use `nurbs_circle` for a rational
circle construction and inspect the documented parameter interval.

Validate interpolation with off-node errors, shape constraints, and boundary
behavior before differentiating or integrating the result. Continue with
[differentiation](diff.md), [integration](integrate.md), and
[approximation](approx.md) for those downstream operations.
