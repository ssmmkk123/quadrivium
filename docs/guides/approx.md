# Approximation and fitting

Approximation replaces a function or dataset with a simpler representation.
Unlike interpolation, a fit can deliberately leave residuals at the supplied
samples. This is useful when measurements contain noise, evaluations are
expensive, or the goal is a compact model that can be differentiated,
integrated, and evaluated repeatedly.

The central choice is the error you want to control: squared error at samples,
maximum error over an interval, relative error after a transformation, or a
sample-based adaptive tolerance. These are different objectives and can
produce different models from the same data.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import approx as ap
>>> x = np.linspace(-1.0, 1.0, 9)
>>> y = 1.0 + 2.0*x + 3.0*x*x
>>> coefficients = ap.polyfit(x, y, degree=2)
>>> np.allclose(coefficients, [3.0, 2.0, 1.0])
True

```

The [approximation reference](../api/approx.md) lists constructors, coefficient
conventions, and available polynomial families. See [interpolation](interpolate.md)
when matching every supplied value is an explicit requirement.

## Choose an objective and representation

| Goal | Starting point | What the method controls |
| --- | --- | --- |
| Low-degree polynomial fit to observations | `polyfit` | Sum of squared sample residuals |
| Unequal observation reliability | `weighted_polyfit` | Weighted squared sample residuals |
| Polynomial fit at a higher degree | `chebyshev_fit`, `legendre_fit` | Sample least squares in a scaled basis |
| Smooth callable, automatically chosen complexity | `chebfun` | Coefficient-tail and off-grid sample checks |
| Uniform error over a finite interval | `remez`, `minimax_polynomial` | Approximate equioscillation search |
| Local series with rational continuation | `pade` | Agreement of Taylor coefficients |
| Rational model from values | `rational_fit`, `aaa` | Linearized or sampled rational approximation |
| Periodic model | `fourier_series`, `trigonometric_fit` | Fourier projection or sample least squares |
| Noisy smooth curve | `spline_fit` | Fidelity balanced against curvature |

A method that fits the samples best may predict worst between them. Keep
training residuals and independent validation errors separate, and decide
whether extrapolation is part of the intended use before choosing the model.

## Polynomial least squares

`polyfit(x, y, degree)` returns coefficients in **descending powers**, suitable
for `np.polyval`. It constructs a Vandermonde matrix and solves with QR rather
than explicitly forming the normal equations.

```pycon
>>> prediction = np.polyval(coefficients, [-0.5, 0.5])
>>> np.allclose(prediction, [0.75, 2.75])
True
>>> training_residual = np.polyval(coefficients, x) - y
>>> float(np.max(np.abs(training_residual))) < 1e-12
True

```

Use matching finite vectors and enough independent sample locations for the
requested degree. Repeated coordinates are acceptable in least squares when
the overall design retains the needed rank, but many repeats do not create
new information about polynomial shape.

A large domain or a high degree can make powers of `x` poorly scaled even
with QR. Center and scale the coordinate, or use an orthogonal basis.
QR improves the numerical solve; it cannot resolve ambiguity in an
underconstrained or noisy model.

### Weights describe the squared-error objective

`weighted_polyfit` uses weights in
`sum(weights[i] * (prediction[i] - y[i])**2)`. Internally the design matrix and
observations are multiplied by square-root weights. If known standard
deviations are `sigma`, inverse-variance weights are `1/sigma**2`, subject to
the assumptions of the measurement model.

```pycon
>>> weighted = ap.weighted_polyfit([0, 1, 2], [1, 3, 5], [1, 4, 1], degree=1)
>>> np.allclose(weighted, [2.0, 1.0])
True

```

Use finite nonnegative weights and avoid zero total information. Weights do
not provide automatic outlier rejection, confidence intervals, or a robust
loss. For a different loss function, construct the objective explicitly and
use the [optimization tools](optimize.md).

## Orthogonal bases and coordinate scaling

Chebyshev and Legendre fitting map the specified domain to `[-1, 1]` and fit
a linear combination of basis functions. The returned object is callable and
stores `.coefficients` in increasing basis degree. Those are basis
coefficients, not monomial coefficients: do not pass them to `np.polyval`.

```pycon
>>> model = ap.chebyshev_fit(x, y, degree=2, domain=(-1.0, 1.0))
>>> np.allclose(model([-0.5, 0.5]), [0.75, 2.75])
True
>>> model.coefficients.shape
(3,)
>>> legendre_model = ap.legendre_fit(x, y, degree=2, domain=(-1.0, 1.0))
>>> np.allclose(legendre_model(x), y)
True

```

When `domain` is omitted, the data's minimum and maximum define it. A zero-width
domain is unsuitable. Save the scaling interval with the model if storing
coefficients for later use. Evaluation outside that interval extrapolates the
basis expansion and can grow rapidly.

An orthogonal polynomial family is orthogonal relative to a continuous weight
and domain. Arbitrarily located sample columns are not automatically orthogonal
in the discrete least-squares problem. The basis often improves conditioning,
but the sample distribution still matters.

## Choose complexity using independent errors

Degree is a model-complexity parameter. A low degree may miss real curvature;
a high degree may reproduce measurement noise and develop large boundary
excursions. Compare candidate models at locations that did not determine their
coefficients.

```pycon
>>> rng = np.random.default_rng(5)
>>> training_x = np.linspace(-1, 1, 25)
>>> training_y = np.sin(training_x) + 0.01*rng.standard_normal(training_x.size)
>>> fitted = ap.chebyshev_fit(training_x, training_y, degree=3, domain=(-1, 1))
>>> validation_x = np.linspace(-0.95, 0.95, 40)
>>> rmse = float(np.sqrt(np.mean((fitted(validation_x) - np.sin(validation_x))**2)))
>>> rmse < 0.02
True

```

Here the exact function is available because the data are synthetic. For real
measurements, hold out observations or collect a separate validation dataset.
If model selection repeatedly uses the same holdout, retain a final independent
set for the ultimate assessment.

<figure markdown="span">
  ![Training error and error against the underlying function as polynomial degree changes](../assets/figures/approx-fit-generalization.svg#only-light)
  ![Training error and error against the underlying function as polynomial degree changes](../assets/figures/approx-fit-generalization-dark.svg#only-dark)
  <figcaption>Chebyshev fits of degrees 1 to 16 use 25 noisy sine samples with Gaussian noise σ=0.08 and seed 20260911. Training RMS is compared with RMS against the noise-free function at 501 query points; the curve profiles show how excess degree can follow noise instead of the underlying function.</figcaption>
</figure>

Report the error statistic and its domain. Maximum absolute error emphasizes
the worst sampled location; root-mean-square error summarizes typical squared
error; relative error needs care near zeros. None of these is a certified
continuous-domain bound when measured only on a finite grid.

## Adaptive piecewise Chebyshev approximation

`chebfun(f, domain, ...)` constructs a `ChebyshevApproximation` of a finite
real-valued scalar callable on a finite interval. It increases polynomial
degree, checks coefficient tails and independent off-grid probes, and bisects
difficult pieces when increasing degree is insufficient.

```pycon
>>> adaptive = ap.chebfun(np.exp, domain=(-1.0, 1.0), atol=1e-12, rtol=1e-10)
>>> adaptive.converged
True
>>> float(np.max(np.abs(adaptive(x) - np.exp(x)))) < 1e-9
True
>>> adaptive.function_calls > 0 and len(adaptive.pieces) >= 1
True

```

`atol` and `rtol` set the sample-based target. `max_degree` bounds the degree
of each attempted piece and must be at least 16; `max_pieces` bounds subdivision.
If the budget is exhausted before the criterion is satisfied,
`.converged` is false and the constructed approximation remains available.
Inspect `.error_estimate`, `.function_calls`, and the piece count alongside
that flag.

The error estimate is based on sampled evidence. A narrow feature can escape
all sample points. Validate at additional locations, particularly around known
singularities, transitions, or rapid oscillations. Providing a tighter
tolerance alone does not certify behavior between samples.

### Calculus on the representation

The adaptive object supports evaluation, `derivative(order)`, `integrate(a,b)`,
and `roots()`. Evaluation and integration limits must remain within the
construction domain; out-of-domain queries raise rather than extrapolate.

```pycon
>>> derivative = adaptive.derivative()
>>> abs(derivative(0.25) - np.exp(0.25)) < 1e-8
True
>>> abs(adaptive.integrate() - (np.e - 1/np.e)) < 1e-9
True
>>> quadratic = ap.chebfun(lambda t: t*t - 0.25, domain=(-1, 1))
>>> np.allclose(quadratic.roots(), [-0.5, 0.5], atol=1e-8)
True

```

Differentiation amplifies coefficient errors. The derivative object does not
supply a fresh derivative-error certificate; its error estimate is unavailable.
Roots are roots of the represented pieces, deduplicated near shared boundaries.
An identically zero piece has infinitely many roots and raises an error.
Check important roots against the original function using [root finding](rootfind.md).

## Transformed elementary models

`exponential_fit` fits `y = a*exp(b*x)` after taking `log(y)` and requires
strictly positive responses. `power_fit` fits `y = a*x**b` in log-log space and
requires positive coordinates and responses. `logarithmic_fit` fits
`y = a + b*log(x)` and requires a positive coordinate domain.

```pycon
>>> amplitude, rate = ap.exponential_fit([0.0, 1.0, 2.0],
...                                    [2.0, 2.0*np.e, 2.0*np.e**2])
>>> abs(amplitude - 2.0) < 1e-12 and abs(rate - 1.0) < 1e-12
True

```

Least squares after a logarithm minimizes errors in transformed space, not
squared errors in the original response. This can be appropriate for
multiplicative noise and inappropriate for additive noise. If the objective
is original-scale error, fit that nonlinear model directly.

## Rational approximation and Padé coefficients

A rational representation `P/Q` can capture pole-like behavior with fewer
coefficients than a polynomial. It can also introduce unwanted denominator
zeros. Check its poles and evaluation range before using it as a surrogate.

`pade(coeffs, m, n)` accepts **ascending Taylor coefficients**, including the
constant term, and returns ascending numerator and denominator coefficients.
It needs at least `m+n+1` terms. Use `pade_evaluate` to avoid mixing conventions.

```pycon
>>> numerator, denominator = ap.pade([1.0, 1.0, 0.5], m=1, n=1)
>>> np.allclose(numerator, [1.0, 0.5])
True
>>> np.allclose(denominator, [1.0, -0.5])
True
>>> abs(float(ap.pade_evaluate(numerator, denominator, 0.1)) - np.exp(0.1)) < 1e-4
True

```

Padé matches a local series; it does not minimize error over an arbitrary
interval. Degenerate coefficient systems may use a minimum-norm solve, so
inspect the represented rational function rather than assuming a requested
nominal degree produces that effective degree.

`rational_fit(x, y, m, n)` uses a linearized least-squares formulation with
`Q(0)=1`. Its objective is not identical to minimizing original rational
prediction residuals. It returns a callable with ascending `.numerator` and
`.denominator` arrays.

`aaa(f, points=..., ...)` adaptively chooses support points from supplied
samples and returns `(evaluator, support, weights, fvals)`. A values-based
call is also available with `values=...`. The sampling set defines what the
algorithm can see; a small sampled residual does not rule out poles or large
errors between samples. Keep denominator behavior and independent validation
in the assessment.

## Minimax and Fourier representations

`remez` and `minimax_polynomial` seek a polynomial with nearly equal alternating
error extrema. The implementation locates candidate extrema on a finite grid.
The returned callable exposes ascending monomial `.coefficients`, `.nodes`,
and `.error`; that last value is an exchange-system error level, not a
certified uniform bound or a convergence flag. Check the error on a finer,
independent grid and refine extrema when a strict bound matters.

Fourier representations are appropriate when the model is periodic.
`fourier_coefficients` computes cosine and sine coefficients by quadrature;
`fourier_series` returns an evaluator; `trigonometric_fit` solves a
least-squares problem using sampled observations and a chosen harmonic count.

```pycon
>>> periodic = ap.trigonometric_fit(np.linspace(0, 2*np.pi, 24, endpoint=False),
...                                np.sin(np.linspace(0, 2*np.pi, 24, endpoint=False)),
...                                n_harmonics=1, period=2*np.pi)
>>> abs(float(periodic(0.3)) - np.sin(0.3)) < 1e-12
True

```

An incorrect period or an endpoint mismatch produces artificial nonsmoothness.
Sharp jumps yield ringing and slower convergence. More harmonics also require
enough independent samples to distinguish their frequencies. See
[transforms](transforms.md) for sampling and Fourier conventions.

## Orthogonal polynomials and quadrature nodes

The module evaluates Legendre, Chebyshev, Hermite, Laguerre, Jacobi, and
Gegenbauer families. The two Hermite conventions are explicitly named:
`hermite_physicists` and `hermite_probabilists`. Their weights and scaling
differ, so do not substitute one for the other by name alone.

`gauss_legendre_nodes` and related helpers return `(nodes, weights)` for their
respective weighted integral. The weight is part of the mathematical rule.
For example, Hermite quadrature includes `exp(-x*x)` in the measure.

```pycon
>>> nodes, weights = ap.gauss_legendre_nodes(4, a=0.0, b=1.0)
>>> abs(float(weights @ nodes**7) - 1.0/8.0) < 1e-12
True

```

Use [integration](integrate.md) to apply these rules to callables. For the
linear algebra underlying fits, including rank deficiency and regularization,
continue with [linear algebra](linalg.md).
