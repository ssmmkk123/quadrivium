# Special functions

Special functions provide reusable numerical representations of integrals,
recurrences, and differential-equation solutions that appear throughout applied
mathematics. The practical question is often which **form** to evaluate:
a logarithm instead of a large gamma value, a complementary function instead
of subtracting nearly equal numbers, or a scaled function instead of a product
that overflows before cancellation.

Quadrivium implements these functions for real arguments, with complex output
for spherical harmonics. It uses series, recurrences, continued fractions, and
asymptotic formulas in different regions. The [special-function reference](../api/special.md)
records complete signatures, domains, and iteration controls.

## Inputs, shapes, and domain behavior

A scalar input usually produces a Python scalar. Array arguments preserve their
shape, and multi-argument elementwise functions broadcast compatible shapes.
Orders such as the `n` in `bessel_jn(n, x)` are scalar integer choices; they are
not array-valued real-order Bessel parameters.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import special as sp
>>> isinstance(sp.gamma(5.0), float)
True
>>> sp.gamma(np.array([[1.0, 2.0], [3.0, 4.0]])).round(8).tolist()
[[1.0, 1.0], [2.0, 6.0]]
>>> sp.beta(np.array([1.0, 2.0])[:, None], np.array([1.0, 2.0, 3.0])).shape
(2, 3)

```

Domain handling is function-specific. Some inputs raise `DomainError`, while
some singular or overflowing values deliberately return infinity. An invalid
member of an array can cause a domain-checking function to reject the whole
call. Check domains before batching measurements from different regimes.

| Function or family | Contract worth checking |
| --- | --- |
| `gamma` | Non-positive integer poles and positive overflow return `+inf` |
| `erfinv` | `-1` and `1` return signed infinity; values beyond them are rejected |
| `regularized_gamma_p/q` | Require `a > 0`, `x >= 0` |
| `bessel_y0/y1/yn`, `bessel_k0/k1/kn` | Positive real arguments |
| `elliptic_k` | Requires parameter `m < 1` |
| `elliptic_e` | Allows `m = 1`; rejects `m > 1` |
| `lambert_w` | Only real branches `0` and `-1` |
| `zeta` | Real argument excluding its pole at `s = 1` |
| `hyp2f1` | Implemented for real `abs(z) < 1`, with denominator-parameter restrictions |
| `logit` | Requires `0 < p < 1`, including rejection of the endpoints |

Most functions return only a value, without a `converged` flag or an error
estimate. A `tol` or iteration limit controls the internal approximation;
it is not a certified relative-error bound everywhere in the domain.

## Gamma, logarithms, and ratios

The gamma recurrence `Gamma(x+1)=x*Gamma(x)` extends the factorial relation
`Gamma(n)=(n-1)!` at positive integers.

```pycon
>>> round(sp.gamma(5.0), 8)
24.0
>>> abs(sp.gamma(0.5) - float(np.sqrt(np.pi))) < 1e-12
True
>>> x = np.array([0.3, 0.8, 2.5, 8.0])
>>> np.allclose(sp.gamma(x + 1), x * sp.gamma(x), rtol=1e-12)
True

```

For positive arguments, use `log_gamma` when the gamma value may overflow or
when it enters a product or ratio. Keeping a calculation in logarithmic form
can prevent large intermediate values from appearing at all.

```pycon
>>> np.isinf(sp.gamma(200.0))
True
>>> round(sp.log_gamma(200.0), 6)
857.93367
>>> np.isinf(sp.gamma(0.0))
True
>>> a, b = 120.0, 80.0
>>> abs(sp.log_beta(a, b) - (sp.log_gamma(a) + sp.log_gamma(b) - sp.log_gamma(a+b))) < 1e-10
True

```

`beta(a,b)` evaluates the beta function; `log_beta(a,b)` is useful for its
logarithm. For probability models use positive `a` and `b`. Subtracting two
very large, nearly equal log-gamma values can still lose precision, so a
logarithmic rewrite is an improvement in numerical range, not immunity to all
cancellation.

`digamma` is the derivative of log gamma. `trigamma` is its derivative, and
`polygamma(n, x)` gives higher orders. These often appear in likelihood
gradients and curvature calculations.

```pycon
>>> euler_constant = 0.5772156649015329
>>> abs(sp.digamma(1.0) + euler_constant) < 1e-12
True
>>> abs(sp.trigamma(1.0) - float(np.pi**2 / 6)) < 1e-10
True

```

`factorial` and `binomial` return floating-point values. Small integer cases
are handled carefully, but these are not arbitrary-precision combinatorial
interfaces. Use Python's integer arithmetic when exact huge integer values
are the actual output required. A factorial represented as float64 eventually
overflows even though the mathematical integer exists.

## Incomplete gamma and beta

An incomplete function integrates over part of the original domain. A
regularized function divides by the complete integral. Distinguishing those
normalizations avoids missing factors in probability and integral formulas.

| Function | Meaning |
| --- | --- |
| `incomplete_gamma_lower(a,x)` | Unregularized lower incomplete gamma |
| `incomplete_gamma_upper(a,x)` | Unregularized upper incomplete gamma |
| `regularized_gamma_p(a,x)` | Lower incomplete gamma divided by `Gamma(a)` |
| `regularized_gamma_q(a,x)` | Upper incomplete gamma divided by `Gamma(a)` |
| `incomplete_beta(a,b,x)` | **Regularized** incomplete beta `I_x(a,b)` |

```pycon
>>> p = sp.regularized_gamma_p(3.0, 4.0)
>>> q = sp.regularized_gamma_q(3.0, 4.0)
>>> abs(p + q - 1.0) < 1e-13
True
>>> abs(sp.incomplete_gamma_lower(3.0, 4.0) / sp.gamma(3.0) - p) < 1e-13
True
>>> abs(sp.incomplete_beta(1.0, 1.0, 0.3) - 0.3) < 1e-13
True

```

The unregularized gamma routines can overflow in a factor even when a related
regularized probability is representable. Use the normalized function directly
when that is the quantity needed. For extreme probabilities, the distribution
objects in the [stochastic guide](stochastic.md) provide CDF, survival, and
log-tail interfaces with more appropriate semantics than manually combining
special functions.

## Error functions and cancellation

`erf` is the scaled Gaussian integral. Its complement `erfc` is mathematically
`1-erf(x)`, but the direct complementary implementation preserves small positive
tails after `erf(x)` has rounded to one.

```pycon
>>> round(sp.erf(1.0), 12)
0.84270079295
>>> 1.0 - sp.erf(8.0) == 0.0
True
>>> sp.erfc(8.0) > 0.0
True
>>> abs(sp.erfinv(sp.erf(0.7)) - 0.7) < 1e-12
True

```

For still larger positive arguments, `erfc` itself underflows.
`erfcx(x)=exp(x*x)*erfc(x)` evaluates the scaled combination directly:

```pycon
>>> sp.erfc(30.0) == 0.0, sp.erfcx(30.0) > 0.0
(True, True)
>>> abs(float(np.sqrt(np.pi) * 30 * sp.erfcx(30.0)) - 1.0) < 0.001
True

```

The second check uses the leading positive-tail asymptotic behavior. Computing
`exp(x*x)` and `erfc(x)` separately can produce infinity times zero, even when
the scaled answer is finite. The positive-tail advantage does not mean
`erfcx` is bounded on the entire real line; large negative arguments grow.

`dawson` supplies another scaled exponential integral. `fresnel_s` and
`fresnel_c` use the convention with integrands `sin(pi*t*t/2)` and
`cos(pi*t*t/2)`. Verify this factor before comparing optical or wave formulas
that define Fresnel integrals with another scaling.

<figure markdown="span">
  ![Spherical Bessel identity and complementary-error-function cancellation diagnostics](../assets/figures/special-identity-checks.svg#only-light)
  ![Spherical Bessel identity and complementary-error-function cancellation diagnostics](../assets/figures/special-identity-checks-dark.svg#only-dark)
  <figcaption>The spherical Bessel check compares j₀(x) with sin(x)/x. The tail check compares erfc(x) and the subtraction 1−erf(x) against a direct complementary-error reference. These diagnostics show why an algebraically equivalent expression can lose useful digits, and why absolute error near a zero needs different interpretation from relative tail error.</figcaption>
</figure>

## Bessel functions and stable recurrences

Ordinary Bessel functions solve an oscillatory second-order differential
equation. `J` is regular at the origin for nonnegative integer order; `Y` is
singular there. Boundary or regularity conditions determine which solution
belongs in a physical model.

```pycon
>>> round(sp.bessel_j0(1.0), 12)
0.765197686558
>>> round(sp.bessel_jn(3, 2.5), 12)
0.216600391039
>>> x = 3.7
>>> residual = sp.bessel_jn(1, x) + sp.bessel_jn(3, x) - 4 * sp.bessel_jn(2, x) / x
>>> abs(residual) < 1e-10
True

```

A three-term recurrence can be stable in one direction and unstable in the
other. `bessel_jn` chooses upward or downward recurrence according to the order
and argument. Small values at large order require a relative-accuracy check
away from zeros; near a root, relative error can be arbitrarily large because
the true value is close to zero.

`bessel_i0/i1/in` and `bessel_k0/k1/kn` are modified Bessel functions. Their
exponential growth or decay creates additional range limits. The package does
not make every modified Bessel evaluation exponentially scaled merely because
it uses a stable recurrence. Inspect finite values and choose the formulation
appropriate to the required argument range.

Spherical Bessel functions use separate interfaces and orders:

```pycon
>>> radius = np.array([0.2, 0.7, 1.5, 3.0])
>>> np.allclose(sp.spherical_bessel_j(0, radius), np.sin(radius)/radius, atol=1e-12)
True
>>> sp.spherical_bessel_j(0, 0.0)
1.0

```

The origin value is defined by a limit, so avoid evaluating `sin(x)/x` blindly
at zero when constructing an independent reference. A recurrence residual
alone is not a complete accuracy proof: two incorrect neighboring solutions
can satisfy the same homogeneous recurrence. Combine identities with known
values, asymptotics, or an independent reference.

## Airy functions and special integrals

`airy_ai` and `airy_bi` solve `y''=x*y`. Negative arguments are oscillatory;
for positive arguments Ai decays and Bi grows. A graph of both on one linear
scale can hide the decaying solution. If growth or decay is what matters,
inspect the relevant scale and numerical range directly.

Complete elliptic integrals take the **parameter** `m`, not the modulus `k`.
A formula written as `K(k)` in a source may therefore require a call with
`m=k*k`. In the usual real interval `0 <= m <= 1`, `K` diverges as `m`
approaches one while `E(1)=1` remains finite.

```pycon
>>> abs(sp.elliptic_k(0.0) - float(np.pi/2)) < 1e-13
True
>>> abs(sp.elliptic_e(0.0) - float(np.pi/2)) < 1e-13
True
>>> sp.elliptic_e(1.0)
1.0

```

`exponential_integral` denotes Ei, while `expint_n(n,x)` denotes
`E_n(x)=integral_1^infinity exp(-x*t)/t**n dt`. Their signs, lower limits,
and branch/domain conventions differ. `sine_integral` and `cosine_integral`
likewise require their stated definitions when comparing formulas.
`expint_n` permits `x=0` only when `n>=2`, where the integral is finite.

## Zeta and Lambert W

`zeta` evaluates the Riemann zeta function on the real line excluding `s=1`.
Values for `s<=1` use continuation rather than the divergent defining series.
The interface does not provide general complex zeta evaluation.

```pycon
>>> abs(sp.zeta(2.0) - float(np.pi**2/6)) < 1e-12
True
>>> abs(sp.zeta(-1.0) + 1/12) < 1e-12
True
>>> sp.zeta(-2.0)
0.0

```

Near the pole, the function is sensitive to small changes in its argument.
A large result is expected; it does not by itself show failure. Check the
argument's precision and the distance to the singularity before demanding a
small absolute error.

`lambert_w` inverts `w*exp(w)=x`. There are two real answers for
`-1/e < x < 0`, so branch choice is part of the model.

```pycon
>>> principal = sp.lambert_w(-0.2, branch=0)
>>> lower = sp.lambert_w(-0.2, branch=-1)
>>> lower < -1 < principal < 0
True
>>> abs(float(principal*np.exp(principal)) + 0.2) < 1e-12
True
>>> abs(float(lower*np.exp(lower)) + 0.2) < 1e-12
True

```

The principal branch is defined for `x>=-1/e`; the lower branch for
`-1/e<=x<0`. They meet at `w=-1`. Near that branch point a small equation
residual does not necessarily imply a comparably small error in `w`, because
the inverse is ill-conditioned there.

## Hypergeometric and angular functions

`hyp1f1(a,b,z)` is the confluent hypergeometric function; for negative
arguments the implementation uses Kummer's transformation to avoid a badly
cancelling direct series. `hyp2f1(a,b,c,z)` is restricted to real
`abs(z)<1` and rejects non-positive integer `c`. It is not a general analytic
continuation engine across the complex plane.

```pycon
>>> np.allclose(sp.hyp1f1(1.0, 1.0, [-3.0, 0.0, 2.0]), np.exp([-3.0, 0.0, 2.0]), rtol=1e-12)
True
>>> abs(sp.hyp2f1(1.0, 1.0, 2.0, 0.3) + float(np.log(0.7))/0.3) < 1e-12
True

```

`associated_legendre(l,m,x)` includes the Condon–Shortley phase and requires
`0<=m<=l`, `abs(x)<=1`. `spherical_harmonic(l,m,theta,phi)` returns complex
values, with **polar angle theta and azimuth phi**, both in radians.
Negative `m` uses the conjugation identity implemented by the function.

```pycon
>>> round(sp.associated_legendre(2, 0, 0.5), 8)
-0.125
>>> harmonic = sp.spherical_harmonic(1, 0, 0.7, 0.3)
>>> abs(float(np.real(harmonic)) - float(np.sqrt(3/(4*np.pi))*np.cos(0.7))) < 1e-12
True

```

For probability transformations, `logistic` avoids overflow for both signs of
its argument and `logit` uses a stable logarithmic expression. The inverse
relation is still limited by rounding: a large positive argument can make
`logistic` equal exactly one, outside `logit`'s open interval.

## Validate the representation you use

Use known values for anchoring, identities for consistency, and separate
absolute/relative checks near zeros and tails. Probe both sides of numerical
regime changes and points near domain boundaries. Record whether an overflow,
a pole, or a rejected domain value is expected in your model.

See [approximation](approx.md) for orthogonal polynomials,
[stochastic methods](stochastic.md) for probability distributions, and
[integration](integrate.md) for independent integral checks.
