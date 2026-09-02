# Special functions

```python
from quadrivium.special import gamma, bessel_jn, zeta, lambert_w
import quadrivium as qd          # qd.gamma, qd.erf, qd.beta, qd.lambert_w
```

52 functions: the gamma and beta families, error functions, Bessel and Airy
functions, elliptic and exponential integrals, the Riemann zeta function,
Lambert W, hypergeometric functions, and spherical harmonics. Full signatures
are in the [`special` reference](../api/special.md).

Each is computed by whichever representation is accurate in the region asked
for — series near the origin, continued fraction or asymptotic expansion far
from it, a reflection formula where the argument is negative — rather than by
one formula stretched past its useful range.

## Gamma and friends

```pycon
>>> import numpy as np
>>> import quadrivium as qd
>>> round(float(qd.gamma(5.0)), 10)                 # Γ(n) = (n-1)!
24.0
>>> round(float(qd.gamma(0.5)), 12) == round(float(np.sqrt(np.pi)), 12)
True

```

`log_gamma` is what to use when the value would overflow — `Γ(200)` is beyond
a double, but its logarithm is not:

```pycon
>>> from quadrivium.special import log_gamma, digamma, polygamma
>>> round(float(log_gamma(200.0)), 6)
857.93367
>>> abs(float(digamma(1.0)) + 0.5772156649015329) < 1e-12    # ψ(1) = −γ
True

```

`polygamma(n, x)` gives higher derivatives (`trigamma` is `n = 1`), `beta` and
`log_beta` the beta function, and `factorial` and `binomial` the exact
combinatorial values. The incomplete forms — `incomplete_gamma_lower`,
`incomplete_gamma_upper`, `regularized_gamma_p`, `regularized_gamma_q`,
`incomplete_beta` — are the CDFs of the chi-square, Poisson, gamma, beta,
Student-t and F distributions, which is where they are usually met.

<figure markdown="span">
  ![The gamma function through its poles, and the logarithm that stays finite](../assets/figures/special-gamma.svg#only-light)
  ![The gamma function through its poles, and the logarithm that stays finite](../assets/figures/special-gamma-dark.svg#only-dark)
  <figcaption>Γ has a pole at every non-positive integer and passes through the factorials at the positive ones. It also overflows a double past x = 172, which is the whole reason `log_gamma` exists as a separate function rather than as `log(gamma(x))`.</figcaption>
</figure>

```pycon
>>> from quadrivium.special import regularized_gamma_p, regularized_gamma_q
>>> p, q = regularized_gamma_p(3.0, 4.0), regularized_gamma_q(3.0, 4.0)
>>> abs(p + q - 1.0) < 1e-14                        # P + Q = 1, exactly
True

```

## Error function family

```pycon
>>> from quadrivium.special import erf, erfc, erfinv, erfcx, dawson
>>> round(float(erf(1.0)), 12)
0.84270079295
>>> abs(float(erfinv(erf(0.7))) - 0.7) < 1e-12
True

```

`erfc(x)` underflows for large `x`; `erfcx(x) = e^{x²} erfc(x)` does not, which
is what makes it the right function for tail probabilities:

```pycon
>>> float(erfc(30.0)) == 0.0                        # underflows
True
>>> float(erfcx(30.0)) > 0.0                        # scaled version survives
True

```

`dawson` is computed by Rybicki's method, because both obvious routes —
`e^{−x²}∫e^{t²}dt` and the scaled complementary error function — overflow
before they meet in the middle. `fresnel_s` and `fresnel_c` complete the set.

<figure markdown="span">
  ![The error function family, and the scaled form that survives the tail](../assets/figures/special-error-family.svg#only-light)
  ![The error function family, and the scaled form that survives the tail](../assets/figures/special-error-family-dark.svg#only-dark)
  <figcaption>erfc underflows to exactly zero at x = 27, which makes any tail probability computed through it zero as well. erfcx carries the e^{x²} factor and keeps returning digits far beyond that point.</figcaption>
</figure>

## Bessel and Airy functions

```pycon
>>> from quadrivium.special import bessel_j0, bessel_jn, bessel_yn
>>> round(float(bessel_j0(1.0)), 12)
0.765197686558
>>> round(float(bessel_jn(3, 2.5)), 12)
0.216600391039

```

The Wronskian identity `J_{n+1}(x)Y_n(x) − J_n(x)Y_{n+1}(x) = 2/(πx)` ties the
two solutions together and holds to machine precision — a check no table of
values could give you:

```pycon
>>> from quadrivium.special import bessel_jn, bessel_yn
>>> x = 3.7
>>> wronskian = (bessel_jn(3, x) * bessel_yn(2, x) - bessel_jn(2, x) * bessel_yn(3, x))
>>> abs(float(wronskian) - 2/(np.pi*x)) < 1e-9
True

```

<figure markdown="span">
  ![Bessel functions of the first and second kind](../assets/figures/special-bessel.svg#only-light)
  ![Bessel functions of the first and second kind](../assets/figures/special-bessel-dark.svg#only-dark)
  <figcaption>J is regular at the origin and Y is not, which is how a physical problem picks between them. The Wronskian identity ties the two families together and holds across the whole range plotted — a check no table of values could give you.</figcaption>
</figure>

Modified Bessel functions (`bessel_i0`, `bessel_i1`, `bessel_in`, `bessel_k0`,
`bessel_k1`, `bessel_kn`) and spherical Bessel functions
(`spherical_bessel_j`, `spherical_bessel_y`) follow the same pattern.
`airy_ai` and `airy_bi` solve `y″ = xy`, the equation that governs the
transition between oscillation and exponential growth.

<figure markdown="span">
  ![Airy functions: oscillation on one side, exponential behaviour on the other](../assets/figures/special-airy.svg#only-light)
  ![Airy functions: oscillation on one side, exponential behaviour on the other](../assets/figures/special-airy-dark.svg#only-dark)
  <figcaption>Both solve y″ = xy. Where x is negative the equation is oscillatory and both functions ring; where x is positive one decays and the other grows. That turning point is why the Airy functions appear wherever a wave meets a barrier.</figcaption>
</figure>

## Integrals and zeta

```pycon
>>> from quadrivium.special import sine_integral, exponential_integral, expint_n
>>> round(float(sine_integral(1.0)), 12)
0.946083070367

```

`zeta(s)` is evaluated on the whole real line: Euler-Maclaurin for `s > 1`,
the alternating Borwein algorithm in the critical strip, and the functional
equation for `s < 0`. The values it must reproduce are exactly known:

```pycon
>>> from quadrivium.special import zeta
>>> abs(float(zeta(2.0)) - np.pi**2/6) < 1e-12
True
>>> abs(float(zeta(-1.0)) + 1/12) < 1e-12           # ζ(−1) = −1/12
True

```

## Lambert W and hypergeometric functions

`lambert_w` inverts `w e^w = x`, with both real branches:

```pycon
>>> round(float(qd.lambert_w(np.e)), 12)            # W(e·1) = 1
1.0
>>> w = float(qd.lambert_w(-0.2, branch=-1))        # the lower branch
>>> bool(abs(w * np.exp(w) + 0.2) < 1e-12)
True

```

The hypergeometric functions `hyp1f1` (confluent, Kummer's `M`) and `hyp2f1`
(Gauss) subsume most of the others as special cases. `hyp1f1` applies Kummer's
transformation for *every* negative argument, because the direct alternating
series loses about `2|z|` nepers of precision before it converges at all:

```pycon
>>> from quadrivium.special import hyp1f1
>>> abs(float(hyp1f1(1.0, 1.0, -30.0)) - float(np.exp(-30.0))) < 1e-15
True

```

## Legendre and spherical harmonics

```pycon
>>> from quadrivium.special import associated_legendre, spherical_harmonic
>>> round(float(associated_legendre(2, 0, 0.5)), 12)    # P₂(x) = (3x²−1)/2
-0.125
>>> y = spherical_harmonic(1, 0, 0.7, 0.3)
>>> abs(float(np.real(y)) - float(np.sqrt(3/(4*np.pi)) * np.cos(0.7))) < 1e-12
True

```

<figure markdown="span">
  ![The zeta function on the real line, and both real branches of Lambert W](../assets/figures/special-zeta-lambert.svg#only-light)
  ![The zeta function on the real line, and both real branches of Lambert W](../assets/figures/special-zeta-lambert-dark.svg#only-dark)
  <figcaption>ζ is evaluated by three different representations either side of its pole at s = 1, and reproduces the values that are known in closed form. Lambert W inverts w·eʷ, which is two-valued on [−1/e, 0): the branch is an argument, not a guess.</figcaption>
</figure>

## Pitfalls

- **Check the domain.** `gamma` has poles at the non-positive integers,
  `log_gamma` is real only for positive arguments, `elliptic_k(m)` needs
  `m < 1`, and `lambert_w` needs `x ≥ −1/e`. Outside these you get a
  `DomainError`, not a quiet `nan`.
- **`erfc` underflows, `erfcx` does not.** Same for `exp(log_gamma(x))` versus
  `gamma(x)` for large `x`.
- **`bessel_jn` for large order and small argument is a tiny number**
  computed by downward recurrence. It is accurate in a *relative* sense, which
  is the useful sense, but do not expect the absolute error to be smaller than
  the value itself times the machine epsilon.
- **Integer versus real order.** The Bessel routines here take integer orders.
  Half-integer orders are the spherical Bessel functions, which have their own
  functions.
- **`zeta` near `s = 1` is a pole.** Values close to 1 lose accuracy;
  `zeta(1.0)` is undefined.

## See also

- [`special` API reference](../api/special.md) — every signature.
- [Approximation guide](approx.md) — the orthogonal polynomial families.
- [Stochastic guide](stochastic.md) — where the incomplete gamma and beta
  functions become distribution functions.
