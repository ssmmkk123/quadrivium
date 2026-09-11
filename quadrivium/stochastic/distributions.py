"""Scalar-parameter distributions with array evaluation and stable tail APIs.

Sampling accepts a seed or the package's Generator. Continuous distributions
expose pdf/logpdf, discrete ones also expose pmf/logpmf. Quantiles use a monotone
bracket and evaluate the smaller tail instead of subtracting nearly equal CDFs.
"""
from __future__ import annotations

import math
import operator
from .. import numeric as np
from ..special import regularized_gamma_p, regularized_gamma_q, incomplete_beta, erfcx

__all__ = ["Distribution", "Normal", "Uniform", "Exponential", "Gamma", "Beta",
           "StudentT", "ChiSquare", "Poisson", "Binomial"]


def _map(fn, x):
    values = np.asarray(x, dtype=float)
    result = np.empty(values.shape)
    for i, value in enumerate(values.ravel()):
        result.ravel()[i] = fn(float(value))
    return float(result) if values.ndim == 0 else result


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _log(p):
    return math.log(p) if p > 0 else -math.inf


def _standardize(x, loc, scale):
    difference = x - loc
    return difference / scale if math.isfinite(difference) else x / scale - loc / scale


def _beta_cf(a, b, x):
    tiny = 1e-300
    c = 1.0
    d = 1 - (a + b) * x / (a + 1)
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    h = d
    for m in range(1, 2001):
        for coefficient in (m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m)),
                            -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))):
            d = 1 + coefficient * d
            c = 1 + coefficient / c
            if abs(d) < tiny:
                d = tiny
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            delta = d * c
            h *= delta
        if abs(delta - 1) < 2e-15:
            break
    return h


def _log_ibeta(a, b, x, logx=None):
    if math.isnan(x):
        return math.nan
    if x <= 0 and logx is None:
        return -math.inf
    if x >= 1:
        return 0.0
    if x > (a + 1) / (a + b + 2):
        tail = _log_ibeta(b, a, 1 - x)
        return math.log1p(-math.exp(tail))
    logx = math.log(x) if logx is None else logx
    return (a * logx + b * math.log1p(-x) -
            (math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)) +
            math.log(_beta_cf(a, b, x) / a))


def _log_gamma_parts(a, rng):
    d = (a if a >= 1 else a + 1) - 1 / 3
    c = 1 / math.sqrt(9 * d)
    while True:
        z = float(rng.normal())
        v = 1 + c * z
        if v <= 0:
            continue
        v = v ** 3
        u = max(float(rng.random()), 1e-300)
        if u < 1 - 0.0331 * z ** 4 or math.log(u) < 0.5*z*z + d*(1-v+math.log(v)):
            value = math.log(d) + math.log(v)
            extra = math.log(max(float(rng.random()), 1e-300)) if a < 1 else 0.0
            return value, extra


class Distribution:
    """Common distribution interface; parameters belong to concrete subclasses."""
    support = (-math.inf, math.inf)
    discrete = False

    def pdf(self, x):
        return _map(lambda v: math.exp(self._logpdf(v)), x)

    def logpdf(self, x):
        return _map(self._logpdf, x)

    def pmf(self, x):
        if not self.discrete:
            raise TypeError("pmf is defined only for discrete distributions")
        return self.pdf(x)

    def logpmf(self, x):
        if not self.discrete:
            raise TypeError("logpmf is defined only for discrete distributions")
        return self.logpdf(x)

    def cdf(self, x):
        return _map(lambda v: self._cdf(v) if not math.isnan(v) else math.nan, x)

    def sf(self, x):
        return _map(lambda v: self._sf(v) if not math.isnan(v) else math.nan, x)

    def logcdf(self, x):
        return _map(lambda v: self._logcdf(v), x)

    def logsf(self, x):
        return _map(lambda v: self._logsf(v), x)

    def _logcdf(self, x):
        if math.isnan(x):
            return math.nan
        p = self._cdf(x)
        return math.log1p(-self._sf(x)) if p > 0.5 else _log(p)

    def _logsf(self, x):
        if math.isnan(x):
            return math.nan
        q = self._sf(x)
        return math.log1p(-self._cdf(x)) if q > 0.5 else _log(q)

    def _quantile(self, p, upper=False):
        if math.isnan(p):
            return math.nan
        if not 0 <= p <= 1:
            raise ValueError("probabilities must lie in [0, 1]")
        low, high = self.support
        if p in (0.0, 1.0):
            return high if (p == 1) != upper else low
        if p > 0.5:
            return self._quantile(1.0 - p, not upper)
        # Compare logarithmic tails, including probabilities below normal range.
        logp = math.log(p)
        fn = self._logsf if upper else self._logcdf
        def below(x):
            return fn(x) > logp if upper else fn(x) < logp
        lo = low if math.isfinite(low) else -1.0
        hi = high if math.isfinite(high) else 1.0
        largest = float.fromhex("0x1.fffffffffffffp+1023")
        for _ in range(1024):
            if below(lo):
                break
            if math.isfinite(low):
                break
            lo = max(-largest, lo * 2) if lo < 0 else -1.0
        for _ in range(1024):
            if not below(hi):
                break
            if math.isfinite(high):
                break
            hi = min(largest, hi * 2) if hi > 0 else 1.0
        if below(hi):
            return high
        if not below(lo):
            return low
        if self.discrete:
            lo, hi = math.floor(lo) - 1, math.ceil(hi)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if below(mid):
                    lo = mid
                else:
                    hi = mid
            return float(hi)
        for _ in range(1100):
            mid = lo * 0.5 + hi * 0.5
            if mid == lo or mid == hi:
                break
            if below(mid):
                lo = mid
            else:
                hi = mid
            if hi - lo <= 4e-15 * max(abs(lo), abs(hi), 1e-300):
                break
        return lo * 0.5 + hi * 0.5

    def ppf(self, q):
        return _map(self._quantile, q)

    def isf(self, q):
        return _map(lambda p: self._quantile(p, True), q)

    def rvs(self, size=None, rng=None):
        generator = np.random.default_rng(rng)
        return self.ppf(generator.random(size=size))


class Normal(Distribution):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc, self.scale = _finite(loc, "loc"), _positive(scale, "scale")

    def _logpdf(self, x):
        z = _standardize(x, self.loc, self.scale)
        return -0.5 * z * z - math.log(self.scale) - 0.5 * math.log(2 * math.pi)

    def _cdf(self, x):
        return 0.5 * math.erfc(-_standardize(x, self.loc, self.scale) / math.sqrt(2))

    def _sf(self, x):
        return 0.5 * math.erfc(_standardize(x, self.loc, self.scale) / math.sqrt(2))

    def _logsf(self, x):
        return self._logtail(_standardize(x, self.loc, self.scale) / math.sqrt(2))

    @staticmethod
    def _logtail(z):
        if z == math.inf:
            return -math.inf
        if z > 0:
            return math.log(0.5 * float(erfcx(z))) - z * z
        return math.log1p(-0.5 * math.erfc(-z))

    def _logcdf(self, x):
        return self._logtail(-_standardize(x, self.loc, self.scale) / math.sqrt(2))

    def rvs(self, size=None, rng=None):
        return np.random.default_rng(rng).normal(self.loc, self.scale, size=size)


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):
        low, high = _finite(low, "low"), _finite(high, "high")
        if high <= low:
            raise ValueError("high must exceed low")
        self.low, self.high, self.support = low, high, (low, high)

    def _logpdf(self, x):
        if math.isnan(x):
            return math.nan
        return -math.log(self.high - self.low) if self.low <= x <= self.high else -math.inf

    def _cdf(self, x):
        return min(1.0, max(0.0, (x - self.low) / (self.high - self.low)))

    def _sf(self, x):
        return min(1.0, max(0.0, (self.high - x) / (self.high - self.low)))

    def rvs(self, size=None, rng=None):
        return np.random.default_rng(rng).uniform(self.low, self.high, size=size)


class Exponential(Distribution):
    support = (0.0, math.inf)

    def __init__(self, scale=1.0):
        self.scale = _positive(scale, "scale")

    def _logpdf(self, x):
        return -math.log(self.scale) - x / self.scale if x >= 0 else (math.nan if math.isnan(x) else -math.inf)

    def _cdf(self, x):
        return -math.expm1(-x / self.scale) if x > 0 else 0.0

    def _sf(self, x):
        return math.exp(-x / self.scale) if x > 0 else 1.0

    def _logsf(self, x):
        return -max(x, 0.0) / self.scale

    def rvs(self, size=None, rng=None):
        return np.random.default_rng(rng).exponential(self.scale, size=size)


class Gamma(Distribution):
    support = (0.0, math.inf)

    def __init__(self, shape, scale=1.0):
        self.shape, self.scale = _positive(shape, "shape"), _positive(scale, "scale")

    def _logpdf(self, x):
        if math.isnan(x):
            return math.nan
        if x < 0 or x == math.inf:
            return -math.inf
        if x == 0:
            return math.inf if self.shape < 1 else (-math.log(self.scale) if self.shape == 1 else -math.inf)
        return ((self.shape - 1) * math.log(x / self.scale) - x / self.scale -
                math.lgamma(self.shape) - math.log(self.scale))

    def _cdf(self, x):
        return 0.0 if x <= 0 else (1.0 if x == math.inf else float(regularized_gamma_p(self.shape, x / self.scale)))

    def _sf(self, x):
        return 1.0 if x <= 0 else (0.0 if x == math.inf else float(regularized_gamma_q(self.shape, x / self.scale)))

    def _logsf(self, x):
        z, a = x / self.scale, self.shape
        if math.isnan(z):
            return math.nan
        if z <= 0:
            return 0.0
        if z == math.inf:
            return -math.inf
        if z < a + 1:
            return super()._logsf(x)
        b, c = z + 1 - a, 1e300
        d = 1.0 / b
        h = d
        for i in range(1, 2001):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            c = b + an / c
            if abs(d) < 1e-300:
                d = 1e-300
            if abs(c) < 1e-300:
                c = 1e-300
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < 2e-15:
                break
        return a * math.log(z) - z - math.lgamma(a) + math.log(h)

    def rvs(self, size=None, rng=None):
        rng = np.random.default_rng(rng)
        shape = () if size is None else ((operator.index(size),) if isinstance(size, int) else tuple(size))
        out = np.empty(shape)
        a = self.shape
        d = (a if a >= 1 else a + 1) - 1 / 3
        c = 1 / math.sqrt(9 * d)
        for i in range(out.size):
            while True:
                z = float(rng.normal())
                v = 1 + c * z
                if v <= 0:
                    continue
                v = v ** 3
                u = max(float(rng.random()), 1e-300)
                if u < 1 - 0.0331 * z ** 4 or math.log(u) < 0.5 * z * z + d * (1 - v + math.log(v)):
                    value = d * v
                    if a < 1:
                        value *= max(float(rng.random()), 1e-300) ** (1 / a)
                    out.ravel()[i] = value * self.scale
                    break
        return float(out) if size is None else out


class Beta(Distribution):
    support = (0.0, 1.0)

    def __init__(self, a, b):
        self.a, self.b = _positive(a, "a"), _positive(b, "b")
        self._normalizer = math.lgamma(self.a) + math.lgamma(self.b) - math.lgamma(self.a + self.b)

    def _logpdf(self, x):
        if math.isnan(x):
            return math.nan
        if x < 0 or x > 1:
            return -math.inf
        if x == 0:
            return math.inf if self.a < 1 else (-self._normalizer if self.a == 1 else -math.inf)
        if x == 1:
            return math.inf if self.b < 1 else (-self._normalizer if self.b == 1 else -math.inf)
        return (self.a - 1) * math.log(x) + (self.b - 1) * math.log1p(-x) - self._normalizer

    def _cdf(self, x):
        return 0.0 if x <= 0 else (1.0 if x >= 1 else float(incomplete_beta(self.a, self.b, x)))

    def _sf(self, x):
        return 1.0 if x <= 0 else (0.0 if x >= 1 else float(incomplete_beta(self.b, self.a, 1 - x)))

    def _logcdf(self, x):
        return _log_ibeta(self.a, self.b, x)

    def _logsf(self, x):
        return _log_ibeta(self.b, self.a, 1 - x)

    def rvs(self, size=None, rng=None):
        rng = np.random.default_rng(rng)
        shape = () if size is None else ((operator.index(size),) if isinstance(size, int) else tuple(size))
        out = np.empty(shape)
        for i in range(out.size):
            base_x, extra_x = _log_gamma_parts(self.a, rng)
            base_y, extra_y = _log_gamma_parts(self.b, rng)
            scale = min(self.a, self.b, 1.0)
            delta = scale * (base_x - base_y) + (scale / self.a) * extra_x - (scale / self.b) * extra_y
            ratio = 0.0 if abs(delta) > 745 * scale else math.exp(-abs(delta) / scale)
            out.ravel()[i] = 1 / (1 + ratio) if delta >= 0 else ratio / (1 + ratio)
        return float(out) if size is None else out


class StudentT(Distribution):
    def __init__(self, df, loc=0.0, scale=1.0):
        self.df, self.loc, self.scale = _positive(df, "df"), _finite(loc, "loc"), _positive(scale, "scale")

    def _logpdf(self, x):
        z, v = _standardize(x, self.loc, self.scale), self.df
        if z == 0:
            term = 0.0
        else:
            logratio = 2 * math.log(abs(z)) - math.log(v)
            term = logratio + math.log1p(math.exp(-logratio)) if logratio > 0 else math.log1p(math.exp(logratio))
        return (math.lgamma((v + 1) / 2) - math.lgamma(v / 2) -
                0.5 * math.log(v * math.pi) - math.log(self.scale) -
                (v + 1) / 2 * term)

    def _sf(self, x):
        return math.exp(self._logsf(x))

    def _cdf(self, x):
        return math.exp(self._logcdf(x))

    def _logtail(self, z):
        if math.isnan(z):
            return math.nan
        if z == 0:
            return -math.log(2)
        if abs(z) < 0.1 * min(1.0, math.sqrt(self.df)):
            # Integrate the density series near the center. Forming
            # v/(v+z*z) here would round to one and create a CDF plateau.
            ratio, alpha = z*z / self.df, (self.df + 1) / 2
            term = total = 1.0
            for k in range(1, 100):
                term *= -(alpha + k - 1) / k * ratio * (2*k - 1) / (2*k + 1)
                total += term
                if abs(term) < 2e-16 * abs(total):
                    break
            density0 = math.exp(math.lgamma(alpha) - math.lgamma(self.df/2) - .5*math.log(self.df*math.pi))
            return -math.log(2) + math.log1p(-2 * density0 * z * total)
        logratio = 2 * math.log(abs(z)) - math.log(self.df)
        logw = -(logratio + math.log1p(math.exp(-logratio)) if logratio > 0 else math.log1p(math.exp(logratio)))
        logtail = -math.log(2) + _log_ibeta(self.df / 2, .5, math.exp(logw), logw)
        return logtail if z > 0 else math.log1p(-math.exp(logtail))

    def _logsf(self, x):
        return self._logtail(_standardize(x, self.loc, self.scale))

    def _logcdf(self, x):
        return self._logtail(-_standardize(x, self.loc, self.scale))

    def rvs(self, size=None, rng=None):
        rng = np.random.default_rng(rng)
        return self.loc + self.scale * rng.normal(size=size) / np.sqrt(Gamma(self.df / 2, 2).rvs(size, rng) / self.df)


class ChiSquare(Gamma):
    def __init__(self, df):
        super().__init__(df / 2, 2.0)
        self.df = 2 * self.shape


class Poisson(Distribution):
    support, discrete = (0.0, math.inf), True

    def __init__(self, mu):
        self.mu = _finite(mu, "mu")
        if self.mu < 0:
            raise ValueError("mu must be nonnegative")

    def _logpdf(self, x):
        if math.isnan(x):
            return math.nan
        if not math.isfinite(x) or x < 0 or x != math.floor(x):
            return -math.inf
        if self.mu == 0:
            return 0.0 if x == 0 else -math.inf
        return x * math.log(self.mu) - self.mu - math.lgamma(x + 1)

    def _cdf(self, x):
        return 0.0 if x < 0 else (1.0 if x == math.inf or self.mu == 0 else float(regularized_gamma_q(math.floor(x) + 1, self.mu)))

    def _sf(self, x):
        return 1.0 if x < 0 else (0.0 if x == math.inf or self.mu == 0 else float(regularized_gamma_p(math.floor(x) + 1, self.mu)))

    def rvs(self, size=None, rng=None):
        return np.random.default_rng(rng).poisson(self.mu, size=size)


class Binomial(Distribution):
    discrete = True

    def __init__(self, n, p):
        self.n, self.p = operator.index(n), float(p)
        if self.n < 0 or not 0 <= self.p <= 1:
            raise ValueError("require n >= 0 and 0 <= p <= 1")
        self.support = (0.0, float(self.n))

    def _logpdf(self, x):
        if math.isnan(x):
            return math.nan
        if not 0 <= x <= self.n or x != math.floor(x):
            return -math.inf
        if self.p in (0.0, 1.0):
            return 0.0 if x == self.n * self.p else -math.inf
        return (math.lgamma(self.n + 1) - math.lgamma(x + 1) - math.lgamma(self.n - x + 1) +
                x * math.log(self.p) + (self.n - x) * math.log1p(-self.p))

    def _cdf(self, x):
        if x < 0:
            return 0.0
        if x >= self.n:
            return 1.0
        return float(incomplete_beta(self.n - math.floor(x), math.floor(x) + 1, 1 - self.p))

    def _sf(self, x):
        if x < 0:
            return 1.0
        if x >= self.n:
            return 0.0
        return float(incomplete_beta(math.floor(x) + 1, self.n - math.floor(x), self.p))
