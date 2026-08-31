"""Sampling from probability distributions."""

from __future__ import annotations

import numpy as np

from ..core.utils import as_vector

__all__ = [
    "inverse_transform",
    "box_muller",
    "marsaglia_polar",
    "rejection_sampling",
    "adaptive_rejection",
    "ratio_of_uniforms",
    "alias_table",
    "sample_discrete",
    "multivariate_normal",
    "gibbs_bivariate_normal",
    "bootstrap",
    "jackknife",
    "permutation_test",
]


def inverse_transform(inv_cdf, n: int = 1000, rng=None):
    """Inverse transform sampling: ``X = F^-1(U)`` with ``U`` uniform.

    Exact whenever the quantile function is available.
    """
    rng = np.random.default_rng(rng)
    u = rng.random(n)
    return np.array([inv_cdf(ui) for ui in u])


def box_muller(n: int = 1000, mu: float = 0.0, sigma: float = 1.0, rng=None):
    """Box-Muller transform: two uniforms give two independent normals."""
    rng = np.random.default_rng(rng)
    m = (n + 1) // 2
    u1 = rng.random(m)
    u2 = rng.random(m)
    r = np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-300)))
    z0 = r * np.cos(2 * np.pi * u2)
    z1 = r * np.sin(2 * np.pi * u2)
    return (mu + sigma * np.concatenate([z0, z1]))[:n]


def marsaglia_polar(n: int = 1000, mu: float = 0.0, sigma: float = 1.0, rng=None):
    """Marsaglia polar method: like Box-Muller but with no trigonometry."""
    rng = np.random.default_rng(rng)
    out = []
    while len(out) < n:
        batch = rng.uniform(-1, 1, size=(max(n, 32), 2))
        s = np.sum(batch**2, axis=1)
        keep = batch[(s < 1) & (s > 0)]
        sk = np.sum(keep**2, axis=1)
        factor = np.sqrt(-2.0 * np.log(sk) / sk)
        out.extend((keep * factor[:, None]).ravel())
    return mu + sigma * np.array(out[:n])


def rejection_sampling(pdf, proposal_sampler, proposal_pdf, M: float, n: int = 1000,
                       rng=None, max_tries: int = 10_000_000):
    """Rejection sampling with an envelope ``M * proposal_pdf >= pdf``.

    Returns the samples and the realized acceptance rate; an ``M`` far larger
    than necessary is correct but wasteful.
    """
    rng = np.random.default_rng(rng)
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        x = proposal_sampler(rng)
        tries += 1
        if rng.random() * M * proposal_pdf(x) <= pdf(x):
            out.append(x)
    return np.array(out), (len(out) / tries if tries else 0.0)


def adaptive_rejection(log_pdf, x_init, n: int = 1000, domain=(-np.inf, np.inf),
                       rng=None, dlog_pdf=None, max_points: int = 60):
    """Adaptive rejection sampling (Gilks-Wild) for log-concave densities.

    Builds a piecewise-linear *upper hull* of ``log pdf`` from tangents at the
    current abscissae. The hull exponentiates to a piecewise-exponential
    density that can be sampled exactly; every rejected point is added as a new
    tangent, so the envelope tightens and the acceptance rate climbs toward one.

    ``x_init`` must bracket the mode -- the tangent slopes have to be positive
    at the left end and negative at the right for an unbounded domain, or the
    envelope has infinite mass.
    """
    from ..diff.finite import central_difference

    rng = np.random.default_rng(rng)
    lo, hi = float(domain[0]), float(domain[1])
    xs = sorted(float(v) for v in np.atleast_1d(np.asarray(x_init, dtype=float)))
    dh = dlog_pdf or (lambda t: central_difference(log_pdf, t, 1e-5))

    def hull(xs):
        """Tangent data and segment boundaries for the current abscissae."""
        h = np.array([float(log_pdf(x)) for x in xs])
        m = np.array([float(dh(x)) for x in xs])
        k = len(xs)
        z = np.empty(k + 1)
        z[0], z[k] = lo, hi
        for j in range(k - 1):
            dm = m[j] - m[j + 1]
            if abs(dm) < 1e-300:
                z[j + 1] = 0.5 * (xs[j] + xs[j + 1])
            else:
                z[j + 1] = ((h[j + 1] - h[j] - xs[j + 1] * m[j + 1]
                             + xs[j] * m[j]) / dm)
            z[j + 1] = min(max(z[j + 1], xs[j]), xs[j + 1])
        return np.array(h), np.array(m), z

    def log_masses(xs, h, m, z):
        """Log integral of exp(hull) over each segment."""
        out = np.empty(len(xs))
        for j in range(len(xs)):
            zl, zr = z[j], z[j + 1]
            if m[j] > 1e-12:
                if not np.isfinite(zr):
                    return None
                a, b = m[j] * (zl - xs[j]), m[j] * (zr - xs[j])
                out[j] = h[j] + b + np.log1p(-np.exp(min(a - b, -1e-16))) - np.log(m[j])
            elif m[j] < -1e-12:
                if not np.isfinite(zl):
                    return None
                a, b = m[j] * (zl - xs[j]), m[j] * (zr - xs[j])
                out[j] = h[j] + a + np.log1p(-np.exp(min(b - a, -1e-16))) - np.log(-m[j])
            else:
                if not (np.isfinite(zl) and np.isfinite(zr)):
                    return None
                out[j] = h[j] + np.log(max(zr - zl, 1e-300))
        return out

    def draw(xs, h, m, z, lm):
        """Sample one point from the piecewise-exponential envelope."""
        w = np.exp(lm - np.max(lm))
        w = w / w.sum()
        j = int(rng.choice(len(xs), p=w))
        zl, zr = z[j], z[j + 1]
        u = rng.random()
        if abs(m[j]) < 1e-12:
            return zl + u * (zr - zl), j
        a, b = m[j] * (zl - xs[j]), m[j] * (zr - xs[j])
        if m[j] > 0:
            lt = b + np.log(np.exp(a - b) + u * (1.0 - np.exp(a - b)))
        else:
            lt = a + np.log1p(u * (np.exp(b - a) - 1.0))
        return xs[j] + lt / m[j], j

    h, m, z = hull(xs)
    lm = log_masses(xs, h, m, z)
    if lm is None:
        raise ValueError("the initial abscissae must bracket the mode: the "
                         "envelope has infinite mass (need a positive slope at "
                         "the left end and a negative one at the right)")
    out = []
    guard = 0
    while len(out) < n and guard < 500 * n:
        guard += 1
        x, j = draw(xs, h, m, z, lm)
        if not np.isfinite(x) or x <= lo or x >= hi:
            continue
        upper = h[j] + m[j] * (x - xs[j])          # value of the hull at x
        if np.log(max(rng.random(), 1e-300)) <= float(log_pdf(x)) - upper:
            out.append(x)
        elif len(xs) < max_points:
            xs = sorted(xs + [x])                  # adapt: tighten the hull
            h, m, z = hull(xs)
            new_lm = log_masses(xs, h, m, z)
            if new_lm is not None:
                lm = new_lm
    return np.array(out)


def ratio_of_uniforms(pdf, n: int = 1000, u_max: float = 1.0, v_range=(-1.0, 1.0),
                      rng=None, max_tries: int = 10_000_000):
    """Ratio-of-uniforms method: sample ``(u, v)`` and return ``v/u``.

    Accepts when ``u <= sqrt(pdf(v/u))``, giving exact samples without an
    explicit envelope function.
    """
    rng = np.random.default_rng(rng)
    out = []
    tries = 0
    while len(out) < n and tries < max_tries:
        u = rng.uniform(0, u_max)
        v = rng.uniform(*v_range)
        tries += 1
        if u > 0 and u <= np.sqrt(max(pdf(v / u), 0.0)):
            out.append(v / u)
    return np.array(out), (len(out) / tries if tries else 0.0)


def alias_table(probabilities):
    """Build Walker's alias table for O(1) sampling from a discrete distribution."""
    p = as_vector(probabilities).astype(float)
    p = p / p.sum()
    n = p.size
    prob = np.zeros(n)
    alias = np.zeros(n, dtype=int)
    scaled = p * n
    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] + scaled[s] - 1.0
        (small if scaled[l] < 1.0 else large).append(l)
    for i in large + small:
        prob[i] = 1.0
        alias[i] = i
    return prob, alias


def sample_discrete(probabilities, n: int = 1000, rng=None, method: str = "alias"):
    """Sample from a discrete distribution by the alias table or by inversion."""
    rng = np.random.default_rng(rng)
    p = as_vector(probabilities).astype(float)
    p = p / p.sum()
    if method == "alias":
        prob, alias = alias_table(p)
        k = p.size
        i = rng.integers(k, size=n)
        u = rng.random(n)
        return np.where(u < prob[i], i, alias[i])
    cdf = np.cumsum(p)
    return np.searchsorted(cdf, rng.random(n))


def multivariate_normal(mean, cov, n: int = 1000, rng=None):
    """Multivariate normal samples via the Cholesky factor of the covariance."""
    rng = np.random.default_rng(rng)
    mean = as_vector(mean)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # fall back to the eigendecomposition for a semi-definite covariance
        w, V = np.linalg.eigh(cov)
        L = V @ np.diag(np.sqrt(np.maximum(w, 0.0)))
    z = rng.standard_normal((n, mean.size))
    return mean + z @ L.T


def gibbs_bivariate_normal(rho: float, n: int = 5000, burn: int = 500, rng=None):
    """Gibbs sampler for a bivariate normal with correlation ``rho``.

    The textbook illustration: each conditional is a one-dimensional normal.
    """
    rng = np.random.default_rng(rng)
    x = y = 0.0
    out = np.empty((n, 2))
    s = np.sqrt(1 - rho * rho)
    for i in range(n + burn):
        x = rng.normal(rho * y, s)
        y = rng.normal(rho * x, s)
        if i >= burn:
            out[i - burn] = (x, y)
    return out


def bootstrap(data, statistic, n_resamples: int = 10000, rng=None,
              confidence: float = 0.95):
    """Nonparametric bootstrap: resample with replacement to get a sampling
    distribution for any statistic.

    Returns the estimate, standard error, percentile interval and the replicates.
    """
    rng = np.random.default_rng(rng)
    data = np.asarray(data)
    n = len(data)
    theta = float(statistic(data))
    reps = np.empty(n_resamples)
    for b in range(n_resamples):
        reps[b] = statistic(data[rng.integers(0, n, n)])
    alpha = (1 - confidence) / 2
    return {
        "estimate": theta,
        "std_error": float(np.std(reps, ddof=1)),
        "ci": (float(np.quantile(reps, alpha)), float(np.quantile(reps, 1 - alpha))),
        "bias": float(np.mean(reps) - theta),
        "replicates": reps,
    }


def jackknife(data, statistic):
    """Leave-one-out jackknife estimate of bias and standard error."""
    data = np.asarray(data)
    n = len(data)
    theta = float(statistic(data))
    loo = np.array([float(statistic(np.delete(data, i))) for i in range(n)])
    mean_loo = loo.mean()
    return {
        "estimate": theta,
        "bias": float((n - 1) * (mean_loo - theta)),
        "std_error": float(np.sqrt((n - 1) / n * np.sum((loo - mean_loo) ** 2))),
        "values": loo,
    }


def permutation_test(a, b, statistic=None, n_permutations: int = 10000, rng=None):
    """Two-sample permutation test.

    Makes no distributional assumption: the null distribution is built by
    relabelling the pooled data.
    """
    rng = np.random.default_rng(rng)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    statistic = statistic or (lambda u, v: abs(np.mean(u) - np.mean(v)))
    observed = float(statistic(a, b))
    pooled = np.concatenate([a, b])
    na = a.size
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(pooled)
        if float(statistic(perm[:na], perm[na:])) >= observed:
            count += 1
    return {"statistic": observed,
            "p_value": (count + 1) / (n_permutations + 1)}
