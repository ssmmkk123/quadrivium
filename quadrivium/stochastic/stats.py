"""Descriptive statistics, regression and hypothesis testing."""

from __future__ import annotations

from .. import numeric as np

from ..core.utils import as_vector

__all__ = [
    "describe",
    "mean",
    "variance",
    "skewness",
    "kurtosis",
    "median",
    "quantile",
    "mode_estimate",
    "covariance_matrix",
    "correlation_matrix",
    "linear_regression",
    "polynomial_regression",
    "logistic_regression",
    "pca",
    "welford_mean_var",
    "histogram_density",
    "kernel_density",
    "t_test",
    "chi_square_test",
    "ks_test",
    "anova_one_way",
    "confidence_interval",
]


def mean(x):
    """Arithmetic mean."""
    return float(np.mean(as_vector(x)))


def variance(x, ddof: int = 1):
    """Sample variance (``ddof=1`` by default, the unbiased estimator)."""
    return float(np.var(as_vector(x), ddof=ddof))


def median(x):
    """Median by sorting."""
    return float(np.median(as_vector(x)))


def quantile(x, q, method: str = "linear"):
    """Empirical quantile with linear interpolation between order statistics."""
    return np.quantile(as_vector(x), q, method=method)


def skewness(x, bias: bool = False):
    """Sample skewness; ``bias=False`` applies the usual small-sample correction."""
    x = as_vector(x)
    n = x.size
    m = x.mean()
    s = np.sqrt(np.mean((x - m) ** 2))
    g1 = float(np.mean((x - m) ** 3) / s**3) if s > 0 else 0.0
    if bias or n < 3:
        return g1
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def kurtosis(x, excess: bool = True, bias: bool = False):
    """Sample kurtosis; ``excess=True`` subtracts 3 (normal reference)."""
    x = as_vector(x)
    n = x.size
    m = x.mean()
    s2 = np.mean((x - m) ** 2)
    g2 = float(np.mean((x - m) ** 4) / s2**2) if s2 > 0 else 0.0
    if not bias and n > 3:
        g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * (g2 - 3) + 6) + 3
    return g2 - 3.0 if excess else g2


def mode_estimate(x, bins: int = 50):
    """Mode estimated from the peak of a histogram."""
    x = as_vector(x)
    counts, edges = np.histogram(x, bins=bins)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i + 1]))


def welford_mean_var(x):
    """Welford's online algorithm for the mean and variance.

    One pass, and numerically stable where the naive
    ``E[x^2] - E[x]^2`` formula catastrophically cancels.
    """
    n = 0
    m = 0.0
    m2 = 0.0
    for v in as_vector(x):
        n += 1
        d = v - m
        m += d / n
        m2 += d * (v - m)
    return m, (m2 / (n - 1) if n > 1 else 0.0)


def describe(x):
    """Summary statistics of a sample."""
    x = as_vector(x)
    # All three quantiles share a single sorted copy of the sample.
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    sample_variance = variance(x)
    return {
        "n": int(x.size),
        "mean": mean(x),
        "std": float(np.sqrt(sample_variance)) if x.size > 1 else 0.0,
        "variance": sample_variance,
        "min": float(x.min()),
        "q1": float(q1),
        "median": float(q2),
        "q3": float(q3),
        "max": float(x.max()),
        "skewness": skewness(x),
        "kurtosis": kurtosis(x),
    }


def covariance_matrix(X, ddof: int = 1):
    """Covariance matrix of observations stored as rows."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Xc = X - X.mean(axis=0)
    return (Xc.T @ Xc) / (X.shape[0] - ddof)


def correlation_matrix(X):
    """Pearson correlation matrix."""
    C = covariance_matrix(X)
    d = np.sqrt(np.diag(C))
    d = np.where(d > 0, d, 1.0)
    return C / np.outer(d, d)


def linear_regression(X, y, intercept: bool = True):
    """Ordinary least squares with the usual inferential summary.

    Returns coefficients, standard errors, t statistics, ``R^2`` and the
    residuals.
    """
    from ..linalg.lstsq import qr_least_squares

    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(y):
        X = X.T
    y = as_vector(y)
    A = np.hstack([np.ones((X.shape[0], 1)), X]) if intercept else X
    beta = qr_least_squares(A, y)
    resid = y - A @ beta
    n, p = A.shape
    dof = max(n - p, 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.diag(cov))
        tvals = beta / np.where(se > 0, se, 1.0)
    except np.linalg.LinAlgError:
        cov, se, tvals = None, None, None
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return {
        "coefficients": beta,
        "std_errors": se,
        "t_values": tvals,
        "residuals": resid,
        "r_squared": r2,
        "adjusted_r_squared": 1 - (1 - r2) * (n - 1) / dof if ss_tot > 0 else np.nan,
        "sigma2": s2,
        "covariance": cov,
        "dof": dof,
    }


def polynomial_regression(x, y, degree: int = 2):
    """Least squares polynomial fit using a Vandermonde design matrix."""
    x, y = as_vector(x), as_vector(y)
    A = np.vander(x, degree + 1, increasing=True)
    res = linear_regression(A[:, 1:], y, intercept=True)
    res["poly_coefficients"] = res["coefficients"]
    return res


def logistic_regression(X, y, tol: float = 1e-10, max_iter: int = 100,
                        intercept: bool = True, ridge: float = 1e-8):
    """Logistic regression by iteratively reweighted least squares.

    IRLS is Newton's method on the log-likelihood; the small ridge term keeps
    the Hessian invertible under perfect separation.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = as_vector(y)
    if X.shape[0] != y.size:
        X = X.T
    A = np.hstack([np.ones((X.shape[0], 1)), X]) if intercept else X
    n, p = A.shape
    beta = np.zeros(p)
    for k in range(1, max_iter + 1):
        eta = A @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        W = np.maximum(mu * (1 - mu), 1e-10)
        z = eta + (y - mu) / W
        H = A.T @ (A * W[:, None]) + ridge * np.eye(p)
        beta_new = np.linalg.solve(H, A.T @ (W * z))
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = A @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
    ll = float(np.sum(y * np.log(np.clip(mu, 1e-300, 1)) +
                      (1 - y) * np.log(np.clip(1 - mu, 1e-300, 1))))
    return {"coefficients": beta, "iterations": k, "fitted": mu,
            "log_likelihood": ll,
            "accuracy": float(np.mean((mu > 0.5) == (y > 0.5)))}


def pca(X, n_components=None, center: bool = True):
    """Principal component analysis via the SVD of the centred data."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s**2 / (X.shape[0] - 1)
    k = n_components or s.size
    return {
        "components": Vt[:k],
        "explained_variance": var[:k],
        "explained_variance_ratio": var[:k] / var.sum(),
        "singular_values": s[:k],
        "mean": mu,
        "scores": Xc @ Vt[:k].T,
    }


def histogram_density(x, bins: int = 30, range_=None):
    """Histogram density estimate: returns ``(centres, density)``."""
    x = as_vector(x)
    counts, edges = np.histogram(x, bins=bins, range=range_, density=True)
    return 0.5 * (edges[:-1] + edges[1:]), counts


# Pairwise distances are evaluated in tiles so the working set does not grow
# as sample_count * point_count. One float64 tile occupies at most 1 MiB.
_KDE_TILE_ELEMENTS = 131072


def kernel_density(x, points=None, bandwidth=None, kernel: str = "gaussian"):
    """Kernel density estimate.

    The default bandwidth is Silverman's rule of thumb, using a unit scale
    for a singleton or constant sample. Pairwise evaluations use bounded
    workspace, even when both the sample and evaluation grid are large.
    Samples and points must be finite and bandwidth must be positive.
    """
    x = as_vector(x)
    n = x.size
    if n == 0 or not np.all(np.isfinite(x)):
        raise ValueError("kernel density requires a nonempty finite sample")
    if kernel not in ("gaussian", "epanechnikov", "uniform", "triangular"):
        raise ValueError(f"unknown kernel {kernel!r}")
    if bandwidth is None:
        std = float(np.std(x, ddof=1)) if n > 1 else 0.0
        q1, q3 = np.quantile(x, [0.25, 0.75])
        sigma = min(std, (q3 - q1) / 1.349)
        sigma = sigma if sigma > 0 else std or 1.0
        bandwidth = 0.9 * sigma * n ** (-1 / 5)
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    pts = np.linspace(x.min() - 3 * bandwidth, x.max() + 3 * bandwidth, 200) \
        if points is None else as_vector(points)
    if not np.all(np.isfinite(pts)):
        raise ValueError("evaluation points must be finite")
    density = np.zeros(pts.size)
    sample_step = min(n, _KDE_TILE_ELEMENTS)
    point_step = max(1, _KDE_TILE_ELEMENTS // sample_step)
    for i in range(0, pts.size, point_step):
        stop = min(i + point_step, pts.size)
        for j in range(0, n, sample_step):
            # Overflow in distances far outside the kernel's support means a
            # zero contribution, including for the Gaussian tail.
            with np.errstate(over="ignore"):
                u = pts[i:stop, None] - x[None, j:j + sample_step]
                u /= bandwidth
                if kernel == "gaussian":
                    np.square(u, out=u)
                    u *= -0.5
                    np.exp(u, out=u)
                    u /= np.sqrt(2 * np.pi)
                elif kernel == "uniform":
                    np.abs(u, out=u)
                    np.less_equal(u, 1, out=u)
                    u *= 0.5
                else:
                    # Clip before squaring to keep compact kernels finite
                    # even for very distant evaluation points.
                    np.abs(u, out=u)
                    np.minimum(u, 1.0, out=u)
                    if kernel == "epanechnikov":
                        np.square(u, out=u)
                    np.subtract(1.0, u, out=u)
                    if kernel == "epanechnikov":
                        u *= 0.75
            density[i:stop] += u.sum(axis=1)
    density /= n
    density /= bandwidth
    return pts, density


def _t_cdf(t: float, dof: float) -> float:
    """CDF of Student's t via the incomplete beta function."""
    from ..special.functions import incomplete_beta

    x = dof / (dof + t * t)
    p = 0.5 * incomplete_beta(dof / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_test(a, b=None, mu0: float = 0.0, paired: bool = False,
           equal_var: bool = True):
    """One-sample, paired or two-sample t test.

    With ``equal_var=False`` this is Welch's test, which does not assume equal
    variances.
    """
    a = as_vector(a)
    if b is None or paired:
        d = a - mu0 if b is None else a - as_vector(b)
        n = d.size
        se = np.std(d, ddof=1) / np.sqrt(n)
        t = float(d.mean() / se) if se > 0 else np.inf
        dof = n - 1
    else:
        b = as_vector(b)
        na, nb = a.size, b.size
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        if equal_var:
            sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
            se = np.sqrt(sp2 * (1 / na + 1 / nb))
            dof = na + nb - 2
        else:
            se = np.sqrt(va / na + vb / nb)
            dof = (va / na + vb / nb) ** 2 / (
                (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        t = float((a.mean() - b.mean()) / se) if se > 0 else np.inf
    p = 2.0 * (1.0 - _t_cdf(abs(t), dof))
    return {"statistic": t, "dof": float(dof), "p_value": float(min(max(p, 0.0), 1.0))}


def chi_square_test(observed, expected=None):
    """Chi-square goodness of fit test."""
    from ..special.functions import regularized_gamma_q

    o = as_vector(observed)
    e = np.full_like(o, o.mean()) if expected is None else as_vector(expected)
    stat = float(np.sum((o - e) ** 2 / np.where(e > 0, e, 1.0)))
    dof = o.size - 1
    p = regularized_gamma_q(dof / 2.0, stat / 2.0) if dof > 0 else 1.0
    return {"statistic": stat, "dof": dof, "p_value": float(p)}


def ks_test(a, cdf=None, b=None):
    """Kolmogorov-Smirnov test, one sample against a CDF or two samples."""
    a = np.sort(as_vector(a))
    n = a.size
    if b is not None:
        b = np.sort(as_vector(b))
        m = b.size
        grid = np.concatenate([a, b])
        Fa = np.searchsorted(a, grid, side="right") / n
        Fb = np.searchsorted(b, grid, side="right") / m
        d = float(np.max(np.abs(Fa - Fb)))
        ne = n * m / (n + m)
    else:
        F = _cdf_values(cdf, a)
        d_plus = np.max(np.arange(1, n + 1) / n - F)
        d_minus = np.max(F - np.arange(n) / n)
        d = float(max(d_plus, d_minus))
        ne = n
    # asymptotic Kolmogorov distribution
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    k = np.arange(1, 101)
    p = 2.0 * float(np.sum((-1.0) ** (k - 1) * np.exp(-2 * k * k * lam * lam)))
    return {"statistic": d, "p_value": float(min(max(p, 0.0), 1.0))}


def _cdf_values(cdf, a):
    """Evaluate ``cdf`` at every sorted sample, on the whole array if it can.

    Most CDFs handed in here are NumPy expressions that take an array
    directly, which is one call instead of one per sample. A CDF written for
    scalars raises or returns the wrong shape; it then gets the element loop,
    and any genuine error inside it surfaces from that loop unchanged.
    """
    try:
        out = np.asarray(cdf(a), dtype=float)
    except Exception:  # noqa: BLE001 - scalar-only CDF; retried elementwise
        out = None
    if out is None or out.shape != a.shape:
        out = np.array([cdf(v) for v in a], dtype=float)
    return out


def anova_one_way(*groups):
    """One-way analysis of variance across several groups."""
    from ..special.functions import incomplete_beta

    gs = [as_vector(g) for g in groups]
    k = len(gs)
    n = sum(g.size for g in gs)
    grand = np.concatenate(gs).mean()
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in gs)
    ss_within = sum(float(np.sum((g - g.mean()) ** 2)) for g in gs)
    df_b, df_w = k - 1, n - k
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w
    F = float(ms_b / ms_w) if ms_w > 0 else np.inf
    x = df_w / (df_w + df_b * F) if np.isfinite(F) else 0.0
    p = float(incomplete_beta(df_w / 2.0, df_b / 2.0, x)) if np.isfinite(F) else 0.0
    return {"F": F, "df_between": df_b, "df_within": df_w,
            "p_value": float(min(max(p, 0.0), 1.0))}


def confidence_interval(x, confidence: float = 0.95, method: str = "t"):
    """Confidence interval for the mean (``'t'``, ``'normal'`` or ``'bootstrap'``)."""
    x = as_vector(x)
    n = x.size
    m = x.mean()
    se = np.std(x, ddof=1) / np.sqrt(n)
    if method == "bootstrap":
        from .sampling import bootstrap

        return bootstrap(x, np.mean, 10000, confidence=confidence)["ci"]
    if method == "normal":
        from ..special.functions import erfinv

        z = np.sqrt(2) * erfinv(confidence)
        return (m - z * se, m + z * se)
    # invert the t CDF by bisection
    target = 0.5 * (1 + confidence)
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, n - 1) < target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return (m - t * se, m + t * se)
