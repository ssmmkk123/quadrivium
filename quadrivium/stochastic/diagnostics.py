"""Rank-based diagnostics for multiple Monte Carlo chains."""
from __future__ import annotations
import math
from statistics import NormalDist
from .. import numeric as np

__all__ = ["split_rhat", "bulk_ess", "tail_ess", "mcse", "mcmc_diagnostics"]


def _chains(chains):
    c = np.asarray(chains, dtype=float)
    if c.ndim < 2 or c.shape[0] < 2 or c.shape[1] < 4 or not np.all(np.isfinite(c)):
        raise ValueError("need at least two finite chains with four draws each")
    return c.reshape(c.shape[:2] + (-1,)), c.shape[2:]


def _split(c):
    n = c.shape[1] // 2
    return np.concatenate((c[:, :n], c[:, -n:]), axis=0)


def _rank_normalize(c):
    flat = c.ravel()
    order = np.argsort(flat)
    ranked = np.empty(flat.size)
    normal = NormalDist()
    i = 0
    while i < flat.size:
        j = i + 1
        while j < flat.size and flat[order[j]] == flat[order[i]]:
            j += 1
        probability = ((i + j + 1) * 0.5 - 0.375) / (flat.size + 0.25)
        ranked[order[i:j]] = normal.inv_cdf(probability)
        i = j
    return ranked.reshape(c.shape)


def _rhat(c):
    if np.all(c == c[:, :1]):
        return math.nan if np.all(c == c[0, 0]) else math.inf
    n = c.shape[1]
    within = float(np.mean(np.var(c, axis=1, ddof=1)))
    between = float(np.var(np.mean(c, axis=1), ddof=1))
    if within == 0:
        return math.nan if between == 0 else math.inf
    return math.sqrt((n - 1) / n + between / within)


def _ess(c):
    if np.all(c == c[0, 0]):
        return math.nan
    m, n = c.shape
    within = float(np.mean(np.var(c, axis=1, ddof=1)))
    variance = (n - 1) / n * within + float(np.var(c.mean(axis=1), ddof=1))
    if variance == 0:
        # A constant trace cannot establish mixing or a sampling error.
        return math.nan
    acov = np.zeros(n)
    length = 1 << (2 * n - 1).bit_length()
    for row in c:
        centered = row - np.mean(row)
        spec = np.fft.fft(centered, n=length)
        acov += np.real(np.fft.ifft(spec * np.conj(spec)))[:n] / (n * m)
    rho = 1 - (within - acov) / variance
    rho[0] = 1.0
    total, previous = 0.0, math.inf
    for t in range(0, n - 1, 2):
        pair = float(rho[t] + rho[t + 1])
        if pair < 0:
            break
        pair = min(previous, pair)
        total += pair
        previous = pair
    tau = max(1.0, -1 + 2 * total)
    return min(float(m * n), m * n / tau)


def _apply(chains, fn):
    c, shape = _chains(chains)
    out = np.array([fn(c[:, :, d]) for d in range(c.shape[2])]).reshape(shape)
    return float(out) if not shape else out


def split_rhat(chains):
    """Maximum rank/folded split R-hat; NaN for identically constant chains.

    Different constant values in different chains yield infinity, indicating
    disagreement. Equal constant traces provide no estimate of mixing.
    """
    def one(c):
        split = _split(c)
        return max(_rhat(_rank_normalize(split)),
                   _rhat(_rank_normalize(np.abs(split - np.median(split)))))
    return _apply(chains, one)


def bulk_ess(chains):
    """Rank-normalized effective sample size using positive monotone lag pairs."""
    return _apply(chains, lambda c: _ess(_rank_normalize(_split(c))))


def tail_ess(chains, probability=0.05):
    """Minimum indicator ESS in the lower and upper probability tails."""
    if not 0 < probability < 0.5:
        raise ValueError("probability must lie between zero and one half")
    def one(c):
        c = _split(c)
        low, high = np.quantile(c, probability), np.quantile(c, 1 - probability)
        return min(_ess((c <= low).astype(float)), _ess((c >= high).astype(float)))
    return _apply(chains, one)


def mcse(chains):
    """Monte Carlo standard error of the posterior mean, per parameter."""
    return _apply(chains, lambda c: math.sqrt(float(np.var(c, ddof=1)) / _ess(_split(c))))


def mcmc_diagnostics(chains):
    """Return split R-hat, bulk/tail ESS and mean Monte Carlo standard errors.

    Degenerate constant traces have undefined (NaN) diagnostics; their zero
    empirical spread must not be mistaken for a precisely estimated posterior.
    """
    return {"rhat": split_rhat(chains), "ess_bulk": bulk_ess(chains),
            "ess_tail": tail_ess(chains), "mcse_mean": mcse(chains)}
