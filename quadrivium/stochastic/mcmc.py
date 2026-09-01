"""Markov chain Monte Carlo.

When direct sampling is impossible, MCMC builds a Markov chain whose stationary
distribution is the target, so the samples are correlated but asymptotically
correct. Diagnostics matter as much as the samplers themselves.
"""

from __future__ import annotations

import numpy as np

from ..core.utils import as_vector

__all__ = [
    "Chain",
    "metropolis_hastings",
    "random_walk_metropolis",
    "gibbs_sampler",
    "hamiltonian_mc",
    "nuts_lite",
    "slice_sampler",
    "parallel_tempering",
    "effective_sample_size",
    "gelman_rubin",
    "autocorrelation_time",
    "acceptance_rate",
]


class Chain(np.ndarray):
    """Array of MCMC draws that also carries the sampler's diagnostics.

    A plain ``ndarray`` cannot hold extra attributes, so the acceptance rate
    and swap count travel with the samples on this thin subclass.
    """

    def __new__(cls, data, acceptance=None, swaps=None):
        obj = np.asarray(data, dtype=float).view(cls)
        obj.acceptance = acceptance
        obj.swaps = swaps
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.acceptance = getattr(obj, "acceptance", None)
        self.swaps = getattr(obj, "swaps", None)


def metropolis_hastings(log_target, x0, proposal, log_proposal_ratio=None,
                        n: int = 10000, burn: int = 1000, thin: int = 1, rng=None):
    """Metropolis-Hastings.

    ``proposal(x, rng)`` returns a candidate; ``log_proposal_ratio(x, y)`` is
    ``log q(x|y) - log q(y|x)`` and defaults to 0 for a symmetric proposal.
    """
    rng = np.random.default_rng(rng)
    x = as_vector(x0).copy()
    lp = float(log_target(x))
    kept = []
    accepted = 0
    total = n + burn
    for i in range(total):
        y = as_vector(proposal(x, rng))
        lp_y = float(log_target(y))
        log_ratio = lp_y - lp
        if log_proposal_ratio is not None:
            log_ratio += float(log_proposal_ratio(x, y))
        if np.log(max(rng.random(), 1e-300)) < log_ratio:
            x, lp = y, lp_y
            accepted += 1
        if i >= burn and (i - burn) % thin == 0:
            kept.append(x.copy())
    return Chain(kept, acceptance=accepted / total)


def random_walk_metropolis(log_target, x0, step: float = 1.0, n: int = 10000,
                           burn: int = 1000, thin: int = 1, rng=None):
    """Random walk Metropolis with an isotropic Gaussian proposal.

    An acceptance rate near 0.234 is optimal in high dimension; tune ``step``
    toward that.
    """
    def proposal(x, r):
        return x + step * r.standard_normal(x.size)

    return metropolis_hastings(log_target, x0, proposal, None, n, burn, thin, rng)


def gibbs_sampler(conditionals, x0, n: int = 10000, burn: int = 1000, rng=None):
    """Systematic-scan Gibbs sampler.

    ``conditionals[i](x, rng)`` draws component ``i`` from its full conditional
    given the current state; every draw is accepted by construction.
    """
    rng = np.random.default_rng(rng)
    x = as_vector(x0).copy()
    out = []
    for i in range(n + burn):
        for j, cond in enumerate(conditionals):
            x[j] = float(cond(x, rng))
        if i >= burn:
            out.append(x.copy())
    return np.array(out)


def hamiltonian_mc(log_target, grad_log_target, x0, step: float = 0.1,
                   n_leapfrog: int = 20, n: int = 5000, burn: int = 500, rng=None,
                   mass=None):
    """Hamiltonian Monte Carlo.

    Augments the state with momentum and follows Hamiltonian dynamics with a
    leapfrog integrator, so proposals travel far while keeping a high
    acceptance rate. The integrator must be symplectic and reversible for
    detailed balance to hold.
    """
    rng = np.random.default_rng(rng)
    x = as_vector(x0).copy()
    d = x.size
    M = np.ones(d) if mass is None else as_vector(mass)
    out = []
    accepted = 0
    total = n + burn
    for i in range(total):
        p = rng.standard_normal(d) * np.sqrt(M)
        x_new = x.copy()
        p_new = p.copy()
        H0 = -float(log_target(x)) + 0.5 * float(np.sum(p * p / M))
        # leapfrog: half kick, full drifts, half kick
        p_new = p_new + 0.5 * step * as_vector(grad_log_target(x_new))
        for _ in range(n_leapfrog):
            x_new = x_new + step * p_new / M
            if _ < n_leapfrog - 1:
                p_new = p_new + step * as_vector(grad_log_target(x_new))
        p_new = p_new + 0.5 * step * as_vector(grad_log_target(x_new))
        p_new = -p_new                       # reversibility
        H1 = -float(log_target(x_new)) + 0.5 * float(np.sum(p_new * p_new / M))
        if np.log(max(rng.random(), 1e-300)) < H0 - H1:
            x = x_new
            accepted += 1
        if i >= burn:
            out.append(x.copy())
    return Chain(out, acceptance=accepted / total)


def nuts_lite(log_target, grad_log_target, x0, step: float = 0.1, n: int = 5000,
              burn: int = 500, max_depth: int = 8, rng=None):
    """HMC with a randomized trajectory length.

    Captures the practical benefit of NUTS -- no hand-tuned path length --
    without the full recursive no-U-turn tree.
    """
    rng = np.random.default_rng(rng)
    x = as_vector(x0).copy()
    out = []
    accepted = 0
    for i in range(n + burn):
        L = int(rng.integers(1, 2**max_depth // 8 + 2))
        sub = hamiltonian_mc(log_target, grad_log_target, x, step, L, 1, 0, rng)
        x_new = sub[-1]
        if not np.allclose(x_new, x):
            accepted += 1
        x = x_new
        if i >= burn:
            out.append(x.copy())
    return Chain(out, acceptance=accepted / (n + burn))


def slice_sampler(log_target, x0, w: float = 1.0, n: int = 10000, burn: int = 1000,
                  rng=None, max_steps: int = 100):
    """Univariate slice sampling with stepping-out and shrinkage.

    Needs no proposal tuning: the step size adapts through the interval
    construction, and every draw is accepted.
    """
    rng = np.random.default_rng(rng)
    x = float(np.atleast_1d(x0)[0])
    out = []
    for i in range(n + burn):
        logy = float(log_target(x)) + np.log(max(rng.random(), 1e-300))
        # stepping out
        u = rng.random()
        L = x - w * u
        R = L + w
        j = 0
        while j < max_steps and float(log_target(L)) > logy:
            L -= w
            j += 1
        j = 0
        while j < max_steps and float(log_target(R)) > logy:
            R += w
            j += 1
        # shrinkage
        for _ in range(max_steps):
            x_new = rng.uniform(L, R)
            if float(log_target(x_new)) > logy:
                x = x_new
                break
            if x_new < x:
                L = x_new
            else:
                R = x_new
        if i >= burn:
            out.append(x)
    return np.array(out)


def parallel_tempering(log_target, x0, temperatures=(1.0, 2.0, 4.0, 8.0),
                       step: float = 1.0, n: int = 10000, burn: int = 1000,
                       swap_every: int = 10, rng=None):
    """Parallel tempering (replica exchange).

    Runs chains at several temperatures and swaps neighbours, so the hot chains
    cross barriers and carry the cold chain out of local modes -- the standard
    fix for multimodal targets.
    """
    rng = np.random.default_rng(rng)
    T = np.asarray(temperatures, dtype=float)
    k = T.size
    X = np.array([as_vector(x0).copy() for _ in range(k)])
    lp = np.array([float(log_target(x)) for x in X])
    out = []
    swaps = 0
    for i in range(n + burn):
        for j in range(k):
            y = X[j] + step * np.sqrt(T[j]) * rng.standard_normal(X[j].size)
            lp_y = float(log_target(y))
            if np.log(max(rng.random(), 1e-300)) < (lp_y - lp[j]) / T[j]:
                X[j], lp[j] = y, lp_y
        if i % swap_every == 0:
            for j in range(k - 1):
                delta = (1.0 / T[j] - 1.0 / T[j + 1]) * (lp[j + 1] - lp[j])
                if np.log(max(rng.random(), 1e-300)) < delta:
                    X[[j, j + 1]] = X[[j + 1, j]]
                    lp[[j, j + 1]] = lp[[j + 1, j]]
                    swaps += 1
        if i >= burn:
            out.append(X[0].copy())
    return Chain(out, swaps=swaps)


def autocorrelation_time(chain, max_lag=None):
    """Integrated autocorrelation time, using Geyer's initial positive sequence.

    ``tau`` measures how many steps the chain needs to produce one independent
    sample.
    """
    from ..transforms.signal import autocorrelation

    x = np.asarray(chain, dtype=float)
    if x.ndim > 1:
        return np.array([autocorrelation_time(x[:, j], max_lag)
                         for j in range(x.shape[1])])
    n = x.size
    max_lag = max_lag or min(n - 1, 1000)
    rho = autocorrelation(x, normalize=True, max_lag=max_lag)
    tau = 1.0
    for k in range(1, len(rho) - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break                     # Geyer's truncation
        tau += 2.0 * pair
    return float(max(tau, 1.0))


def effective_sample_size(chain):
    """Effective sample size ``N / tau``: the number of independent samples."""
    x = np.asarray(chain, dtype=float)
    n = x.shape[0]
    tau = autocorrelation_time(x)
    return n / tau if np.isscalar(tau) else n / np.asarray(tau)


def gelman_rubin(chains):
    """Gelman-Rubin ``R-hat`` from several independent chains.

    Compares within-chain and between-chain variance; values above about 1.01
    indicate the chains have not mixed.
    """
    C = np.asarray(chains, dtype=float)
    if C.ndim == 2:
        C = C[:, :, None]
    m, n, d = C.shape
    means = C.mean(axis=1)
    B = n * np.var(means, axis=0, ddof=1)
    W = np.mean(np.var(C, axis=1, ddof=1), axis=0)
    var_hat = (n - 1) / n * W + B / n
    rhat = np.sqrt(var_hat / np.where(W > 0, W, 1.0))
    return rhat if d > 1 else float(rhat[0])


def acceptance_rate(chain):
    """Fraction of proposals accepted, if the sampler recorded it."""
    return getattr(chain, "acceptance", None)
