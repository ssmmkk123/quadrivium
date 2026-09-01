"""Global optimization: stochastic and population-based search.

Local methods find the nearest minimum; these explore the whole domain, trading
guarantees for the ability to escape local optima.
"""

from __future__ import annotations

import numpy as np

from ..core.types import OptimizeResult
from ..core.utils import CountedFunction, as_vector

__all__ = [
    "simulated_annealing",
    "particle_swarm",
    "differential_evolution",
    "genetic_algorithm",
    "basin_hopping",
    "random_search",
    "cma_es",
    "cma_es_lite",
    "dual_annealing_lite",
]


def _bounds_arrays(bounds):
    b = np.atleast_2d(np.asarray(bounds, dtype=float))
    return b[:, 0], b[:, 1]


def simulated_annealing(f, x0, bounds=None, T0: float = 10.0, cooling: float = 0.995,
                        max_iter: int = 20000, step: float = 0.5, rng=None,
                        T_min: float = 1e-12):
    """Simulated annealing with geometric cooling.

    Uphill moves are accepted with probability ``exp(-dE/T)``, so the search can
    leave a local basin early and settles as the temperature falls.
    """
    rng = np.random.default_rng(rng)
    fc = CountedFunction(lambda v: float(f(v)))
    x = as_vector(x0).copy()
    lo, hi = _bounds_arrays(bounds) if bounds is not None else (None, None)
    fx = fc(x)
    best, f_best = x.copy(), fx
    T = T0
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        if T < T_min:
            break
        cand = x + step * T / T0 * rng.standard_normal(x.size)
        if lo is not None:
            cand = np.clip(cand, lo, hi)
        f_cand = fc(cand)
        dE = f_cand - fx
        if dE < 0 or rng.random() < np.exp(-dE / T):
            x, fx = cand, f_cand
            if fx < f_best:
                best, f_best = x.copy(), fx
                history.append(best.copy())
        T *= cooling
    return OptimizeResult(best, float(f_best), None, None, k, True, fc.calls, 0,
                          "simulated_annealing", history, "annealing completed")


def particle_swarm(f, bounds, n_particles: int = 40, max_iter: int = 500,
                   w: float = 0.729, c1: float = 1.49445, c2: float = 1.49445,
                   rng=None, tol: float = 1e-12):
    """Particle swarm optimization.

    Particles are drawn toward their own best position and the swarm's best;
    the inertia weight ``w`` balances exploration against convergence.
    """
    rng = np.random.default_rng(rng)
    fc = CountedFunction(lambda v: float(f(v)))
    lo, hi = _bounds_arrays(bounds)
    n = lo.size
    X = rng.uniform(lo, hi, size=(n_particles, n))
    V = rng.uniform(-abs(hi - lo), abs(hi - lo), size=(n_particles, n)) * 0.1
    F = np.array([fc(p) for p in X])
    P, FP = X.copy(), F.copy()
    g_idx = int(np.argmin(FP))
    g_best, f_best = P[g_idx].copy(), FP[g_idx]
    history = [g_best.copy()]
    for k in range(1, max_iter + 1):
        r1 = rng.random((n_particles, n))
        r2 = rng.random((n_particles, n))
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g_best - X)
        X = np.clip(X + V, lo, hi)
        F = np.array([fc(p) for p in X])
        better = F < FP
        P[better], FP[better] = X[better], F[better]
        i = int(np.argmin(FP))
        if FP[i] < f_best:
            g_best, f_best = P[i].copy(), FP[i]
            history.append(g_best.copy())
        if np.max(np.abs(X - g_best)) < tol:
            break
    return OptimizeResult(g_best, float(f_best), None, None, k, True, fc.calls, 0,
                          "particle_swarm", history, "swarm converged")


def differential_evolution(f, bounds, pop_size: int = 30, F: float = 0.8,
                           CR: float = 0.9, max_iter: int = 1000, rng=None,
                           tol: float = 1e-12, strategy: str = "rand1bin"):
    """Differential evolution.

    Mutation uses scaled differences of population members, so the step size
    adapts to the population's spread automatically.

    ``strategy='rand1bin'`` mutates around a random member and explores widely;
    ``'best1bin'`` mutates around the incumbent best and converges faster on
    unimodal problems but is prone to stalling in a local basin on multimodal
    ones.
    """
    rng = np.random.default_rng(rng)
    fc = CountedFunction(lambda v: float(f(v)))
    lo, hi = _bounds_arrays(bounds)
    n = lo.size
    pop = rng.uniform(lo, hi, size=(pop_size, n))
    fit = np.array([fc(p) for p in pop])
    history = []
    for k in range(1, max_iter + 1):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            if strategy == "best1bin":
                base = pop[int(np.argmin(fit))]
                a, b = pop[rng.choice(idxs, 2, replace=False)]
                mutant = base + F * (a - b)
            else:
                a, b, c = pop[rng.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
            mutant = np.clip(mutant, lo, hi)
            cross = rng.random(n) < CR
            if not cross.any():
                cross[rng.integers(n)] = True
            trial = np.where(cross, mutant, pop[i])
            f_trial = fc(trial)
            if f_trial < fit[i]:
                pop[i], fit[i] = trial, f_trial
        history.append(pop[int(np.argmin(fit))].copy())
        if np.max(fit) - np.min(fit) < tol:
            break
    best = int(np.argmin(fit))
    return OptimizeResult(pop[best], float(fit[best]), None, None, k, True,
                          fc.calls, 0, f"differential_evolution_{strategy}",
                          history, "population converged")


def genetic_algorithm(f, bounds, pop_size: int = 60, max_iter: int = 500,
                      crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                      elite: int = 2, rng=None, tournament: int = 3):
    """Real-coded genetic algorithm with tournament selection and BLX crossover."""
    rng = np.random.default_rng(rng)
    fc = CountedFunction(lambda v: float(f(v)))
    lo, hi = _bounds_arrays(bounds)
    n = lo.size
    pop = rng.uniform(lo, hi, size=(pop_size, n))
    fit = np.array([fc(p) for p in pop])
    history = []
    for k in range(1, max_iter + 1):
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        history.append(pop[0].copy())
        new = [pop[i].copy() for i in range(elite)]
        while len(new) < pop_size:
            # tournament selection
            p1 = pop[min(rng.choice(pop_size, tournament, replace=False))]
            p2 = pop[min(rng.choice(pop_size, tournament, replace=False))]
            if rng.random() < crossover_rate:
                alpha = rng.uniform(-0.25, 1.25, n)      # blend crossover
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()
            mask = rng.random(n) < mutation_rate
            child[mask] += 0.1 * (hi - lo)[mask] * rng.standard_normal(int(mask.sum()))
            new.append(np.clip(child, lo, hi))
        pop = np.array(new)
        fit = np.array([fc(p) for p in pop])
    best = int(np.argmin(fit))
    return OptimizeResult(pop[best], float(fit[best]), None, None, max_iter, True,
                          fc.calls, 0, "genetic_algorithm", history,
                          "generations completed")


def basin_hopping(f, x0, n_hops: int = 100, step: float = 0.5, T: float = 1.0,
                  rng=None, local=None, bounds=None):
    """Basin hopping: random perturbation followed by local minimization.

    Very effective when the landscape is a set of smooth basins.
    """
    from .quasinewton import bfgs

    rng = np.random.default_rng(rng)
    local = local or (lambda g, x: bfgs(g, x, tol=1e-10, max_iter=500))
    fc = CountedFunction(lambda v: float(f(v)))
    x = as_vector(x0).copy()
    res = local(fc, x)
    x, fx = as_vector(res.x), float(res.fun)
    best, f_best = x.copy(), fx
    history = [best.copy()]
    lo, hi = _bounds_arrays(bounds) if bounds is not None else (None, None)
    for k in range(1, n_hops + 1):
        cand = x + step * rng.standard_normal(x.size)
        if lo is not None:
            cand = np.clip(cand, lo, hi)
        r = local(fc, cand)
        xc, f_cand = as_vector(r.x), float(r.fun)
        if f_cand < fx or rng.random() < np.exp(-(f_cand - fx) / T):
            x, fx = xc, f_cand
        if f_cand < f_best:
            best, f_best = xc.copy(), f_cand
            history.append(best.copy())
    return OptimizeResult(best, float(f_best), None, None, n_hops, True, fc.calls,
                          0, "basin_hopping", history, "hops completed")


def random_search(f, bounds, n_samples: int = 10000, rng=None):
    """Pure random search: the baseline every other global method must beat."""
    rng = np.random.default_rng(rng)
    fc = CountedFunction(lambda v: float(f(v)))
    lo, hi = _bounds_arrays(bounds)
    X = rng.uniform(lo, hi, size=(n_samples, lo.size))
    F = np.array([fc(p) for p in X])
    i = int(np.argmin(F))
    return OptimizeResult(X[i], float(F[i]), None, None, n_samples, True, fc.calls,
                          0, "random_search", [], "sampling completed")


def cma_es(f, x0, sigma0: float = 0.5, pop_size=None, max_iter: int = 1000,
           rng=None, tol: float = 1e-14):
    """Covariance Matrix Adaptation Evolution Strategy.

    Samples from a multivariate normal, moves the mean toward the best points,
    and adapts the full covariance from the evolution path -- which lets the
    search learn the local metric and handle ill-conditioned, non-separable
    landscapes that defeat coordinate-wise methods.

    The default population ``lambda = 4 + 3 ln n`` is tuned for unimodal
    problems. On a strongly multimodal landscape that small a sample gets
    trapped; raise ``pop_size`` (a few dozen is usually enough) to restore
    global behaviour.
    """
    rng = np.random.default_rng(rng)
    fc = CountedFunction(lambda v: float(f(v)))
    xmean = as_vector(x0).copy()
    n = xmean.size
    sigma = float(sigma0)
    lam = pop_size or (4 + int(3 * np.log(n)))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    # adaptation rates (Hansen's standard settings)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    invsqrtC = np.eye(n)
    eigeneval = 0
    chiN = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
    counteval = 0
    best, f_best = xmean.copy(), fc(xmean)
    history = [best.copy()]
    for k in range(1, max_iter + 1):
        Z = rng.standard_normal((lam, n))
        X = xmean + sigma * (Z * D) @ B.T
        F = np.array([fc(p) for p in X])
        counteval += lam
        order = np.argsort(F)
        if F[order[0]] < f_best:
            best, f_best = X[order[0]].copy(), float(F[order[0]])
            history.append(best.copy())
        xold = xmean.copy()
        xmean = weights @ X[order[:mu]]
        # cumulative step-size adaptation
        ps = ((1 - cs) * ps
              + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (xmean - xold)) / sigma)
        hsig = (np.linalg.norm(ps)
                / np.sqrt(1 - (1 - cs) ** (2 * counteval / lam)) / chiN
                < 1.4 + 2.0 / (n + 1))
        pc = ((1 - cc) * pc
              + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma)
        artmp = (X[order[:mu]] - xold) / sigma
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (artmp.T * weights) @ artmp)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        # refresh the eigendecomposition occasionally, not every generation
        if counteval - eigeneval > lam / (c1 + cmu) / n / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T
            vals, B = np.linalg.eigh(C)
            vals = np.maximum(vals, 1e-20)
            D = np.sqrt(vals)
            invsqrtC = B @ np.diag(1.0 / D) @ B.T
        if sigma * np.max(D) < tol or not np.isfinite(sigma):
            break
    return OptimizeResult(best, float(f_best), None, None, k, True, fc.calls, 0,
                          "cma_es", history, "evolution completed")


def cma_es_lite(f, x0, **kwargs):
    """Alias of :func:`cma_es`."""
    return cma_es(f, x0, **kwargs)


def dual_annealing_lite(f, bounds, max_iter: int = 2000, rng=None, local=True,
                        restarts: int = 5):
    """Annealing over the whole domain, polished by a local search.

    A simplified dual annealing: global exploration then local refinement.
    """
    from .quasinewton import lbfgs

    rng = np.random.default_rng(rng)
    lo, hi = _bounds_arrays(bounds)
    width = float(np.max(hi - lo))
    best_x, best_f = None, np.inf
    calls = 0
    # A proposal of fixed size cannot explore a wide box, so the step is scaled
    # to the domain; several annealing runs are polished and the best kept.
    for _ in range(max(1, restarts)):
        x0 = rng.uniform(lo, hi)
        res = simulated_annealing(f, x0, bounds, T0=max(width, 1.0),
                                  cooling=0.999, max_iter=max_iter // max(1, restarts),
                                  step=0.25 * width, rng=rng)
        cand_x, cand_f = res.x, res.fun
        calls += res.function_calls
        if local:
            polished = lbfgs(f, res.x, tol=1e-12, max_iter=500)
            calls += polished.function_calls
            if polished.fun < cand_f:
                cand_x, cand_f = np.clip(polished.x, lo, hi), float(polished.fun)
        if cand_f < best_f:
            best_x, best_f = cand_x, cand_f
    return OptimizeResult(best_x, float(best_f), None, None, restarts, True, calls,
                          0, "dual_annealing_lite", [],
                          "annealing with local polish")
