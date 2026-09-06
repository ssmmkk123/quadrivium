"""Figures for the stochastic methods guide."""

from __future__ import annotations

from quadrivium import numeric as np

from quadrivium.stochastic import (LCG, autocorrelation_time, bootstrap,
                                   effective_sample_size,
                                   euler_maruyama, geometric_brownian_motion,
                                   halton, hamiltonian_mc, kernel_density,
                                   milstein, random_walk_metropolis, sobol,
                                   spectral_test, strong_error)

from . import figure
from figstyle import annotate, finish

GUIDE = "guides/stochastic.md"


@figure("stochastic-spectral-test", GUIDE,
        "RANDU's triples lie on a family of parallel lines; a modern "
        "generator's do not", size=(7.0, 3.8))
def spectral_test_figure(fig, scheme):
    n = 200_000
    # RANDU: a = 65539, c = 0, m = 2^31.  Its successive triples satisfy
    # x[i+2] = 6x[i+1] - 9x[i] (mod m) exactly, so they lie on a family of
    # parallel planes -- which `spectral_test` finds on its own, without being
    # told what to look for.
    randu = LCG(seed=1, a=65539, c=0, m=2 ** 31)
    stream = np.array([randu.random() for _ in range(n)])
    report = spectral_test(LCG(seed=1, a=65539, c=0, m=2 ** 31), n=4000, dim=3)
    good = np.random.default_rng(0).random(n)

    # A thin slab of the unit cube: fix the first coordinate and look at the
    # other two, which is the plane structure seen from the side.
    lo, hi = 0.50, 0.51
    for k, (name, values, colour) in enumerate(
            [("RANDU", stream, scheme.series[0]),
             ("NumPy PCG64", good, scheme.series[1])]):
        first, second, third = values[:-2], values[1:-1], values[2:]
        slab = (first > lo) & (first < hi)
        ax = fig.add_subplot(1, 2, k + 1)
        ax.scatter(second[slab], third[slab], s=2.0, color=colour,
                   linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        finish(ax, scheme, title=f"{name} — {int(slab.sum()):,} triples",
               xlabel=r"$u_{i+1}$", ylabel=r"$u_{i+2}$" if k == 0 else None,
               grid="both")
    coefficients = tuple(int(c) for c in np.asarray(report["coefficients"]))
    fig.suptitle(f"Triples with $u_i$ in [{lo}, {hi}], out of {n:,} draws — "
                 f"spectral_test reports the lattice "
                 f"{coefficients}, detected: {bool(report['lattice_detected'])}",
                 x=0.02, ha="left", fontsize=9, color=scheme.secondary)
    fig.tight_layout(rect=(0, 0, 1, 0.94))


@figure("stochastic-low-discrepancy", GUIDE,
        "Pseudorandom points against Halton and Sobol points in the unit "
        "square", size=(7.0, 2.9))
def low_discrepancy(fig, scheme):
    n, bins = 256, 16
    sets = [("pseudorandom", np.random.default_rng(0).random((n, 2))),
            ("Halton", np.asarray(halton(n, dim=2))),
            ("Sobol", np.asarray(sobol(n, dim=2)))]
    for k, (name, points) in enumerate(sets):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.scatter(points[:, 0], points[:, 1], s=5, color=scheme.series[k],
                   linewidths=0, rasterized=True)
        # How many of the 256 cells stay empty: the clumping is what costs
        # plain Monte Carlo its convergence rate.
        counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins,
                                      range=[[0, 1], [0, 1]])
        empty = int(np.sum(counts == 0))
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks((0, 0.5, 1))
        ax.set_yticks((0, 0.5, 1))
        finish(ax, scheme,
               title=f"{name}\n{empty} of {bins * bins} cells empty",
               grid="both")
        ax.title.set_fontsize(9)
    fig.suptitle(f"{n} points each, on a {bins}x{bins} grid", x=0.02, ha="left",
                 fontsize=9, color=scheme.secondary)
    fig.tight_layout(rect=(0, 0, 1, 0.95))


@figure("stochastic-mcmc-diagnostics", GUIDE,
        "Random-walk Metropolis against Hamiltonian Monte Carlo on the same "
        "target", size=(7.0, 4.0))
def mcmc_diagnostics(fig, scheme):
    dim = 10
    log_target = lambda x: -0.5 * float(np.asarray(x) @ np.asarray(x))
    grad = lambda x: -np.asarray(x)
    walk = random_walk_metropolis(log_target, np.zeros(dim), step=0.8, n=6000,
                                  burn=1000, rng=0)
    hmc = hamiltonian_mc(log_target, grad, np.zeros(dim), step=0.25,
                         n_leapfrog=12, n=6000, burn=1000, rng=0)
    ess_walk = float(np.mean(effective_sample_size(walk)))
    ess_hmc = float(np.mean(effective_sample_size(hmc)))

    layout = fig.add_gridspec(2, 2, height_ratios=(1, 1), hspace=0.5,
                              wspace=0.22)
    chains = [("random-walk Metropolis", walk, ess_walk),
              ("Hamiltonian Monte Carlo", hmc, ess_hmc)]
    for k, (name, chain, ess) in enumerate(chains):
        ax = fig.add_subplot(layout[0, k])
        ax.plot(np.asarray(chain)[:800, 0], color=scheme.series[k],
                linewidth=0.8)
        finish(ax, scheme, title=f"{name}: first 800 draws", xlabel="step",
               ylabel=r"$x_1$" if k == 0 else None, grid="both")
        ax.set_ylim(-4, 4)

    ax3 = fig.add_subplot(layout[1, :])
    for k, (name, chain, ess) in enumerate(chains):
        column = np.asarray(chain)[:, 0]
        running = np.cumsum(column) / np.arange(1, column.size + 1)
        ax3.plot(np.arange(1, column.size + 1), running, color=scheme.series[k],
                 label=f"{name}: {ess:,.0f} effective draws of "
                       f"{column.size:,}")
    ax3.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax3, scheme,
           title=r"Running mean of $x_1$, whose true value is 0",
           xlabel="draws used", ylabel="running mean", grid="both", legend=True,
           legend_kw=dict(loc="lower right", fontsize=8))
    ax3.set_ylim(-0.6, 0.6)


@figure("stochastic-sde-paths", GUIDE,
        "Geometric Brownian motion: sample paths, and the mean they average "
        "to", size=(7.0, 3.6))
def sde_paths(fig, scheme):
    mu, sigma, x0 = 1.2, 0.5, 1.0
    drift = lambda x, t: mu * x
    diffusion = lambda x, t: sigma * x

    ax1 = fig.add_subplot(1, 2, 1)
    finals = []
    for seed in range(60):
        path = milstein(drift, diffusion, (0, 1), [x0], n=500, rng=seed)
        finals.append(float(path.y[-1, 0]))
        if seed < 12:
            ax1.plot(path.t, path.y[:, 0], color=scheme.series[0], linewidth=0.9,
                     alpha=0.55)
    t = np.linspace(0, 1, 200)
    ax1.plot(t, x0 * np.exp(mu * t), color=scheme.series[1], linewidth=2.0,
             label=r"$E[X_t] = x_0e^{\mu t}$")
    ax1.plot([], [], color=scheme.series[0], linewidth=0.9,
             label="12 of 60 Milstein paths")
    finish(ax1, scheme, title=r"$dX = \mu X\,dt + \sigma X\,dW$", xlabel="t",
           ylabel="X", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=8))

    ax2 = fig.add_subplot(1, 2, 2)
    grid = np.linspace(0, max(finals) * 1.05, 300)
    # kernel_density returns (points, density).
    points, density = kernel_density(np.array(finals), grid)
    ax2.plot(points, density, color=scheme.series[0],
             label="kernel density of 60 paths")
    # The exact law: log X_1 is normal with mean log x0 + (mu - sigma^2/2)
    # and standard deviation sigma.
    mean = np.log(x0) + (mu - 0.5 * sigma ** 2)
    exact = (np.exp(-((np.log(np.maximum(grid, 1e-9)) - mean) ** 2)
                    / (2 * sigma ** 2)) / (np.maximum(grid, 1e-9) * sigma
                                           * np.sqrt(2 * np.pi)))
    ax2.plot(grid, exact, color=scheme.series[1], linestyle=(0, (5, 2)),
             label="exact lognormal density")
    finish(ax2, scheme, title=r"Distribution of $X_1$", xlabel="X(1)",
           ylabel="density", grid="both", legend=True,
           legend_kw=dict(loc="upper right", fontsize=8))
    fig.tight_layout()


@figure("stochastic-strong-order", GUIDE,
        "Measured strong convergence orders of Euler-Maruyama and Milstein",
        size=(7.0, 3.6))
def strong_order(fig, scheme):
    mu, sigma, x0 = 1.0, 1.0, 1.0
    drift = lambda x, t: mu * x
    diffusion = lambda x, t: sigma * x
    derivative = lambda x, t: sigma * np.ones_like(np.atleast_1d(x))
    exact = lambda t, w: x0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * w)
    steps = np.array([32, 64, 128, 256, 512])

    ax = fig.add_subplot()
    for i, (name, solver, extra) in enumerate(
            [("Euler-Maruyama", euler_maruyama, {}),
             ("Milstein", milstein, {"db": derivative})]):
        errors = [strong_error(solver, exact, (0, 1), [x0], n=int(n), paths=600,
                               rng=1, a=drift, b=diffusion, **extra)
                  for n in steps]
        slope = -np.polyfit(np.log(steps), np.log(errors), 1)[0]
        ax.loglog(1.0 / steps, errors, color=scheme.series[i], marker="o",
                  markersize=4, label=f"{name} — measured order {slope:.2f}")
    reference = 0.9 * (1.0 / steps) ** 0.5
    ax.loglog(1.0 / steps, reference, color=scheme.muted, linewidth=1.0,
              linestyle=(0, (4, 3)))
    finish(ax, scheme,
           title=r"$E|X_N - X(1)|$ on the same Brownian paths",
           xlabel="step size", ylabel="mean pathwise error", grid="both",
           legend=True, legend_kw=dict(loc="lower right"))
    annotate(ax, r"slope 1/2", (1.0 / steps[1], reference[1]),
             (1.0 / steps[1], reference[1] * 2.2), scheme, arrow=False,
             ha="center")


@figure("stochastic-bootstrap", GUIDE,
        "The bootstrap distribution of a statistic, and the interval it "
        "gives", size=(7.0, 3.4))
def bootstrap_figure(fig, scheme):
    rng = np.random.default_rng(3)
    data = rng.gamma(shape=2.0, scale=1.5, size=60)
    result = bootstrap(data, np.median, n_resamples=4000, rng=0)
    low, high = result["ci"]

    ax = fig.add_subplot()
    ax.hist(result["replicates"], bins=40, color=scheme.series[0], alpha=0.85)
    ax.axvline(float(np.median(data)), color=scheme.ink, linewidth=1.4,
               label=f"median of the sample: {float(np.median(data)):.2f}")
    for value, label in ((low, "2.5%"), (high, "97.5%")):
        ax.axvline(value, color=scheme.series[1], linewidth=1.4,
                   linestyle=(0, (4, 3)))
        ax.text(value, ax.get_ylim()[1] * 0.92, f" {label}: {value:.2f}",
                fontsize=8, color=scheme.secondary, ha="left")
    finish(ax, scheme,
           title="4,000 resamples of 60 observations, no distributional "
                 "assumption",
           xlabel="median of the resample", ylabel="count", grid="y",
           legend=True, legend_kw=dict(loc="upper left", fontsize=8))
