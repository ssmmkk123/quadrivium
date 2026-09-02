"""Figures for the integration guide."""

from __future__ import annotations

import numpy as np

from quadrivium.integrate import (adaptive_gauss_kronrod, boole_rule, filon,
                                  gauss_legendre, monte_carlo,
                                  quasi_monte_carlo, simpson_rule,
                                  stratified_sampling, tanh_sinh,
                                  trapezoid_rule)

from . import figure
from figstyle import annotate, finish, label_line

GUIDE = "guides/integrate.md"


class Recorder:
    """Wrap an integrand and keep every abscissa it was asked about."""

    def __init__(self, f):
        self.f = f
        self.points = []

    def __call__(self, x):
        self.points.append(float(x))
        return self.f(x)


@figure("integrate-newton-cotes-orders", GUIDE,
        "Measured convergence orders of the trapezoid, Simpson and Boole "
        "rules", size=(7.0, 3.6))
def newton_cotes_orders(fig, scheme):
    exact = 1.0 - np.cos(1.0)
    panels = np.array([4, 8, 16, 32, 64, 128, 256])
    rules = [("trapezoid, $O(h^2)$", trapezoid_rule),
             ("Simpson, $O(h^4)$", simpson_rule),
             ("Boole, $O(h^6)$", boole_rule)]

    ax = fig.add_subplot()
    h = 1.0 / panels
    for i, (name, rule) in enumerate(rules):
        errors = [max(abs(float(rule(np.sin, 0, 1, n=int(n))) - exact), 1e-18)
                  for n in panels]
        usable = [(hi, e) for hi, e in zip(h, errors) if e > 1e-15]
        slope = np.polyfit(np.log([hi for hi, _ in usable]),
                           np.log([e for _, e in usable]), 1)[0]
        ax.loglog(h, errors, color=scheme.series[i], marker="o", markersize=3.5,
                  label=f"{name} — measured {slope:.1f}")
    ax.axhline(np.finfo(float).eps, color=scheme.muted, linewidth=0.9,
               linestyle=(0, (4, 3)))
    finish(ax, scheme,
           title="Each rule earns its order, until rounding takes over",
           xlabel="panel width h", ylabel=r"error in $\int_0^1 \sin x\,dx$",
           grid="both", legend=True, legend_kw=dict(loc="lower right"))
    ax.set_ylim(1e-18, 1e-2)
    annotate(ax, "below machine epsilon a rule is\nmeasuring rounding, "
                 "not itself",
             (h[-2], 2e-16), (h[-2], 3e-13), scheme, ha="left", va="bottom")


@figure("integrate-adaptive-nodes", GUIDE,
        "Where an adaptive rule puts its evaluations on an integrand with a "
        "narrow spike", size=(7.0, 4.0))
def adaptive_nodes(fig, scheme):
    spike = lambda x: 1.0 / (1e-6 + x ** 2)
    exact = float(2 * np.arctan(1e3) * 1e3)
    recorder = Recorder(spike)
    adaptive = adaptive_gauss_kronrod(recorder, -1, 1, tol=1e-10)
    used = np.array(recorder.points)
    uniform_n = used.size
    uniform = simpson_rule(spike, -1, 1, n=uniform_n)
    uniform_nodes = np.linspace(-1, 1, uniform_n + 1)

    layout = fig.add_gridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.32)
    ax = fig.add_subplot(layout[0])
    grid = np.linspace(-1, 1, 4000)
    ax.plot(grid, spike(grid), color=scheme.secondary, linewidth=1.5)
    ax.set_yscale("log")
    finish(ax, scheme, title=r"$1/(10^{-6} + x^2)$ — nearly all of the area "
                             "lies within 0.1% of the origin",
           ylabel="integrand", grid="both")
    ax.set_xlim(-1, 1)

    ax2 = fig.add_subplot(layout[1])
    ax2.scatter(used, np.ones(used.size), s=10, marker="|",
                color=scheme.series[0], linewidths=1.0, rasterized=True)
    ax2.scatter(uniform_nodes, np.zeros(uniform_nodes.size), s=10, marker="|",
                color=scheme.series[1], linewidths=1.0, rasterized=True)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["uniform Simpson", "adaptive Gauss-Kronrod"],
                        fontsize=8)
    ax2.tick_params(axis="y", length=0)
    ax2.set_ylim(-0.6, 1.6)
    ax2.set_xlim(-1, 1)
    ax2.grid(False)
    ax2.set_xlabel(f"abscissa — {used.size} evaluations each")
    for spine in ("left",):
        ax2.spines[spine].set_visible(False)
    adaptive_error = abs(float(adaptive) - exact) / exact
    uniform_error = abs(float(uniform) - exact) / exact
    annotate(ax2, f"relative error {adaptive_error:.0e}", (0.55, 1.0),
             (0.55, 1.35), scheme, arrow=False, ha="center", va="center",
             color=scheme.secondary)
    annotate(ax2, f"relative error {uniform_error:.0%} — the peak "
                  "falls between the nodes", (0.55, 0.0),
             (0.55, -0.35), scheme, arrow=False, ha="center", va="center",
             color=scheme.secondary)


@figure("integrate-gauss-exactness", GUIDE,
        "An n-point Gauss rule is exact through degree 2n-1 and no further",
        size=(7.0, 3.4))
def gauss_exactness(fig, scheme):
    degrees = np.arange(0, 16)
    ax = fig.add_subplot()
    for i, n in enumerate((3, 5, 7)):
        errors = []
        for d in degrees:
            exact = 1.0 / (d + 1)
            value = float(gauss_legendre(lambda x, d=int(d): x ** d, 0, 1, n=int(n)))
            errors.append(max(abs(value - exact) / exact, 1e-18))
        ax.semilogy(degrees, errors, color=scheme.series[i], marker="o",
                    markersize=3.5, label=f"{n}-point rule — exact to degree "
                                          f"{2 * n - 1}")
        ax.axvline(2 * n - 1, color=scheme.series[i], linewidth=0.9,
                   linestyle=(0, (4, 3)), alpha=0.6)
    finish(ax, scheme,
           title=r"Relative error integrating $x^d$ over $[0, 1]$",
           xlabel="degree d of the integrand", ylabel="relative error",
           grid="both", legend=True, legend_kw=dict(loc="upper left"))
    ax.set_ylim(1e-18, 1e3)


@figure("integrate-monte-carlo", GUIDE,
        "Monte Carlo, stratified sampling and quasi-Monte Carlo error against "
        "sample count", size=(7.0, 3.6))
def monte_carlo_rates(fig, scheme):
    target = lambda x: np.exp(-x) * np.cos(4 * x)
    exact = float((1 - np.exp(-1) * (np.cos(4.0) - 4 * np.sin(4.0))) / 17)
    counts = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    plain, strat, qmc = [], [], []
    for n in counts:
        n = int(n)
        # Ten seeds, so the plotted curve is a typical error rather than one
        # lucky or unlucky draw.
        errors = [abs(float(monte_carlo(target, 0, 1, n=n, rng=seed)) - exact)
                  for seed in range(10)]
        plain.append(float(np.mean(errors)))
        strat.append(float(np.mean([
            abs(float(stratified_sampling(target, 0, 1, n=n,
                                          strata=max(n // 16, 2), rng=seed)) - exact)
            for seed in range(10)])))
        qmc.append(abs(float(quasi_monte_carlo(target, [0.0], [1.0], n=n)) - exact))

    ax = fig.add_subplot()
    ax.loglog(counts, plain, color=scheme.series[1], marker="o", markersize=3.5,
              label="plain Monte Carlo")
    ax.loglog(counts, strat, color=scheme.series[2], marker="s", markersize=3.5,
              label="stratified sampling")
    ax.loglog(counts, np.maximum(qmc, 1e-17), color=scheme.series[0], marker="^",
              markersize=3.5, label="quasi-Monte Carlo (Halton)")
    reference = plain[0] * (counts / counts[0]) ** -0.5
    ax.loglog(counts, reference, color=scheme.muted, linewidth=1.0,
              linestyle=(0, (4, 3)))
    label_line(ax, counts[-3], reference[-3] * 1.6, r"$n^{-1/2}$", scheme,
               scheme.muted)
    reference2 = plain[0] * (counts / counts[0]) ** -1.0
    ax.loglog(counts, reference2, color=scheme.muted, linewidth=1.0,
              linestyle=(0, (1, 2)))
    label_line(ax, counts[-4], reference2[-4] * 0.35, r"$n^{-1}$", scheme,
               scheme.muted)
    finish(ax, scheme, title="Four times the samples for two more digits — or "
                             "not, with better points",
           xlabel="samples", ylabel="mean absolute error", grid="both",
           legend=True, legend_kw=dict(loc="lower left"))


@figure("integrate-singular", GUIDE,
        "The tanh-sinh transformation turns an endpoint singularity into a "
        "decaying integrand", size=(7.0, 2.9))
def singular(fig, scheme):
    ax0 = fig.add_subplot(1, 3, 1)
    x = np.linspace(1e-6, 1, 1000)
    ax0.plot(x, 1 / np.sqrt(x), color=scheme.series[1])
    ax0.set_yscale("log")
    finish(ax0, scheme, title=r"$1/\sqrt{x}$ on $[0, 1]$", xlabel="x",
           ylabel="integrand", grid="both")
    annotate(ax0, "unbounded here", (0.02, 7.0), (0.35, 60.0), scheme,
             ha="center")

    ax1 = fig.add_subplot(1, 3, 2)
    t = np.linspace(-3.4, 3.4, 1200)
    # The change of variable tanh-sinh applies, x = (1 + tanh(pi/2 sinh t))/2,
    # with the weight it brings with it.
    u = np.tanh(0.5 * np.pi * np.sinh(t))
    weight = (0.5 * np.pi * np.cosh(t)) / np.cosh(0.5 * np.pi * np.sinh(t)) ** 2
    # (u + 1)/2 underflows to zero at the ends of this range, which is the
    # point being made -- but dividing by it would warn rather than plot.
    position = np.maximum((u + 1) / 2, np.finfo(float).tiny)
    transformed = weight / np.sqrt(position) / 2
    ax1.plot(t, transformed, color=scheme.series[0])
    finish(ax1, scheme, title="after the change of variable", xlabel="t",
           ylabel="transformed integrand", grid="both")
    annotate(ax1, "decays doubly\nexponentially", (2.6, 0.05), (1.0, 0.9),
             scheme, ha="center")

    ax2 = fig.add_subplot(1, 3, 3)
    exact = 2.0
    panels = np.array([8, 32, 128, 512, 2048, 8192])
    # Composite Simpson with the singular endpoint value replaced by zero --
    # what an ordinary rule can do here -- converges as the square root of the
    # panel width, so digits cost powers of ten.
    simpson_errors = [abs(float(simpson_rule(lambda x: 1 / np.sqrt(x) if x > 0 else 0.0,
                                             0, 1, n=int(n))) - exact)
                      for n in panels]
    levels = np.arange(2, 9)
    ts_errors, ts_calls = [], []
    for level in levels:
        recorder = Recorder(lambda x: 1 / np.sqrt(x))
        value = tanh_sinh(recorder, 0, 1, levels=int(level), tol=1e-15)
        ts_errors.append(max(abs(float(value) - exact), 1e-17))
        ts_calls.append(len(recorder.points))
    simpson_calls = panels + 1
    ax2.loglog(simpson_calls, np.maximum(simpson_errors, 1e-17),
               color=scheme.series[1], marker="o", markersize=3.5,
               label="composite Simpson")
    ax2.loglog(ts_calls, ts_errors, color=scheme.series[0], marker="s",
               markersize=3.5, label="tanh-sinh")
    finish(ax2, scheme, title="cost of an answer", xlabel="function evaluations",
           ylabel=r"error in $\int_0^1 x^{-1/2}dx$", grid="both", legend=True,
           legend_kw=dict(loc="lower left", fontsize=7.5))
    ax2.set_ylim(1e-17, 1e0)
    fig.tight_layout()


@figure("integrate-oscillatory", GUIDE,
        "Filon quadrature against a general adaptive rule as the frequency "
        "rises", size=(7.0, 3.6))
def oscillatory(fig, scheme):
    omegas = np.array([10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0])
    filon_cost, adaptive_cost = [], []
    for omega in omegas:
        recorder = Recorder(lambda x: 1.0)
        filon(recorder, 0, 1, omega=float(omega), n=100, kind="sin")
        filon_cost.append(len(recorder.points))
        recorder = Recorder(lambda x, w=float(omega): np.sin(w * x))
        adaptive_gauss_kronrod(recorder, 0, 1, tol=1e-10)
        adaptive_cost.append(len(recorder.points))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.loglog(omegas, adaptive_cost, color=scheme.series[1], marker="o",
               markersize=3.5, label="adaptive Gauss-Kronrod")
    ax1.loglog(omegas, filon_cost, color=scheme.series[0], marker="s",
               markersize=3.5, label="Filon")
    finish(ax1, scheme, title="Evaluations to reach 1e-10",
           xlabel=r"frequency $\omega$",
           ylabel="function evaluations", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=8))

    ax2 = fig.add_subplot(1, 2, 2)
    budget_gauss, budget_filon = [], []
    for omega in omegas:
        exact = (1 - np.cos(omega)) / omega
        gauss = gauss_legendre(lambda x, w=float(omega): np.sin(w * x), 0, 1, n=100)
        budget_gauss.append(max(abs(float(gauss) - exact), 1e-17))
        value = filon(lambda x: 1.0, 0, 1, omega=float(omega), n=100, kind="sin")
        budget_filon.append(max(abs(float(value) - exact), 1e-17))
    ax2.loglog(omegas, budget_gauss, color=scheme.series[1], marker="o",
               markersize=3.5, label="100-point Gauss-Legendre")
    ax2.loglog(omegas, budget_filon, color=scheme.series[0], marker="s",
               markersize=3.5, label="100-point Filon")
    finish(ax2, scheme, title="Error on a fixed budget of 100 points",
           xlabel=r"frequency $\omega$",
           ylabel=r"error in $\int_0^1 \sin(\omega x)\,dx$", grid="both",
           legend=True, legend_kw=dict(loc="upper left", fontsize=8))
    fig.tight_layout()
