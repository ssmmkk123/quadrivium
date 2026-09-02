"""Figures for the root finding guide."""

from __future__ import annotations

import numpy as np

from quadrivium.rootfind import (aberth_ehrlich, bisection, brent, companion_roots,
                                 illinois, itp, newton, ridders, root_bounds,
                                 secant, steffensen)

from . import figure
from figstyle import annotate, finish

GUIDE = "guides/rootfind.md"

# x^3 - 2x - 5 = 0, Wallis's cubic: the example the guide uses throughout.
CUBIC = lambda x: x ** 3 - 2 * x - 5
D_CUBIC = lambda x: 3 * x ** 2 - 2
ROOT = 2.0945514815423265


@figure("rootfind-convergence-rates", GUIDE,
        "Error against iteration for five scalar root finders on the same "
        "equation", size=(7.0, 3.6))
def convergence_rates(fig, scheme):
    runs = [("bisection (order 1)", bisection(CUBIC, 1, 3, tol=1e-15)),
            ("secant (1.618)", secant(CUBIC, 1.0, 3.0, tol=1e-15)),
            ("Newton (2)", newton(CUBIC, 3.0, D_CUBIC, tol=1e-15)),
            ("Ridders (2)", ridders(CUBIC, 1, 3, tol=1e-15)),
            ("Brent", brent(CUBIC, 1, 3, tol=1e-15))]

    ax = fig.add_subplot()
    floor = 1e-17
    for i, (name, res) in enumerate(runs):
        errors = np.maximum(np.abs(np.asarray(res.history, dtype=float) - ROOT), floor)
        ax.semilogy(np.arange(1, errors.size + 1), errors, label=name,
                    color=scheme.series[i], marker="o", markersize=3.5,
                    markeredgecolor=scheme.surface, markeredgewidth=0.6)
    finish(ax, scheme,
           title="One bit per step, or the number of correct digits doubling",
           xlabel="iteration", ylabel=r"$|x_k - x^*|$", grid="both", legend=True,
           legend_kw=dict(loc="upper right", ncol=2))
    ax.set_xlim(0, 24)
    ax.set_ylim(floor, 5)
    annotate(ax, "bisection is a straight line on a log scale:\n"
                 "one bit of the answer per evaluation",
             (14, 1e-5), (14, 1e-5), scheme, arrow=False, ha="left", va="top")


@figure("rootfind-cost", GUIDE,
        "Function evaluations each scalar method needs to reach 1e-12",
        size=(7.0, 3.2))
def cost(fig, scheme):
    methods = [("bisection", lambda: bisection(CUBIC, 1, 3, tol=1e-12)),
               ("Illinois", lambda: illinois(CUBIC, 1, 3, tol=1e-12)),
               ("ITP", lambda: itp(CUBIC, 1, 3, tol=1e-12)),
               ("Ridders", lambda: ridders(CUBIC, 1, 3, tol=1e-12)),
               ("Brent", lambda: brent(CUBIC, 1, 3, tol=1e-12)),
               ("secant", lambda: secant(CUBIC, 1.0, 3.0, tol=1e-12)),
               ("Steffensen", lambda: steffensen(CUBIC, 2.5, tol=1e-12)),
               ("Newton", lambda: newton(CUBIC, 3.0, D_CUBIC, tol=1e-12))]
    results = [(name, run()) for name, run in methods]
    results.sort(key=lambda pair: pair[1].function_calls)

    ax = fig.add_subplot()
    names = [name for name, _ in results]
    calls = [res.function_calls for _, res in results]
    bracketing = {"bisection", "Illinois", "ITP", "Ridders", "Brent"}
    colors = [scheme.series[0] if name in bracketing else scheme.series[1]
              for name in names]
    bars = ax.barh(names, calls, height=0.62, color=colors)
    for bar, value in zip(bars, calls):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                str(value), va="center", fontsize=8, color=scheme.secondary)
    ax.invert_yaxis()
    handles = [__import__("matplotlib").patches.Patch(color=scheme.series[0],
                                                      label="bracketing — convergence guaranteed"),
               __import__("matplotlib").patches.Patch(color=scheme.series[1],
                                                      label="open — faster, no guarantee")]
    finish(ax, scheme,
           title="Cost is function evaluations, not iterations",
           xlabel="function evaluations to reach a tolerance of 1e-12", grid="x")
    leg = ax.legend(handles=handles, loc="upper right")
    for text in leg.get_texts():
        text.set_color(scheme.secondary)
    ax.set_xlim(0, max(calls) * 1.12)
    ax.tick_params(axis="y", length=0)


@figure("rootfind-newton-basins", GUIDE,
        "Which root Newton's method lands on, as a function of where it "
        "starts", size=(7.0, 4.0))
def newton_basins(fig, scheme):
    # f(x) = x^3 - x has roots at -1, 0 and 1.  Newton's iterate is a rational
    # map; near the points where f'(x) = 0 it throws the iterate far away, and
    # which root it eventually reaches stops being a continuous function of the
    # starting point.
    f = lambda x: x ** 3 - x
    df = lambda x: 3 * x ** 2 - 1
    starts = np.linspace(-1.6, 1.6, 1400)
    roots = np.array([-1.0, 0.0, 1.0])
    landed = np.full(starts.size, np.nan)
    for i, x0 in enumerate(starts):
        res = newton(f, float(x0), df, tol=1e-12, max_iter=60)
        if res.converged:
            landed[i] = int(np.argmin(np.abs(roots - res.root)))

    grid = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.12)
    ax = fig.add_subplot(grid[0])
    xs = np.linspace(-1.6, 1.6, 600)
    ax.plot(xs, f(xs), color=scheme.secondary, linewidth=1.6)
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    for k, root in enumerate(roots):
        ax.scatter([root], [0], s=46, zorder=3, color=scheme.series[k],
                   edgecolors=scheme.surface, linewidths=1.5,
                   label=f"root at {root:g}")
    for critical in (-np.sqrt(1 / 3), np.sqrt(1 / 3)):
        ax.axvline(critical, color=scheme.muted, linewidth=0.9,
                   linestyle=(0, (4, 3)))
    annotate(ax, r"$f'(x) = 0$: the step blows up", (np.sqrt(1 / 3), -0.45),
             (1.25, -1.6), scheme, ha="center")
    finish(ax, scheme, title="Newton's method is not local, however local it looks",
           ylabel=r"$f(x) = x^3 - x$", grid="both", legend=True,
           legend_kw=dict(loc="upper left", ncol=3))
    ax.set_xlim(-1.6, 1.6)
    ax.set_xticklabels([])

    ax2 = fig.add_subplot(grid[1])
    for k in range(3):
        mask = landed == k
        ax2.scatter(starts[mask], np.zeros(mask.sum()), s=8, marker="|",
                    color=scheme.series[k], linewidths=1.1, rasterized=True)
    mask = np.isnan(landed)
    ax2.scatter(starts[mask], np.zeros(mask.sum()), s=8, marker="|",
                color=scheme.muted, linewidths=1.1, rasterized=True)
    ax2.set_yticks([])
    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(False)
    ax2.set_xlabel("starting point $x_0$ — coloured by the root Newton reaches")
    for spine in ("left", "top", "right"):
        ax2.spines[spine].set_visible(False)


@figure("rootfind-multiplicity", GUIDE,
        "Newton at a triple root, with and without the multiplicity "
        "correction", size=(7.0, 3.4))
def multiplicity(fig, scheme):
    # A triple root at 1 that is not exactly a cubic, so the corrected
    # iteration still has work to do rather than landing on the root at once.
    g = lambda x: (x - 1.0) ** 3 * (x + 2.0)
    dg = lambda x: 3.0 * (x - 1.0) ** 2 * (x + 2.0) + (x - 1.0) ** 3
    plain = newton(g, 2.0, dg, tol=1e-14, max_iter=200)
    fixed = newton(g, 2.0, dg, tol=1e-14, max_iter=200, multiplicity=3)

    ax = fig.add_subplot()
    for i, (name, res) in enumerate([("plain Newton, linear", plain),
                                     ("multiplicity=3, quadratic", fixed)]):
        errors = np.maximum(np.abs(np.asarray(res.history, dtype=float) - 1.0), 1e-17)
        ax.semilogy(np.arange(1, errors.size + 1), errors,
                    label=f"{name} — {res.iterations} iterations",
                    color=scheme.series[i], marker="o", markersize=3.5,
                    markeredgecolor=scheme.surface, markeredgewidth=0.6)
    finish(ax, scheme,
           title="A multiple root costs Newton its quadratic convergence",
           xlabel="iteration", ylabel=r"$|x_k - 1|$", grid="both", legend=True)
    ax.set_ylim(1e-10, 5)


@figure("rootfind-polynomial-roots", GUIDE,
        "Polynomial roots in the complex plane inside the Cauchy and Fujiwara "
        "bounds", size=(7.0, 4.0))
def polynomial_roots_figure(fig, scheme):
    # z^6 + z^5 - 3z^4 - 3z^3 + 2z^2 + 2z - 4: two real roots and two complex
    # pairs, so both the bounds and the two solvers have something to show.
    coeffs = [1.0, 1.0, -3.0, -3.0, 2.0, 2.0, -4.0]
    companion = np.asarray(companion_roots(coeffs), dtype=complex)
    aberth = np.asarray(aberth_ehrlich(coeffs), dtype=complex)
    bounds = root_bounds(coeffs)

    ax = fig.add_subplot()
    theta = np.linspace(0, 2 * np.pi, 400)
    for key, style in (("cauchy", (0, (5, 2))), ("fujiwara", (0, (2, 2))),
                       ("lower", (0, (1, 2)))):
        radius = float(bounds[key])
        ax.plot(radius * np.cos(theta), radius * np.sin(theta),
                color=scheme.muted, linewidth=1.0, linestyle=style,
                label=f"{key} bound, |z| = {radius:.2f}")
    ax.scatter(companion.real, companion.imag, s=90, color=scheme.series[0],
               marker="o", facecolors="none", linewidths=1.8,
               label="companion matrix eigenvalues")
    ax.scatter(aberth.real, aberth.imag, s=26, color=scheme.series[1],
               marker="x", linewidths=1.6, label="Aberth-Ehrlich, all at once")
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    ax.axvline(0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme,
           title="Two methods, one answer, inside bounds computed without solving",
           xlabel="real part", ylabel="imaginary part", grid="both", legend=True,
           legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                          fontsize=8))
    ax.set_aspect("equal")
    largest = float(bounds["cauchy"])
    ax.set_xlim(-largest * 1.1, largest * 1.1)
    ax.set_ylim(-largest * 1.1, largest * 1.1)
    fig.tight_layout()
