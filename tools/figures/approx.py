"""Figures for the approximation guide."""

from __future__ import annotations

import numpy as np

from quadrivium.approx import (aaa, chebyshev_t, fourier_series,
                               gauss_chebyshev_nodes, gauss_hermite_nodes,
                               gauss_laguerre_nodes, gauss_legendre_nodes,
                               gauss_lobatto_nodes, gauss_radau_nodes, legendre,
                               pade, pade_evaluate, polyfit, remez)

from . import figure
from figstyle import annotate, finish, ordinal

GUIDE = "guides/approx.md"


@figure("approx-orthogonal-families", GUIDE,
        "Legendre and Chebyshev polynomials of the first six degrees",
        size=(7.0, 3.4))
def orthogonal_families(fig, scheme):
    x = np.linspace(-1, 1, 600)
    colours = ordinal(scheme, 6)
    for panel, (name, func, ylabel) in enumerate(
            [("Legendre $P_n$", legendre, r"$P_n(x)$"),
             ("Chebyshev $T_n$", chebyshev_t, r"$T_n(x)$")]):
        ax = fig.add_subplot(1, 2, panel + 1)
        for n in range(6):
            y = np.array([func(n, float(v)) for v in x])
            ax.plot(x, y, color=colours[n], linewidth=1.6, label=f"n = {n}")
        ax.axhline(0, color=scheme.axis, linewidth=0.8)
        finish(ax, scheme, title=name, xlabel="x", ylabel=ylabel, grid="both",
               legend=True, legend_kw=dict(loc="lower center", ncol=3,
                                           fontsize=7.5))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1.55, 1.25)
    fig.tight_layout()


@figure("approx-gauss-nodes", GUIDE,
        "Where the Gauss families put their nodes, and how much weight each "
        "one carries", size=(7.0, 3.6))
def gauss_nodes(fig, scheme):
    n = 12
    families = [("Gauss-Legendre", *gauss_legendre_nodes(n, -1, 1)),
                ("Gauss-Chebyshev", *gauss_chebyshev_nodes(n)),
                ("Gauss-Lobatto", *gauss_lobatto_nodes(n, -1, 1)),
                ("Gauss-Radau", *gauss_radau_nodes(n, -1, 1)),
                ("equally spaced", np.linspace(-1, 1, n),
                 np.full(n, 2.0 / n))]

    ax = fig.add_subplot()
    for row, (name, nodes, weights) in enumerate(families):
        y = np.full(np.size(nodes), len(families) - row - 1.0)
        colour = scheme.series[0] if row < 4 else scheme.muted
        ax.scatter(nodes, y, s=380 * np.asarray(weights, dtype=float) /
                   float(np.max(weights)), color=colour, alpha=0.85,
                   edgecolors=scheme.surface, linewidths=0.8)
        ax.text(-1.08, len(families) - row - 1.0, name, ha="right", va="center",
                fontsize=8.5, color=scheme.secondary)
    finish(ax, scheme,
           title="Marker area is the quadrature weight; every family clusters "
                 "at the ends",
           xlabel="node position on [-1, 1]", grid="x")
    ax.set_yticks([])
    ax.set_xlim(-1.75, 1.1)
    ax.set_ylim(-0.7, len(families) + 0.6)
    for spine in ("left",):
        ax.spines[spine].set_visible(False)
    annotate(ax, "Lobatto keeps both endpoints; Radau keeps one; the "
                 "interior families keep neither",
             (0.0, 5.0), (0.0, 5.0), scheme, arrow=False, ha="center")


@figure("approx-remez-equioscillation", GUIDE,
        "The minimax error equioscillates; the least squares error does not",
        size=(7.0, 3.6))
def remez_equioscillation(fig, scheme):
    degree = 4
    grid = np.linspace(-1, 1, 2001)
    best = remez(np.exp, degree, -1, 1)
    minimax_error = best(grid) - np.exp(grid)
    ls = polyfit(grid, np.exp(grid), degree=degree)
    ls_error = np.polyval(ls, grid) - np.exp(grid)

    ax = fig.add_subplot()
    ax.plot(grid, ls_error, color=scheme.series[1],
            label=f"least squares — worst error {np.max(np.abs(ls_error)):.2e}")
    ax.plot(grid, minimax_error, color=scheme.series[0],
            label=f"Remez minimax — worst error {np.max(np.abs(minimax_error)):.2e}")
    peak = float(np.max(np.abs(minimax_error)))
    for sign in (1, -1):
        ax.axhline(sign * peak, color=scheme.muted, linewidth=0.9,
                   linestyle=(0, (4, 3)))
    # The alternation points: interior extrema of the minimax error, plus the
    # two endpoints.  Degree + 2 of them is the equioscillation theorem.
    interior = np.where((np.abs(minimax_error[1:-1]) > np.abs(minimax_error[:-2]))
                        & (np.abs(minimax_error[1:-1]) > np.abs(minimax_error[2:])))[0] + 1
    marks = np.concatenate(([0], interior, [grid.size - 1]))
    ax.scatter(grid[marks], minimax_error[marks], s=34, zorder=4,
               color=scheme.series[0], edgecolors=scheme.surface, linewidths=1.2)
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme,
           title=f"{marks.size} alternation points for degree {degree} — "
                 f"the theorem says {degree + 2}",
           xlabel="x", ylabel=r"approximation $-\ e^x$", grid="both", legend=True,
           legend_kw=dict(loc="lower center"))
    ax.set_ylim(-3.1 * peak, 2.6 * peak)


@figure("approx-pade", GUIDE,
        "A Pade approximant against the Taylor polynomial it was built from",
        size=(7.0, 3.4))
def pade_figure(fig, scheme):
    taylor = [1.0, 1.0, 0.5, 1 / 6, 1 / 24]  # e^x through x^4
    num, den = pade(taylor, 2, 2)
    x = np.linspace(-3, 3, 600)
    exact = np.exp(x)
    taylor_values = np.polyval(taylor[::-1], x)
    pade_values = np.array([pade_evaluate(num, den, float(v)) for v in x])

    ax = fig.add_subplot()
    ax.semilogy(x, np.abs(taylor_values - exact) / exact, color=scheme.series[1],
                label="Taylor, degree 4")
    ax.semilogy(x, np.abs(pade_values - exact) / exact, color=scheme.series[0],
                label="Pade [2/2], same five coefficients")
    finish(ax, scheme,
           title="Same information, rearranged: the ratio extrapolates further",
           xlabel="x", ylabel="relative error", grid="both", legend=True,
           legend_kw=dict(loc="lower right"))
    ax.set_ylim(1e-9, 5)
    annotate(ax, "both are exact to order $x^4$ at the origin", (0.0, 1e-8),
             (0.0, 1e-6), scheme, ha="center")


@figure("approx-gibbs", GUIDE,
        "Gibbs' phenomenon: the overshoot at a jump does not shrink with more "
        "harmonics", size=(7.0, 3.6))
def gibbs(fig, scheme):
    square = lambda t: np.sign(np.sin(t))
    t = np.linspace(-0.4, 1.4, 1400)
    orders = (5, 15, 45)
    colours = ordinal(scheme, len(orders))

    layout = fig.add_gridspec(1, 2, width_ratios=(2.0, 1.0), wspace=0.28)
    ax = fig.add_subplot(layout[0])
    ax.plot(t, square(t), color=scheme.secondary, linewidth=1.3,
            linestyle=(0, (5, 2)), label="square wave")
    overshoot = []
    for colour, n in zip(colours, orders):
        series = fourier_series(square, n=n)
        values = np.array([series(float(v)) for v in t])
        ax.plot(t, values, color=colour, linewidth=1.5, label=f"{n} harmonics")
        overshoot.append(float(np.max(values)))
    ax.axhline(1.0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme, title="Partial sums near the jump at x = 0", xlabel="x",
           ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="lower right", fontsize=8))
    ax.set_ylim(-0.4, 1.45)

    ax2 = fig.add_subplot(layout[1])
    # The jump is from -1 to 1, so the overshoot is measured against a height
    # of 2 -- that is the 8.95% the theory names.
    percent = [100 * (v - 1.0) / 2.0 for v in overshoot]
    bars = ax2.bar([str(n) for n in orders], percent, width=0.55,
                   color=[scheme.series[0]] * len(orders))
    for bar, value in zip(bars, percent):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{value:.1f}%", ha="center", fontsize=8,
                 color=scheme.secondary)
    ax2.axhline(8.95, color=scheme.muted, linewidth=0.9, linestyle=(0, (4, 3)))
    finish(ax2, scheme, title="Overshoot", xlabel="harmonics",
           ylabel="percent of the jump height", grid="y")
    ax2.set_ylim(0, 13.5)
    annotate(ax2, "the limit, 8.95%", (2.4, 8.95), (2.45, 12.2), scheme,
             ha="right")


@figure("approx-rational-vs-polynomial", GUIDE,
        "Rational approximation against polynomial approximation of a function "
        "with a nearby singularity", size=(7.0, 3.6))
def rational_vs_polynomial(fig, scheme):
    # tan has poles just outside the interval: a polynomial has to spend
    # degree on them, a rational approximant simply has poles.
    target = np.tan
    sample = np.linspace(-1.4, 1.4, 601)
    values = target(sample)

    parameters, polynomial_error, rational_error = [], [], []
    for degree in range(2, 27, 2):
        coeffs = polyfit(sample, values, degree=int(degree))
        polynomial_error.append(
            float(np.max(np.abs(np.polyval(coeffs, sample) - values))))
        parameters.append(degree + 1)
    for tol in np.logspace(-2, -13, 12):
        approximant, support, _, _ = aaa(target, sample, tol=float(tol))
        rational_error.append(
            float(np.max(np.abs(approximant(sample) - values))))
        parameters.append(2 * len(support))

    ax1 = fig.add_subplot(1, 2, 1)
    coeffs = polyfit(sample, values, degree=12)
    approximant = aaa(target, sample, tol=1e-10)[0]
    ax1.semilogy(sample, np.abs(np.polyval(coeffs, sample) - values) + 1e-18,
                 color=scheme.series[1], label="degree 12 polynomial")
    ax1.semilogy(sample, np.abs(approximant(sample) - values) + 1e-18,
                 color=scheme.series[0], label="AAA rational, 16 coefficients")
    finish(ax1, scheme, title=r"Error in approximating $\tan x$", xlabel="x",
           ylabel="absolute error", grid="both", legend=True,
           legend_kw=dict(loc="lower center", fontsize=7.5))
    ax1.set_ylim(1e-17, 1e2)

    ax2 = fig.add_subplot(1, 2, 2)
    half = len(polynomial_error)
    ax2.semilogy(parameters[:half], polynomial_error, color=scheme.series[1],
                 marker="o", markersize=3, label="polynomial least squares")
    ax2.semilogy(parameters[half:], np.maximum(rational_error, 1e-16),
                 color=scheme.series[0], marker="s", markersize=3,
                 label="AAA rational")
    finish(ax2, scheme, title="Error against the number of coefficients",
           xlabel="coefficients used", ylabel="max error", grid="both",
           legend=True, legend_kw=dict(loc="upper right", fontsize=7.5))
    fig.tight_layout()
