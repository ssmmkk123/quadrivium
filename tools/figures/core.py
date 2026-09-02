"""Figures for the core guide."""

from __future__ import annotations

import numpy as np

from quadrivium.core import EPS, condition_number, norm
from quadrivium.linalg import qr_solve, solve

from . import figure
from figstyle import annotate, finish, ordinal

GUIDE = "guides/core.md"


@figure("core-norms", GUIDE,
        "The unit ball of the p-norms the library computes", size=(7.0, 3.4))
def norms(fig, scheme):
    grid = np.linspace(-1.6, 1.6, 301)
    X, Y = np.meshgrid(grid, grid)
    orders = [(1, "p = 1"), (2, "p = 2"), (np.inf, r"p = $\infty$")]
    extra = [(0.5, "p = 0.5")]
    colours = list(scheme.series[:3]) + [scheme.muted]

    ax = fig.add_subplot(1, 2, 1)
    for colour, (p, label) in zip(colours, orders + extra):
        values = np.array([[norm(np.array([x, y]), p) for x in grid]
                           for y in grid])
        ax.contour(X, Y, values, levels=[1.0], colors=[colour], linewidths=1.8)
        ax.plot([], [], color=colour, label=label)
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    ax.axvline(0, color=scheme.axis, linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    finish(ax, scheme, title=r"$\{x : \|x\|_p = 1\}$", xlabel=r"$x_1$",
           ylabel=r"$x_2$", grid="both", legend=True,
           legend_kw=dict(loc="upper right", fontsize=7.5))

    ax2 = fig.add_subplot(1, 2, 2)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((400, 8))
    ps = np.linspace(1, 12, 60)
    ratios = np.array([[norm(v, float(p)) / norm(v, np.inf) for p in ps]
                       for v in vectors])
    ax2.plot(ps, ratios.mean(axis=0), color=scheme.series[0],
             label="mean over 400 random vectors in $R^8$")
    ax2.fill_between(ps, ratios.min(axis=0), ratios.max(axis=0),
                     color=scheme.series[0], alpha=0.12)
    ax2.axhline(1.0, color=scheme.muted, linewidth=0.9, linestyle=(0, (4, 3)))
    finish(ax2, scheme, title=r"Every $p$-norm collapses onto $\|x\|_\infty$",
           xlabel="p", ylabel=r"$\|x\|_p\,/\,\|x\|_\infty$", grid="both",
           legend=True, legend_kw=dict(loc="upper right", fontsize=8))
    fig.tight_layout()


@figure("core-conditioning", GUIDE,
        "The condition number predicts how many digits a solve can lose",
        size=(7.0, 3.6))
def conditioning(fig, scheme):
    sizes = np.arange(2, 13)
    conditions, errors = [], []
    for n in sizes:
        hilbert = np.array([[1.0 / (i + j + 1) for j in range(n)]
                            for i in range(n)])
        x_true = np.ones(n)
        b = hilbert @ x_true
        conditions.append(condition_number(hilbert))
        # Least squares through QR rather than a plain solve, so what is
        # plotted is the conditioning of the problem and not a breakdown of
        # the factorization.
        errors.append(max(float(np.max(np.abs(qr_solve(hilbert, b) - x_true))),
                          1e-18))

    ax = fig.add_subplot()
    ax.semilogy(sizes, conditions, color=scheme.series[0], marker="o",
                markersize=4, label=r"$\kappa_2(H_n)$")
    ax.semilogy(sizes, np.array(conditions) * EPS, color=scheme.series[1],
                linestyle=(0, (5, 2)),
                label=r"$\kappa_2(H_n)\cdot\varepsilon$ — the error to expect")
    ax.semilogy(sizes, errors, color=scheme.series[2], marker="s",
                markersize=4, label="error actually made")
    finish(ax, scheme,
           title="The Hilbert matrix: a bound that is met, not merely quoted",
           xlabel="n", ylabel="condition number, or error", grid="both",
           legend=True, legend_kw=dict(loc="center left"))
    ax.set_ylim(1e-18, 1e18)
    annotate(ax, "the error tracks the bound: by n = 12 the\n"
                 "answer has no correct digits left",
             (sizes[-1], errors[-1]), (2.2, 1e10), scheme, ha="left")
