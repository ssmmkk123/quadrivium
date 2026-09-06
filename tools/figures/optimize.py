"""Figures for the optimization guide."""

from __future__ import annotations

from quadrivium import numeric as np

import quadrivium as qd
from quadrivium.optimize import (backtracking, bfgs, curve_fit,
                                 differential_evolution, gradient_descent,
                                 lasso, linprog, nelder_mead, nonlinear_cg,
                                 powell, ridge, steepest_descent_opt,
                                 strong_wolfe)

from . import figure
from figstyle import annotate, finish, ordinal, sequential_cmap

GUIDE = "guides/optimize.md"

ROSENBROCK = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
ROSENBROCK_GRAD = lambda v: np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
                                      200 * (v[1] - v[0] ** 2)])


@figure("optimize-rosenbrock-paths", GUIDE,
        "The path three optimizers take across the Rosenbrock valley",
        size=(7.0, 4.0))
def rosenbrock_paths(fig, scheme):
    runs = [("BFGS", bfgs(ROSENBROCK, [-1.2, 1.0], ROSENBROCK_GRAD, tol=1e-10)),
            ("Nelder-Mead", nelder_mead(ROSENBROCK, [-1.2, 1.0], tol=1e-10)),
            ("gradient descent, lr = 0.001",
             gradient_descent(ROSENBROCK, [-1.2, 1.0], ROSENBROCK_GRAD,
                              lr=1e-3, tol=1e-10, max_iter=3000))]

    ax = fig.add_subplot()
    gx = np.linspace(-1.7, 1.7, 320)
    gy = np.linspace(-0.6, 1.9, 320)
    X, Y = np.meshgrid(gx, gy)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    ax.contour(X, Y, Z, levels=np.logspace(-0.7, 3.2, 16),
               colors=[scheme.grid], linewidths=0.7)
    for i, (name, result) in enumerate(runs):
        path = np.asarray(result.history, dtype=float)
        ax.plot(path[:, 0], path[:, 1], color=scheme.series[i], linewidth=1.6,
                marker="o", markersize=2.5, markevery=max(len(path) // 40, 1),
                label=f"{name} — {result.iterations:,} iterations")
    ax.scatter([-1.2], [1.0], s=44, color=scheme.ink, zorder=4, marker="s",
               label="start")
    ax.scatter([1.0], [1.0], s=70, color=scheme.ink, zorder=4, marker="*",
               label="minimum at (1, 1)")
    finish(ax, scheme,
           title="Same function, same start, three ideas about what to do next",
           xlabel=r"$x_1$", ylabel=r"$x_2$", grid=None, legend=True,
           legend_kw=dict(loc="upper left", ncol=2, fontsize=8))
    ax.set_ylim(-0.6, 2.3)


@figure("optimize-convergence", GUIDE,
        "How fast the objective falls, by method", size=(7.0, 3.6))
def convergence(fig, scheme):
    runs = [("BFGS (quasi-Newton)", bfgs(ROSENBROCK, [-1.2, 1.0], ROSENBROCK_GRAD,
                                         tol=1e-12)),
            ("nonlinear CG", nonlinear_cg(ROSENBROCK, [-1.2, 1.0], ROSENBROCK_GRAD,
                                          tol=1e-12)),
            ("Powell (derivative free)", powell(ROSENBROCK, [-1.2, 1.0], tol=1e-12)),
            ("steepest descent", steepest_descent_opt(ROSENBROCK, [-1.2, 1.0],
                                                      ROSENBROCK_GRAD,
                                                      tol=1e-12,
                                                      max_iter=5000))]

    ax = fig.add_subplot()
    for i, (name, result) in enumerate(runs):
        path = np.asarray(result.history, dtype=float)
        values = np.maximum([ROSENBROCK(p) for p in path], 1e-18)
        ax.semilogy(np.arange(values.size), values, color=scheme.series[i],
                    label=f"{name} — {result.iterations:,} iterations")
    finish(ax, scheme,
           title="Superlinear methods finish; steepest descent crawls the valley",
           xlabel="iteration", ylabel=r"$f(x_k)$ — the minimum is 0", grid="both",
           legend=True, legend_kw=dict(loc="upper right", fontsize=8))
    ax.set_xlim(0, 210)
    ax.set_ylim(1e-18, 1e2)


@figure("optimize-line-search", GUIDE,
        "What the Armijo and Wolfe conditions accept along a search direction",
        size=(7.0, 3.8))
def line_search(fig, scheme):
    point = np.array([-1.2, 1.0])
    grad = ROSENBROCK_GRAD(point)
    direction = -grad / np.linalg.norm(grad)
    slope = float(grad @ direction)
    alphas = np.linspace(0, 0.9, 600)
    phi = np.array([ROSENBROCK(point + a * direction) for a in alphas])
    c1, c2 = 1e-4, 0.9
    armijo_line = phi[0] + c1 * slope * alphas
    accepted_armijo = phi <= armijo_line
    curvature = np.array([float(ROSENBROCK_GRAD(point + a * direction) @ direction)
                          for a in alphas])
    accepted_wolfe = accepted_armijo & (np.abs(curvature) <= c2 * abs(slope))

    step_backtracking = backtracking(ROSENBROCK, point, direction, grad)
    step_wolfe = strong_wolfe(ROSENBROCK, ROSENBROCK_GRAD, point, direction)

    ax = fig.add_subplot()
    ax.plot(alphas, phi, color=scheme.series[0], label=r"$\phi(\alpha)$")
    ax.plot(alphas, armijo_line, color=scheme.series[1],
            linestyle=(0, (5, 2)),
            label=r"Armijo line $\phi(0) + c_1\alpha\,\phi'(0)$")
    ax.fill_between(alphas, phi.min() - 1, phi.max() + 1, where=accepted_armijo,
                    color=scheme.series[1], alpha=0.10)
    ax.fill_between(alphas, phi.min() - 1, phi.max() + 1, where=accepted_wolfe,
                    color=scheme.series[2], alpha=0.22)
    for name, step, colour in (("backtracking", step_backtracking, scheme.series[1]),
                               ("strong Wolfe", step_wolfe, scheme.series[2])):
        step = float(np.ravel(step)[0])
        ax.scatter([step], [ROSENBROCK(point + step * direction)], s=48, zorder=4,
                   color=colour, edgecolors=scheme.surface, linewidths=1.2,
                   label=f"{name} accepts {step:.3f}")
    finish(ax, scheme,
           title="Sufficient decrease alone accepts a tiny step; curvature "
                 "rules it out",
           xlabel=r"step length $\alpha$", ylabel=r"$\phi(\alpha) = f(x + \alpha d)$",
           grid="both", legend=True, legend_kw=dict(loc="upper right",
                                                    fontsize=8))
    ax.set_ylim(0, float(phi.max()) * 1.05)
    annotate(ax, "shaded: accepted by Armijo (light) and by\n"
                 "the strong Wolfe conditions (dark)", (0.72, phi.max() * 0.30),
             (0.72, phi.max() * 0.30), scheme, arrow=False, ha="center")


@figure("optimize-global-landscape", GUIDE,
        "A landscape full of local minima, and where differential evolution "
        "looks", size=(7.0, 3.8))
def global_landscape(fig, scheme):
    rastrigin = lambda v: 20 + sum(x ** 2 - 10 * np.cos(2 * np.pi * x) for x in v)
    bounds = [(-5.12, 5.12)] * 2
    result = differential_evolution(rastrigin, bounds, rng=0, tol=1e-10)
    local = qd.minimize(rastrigin, [4.0, -3.6], method="bfgs")

    ax = fig.add_subplot()
    grid = np.linspace(-5.12, 5.12, 300)
    X, Y = np.meshgrid(grid, grid)
    Z = 20 + (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y))
    ax.set_rasterization_zorder(0)
    image = ax.contourf(X, Y, Z, levels=24, cmap=sequential_cmap(scheme),
                        zorder=-1)
    ax.scatter([local.x[0]], [local.x[1]], s=70, marker="X", zorder=4,
               color=scheme.series[1], edgecolors=scheme.surface, linewidths=1.2,
               label=f"BFGS from (4.0, -3.6) stops at f = {local.fun:.2f}")
    ax.scatter([result.x[0]], [result.x[1]], s=90, marker="*", zorder=5,
               color=scheme.series[3], edgecolors=scheme.surface, linewidths=1.2,
               label=f"differential evolution finds f = {result.fun:.1e}")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.ax.tick_params(labelsize=7, color=scheme.muted, labelcolor=scheme.muted)
    bar.outline.set_edgecolor(scheme.axis)
    ax.set_aspect("equal")
    finish(ax, scheme,
           title="Rastrigin: every local method finds the nearest bowl",
           xlabel=r"$x_1$", ylabel=r"$x_2$", grid=None, legend=True,
           legend_kw=dict(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                          fontsize=8))


@figure("optimize-sparse-recovery", GUIDE,
        "L1 recovers a sparse signal; L2 spreads the answer over every "
        "coefficient", size=(7.0, 3.6))
def sparse_recovery(fig, scheme):
    rng = np.random.default_rng(0)
    # 30 measurements of 60 unknowns: the system cannot be solved without a
    # prior, and the two penalties encode different ones.
    A = rng.standard_normal((30, 60)) / np.sqrt(30)
    truth = np.zeros(60)
    truth[[7, 23, 44]] = [2.0, -3.0, 1.5]
    b = A @ truth
    l1 = np.asarray(lasso(A, b, lam=0.02).x, dtype=float)
    l2 = np.asarray(ridge(A, b, lam=0.02).x, dtype=float).ravel()

    ax = fig.add_subplot()
    index = np.arange(truth.size)
    ax.bar(index - 0.28, truth, width=0.28, color=scheme.muted,
           label="true signal — 3 nonzeros")
    ax.bar(index, l1, width=0.28, color=scheme.series[0],
           label=f"lasso (L1) — {int(np.sum(np.abs(l1) > 1e-6))} nonzeros, "
                 f"max error {float(np.max(np.abs(l1 - truth))):.2f}")
    ax.bar(index + 0.28, l2, width=0.28, color=scheme.series[1],
           label=f"ridge (L2) — {int(np.sum(np.abs(l2) > 1e-6))} nonzeros, "
                 f"max error {float(np.max(np.abs(l2 - truth))):.2f}")
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme,
           title="Thirty measurements of sixty unknowns, same penalty strength",
           xlabel="coefficient index", ylabel="value", grid="y", legend=True,
           legend_kw=dict(loc="lower left", fontsize=8))
    ax.set_ylim(-4.6, 3.2)


@figure("optimize-linprog", GUIDE,
        "A linear program: the feasible polygon, the objective, and the vertex "
        "the simplex method reaches", size=(7.0, 3.8))
def linear_program(fig, scheme):
    # maximize 3x + 2y subject to x + y <= 4, x + 3y <= 6, x, y >= 0
    result = linprog([-3.0, -2.0], A_ub=[[1.0, 1.0], [1.0, 3.0]], b_ub=[4.0, 6.0])
    vertices = np.array([[0.0, 0.0], [4.0, 0.0], [3.0, 1.0], [0.0, 2.0]])

    ax = fig.add_subplot()
    ax.fill(vertices[:, 0], vertices[:, 1], color=scheme.series[0], alpha=0.12)
    ax.plot(np.append(vertices[:, 0], vertices[0, 0]),
            np.append(vertices[:, 1], vertices[0, 1]), color=scheme.series[0],
            linewidth=1.6, label="feasible region")
    x = np.linspace(-0.2, 5.0, 100)
    colours = ordinal(scheme, 4)
    for k, (colour, level) in enumerate(zip(colours, (3.0, 6.0, 9.0, 12.0))):
        ax.plot(x, (level - 3 * x) / 2, color=colour, linewidth=1.0,
                linestyle=(0, (3, 2)),
                label="objective contours 3x + 2y" if k == 0 else None)
    ax.text(4.05, 0.22, "3x + 2y = 12", fontsize=7.5, color=scheme.secondary,
            ha="right")
    ax.scatter(vertices[:, 0], vertices[:, 1], s=30, color=scheme.series[0],
               zorder=3, label="vertices — the only candidates")
    ax.scatter([result.x[0]], [result.x[1]], s=90, marker="*", zorder=4,
               color=scheme.series[1], edgecolors=scheme.surface, linewidths=1.2,
               label=f"optimum ({result.x[0]:.0f}, {result.x[1]:.0f}), "
                     f"value {-result.fun:.0f}")
    finish(ax, scheme,
           title="The optimum of a linear program is always at a vertex",
           xlabel="x", ylabel="y", grid="both", legend=True,
           legend_kw=dict(loc="upper right", fontsize=8))
    ax.set_xlim(-0.3, 5.2)
    ax.set_ylim(-0.3, 3.4)
    annotate(ax, "the objective rises to the north-east; the last\ncontour "
                 "to touch the region touches it at a corner",
             (3.75, 0.35), (1.9, 1.1), scheme, ha="center")


@figure("optimize-curve-fit", GUIDE,
        "Nonlinear least squares: the fitted model and what is left over",
        size=(7.0, 3.4))
def curve_fitting(fig, scheme):
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4, 40)
    truth = 2.5 * np.exp(-1.3 * t) + 0.5
    y = truth + 0.04 * rng.standard_normal(t.size)
    model = lambda x, a, b, c: a * np.exp(-b * x) + c
    fit = curve_fit(model, t, y, [1.0, 1.0, 0.0])
    fitted = model(t, *fit.x)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(t, y, s=18, color=scheme.muted, label="data")
    dense = np.linspace(0, 4, 300)
    ax1.plot(dense, model(dense, *fit.x), color=scheme.series[0],
             label=f"fit: {fit.x[0]:.2f} exp(-{fit.x[1]:.2f} t) + {fit.x[2]:.2f}")
    finish(ax1, scheme, title="Levenberg-Marquardt fit", xlabel="t", ylabel="y",
           grid="both", legend=True, legend_kw=dict(loc="upper right",
                                                    fontsize=8))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axhline(0, color=scheme.axis, linewidth=0.8)
    ax2.scatter(t, y - fitted, s=18, color=scheme.series[1])
    finish(ax2, scheme,
           title="Residuals — structureless is what you want", xlabel="t",
           ylabel="data - fit", grid="both")
    fig.tight_layout()
