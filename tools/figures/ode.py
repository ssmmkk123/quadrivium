"""Figures for the ordinary differential equations guide."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import quadrivium as qd
from quadrivium.ode import (backward_euler, energy_drift, euler, heun,
                            finite_difference_bvp, leapfrog, radau_iia, rk4,
                            solve_ivp, solve_ivp_events, sturm_liouville,
                            trapezoidal)

from . import figure
from figstyle import annotate, finish, label_line, ordinal

GUIDE = "guides/ode.md"


def amplification(method, z: complex, **kwargs) -> float:
    """|y1| after one step of `method` on y' = z y, written as a real system.

    The library's integrators work in real arithmetic, so the complex test
    equation is carried as the 2x2 rotation-and-scaling it is equivalent to.
    Nothing here assumes a stability function: the number is whatever the
    method being described actually produces in one step of size 1.
    """
    a, b = z.real, z.imag
    rhs = lambda t, y: np.array([a * y[0] - b * y[1], b * y[0] + a * y[1]])
    solution = method(rhs, (0.0, 1.0), [1.0, 0.0], n=1, **kwargs)
    final = solution.y[-1]
    return float(np.hypot(final[0], final[1]))


@lru_cache(maxsize=None)
def stability_grid(name, re_lo, re_hi, im_hi, nx, ny):
    """|amplification| over a rectangle of the complex plane, cached.

    Each figure is drawn once per colour scheme; the arithmetic does not
    depend on the colours, so it is done once.
    """
    methods = {"euler": euler, "heun": heun, "rk4": rk4,
               "backward_euler": backward_euler, "trapezoidal": trapezoidal,
               "radau_iia": radau_iia}
    method = methods[name]
    re = np.linspace(re_lo, re_hi, nx)
    im = np.linspace(-im_hi, im_hi, ny)
    grid = np.empty((ny, nx))
    for j, y in enumerate(im):
        for i, x in enumerate(re):
            grid[j, i] = amplification(method, complex(x, y))
    return re, im, grid


@figure("ode-convergence-orders", GUIDE,
        "Measured convergence orders of four fixed-step methods on a linear "
        "test problem", size=(7.0, 3.6))
def convergence_orders(fig, scheme):
    exact = float(np.exp(-1.0))
    steps = np.array([10, 20, 40, 80, 160, 320, 640])
    methods = [("Euler", euler, 1), ("Heun", heun, 2),
               ("RK3", qd.ode.rk3, 3), ("RK4", rk4, 4)]

    ax = fig.add_subplot()
    h = 1.0 / steps
    for i, (name, method, order) in enumerate(methods):
        errors = [max(abs(float(method(lambda t, y: -y, (0, 1), [1.0],
                                       n=int(n)).y[-1, 0]) - exact), 1e-17)
                  for n in steps]
        slope = np.polyfit(np.log(h), np.log(errors), 1)[0]
        ax.loglog(h, errors, color=scheme.series[i], marker="o", markersize=3.5,
                  label=f"{name} — order {order}, measured {slope:.2f}")
    finish(ax, scheme, title=r"Halving the step divides the error by $2^p$",
           xlabel="step size h", ylabel=r"error in $y(1)$ for $y' = -y$",
           grid="both", legend=True, legend_kw=dict(loc="lower right"))


@figure("ode-stability-regions", GUIDE,
        "Regions of absolute stability, measured by taking one step of each "
        "method", size=(7.0, 3.6))
def stability_regions(fig, scheme):
    ax1 = fig.add_subplot(1, 2, 1)
    explicit = [("Euler", "euler"), ("Heun", "heun"), ("RK4", "rk4")]
    for i, (label, key) in enumerate(explicit):
        re, im, grid = stability_grid(key, -4.0, 1.0, 3.6, 260, 200)
        ax1.contour(re, im, grid, levels=[1.0], colors=[scheme.series[i]],
                    linewidths=1.8)
        ax1.contourf(re, im, grid, levels=[0.0, 1.0], colors=[scheme.series[i]],
                     alpha=0.13)
        ax1.plot([], [], color=scheme.series[i], label=label)
    ax1.axhline(0, color=scheme.axis, linewidth=0.8)
    ax1.axvline(0, color=scheme.axis, linewidth=0.8)
    finish(ax1, scheme, title="Explicit: a bounded region",
           xlabel=r"Re$(h\lambda)$", ylabel=r"Im$(h\lambda)$", grid="both",
           legend=True, legend_kw=dict(loc="upper left", fontsize=8))
    ax1.set_aspect("equal")

    ax2 = fig.add_subplot(1, 2, 2)
    implicit = [("backward Euler", "backward_euler"),
                ("trapezoidal", "trapezoidal"), ("Radau IIA", "radau_iia")]
    for i, (label, key) in enumerate(implicit):
        re, im, grid = stability_grid(key, -6.0, 4.0, 4.0, 140, 110)
        ax2.contour(re, im, grid, levels=[1.0], colors=[scheme.series[i]],
                    linewidths=1.8)
        ax2.plot([], [], color=scheme.series[i], label=label)
    ax2.axvspan(-6.0, 0.0, color=scheme.series[0], alpha=0.10)
    ax2.axhline(0, color=scheme.axis, linewidth=0.8)
    ax2.axvline(0, color=scheme.axis, linewidth=0.8)
    finish(ax2, scheme, title="Implicit: the whole left half plane, and more",
           xlabel=r"Re$(h\lambda)$", grid="both", legend=True,
           legend_kw=dict(loc="lower left", fontsize=8))
    ax2.set_aspect("equal")
    annotate(ax2, "backward Euler damps inside\nthis circle too — more than\n"
                  "A-stability asks for", (1.9, 0.6), (2.6, 2.6), scheme,
             ha="center")
    fig.tight_layout()


@figure("ode-stiffness", GUIDE,
        "An explicit method above its stability limit does not lose accuracy, "
        "it diverges", size=(7.0, 3.6))
def stiffness(fig, scheme):
    stiff = lambda t, y: [-1000 * (y[0] - np.cos(t)) - np.sin(t)]
    explicit = rk4(stiff, (0, 1), [1.0], n=100)
    implicit = radau_iia(stiff, (0, 1), [1.0], n=100)
    fine = rk4(stiff, (0, 1), [1.0], n=2000)

    ax1 = fig.add_subplot(1, 2, 1)
    t = np.linspace(0, 1, 400)
    ax1.plot(t, np.cos(t), color=scheme.secondary, linewidth=1.4,
             linestyle=(0, (5, 2)), label=r"$\cos t$, the solution")
    ax1.plot(implicit.t, implicit.y[:, 0], color=scheme.series[0],
             label="Radau IIA, 100 steps")
    ax1.plot(fine.t, fine.y[:, 0], color=scheme.series[2], linewidth=1.2,
             label="RK4, 2000 steps")
    finish(ax1, scheme, title="What the solution looks like", xlabel="t",
           ylabel="y", grid="both", legend=True,
           legend_kw=dict(loc="lower left", fontsize=8))
    ax1.set_ylim(0.4, 1.15)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.semilogy(explicit.t, np.abs(explicit.y[:, 0]) + 1e-16,
                 color=scheme.series[1], label="RK4, 100 steps")
    ax2.semilogy(implicit.t, np.abs(implicit.y[:, 0]), color=scheme.series[0],
                 label="Radau IIA, 100 steps")
    finish(ax2, scheme, title=r"$|y|$ with a step of 0.01 — the limit is 0.0028",
           xlabel="t", ylabel="|y|", grid="both", legend=True,
           legend_kw=dict(loc="lower right", fontsize=8))
    ax2.set_ylim(1e-2, 1e40)
    fig.tight_layout()


@figure("ode-adaptive-steps", GUIDE,
        "How an adaptive integrator spends its steps on a problem with a fast "
        "transient", size=(7.0, 4.0))
def adaptive_steps(fig, scheme):
    # Van der Pol at mu = 12: smooth stretches punctuated by fast switches.
    mu = 12.0
    vdp = lambda t, y: np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])
    solution = solve_ivp(vdp, (0, 40), [2.0, 0.0], rtol=1e-8, atol=1e-10)

    layout = fig.add_gridspec(2, 1, height_ratios=(1.5, 1.0), hspace=0.18)
    ax = fig.add_subplot(layout[0])
    ax.plot(solution.t, solution.y[:, 0], color=scheme.series[0],
            label="position")
    finish(ax, scheme,
           title=f"Van der Pol at $\\mu$ = {mu:g}: "
                 f"{solution.n_accepted} steps accepted, "
                 f"{solution.n_rejected} rejected",
           ylabel="y", grid="both")
    ax.set_xticklabels([])
    ax.set_xlim(0, 40)

    ax2 = fig.add_subplot(layout[1])
    steps = np.diff(solution.t)
    ax2.semilogy(solution.t[:-1], steps, color=scheme.series[1], linewidth=1.4)
    finish(ax2, scheme, xlabel="t", ylabel="step size", grid="both")
    ax2.set_xlim(0, 40)
    annotate(ax2, "the step collapses at each fast switch\nand recovers "
                  "over the slow stretch between",
             (float(solution.t[int(np.argmin(steps))]), float(np.min(steps))),
             (17.0, float(np.min(steps)) * 2.2), scheme, ha="center",
             va="bottom")


@figure("ode-symplectic-energy", GUIDE,
        "Energy of a harmonic oscillator over 200 time units, symplectic "
        "against general purpose", size=(7.0, 3.6))
def symplectic_energy(fig, scheme):
    span, n = (0, 200), 20000
    sym = leapfrog(lambda q: q, lambda p: p, span, [1.0], [0.0], n=n)
    q, p = sym.y[:, 0], sym.y[:, 1]
    sym_energy = 0.5 * (p ** 2 + q ** 2)

    harmonic = lambda t, y: np.array([y[1], -y[0]])
    explicit = rk4(harmonic, span, [1.0, 0.0], n=n)
    rk_energy = 0.5 * (explicit.y[:, 1] ** 2 + explicit.y[:, 0] ** 2)
    forward = euler(harmonic, span, [1.0, 0.0], n=n)
    euler_energy = 0.5 * (forward.y[:, 1] ** 2 + forward.y[:, 0] ** 2)

    ax = fig.add_subplot()
    ax.semilogy(sym.t, np.abs(sym_energy - 0.5) + 1e-18, color=scheme.series[0],
                label=f"leapfrog (symplectic) — drift "
                      f"{energy_drift(sym, lambda y: 0.5 * (y[1] ** 2 + y[0] ** 2)):.1e}")
    ax.semilogy(explicit.t, np.abs(rk_energy - 0.5) + 1e-18,
                color=scheme.series[2], label="RK4 — fourth order, still drifts")
    ax.semilogy(forward.t, np.abs(euler_energy - 0.5) + 1e-18,
                color=scheme.series[1], label="forward Euler — energy grows")
    finish(ax, scheme,
           title="What matters over long times is the drift, not the order",
           xlabel="t", ylabel="|energy error|", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=8))
    ax.set_ylim(1e-13, 1e4)


@figure("ode-events", GUIDE,
        "Event location finds the crossing on the dense output, not on the "
        "step grid", size=(7.0, 3.6))
def events(fig, scheme):
    throw = lambda t, y: [y[1], -9.81]
    hits_ground = lambda t, y: y[0]
    solution, t_events, y_events = solve_ivp_events(
        throw, (0, 10), [5.0, 10.0], events=hits_ground, terminal=True,
        rtol=1e-10)
    landing = (10 + np.sqrt(100 + 2 * 9.81 * 5)) / 9.81

    ax = fig.add_subplot()
    dense = np.linspace(0, float(solution.t[-1]), 400)
    ax.plot(dense, solution(dense)[:, 0], color=scheme.series[0],
            label="dense output")
    ax.scatter(solution.t, solution.y[:, 0], s=26, color=scheme.series[0],
               zorder=3, edgecolors=scheme.surface, linewidths=1.0,
               label=f"the {solution.t.size} steps the integrator took")
    ax.scatter(t_events[0], y_events[0][:, 0], s=70, marker="*", zorder=4,
               color=scheme.series[1], label="located event")
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    error = abs(float(t_events[0][0]) - float(landing))
    finish(ax, scheme,
           title=f"The event time is right to {error:.0e}, with steps of "
                 f"{float(np.max(np.diff(solution.t))):.2f}",
           xlabel="t", ylabel="height", grid="both", legend=True,
           legend_kw=dict(loc="lower left", fontsize=8))


@figure("ode-bvp-eigenfunctions", GUIDE,
        "Sturm-Liouville eigenfunctions and eigenvalues against their exact "
        "values", size=(7.0, 3.6))
def bvp_eigenfunctions(fig, scheme):
    # -u'' = lambda u on (0, 1) with u(0) = u(1) = 0: eigenvalues (n pi)^2.
    values, x, vectors = sturm_liouville(lambda t: 1.0, lambda t: 0.0,
                                         lambda t: 1.0, (0.0, 1.0), n=200,
                                         n_eigen=12)
    ax1 = fig.add_subplot(1, 2, 1)
    colours = ordinal(scheme, 4)
    for k in range(4):
        shape = vectors[:, k]
        shape = shape / np.max(np.abs(shape)) * np.sign(shape[1])
        ax1.plot(x, shape, color=colours[k], label=f"n = {k + 1}")
    ax1.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax1, scheme, title="Eigenfunctions", xlabel="x", ylabel="shape",
           grid="both", legend=True, legend_kw=dict(loc="lower center", ncol=4,
                                                    fontsize=7.5))
    ax1.set_ylim(-1.5, 1.35)

    ax2 = fig.add_subplot(1, 2, 2)
    modes = np.arange(1, 13)
    exact = (modes * np.pi) ** 2
    relative = np.abs(values[:12] - exact) / exact
    ax2.semilogy(modes, relative, color=scheme.series[0], marker="o",
                 markersize=4)
    finish(ax2, scheme, title=r"Relative error in $\lambda_n = (n\pi)^2$",
           xlabel="mode n", ylabel="relative error", grid="both")
    annotate(ax2, "the discretization resolves the low modes\n"
                  "far better than the high ones",
             (modes[-1], relative[-1]), (modes[1], relative[-1] * 0.55), scheme,
             ha="left", va="top")
    fig.tight_layout()


@figure("ode-bvp-solution", GUIDE,
        "A boundary value problem solved by finite differences, and the order "
        "of the method", size=(7.0, 3.4))
def bvp_solution(fig, scheme):
    # The solver's form is y'' = p y' + q y + r, so q = -1 is y'' = -y, whose
    # solution here is sin(x)/sin(1).  No difference formula reproduces that
    # exactly, so the convergence panel measures the method rather than an
    # accident of the data.
    exact = lambda x: np.sin(x) / np.sin(1.0)
    solve = lambda n: finite_difference_bvp(lambda x: 0.0, lambda x: -1.0,
                                            lambda x: 0.0, (0, 1), 0.0, 1.0,
                                            n=int(n))
    ax1 = fig.add_subplot(1, 2, 1)
    solution = solve(20)
    dense = np.linspace(0, 1, 400)
    ax1.plot(dense, exact(dense), color=scheme.secondary, linewidth=1.6,
             linestyle=(0, (5, 2)), label=r"exact, $\sin x/\sin 1$")
    ax1.scatter(solution.t, solution.y[:, 0], s=16, color=scheme.series[0],
                zorder=3, label="finite differences, n = 20")
    ax1.scatter([0, 1], [0.0, 1.0], s=52, color=scheme.series[1], zorder=4,
                marker="s", label="prescribed at both ends")
    finish(ax1, scheme, title="Conditions at both ends, so it cannot be marched",
           xlabel="x", ylabel="y", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=8))

    ax2 = fig.add_subplot(1, 2, 2)
    counts = np.array([10, 20, 40, 80, 160, 320])
    errors = []
    for n in counts:
        s = solve(n)
        errors.append(float(np.max(np.abs(s.y[:, 0] - exact(s.t)))))
    slope = np.polyfit(np.log(1.0 / counts), np.log(errors), 1)[0]
    ax2.loglog(1.0 / counts, errors, color=scheme.series[0], marker="o",
               markersize=4, label=f"measured order {slope:.2f}")
    finish(ax2, scheme, title="Error against grid spacing", xlabel="h",
           ylabel="max error", grid="both", legend=True,
           legend_kw=dict(loc="lower right"))
    fig.tight_layout()
