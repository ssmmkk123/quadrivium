"""Figures for the partial differential equations guide."""

from __future__ import annotations

from functools import lru_cache

from quadrivium import numeric as np

from quadrivium.pde import (advection_upwind, heat_crank_nicolson, heat_ftcs,
                            lax_wendroff, lid_driven_cavity, multigrid_solve,
                            poisson_2d_direct, poisson_2d_iterative,
                            poisson_9point, stability_ratio, tvd_scheme,
                            weno_burgers)

from . import figure
from figstyle import annotate, diverging_cmap, finish, ordinal

GUIDE = "guides/pde.md"


@figure("pde-heat-evolution", GUIDE,
        "Crank-Nicolson diffusion profiles against the analytic solution",
        size=(7.0, 3.6))
def heat_evolution(fig, scheme):
    alpha = 0.1
    u0 = lambda x: np.sin(np.pi * x)
    solution = heat_crank_nicolson(u0, alpha, (0, 1), (0, 1.0), nx=80, nt=200)
    x = solution.grids[0]
    times = np.linspace(0, 1.0, 6)
    colours = ordinal(scheme, times.size)

    ax1 = fig.add_subplot(1, 2, 1)
    for colour, t in zip(colours, times):
        level = int(round(t / 1.0 * (solution.u.shape[0] - 1)))
        ax1.plot(x, solution.u[level], color=colour, linewidth=1.6,
                 label=f"t = {t:.1f}")
        exact = np.exp(-alpha * np.pi ** 2 * t) * np.sin(np.pi * x)
        ax1.plot(x, exact, color=scheme.ink, linewidth=0.8,
                 linestyle=(0, (2, 2)))
    finish(ax1, scheme, title="Profiles, with the exact solution dotted",
           xlabel="x", ylabel="u", grid="both", legend=True,
           legend_kw=dict(loc="upper right", fontsize=7.5, ncol=2))
    ax1.set_ylim(0, 1.25)

    ax2 = fig.add_subplot(1, 2, 2)
    for i, nx in enumerate((20, 40, 80)):
        s = heat_crank_nicolson(u0, alpha, (0, 1), (0, 1.0), nx=nx, nt=4 * nx)
        xs = s.grids[0]
        errors = [float(np.max(np.abs(
            s.u[k] - np.exp(-alpha * np.pi ** 2 * t) * np.sin(np.pi * xs))))
            for k, t in enumerate(np.linspace(0, 1.0, s.u.shape[0]))]
        ax2.semilogy(np.linspace(0, 1.0, s.u.shape[0]), np.maximum(errors, 1e-16),
                     color=scheme.series[i], label=f"nx = {nx}")
    finish(ax2, scheme, title="Error over time, three grids", xlabel="t",
           ylabel="max error", grid="both", legend=True,
           legend_kw=dict(loc="lower right"))
    fig.tight_layout()


@figure("pde-ftcs-stability", GUIDE,
        "The explicit diffusion scheme above and below its stability limit",
        size=(7.0, 3.6))
def ftcs_stability(fig, scheme):
    alpha, nx = 0.1, 40
    # The amplification factor is 1 - 4r sin^2(k dx/2), which first exceeds 1
    # in magnitude at the shortest wavelength the grid carries.  A pure lowest
    # mode therefore stays bounded even above the limit; the 2% Nyquist
    # component added here -- alternating +/- at successive grid points -- is
    # what any real data, or plain rounding, supplies.
    u0 = lambda x: np.sin(np.pi * x) + 0.02 * np.cos(nx * np.pi * x)
    dx = 1.0 / nx
    ratios = (0.25, 0.45, 0.50, 0.52, 0.60)
    colours = ordinal(scheme, len(ratios))

    ax = fig.add_subplot()
    for colour, r in zip(colours, ratios):
        dt = r * dx ** 2 / alpha
        nt = int(round(0.5 / dt))
        # check_stability=False is the only way to see what the guard prevents.
        solution = heat_ftcs(u0, alpha, (0, 1), (0, nt * dt), nx=nx, nt=nt,
                             check_stability=False)
        peak = np.max(np.abs(solution.u), axis=1)
        ax.semilogy(np.linspace(0, nt * dt, peak.size), np.maximum(peak, 1e-6),
                    color=colour,
                    label=f"r = {r:.2f}"
                          + ("  (the limit)" if r == 0.5 else ""))
    ax.axhline(1.0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme,
           title=r"$r = \alpha\,\Delta t/\Delta x^2$ — above 1/2 the scheme "
                 "amplifies every step",
           xlabel="t", ylabel=r"$\max_x |u|$", grid="both", legend=True,
           legend_kw=dict(loc="lower right", ncol=2))
    ax.set_ylim(1e-2, 1e20)
    annotate(ax, "0.52 is not slightly worse than 0.50 — it is\n"
                 "the difference between decay and growth",
             (0.42, 3e2), (0.03, 1e9), scheme, ha="left")


@figure("pde-godunov-barrier", GUIDE,
        "Advecting a square pulse: monotone, second order, and the limiter "
        "that gets both", size=(7.0, 3.6))
def godunov_barrier(fig, scheme):
    square = lambda x: np.where(np.abs(x - 0.3) < 0.1, 1.0, 0.0)
    span, t_span, nx, nt = (0, 1), (0, 0.2), 200, 400
    runs = [("upwind — monotone, smeared",
             advection_upwind(square, 1.0, span, t_span, nx=nx, nt=nt)),
            ("Lax-Wendroff — sharp, oscillates",
             lax_wendroff(square, 1.0, span, t_span, nx=nx, nt=nt)),
            ("TVD, van Leer limiter — both",
             tvd_scheme(square, 1.0, span, t_span, nx=nx, nt=nt,
                        limiter="van_leer"))]

    ax = fig.add_subplot()
    x = runs[0][1].grids[0]
    ax.plot(x, square(x - 0.2), color=scheme.secondary, linewidth=1.3,
            linestyle=(0, (5, 2)), label="exact after one transit")
    for i, (name, run) in enumerate(runs):
        undershoot = float(np.min(run.u[-1]))
        ax.plot(x, run.u[-1], color=scheme.series[i],
                label=f"{name} (min {undershoot:+.3f})")
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme,
           title="Godunov's theorem in one picture",
           xlabel="x", ylabel="u", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=8))
    ax.set_xlim(0.25, 0.85)
    ax.set_ylim(-0.25, 1.45)


@figure("pde-multigrid", GUIDE,
        "Multigrid against single-grid iterations, and the mesh independence "
        "that is the point", size=(7.0, 3.6))
def multigrid(fig, scheme):
    source = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    ax1 = fig.add_subplot(1, 2, 1)
    n = 32
    runs = [("Jacobi", poisson_2d_iterative(source, (0, 1), (0, 1), nx=n, ny=n,
                                            method="jacobi", tol=1e-10,
                                            max_iter=20000)),
            ("Gauss-Seidel", poisson_2d_iterative(source, (0, 1), (0, 1), nx=n,
                                                  ny=n, method="gauss_seidel",
                                                  tol=1e-10, max_iter=20000)),
            ("SOR, optimal omega", poisson_2d_iterative(source, (0, 1), (0, 1),
                                                        nx=n, ny=n, method="sor",
                                                        tol=1e-10,
                                                        max_iter=20000)),
            ("multigrid V-cycles", multigrid_solve(source, (0, 1), (0, 1), n=n,
                                                   tol=1e-10))]
    for i, (name, run) in enumerate(runs):
        residuals = np.asarray(run.residuals, dtype=float)
        ax1.semilogy(np.arange(residuals.size), np.maximum(residuals, 1e-14),
                     color=scheme.series[i],
                     label=f"{name} — {run.iterations:,}")
    finish(ax1, scheme, title=f"Residual on a {n}x{n} grid",
           xlabel="iteration (V-cycle for multigrid)", ylabel="residual norm",
           grid="both", legend=True, legend_kw=dict(loc="upper right",
                                                    fontsize=8))
    ax1.set_xlim(0, 220)

    ax2 = fig.add_subplot(1, 2, 2)
    sizes = np.array([16, 32, 64, 128])
    cycles, sweeps = [], []
    for size in sizes:
        cycles.append(multigrid_solve(source, (0, 1), (0, 1), n=int(size),
                                      tol=1e-10).iterations)
        sweeps.append(poisson_2d_iterative(source, (0, 1), (0, 1), nx=int(size),
                                           ny=int(size), method="sor", tol=1e-10,
                                           max_iter=40000).iterations)
    ax2.loglog(sizes, sweeps, color=scheme.series[2], marker="o", markersize=4,
               label="SOR sweeps")
    ax2.loglog(sizes, cycles, color=scheme.series[3], marker="s", markersize=4,
               label="multigrid V-cycles")
    finish(ax2, scheme, title="Iterations to the same tolerance",
           xlabel="grid points per side", ylabel="iterations", grid="both",
           legend=True, legend_kw=dict(loc="upper left", fontsize=8))
    annotate(ax2, "flat: the work per unknown\ndoes not grow with the mesh",
             (sizes[-2], cycles[-2]), (sizes[0] * 1.15, cycles[0] * 2.4),
             scheme, ha="left")
    fig.tight_layout()


@figure("pde-poisson-accuracy", GUIDE,
        "The 2-D Poisson solution, and what the fourth-order stencil buys",
        size=(7.0, 3.4))
def poisson_accuracy(fig, scheme):
    source = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    solution = poisson_2d_direct(source, (0, 1), (0, 1), nx=60, ny=60)
    X, Y = np.meshgrid(solution.grids[0], solution.grids[1], indexing="ij")

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_rasterization_zorder(0)
    filled = ax1.contourf(X, Y, solution.u, levels=14,
                          cmap=diverging_cmap(scheme), zorder=-1)
    ax1.contour(X, Y, solution.u, levels=14, colors=scheme.surface,
                linewidths=0.4)
    ax1.set_aspect("equal")
    ax1.grid(False)
    bar = fig.colorbar(filled, ax=ax1, fraction=0.046, pad=0.03)
    bar.ax.tick_params(labelsize=7, color=scheme.muted, labelcolor=scheme.muted)
    bar.outline.set_edgecolor(scheme.axis)
    finish(ax1, scheme, title=r"$-\nabla^2 u = f$, five-point stencil",
           xlabel="x", ylabel="y", grid=None)

    ax2 = fig.add_subplot(1, 2, 2)
    sizes = np.array([10, 20, 40, 80])
    for i, (name, solve) in enumerate([("5-point", poisson_2d_direct),
                                       ("9-point (Mehrstellen)", poisson_9point)]):
        errors = []
        for size in sizes:
            s = solve(source, (0, 1), (0, 1), int(size), int(size))
            gx, gy = np.meshgrid(s.grids[0], s.grids[1], indexing="ij")
            exact = np.sin(np.pi * gx) * np.sin(np.pi * gy)
            errors.append(float(np.max(np.abs(np.asarray(s.u) - exact))))
        slope = np.polyfit(np.log(1.0 / sizes), np.log(errors), 1)[0]
        ax2.loglog(1.0 / sizes, errors, color=scheme.series[i], marker="o",
                   markersize=4, label=f"{name} — order {slope:.1f}")
    finish(ax2, scheme, title="Error against grid spacing", xlabel="h",
           ylabel="max error", grid="both", legend=True,
           legend_kw=dict(loc="lower right", fontsize=8))
    fig.tight_layout()


@figure("pde-weno-burgers", GUIDE,
        "A shock forming in Burgers' equation, captured without oscillation",
        size=(7.0, 3.6))
def weno_burgers_figure(fig, scheme):
    smooth = lambda x: 0.5 + np.sin(2 * np.pi * x)
    solution = weno_burgers(smooth, (0, 1), (0, 0.35), nx=256, cfl=0.4)
    x = solution.grids[0]
    times = np.asarray(solution.t, dtype=float)
    wanted = [0.0, 0.08, 0.16, 0.35]
    colours = ordinal(scheme, len(wanted))

    ax = fig.add_subplot()
    for colour, t in zip(colours, wanted):
        level = int(np.argmin(np.abs(times - t)))
        ax.plot(x, solution.u[level], color=colour, linewidth=1.6,
                label=f"t = {times[level]:.2f}")
    finish(ax, scheme,
           title="Smooth data steepens into a shock; WENO5 keeps it sharp and "
                 "monotone",
           xlabel="x", ylabel="u", grid="both", legend=True,
           legend_kw=dict(loc="upper right", ncol=2, fontsize=8))
    ax.set_ylim(-0.8, 2.2)
    annotate(ax, "no overshoot at the discontinuity", (0.62, 1.35),
             (0.25, 1.9), scheme, ha="center")


@lru_cache(maxsize=None)
def cavity(re: float, n: int):
    """The lid-driven cavity, computed once and shared by both schemes."""
    return lid_driven_cavity(re=re, n=n, tol=1e-7, max_iter=40000)


@figure("pde-cavity", GUIDE,
        "The lid-driven cavity at Re = 100, against the Ghia benchmark",
        size=(7.0, 3.6))
def cavity_figure(fig, scheme):
    psi, vorticity, x = cavity(100.0, 65)
    X, Y = np.meshgrid(x, x, indexing="xy")

    ax1 = fig.add_subplot(1, 2, 1)
    levels = np.concatenate([np.linspace(float(psi.min()), 0, 9)[:-1],
                             np.linspace(0, float(psi.max()), 6)])
    ax1.contour(X, Y, psi, levels=levels, colors=[scheme.series[0]],
                linewidths=0.8, linestyles="solid")
    index = np.unravel_index(int(np.argmin(psi)), psi.shape)
    ax1.scatter([x[index[1]]], [x[index[0]]], s=44, color=scheme.series[1],
                zorder=3, label="primary vortex centre")
    ax1.set_aspect("equal")
    finish(ax1, scheme, title="Streamlines, Re = 100", xlabel="x", ylabel="y",
           grid=None, legend=True, legend_kw=dict(loc="lower center",
                                                  fontsize=7.5))

    ax2 = fig.add_subplot(1, 2, 2)
    # Ghia, Ghia & Shin (1982): vortex centre (0.6172, 0.7344), psi = -0.1034.
    measured = (float(x[index[1]]), float(x[index[0]]), float(psi.min()))
    reference = (0.6172, 0.7344, -0.1034)
    labels = ["vortex x", "vortex y", r"$\psi_{\min}$"]
    positions = np.arange(3)
    ax2.barh(positions - 0.19, reference, height=0.34, color=scheme.muted,
             label="Ghia, Ghia & Shin (1982)")
    ax2.barh(positions + 0.19, measured, height=0.34, color=scheme.series[0],
             label="quadrivium, n = 65")
    for pos, (ref, got) in enumerate(zip(reference, measured)):
        ax2.text(0.92, pos, f"{got:+.4f} vs {ref:+.4f}", va="center",
                 fontsize=7.5, color=scheme.secondary)
    ax2.set_yticks(positions)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlim(-0.35, 1.75)
    ax2.set_ylim(-0.6, 3.1)
    finish(ax2, scheme, title="Against the published benchmark", grid="x",
           legend=True, legend_kw=dict(loc="upper right", fontsize=7.5))
    fig.tight_layout()
