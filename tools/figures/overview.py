"""Figures for the pages that describe the library as a whole."""

from __future__ import annotations

import inspect

from quadrivium import numeric as np

import quadrivium as qd
from quadrivium.diff import central_difference, forward_difference
from quadrivium.integrate import (boole_rule, monte_carlo, simpson_rule,
                                  trapezoid_rule)
from quadrivium.interpolate import barycentric, chebyshev_nodes, cubic_spline
from quadrivium.linalg import conjugate_gradient, gmres, kron
from quadrivium.ode import euler, heun, rk4
from quadrivium.pde import heat_crank_nicolson, poisson_2d_direct, weno_burgers
from quadrivium.transforms import fft, fftfreq
from quadrivium.rootfind import bisection, brent, newton
from quadrivium.stochastic import milstein

from . import figure
from figstyle import annotate, finish, ordinal


@figure("overview-gallery", "index.md",
        "One result from six of the thirteen subpackages", size=(7.0, 4.6))
def gallery(fig, scheme):
    layout = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)

    # 1. A chaotic trajectory, from the adaptive integrator.
    def lorenz(t, y):
        return np.array([10.0 * (y[1] - y[0]),
                         y[0] * (28.0 - y[2]) - y[1],
                         y[0] * y[1] - 8.0 / 3.0 * y[2]])

    solution = qd.solve_ivp(lorenz, (0, 40), [1.0, 1.0, 20.0], rtol=1e-9,
                            atol=1e-10)
    ax = fig.add_subplot(layout[0, 0])
    ax.plot(solution.y[:, 0], solution.y[:, 2], color=scheme.series[0],
            linewidth=0.5)
    finish(ax, scheme, title="ode: Lorenz", xlabel="x", ylabel="z",
           grid="both")

    # 2. Krylov convergence.
    tri = (np.diag(2.0 * np.ones(24)) + np.diag(-np.ones(23), 1)
           + np.diag(-np.ones(23), -1))
    A = kron(np.eye(24), tri) + kron(tri, np.eye(24))
    b = np.ones(A.shape[0])
    ax = fig.add_subplot(layout[0, 1])
    for i, (name, run) in enumerate([("CG", conjugate_gradient(A, b, tol=1e-12,
                                                               max_iter=400)),
                                     ("GMRES", gmres(A, b, tol=1e-12,
                                                     max_iter=400))]):
        ax.semilogy(run.residuals, color=scheme.series[i], label=name)
    finish(ax, scheme, title="linalg: Krylov", xlabel="iteration",
           ylabel="residual", grid="both", legend=True,
           legend_kw=dict(loc="upper right", fontsize=7))

    # 3. Runge's phenomenon.
    runge = lambda t: 1 / (1 + 25 * t ** 2)
    grid = np.linspace(-1, 1, 500)
    equi = np.linspace(-1, 1, 15)
    cheb = chebyshev_nodes(15, -1, 1)
    ax = fig.add_subplot(layout[0, 2])
    ax.plot(grid, runge(grid), color=scheme.secondary, linewidth=1.2)
    ax.plot(grid, barycentric(equi, runge(equi))(grid), color=scheme.series[1],
            linewidth=1.2, label="equispaced")
    ax.plot(grid, barycentric(cheb, runge(cheb))(grid), color=scheme.series[0],
            linewidth=1.2, label="Chebyshev")
    ax.set_ylim(-0.5, 1.6)
    finish(ax, scheme, title="interpolate: nodes", xlabel="x",
           ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="upper right", fontsize=7))

    # 4. A shock, captured.
    burgers = weno_burgers(lambda x: 0.5 + np.sin(2 * np.pi * x), (0, 1),
                           (0, 0.3), nx=200, cfl=0.4)
    ax = fig.add_subplot(layout[1, 0])
    colours = ordinal(scheme, 3)
    for colour, level in zip(colours, (0, burgers.u.shape[0] // 2, -1)):
        ax.plot(burgers.grids[0], burgers.u[level], color=colour, linewidth=1.3)
    finish(ax, scheme, title="pde: WENO5", xlabel="x", ylabel="u",
           grid="both")

    # 5. A spectrum.
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    signal = (np.sin(2 * np.pi * 40 * t) + 0.4 * np.sin(2 * np.pi * 120 * t)
              + 0.4 * np.random.default_rng(0).standard_normal(n))
    ax = fig.add_subplot(layout[1, 1])
    magnitude = np.abs(fft(signal)) * 2 / n
    freqs = fftfreq(n, 1 / fs)
    ax.plot(freqs[:n // 2], magnitude[:n // 2], color=scheme.series[0],
            linewidth=1.0)
    ax.set_xlim(0, 200)
    finish(ax, scheme, title="transforms: FFT",
           xlabel="frequency (Hz)", ylabel="amplitude", grid="both")

    # 6. Stochastic paths.
    ax = fig.add_subplot(layout[1, 2])
    for seed in range(10):
        path = milstein(lambda x, t: 1.0 * x, lambda x, t: 0.6 * x, (0, 1),
                        [1.0], n=400, rng=seed)
        ax.plot(path.t, path.y[:, 0], color=scheme.series[0], linewidth=0.8,
                alpha=0.6)
    finish(ax, scheme, title="stochastic: SDE", xlabel="t",
           ylabel="X", grid="both")


@figure("getting-started-tolerance", "getting-started.md",
        "What asking for more digits costs, by method", size=(7.0, 3.6))
def tolerance_cost(fig, scheme):
    f = lambda x: x ** 3 - 2 * x - 5
    df = lambda x: 3 * x ** 2 - 2
    tolerances = np.logspace(-2, -15, 14)
    runs = {"bisection": [], "Brent": [], "Newton": []}
    accuracy = {"bisection": [], "Brent": [], "Newton": []}
    root = 2.0945514815423265
    for tol in tolerances:
        for name, call in (("bisection", lambda t: bisection(f, 1, 3, tol=t)),
                           ("Brent", lambda t: brent(f, 1, 3, tol=t)),
                           ("Newton", lambda t: newton(f, 3.0, df, tol=t))):
            result = call(float(tol))
            runs[name].append(result.function_calls)
            accuracy[name].append(max(abs(float(result.root) - root), 1e-17))

    ax1 = fig.add_subplot(1, 2, 1)
    for i, name in enumerate(runs):
        ax1.semilogx(tolerances, runs[name], color=scheme.series[i], marker="o",
                     markersize=3.5, label=name)
    ax1.invert_xaxis()
    finish(ax1, scheme, title="Cost of the tolerance you ask for",
           xlabel="requested tol", ylabel="function evaluations", grid="both",
           legend=True, legend_kw=dict(loc="upper left"))

    ax2 = fig.add_subplot(1, 2, 2)
    for i, name in enumerate(accuracy):
        ax2.loglog(tolerances, accuracy[name], color=scheme.series[i],
                   marker="o", markersize=3.5, label=name)
    ax2.loglog(tolerances, tolerances, color=scheme.muted, linewidth=1.0,
               linestyle=(0, (4, 3)))
    ax2.invert_xaxis()
    finish(ax2, scheme, title="Accuracy you actually get", xlabel="requested tol",
           ylabel="error in the root", grid="both", legend=True,
           legend_kw=dict(loc="lower left"))
    annotate(ax2, "the dashed line is\nwhat you asked for", (1e-8, 1e-8),
             (1e-4, 1e-13), scheme, ha="center")
    fig.tight_layout()


def measured_order(errors, sizes):
    """Slope of log error against log h, which is the observed order."""
    return float(-np.polyfit(np.log(np.asarray(sizes, dtype=float)),
                             np.log(np.asarray(errors, dtype=float)), 1)[0])


@figure("design-measured-orders", "design.md",
        "Convergence orders measured across the library against the orders "
        "theory promises", size=(7.0, 4.0))
def measured_orders(fig, scheme):
    checks = []

    # Time stepping.
    exact = float(np.exp(-1.0))
    counts = np.array([20, 40, 80, 160])
    for name, method, order in (("ode: forward Euler", euler, 1),
                                ("ode: Heun", heun, 2),
                                ("ode: RK4", rk4, 4)):
        errors = [abs(float(method(lambda t, y: -y, (0, 1), [1.0],
                                   n=int(n)).y[-1, 0]) - exact) for n in counts]
        checks.append((name, order, measured_order(errors, counts)))

    # Quadrature.
    integral = 1 - np.cos(1.0)
    for name, rule, order in (("integrate: trapezoid", trapezoid_rule, 2),
                              ("integrate: Simpson", simpson_rule, 4),
                              ("integrate: Boole", boole_rule, 6)):
        errors = [abs(float(rule(np.sin, 0, 1, n=int(n))) - integral)
                  for n in (8, 16, 32, 64)]
        checks.append((name, order, measured_order(errors, (8, 16, 32, 64))))

    # Interpolation.
    grid = np.linspace(0, 1, 601)
    knots = (9, 17, 33, 65)
    errors = []
    for n in knots:
        xs = np.linspace(0, 1, n)
        errors.append(float(np.max(np.abs(cubic_spline(xs, np.exp(xs))(grid)
                                          - np.exp(grid)))))
    checks.append(("interpolate: cubic spline", 4, measured_order(errors, knots)))

    # Differentiation, at the step size each formula is meant to use.
    for name, rule, order in (("diff: forward difference", forward_difference, 1),
                              ("diff: central difference", central_difference, 2)):
        steps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
        errors = [abs(rule(np.sin, 1.0, h=float(h)) - float(np.cos(1.0)))
                  for h in steps]
        checks.append((name, order, measured_order(errors, 1.0 / steps)))

    # A PDE, in space.
    source = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    sizes = (10, 20, 40)
    errors = []
    for n in sizes:
        solution = poisson_2d_direct(source, (0, 1), (0, 1), n, n)
        X, Y = np.meshgrid(solution.grids[0], solution.grids[1], indexing="ij")
        errors.append(float(np.max(np.abs(np.asarray(solution.u)
                                          - np.sin(np.pi * X) * np.sin(np.pi * Y)))))
    checks.append(("pde: Poisson, 5-point", 2, measured_order(errors, sizes)))

    # Diffusion, in time.
    u0 = lambda x: np.sin(np.pi * x)
    errors, steps = [], (5, 10, 20)
    for nt in steps:
        solution = heat_crank_nicolson(u0, 0.1, (0, 1), (0, 1.0), nx=400, nt=nt)
        xs = solution.grids[0]
        errors.append(float(np.max(np.abs(
            solution.u[-1] - np.exp(-0.1 * np.pi ** 2) * np.sin(np.pi * xs)))))
    checks.append(("pde: Crank-Nicolson, in time", 2,
                   measured_order(errors, steps)))

    ax = fig.add_subplot()
    names = [name for name, _, _ in checks]
    positions = np.arange(len(checks))
    theory = [order for _, order, _ in checks]
    got = [value for _, _, value in checks]
    ax.barh(positions - 0.2, theory, height=0.36, color=scheme.muted,
            label="order the theory promises")
    ax.barh(positions + 0.2, got, height=0.36, color=scheme.series[0],
            label="order measured here")
    for pos, value in zip(positions, got):
        ax.text(value + 0.08, pos + 0.2, f"{value:.2f}", va="center",
                fontsize=7.5, color=scheme.secondary)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    finish(ax, scheme,
           title="Every one of these is checked by the test suite on every "
                 "commit",
           xlabel="convergence order", grid="x", legend=True,
           legend_kw=dict(loc="lower right"))
    ax.set_xlim(0, 7.2)


@figure("limitations-accuracy-floors", "limitations.md",
        "What each technique can and cannot deliver in double precision",
        size=(7.0, 3.6))
def accuracy_floors(fig, scheme):
    truth = float(np.cos(1.0))
    forward = min(abs(forward_difference(np.sin, 1.0, h=float(h)) - truth)
                  for h in np.logspace(-1, -12, 60))
    central = min(abs(central_difference(np.sin, 1.0, h=float(h)) - truth)
                  for h in np.logspace(-1, -12, 60))
    automatic = abs(float(qd.derivative(np.sin, 1.0)) - truth)
    monte = float(np.mean([abs(float(monte_carlo(lambda x: x ** 2, 0, 1,
                                                 n=1_000_000, rng=seed)) - 1 / 3)
                           for seed in range(5)]))
    entries = [("forward difference", forward, r"$\sqrt{\varepsilon}$"),
               ("central difference", central, r"$\varepsilon^{2/3}$"),
               ("automatic differentiation", max(automatic, 1e-16),
                "exact" if automatic == 0.0 else r"$\varepsilon$"),
               ("Monte Carlo, $10^6$ samples", monte, r"$n^{-1/2}$"),
               ("double precision itself", float(np.finfo(float).eps),
                r"$\varepsilon$")]

    ax = fig.add_subplot()
    positions = np.arange(len(entries))
    values = [value for _, value, _ in entries]
    colours = [scheme.series[1], scheme.series[1], scheme.series[0],
               scheme.series[2], scheme.muted]
    ax.barh(positions, values, height=0.5, color=colours)
    ax.set_xscale("log")
    for pos, (name, value, law) in zip(positions, entries):
        text = ("0 — exact to the last bit" if law == "exact"
                else f"{value:.1e}   ({law})")
        ax.text(value * 1.7, pos, text, va="center", fontsize=8,
                color=scheme.secondary)
    ax.set_yticks(positions)
    ax.set_yticklabels([name for name, _, _ in entries], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    finish(ax, scheme,
           title="Best achievable error, measured — not the method's fault",
           xlabel="smallest error achievable (log scale)", grid="x")
    ax.set_xlim(1e-17, 1e2)


@figure("api-public-names", "api/index.md",
        "How the library's public names are distributed across its "
        "subpackages", size=(7.0, 3.8))
def public_names(fig, scheme):
    packages = ["core", "linalg", "rootfind", "interpolate", "approx", "diff",
                "integrate", "ode", "pde", "optimize", "transforms",
                "stochastic", "special"]
    counts = []
    top_level = []
    for name in packages:
        module = getattr(qd, name)
        exported = [n for n in getattr(module, "__all__", [])
                    if not inspect.ismodule(getattr(module, n, None))]
        counts.append(len(exported))
        top_level.append(sum(1 for n in exported if getattr(qd, n, None)
                             is getattr(module, n, None) and n in qd.__all__))

    order = np.argsort(counts)
    packages = [packages[i] for i in order]
    counts = [counts[i] for i in order]
    top_level = [top_level[i] for i in order]

    ax = fig.add_subplot()
    positions = np.arange(len(packages))
    ax.barh(positions, counts, height=0.6, color=scheme.series[0],
            label="public names in the subpackage")
    ax.barh(positions, top_level, height=0.6, color=scheme.series[1],
            label="also re-exported at the top level")
    for pos, (total, top) in zip(positions, zip(counts, top_level)):
        ax.text(total + 2, pos, f"{total}  ({top} at top level)", va="center",
                fontsize=7.5, color=scheme.secondary)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"`{name}`".strip("`") for name in packages], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    finish(ax, scheme, title=f"{sum(counts)} public names across "
                             f"{len(packages)} subpackages",
           xlabel="public names", grid="x", legend=True,
           legend_kw=dict(loc="lower right"))
    ax.set_xlim(0, max(counts) * 1.35)
