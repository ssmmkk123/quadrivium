"""Figures for the interpolation guide."""

from __future__ import annotations

from quadrivium import numeric as np

from quadrivium.interpolate import (akima_spline, barycentric, bezier,
                                    bspline_basis, chebyshev_nodes, cubic_spline,
                                    floater_hormann, kriging, natural_cubic_spline,
                                    nurbs_circle, open_uniform_knots, pchip,
                                    rbf_interpolation, smoothing_spline)

from . import figure
from figstyle import (annotate, diverging_cmap, finish, label_line, ordinal,
                      sequential_cmap)

GUIDE = "guides/interpolate.md"

RUNGE = lambda t: 1.0 / (1.0 + 25.0 * t ** 2)


@figure("interpolate-runge", GUIDE,
        "Runge's phenomenon: the same degree-20 interpolant on equally spaced "
        "and on Chebyshev nodes", size=(7.0, 4.4))
def runge(fig, scheme):
    n = 21
    grid = np.linspace(-1, 1, 800)
    equi = np.linspace(-1, 1, n)
    cheb = chebyshev_nodes(n, -1, 1)
    p_equi = barycentric(equi, RUNGE(equi))
    p_cheb = barycentric(cheb, RUNGE(cheb))

    layout = fig.add_gridspec(2, 1, height_ratios=(2.1, 1.0), hspace=0.35)
    ax = fig.add_subplot(layout[0])
    ax.plot(grid, RUNGE(grid), color=scheme.secondary, linewidth=1.6,
            label=r"$1/(1+25x^2)$")
    ax.plot(grid, p_equi(grid), color=scheme.series[1],
            label="degree 20, equally spaced")
    ax.plot(grid, p_cheb(grid), color=scheme.series[0],
            label="degree 20, Chebyshev nodes")
    ax.scatter(equi, RUNGE(equi), s=14, color=scheme.series[1], zorder=3,
               edgecolors=scheme.surface, linewidths=0.6)
    ax.scatter(cheb, RUNGE(cheb), s=14, color=scheme.series[0], zorder=3,
               marker="s", edgecolors=scheme.surface, linewidths=0.6)
    ax.set_ylim(-1.4, 3.0)
    finish(ax, scheme,
           title="Both polynomials pass through every point they were given",
           ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="upper center", ncol=3, fontsize=8))
    worst = float(np.max(np.abs(p_equi(grid))))
    annotate(ax, f"the equally spaced interpolant\nreaches {worst:.0f} near the ends",
             (-0.955, 1.4), (-0.5, 0.9), scheme, ha="left")

    ax2 = fig.add_subplot(layout[1])
    degrees = np.arange(4, 41, 2)
    equi_err, cheb_err = [], []
    for d in degrees:
        xs = np.linspace(-1, 1, d + 1)
        equi_err.append(float(np.max(np.abs(barycentric(xs, RUNGE(xs))(grid)
                                            - RUNGE(grid)))))
        xc = chebyshev_nodes(d + 1, -1, 1)
        cheb_err.append(float(np.max(np.abs(barycentric(xc, RUNGE(xc))(grid)
                                            - RUNGE(grid)))))
    ax2.semilogy(degrees, equi_err, color=scheme.series[1], marker="o",
                 markersize=3, label="equally spaced")
    ax2.semilogy(degrees, cheb_err, color=scheme.series[0], marker="s",
                 markersize=3, label="Chebyshev")
    finish(ax2, scheme, title="Maximum error against degree",
           xlabel="polynomial degree", ylabel="max error", grid="both",
           legend=True, legend_kw=dict(loc="center right", fontsize=8))


@figure("interpolate-spline-shape", GUIDE,
        "Cubic spline, PCHIP and Akima on data with a jump: overshoot against "
        "shape preservation", size=(7.0, 3.6))
def spline_shape(fig, scheme):
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    grid = np.linspace(0, 5, 600)
    curves = [("cubic spline", cubic_spline(x, y), scheme.series[1]),
              ("PCHIP", pchip(x, y), scheme.series[0]),
              ("Akima", akima_spline(x, y), scheme.series[2])]

    ax = fig.add_subplot()
    for name, interp, colour in curves:
        ax.plot(grid, interp(grid), color=colour, label=name)
    ax.scatter(x, y, s=34, color=scheme.ink, zorder=4, label="data",
               edgecolors=scheme.surface, linewidths=1.2)
    ax.axhline(0.0, color=scheme.axis, linewidth=0.8)
    ax.axhline(1.0, color=scheme.axis, linewidth=0.8)
    undershoot = float(np.min(cubic_spline(x, y)(grid)))
    finish(ax, scheme,
           title="A cubic spline is smooth; it is not shape preserving",
           xlabel="x", ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="upper left", ncol=2))
    annotate(ax, f"the cubic spline dips to {undershoot:.3f}\n"
                 "below data that never decreases",
             (2.45, undershoot), (3.0, -0.42), scheme, ha="left")
    ax.set_ylim(-0.5, 1.5)


@figure("interpolate-spline-convergence", GUIDE,
        "Measured convergence rates of piecewise interpolants as the knot "
        "spacing shrinks", size=(7.0, 3.4))
def spline_convergence(fig, scheme):
    f = np.exp
    grid = np.linspace(0, 1, 1001)
    counts = np.array([5, 9, 17, 33, 65, 129])
    h = 1.0 / (counts - 1)
    methods = [("cubic spline", cubic_spline, 4),
               ("PCHIP", pchip, 3),
               ("Akima", akima_spline, 3)]

    ax = fig.add_subplot()
    for i, (name, build, order) in enumerate(methods):
        errors = []
        for n in counts:
            xs = np.linspace(0, 1, n)
            errors.append(float(np.max(np.abs(build(xs, f(xs))(grid) - f(grid)))))
        rate = np.polyfit(np.log(h), np.log(errors), 1)[0]
        ax.loglog(h, errors, color=scheme.series[i], marker="o", markersize=3.5,
                  label=f"{name} — measured order {rate:.1f}")
    reference = 3e-2 * h ** 4
    ax.loglog(h, reference, color=scheme.muted, linewidth=1.0,
              linestyle=(0, (4, 3)))
    label_line(ax, h[1] * 1.05, reference[1] * 2.2, r"slope 4 ($h^4$)", scheme,
               scheme.muted)
    finish(ax, scheme, title="Halving the spacing divides a cubic spline's error by 16",
           xlabel="knot spacing h", ylabel="max error", grid="both", legend=True,
           legend_kw=dict(loc="lower right"))


@figure("interpolate-bspline-basis", GUIDE,
        "Cubic B-spline basis functions on an open uniform knot vector, "
        "summing to one", size=(7.0, 3.4))
def bspline_basis_figure(fig, scheme):
    degree, n_control = 3, 8
    knots = open_uniform_knots(n_control, degree)
    t = np.linspace(float(knots[degree]), float(knots[n_control]) - 1e-12, 600)

    ax = fig.add_subplot()
    total = np.zeros_like(t)
    colours = ordinal(scheme, n_control)
    for i in range(n_control):
        values = np.array([bspline_basis(i, degree, knots, float(tv)) for tv in t])
        total += values
        ax.plot(t, values, color=colours[i], linewidth=1.5)
        peak = int(np.argmax(values))
        if values[peak] > 0.2:
            ax.text(t[peak], values[peak] + 0.03, f"$N_{{{i},3}}$", fontsize=7.5,
                    ha="center", color=scheme.secondary)
    ax.plot(t, total, color=scheme.ink, linewidth=1.4, linestyle=(0, (5, 2)),
            label="sum of all eight — partition of unity")
    ax.scatter(knots, np.zeros_like(knots), s=16, marker="|",
               color=scheme.muted, linewidths=1.4, label="knots")
    finish(ax, scheme,
           title="Each basis function is local; together they sum to 1 everywhere",
           xlabel="parameter t", ylabel="basis value", grid="both", legend=True,
           legend_kw=dict(loc="upper center", ncol=2))
    ax.set_ylim(-0.05, 1.30)


@figure("interpolate-curves", GUIDE,
        "A Bezier curve with its control polygon and a NURBS circle, which no "
        "polynomial curve can represent", size=(7.0, 3.6))
def curves(fig, scheme):
    control = np.array([[0.0, 0.0], [0.25, 1.15], [0.8, -0.35], [1.0, 0.75]])
    curve = bezier(control)
    t = np.linspace(0, 1, 400)
    points = np.array([curve(float(tv)) for tv in t])

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(control[:, 0], control[:, 1], color=scheme.muted, linewidth=1.0,
             linestyle=(0, (4, 3)), marker="o", markersize=4,
             label="control polygon")
    ax1.plot(points[:, 0], points[:, 1], color=scheme.series[0],
             label="cubic Bezier")
    finish(ax1, scheme, title="Shape from control points", xlabel="x", ylabel="y",
           grid="both", legend=True, legend_kw=dict(loc="upper right",
                                                    fontsize=8))

    circle = nurbs_circle(radius=1.0)
    s = np.linspace(0, 1, 400)
    on_circle = np.array([circle(float(sv)) for sv in s])
    radii = np.hypot(on_circle[:, 0], on_circle[:, 1])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(on_circle[:, 0], on_circle[:, 1], color=scheme.series[1],
             label="NURBS circle")
    ax2.set_aspect("equal")
    finish(ax2, scheme,
           title="A circle, exactly", xlabel="x", ylabel="y", grid="both")
    annotate(ax2, f"every point is within\n{float(np.max(np.abs(radii - 1))):.0e} of radius 1",
             (0.0, 0.0), (0.0, 0.0), scheme, arrow=False, ha="center", va="center")
    fig.tight_layout()


@figure("interpolate-scattered", GUIDE,
        "Radial basis interpolation of scattered data, with the kriging "
        "variance that says where the data is thin", size=(7.0, 3.6))
def scattered(fig, scheme):
    rng = np.random.default_rng(3)
    points = rng.random((40, 2))
    truth = lambda p: np.sin(np.pi * p[..., 0]) * np.cos(np.pi * p[..., 1])
    values = truth(points)
    gx = np.linspace(0, 1, 90)
    X, Y = np.meshgrid(gx, gx, indexing="ij")
    query = np.stack([X, Y], axis=-1)

    rbf = rbf_interpolation(points, values, kernel="thin_plate")
    surface = np.array([[rbf(query[i, j]) for j in range(gx.size)]
                        for i in range(gx.size)])
    krige = kriging(points, values, sill=1.0, range_=0.35)
    variance = np.array([[krige(query[i, j])[1] for j in range(gx.size)]
                         for i in range(gx.size)])

    ax1 = fig.add_subplot(1, 2, 1)
    levels = np.linspace(-1.05, 1.05, 15)
    ax1.set_rasterization_zorder(0)
    filled = ax1.contourf(X, Y, surface, levels=levels,
                          cmap=diverging_cmap(scheme), zorder=-1)
    ax1.contour(X, Y, surface, levels=levels, colors=scheme.surface,
                linewidths=0.4)
    ax1.scatter(points[:, 0], points[:, 1], s=12, color=scheme.ink,
                edgecolors=scheme.surface, linewidths=0.6, zorder=3)
    ax1.set_aspect("equal")
    ax1.grid(False)
    bar1 = fig.colorbar(filled, ax=ax1, fraction=0.046, pad=0.03,
                        ticks=(-1, 0, 1))
    bar1.ax.tick_params(labelsize=7, color=scheme.muted, labelcolor=scheme.muted)
    bar1.outline.set_edgecolor(scheme.axis)
    finish(ax1, scheme, title="Thin-plate RBF surface", xlabel="x", ylabel="y",
           grid=None)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_rasterization_zorder(0)
    mesh = ax2.contourf(X, Y, variance, levels=12, cmap=sequential_cmap(scheme),
                        zorder=-1)
    ax2.scatter(points[:, 0], points[:, 1], s=12, color=scheme.ink,
                edgecolors=scheme.surface, linewidths=0.6, zorder=3)
    ax2.set_aspect("equal")
    ax2.grid(False)
    bar = fig.colorbar(mesh, ax=ax2, fraction=0.046, pad=0.03)
    bar.ax.tick_params(labelsize=7, color=scheme.muted,
                       labelcolor=scheme.muted)
    bar.outline.set_edgecolor(scheme.axis)
    finish(ax2, scheme, title="Kriging variance — where the data is thin",
           xlabel="x", grid=None)
    fig.tight_layout()


@figure("interpolate-smoothing", GUIDE,
        "Interpolating noisy data reproduces the noise; a smoothing spline "
        "does not", size=(7.0, 3.4))
def smoothing(fig, scheme):
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 40)
    truth = np.sin(2 * np.pi * x)
    y = truth + 0.12 * rng.standard_normal(x.size)
    grid = np.linspace(0, 1, 600)

    ax = fig.add_subplot()
    ax.plot(grid, np.sin(2 * np.pi * grid), color=scheme.secondary,
            linewidth=1.4, linestyle=(0, (5, 2)), label="the signal")
    ax.scatter(x, y, s=16, color=scheme.muted, label="noisy samples", zorder=2)
    ax.plot(grid, natural_cubic_spline(x, y)(grid), color=scheme.series[1],
            label="interpolating spline", linewidth=1.6)
    for lam, colour in ((1e-3, scheme.series[0]), (1e-1, scheme.series[2])):
        ax.plot(grid, smoothing_spline(x, y, lam=lam)(grid), color=colour,
                label=f"smoothing spline, lam={lam:g}")
    finish(ax, scheme,
           title="An interpolant through noisy data interpolates the noise",
           xlabel="x", ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="upper center", ncol=2, fontsize=8))
    ax.set_ylim(-1.9, 2.9)
