"""Figures for the differentiation guide."""

from __future__ import annotations

from quadrivium import numeric as np

from quadrivium.core import CountedFunction
from quadrivium.diff import (central_difference, chebyshev_derivative,
                             forward_gradient,
                             chebyshev_diff_matrix, complex_step_derivative,
                             differentiate_data, differentiation_matrix,
                             forward_difference, fourier_derivative, gradient,
                             gradient_fd, optimal_step_size,
                             richardson_derivative, savitzky_golay_derivative)

from . import figure
from figstyle import annotate, diverging_cmap, finish, label_line

GUIDE = "guides/diff.md"


@figure("diff-step-size", GUIDE,
        "The finite-difference step-size dilemma, and the three ways around "
        "it", size=(7.0, 3.8))
def step_size(fig, scheme):
    exact = float(np.cos(1.0))
    steps = np.logspace(-1, -16, 61)
    forward = [abs(forward_difference(np.sin, 1.0, h=float(h)) - exact) for h in steps]
    central = [abs(central_difference(np.sin, 1.0, h=float(h)) - exact) for h in steps]
    complex_step = abs(complex_step_derivative(np.sin, 1.0) - exact)
    richardson = abs(richardson_derivative(np.sin, 1.0, h=0.1) - exact)
    floor = 1e-18

    ax = fig.add_subplot()
    ax.loglog(steps, np.maximum(forward, floor), color=scheme.series[1],
              label="forward difference, $O(h)$")
    ax.loglog(steps, np.maximum(central, floor), color=scheme.series[0],
              label="central difference, $O(h^2)$")
    ax.axhline(max(complex_step, floor), color=scheme.series[2],
               linestyle=(0, (5, 2)), label="complex step — no subtraction")
    ax.axhline(max(richardson, floor), color=scheme.series[3],
               linestyle=(0, (1, 2)), label="Richardson from h = 0.1")
    best = optimal_step_size(order=1, accuracy=2)
    ax.axvline(best, color=scheme.muted, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_ylim(3e-18, 1e2)
    finish(ax, scheme,
           title="Truncation falls with h; cancellation rises as h shrinks",
           xlabel="step size h", ylabel="error in $f'(1)$", grid="both",
           legend=True, legend_kw=dict(loc="upper center", ncol=2, fontsize=8))
    ax.set_xlim(2e-1, 1e-16)
    annotate(ax, f"optimal_step_size() = {best:.1e}", (best, 1e-9),
             (best * 25, 3e-12), scheme, ha="center")


@figure("diff-gradient-cost", GUIDE,
        "What a gradient costs and what it is worth: evaluations against "
        "dimension, and accuracy", size=(7.0, 3.4))
def gradient_cost(fig, scheme):
    def rosenbrock_like(v):
        # Written in plain arithmetic so the same function can be called with
        # floats, with Dual numbers, and with taped Variables.
        total = 0.0
        for i in range(len(v) - 1):
            total = total + 100.0 * (v[i + 1] - v[i] * v[i]) ** 2 + (1 - v[i]) ** 2
        return total

    def exact_gradient(v):
        v = np.asarray(v, dtype=float)
        g = np.zeros_like(v)
        g[:-1] += -400.0 * v[:-1] * (v[1:] - v[:-1] ** 2) - 2 * (1 - v[:-1])
        g[1:] += 200.0 * (v[1:] - v[:-1] ** 2)
        return g

    dims = np.array([2, 4, 8, 16, 32, 64])
    counts = {"reverse-mode AD (gradient)": [], "forward-mode AD": [],
              "central differences": []}
    for n in dims:
        point = 0.5 * np.ones(int(n))
        for name, call in (("reverse-mode AD (gradient)", gradient),
                           ("forward-mode AD", forward_gradient),
                           ("central differences", gradient_fd)):
            counted = CountedFunction(rosenbrock_like)
            call(counted, point)
            counts[name].append(counted.calls)

    ax1 = fig.add_subplot(1, 2, 1)
    for i, (name, values) in enumerate(counts.items()):
        ax1.plot(dims, values, color=scheme.series[[0, 2, 1][i]], marker="o",
                 markersize=4, label=name)
    finish(ax1, scheme, title="Calls to your function for one gradient",
           xlabel="dimension", ylabel="evaluations", grid="both", legend=True,
           legend_kw=dict(loc="upper left"))

    point = 0.5 * np.ones(8)
    truth = exact_gradient(point)
    errors = {
        "automatic differentiation": float(np.max(np.abs(
            np.asarray(gradient(rosenbrock_like, point), dtype=float) - truth))),
        "central differences": float(np.max(np.abs(
            np.asarray(gradient_fd(rosenbrock_like, point, method="central"),
                       dtype=float) - truth))),
        "forward differences": float(np.max(np.abs(
            np.asarray(gradient_fd(rosenbrock_like, point, method="forward"),
                       dtype=float) - truth))),
    }
    ax2 = fig.add_subplot(1, 2, 2)
    names = list(errors)
    values = [max(v, 1e-18) for v in errors.values()]
    bars = ax2.barh(names, values, height=0.55,
                    color=[scheme.series[0], scheme.series[1], scheme.series[1]])
    ax2.set_xscale("log")
    for bar, value in zip(bars, values):
        ax2.text(value * 1.6, bar.get_y() + bar.get_height() / 2,
                 "exact" if value < 1e-15 else f"{value:.0e}", va="center",
                 fontsize=8, color=scheme.secondary)
    ax2.invert_yaxis()
    finish(ax2, scheme, title="Error in the gradient (dimension 8)",
           xlabel="max component error", grid="x")
    ax2.set_xlim(1e-18, 1e2)
    ax2.tick_params(axis="y", length=0, labelsize=8)
    fig.tight_layout()


@figure("diff-noisy-data", GUIDE,
        "Differentiating noisy samples directly against fitting a local "
        "polynomial first", size=(7.0, 3.6))
def noisy_data(fig, scheme):
    t = np.linspace(0, 2 * np.pi, 300)
    clean = np.sin(t)
    noisy = clean + 1e-3 * np.random.default_rng(0).standard_normal(t.size)
    dt = float(t[1] - t[0])
    raw = differentiate_data(t, noisy)
    smooth = savitzky_golay_derivative(noisy, window=21, poly_order=3, order=1,
                                       dx=dt)

    ax = fig.add_subplot()
    ax.plot(t, raw, color=scheme.series[1], linewidth=1.2,
            label=f"plain differences — max error {np.max(np.abs(raw - np.cos(t))):.2f}")
    ax.plot(t, smooth, color=scheme.series[0], linewidth=1.8,
            label="Savitzky-Golay, window 21, cubic — max error "
                  f"{np.max(np.abs((smooth - np.cos(t))[10:-10])):.3f}")
    ax.plot(t, np.cos(t), color=scheme.secondary, linewidth=1.2,
            linestyle=(0, (5, 2)), label=r"$\cos t$, the answer")
    finish(ax, scheme,
           title="Noise of size 1e-3 becomes noise of size 1e-3/h in a derivative",
           xlabel="t", ylabel="derivative", grid="both", legend=True,
           legend_kw=dict(loc="lower left", fontsize=8))
    ax.set_ylim(-1.9, 2.2)


@figure("diff-spectral-accuracy", GUIDE,
        "Finite differences against spectral differentiation as the grid is "
        "refined", size=(7.0, 3.6))
def spectral_accuracy(fig, scheme):
    counts = np.array([8, 12, 16, 24, 32, 48, 64, 96])
    fd2, fd4, fourier, chebyshev = [], [], [], []
    for n in counts:
        n = int(n)
        # The same smooth function three ways.  The finite differences use a
        # grid including both endpoints; the Fourier method needs the periodic
        # grid that omits the repeated one.
        x = np.linspace(0, 2 * np.pi, n)
        f = np.exp(np.sin(x))
        exact = np.cos(x) * f
        for stencil, into in ((3, fd2), (5, fd4)):
            D = differentiation_matrix(x, order=1, stencil=stencil)
            into.append(float(np.max(np.abs(D @ f - exact))))
        xp = np.linspace(0, 2 * np.pi, n, endpoint=False)
        fp = np.exp(np.sin(xp))
        fourier.append(float(np.max(np.abs(
            fourier_derivative(fp, L=2 * np.pi) - np.cos(xp) * fp))))
        # The same function on Chebyshev points, where periodicity is not needed.
        xc, dc = chebyshev_derivative(lambda t: np.exp(np.sin(t)), n=n, a=0,
                                      b=2 * np.pi)
        chebyshev.append(float(np.max(np.abs(dc - np.cos(xc) * np.exp(np.sin(xc))))))

    ax = fig.add_subplot()
    series = [("3-point stencil, $O(h^2)$", fd2, scheme.series[1]),
              ("5-point stencil, $O(h^4)$", fd4, scheme.series[3]),
              ("Fourier (periodic)", fourier, scheme.series[0]),
              ("Chebyshev (any smooth $f$)", chebyshev, scheme.series[2])]
    for name, values, colour in series:
        ax.loglog(counts, np.maximum(values, 1e-17), color=colour, marker="o",
                  markersize=3.5, label=name)
    ax.axhline(np.finfo(float).eps, color=scheme.muted, linewidth=0.9,
               linestyle=(0, (4, 3)))
    finish(ax, scheme,
           title="Spectral methods gain digits per point, not per halving",
           xlabel="grid points", ylabel="max error in the derivative",
           grid="both", legend=True, legend_kw=dict(loc="lower left", fontsize=8))
    ax.set_ylim(1e-17, 1e1)


@figure("diff-differentiation-matrices", GUIDE,
        "The finite-difference and Chebyshev differentiation matrices side by "
        "side", size=(7.0, 3.4))
def differentiation_matrices(fig, scheme):
    n = 24
    uniform = np.linspace(-1, 1, n)
    fd = differentiation_matrix(uniform, order=1, stencil=3)
    cheb, _ = chebyshev_diff_matrix(n - 1)
    limit = float(np.max(np.abs(fd)))

    from matplotlib.colors import SymLogNorm

    panels = [(fd / limit, "central differences", int(np.sum(np.abs(fd) > 1e-12))),
              (cheb / float(np.max(np.abs(cheb))), "Chebyshev",
               int(np.sum(np.abs(cheb) > 1e-12)))]
    for k, (M, title, nonzeros) in enumerate(panels):
        ax = fig.add_subplot(1, 2, k + 1)
        title = f"{title} — {nonzeros} nonzeros of {M.size}"
        # A symmetric log scale, because the corner entries of the Chebyshev
        # matrix are three orders of magnitude larger than its interior ones.
        image = ax.imshow(M, cmap=diverging_cmap(scheme),
                          interpolation="nearest",
                          norm=SymLogNorm(linthresh=1e-3, vmin=-1, vmax=1))
        ax.set_title(title, color=scheme.ink, fontsize=9.5)
        ax.set_xlabel("column")
        if k == 0:
            ax.set_ylabel("row")
        ax.grid(False)
        ax.tick_params(labelsize=7)
        bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03,
                           ticks=(-1, 0, 1))
        bar.ax.set_yticklabels(["$-$max", "0", "$+$max"], fontsize=7,
                               color=scheme.muted)
        bar.outline.set_edgecolor(scheme.axis)
    fig.suptitle("Both matrices differentiate; only one of them is sparse",
                 x=0.02, ha="left", fontsize=10, fontweight="bold",
                 color=scheme.ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
