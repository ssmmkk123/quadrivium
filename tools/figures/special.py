"""Figures for the special functions guide."""

from __future__ import annotations

from quadrivium import numeric as np

from quadrivium.special import (airy_ai, airy_bi, bessel_jn, bessel_yn, dawson,
                                erf, erfc, erfcx, gamma, lambert_w, log_gamma,
                                zeta)

from . import figure
from figstyle import annotate, finish, ordinal

GUIDE = "guides/special.md"


def sampled(func, x):
    return np.array([float(func(float(v))) for v in x])


@figure("special-gamma", GUIDE,
        "The gamma function through its poles, and the logarithm that stays "
        "finite", size=(7.0, 3.6))
def gamma_figure(fig, scheme):
    ax1 = fig.add_subplot(1, 2, 1)
    # Between each pair of non-positive integers, where gamma is finite.
    for k in range(-4, 4):
        piece = np.linspace(k + 0.02, k + 0.98, 300)
        values = sampled(gamma, piece)
        ax1.plot(piece, values, color=scheme.series[0], linewidth=1.8)
    for pole in range(-4, 1):
        ax1.axvline(pole, color=scheme.muted, linewidth=0.8,
                    linestyle=(0, (2, 3)))
    integers = np.arange(1, 6)
    ax1.scatter(integers, sampled(gamma, integers), s=30, zorder=3,
                color=scheme.series[1], edgecolors=scheme.surface,
                linewidths=1.0, label=r"$\Gamma(n) = (n-1)!$")
    ax1.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax1, scheme, title=r"$\Gamma(x)$ on the real line", xlabel="x",
           ylabel=r"$\Gamma(x)$", grid="both", legend=True,
           legend_kw=dict(loc="lower right", fontsize=8))
    ax1.set_ylim(-6, 6)
    ax1.set_xlim(-4.2, 4.6)

    ax2 = fig.add_subplot(1, 2, 2)
    x = np.linspace(1, 200, 400)
    ax2.plot(x, sampled(log_gamma, x), color=scheme.series[0],
             label=r"$\log\Gamma(x)$ — computed directly")
    overflow = np.log(np.finfo(float).max)
    ax2.axhline(overflow, color=scheme.muted, linewidth=1.0,
                linestyle=(0, (4, 3)))
    crossing = x[int(np.argmax(sampled(log_gamma, x) > overflow))]
    ax2.axvline(crossing, color=scheme.muted, linewidth=1.0,
                linestyle=(0, (4, 3)))
    finish(ax2, scheme, title="Why log_gamma exists", xlabel="x",
           ylabel=r"$\log\Gamma(x)$", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=8))
    annotate(ax2, f"$\\Gamma(x)$ overflows a double\npast x = {crossing:.0f}",
             (crossing, overflow), (crossing + 8, overflow * 0.45), scheme,
             ha="left")
    fig.tight_layout()


@figure("special-bessel", GUIDE,
        "Bessel functions of the first and second kind, and the Wronskian "
        "identity that ties them together", size=(7.0, 3.6))
def bessel_figure(fig, scheme):
    x = np.linspace(0.05, 20, 700)
    ax1 = fig.add_subplot(1, 2, 1)
    colours = ordinal(scheme, 4)
    for n in range(4):
        ax1.plot(x, np.array([float(bessel_jn(n, float(v))) for v in x]),
                 color=colours[n], label=f"$J_{n}$")
    ax1.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax1, scheme, title="First kind", xlabel="x", ylabel=r"$J_n(x)$",
           grid="both", legend=True, legend_kw=dict(loc="upper right", ncol=4,
                                                    fontsize=8))
    ax1.set_ylim(-0.65, 1.35)

    ax2 = fig.add_subplot(1, 2, 2)
    for n in range(3):
        ax2.plot(x, np.array([float(bessel_yn(n, float(v))) for v in x]),
                 color=colours[n], label=f"$Y_{n}$")
    ax2.axhline(0, color=scheme.axis, linewidth=0.8)
    residual = np.array([abs(float(bessel_jn(3, float(v)) * bessel_yn(2, float(v))
                                   - bessel_jn(2, float(v)) * bessel_yn(3, float(v)))
                             - 2 / (np.pi * float(v))) for v in x])
    finish(ax2, scheme,
           title=f"Second kind — Wronskian holds to {float(np.max(residual)):.0e}",
           xlabel="x", ylabel=r"$Y_n(x)$", grid="both", legend=True,
           legend_kw=dict(loc="lower right", ncol=3, fontsize=8))
    ax2.set_ylim(-1.6, 0.85)
    fig.tight_layout()


@figure("special-airy", GUIDE,
        "Airy functions: the transition between oscillation and exponential "
        "growth", size=(7.0, 3.4))
def airy_figure(fig, scheme):
    x = np.linspace(-12, 3, 800)
    ai = sampled(airy_ai, x)
    bi = sampled(airy_bi, x)

    ax = fig.add_subplot()
    ax.plot(x, ai, color=scheme.series[0], label=r"$\mathrm{Ai}(x)$")
    ax.plot(x, bi, color=scheme.series[1], label=r"$\mathrm{Bi}(x)$")
    ax.axvline(0, color=scheme.axis, linewidth=0.8)
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme,
           title=r"Both solve $y'' = xy$; the sign of $x$ decides the character",
           xlabel="x", ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="upper left"))
    ax.set_ylim(-1.0, 1.6)
    annotate(ax, "oscillatory for x < 0", (-8.0, 0.35), (-8.0, 1.15), scheme,
             ha="center")
    annotate(ax, "Ai decays, Bi grows", (1.6, 0.9), (0.2, 1.35), scheme,
             ha="center")


@figure("special-error-family", GUIDE,
        "The error function family, and the scaled form that survives the "
        "tail", size=(7.0, 3.4))
def error_family(fig, scheme):
    ax1 = fig.add_subplot(1, 2, 1)
    x = np.linspace(-3, 3, 500)
    ax1.plot(x, sampled(erf, x), color=scheme.series[0], label=r"$\mathrm{erf}$")
    ax1.plot(x, sampled(erfc, x), color=scheme.series[1],
             label=r"$\mathrm{erfc} = 1 - \mathrm{erf}$")
    ax1.plot(x, sampled(dawson, x), color=scheme.series[2],
             label="Dawson $F$")
    ax1.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax1, scheme, title="On a scale where all three are visible",
           xlabel="x", ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=8))

    ax2 = fig.add_subplot(1, 2, 2)
    tail = np.linspace(0, 30, 400)
    ax2.semilogy(tail, np.maximum(sampled(erfc, tail), 1e-320),
                 color=scheme.series[1], label=r"$\mathrm{erfc}(x)$")
    ax2.semilogy(tail, sampled(erfcx, tail), color=scheme.series[0],
                 label=r"$\mathrm{erfcx}(x) = e^{x^2}\mathrm{erfc}(x)$")
    underflow = tail[int(np.argmax(sampled(erfc, tail) == 0.0))]
    ax2.axvline(underflow, color=scheme.muted, linewidth=1.0,
                linestyle=(0, (4, 3)))
    finish(ax2, scheme, title="In the tail", xlabel="x", ylabel="value",
           grid="both", legend=True, legend_kw=dict(loc="lower left",
                                                    fontsize=8))
    ax2.set_ylim(1e-320, 1e2)
    annotate(ax2, f"erfc underflows to zero\nat x = {underflow:.0f}",
             (underflow, 1e-160), (underflow - 1.5, 1e-100), scheme, ha="right")
    fig.tight_layout()


@figure("special-zeta-lambert", GUIDE,
        "The Riemann zeta function on the real line and both real branches of "
        "Lambert W", size=(7.0, 3.4))
def zeta_lambert(fig, scheme):
    ax1 = fig.add_subplot(1, 2, 1)
    for piece in (np.linspace(-6, 0.98, 400), np.linspace(1.02, 8, 400)):
        ax1.plot(piece, sampled(zeta, piece), color=scheme.series[0],
                 linewidth=1.8)
    ax1.axvline(1.0, color=scheme.muted, linewidth=1.0, linestyle=(0, (2, 3)))
    ax1.axhline(0, color=scheme.axis, linewidth=0.8)
    known = {2.0: np.pi ** 2 / 6, -1.0: -1 / 12, 4.0: np.pi ** 4 / 90}
    ax1.scatter(list(known), [known[k] for k in known], s=32, zorder=3,
                color=scheme.series[1], edgecolors=scheme.surface,
                linewidths=1.0, label=r"$\zeta(2), \zeta(4), \zeta(-1)$ exact")
    finish(ax1, scheme, title=r"$\zeta(s)$, with its pole at $s = 1$",
           xlabel="s", ylabel=r"$\zeta(s)$", grid="both", legend=True,
           legend_kw=dict(loc="lower right", fontsize=8))
    ax1.set_ylim(-2.5, 4)

    ax2 = fig.add_subplot(1, 2, 2)
    upper = np.linspace(-1 / np.e + 1e-9, 4, 400)
    lower = np.linspace(-1 / np.e + 1e-9, -1e-3, 400)
    ax2.plot(upper, np.array([float(lambert_w(float(v))) for v in upper]),
             color=scheme.series[0], label=r"principal branch $W_0$")
    ax2.plot(lower, np.array([float(lambert_w(float(v), branch=-1))
                              for v in lower]), color=scheme.series[1],
             label=r"branch $W_{-1}$")
    ax2.axvline(-1 / np.e, color=scheme.muted, linewidth=1.0,
                linestyle=(0, (4, 3)))
    ax2.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax2, scheme, title=r"$W(x)e^{W(x)} = x$", xlabel="x", ylabel="W(x)",
           grid="both", legend=True, legend_kw=dict(loc="lower right",
                                                    fontsize=8))
    ax2.set_ylim(-6, 2)
    annotate(ax2, r"the branch point at $-1/e$", (-1 / np.e, -1.0),
             (0.6, -2.2), scheme, ha="left")
    fig.tight_layout()
