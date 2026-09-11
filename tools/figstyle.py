"""Shared, accessible styling for the documentation's measured experiments.

Series use both colour and line/marker shape. SVG text remains searchable.
The two palettes are chosen for light and Material's slate backgrounds; no
unverified claim about simulated colour vision is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import textwrap

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Scheme:
    name: str
    series: tuple[str, ...]
    ink: str
    secondary: str
    grid: str
    axis: str
    surface: str


LIGHT = Scheme("light", ("#1264a3", "#be5214", "#087d63", "#8554a0"),
               "#182735", "#465b6b", "#e0e6eb", "#b4c1cc", "#ffffff")
DARK = Scheme("dark", ("#72bdff", "#ffad71", "#6bd8b3", "#d5adf5"),
              "#edf5fc", "#bacbdb", "#39424f", "#718092", "#1e2029")
SCHEMES = (LIGHT, DARK)
LINE_STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")


def rc_params(scheme: Scheme) -> dict:
    return {
        "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none", "savefig.transparent": True,
        "axes.edgecolor": scheme.axis, "axes.labelcolor": scheme.secondary,
        "axes.titlecolor": scheme.ink, "text.color": scheme.ink,
        "xtick.color": scheme.secondary, "ytick.color": scheme.secondary,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.grid": True,
        "grid.color": scheme.grid, "grid.linewidth": 0.65,
        "axes.prop_cycle": plt.cycler(color=list(scheme.series)),
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "DejaVu Sans", "font.size": 9,
        "mathtext.fontset": "dejavusans", "legend.fontsize": 8,
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.facecolor": scheme.surface, "legend.edgecolor": scheme.grid,
        "legend.labelcolor": scheme.secondary, "legend.handlelength": 2.2,
        "lines.linewidth": 1.8, "lines.markersize": 4,
        "figure.dpi": 110, "savefig.dpi": 150,
        "svg.fonttype": "none", "svg.hashsalt": "quadrivium-experiments-v2",
        "path.simplify": False,
    }


def panels(fig, title: str, subtitle: str):
    """Two spacious panels with room reserved for an explanatory header."""
    fig.suptitle(title, x=0.075, y=0.98, ha="left", fontsize=14,
                 fontweight="bold")
    fig.text(0.075, 0.90, subtitle, ha="left", fontsize=9)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17,
                        top=0.75, wspace=0.36)
    return axes


def line(ax, x, y, scheme, index=0, *, label=None, markers=False, **kwargs):
    """Keep curve identity available without colour."""
    return ax.plot(np.asarray(x), np.asarray(y), color=scheme.series[index % 4],
                   linestyle=LINE_STYLES[index % 4],
                   marker=MARKERS[index % 4] if markers else None,
                   label=label, **kwargs)


def truth(ax, x, y, scheme, label="Analytic reference"):
    return ax.plot(np.asarray(x), np.asarray(y), color=scheme.ink, linestyle=(0, (2, 3)),
                   linewidth=1.2, label=label, zorder=2)


def finish(ax, title, xlabel, ylabel, *, legend=True, logx=False, logy=False):
    ax.set(title=textwrap.fill(title, width=43), xlabel=xlabel, ylabel=ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="major")
    if legend:
        ax.legend(loc="best")
    ax.margins(x=0.04)
