#!/usr/bin/env python3
"""The house style for the generated documentation figures.

Everything a figure needs that is not data lives here: the two colour
schemes, the Matplotlib settings, and the helpers that keep the panels
consistent.  ``tools/gen_figures.py`` draws each figure twice -- once per
scheme -- because a figure is a static image and cannot see the theme the
reader has chosen; MkDocs Material swaps the two on the ``#only-light`` and
``#only-dark`` URL fragments.

The palette is a validated categorical set: adjacent slots stay apart under
simulated protanopia and deuteranopia (worst OKLab dE 9.1 light, 8.4 dark,
against a target of 8), and the dark column is the same eight hues re-stepped
for a dark surface rather than a mechanical inversion.  Series identity is
never carried by colour alone -- every figure with more than one series has a
legend or a direct label -- because three of the light steps sit below a 3:1
contrast ratio against white.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class Scheme:
    """One colour scheme: the marks, and the chrome they are drawn on."""

    name: str
    series: tuple  # categorical slots, assigned in this fixed order
    ink: str  # primary text
    secondary: str  # subordinate text
    muted: str  # axis labels and ticks
    grid: str  # hairline gridlines
    axis: str  # baselines and spines
    surface: str  # what the page behind the figure looks like
    wash: str  # fills and shaded regions
    sequential: tuple = field(default=())  # one-hue ramp, ordered light -> dark



LIGHT = Scheme(
    name="light",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    ink="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    surface="#ffffff",
    wash="#f0efec",
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#256abf", "#184f95", "#0d366b"),
)

DARK = Scheme(
    name="dark",
    series=("#3987e5", "#d95926", "#199e70", "#c98500",
            "#d55181", "#008300", "#9085e9", "#e66767"),
    ink="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    # The Material slate background, hsl(232, 15%, 14%); the palette's dark
    # steps were checked for contrast against exactly this.
    surface="#1e2029",
    wash="#383835",
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5",
                "#6da7ec", "#9ec5f4", "#cde2fb"),
)

SCHEMES = (LIGHT, DARK)


def rc_params(scheme: Scheme) -> dict:
    """Matplotlib settings for one scheme.

    The figure and axes backgrounds stay transparent so the page's own
    background shows through -- a figure with a painted white panel would
    glow on the dark theme.
    """
    return {
        "figure.facecolor": "none",
        "figure.edgecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
        "savefig.transparent": True,
        "axes.facecolor": "none",
        "axes.edgecolor": scheme.axis,
        "axes.labelcolor": scheme.secondary,
        "axes.titlecolor": scheme.ink,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.labelsize": 9,
        "axes.labelpad": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=list(scheme.series)),
        "grid.color": scheme.grid,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": scheme.muted,
        "ytick.color": scheme.muted,
        "xtick.labelcolor": scheme.muted,
        "ytick.labelcolor": scheme.muted,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": scheme.secondary,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.6,
        "legend.borderaxespad": 0.4,
        "legend.columnspacing": 1.2,
        "font.family": "sans-serif",
        # DejaVu ships with Matplotlib, so the metrics are the same on every
        # machine that regenerates these files.
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "text.color": scheme.ink,
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 100,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "svg.fonttype": "path",
        # Dense marks -- point clouds, filled contours, spectrograms -- are
        # drawn with `rasterized=True`, which embeds them as an image at this
        # resolution instead of tens of thousands of vector paths.
        "savefig.dpi": 200,
        "path.simplify": True,
        "path.simplify_threshold": 1.0,
    }


def sequential_cmap(scheme):
    """A one-hue colour map running from the page outwards.

    The ramp is stored light-to-dark for the light scheme and dark-to-light
    for the dark one, so in both cases the low end of a magnitude scale sits
    near the page and the high end stands away from it.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("quadrivium-sequential",
                                             scheme.sequential)


def diverging_cmap(scheme):
    """Blue to red through the scheme's own neutral, for signed quantities.

    A diverging scale needs a midpoint that reads as *nothing*; a light grey
    would read as a value on the dark scheme, so the neutral comes from the
    scheme rather than from the colour map's own idea of white.
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Both arms end as far from the page as the sequential ramp does: darkest
    # on the light scheme, lightest on the dark one, so the two sides carry
    # equal weight and the middle reads as zero.
    far_blue = scheme.sequential[-1]
    far_red = "#8f1f1f" if scheme.name == "light" else "#f2a0a0"
    return LinearSegmentedColormap.from_list(
        "quadrivium-diverging",
        [far_blue, scheme.series[0], scheme.wash, scheme.series[7], far_red])


def ordinal(scheme, n: int):
    """`n` colours from the one-hue ramp, for a family with a natural order.

    A set of curves that differ by an index -- B-spline basis functions,
    successive partial sums, refinement levels -- is ordinal data, not
    categorical: it wants one hue stepped by lightness, not `n` unrelated hues.
    The ramp's ends are trimmed so that even the step nearest the page still
    reads against it.
    """
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("ordinal", scheme.sequential)
    # Trim the end nearest the page: on light that is the palest step, on dark
    # the darkest, and either one would sink into the background.
    lo, hi = (0.34, 1.0) if scheme.name == "light" else (0.17, 1.0)
    return [ramp(lo + (hi - lo) * (i / max(n - 1, 1))) for i in range(n)]


def annotate(ax, text, xy, xytext, scheme, *, color=None, arrow=True, **kw):
    """A short note pointing at something, in text ink rather than a series hue."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("ha", "center")
    arrowprops = (dict(arrowstyle="-", color=color or scheme.muted, linewidth=0.8,
                       shrinkA=2, shrinkB=3) if arrow else None)
    ax.annotate(text, xy=xy, xytext=xytext, color=color or scheme.secondary,
                arrowprops=arrowprops, **kw)


def label_line(ax, x, y, text, scheme, color, **kw):
    """Direct label riding a line, so identity never depends on colour alone."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("va", "center")
    kw.setdefault("fontweight", "bold")
    ax.text(x, y, text, color=color, **kw)


def finish(ax, scheme, *, title=None, xlabel=None, ylabel=None, grid="y",
           legend=False, legend_kw=None):
    """Apply the parts of the house style that depend on the axes' contents."""
    if title:
        ax.set_title(title, color=scheme.ink)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid in (None, "none"):
        ax.grid(False)
    else:
        ax.grid(visible=True, axis=grid)
    if legend:
        kw = dict(loc="best")
        kw.update(legend_kw or {})
        leg = ax.legend(**kw)
        for text in leg.get_texts():
            text.set_color(scheme.secondary)
    return ax
