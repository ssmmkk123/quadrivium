"""The catalogue of documentation figures.

Every figure is a function that draws one Matplotlib figure from data the
library itself computes -- a convergence history, an iterate path, a spectrum
-- so a figure cannot drift from the code any more than the API reference can.
Registration is by decorator; ``tools/gen_figures.py`` renders whatever is
registered here, once per colour scheme.

Each module corresponds to one documentation page, and the ``page`` recorded
with every figure is checked against the page that actually embeds it, so a
figure nobody shows and a reference to a figure nobody draws are both test
failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = ["FIGURES", "Figure", "figure", "load_all"]


@dataclass(frozen=True)
class Figure:
    """One registered figure."""

    name: str  # file stem under docs/assets/figures
    page: str  # documentation page that embeds it, relative to docs/
    summary: str  # what the figure shows, for the manifest and the alt text
    size: tuple  # figure size in inches
    draw: Callable  # draw(fig, scheme) -> None


FIGURES: list = []


def figure(name: str, page: str, summary: str, size: tuple = (7.0, 3.4)):
    """Register a drawing function as a documentation figure."""

    def decorate(func):
        if any(f.name == name for f in FIGURES):
            raise ValueError(f"duplicate figure name: {name}")
        FIGURES.append(Figure(name=name, page=page, summary=summary,
                              size=size, draw=func))
        return func

    return decorate


def load_all() -> list:
    """Import every figure module, so every figure is registered."""
    from . import (  # noqa: F401
        overview, core, linalg, rootfind, interpolate, approx, diff,
        integrate, ode, pde, optimize, transforms, stochastic, special,
    )
    return FIGURES
