"""Catalogue of reproducible, measured documentation experiments.

Each registered Figure owns one page and a light/dark pair. Numeric results
come from Quadrivium; analytic references and diagnostic reductions are
explicit in experiments.py. The catalogue is also used by tests/test_docs.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

__all__ = ["FIGURES", "Figure", "figure", "load_all"]


@dataclass(frozen=True)
class Figure:
    name: str
    page: str
    summary: str
    size: tuple[float, float]
    draw: Callable
    parameters: dict = field(default_factory=dict)


FIGURES: list[Figure] = []


def figure(name, page, summary, size=(9.6, 4.7), *, parameters=None):
    """Register an experiment and its reproducibility details."""
    def decorate(func):
        if any(spec.name == name for spec in FIGURES):
            raise ValueError(f"duplicate figure name: {name}")
        FIGURES.append(Figure(name, page, summary, size, func, parameters or {}))
        return func
    return decorate


def load_all() -> list[Figure]:
    """Import the complete catalogue once; repeated calls are idempotent."""
    from . import experiments  # noqa: F401
    return FIGURES
