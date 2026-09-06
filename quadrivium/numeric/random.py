"""Random number generation, mirroring ``numpy.random``.

The bit generator is PCG64 seeded through NumPy's SeedSequence, and the
samplers use NumPy's ziggurat tables, so a given seed reproduces NumPy's stream
exactly.
"""

from __future__ import annotations

from .. import _qnp as _c

default_rng = _c.default_rng
Generator = _c.Generator

__all__ = ["default_rng", "Generator", "PCG64"]


def PCG64(seed=None):
    """Provided so ``default_rng(PCG64(seed))`` keeps working."""
    return _c.default_rng(seed)
