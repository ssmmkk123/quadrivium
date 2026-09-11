"""Root finding: scalar equations, nonlinear systems, and polynomials."""

from . import polynomial as _polynomial
from . import scalar as _scalar
from . import systems as _systems

__all__ = _scalar.__all__ + _systems.__all__ + _polynomial.__all__

from .polynomial import *  # noqa: F401,F403,E402
from .scalar import *  # noqa: F401,F403,E402
from .systems import *  # noqa: F401,F403,E402

from .continuation import ContinuationResult, pseudo_arclength
from .systems import continuation  # preserve the historical callable, not the module
__all__ += ["ContinuationResult", "pseudo_arclength"]
