"""Function approximation: orthogonal polynomials, least squares fitting,
Pade approximants, minimax and Fourier series."""

from . import orthopoly as _orthopoly

__all__ = list(_orthopoly.__all__)

from .orthopoly import *  # noqa: F401,F403,E402

try:  # the remaining approximation modules are optional at import time
    from . import fitting as _fitting
    from .fitting import *  # noqa: F401,F403
    __all__ = __all__ + _fitting.__all__
except ImportError:  # pragma: no cover
    pass

from .adaptive import ChebyshevApproximation, chebfun
__all__ += ["ChebyshevApproximation", "chebfun"]
