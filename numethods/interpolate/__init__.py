"""Interpolation: polynomial, spline, rational, and multivariate."""

from . import multivariate as _multivariate
from . import polynomial as _polynomial
from . import rational as _rational
from . import spline as _spline

__all__ = (_polynomial.__all__ + _spline.__all__ + _rational.__all__
           + _multivariate.__all__)

from .multivariate import *  # noqa: F401,F403,E402
from .polynomial import *  # noqa: F401,F403,E402
from .rational import *  # noqa: F401,F403,E402
from .spline import *  # noqa: F401,F403,E402
