"""Numerical differentiation: finite differences, automatic differentiation,
and spectral methods."""

from . import autodiff as _autodiff
from . import finite as _finite
from . import spectral as _spectral

__all__ = _finite.__all__ + _autodiff.__all__ + _spectral.__all__

from .autodiff import *  # noqa: F401,F403,E402
from .finite import *  # noqa: F401,F403,E402
from .spectral import *  # noqa: F401,F403,E402

from . import tensor as _tensor
from .tensor import *  # noqa: F401,F403
__all__ += _tensor.__all__
