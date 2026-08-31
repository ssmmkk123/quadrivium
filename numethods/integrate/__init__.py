"""Numerical integration: Newton-Cotes, Gauss, adaptive, Monte Carlo and
multidimensional quadrature."""

# Private aliases: several public functions share a name with their module
# (``newton_cotes``, ``romberg``, ``monte_carlo``), and the star imports below
# deliberately let the function win that name.
from . import adaptive as _adaptive
from . import gauss as _gauss
from . import monte_carlo as _monte_carlo
from . import multidim as _multidim
from . import newton_cotes as _newton_cotes
from . import romberg as _romberg

__all__ = (_newton_cotes.__all__ + _romberg.__all__ + _adaptive.__all__
           + _gauss.__all__ + _monte_carlo.__all__ + _multidim.__all__)

from .adaptive import *  # noqa: F401,F403,E402
from .gauss import *  # noqa: F401,F403,E402
from .monte_carlo import *  # noqa: F401,F403,E402
from .multidim import *  # noqa: F401,F403,E402
from .newton_cotes import *  # noqa: F401,F403,E402
from .romberg import *  # noqa: F401,F403,E402
