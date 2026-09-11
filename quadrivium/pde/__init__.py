"""Partial differential equations: parabolic, hyperbolic and elliptic solvers,
plus multigrid, finite elements, finite volumes and spectral methods."""

from . import elliptic as _elliptic
from . import fem as _fem
from . import fvm as _fvm
from . import highres as _highres
from . import hyperbolic as _hyperbolic
from . import multigrid as _multigrid
from . import parabolic as _parabolic
from . import spectral as _spectral

__all__ = (_parabolic.__all__ + _hyperbolic.__all__ + _elliptic.__all__
           + _multigrid.__all__ + _fem.__all__ + _fvm.__all__ + _spectral.__all__ + _highres.__all__)

from .elliptic import *  # noqa: F401,F403,E402
from .fem import *  # noqa: F401,F403,E402
from .fvm import *  # noqa: F401,F403,E402
from .highres import *  # noqa: F401,F403,E402
from .hyperbolic import *  # noqa: F401,F403,E402
from .multigrid import *  # noqa: F401,F403,E402
from .parabolic import *  # noqa: F401,F403,E402
from .spectral import *  # noqa: F401,F403,E402

from . import adaptive_fem as _adaptive_fem
from .adaptive_fem import *  # noqa: F401,F403
__all__ += _adaptive_fem.__all__
