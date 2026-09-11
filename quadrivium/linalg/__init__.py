"""Linear algebra: direct factorizations, eigenvalue algorithms, iterative
solvers, least squares, sparse storage, matrix functions and matrix equations."""

from . import direct as _direct
from . import eigen as _eigen
from . import iterative as _iterative
from . import lstsq as _lstsq
from . import matfun as _matfun
from . import sparse as _sparse
from . import operators as _operators
from . import factors as _factors

__all__ = (_direct.__all__ + _eigen.__all__ + _iterative.__all__
           + _lstsq.__all__ + _sparse.__all__ + _matfun.__all__
           + _operators.__all__ + _factors.__all__)

from .direct import *  # noqa: F401,F403,E402
from .eigen import *  # noqa: F401,F403,E402
from .iterative import *  # noqa: F401,F403,E402
from .lstsq import *  # noqa: F401,F403,E402
from .matfun import *  # noqa: F401,F403,E402
from .sparse import *  # noqa: F401,F403,E402

from .operators import *  # noqa: F401,F403,E402
from .factors import *  # noqa: F401,F403,E402
