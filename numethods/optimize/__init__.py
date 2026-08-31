"""Optimization: line searches, scalar minimization, gradient and quasi-Newton
methods, trust regions, derivative-free search, global optimization,
constrained and proximal algorithms, and linear programming."""

from . import constrained as _constrained
from . import derivative_free as _derivative_free
from . import global_opt as _global_opt
from . import gradient as _gradient
from . import linesearch as _linesearch
from . import linprog as _linprog
from . import proximal as _proximal
from . import quasinewton as _quasinewton
from . import scalar as _scalar
from . import trustregion as _trustregion

__all__ = (_scalar.__all__ + _linesearch.__all__ + _gradient.__all__
           + _quasinewton.__all__ + _trustregion.__all__
           + _derivative_free.__all__ + _global_opt.__all__
           + _constrained.__all__ + _proximal.__all__ + _linprog.__all__
           + ["minimize"])

from .constrained import *  # noqa: F401,F403,E402
from .derivative_free import *  # noqa: F401,F403,E402
from .global_opt import *  # noqa: F401,F403,E402
from .gradient import *  # noqa: F401,F403,E402
from .linesearch import *  # noqa: F401,F403,E402
from .linprog import *  # noqa: F401,F403,E402
from .proximal import *  # noqa: F401,F403,E402
from .quasinewton import *  # noqa: F401,F403,E402
from .scalar import *  # noqa: F401,F403,E402
from .trustregion import *  # noqa: F401,F403,E402


def minimize(f, x0, method: str = "bfgs", grad_f=None, **kwargs):
    """Minimize a scalar objective with the named method.

    Gradient-based: ``bfgs``, ``lbfgs``, ``dfp``, ``sr1``, ``newton``,
    ``modified_newton``, ``cg_fr``, ``cg_pr``, ``cg_hs``, ``gradient_descent``,
    ``momentum``, ``nesterov``, ``adam``, ``adagrad``, ``rmsprop``,
    ``barzilai_borwein``, ``trust_region``.
    Derivative-free: ``nelder_mead``, ``powell``, ``hooke_jeeves``,
    ``compass_search``, ``coordinate_descent``.
    Global (these take ``bounds`` rather than ``x0``): ``differential_evolution``,
    ``particle_swarm``, ``simulated_annealing``, ``genetic_algorithm``,
    ``basin_hopping``, ``cma_es``, ``dual_annealing_lite``.
    """
    gradient_based = {
        "bfgs": bfgs, "lbfgs": lbfgs, "dfp": dfp, "sr1": sr1,
        "newton": newton_method, "modified_newton": modified_newton,
        "cg_fr": conjugate_gradient_fr, "cg_pr": conjugate_gradient_pr,
        "cg_hs": conjugate_gradient_hs, "gradient_descent": gradient_descent,
        "momentum": momentum, "nesterov": nesterov, "adam": adam,
        "adagrad": adagrad, "rmsprop": rmsprop,
        "barzilai_borwein": barzilai_borwein,
    }
    if method in gradient_based:
        return gradient_based[method](f, x0, grad_f, **kwargs)
    if method == "trust_region":
        return trust_region(f, x0, grad_f, **kwargs)
    derivative_free = {
        "nelder_mead": nelder_mead, "powell": powell,
        "hooke_jeeves": hooke_jeeves, "compass_search": compass_search,
        "coordinate_descent": coordinate_descent,
    }
    if method in derivative_free:
        return derivative_free[method](f, x0, **kwargs)
    global_methods = {
        "differential_evolution": differential_evolution,
        "particle_swarm": particle_swarm, "genetic_algorithm": genetic_algorithm,
        "random_search": random_search,
    }
    if method in global_methods:
        return global_methods[method](f, x0, **kwargs)   # x0 is `bounds` here
    if method in ("simulated_annealing", "basin_hopping", "cma_es"):
        return {"simulated_annealing": simulated_annealing,
                "basin_hopping": basin_hopping, "cma_es": cma_es}[method](
                    f, x0, **kwargs)
    raise ValueError(f"unknown method {method!r}")
