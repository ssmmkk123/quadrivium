"""quadrivium -- a comprehensive library of numerical methods.

A from-scratch implementation of the standard algorithms of scientific
computing, organised by problem area. Every method is written out explicitly
rather than delegated to a compiled library, so the algorithm itself is
readable, and each one is checked against analytic solutions, convergence
orders, or exact identities.

Subpackages
-----------
``core``          result records, exceptions, norms and numerical derivatives
``linalg``        factorizations, eigenvalues, Krylov solvers, least squares,
                  sparse storage, matrix functions and equations, randomized
``rootfind``      scalar equations, nonlinear systems, polynomial roots
``interpolate``   polynomial, spline, rational and multivariate interpolation
``approx``        orthogonal polynomials, least squares, Pade, minimax, Fourier
``diff``          finite differences, automatic differentiation, spectral
``integrate``     Newton-Cotes, Gauss, adaptive, Monte Carlo, oscillatory,
                  singular, sparse-grid and multidimensional rules
``ode``           explicit/implicit one-step, multistep, symplectic, exponential,
                  extrapolation, events, BVP, DAEs and delay equations
``pde``           parabolic, hyperbolic, elliptic, multigrid, FEM, FVM, spectral,
                  WENO/SSP high-resolution schemes, incompressible Navier-Stokes
``optimize``      line searches, quasi-Newton, trust region, global, constrained, LP
``transforms``    DFT/FFT family, signal processing, wavelets
``stochastic``    random generation, sampling, MCMC, statistics, SDE solvers
``special``       gamma, beta, error, Bessel, Airy, elliptic, hypergeometric,
                  Lambert W, Fresnel, spherical harmonics

Quick start
-----------
Results are rounded below to the accuracy each method is actually asked for --
``solve_ivp`` defaults to ``rtol=1e-8``, so its last digits are not meaningful.

>>> import numpy as np
>>> import quadrivium as qd
>>> round(qd.brent(lambda x: x**3 - 2*x - 5, 1, 3).root, 12)
2.094551481542
>>> round(float(qd.quad(lambda x: np.exp(-x*x), -np.inf, np.inf).value), 12)
1.772453850906
>>> round(float(qd.solve_ivp(lambda t, y: -2*y, (0, 1), [1.0]).y[-1, 0]), 9)
0.135335283
>>> round(float(qd.sqrtm([[4.0, 1.0], [2.0, 3.0]])[0, 0]), 12)
1.962116505791
>>> round(qd.lambert_w(np.e), 12)
1.0

Each subpackage can also be imported directly::

    from quadrivium.linalg import householder_qr
    from quadrivium.ode import dormand_prince
"""

from __future__ import annotations

__version__ = "1.1.0"

from . import (approx, core, diff, integrate, interpolate, linalg, ode,
               optimize, pde, rootfind, special, stochastic, transforms)
from .core import *  # noqa: F401,F403

# Curated top-level namespace: the routines most often reached for. Anything
# not re-exported here is available from its subpackage.
from .approx import (aaa, chebyshev_fit, fourier_series,
                     gauss_legendre_nodes, legendre, pade, polyfit,
                     remez)  # noqa: F401
from .diff import (Dual, HyperDual, Variable, central_difference, derivative,
                   gradient, hessian, jacobian, jacobian_fd,
                   richardson_derivative)  # noqa: F401
from .integrate import (adaptive_gauss_kronrod, filon, gauss_legendre,
                        monte_carlo, quad, romberg, simpson_rule,
                        sparse_grid_quadrature, tanh_sinh,
                        trapezoid_rule)  # noqa: F401
from .interpolate import (barycentric, bezier, cubic_spline, lagrange,
                          natural_cubic_spline, newton_divided_differences,
                          nurbs, pchip, rbf_interpolation)  # noqa: F401
from .linalg import (cholesky, conjugate_gradient, generalized_eigh, gmres,
                     householder_qr, jacobi_eigen, logm, lu_decomposition,
                     lyapunov, plu_decomposition, power_iteration, qr_algorithm,
                     randomized_svd, schur, solve, sqrtm, svd_jacobi,
                     sylvester)  # noqa: F401
from .ode import (adaptive_rk, bdf, dae_index1_bdf, dde_method_of_steps,
                  dormand_prince, euler, gragg_bulirsch_stoer, radau_iia, rk4,
                  rk_nystrom, shooting, solve_ivp, solve_ivp_events,
                  velocity_verlet)  # noqa: F401
from .optimize import (bfgs, brent_minimize, differential_evolution, lbfgs,
                       lbfgsb, linprog, minimize, nelder_mead, newton_cg,
                       newton_method, trust_region)  # noqa: F401
from .pde import (heat_crank_nicolson, multigrid_solve, navier_stokes_2d,
                  poisson_2d_direct, wave_explicit, weno_conservation_law)  # noqa: F401
from .rootfind import (anderson_acceleration, bisection, brent, newton,
                       newton_krylov, newton_system, polynomial_roots,
                       secant)  # noqa: F401
from .special import beta, erf, gamma, lambert_w  # noqa: F401
from .stochastic import (bootstrap, describe, euler_maruyama, gillespie_ssa,
                         metropolis_hastings, milstein)  # noqa: F401
from .transforms import (convolve, cwt, fft, ifft, power_spectrum, wavedec,
                         waverec)  # noqa: F401

__all__ = [
    # subpackages
    "core", "linalg", "rootfind", "interpolate", "approx", "diff", "integrate",
    "ode", "pde", "optimize", "transforms", "stochastic", "special",
    # linear algebra
    "solve", "lu_decomposition", "plu_decomposition", "cholesky",
    "householder_qr", "qr_algorithm", "jacobi_eigen", "power_iteration",
    "svd_jacobi", "conjugate_gradient", "gmres", "schur", "sqrtm", "logm",
    "sylvester", "lyapunov", "generalized_eigh", "randomized_svd",
    # root finding
    "bisection", "brent", "newton", "secant", "newton_system",
    "polynomial_roots", "anderson_acceleration", "newton_krylov",
    # interpolation
    "lagrange", "newton_divided_differences", "barycentric", "cubic_spline",
    "natural_cubic_spline", "pchip", "rbf_interpolation", "bezier", "nurbs",
    # approximation
    "polyfit", "chebyshev_fit", "pade", "remez", "fourier_series", "legendre",
    "gauss_legendre_nodes", "aaa",
    # differentiation
    "central_difference", "richardson_derivative", "derivative", "gradient",
    "jacobian", "hessian", "jacobian_fd", "Dual", "HyperDual", "Variable",
    # integration
    "quad", "trapezoid_rule", "simpson_rule", "romberg", "gauss_legendre",
    "adaptive_gauss_kronrod", "tanh_sinh", "monte_carlo", "filon",
    "sparse_grid_quadrature",
    # ODEs
    "solve_ivp", "euler", "rk4", "dormand_prince", "adaptive_rk", "radau_iia",
    "bdf", "velocity_verlet", "shooting", "gragg_bulirsch_stoer",
    "solve_ivp_events", "rk_nystrom", "dae_index1_bdf", "dde_method_of_steps",
    # PDEs
    "heat_crank_nicolson", "wave_explicit", "poisson_2d_direct",
    "multigrid_solve", "weno_conservation_law", "navier_stokes_2d",
    # optimization
    "minimize", "bfgs", "lbfgs", "newton_method", "trust_region",
    "nelder_mead", "brent_minimize", "differential_evolution", "linprog",
    "newton_cg", "lbfgsb",
    # transforms
    "fft", "ifft", "convolve", "power_spectrum", "wavedec", "waverec", "cwt",
    # stochastic and special
    "metropolis_hastings", "bootstrap", "describe", "gamma", "beta", "erf",
    "euler_maruyama", "milstein", "gillespie_ssa", "lambert_w",
    "__version__",
] + [n for n in core.__all__]
