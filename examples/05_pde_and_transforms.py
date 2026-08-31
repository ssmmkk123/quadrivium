"""PDEs, transforms, and stochastic methods."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numethods.core.exceptions import DomainError
from numethods.pde import *
from numethods.stochastic import *
from numethods.transforms import *


def banner(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


banner("1. The heat equation: conditional versus unconditional stability")
u0 = lambda x: np.sin(np.pi * x)
exact = lambda x, t: np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x)
T = 0.1
print("  u_t = u_xx on [0,1], u(x,0) = sin(pi x), zero at the ends\n")
print(f"  {'scheme':<26}{'steps':>8}{'r = dt/dx^2':>14}{'error':>12}")
for name, nt in (("FTCS (explicit)", 4000), ("BTCS (implicit)", 400),
                 ("Crank-Nicolson", 400)):
    run = {"FTCS (explicit)": heat_ftcs, "BTCS (implicit)": heat_btcs,
           "Crank-Nicolson": heat_crank_nicolson}[name]
    s = run(u0, 1.0, (0, 1), (0, T), nx=40, nt=nt)
    r = stability_ratio(1.0, T / nt, 1 / 40)
    print(f"  {name:<26}{nt:>8}{r:>14.4f}"
          f"{np.max(np.abs(s.final - exact(s.x, T))):>12.2e}")

try:
    heat_ftcs(u0, 1.0, (0, 1), (0, T), nx=40, nt=100)
except DomainError as e:
    print(f"\n  FTCS with only 100 steps is refused:\n    {e}")
blown = heat_ftcs(u0, 1.0, (0, 1), (0, T), nx=40, nt=100, check_stability=False)
print(f"  Forcing it through anyway gives max|u| = {np.max(np.abs(blown.final)):.2e}")

banner("2. Godunov's theorem in action")
sq = lambda x: 1.0 if 0.3 <= x <= 0.6 else 0.0
print("  A square pulse advected once around a periodic domain.")
print("  Exact solution stays in [0, 1] with total variation 2.\n")
print(f"  {'scheme':<26}{'order':>7}{'min':>10}{'max':>10}{'TV':>8}")
for name, run, order in (("upwind", advection_upwind, 1),
                         ("lax_friedrichs", lax_friedrichs, 1),
                         ("lax_wendroff", lax_wendroff, 2),
                         ("beam_warming", beam_warming, 2)):
    s = run(sq, 1.0, (0, 1), (0, 1.0), 200, 400)
    tv = np.sum(np.abs(np.diff(np.concatenate([s.final, [s.final[0]]]))))
    print(f"  {name:<26}{order:>7}{s.final.min():>10.4f}{s.final.max():>10.4f}{tv:>8.3f}")
for lim in ("minmod", "van_leer", "superbee", "mc"):
    s = tvd_scheme(sq, 1.0, (0, 1), (0, 1.0), 200, 400, lim)
    tv = np.sum(np.abs(np.diff(np.concatenate([s.final, [s.final[0]]]))))
    print(f"  {'tvd/' + lim:<26}{2:>7}{s.final.min():>10.4f}{s.final.max():>10.4f}"
          f"{tv:>8.3f}")
print("\n  Second-order linear schemes must oscillate at a discontinuity")
print("  (Godunov, 1959). TVD schemes escape by being nonlinear.")

banner("3. Shock capturing for Burgers' equation")
riemann = lambda x: 1.0 if x < 0.5 else 0.0
print("  u_t + (u^2/2)_x = 0 with a step at x = 0.5.")
print("  Rankine-Hugoniot gives shock speed 1/2, so at t = 0.8 the front")
print("  should sit at x = 0.9.\n")
for name, run in (("Godunov (exact Riemann)", godunov_burgers),
                  ("Lax-Friedrichs", lax_friedrichs_burgers)):
    s = run(riemann, (0, 2), (0, 0.8), 400, 1600)
    pos = s.x[np.argmin(np.diff(s.final))]
    print(f"  {name:<28} shock at x = {pos:.4f}")
s = godunov_burgers(lambda x: 0.5 + 0.5 * np.sin(2 * np.pi * x), (0, 1), (0, 0.5),
                    400, 1000)
print(f"\n  Mass conservation from smooth data through shock formation:")
print(f"    initial mean {s.u[0].mean():.14f}")
print(f"    final   mean {s.final.mean():.14f}")

banner("4. Elliptic problems: discretization order and solver cost")
uex = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
f2 = lambda x, y: -2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
print("  Poisson on the unit square with a manufactured solution\n")
print(f"  {'n':>5}{'5-point':>14}{'ratio':>8}{'9-point':>14}{'ratio':>8}")
prev5 = prev9 = None
for n in (8, 16, 32, 64):
    g = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    e5 = np.max(np.abs(poisson_2d_direct(f2, (0, 1), (0, 1), n, n, 0.0, 5).u - uex(X, Y)))
    e9 = np.max(np.abs(poisson_2d_direct(f2, (0, 1), (0, 1), n, n, 0.0, 9).u - uex(X, Y)))
    r5 = "" if prev5 is None else f"{prev5 / e5:.1f}"
    r9 = "" if prev9 is None else f"{prev9 / e9:.1f}"
    print(f"  {n:>5}{e5:>14.2e}{r5:>8}{e9:>14.2e}{r9:>8}")
    prev5, prev9 = e5, e9
print("\n  The 9-point Mehrstellen stencil is fourth order (once the h^2/12")
print("  correction is applied to the right-hand side).")

print("\n  Solver cost on the same problem (n = 32):")
n = 32
for method in ("jacobi", "gauss_seidel", "sor", "cg"):
    s = poisson_2d_iterative(f2, (0, 1), (0, 1), n, n, 0.0, method=method,
                             tol=1e-10, max_iter=40000)
    print(f"    {method:<16}{s.iterations:>8} iterations")

banner("5. Multigrid: convergence independent of the mesh")
print(f"  {'n':>6}{'unknowns':>12}{'V-cycles':>11}{'final residual':>17}{'error':>12}")
for n in (32, 64, 128):
    s = multigrid_solve(f2, (0, 1), (0, 1), n, tol=1e-10, max_cycles=50)
    X, Y = np.meshgrid(s.x, s.y, indexing="ij")
    print(f"  {n:>6}{(n - 1) ** 2:>12}{s.iterations:>11}{s.residuals[-1]:>17.2e}"
          f"{np.max(np.abs(s.u - uex(X, Y))):>12.2e}")
s = multigrid_solve(f2, (0, 1), (0, 1), 64, tol=1e-12, max_cycles=20)
rates = [s.residuals[i + 1] / s.residuals[i] for i in range(4)]
print(f"\n  Residual reduction per cycle: {' '.join(f'{r:.3f}' for r in rates)}")
print("  The cycle count does not grow with the problem size: that is what")
print("  makes multigrid an O(N) method.")

banner("6. Finite elements and finite volumes")
fx = lambda x: np.pi ** 2 * np.sin(np.pi * x)
print("  1-D FEM for -u'' = f. The P1 Galerkin solution is nodally exact,")
print("  so only the load quadrature limits accuracy:\n")
s = fem_1d_linear(lambda x: 2.0, (0, 1), (0.0, 0.0), 10)
print(f"    -u'' = 2 (exact solution x(1-x)): nodal error "
      f"{np.max(np.abs(s.u - s.x * (1 - s.x))):.2e}")
errs = [np.max(np.abs(fem_1d_linear(fx, (0, 1), (0.0, 0.0), n).u
                      - np.sin(np.pi * np.linspace(0, 1, n + 1)))) for n in (20, 40, 80)]
print(f"    -u'' = pi^2 sin(pi x): errors {errs[0]:.2e} -> {errs[1]:.2e} -> "
      f"{errs[2]:.2e}")

print("\n  2-D FEM on a triangular mesh:")
f2d = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
for n in (8, 16, 32):
    s = fem_2d_triangular(f2d, n=n)
    P = np.column_stack([s.grids[0], s.grids[1]])
    print(f"    {2 * n * n:>5} triangles, {len(P):>5} nodes: error "
          f"{np.max(np.abs(s.u - uex(P[:, 0], P[:, 1]))):.2e}")

print("\n  Finite volume conserves mass to round-off by construction:")
flux = lambda u: 1.0 * u
speed = lambda u: np.ones_like(u)
for nf in ("rusanov", "hll"):
    s = fvm_1d_conservation(sq, flux, speed, (0, 1), (0, 1.0), 200, 400,
                            numerical_flux=nf)
    print(f"    {nf:<10} mass change {abs(s.u[0].mean() - s.final.mean()):.2e}")

banner("7. Spectral methods")
u0s = lambda x: np.sin(x) + 0.5 * np.sin(3 * x)
s = fourier_heat(u0s, 1.0, 2 * np.pi, 64, (0, 0.5), 50)
ex = np.exp(-0.5) * np.sin(s.x) + 0.5 * np.exp(-4.5) * np.sin(3 * s.x)
print(f"  Fourier heat equation (exact in spectral space): error "
      f"{np.max(np.abs(s.final - ex)):.2e}")
s = fourier_advection(u0s, 1.0, 2 * np.pi, 64, (0, 10.0), 50)
ex = np.sin(s.x - 10) + 0.5 * np.sin(3 * (s.x - 10))
print(f"  Fourier advection after t = 10:                  error "
      f"{np.max(np.abs(s.final - ex)):.2e}")
print("\n  Chebyshev collocation for u'' = -pi^2 sin(pi x) converges exponentially:")
for n in (8, 16, 32):
    s = chebyshev_poisson_1d(lambda x: -np.pi ** 2 * np.sin(np.pi * x), (-1, 1),
                             (0.0, 0.0), n)
    print(f"    n = {n:>3}: error {np.max(np.abs(s.u - np.sin(np.pi * s.x))):.2e}")

banner("8. Transforms")
rng = np.random.default_rng(0)
print(f"  {'length':>8}  {'kind':<12}{'error vs reference':>22}")
for n in (64, 97, 128):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    kind = "power of 2" if n & (n - 1) == 0 else "prime"
    print(f"  {n:>8}  {kind:<12}{np.max(np.abs(fft(x) - np.fft.fft(x))):>20.2e}")
print("\n  Bluestein's chirp-z algorithm keeps prime lengths at O(n log n).")

fs, dt, n = 1000.0, 1e-3, 4096
t = np.arange(n) * dt
sig = (2 * np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
       + 0.1 * rng.standard_normal(n))
f, p = power_spectrum(sig, dt)
print(f"\n  Spectral analysis of a two-tone signal in noise:")
for target in (50, 120):
    i = int(np.argmin(np.abs(f - target)))
    print(f"    {target:>3} Hz component stands {p[max(i - 2, 0):i + 3].max() / np.median(p):>8.0f}x above the noise floor")
print(f"    Parseval check: integral of PSD = {np.trapezoid(p, f):.6f}, "
      f"variance = {np.var(sig):.6f}")

banner("9. Random number generation and MCMC")
print("  Testing generators for lattice structure in consecutive triples:\n")
for name, make in (("RANDU (a=65539)", lambda: LCG(1, a=65539, c=0, m=2 ** 31)),
                   ("Park-Miller", lambda: ParkMiller(1)),
                   ("xorshift", lambda: XorShift()),
                   ("MT19937", lambda: MersenneTwister(1))):
    r = spectral_test(make(), 1500, dim=3)
    verdict = (f"LATTICE via {tuple(int(c) for c in r['coefficients'])}"
               if r["lattice_detected"] else "no small dual vector")
    print(f"    {name:<20} {verdict}")
print("\n  RANDU's triples lie on 15 planes, which is why it was withdrawn.")

lt = lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2))
gl = lambda x: -np.asarray(x, dtype=float)
print(f"\n  Sampling a standard normal:")
print(f"  {'sampler':<22}{'mean':>9}{'sd':>8}{'ESS/n':>9}{'tau':>8}")
for name, run in (("random walk MH", lambda: random_walk_metropolis(lt, [0.0], step=2.4, n=20000, burn=1000, rng=0)),
                  ("Hamiltonian MC", lambda: hamiltonian_mc(lt, gl, [0.0], step=0.15, n_leapfrog=20, n=10000, burn=500, rng=0)),
                  ("slice sampling", lambda: slice_sampler(lt, 0.0, w=2.0, n=10000, burn=500, rng=0))):
    ch = run()
    x = np.asarray(ch)[:, 0] if np.asarray(ch).ndim > 1 else np.asarray(ch)
    print(f"  {name:<22}{x.mean():>9.4f}{x.std():>8.4f}"
          f"{effective_sample_size(x) / len(x):>9.3f}{autocorrelation_time(x):>8.2f}")

bimodal = lambda x: float(np.logaddexp(-0.5 * (np.asarray(x)[0] - 5) ** 2,
                                       -0.5 * (np.asarray(x)[0] + 5) ** 2))
pt = parallel_tempering(bimodal, [5.0], temperatures=(1.0, 3.0, 9.0, 27.0),
                        step=1.5, n=20000, burn=2000, rng=0)
rw = random_walk_metropolis(bimodal, [5.0], step=1.5, n=20000, burn=2000, rng=0)
print(f"\n  Bimodal target with modes at +-5:")
print(f"    parallel tempering visited both modes: "
      f"{bool((pt < 0).any() and (pt > 0).any())} ({pt.swaps} replica swaps)")
print(f"    plain random walk visited both modes:  "
      f"{bool((rw < 0).any() and (rw > 0).any())}")
