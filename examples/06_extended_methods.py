"""The extended methods: SDEs, wavelets, matrix equations, high-resolution PDEs.

Each section checks the result against something independent -- an exact
identity, a convergence order, an analytic solution, or a published benchmark --
rather than just printing numbers.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numethods.integrate import filon, gauss_legendre, sparse_grid_quadrature
from numethods.linalg import (condition_estimate, lyapunov, qz_decomposition,
                              qz_eigenvalues, randomized_svd, sqrtm, sylvester)
from numethods.ode import (dormand_prince, gragg_bulirsch_stoer,
                           solve_ivp_events)
from numethods.pde import lid_driven_cavity, navier_stokes_2d, weno_burgers
from numethods.rootfind import anderson_acceleration, fixed_point_system
from numethods.special import dawson, lambert_w, spherical_harmonic, zeta
from numethods.stochastic import gillespie_ssa, milstein
from numethods.transforms import wavedec, wavelet_denoise, waverec

np.set_printoptions(precision=6, suppress=True)


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


banner("1. Stochastic differential equations: Milstein is strong order 1")
mu, sig, x0, T = 1.5, 0.6, 1.0, 1.0
a = lambda x, t: mu * x
b = lambda x, t: sig * x
db = lambda x, t: sig * np.ones_like(x)
exact = lambda wt: x0 * np.exp((mu - 0.5 * sig**2) * T + sig * wt)
prev = None
for n in (16, 32, 64, 128, 256):
    rng = np.random.default_rng(0)
    dt, total = T / n, 0.0
    for _ in range(400):
        dW = np.sqrt(dt) * rng.standard_normal((n, 1))
        total += abs(milstein(a, b, (0, T), [x0], n=n, dW=dW, db=db).y[-1][0]
                     - exact(dW.sum()))
    err = total / 400
    rate = "" if prev is None else f"   rate {math.log2(prev / err):.2f}"
    print(f"  n = {n:4d}   E|X_N - X(T)| = {err:.3e}{rate}")
    prev = err
print("  (Euler-Maruyama would give rate 1/2 here -- it drops the Ito correction)")

banner("2. Gillespie SSA is exact, not a discretization")
k, X0, TT = 0.8, 40, 1.5
prop = lambda x, t: np.array([k * x[0]])
finals = np.array([gillespie_ssa(prop, np.array([[-1.0]]), [X0], (0, TT),
                                 rng=np.random.default_rng(s))[1][-1, 0]
                   for s in range(2000)])
p = math.exp(-k * TT)
print(f"  pure death process, 2000 realizations")
print(f"    mean     {finals.mean():8.4f}   exact (binomial) {X0 * p:8.4f}")
print(f"    variance {finals.var():8.4f}   exact            {X0 * p * (1 - p):8.4f}")

banner("3. Wavelets: perfect reconstruction and denoising")
rng = np.random.default_rng(1)
for wav in ("haar", "db4", "coif2"):
    x = rng.standard_normal(512)
    err = np.abs(waverec(wavedec(x, wav, 5), wav, 512) - x).max()
    print(f"  {wav:6s} 5-level decompose then reconstruct: {err:.2e}")
t = np.linspace(0, 1, 1024)
clean = np.sin(4 * np.pi * t) + np.where(t > 0.5, 1.0, 0.0)
noisy = clean + 0.3 * rng.standard_normal(1024)
print(f"  noisy signal RMSE {np.sqrt(np.mean((noisy - clean)**2)):.4f}")
for wav in ("haar", "db4", "coif2"):
    den = wavelet_denoise(noisy, wav, level=5)
    print(f"    denoised with {wav:6s}: {np.sqrt(np.mean((den - clean)**2)):.4f}")
print("  (the jump at t = 0.5 survives -- a Fourier low-pass would ring around it)")

banner("4. Matrix equations solved in O(n^3), not O(n^6)")
rng = np.random.default_rng(2)
A = rng.standard_normal((6, 6))
B = rng.standard_normal((5, 5))
C = rng.standard_normal((6, 5))
X = sylvester(A, B, C)
print(f"  Sylvester  A X + X B = C:      residual {np.abs(A @ X + X @ B - C).max():.2e}")
As = rng.standard_normal((6, 6)) - 3.5 * np.eye(6)
Q = rng.standard_normal((6, 6))
Q = Q @ Q.T
Y = lyapunov(As, Q)
print(f"  Lyapunov   A Y + Y A' + Q = 0: residual {np.abs(As @ Y + Y @ As.T + Q).max():.2e}")
print(f"             solution is positive semidefinite: "
      f"{np.linalg.eigvalsh(Y).min() > -1e-10}")
M = np.array([[4.0, 1.0], [2.0, 3.0]])
print(f"  sqrtm:     ||X^2 - A||         {np.abs(sqrtm(M) @ sqrtm(M) - M).max():.2e}")
Aq, Bq = rng.standard_normal((6, 6)), rng.standard_normal((6, 6))
Qz, Zz, S, T2 = qz_decomposition(Aq, Bq)
got = qz_eigenvalues(S, T2)
ref = np.linalg.eigvals(np.linalg.solve(Bq, Aq))
d = np.abs(got[:, None] - ref[None, :])
print(f"  QZ pencil eigenvalues:         {max(d.min(axis=1).max(), d.min(axis=0).max()):.2e}")
Ah = np.array([[1 / (i + j + 1) for j in range(6)] for i in range(6)])
print(f"  Hager condition estimate:      {condition_estimate(Ah):.4g}   "
      f"true {np.linalg.cond(Ah, 1):.4g}   (no inverse formed)")

banner("5. Randomized SVD attains the Eckart-Young optimum")
U, _, V = np.linalg.svd(rng.standard_normal((300, 200)), full_matrices=False)
s = np.exp(-np.arange(200) / 12.0)
A = U @ np.diag(s) @ V
for k in (5, 15, 30):
    Ur, sr, Vr = randomized_svd(A, k, rng=rng)
    err = np.linalg.norm(A - Ur @ np.diag(sr) @ Vr, 2)
    print(f"  k = {k:3d}   ||A - A_k||_2 = {err:.4e}   optimal {s[k]:.4e}   "
          f"ratio {err / s[k]:.4f}")

banner("6. Extrapolation: same accuracy for a fraction of the work")
f = lambda t, y: -y + np.sin(t)
ex = lambda t: (np.sin(t) - np.cos(t) + np.exp(-t)) / 2
gbs = gragg_bulirsch_stoer(f, (0, 10), [0.0], rtol=1e-12, atol=1e-14)
dp = dormand_prince(f, (0, 10), [0.0], rtol=1e-12, atol=1e-14)
print(f"  Gragg-Bulirsch-Stoer  err {abs(gbs.y[-1][0] - ex(10.0)):.2e}   "
      f"{gbs.n_rhs_evals:5d} f-evals")
print(f"  Dormand-Prince 5(4)   err {abs(dp.y[-1][0] - ex(10.0)):.2e}   "
      f"{dp.n_rhs_evals:5d} f-evals   ({dp.n_rhs_evals / gbs.n_rhs_evals:.1f}x more)")

banner("7. Event location is as accurate as the integrator")
g = 9.81
rhs = lambda t, y: np.array([y[1], -g])
sol, te, ye = solve_ivp_events(rhs, (0, 3), [10.0, 0.0],
                               events=lambda t, y: y[0], terminal=True,
                               rtol=1e-12, atol=1e-14)
t_exact = math.sqrt(20.0 / g)
print(f"  ball dropped from 10 m hits the ground at t = {te[0][0]:.14f}")
print(f"                                       exact   {t_exact:.14f}")
print(f"                                       error   {abs(te[0][0] - t_exact):.2e}")
print(f"  impact speed {ye[0][0][1]:.10f} m/s   (exact {-g * t_exact:.10f})")

banner("8. Anderson acceleration turns a crawl into a sprint")
Am = np.array([[0.0, 0.45, 0.4], [0.45, 0.0, 0.45], [0.4, 0.45, 0.0]])
bm = np.array([1.0, 2.0, 3.0])
gfun = lambda x: Am @ x + bm
xstar = np.linalg.solve(np.eye(3) - Am, bm)
for m in (0, 2, 5):
    r = anderson_acceleration(gfun, np.zeros(3), m=m, tol=1e-12)
    label = "plain Picard" if m == 0 else f"Anderson m={m}"
    print(f"  {label:16s}: {r.iterations:4d} iterations, "
          f"error {np.linalg.norm(r.root - xstar):.2e}")

banner("9. WENO5 captures a shock without oscillation")
s = weno_burgers(lambda x: np.where(x < np.pi, 1.0, 0.0), (0, 2 * np.pi),
                 (0, 1.0), nx=400)
u, xg = np.asarray(s.final), np.asarray(s.x)
band = (xg > 2.5) & (xg < 5.0)
loc = xg[band][int(np.argmin(np.abs(u[band] - 0.5)))]
cells = int(np.sum((u[band] > 0.05) & (u[band] < 0.95)))
print(f"  Burgers shock at t = 1: located at x = {loc:.4f}, "
      f"exact pi + t/2 = {np.pi + 0.5:.4f}")
print(f"  resolved in {cells} transition cells, overshoot {max(u.max() - 1, 0):.2e}")

banner("10. Incompressible flow: the Ghia lid-driven cavity benchmark")
psi, w, xc = lid_driven_cavity(re=100.0, n=65, tol=1e-7, max_iter=40000)
i = np.unravel_index(int(np.argmin(psi)), psi.shape)
print(f"  Re = 100   primary vortex at ({xc[i[1]]:.4f}, {xc[i[0]]:.4f})   "
      f"Ghia et al. (1982): (0.6172, 0.7344)")
print(f"             psi_min = {psi.min():.5f}                  "
      f"Ghia et al. (1982): -0.1034")
nu = 0.05
ns = navier_stokes_2d(lambda X, Y: np.cos(X) * np.sin(Y),
                      lambda X, Y: -np.sin(X) * np.cos(Y), nu, (0, 1.0),
                      nx=64, ny=64, nt=400)
X, Y = np.meshgrid(ns.x, ns.y, indexing="xy")
uu, vv = np.asarray(ns.u)[-1]
dx = ns.x[1] - ns.x[0]
div = ((np.roll(uu, -1, axis=1) - np.roll(uu, 1, axis=1)) / (2 * dx)
       + (np.roll(vv, -1, axis=0) - np.roll(vv, 1, axis=0)) / (2 * dx))
print(f"  Taylor-Green vortex: velocity error "
      f"{np.abs(uu - np.cos(X) * np.sin(Y) * math.exp(-2 * nu)).max():.2e}")
print(f"                       max |div u| after projection {np.abs(div).max():.2e}")

banner("11. Quadrature for problems ordinary rules cannot touch")
exact_osc = lambda w: (math.exp(1) * (math.sin(w) - w * math.cos(w)) + w) / (1 + w * w)
print("  int_0^1 e^x sin(wx) dx, 200 nodes:")
for w in (10.0, 1000.0, 100000.0):
    fv = filon(np.exp, 0, 1, w, n=200, kind="sin").value
    gv = gauss_legendre(lambda x: np.exp(x) * np.sin(w * x), 0, 1, 200).value
    print(f"    w = {w:9.0f}   Filon {abs(fv - exact_osc(w)):.2e}   "
          f"Gauss {abs(gv - exact_osc(w)):.2e}")
print("  int over [0,1]^d of exp(sum x):")
for d in (4, 6, 8):
    r = sparse_grid_quadrature(lambda pt: np.exp(np.sum(pt)), [0.0] * d,
                               [1.0] * d, level=5)
    exact_d = (math.e - 1) ** d
    print(f"    d = {d}: {r.subintervals:6d} Smolyak nodes, "
          f"rel error {abs(r.value - exact_d) / exact_d:.2e}   "
          f"(tensor grid: {17**d:,} nodes)")

banner("12. Special functions checked by their own identities")
print(f"  Lambert W:  W(e) = {lambert_w(math.e):.15f}   (exactly 1)")
print(f"              W(x) e^W(x) - x over 8 arguments: "
      f"{max(abs(lambert_w(x) * math.exp(lambert_w(x)) - x) / max(abs(x), 1e-3) for x in [-0.36, -0.1, 0.5, 1, 10, 1e3, 1e6, 1e10]):.2e}")
print(f"  Dawson:     F' - (1 - 2xF) over 7 arguments:  "
      f"{max(abs((dawson(x + 1e-5) - dawson(x - 1e-5)) / 2e-5 - (1 - 2 * x * dawson(x))) for x in [0.05, 0.5, 1, 3, 5, 12, 40]):.2e}")
print(f"  Riemann zeta beyond its series' half-plane:")
print(f"              zeta(-1)  = {zeta(-1.0):+.15f}   (exactly -1/12)")
print(f"              zeta(0.5) = {zeta(0.5):+.15f}   (critical strip)")
print(f"  Spherical harmonics orthonormal to "
      f"{abs(2 * np.pi * np.sum(np.polynomial.legendre.leggauss(60)[1] * np.array([abs(spherical_harmonic(2, 1, np.arccos(c), 0.0))**2 for c in np.polynomial.legendre.leggauss(60)[0]])) - 1):.2e}")

print(f"\n{'=' * 72}\nAll checks above are against exact identities, convergence")
print(f"orders, analytic solutions or published benchmarks.\n{'=' * 72}")
