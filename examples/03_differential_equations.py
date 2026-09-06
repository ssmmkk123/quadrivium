"""ODEs: accuracy, stiffness, and structure preservation."""

import os
import sys

from quadrivium import numeric as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrivium.ode import *


def banner(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


banner("1. Convergence orders on y' = -2y + t")
f = lambda t, y: -2 * y + t
exact = lambda t: 1.25 * np.exp(-2 * t) + t / 2 - 0.25
print(f"  {'method':<20}{'n=40':>12}{'n=80':>12}{'ratio':>9}{'order':>8}")
for m, order in ((euler, 1), (heun, 2), (rk3, 3), (rk4, 4)):
    e1 = abs(m(f, (0, 2), [1.0], 40).y[-1, 0] - exact(2))
    e2 = abs(m(f, (0, 2), [1.0], 80).y[-1, 0] - exact(2))
    print(f"  {m.__name__:<20}{e1:>12.2e}{e2:>12.2e}{e1 / e2:>9.1f}{order:>8}")

banner("2. Adaptive step size control")
print(f"  {'method':<22}{'error':>12}{'steps':>9}{'rejected':>10}{'f evals':>10}")
for tableau in ("rkf45", "cash_karp", "dormand_prince", "bogacki_shampine"):
    r = adaptive_rk(f, (0, 2), [1.0], tableau, rtol=1e-10, atol=1e-12)
    print(f"  {tableau:<22}{abs(r.y[-1, 0] - exact(2)):>12.2e}"
          f"{r.n_accepted:>9}{r.n_rejected:>10}{r.n_rhs_evals:>10}")

banner("3. Stiffness: where explicit methods fail")
lam = -1000.0
stiff = lambda t, y: lam * (y - np.cos(t)) - np.sin(t)     # solution: cos(t)
print("  y' = -1000(y - cos t) - sin t,  y(0) = 1,  exact solution cos(t)")
print(f"  Using only 50 steps on [0, 1] (h = 0.02, |lambda|h = 20):\n")
print(f"  {'method':<24}{'type':<12}{'error at t=1':>16}")
for name, kind, run in (
        ("rk4", "explicit", lambda: rk4(stiff, (0, 1), [1.0], 50)),
        ("dormand_prince", "explicit", lambda: dormand_prince(stiff, (0, 1), [1.0], rtol=1e-6)),
        ("backward_euler", "implicit", lambda: backward_euler(stiff, (0, 1), [1.0], 50)),
        ("trapezoidal", "implicit", lambda: trapezoidal(stiff, (0, 1), [1.0], 50)),
        ("radau_iia", "implicit", lambda: radau_iia(stiff, (0, 1), [1.0], 50)),
        ("tr_bdf2", "implicit", lambda: tr_bdf2(stiff, (0, 1), [1.0], 50)),
        ("bdf(order=2)", "implicit", lambda: bdf(stiff, (0, 1), [1.0], 50, order=2)),
        ("rosenbrock", "implicit", lambda: rosenbrock(stiff, (0, 1), [1.0], 50))):
    r = run()
    err = abs(r.y[-1, 0] - np.cos(1))
    extra = f" ({r.n_accepted} adaptive steps)" if name == "dormand_prince" else ""
    shown = "unstable" if not np.isfinite(err) or err > 1 else f"{err:.2e}"
    print(f"  {name:<24}{kind:<12}{shown:>16}{extra}")
print("\n  Explicit RK4 is limited by stability, not accuracy: it needs")
print("  h < 2/|lambda| however smooth the solution is.")

banner("4. L-stability: damping a very stiff transient")
decay = lambda t, y: -1e6 * y
print("  y' = -10^6 y, y(0) = 1, ten steps of h = 0.1. Exact answer ~ 0.\n")
for name, run in (("backward_euler (L-stable)", lambda: backward_euler(decay, (0, 1), [1.0], 10)),
                  ("radau_iia (L-stable)", lambda: radau_iia(decay, (0, 1), [1.0], 10)),
                  ("tr_bdf2 (L-stable)", lambda: tr_bdf2(decay, (0, 1), [1.0], 10)),
                  ("trapezoidal (A-stable only)", lambda: trapezoidal(decay, (0, 1), [1.0], 10)),
                  ("esdirk (A-stable only)", lambda: esdirk(decay, (0, 1), [1.0], 10))):
    print(f"  {name:<30} y(1) = {abs(run().y[-1, 0]):.3e}")
print("\n  A-stable methods stay bounded; L-stable ones actually damp the mode.")

banner("5. Symplectic integrators conserve structure")
dHdq = lambda q: q
dHdp = lambda p: p
energy = lambda y: 0.5 * (y[0] ** 2 + y[1] ** 2)
T, n = 500.0, 50000
print(f"  Harmonic oscillator integrated to t = {T:.0f} with {n} steps\n")
print(f"  {'method':<22}{'energy drift':>16}{'position error':>17}")
for m in (symplectic_euler, leapfrog, ruth3, yoshida4, pefrl):
    r = m(dHdq, dHdp, (0, T), [1.0], [0.0], n)
    print(f"  {m.__name__:<22}{energy_drift(r, energy):>16.2e}"
          f"{abs(r.y[-1, 0] - np.cos(T)):>17.2e}")
r = rk4(lambda t, y: np.array([y[1], -y[0]]), (0, T), [1.0, 0.0], n)
print(f"  {'rk4 (not symplectic)':<22}{energy_drift(r, energy):>16.2e}"
      f"{abs(r.y[-1, 0] - np.cos(T)):>17.2e}")

print("\n  Kepler two-body problem, t = 0 to 100:")
force = lambda q: q / np.linalg.norm(q) ** 3
kep = yoshida4(force, lambda p: p, (0, 100), [1.0, 0.0], [0.0, 1.0], 20000)
E = lambda y: 0.5 * (y[2] ** 2 + y[3] ** 2) - 1.0 / np.hypot(y[0], y[1])
L = lambda y: y[0] * y[3] - y[1] * y[2]
print(f"    energy drift           {energy_drift(kep, E):.2e}")
print(f"    angular momentum drift "
      f"{max(abs(L(y) - L(kep.y[0])) for y in kep.y):.2e}")

banner("6. Exponential integrators for semilinear problems")
A = np.array([[-100.0]])
g = lambda t, y: np.array([100 * np.cos(t) - np.sin(t)])
print("  y' = -100 y + (100 cos t - sin t),  exact solution cos(t)\n")
print(f"  {'method':<24}{'n=20':>12}{'n=40':>12}{'ratio':>9}")
for m in (exponential_euler, etd_rk2, etd_rk4):
    e = [abs(m(A, g, (0, 1), [1.0], k).y[-1, 0] - np.cos(1)) for k in (20, 40)]
    print(f"  {m.__name__:<24}{e[0]:>12.2e}{e[1]:>12.2e}{e[0] / e[1]:>9.1f}")
print("\n  The stiff linear part is integrated exactly by the matrix exponential.")

banner("7. Boundary value problems")
print("  y'' = y,  y(0) = 0,  y(1) = 1   (exact: sinh(x)/sinh(1))\n")
ex = lambda t: np.sinh(t) / np.sinh(1)
for name, run in (("shooting", lambda: shooting(lambda t, y, yp: y, (0, 1), 0.0, 1.0, n=200)),
                  ("multiple shooting", lambda: multiple_shooting(lambda t, y, yp: y, (0, 1), 0.0, 1.0, 4, 200)),
                  ("finite differences", lambda: finite_difference_bvp(lambda t: 0.0, lambda t: 1.0, lambda t: 0.0, (0, 1), 0.0, 1.0, 200)),
                  ("Galerkin FEM", lambda: galerkin_bvp(lambda t: 0.0, lambda t: 1.0, lambda t: 0.0, (0, 1), 0.0, 1.0, 200)),
                  ("Chebyshev collocation", lambda: collocation_bvp(lambda t, y, yp: y, (0, 1), 0.0, 1.0, 16))):
    s = run()
    print(f"  {name:<26} max error {np.max(np.abs(s.y[:, 0] - ex(s.t))):.2e}")

print("\n  Sturm-Liouville: -y'' = lambda y on (0, pi), Dirichlet ends")
vals, _, _ = sturm_liouville(lambda t: 1.0, lambda t: 0.0, lambda t: 1.0,
                             (0, np.pi), 400, 5)
print(f"    computed eigenvalues {np.round(vals, 5)}")
print(f"    exact  n^2           {np.arange(1, 6) ** 2}")

banner("8. A chaotic system: Lorenz")
sigma, rho, beta = 10.0, 28.0, 8 / 3
lorenz = lambda t, y: np.array([sigma * (y[1] - y[0]),
                                y[0] * (rho - y[2]) - y[1],
                                y[0] * y[1] - beta * y[2]])
r = dormand_prince(lorenz, (0, 40), [1.0, 1.0, 1.0], rtol=1e-11, atol=1e-13)
print(f"  {r.n_accepted} accepted steps, {r.n_rejected} rejected, "
      f"{r.n_rhs_evals} evaluations")
print(f"  state at t=40: {np.round(r.y[-1], 5)}")
r2 = dormand_prince(lorenz, (0, 40), [1.0 + 1e-9, 1.0, 1.0], rtol=1e-11, atol=1e-13)
print(f"  after perturbing y0 by 1e-9: {np.round(r2.y[-1], 5)}")
print(f"  separation at t=40: {np.linalg.norm(r.y[-1] - r2.y[-1]):.3f}")
print("\n  Both trajectories are computed accurately; they diverge because the")
print("  system is chaotic, not because the integrator is inaccurate.")
