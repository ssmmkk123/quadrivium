"""Optimization: local, global, constrained, and structured problems."""

import os
import sys

from quadrivium import numeric as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrivium.optimize import *


def banner(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def rosenbrock(v):
    return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2


def rosen_grad(v):
    return np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
                     200 * (v[1] - v[0] ** 2)])


def rosen_hess(v):
    return np.array([[2 - 400 * (v[1] - 3 * v[0] ** 2), -400 * v[0]],
                     [-400 * v[0], 200.0]])


banner("1. Rosenbrock from (-1.2, 1.0): the classic hard valley")
print(f"  {'method':<24}{'x1':>12}{'x2':>12}{'f':>12}{'iters':>8}{'f evals':>9}")
runs = [
    ("gradient_descent", lambda: gradient_descent(rosenbrock, [-1.2, 1.0], rosen_grad, line_search=True, tol=1e-8, max_iter=50000)),
    ("conjugate_gradient_pr", lambda: conjugate_gradient_pr(rosenbrock, [-1.2, 1.0], rosen_grad, tol=1e-8)),
    ("barzilai_borwein", lambda: barzilai_borwein(rosenbrock, [-1.2, 1.0], rosen_grad, tol=1e-8)),
    ("bfgs", lambda: bfgs(rosenbrock, [-1.2, 1.0], rosen_grad, tol=1e-10)),
    ("lbfgs", lambda: lbfgs(rosenbrock, [-1.2, 1.0], rosen_grad, tol=1e-10)),
    ("newton", lambda: newton_method(rosenbrock, [-1.2, 1.0], rosen_grad, rosen_hess, tol=1e-10)),
    ("trust_region (dogleg)", lambda: trust_region(rosenbrock, [-1.2, 1.0], rosen_grad, rosen_hess, tol=1e-10)),
    ("nelder_mead", lambda: nelder_mead(rosenbrock, [-1.2, 1.0], tol=1e-12, max_iter=10000)),
    ("powell", lambda: powell(rosenbrock, [-1.2, 1.0], tol=1e-13)),
    ("hooke_jeeves", lambda: hooke_jeeves(rosenbrock, [-1.2, 1.0], step=0.5, tol=1e-10)),
]
for name, run in runs:
    r = run()
    print(f"  {name:<24}{r.x[0]:>12.8f}{r.x[1]:>12.8f}{r.fun:>12.1e}"
          f"{r.iterations:>8}{r.function_calls:>9}")
print("\n  The last three use no derivatives at all.")

banner("2. Line searches and why the curvature condition matters")
Q = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([1.0, -1.0])
fq = lambda v: 0.5 * v @ Q @ v - b @ v
gq = lambda v: Q @ v - b
x = np.array([0.3, -0.4])
d = -gq(x)
a_star = -(gq(x) @ d) / (d @ Q @ d)
print(f"  exact minimizer along the search direction: alpha* = {a_star:.12f}")
for c2 in (0.9, 0.1, 0.01):
    a = strong_wolfe(fq, gq, x, d, c2=c2)
    print(f"    strong Wolfe with c2={c2:<5} -> alpha = {a:.12f}   "
          f"(relative error {abs(a - a_star) / a_star:.1e})")
print("\n  With an interpolating zoom the search lands on the exact minimizer of")
print("  a quadratic, which restores CG's finite-termination property:")
for v in ("fr", "pr", "hs"):
    r = nonlinear_cg(fq, [0.0, 0.0], gq, v, tol=1e-12)
    print(f"    nonlinear CG ({v}): {r.iterations} iterations for a 2-D quadratic")

banner("3. Global optimization on a landscape full of local minima")
def rastrigin(v):
    v = np.asarray(v)
    return 10 * len(v) + np.sum(v ** 2 - 10 * np.cos(2 * np.pi * v))

bounds = [(-5.12, 5.12)] * 3
print("  Rastrigin in 3-D: global minimum 0 at the origin, ~10^3 local minima\n")
print(f"  {'method':<26}{'f found':>14}{'f evals':>10}")
for name, run in (
        ("random_search (baseline)", lambda: random_search(rastrigin, bounds, 20000, rng=0)),
        ("simulated_annealing", lambda: simulated_annealing(rastrigin, [4.0, 4.0, 4.0], bounds, T0=20, cooling=0.9995, max_iter=40000, rng=0)),
        ("particle_swarm", lambda: particle_swarm(rastrigin, bounds, 40, 300, rng=0)),
        ("differential_evolution", lambda: differential_evolution(rastrigin, bounds, 30, max_iter=300, rng=0)),
        ("genetic_algorithm", lambda: genetic_algorithm(rastrigin, bounds, 60, 200, rng=0)),
        ("cma_es (pop 50)", lambda: cma_es(rastrigin, [2.0, 2.0, 2.0], sigma0=2.0, pop_size=50, max_iter=2000, rng=0)),
        ("basin_hopping", lambda: basin_hopping(rastrigin, [4.0, 4.0, 4.0], 60, step=1.5, rng=0)),
        ("dual_annealing", lambda: dual_annealing_lite(rastrigin, bounds, 20000, rng=0))):
    r = run()
    print(f"  {name:<26}{r.fun:>14.3e}{r.function_calls:>10}")

print("\n  CMA-ES earns its keep on ill-conditioned, non-separable problems:")
def elli(v):
    v = np.asarray(v)
    n = len(v)
    return float(np.sum((1e6 ** (np.arange(n) / (n - 1))) * v ** 2))
print(f"    10-D ellipsoid with condition number 10^6: f = "
      f"{cma_es(elli, np.ones(10), sigma0=1.0, max_iter=3000, rng=0).fun:.2e}")

banner("4. Constrained optimization")
f = lambda v: v[0] ** 2 + v[1] ** 2
print("  minimize x^2 + y^2 subject to x + y = 1   (solution: (0.5, 0.5))\n")
for name, run in (("penalty", lambda: penalty_method(f, [2.0, -1.0], eq=lambda v: np.array([v[0] + v[1] - 1.0]))),
                  ("augmented Lagrangian", lambda: augmented_lagrangian(f, [2.0, -1.0], eq=lambda v: np.array([v[0] + v[1] - 1.0]))),
                  ("SQP", lambda: sqp(f, [2.0, -1.0], eq=lambda v: np.array([v[0] + v[1] - 1.0])))):
    r = run()
    print(f"  {name:<24} x = ({r.x[0]:.10f}, {r.x[1]:.10f})")

print("\n  minimize x^2 + y^2 subject to x + y >= 2   (solution: (1, 1))\n")
ineq = lambda v: np.array([2.0 - v[0] - v[1]])
for name, run in (("penalty", lambda: penalty_method(f, [0.0, 0.0], ineq=ineq)),
                  ("log barrier", lambda: barrier_method(f, [3.0, 3.0], ineq=ineq)),
                  ("augmented Lagrangian", lambda: augmented_lagrangian(f, [0.0, 0.0], ineq=ineq)),
                  ("SQP", lambda: sqp(f, [0.0, 0.0], ineq=ineq))):
    r = run()
    print(f"  {name:<24} x = ({r.x[0]:.10f}, {r.x[1]:.10f})")

banner("5. Sparse recovery: L1 versus L2 regularization")
rng = np.random.default_rng(0)
m, n, k = 50, 100, 5
A = rng.standard_normal((m, n)) / np.sqrt(m)
x_true = np.zeros(n)
idx = rng.choice(n, k, replace=False)
x_true[idx] = rng.standard_normal(k) * 2 + 1
b = A @ x_true + 0.01 * rng.standard_normal(m)
print(f"  {m} measurements of a {n}-dimensional signal with only {k} nonzeros\n")
lam = 0.05
star = lasso(A, b, lam, tol=1e-14, max_iter=200000)
ridge_sol = ridge(A, b, 1.0)
print(f"  {'method':<22}{'nonzeros':>10}{'support found':>16}")
print(f"  {'ridge (L2)':<22}{int(np.sum(np.abs(ridge_sol.x) > 1e-6)):>10}"
      f"{'no':>16}")
print(f"  {'lasso (L1)':<22}{int(np.sum(np.abs(star.x) > 1e-6)):>10}"
      f"{str(set(idx) <= set(np.flatnonzero(np.abs(star.x) > 1e-6))):>16}")

print("\n  ISTA vs FISTA: suboptimality at a fixed iteration budget")
print(f"  {'iterations':>12}{'ISTA':>14}{'FISTA':>14}{'speedup':>12}")
for budget in (20, 50, 100):
    gi = lasso(A, b, lam, accelerate=False, tol=0.0, max_iter=budget).fun - star.fun
    gf = lasso(A, b, lam, accelerate=True, tol=0.0, max_iter=budget).fun - star.fun
    print(f"  {budget:>12}{gi:>14.2e}{gf:>14.2e}{gi / gf:>11.0f}x")
print("\n  Nesterov acceleration turns O(1/k) into O(1/k^2).")

banner("6. Linear programming")
print("  maximize 3x + 5y  s.t.  x <= 4,  2y <= 12,  3x + 2y <= 18")
c = np.array([-3.0, -5.0])
A_ub = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
b_ub = np.array([4.0, 12.0, 18.0])
for method in ("simplex", "interior_point"):
    r = linprog(c, A_ub, b_ub, method=method)
    print(f"    {method:<16} x = ({r.x[0]:.6f}, {r.x[1]:.6f})   objective = "
          f"{-r.fun:.6f}")

print("\n  Assignment problem (Hungarian algorithm):")
C = np.array([[10.0, 19, 8, 15], [10, 18, 7, 17], [13, 16, 9, 14], [12, 19, 8, 18]])
rows, cols, total = assignment_problem(C)
print(f"    cost matrix rows -> columns: {dict(zip(rows.tolist(), cols.tolist()))}")
print(f"    total cost: {total:.0f}")

banner("7. Nonlinear least squares with uncertainty estimates")
xd = np.linspace(0, 2, 40)
true = [2.5, -1.3]
yd = true[0] * np.exp(true[1] * xd) + 0.01 * rng.standard_normal(40)
r = curve_fit(lambda x, a, b: a * np.exp(b * x), xd, yd, [1.0, -0.5])
print(f"  fitting y = a exp(b x) to 40 noisy points")
print(f"    a = {r.x[0]:.5f} +- {r.std_errors[0]:.5f}   (true {true[0]})")
print(f"    b = {r.x[1]:.5f} +- {r.std_errors[1]:.5f}   (true {true[1]})")
print(f"    converged in {r.iterations} Levenberg-Marquardt iterations")
