"""Differentiation and integration: accuracy, and where naive methods fail."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numethods.diff import *
from numethods.integrate import *
from numethods.special import *


def banner(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


banner("1. The step-size dilemma, and three ways around it")
f = lambda t: np.exp(np.sin(t))
df = lambda t: np.cos(t) * np.exp(np.sin(t))
x = 1.3
print(f"  {'h':>10}{'forward':>14}{'central':>14}{'5-point':>14}")
for h in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
    e_f = abs(forward_difference(f, x, h) - df(x))
    e_c = abs(central_difference(f, x, h) - df(x))
    e_5 = abs(five_point_stencil(f, x, h) - df(x))
    print(f"  {h:>10.0e}{e_f:>14.2e}{e_c:>14.2e}{e_5:>14.2e}")
print("\n  Truncation error falls with h until round-off takes over. Three")
print("  techniques sidestep the tradeoff entirely:")
print(f"    Richardson extrapolation : error {abs(richardson_derivative(f, x) - df(x)):.2e}")
print(f"    complex-step derivative  : error {abs(complex_step_derivative(f, x) - df(x)):.2e}")
print(f"    automatic differentiation: error {abs(derivative(f, x) - df(x)):.2e}")

banner("2. Automatic differentiation is exact, not approximate")
g = lambda v: v[0] * v[1] + np.sin(v[0] * v[2]) + v[1] ** 2 / v[2]
pt = [1.5, 2.0, 0.7]
exact = np.array([2.0 + 0.7 * math.cos(1.05),
                  1.5 + 4.0 / 0.7,
                  1.5 * math.cos(1.05) - 4.0 / 0.49])
print(f"  analytic gradient : {exact}")
print(f"  reverse mode      : {gradient(g, pt)}")
print(f"  forward mode      : {forward_gradient(g, pt)}")
print(f"  max error         : {np.max(np.abs(gradient(g, pt) - exact)):.2e}")

q = lambda v: v[0] ** 3 * v[1] + v[1] ** 2 + np.sin(v[0] * v[1])
a, b = 2.0, 3.0
H_exact = np.array([
    [6 * a * b - b * b * math.sin(a * b),
     3 * a * a + math.cos(a * b) - a * b * math.sin(a * b)],
    [3 * a * a + math.cos(a * b) - a * b * math.sin(a * b),
     2 - a * a * math.sin(a * b)]])
print(f"\n  Hessian by hyper-dual numbers vs analytic: "
      f"{np.max(np.abs(hessian(q, [a, b]) - H_exact)):.2e}")
print(f"  Hessian by finite differences            : "
      f"{np.max(np.abs(hessian_fd(q, [a, b]) - H_exact)):.2e}")

banner("3. Quadrature: convergence orders")
integrand = lambda t: np.exp(-t) * np.sin(3 * t)
exact_int = (3 - np.exp(-2) * (np.sin(6) + 3 * np.cos(6))) / 10
print(f"  integral of e^-x sin(3x) over [0, 2] = {exact_int:.15f}\n")
print(f"  {'rule':<20}{'n=20':>12}{'n=40':>12}{'ratio':>9}{'order':>8}")
for rule, order in ((trapezoid_rule, 2), (simpson_rule, 4), (boole_rule, 6)):
    e1 = abs(rule(integrand, 0, 2, 20).value - exact_int)
    e2 = abs(rule(integrand, 0, 2, 40).value - exact_int)
    print(f"  {rule.__name__:<20}{e1:>12.2e}{e2:>12.2e}{e1 / e2:>9.1f}{order:>8}")

banner("4. Cost of reaching machine precision")
print(f"  {'method':<26}{'error':>12}{'f evaluations':>16}")
for name, run in (("trapezoid (n=10^5)", lambda: trapezoid_rule(integrand, 0, 2, 100000)),
                  ("simpson (n=10^4)", lambda: simpson_rule(integrand, 0, 2, 10000)),
                  ("romberg", lambda: romberg(integrand, 0, 2)),
                  ("adaptive Simpson", lambda: adaptive_simpson(integrand, 0, 2, 1e-13)),
                  ("Gauss-Kronrod", lambda: adaptive_gauss_kronrod(integrand, 0, 2, 1e-13)),
                  ("Gauss-Legendre (n=20)", lambda: gauss_legendre(integrand, 0, 2, 20)),
                  ("Clenshaw-Curtis (n=32)", lambda: clenshaw_curtis(integrand, 0, 2, 32)),
                  ("tanh-sinh", lambda: tanh_sinh(integrand, 0, 2))):
    r = run()
    print(f"  {name:<26}{abs(r.value - exact_int):>12.2e}{r.function_calls:>16}")

banner("5. Integrands that defeat ordinary rules")
cases = [("1/sqrt(x) on [0,1]", lambda t: 1 / np.sqrt(t), 0, 1, 2.0),
         ("log(x) on [0,1]", np.log, 0, 1, -1.0),
         ("x^-0.9 on [0,1]", lambda t: t ** (-0.9), 0, 1, 10.0)]
print(f"  {'integrand':<22}{'Gauss-Kronrod':>16}{'tanh-sinh':>14}")
for name, fn, lo, hi, exact in cases:
    try:
        gk = abs(adaptive_gauss_kronrod(fn, lo, hi, 1e-10).value - exact)
        gk_s = f"{gk:.2e}"
    except Exception:
        gk_s = "failed"
    ts = abs(tanh_sinh(fn, lo, hi).value - exact)
    print(f"  {name:<22}{gk_s:>16}{ts:>14.2e}")
print("\n  The double-exponential transformation flattens endpoint singularities.")

print("\n  Infinite intervals are handled by variable transformation:")
for name, fn, lo, hi, exact in (
        ("gaussian over R", lambda t: np.exp(-t * t), -np.inf, np.inf, np.sqrt(np.pi)),
        ("1/(1+x^2) on [0,inf)", lambda t: 1 / (1 + t * t), 0, np.inf, np.pi / 2)):
    print(f"    {name:<24} error {abs(quad(fn, lo, hi).value - exact):.2e}")

banner("6. High dimensions: Monte Carlo and its refinements")
g4 = lambda p: np.exp(-np.sum(np.asarray(p) ** 2))
exact4 = (math.erf(1) * math.sqrt(math.pi) / 2) ** 4
lo, hi = np.zeros(4), np.ones(4)
print(f"  4-D integral, exact value {exact4:.10f}, 4096 samples\n")
for name, run in (("plain Monte Carlo", lambda: monte_carlo_nd(g4, lo, hi, 4096, rng=0)),
                  ("quasi-MC (Halton)", lambda: quasi_monte_carlo(g4, lo, hi, 4096, "halton")),
                  ("quasi-MC (Sobol)", lambda: quasi_monte_carlo(g4, lo, hi, 4096, "sobol")),
                  ("tensor Gauss (8^4)", lambda: tensor_gauss(g4, lo, hi, 8))):
    r = run()
    print(f"  {name:<22} error {abs(r.value - exact4):.2e}")
print("\n  Low-discrepancy points beat pseudorandom ones; tensor product rules")
print("  are best of all here but cost n^d and become impractical by d ~ 8.")

banner("7. Special functions")
print(f"  Gamma(1/2)      = {gamma(0.5):.15f}   (sqrt(pi) = {math.sqrt(math.pi):.15f})")
print(f"  zeta(2)         = {zeta(2):.15f}   (pi^2/6   = {math.pi**2 / 6:.15f})")
print(f"  K(0.5), E(0.5)  = {elliptic_k(0.5):.12f}, {elliptic_e(0.5):.12f}")
print(f"  J0(1), Y0(1)    = {bessel_j0(1):.12f}, {bessel_y0(1):.12f}")
w = bessel_j1(2.0) * bessel_y0(2.0) - bessel_j0(2.0) * bessel_y1(2.0)
print(f"  Wronskian check : J1*Y0 - J0*Y1 = {w:.12f}  vs  2/(pi x) = "
      f"{2 / (math.pi * 2):.12f}")
