"""Tests for root finding, interpolation and approximation."""

import unittest

import numpy as np

from numethods.approx import *
from numethods.core.exceptions import BracketError
from numethods.interpolate import *
from numethods.rootfind import *

CUBIC_ROOT = 2.0945514815423265        # real root of x^3 - 2x - 5


class TestScalarRootFinding(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: x**3 - 2 * x - 5
        self.df = lambda x: 3 * x**2 - 2
        self.d2f = lambda x: 6 * x

    def test_bracketing_methods(self):
        for m in (bisection, false_position, illinois, pegasus, ridders, brent, itp):
            with self.subTest(method=m.__name__):
                r = m(self.f, 1, 3)
                self.assertTrue(r.converged)
                self.assertAlmostEqual(r.root, CUBIC_ROOT, places=8)

    def test_open_methods(self):
        cases = [(secant, (2.0, 2.5)), (newton, (2.0,)), (steffensen, (2.0,)),
                 (muller, (1.0, 2.0, 3.0))]
        for m, args in cases:
            with self.subTest(method=m.__name__):
                r = m(self.f, *args)
                self.assertTrue(r.converged)
                self.assertAlmostEqual(float(np.real(r.root)), CUBIC_ROOT, places=8)

    def test_higher_order_methods(self):
        for m in (newton, halley, chebyshev_method):
            args = (self.f, 2.0, self.df) if m is newton else (self.f, 2.0, self.df, self.d2f)
            r = m(*args)
            self.assertAlmostEqual(r.root, CUBIC_ROOT, places=12)
        self.assertLessEqual(halley(self.f, 2.0, self.df, self.d2f).iterations,
                             newton(self.f, 2.0, self.df).iterations)

    def test_newton_beats_bisection(self):
        self.assertLess(newton(self.f, 2.0, self.df).iterations,
                        bisection(self.f, 1, 3).iterations)

    def test_multiplicity_correction(self):
        g, dg = lambda x: (x - 2) ** 3, lambda x: 3 * (x - 2) ** 2
        r = newton(g, 1.0, dg, multiplicity=3)
        self.assertAlmostEqual(r.root, 2.0, places=13)
        self.assertEqual(r.iterations, 1)

    def test_fixed_point_and_aitken(self):
        g = lambda x: (2 * x + 5) ** (1 / 3)
        self.assertAlmostEqual(fixed_point(g, 2.0).root, CUBIC_ROOT, places=8)
        self.assertAlmostEqual(aitken_accelerated(g, 2.0).root, CUBIC_ROOT, places=8)

    def test_bad_bracket_raises(self):
        with self.assertRaises(BracketError):
            bisection(lambda x: x * x + 1, -1, 1)

    def test_find_all_roots(self):
        roots = find_all_roots(np.sin, -0.5, 10.0)
        np.testing.assert_allclose(roots, [0, np.pi, 2 * np.pi, 3 * np.pi], atol=1e-9)

    def test_complex_root_via_muller(self):
        r = muller(lambda z: z**2 + 1, 1.0, 0.5j, 1.0 + 1j)
        self.assertAlmostEqual(abs(r.root), 1.0, places=8)


class TestNonlinearSystems(unittest.TestCase):
    def setUp(self):
        self.F = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 4, v[0] * v[1] - 1])
        self.J = lambda v: np.array([[2 * v[0], 2 * v[1]], [v[1], v[0]]])
        self.sol = np.array([1.9318516525781366, 0.5176380902050415])

    def test_all_system_solvers(self):
        for m in (newton_system, damped_newton_system, broyden_good, broyden_bad,
                  secant_system, trust_region_dogleg_root):
            with self.subTest(method=m.__name__):
                r = m(self.F, [2.0, 0.5])
                self.assertTrue(r.converged, r.message)
                np.testing.assert_allclose(r.root, self.sol, atol=1e-7)

    def test_analytic_jacobian(self):
        r = newton_system(self.F, [2.0, 0.5], self.J)
        np.testing.assert_allclose(r.root, self.sol, atol=1e-12)

    def test_three_variable_system(self):
        F = lambda v: np.array([
            3 * v[0] - np.cos(v[1] * v[2]) - 0.5,
            v[0] ** 2 - 81 * (v[1] + 0.1) ** 2 + np.sin(v[2]) + 1.06,
            np.exp(-v[0] * v[1]) + 20 * v[2] + (10 * np.pi - 3) / 3])
        for m in (newton_system, damped_newton_system, broyden_good, secant_system):
            r = m(F, [0.1, 0.1, -0.1])
            self.assertLess(np.linalg.norm(F(r.root)), 1e-8)

    def test_continuation(self):
        r = continuation(self.F, [2.0, 0.5])
        self.assertLess(np.linalg.norm(self.F(r.root)), 1e-8)


class TestPolynomialRoots(unittest.TestCase):
    def test_real_roots_all_methods(self):
        c = np.poly([1.0, 2.0, 3.0, -4.0])
        expected = np.sort([-4.0, 1.0, 2.0, 3.0])
        for m in ("companion", "durand_kerner", "aberth", "bairstow",
                  "jenkins_traub"):
            with self.subTest(method=m):
                r = polynomial_roots(c, m)
                np.testing.assert_allclose(np.sort(np.real(r)), expected, atol=1e-8)

    def test_complex_roots(self):
        c = np.real(np.poly([1 + 2j, 1 - 2j, 3.0]))
        expected = np.array([1 - 2j, 1 + 2j, 3 + 0j])
        for m in ("companion", "durand_kerner", "aberth", "bairstow"):
            r = np.asarray(polynomial_roots(c, m), dtype=complex)
            np.testing.assert_allclose(r, expected, atol=1e-7)

    def test_horner_and_deflation(self):
        self.assertAlmostEqual(horner([1, -2, -5], 3), -2.0)
        p, dp = horner_derivative([1.0, 0.0, -2.0, -5.0], 2.0)
        self.assertAlmostEqual(p, -1.0)
        self.assertAlmostEqual(dp, 10.0)
        q, rem = synthetic_division([1.0, -6.0, 11.0, -6.0], 1.0)
        np.testing.assert_allclose(q, [1, -5, 6])
        self.assertAlmostEqual(rem, 0.0)

    def test_sturm_root_counting(self):
        c = [1.0, -6.0, 11.0, -6.0]      # roots 1, 2, 3
        self.assertEqual(count_real_roots(c, 0, 4), 3)
        self.assertEqual(count_real_roots(c, 0, 1.5), 1)

    def test_degree_10_polynomial(self):
        w = np.poly(np.arange(1.0, 11.0))
        r = np.sort(np.real(polynomial_roots(w, "aberth")))
        np.testing.assert_allclose(r, np.arange(1.0, 11.0), atol=1e-6)


class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self.f = lambda t: np.exp(-t) * np.sin(3 * t)
        self.x = np.linspace(0, 2, 8)
        self.y = self.f(self.x)
        self.t = np.linspace(0, 2, 101)

    def test_polynomial_forms_agree(self):
        forms = [c(self.x, self.y) for c in (lagrange, newton_divided_differences,
                                             barycentric)]
        for p in forms:
            np.testing.assert_allclose(p(self.x), self.y, atol=1e-10)
        for p in forms[1:]:
            np.testing.assert_allclose(p(self.t), forms[0](self.t), atol=1e-10)

    def test_hermite_matches_values_and_slopes(self):
        xh = np.array([0.0, 0.5, 1.0])
        h = hermite(xh, np.sin(xh), np.cos(xh))
        np.testing.assert_allclose(h(xh), np.sin(xh), atol=1e-13)
        eps = 1e-6
        np.testing.assert_allclose((h(xh + eps) - h(xh - eps)) / (2 * eps),
                                   np.cos(xh), atol=1e-8)

    def test_runge_phenomenon(self):
        eq = [runge_demo_error(n, "equispaced") for n in (11, 21, 31)]
        ch = [runge_demo_error(n, "chebyshev") for n in (11, 21, 31)]
        self.assertLess(eq[0], eq[1])          # equispaced diverges
        self.assertLess(eq[1], eq[2])
        self.assertGreater(ch[0], ch[1])       # Chebyshev converges
        self.assertGreater(ch[1], ch[2])

    def test_splines_interpolate_and_converge(self):
        g = lambda t: np.sin(t) + 0.3 * t
        for ctor in (natural_cubic_spline, not_a_knot_spline, pchip, akima_spline,
                     linear_spline, catmull_rom):
            with self.subTest(spline=ctor.__name__):
                x = np.linspace(0, 6, 13)
                s = ctor(x, g(x))
                np.testing.assert_allclose(s(x), g(x), atol=1e-10)

    def test_natural_spline_boundary_and_smoothness(self):
        g = lambda t: np.sin(t) + 0.3 * t
        x = np.linspace(0, 6, 13)
        s = natural_cubic_spline(x, g(x))
        d2 = s.derivative(2)
        self.assertAlmostEqual(float(d2(x[0])), 0.0, places=10)
        self.assertAlmostEqual(float(d2(x[-1])), 0.0, places=10)
        d1 = s.derivative(1)
        for xi in x[1:-1]:
            self.assertLess(abs(d1(xi - 1e-7) - d1(xi + 1e-7)), 1e-5)

    def test_pchip_preserves_monotonicity(self):
        x = np.array([0.0, 1, 2, 3, 4, 5])
        y = np.array([0.0, 0, 0, 1, 1, 1])
        t = np.linspace(0, 5, 300)
        self.assertTrue(np.all(np.diff(pchip(x, y)(t)) >= -1e-12))
        self.assertLess(np.min(natural_cubic_spline(x, y)(t)), -1e-3)

    def test_spline_calculus(self):
        g = lambda t: np.sin(t) + 0.3 * t
        x = np.linspace(0, 6, 41)
        s = natural_cubic_spline(x, g(x))
        exact = 1 - np.cos(6) + 0.15 * 36
        self.assertAlmostEqual(s.integrate(0, 6), exact, places=4)

    def test_bspline_partition_of_unity(self):
        kv = np.array([0.0, 0, 0, 0, 1, 2, 3, 3, 3, 3])
        total = sum(bspline_basis(i, 3, kv, np.array([1.5]))[0] for i in range(6))
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_rational_interpolation(self):
        f = lambda t: 1.0 / (1.0 + 25 * t**2)
        x = np.linspace(-1, 1, 15)
        y = f(x)
        t = np.linspace(-0.99, 0.99, 101)
        # the target is rational, so these are exact
        self.assertLess(np.max(np.abs(rational_interpolation(x, y)(t) - f(t))), 1e-10)
        self.assertLess(np.max(np.abs(thiele(x, y)(t) - f(t))), 1e-10)
        self.assertLess(abs(bulirsch_stoer_rational(x, y, 0.05) - f(0.05)), 1e-12)

    def test_floater_hormann_converges(self):
        f = lambda t: 1.0 / (1.0 + 25 * t**2)
        t = np.linspace(-0.99, 0.99, 201)
        errs = []
        for n in (11, 21, 41):
            x = np.linspace(-1, 1, n)
            errs.append(np.max(np.abs(floater_hormann(x, f(x), 3)(t) - f(t))))
        self.assertLess(errs[2], errs[1])
        self.assertLess(errs[1], errs[0])

    def test_multivariate(self):
        g = lambda X, Y: np.sin(X) * np.cos(Y) + 0.5 * X
        x = y = np.linspace(0, 3, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        Z = g(X, Y)
        self.assertLess(abs(bilinear(x, y, Z)(1.21, 2.2) - g(1.21, 2.2)), 2e-2)
        self.assertLess(abs(bicubic(x, y, Z)(1.21, 2.2) - g(1.21, 2.2)), 1e-4)
        rng = np.random.default_rng(7)
        P = rng.uniform(0, 3, (60, 2))
        v = g(P[:, 0], P[:, 1])
        for kernel in ("multiquadric", "gaussian", "thin_plate", "cubic"):
            r = rbf_interpolation(P, v, kernel)
            np.testing.assert_allclose(r(P), v, atol=1e-6)

    def test_barycentric_triangle(self):
        lam = barycentric_triangle([0.25, 0.25], [0, 0], [1, 0], [0, 1])
        np.testing.assert_allclose(lam, [0.5, 0.25, 0.25])
        self.assertAlmostEqual(float(np.sum(lam)), 1.0, places=14)


class TestApproximation(unittest.TestCase):
    def test_orthogonal_polynomials(self):
        x = np.linspace(-0.99, 0.99, 50)
        np.testing.assert_allclose(legendre(2, x), 0.5 * (3 * x**2 - 1))
        np.testing.assert_allclose(chebyshev_t(3, x), 4 * x**3 - 3 * x)
        np.testing.assert_allclose(chebyshev_u(2, x), 4 * x**2 - 1)
        np.testing.assert_allclose(hermite_physicists(3, x), 8 * x**3 - 12 * x)
        np.testing.assert_allclose(laguerre(2, x), 1 - 2 * x + x**2 / 2)
        np.testing.assert_allclose(jacobi_polynomial(2, 0, 0, x), legendre(2, x))

    def test_gauss_rules_exactness(self):
        for n in (2, 3, 5, 8):
            x, w = gauss_legendre_nodes(n, -1, 1)
            for d in range(2 * n):     # exact through degree 2n-1
                exact = 0.0 if d % 2 else 2.0 / (d + 1)
                self.assertAlmostEqual(float(np.sum(w * x**d)), exact, places=10)

    def test_lobatto_and_radau(self):
        x, w = gauss_lobatto_nodes(6, -1, 1)
        self.assertAlmostEqual(x[0], -1.0, places=13)
        self.assertAlmostEqual(x[-1], 1.0, places=13)
        for d in range(2 * 6 - 2):
            exact = 0.0 if d % 2 else 2.0 / (d + 1)
            self.assertAlmostEqual(float(np.sum(w * x**d)), exact, places=10)
        xr, wr = gauss_radau_nodes(6, -1, 1)
        self.assertAlmostEqual(xr[0], -1.0, places=13)

    def test_fitting(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 60)
        y = 1 - 2 * x + 0.5 * x**2 + 0.02 * rng.standard_normal(60)
        np.testing.assert_allclose(polyfit(x, y, 2), [0.5, -2, 1], atol=0.05)
        a, b = exponential_fit(x, 2.5 * np.exp(-1.3 * x))
        self.assertAlmostEqual(a, 2.5, places=6)
        self.assertAlmostEqual(b, -1.3, places=6)

    def test_pade_beats_taylor(self):
        from math import factorial
        c = np.array([0.0] + [(-1) ** (k + 1) / k for k in range(1, 11)])
        p, q = pade(c, 5, 5)
        t = np.linspace(0, 0.95, 50)
        taylor = np.max(np.abs(np.polyval(c[::-1], t) - np.log(1 + t)))
        approx = np.max(np.abs(pade_evaluate(p, q, t) - np.log(1 + t)))
        self.assertLess(approx, taylor / 10)

    def test_remez_equioscillates(self):
        ap = remez(np.exp, 3, -1, 1)
        t = np.linspace(-1, 1, 4000)
        err = ap(t) - np.exp(t)
        idx = [0] + [i for i in range(1, len(t) - 1)
                     if (err[i] - err[i - 1]) * (err[i + 1] - err[i]) < 0] + [len(t) - 1]
        signs = np.sign(err[idx])
        self.assertGreaterEqual(len(idx), 5)
        self.assertTrue(np.all(signs[:-1] * signs[1:] < 0))     # alternation
        mags = np.abs(err[idx])
        self.assertLess((mags.max() - mags.min()) / mags.max(), 0.02)

    def test_fourier_series_of_square_wave(self):
        sq = lambda t: 1.0 if (t % (2 * np.pi)) < np.pi else -1.0
        s = fourier_series(sq, 15)
        self.assertAlmostEqual(s.b[0], 4 / np.pi, places=2)
        self.assertAlmostEqual(s.b[1], 0.0, places=2)
        self.assertAlmostEqual(s.b[2], 4 / (3 * np.pi), places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
