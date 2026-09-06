"""Tests for differentiation, integration and special functions."""

import math
import unittest
from fractions import Fraction

from quadrivium import numeric as np

from quadrivium.core.exceptions import DomainError
from quadrivium.diff import *
from quadrivium.integrate import *
from quadrivium.special import *

# int_0^2 e^-t sin(3t) dt
QUAD_REF = (3 - np.exp(-2) * (np.sin(6) + 3 * np.cos(6))) / 10


def integrand(t):
    return np.exp(-t) * np.sin(3 * t)


class TestFiniteDifferences(unittest.TestCase):
    def setUp(self):
        self.f = lambda t: np.exp(np.sin(t))
        self.df = lambda t: np.cos(t) * np.exp(np.sin(t))
        self.d2f = lambda t: (np.cos(t) ** 2 - np.sin(t)) * np.exp(np.sin(t))
        self.x = 1.3

    def test_first_derivative_accuracy_ordering(self):
        e_fwd = abs(forward_difference(self.f, self.x) - self.df(self.x))
        e_ctr = abs(central_difference(self.f, self.x) - self.df(self.x))
        e_five = abs(five_point_stencil(self.f, self.x) - self.df(self.x))
        self.assertLess(e_ctr, e_fwd)
        self.assertLess(e_five, e_ctr)

    def test_richardson_and_complex_step(self):
        self.assertLess(abs(richardson_derivative(self.f, self.x) - self.df(self.x)),
                        1e-11)
        self.assertLess(abs(complex_step_derivative(self.f, self.x) - self.df(self.x)),
                        1e-15)

    def test_second_derivative(self):
        self.assertLess(abs(second_derivative(self.f, self.x) - self.d2f(self.x)), 1e-5)

    def test_fornberg_standard_stencils(self):
        _, w = finite_difference_weights(1, 2, "central")
        np.testing.assert_allclose(w, [-0.5, 0, 0.5], atol=1e-14)
        _, w = finite_difference_weights(2, 2, "central")
        np.testing.assert_allclose(w, [1, -2, 1], atol=1e-14)
        _, w = finite_difference_weights(1, 4, "central")
        np.testing.assert_allclose(w, [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12], atol=1e-14)
        _, w = finite_difference_weights(1, 2, "forward")
        np.testing.assert_allclose(w, [-1.5, 2, -0.5], atol=1e-14)

    def test_differentiation_matrix_fourth_order(self):
        errs = []
        for n in (30, 60, 120):
            xs = np.linspace(0, 2 * np.pi, n)
            D = differentiation_matrix(xs, 1, 5)
            errs.append(np.max(np.abs(D @ np.sin(xs) - np.cos(xs))))
        self.assertGreater(errs[0] / errs[1], 8)      # ~16 for 4th order
        self.assertGreater(errs[1] / errs[2], 8)

    def test_nonuniform_grid(self):
        # deliberately uneven: dense on [0, 1], coarse beyond, so the coarse
        # region sets the error level. Fornberg weights still give fourth order.
        errs = []
        for k in (1, 2, 4):
            xn = np.sort(np.concatenate([np.linspace(0, 1, 20 * k),
                                         np.linspace(1, 2 * np.pi, 40 * k)[1:]]))
            D = differentiation_matrix(xn, 1, 5)
            errs.append(np.max(np.abs(D @ np.sin(xn) - np.cos(xn))))
        self.assertLess(errs[0], 1e-4)
        self.assertGreater(errs[0] / errs[1], 8)     # ~16 for fourth order
        self.assertGreater(errs[1] / errs[2], 8)

    def test_savitzky_golay_beats_raw_gradient(self):
        rng = np.random.default_rng(0)
        t = np.linspace(0, 2 * np.pi, 300)
        y = np.sin(t) + 0.01 * rng.standard_normal(300)
        dt = t[1] - t[0]
        sg = savitzky_golay_derivative(y, 21, 3, 1, dt)
        raw = np.gradient(y, dt)
        e_sg = np.max(np.abs(sg[20:-20] - np.cos(t)[20:-20]))
        e_raw = np.max(np.abs(raw[20:-20] - np.cos(t)[20:-20]))
        self.assertLess(e_sg, e_raw / 3)


class TestAutodiff(unittest.TestCase):
    def test_forward_mode_scalar(self):
        f = lambda t: np.exp(np.sin(3 * t**2 + 1 / t))
        num = (f(1.3 + 1e-6) - f(1.3 - 1e-6)) / 2e-6
        self.assertLess(abs(derivative(f, 1.3) - num), 1e-6)

    def test_reverse_mode_gradient_exact(self):
        g = lambda v: v[0] * v[1] + np.sin(v[0] * v[2]) + v[1] ** 2 / v[2]
        pt = [1.5, 2.0, 0.7]
        exact = np.array([2.0 + 0.7 * math.cos(1.05),
                          1.5 + 4.0 / 0.7,
                          1.5 * math.cos(1.05) - 4.0 / 0.49])
        np.testing.assert_allclose(gradient(g, pt), exact, atol=1e-12)
        np.testing.assert_allclose(forward_gradient(g, pt), exact, atol=1e-12)
        val, grad = value_and_grad(g, pt)
        np.testing.assert_allclose(grad, exact, atol=1e-12)

    def test_jacobian(self):
        F = lambda v: [v[0] * v[1], np.exp(v[0]) + v[1]]
        expected = [[3.0, 2.0], [math.exp(2), 1.0]]
        np.testing.assert_allclose(jacobian(F, [2.0, 3.0]), expected, atol=1e-12)
        np.testing.assert_allclose(forward_jacobian(F, [2.0, 3.0]), expected, atol=1e-12)

    def test_hyperdual_hessian_is_exact(self):
        q = lambda v: v[0] ** 3 * v[1] + v[1] ** 2 + np.sin(v[0] * v[1])
        a, b = 2.0, 3.0
        H = hessian(q, [a, b])
        exact = np.array([
            [6 * a * b - b * b * math.sin(a * b),
             3 * a * a + math.cos(a * b) - a * b * math.sin(a * b)],
            [3 * a * a + math.cos(a * b) - a * b * math.sin(a * b),
             2 - a * a * math.sin(a * b)]])
        np.testing.assert_allclose(H, exact, atol=1e-12)
        np.testing.assert_allclose(hessian_vector_product(q, [a, b], [0.4, -1.2]),
                                   exact @ np.array([0.4, -1.2]), atol=1e-12)

    def test_numpy_ufuncs_dispatch(self):
        self.assertAlmostEqual(np.sin(Dual(1.3, 1.0)).deriv, math.cos(1.3))
        self.assertAlmostEqual(np.exp(Variable(2.0)).value, math.exp(2.0))

    def test_taylor_coefficients(self):
        np.testing.assert_allclose(taylor_coefficients(np.exp, 0.0, 5),
                                   [1, 1, 0.5, 1 / 6, 1 / 24, 1 / 120], atol=1e-8)


class TestSpectralDifferentiation(unittest.TestCase):
    def test_fourier_exact_for_band_limited(self):
        n, L = 32, 2 * np.pi
        xs = np.linspace(0, L, n, endpoint=False)
        g = np.sin(xs) + 0.5 * np.cos(3 * xs)
        dg = np.cos(xs) - 1.5 * np.sin(3 * xs)
        self.assertLess(np.max(np.abs(fourier_derivative(g, L, 1) - dg)), 1e-12)
        self.assertLess(np.max(np.abs(fourier_diff_matrix(n, L, 1) @ g - dg)), 1e-12)

    def test_fourier_odd_length(self):
        n, L = 33, 2 * np.pi
        xs = np.linspace(0, L, n, endpoint=False)
        g = np.sin(xs) + 0.5 * np.cos(3 * xs)
        dg = np.cos(xs) - 1.5 * np.sin(3 * xs)
        self.assertLess(np.max(np.abs(fourier_diff_matrix(n, L, 1) @ g - dg)), 1e-11)

    def test_chebyshev_spectral_convergence(self):
        f = lambda t: np.exp(np.sin(np.pi * t))
        df = lambda t: np.pi * np.cos(np.pi * t) * np.exp(np.sin(np.pi * t))
        errs = []
        for n in (8, 16, 32):
            x, d = chebyshev_derivative(f, n, -1, 1)
            errs.append(np.max(np.abs(d - df(x))))
        self.assertLess(errs[1], errs[0])
        self.assertLess(errs[2], 1e-6)

    def test_chebyshev_matrix_identities(self):
        D, x = chebyshev_diff_matrix(20, -1, 1)
        np.testing.assert_allclose(D @ np.ones_like(x), 0, atol=1e-10)
        np.testing.assert_allclose(D @ x, 1, atol=1e-10)
        np.testing.assert_allclose(D @ (x**2), 2 * x, atol=1e-9)


class TestQuadrature(unittest.TestCase):
    def test_newton_cotes_convergence_orders(self):
        for rule, factor in ((trapezoid_rule, 4), (simpson_rule, 16),
                             (boole_rule, 64)):
            e1 = abs(rule(integrand, 0, 2, 20).value - QUAD_REF)
            e2 = abs(rule(integrand, 0, 2, 40).value - QUAD_REF)
            self.assertGreater(e1 / e2, factor * 0.7)

    def test_newton_cotes_weights(self):
        for n, expected in ((1, [0.5, 0.5]), (2, [1 / 6, 4 / 6, 1 / 6]),
                            (3, [1 / 8, 3 / 8, 3 / 8, 1 / 8]),
                            (4, [7 / 90, 32 / 90, 12 / 90, 32 / 90, 7 / 90])):
            _, w = newton_cotes_weights(n)
            np.testing.assert_allclose(w, expected, atol=1e-13)

    def test_romberg_and_adaptive(self):
        self.assertLess(abs(romberg(integrand, 0, 2).value - QUAD_REF), 1e-12)
        for m in ("simpson", "trapezoid", "gauss_kronrod"):
            v = adaptive_quadrature(integrand, 0, 2, 1e-11, m)
            self.assertLess(abs(v.value - QUAD_REF), 1e-9)

    def test_gauss_kronrod_efficiency(self):
        gk = adaptive_quadrature(integrand, 0, 2, 1e-11, "gauss_kronrod")
        tr = adaptive_quadrature(integrand, 0, 2, 1e-11, "trapezoid")
        self.assertLess(gk.function_calls, tr.function_calls / 100)

    def test_gauss_family(self):
        for rule in (gauss_legendre, gauss_lobatto, gauss_radau):
            self.assertLess(abs(rule(integrand, 0, 2, n=20).value - QUAD_REF), 1e-12)
        self.assertLess(abs(clenshaw_curtis(integrand, 0, 2, 32).value - QUAD_REF), 1e-12)
        for k in (1, 2):
            self.assertLess(abs(fejer(integrand, 0, 2, 40, k).value - QUAD_REF), 1e-10)

    def test_weighted_gauss_rules(self):
        self.assertAlmostEqual(gauss_hermite(lambda t: t**2, 20).value,
                               np.sqrt(np.pi) / 2, places=12)
        self.assertAlmostEqual(gauss_laguerre(lambda t: t**3, 20).value, 6.0, places=9)
        self.assertAlmostEqual(gauss_chebyshev(lambda t: t**2, 20).value,
                               np.pi / 2, places=12)

    def test_tanh_sinh_on_singular_integrands(self):
        cases = [(lambda t: 1 / np.sqrt(t), 0, 1, 2.0),
                 (np.log, 0, 1, -1.0),
                 (lambda t: t ** (-0.9), 0, 1, 10.0),
                 (lambda t: np.sqrt(t) * np.log(1 / t), 0, 1, 4 / 9)]
        for f, a, b, exact in cases:
            with self.subTest(exact=exact):
                self.assertLess(abs(tanh_sinh(f, a, b).value - exact), 1e-11)

    def test_tanh_sinh_offset_form(self):
        v = tanh_sinh(lambda t: 1 / np.sqrt(1 - t * t), -1, 1, levels=12,
                      f_offset=lambda d, side: 1 / np.sqrt(d * (2 - d)))
        self.assertLess(abs(v.value - np.pi), 1e-13)

    def test_improper_integrals(self):
        cases = [(lambda t: np.exp(-t * t), -np.inf, np.inf, np.sqrt(np.pi)),
                 (lambda t: np.exp(-t), 0, np.inf, 1.0),
                 (lambda t: 1 / (1 + t * t), 0, np.inf, np.pi / 2),
                 (lambda t: np.exp(t), -np.inf, 0, 1.0)]
        for f, a, b, exact in cases:
            with self.subTest(exact=exact):
                self.assertLess(abs(quad(f, a, b).value - exact), 1e-8)

    def test_monte_carlo_and_variance_reduction(self):
        mc = monte_carlo(integrand, 0, 2, 200000, rng=0)
        self.assertLess(abs(mc.value - QUAD_REF), 4 * mc.error_estimate)
        st = stratified_sampling(integrand, 0, 2, 200000, 200, rng=0)
        self.assertLess(st.error_estimate, mc.error_estimate / 5)

    def test_qmc_beats_mc(self):
        g = lambda p: np.exp(-np.sum(np.asarray(p) ** 2))
        exact = (math.erf(1) * math.sqrt(math.pi) / 2) ** 4
        lo, hi = np.zeros(4), np.ones(4)
        q = quasi_monte_carlo(g, lo, hi, 4096, "halton")
        m = monte_carlo_nd(g, lo, hi, 4096, rng=0)
        self.assertLess(abs(q.value - exact), abs(m.value - exact))

    def test_multidimensional(self):
        v = double_integral(lambda x, y: x * y * np.exp(-x - y), 0, 1, 0, 1, 20, 20)
        self.assertAlmostEqual(v.value, (1 - 2 * np.exp(-1)) ** 2, places=12)
        v = double_integral(lambda x, y: x * y, 0, 1, 0, lambda x: 1 - x, 20, 20)
        self.assertAlmostEqual(v.value, 1 / 24, places=12)
        v = triple_integral(lambda x, y, z: x * y * z, 0, 1, 0, 1, 0, 1, 8, 8, 8)
        self.assertAlmostEqual(v.value, 0.125, places=12)

    def test_simplex_rules(self):
        tri = [[0, 0], [1, 0], [0, 1]]
        self.assertAlmostEqual(triangle_quadrature(lambda x, y: 1.0, tri, 1).value,
                               0.5, places=14)
        self.assertAlmostEqual(triangle_quadrature(lambda x, y: x * y, tri, 4).value,
                               1 / 24, places=14)
        tet = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertAlmostEqual(
            tetrahedron_quadrature(lambda x, y, z: 1.0, tet, 1).value, 1 / 6, places=14)

    def test_polar_and_spherical(self):
        self.assertAlmostEqual(polar_integral(lambda r, t: 1.0, 0, 2).value,
                               4 * np.pi, places=9)
        self.assertAlmostEqual(spherical_integral(lambda r, t, p: 1.0, 0, 1).value,
                               4 * np.pi / 3, places=7)


class TestSpecialFunctions(unittest.TestCase):
    def test_gamma_and_beta(self):
        self.assertAlmostEqual(gamma(5), 24.0, places=9)
        self.assertAlmostEqual(gamma(0.5), math.sqrt(math.pi), places=12)
        self.assertAlmostEqual(gamma(-0.5), -2 * math.sqrt(math.pi), places=10)
        self.assertAlmostEqual(log_gamma(10), math.lgamma(10), places=12)
        self.assertAlmostEqual(beta(2, 3), 1 / 12, places=12)

    def test_digamma_full_precision(self):
        for x, expected in ((1, -0.5772156649015329), (0.5, -1.9635100260214235),
                            (10, 2.2517525890667211)):
            self.assertAlmostEqual(digamma(x), expected, places=14)

    def test_error_functions(self):
        self.assertAlmostEqual(erf(1), math.erf(1), places=15)
        self.assertAlmostEqual(erfinv(math.erf(0.7)), 0.7, places=12)

    def test_incomplete_gamma_complementarity(self):
        self.assertAlmostEqual(regularized_gamma_p(2.5, 4)
                               + regularized_gamma_q(2.5, 4), 1.0, places=14)

    def test_bessel_against_exact_series(self):
        def exact(n, x, terms=120):
            xf = Fraction(x).limit_denominator(10**9)
            s = Fraction(0)
            for k in range(terms):
                s += (Fraction((-1) ** k) * (xf / 2) ** (2 * k + n)
                      / (math.factorial(k) * math.factorial(k + n)))
            return float(s)

        worst = 0.0
        for n in range(8):
            for x in (0.5, 1.0, 2.5, 5.0, 7.0, 10.0, 15.0):
                v = bessel_jn(n, x) if n > 1 else (bessel_j0(x) if n == 0
                                                   else bessel_j1(x))
                worst = max(worst, abs(v - exact(n, x)))
        self.assertLess(worst, 1e-9)

    def test_bessel_wronskian(self):
        for x in (0.5, 1.0, 3.0, 7.0, 12.0):
            w = bessel_j1(x) * bessel_y0(x) - bessel_j0(x) * bessel_y1(x)
            self.assertAlmostEqual(w, 2 / (math.pi * x), places=7)

    def test_modified_bessel_wronskian(self):
        for x in (0.5, 1.0, 2.0, 3.0):
            w = bessel_i0(x) * bessel_k1(x) + bessel_i1(x) * bessel_k0(x)
            self.assertAlmostEqual(w, 1 / x, places=6)

    def test_elliptic_integrals(self):
        self.assertAlmostEqual(elliptic_k(0.5), 1.8540746773013719, places=14)
        self.assertAlmostEqual(elliptic_e(0.5), 1.3506438810476755, places=14)
        self.assertAlmostEqual(elliptic_k(0.0), math.pi / 2, places=14)
        self.assertAlmostEqual(elliptic_e(1.0), 1.0, places=14)

    def test_zeta_and_integrals(self):
        self.assertAlmostEqual(zeta(2), math.pi**2 / 6, places=12)
        self.assertAlmostEqual(zeta(4), math.pi**4 / 90, places=12)
        self.assertAlmostEqual(zeta(3), 1.2020569031595943, places=12)
        self.assertAlmostEqual(exponential_integral(1), 1.8951178163559368, places=10)
        self.assertAlmostEqual(sine_integral(1), 0.9460830703671830, places=12)
        self.assertAlmostEqual(cosine_integral(1), 0.3374039229009681, places=12)

    def test_airy(self):
        self.assertAlmostEqual(airy_ai(0), 0.35502805388781723, places=12)
        self.assertAlmostEqual(airy_bi(0), 0.6149266274460007, places=12)
        self.assertAlmostEqual(airy_ai(1), 0.13529241631288141, places=9)

    def test_zeta_off_the_convergence_half_plane(self):
        # The Euler-Maclaurin sum only converges for s > 1; the critical strip
        # comes from Borwein's alternating series and s < 0 from the
        # functional equation.
        self.assertAlmostEqual(zeta(0.5), -1.4603545088095868, places=12)
        self.assertAlmostEqual(zeta(0.0), -0.5, places=14)
        self.assertAlmostEqual(zeta(-1.0), -1.0 / 12.0, places=14)
        self.assertAlmostEqual(zeta(-3.0), 1.0 / 120.0, places=14)
        for s in (-2.0, -4.0, -6.0):
            self.assertEqual(zeta(s), 0.0)          # trivial zeros

    def test_zeta_satisfies_its_functional_equation(self):
        for s in (0.01, 0.25, 0.6, 0.99, 2.3, -0.5, -7.3):
            lhs = zeta(s)
            rhs = (2.0**s * math.pi ** (s - 1.0) * math.sin(0.5 * math.pi * s)
                   * gamma(1.0 - s) * zeta(1.0 - s))
            self.assertAlmostEqual(lhs / rhs, 1.0, places=12)

    def test_domain_errors(self):
        with self.assertRaises(DomainError):
            elliptic_k(1.5)
        with self.assertRaises(DomainError):
            zeta(1.0)               # the pole


if __name__ == "__main__":
    unittest.main(verbosity=2)
