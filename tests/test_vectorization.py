"""Array-native behaviour, scale invariance, and batched solves.

These pin down the properties the optimization pass introduced, each of which
is a contract a caller can rely on rather than an implementation detail:

* every special function evaluates an array in one call, and agrees elementwise
  with the scalar spelling;
* the eigenvalue iterations converge on a matrix of any magnitude, and report
  the same answer for a scaled copy of the same problem;
* barycentric interpolation does not depend on the affine placement of its
  nodes;
* a tridiagonal solve accepts many right-hand sides without consuming them;
* the low-discrepancy sequences match the scalar recurrences they replaced.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import quadrivium as qd
from quadrivium import _accel, special
from quadrivium.core.exceptions import SingularMatrixError

# (name, extra leading arguments, sample points). The array call passes the
# points as one array; the scalar calls pass them one at a time.
ARRAY_CASES = [
    ("log_gamma", (), np.linspace(0.3, 12.0, 9)),
    ("gamma", (), np.linspace(0.3, 9.0, 9)),
    ("digamma", (), np.linspace(0.3, 12.0, 9)),
    ("trigamma", (), np.linspace(0.4, 9.0, 9)),
    ("polygamma", (2,), np.linspace(0.4, 9.0, 9)),
    ("factorial", (), np.arange(0.0, 9.0)),
    ("binomial", (12.0,), np.arange(0.0, 7.0)),
    ("beta", (2.5,), np.linspace(0.4, 6.0, 9)),
    ("log_beta", (2.5,), np.linspace(0.4, 6.0, 9)),
    ("erf", (), np.linspace(-3.0, 3.0, 9)),
    ("erfc", (), np.linspace(-3.0, 3.0, 9)),
    ("erfinv", (), np.linspace(-0.9, 0.9, 9)),
    ("erfcx", (), np.linspace(-3.0, 8.0, 9)),
    ("regularized_gamma_p", (2.0,), np.linspace(0.1, 9.0, 9)),
    ("regularized_gamma_q", (2.0,), np.linspace(0.1, 9.0, 9)),
    ("incomplete_gamma_lower", (2.0,), np.linspace(0.1, 9.0, 9)),
    ("incomplete_gamma_upper", (2.0,), np.linspace(0.1, 9.0, 9)),
    ("incomplete_beta", (2.0, 3.0), np.linspace(0.05, 0.95, 9)),
    ("bessel_j0", (), np.linspace(0.1, 25.0, 9)),
    ("bessel_j1", (), np.linspace(0.1, 25.0, 9)),
    ("bessel_jn", (3,), np.linspace(0.1, 25.0, 9)),
    ("bessel_y0", (), np.linspace(0.2, 25.0, 9)),
    ("bessel_y1", (), np.linspace(0.2, 25.0, 9)),
    ("bessel_yn", (3,), np.linspace(0.2, 25.0, 9)),
    ("bessel_i0", (), np.linspace(-20.0, 20.0, 9)),
    ("bessel_i1", (), np.linspace(-20.0, 20.0, 9)),
    ("bessel_in", (3,), np.linspace(0.2, 18.0, 9)),
    ("bessel_k0", (), np.linspace(0.05, 20.0, 9)),
    ("bessel_k1", (), np.linspace(0.05, 20.0, 9)),
    ("bessel_kn", (3,), np.linspace(0.05, 20.0, 9)),
    ("spherical_bessel_j", (3,), np.linspace(0.2, 20.0, 9)),
    ("spherical_bessel_y", (3,), np.linspace(0.2, 20.0, 9)),
    ("airy_ai", (), np.linspace(-4.0, 4.0, 9)),
    ("airy_bi", (), np.linspace(-4.0, 4.0, 9)),
    ("elliptic_k", (), np.linspace(-3.0, 0.95, 9)),
    ("elliptic_e", (), np.linspace(-3.0, 1.0, 9)),
    ("exponential_integral", (), np.linspace(0.1, 30.0, 9)),
    ("sine_integral", (), np.linspace(-15.0, 15.0, 9)),
    ("cosine_integral", (), np.linspace(0.1, 15.0, 9)),
    ("zeta", (), np.linspace(1.4, 12.0, 9)),
    ("lambert_w", (), np.linspace(-0.3, 30.0, 9)),
    ("dawson", (), np.linspace(-12.0, 12.0, 9)),
    ("fresnel_s", (), np.linspace(-8.0, 8.0, 9)),
    ("fresnel_c", (), np.linspace(-8.0, 8.0, 9)),
    ("associated_legendre", (5, 2), np.linspace(-0.95, 0.95, 9)),
    ("hyp1f1", (0.5, 1.5), np.linspace(-15.0, 15.0, 9)),
    ("hyp2f1", (0.5, 1.0, 1.5), np.linspace(-0.9, 0.9, 9)),
    ("expint_n", (2,), np.linspace(0.05, 20.0, 9)),
    ("struve_h0", (), np.linspace(-9.0, 9.0, 9)),
    ("logistic", (), np.linspace(-6.0, 6.0, 9)),
    ("logit", (), np.linspace(0.05, 0.95, 9)),
]


class TestSpecialFunctionsAreArrayNative(unittest.TestCase):
    """Each function takes a whole array and matches the scalar call."""

    def test_array_call_matches_elementwise(self):
        for name, head, xs in ARRAY_CASES:
            with self.subTest(function=name):
                fn = getattr(special, name)
                out = np.asarray(fn(*head, xs))
                self.assertEqual(out.shape, xs.shape)
                one_at_a_time = np.array([fn(*head, float(v)) for v in xs])
                np.testing.assert_allclose(out, one_at_a_time, rtol=1e-12,
                                           atol=1e-300)

    def test_scalar_input_returns_a_python_scalar(self):
        for name, head, xs in ARRAY_CASES:
            with self.subTest(function=name):
                got = getattr(special, name)(*head, float(xs[len(xs) // 2]))
                self.assertNotIsInstance(got, np.ndarray)
                self.assertIsInstance(got, (float, complex, int))

    def test_shape_is_preserved(self):
        block = np.linspace(0.5, 6.0, 6).reshape(2, 3)
        for name in ("gamma", "digamma", "erf", "bessel_j0", "lambert_w",
                     "dawson", "airy_ai", "logistic"):
            with self.subTest(function=name):
                out = getattr(special, name)(block)
                self.assertEqual(np.shape(out), (2, 3))

    def test_empty_input_gives_empty_output(self):
        empty = np.zeros(0)
        for name in ("gamma", "digamma", "erf", "bessel_j0", "lambert_w"):
            with self.subTest(function=name):
                self.assertEqual(np.size(getattr(special, name)(empty)), 0)

    def test_spherical_harmonic_takes_angle_arrays(self):
        theta = np.linspace(0.2, 2.9, 5)
        phi = np.linspace(0.0, 6.0, 5)
        out = np.asarray(special.spherical_harmonic(4, 2, theta, phi))
        self.assertEqual(out.shape, theta.shape)
        for i, (t, p) in enumerate(zip(theta, phi)):
            self.assertAlmostEqual(out[i], special.spherical_harmonic(4, 2, t, p),
                                   places=13)

    def test_overflow_returns_infinity_rather_than_raising(self):
        # Past the double range the answer is +inf; raising there would make a
        # single bad element poison a whole array.
        self.assertTrue(math.isinf(special.gamma(200.0)))
        self.assertTrue(np.isinf(np.asarray(special.gamma([2.0, 200.0]))[1]))
        self.assertTrue(math.isinf(special.factorial(400.0)))


class TestSpecialFunctionAccuracy(unittest.TestCase):
    """Identities the vectorized forms must still satisfy."""

    def test_sine_and_cosine_integral_derivatives(self):
        # Si'(x) = sin x / x and Ci'(x) = cos x / x, checked across the
        # crossover between the ascending series and the continued fraction.
        h = 1e-5
        for x in (1.0, 3.5, 4.0, 4.5, 8.0, 20.0):
            with self.subTest(x=x):
                dsi = (special.sine_integral(x + h) - special.sine_integral(x - h)) / (2 * h)
                dci = (special.cosine_integral(x + h) - special.cosine_integral(x - h)) / (2 * h)
                self.assertAlmostEqual(dsi, math.sin(x) / x, places=8)
                self.assertAlmostEqual(dci, math.cos(x) / x, places=8)

    def test_sine_integral_is_odd_and_matches_its_asymptotic_form(self):
        self.assertAlmostEqual(special.sine_integral(-7.0),
                               -special.sine_integral(7.0), places=15)
        # Si oscillates about pi/2 with amplitude ~1/x rather than settling on
        # it, so the check is against the asymptotic expansion.
        for x in (100.0, 400.0, 1000.0):
            with self.subTest(x=x):
                asymptotic = (0.5 * math.pi
                              - math.cos(x) / x * (1 - 2 / x**2 + 24 / x**4)
                              - math.sin(x) / x**2 * (1 - 6 / x**2 + 120 / x**4))
                self.assertAlmostEqual(special.sine_integral(x), asymptotic,
                                       places=10)

    def test_large_argument_accuracy_beats_the_series(self):
        # Si(20) and Ci(20) to 15 digits; the ascending series alone loses
        # about nine of them to cancellation at this argument.
        self.assertAlmostEqual(special.sine_integral(20.0), 1.5482417010434398,
                               places=14)
        self.assertAlmostEqual(special.cosine_integral(20.0), 0.04441982084535316,
                               places=15)

    def test_elliptic_e_agrees_elementwise_with_the_scalar_call(self):
        # Each argument converges after a different number of AGM steps; an
        # element carried past its own convergence would keep accumulating a
        # doubling weight.
        m = np.linspace(-2.0, 0.99, 25)
        np.testing.assert_allclose(special.elliptic_e(m),
                                   [special.elliptic_e(v) for v in m], rtol=1e-14)

    def test_expint_n_of_order_one_is_defined(self):
        # 1/(n-1) must not be evaluated for n == 1, even when no element needs it.
        self.assertAlmostEqual(special.expint_n(1, 1.0), 0.21938393439552029,
                               places=13)
        np.testing.assert_allclose(
            special.expint_n(1, np.array([0.5, 1.0, 2.0])),
            [special.expint_n(1, v) for v in (0.5, 1.0, 2.0)], rtol=1e-13)


class TestEigenvalueIterationsAreScaleInvariant(unittest.TestCase):
    """A scaled copy of a problem is the same problem."""

    @staticmethod
    def _symmetric(n, seed=0):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        return q @ np.diag(np.linspace(1.0, 20.0, n)) @ q.T

    def test_jacobi_converges_at_every_magnitude(self):
        base = self._symmetric(30)
        for scale in (1e-6, 1.0, 1e3, 1e8):
            with self.subTest(scale=scale):
                res = qd.jacobi_eigen(scale * base)
                self.assertTrue(res.converged)
                residual = (scale * base) @ res.eigenvectors - res.eigenvectors * res.eigenvalues
                self.assertLess(np.max(np.abs(residual)) / scale, 1e-12)

    def test_jacobi_eigenvalues_scale_with_the_matrix(self):
        base = self._symmetric(24)
        plain = np.sort(qd.jacobi_eigen(base).eigenvalues)
        scaled = np.sort(qd.jacobi_eigen(1e5 * base).eigenvalues) / 1e5
        np.testing.assert_allclose(scaled, plain, rtol=1e-12)

    def test_qr_algorithm_converges_at_every_magnitude(self):
        base = self._symmetric(20, seed=3)
        counts = []
        for scale in (1e-4, 1.0, 1e4):
            with self.subTest(scale=scale):
                res = qd.qr_algorithm(scale * base)
                self.assertTrue(res.converged)
                counts.append(res.iterations)
                np.testing.assert_allclose(
                    np.sort(res.eigenvalues) / scale,
                    np.sort(np.linalg.eigvalsh(base)), rtol=1e-8)
        # The iteration is driven by eigenvalue ratios, which scaling leaves
        # alone, so the work must not depend on the magnitude either.
        self.assertEqual(len(set(counts)), 1)

    def test_lobpcg_returns_matched_pairs_when_it_runs_out_of_iterations(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal((40, 40))
        s = a @ a.T + 40 * np.eye(40)
        res = qd.linalg.lobpcg(s, k=2, max_iter=6)
        self.assertFalse(res.converged)
        self.assertTrue(np.all(np.isfinite(res.eigenvalues)))
        # Whatever it reached, the values must belong to the vectors beside
        # them: the Rayleigh quotient of each returned vector is its value.
        for j in range(res.eigenvectors.shape[1]):
            v = res.eigenvectors[:, j]
            self.assertAlmostEqual(float(v @ s @ v) / float(v @ v),
                                   float(res.eigenvalues[j]), places=8)


class TestBarycentricIsScaleInvariant(unittest.TestCase):
    def test_same_problem_on_differently_placed_nodes(self):
        n = 200
        cheb = np.cos(np.pi * np.arange(n) / (n - 1))
        for lo, hi in ((-1.0, 1.0), (0.0, 1.0), (0.0, 0.01), (100.0, 100.5)):
            with self.subTest(interval=(lo, hi)):
                nodes = 0.5 * (cheb + 1.0) * (hi - lo) + lo
                p = qd.barycentric(nodes, np.exp(nodes - lo))
                query = np.linspace(lo, hi, 200)
                err = np.max(np.abs(p(query) - np.exp(query - lo)))
                self.assertLess(err, 1e-12)

    def test_weights_stay_finite_for_many_nodes(self):
        for n in (50, 200, 500):
            with self.subTest(n=n):
                nodes = np.linspace(0.0, 1.0, n)
                self.assertTrue(np.all(np.isfinite(
                    qd.interpolate.barycentric_weights(nodes))))

    def test_evaluation_at_a_node_returns_that_value(self):
        x = np.linspace(0.0, 1.0, 40)
        y = np.sin(5 * x)
        p = qd.barycentric(x, y)
        np.testing.assert_allclose(p(x), y, rtol=1e-12)
        self.assertAlmostEqual(float(p(x[7])), float(y[7]), places=15)


class TestTridiagonalBlockSolve(unittest.TestCase):
    def _system(self, n, m, seed=0):
        rng = np.random.default_rng(seed)
        diag = rng.uniform(3.0, 4.0, n)
        sub = rng.uniform(-1.0, 1.0, max(n - 1, 0))
        sup = rng.uniform(-1.0, 1.0, max(n - 1, 0))
        return sub, diag, sup, rng.standard_normal((n, m))

    def test_columns_match_single_right_hand_sides(self):
        for n, m in ((1, 1), (2, 3), (9, 4), (64, 7)):
            with self.subTest(n=n, m=m):
                sub, diag, sup, rhs = self._system(n, m)
                block = qd.linalg.thomas(sub, diag, sup, rhs)
                self.assertEqual(block.shape, rhs.shape)
                for k in range(m):
                    np.testing.assert_allclose(
                        block[:, k], qd.linalg.thomas(sub, diag, sup, rhs[:, k].copy()),
                        rtol=1e-12, atol=1e-14)

    def test_both_backends_agree(self):
        sub, diag, sup, rhs = self._system(40, 6, seed=2)
        with _accel.disabled():
            slow = qd.linalg.thomas(sub, diag, sup, rhs)
        np.testing.assert_allclose(qd.linalg.thomas(sub, diag, sup, rhs), slow,
                                   rtol=1e-13, atol=1e-15)

    def test_right_hand_side_is_not_consumed(self):
        # The kernel solves in place; the caller's array must survive, since
        # an alternating-direction step reuses it.
        sub, diag, sup, rhs = self._system(12, 3, seed=4)
        keep = rhs.copy()
        qd.linalg.thomas(sub, diag, sup, rhs)
        np.testing.assert_array_equal(rhs, keep)

    def test_noncontiguous_right_hand_side(self):
        sub, diag, sup, rhs = self._system(20, 4, seed=6)
        strided = np.repeat(rhs, 2, axis=1)[:, ::2]
        fortran = np.asfortranarray(rhs)
        expected = qd.linalg.thomas(sub, diag, sup, rhs)
        np.testing.assert_allclose(qd.linalg.thomas(sub, diag, sup, strided),
                                   expected, rtol=1e-14)
        np.testing.assert_allclose(qd.linalg.thomas(sub, diag, sup, fortran),
                                   expected, rtol=1e-14)

    def test_zero_pivot_raises(self):
        sub, sup = np.array([1.0, 1.0]), np.array([1.0, 1.0])
        diag, rhs = np.array([0.0, 1.0, 1.0]), np.ones((3, 2))
        self.assertRaises(SingularMatrixError, qd.linalg.thomas, sub, diag, sup, rhs)
        with _accel.disabled():
            self.assertRaises(SingularMatrixError, qd.linalg.thomas,
                              sub, diag, sup, rhs)


class TestLowDiscrepancySequences(unittest.TestCase):
    """The blocked forms must reproduce the scalar recurrences exactly."""

    @staticmethod
    def _radical_inverse(n, base):
        q, weight, k = 0.0, 1.0 / base, n
        while k:
            k, rem = divmod(k, base)
            q += rem * weight
            weight /= base
        return q

    def test_van_der_corput_matches_the_scalar_recurrence(self):
        for base in (2, 3, 7):
            with self.subTest(base=base):
                got = qd.stochastic.van_der_corput(200, base)
                want = [self._radical_inverse(i + 1, base) for i in range(200)]
                np.testing.assert_array_equal(got, want)

    def test_halton_matches_the_scalar_recurrence(self):
        primes = (2, 3, 5, 7)
        got = np.atleast_2d(qd.stochastic.halton(64, len(primes)))
        for j, base in enumerate(primes):
            with self.subTest(base=base):
                want = [self._radical_inverse(i + 1, base) for i in range(64)]
                np.testing.assert_array_equal(got[:, j], want)

    def test_sobol_is_a_running_exclusive_or(self):
        # Consecutive points differ by exactly one direction number, which is
        # the property the cumulative form relies on.
        pts = np.atleast_2d(qd.stochastic.sobol(256, 3))
        self.assertEqual(pts.shape, (256, 3))
        self.assertTrue(np.all((pts >= 0.0) & (pts < 1.0)))
        scaled = np.rint(pts * (1 << 10)).astype(np.int64)
        for j in range(3):
            changes = {int(a) ^ int(b) for a, b in zip(scaled[:-1, j], scaled[1:, j])}
            self.assertLessEqual(len(changes), 12)

    def test_short_and_single_point_sequences(self):
        for n in (0, 1, 2, 3):
            with self.subTest(n=n):
                self.assertEqual(np.atleast_2d(qd.stochastic.sobol(n, 2)).shape[0],
                                 max(n, 0) if n else 0)
                self.assertEqual(np.size(qd.stochastic.halton(n, 1)), n)


@unittest.skipUnless(_accel.available(), "compiled backend not built")
class TestNewKernelsMatchPython(unittest.TestCase):
    def test_givens_qr(self):
        rng = np.random.default_rng(11)
        for shape in ((5, 3), (12, 12), (20, 8), (1, 1), (6, 1)):
            with self.subTest(shape=shape):
                a = rng.standard_normal(shape)
                with _accel.disabled():
                    q_py, r_py = qd.linalg.givens_qr(a)
                q, r = qd.linalg.givens_qr(a)
                np.testing.assert_allclose(q, q_py, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(r, r_py, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(q @ r, a, rtol=1e-11, atol=1e-13)

    def test_svd_jacobi(self):
        rng = np.random.default_rng(12)
        for shape in ((4, 3), (10, 10), (30, 12), (12, 30), (1, 1)):
            with self.subTest(shape=shape):
                a = rng.standard_normal(shape)
                with _accel.disabled():
                    _, s_py, _ = qd.svd_jacobi(a)
                u, s, vt = qd.svd_jacobi(a)
                np.testing.assert_allclose(s, s_py, rtol=1e-10, atol=1e-13)
                np.testing.assert_allclose(u @ np.diag(s) @ vt, a,
                                           rtol=1e-10, atol=1e-12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
