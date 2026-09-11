"""Contracts and allocation bounds for the C direct-factorisation kernels."""
import tracemalloc
import unittest

from quadrivium import _accel, numeric as np
from quadrivium.core.exceptions import DimensionError, SingularMatrixError
from quadrivium.linalg import determinant, householder_qr, plu_solve, qr_solve


@unittest.skipUnless(_accel.available(), "compiled backend not built")
class TestCDirect(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1729)

    def kernel(self, name):
        return _accel.kernel(name)

    def test_factorizations_cross_every_block_boundary(self):
        for n in (17, 33, 65, 129, 193):
            with self.subTest(n=n):
                a = self.rng.standard_normal((n, n))
                saved = a.copy()
                perm, lu = self.kernel("plu")(a=a)
                lower = np.tril(lu, -1) + np.eye(n)
                upper = np.triu(lu)
                np.testing.assert_allclose(lower @ upper, a[perm], atol=2e-11)
                np.testing.assert_array_equal(a, saved)
                spd = a @ a.T + n * np.eye(n)
                lower = self.kernel("cholesky")(a=spd)
                np.testing.assert_allclose(lower @ lower.T, spd, atol=2e-11)

    def test_reduced_full_and_r_only_qr(self):
        for m, n in ((1, 1), (8, 4), (4, 8), (132, 67), (67, 132)):
            with self.subTest(m=m, n=n):
                a = self.rng.standard_normal((m, n))
                saved = a.copy()
                q, r = self.kernel("householder_qr")(a=a)
                thin_q, thin_r = self.kernel("householder_qr")(a=a, reduced=True)
                no_q, r_only = self.kernel("householder_qr")(a=a, want_q=False)
                k = min(m, n)
                self.assertEqual(thin_q.shape, (m, k))
                self.assertEqual(thin_r.shape, (k, n))
                self.assertEqual(no_q.shape, (0, m))
                np.testing.assert_allclose(q @ r, a, atol=2e-12)
                np.testing.assert_allclose(thin_q @ thin_r, a, atol=2e-12)
                np.testing.assert_allclose(thin_q.T @ thin_q, np.eye(k), atol=2e-12)
                np.testing.assert_allclose(thin_q, q[:, :k], atol=2e-12)
                np.testing.assert_allclose(r_only, r, atol=2e-12)
                np.testing.assert_array_equal(a, saved)

    def test_tall_reduced_qr_memory_scales_with_input(self):
        a = self.rng.standard_normal((4096, 4))
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            q, r = householder_qr(a)
            peak = tracemalloc.get_traced_memory()[1] - before
        finally:
            if not already_tracing:
                tracemalloc.stop()
        self.assertLess(peak, 4 * 1024 * 1024)
        self.assertEqual(q.shape, (4096, 4))
        self.assertEqual(r.shape, (4, 4))
        np.testing.assert_allclose(q @ r, a, atol=2e-12)

    def test_zero_columns_and_empty_shapes(self):
        for shape in ((0, 0), (0, 4), (4, 0), (6, 3)):
            a = np.zeros(shape)
            q, r = self.kernel("householder_qr")(a)
            np.testing.assert_allclose(q @ r, a, atol=0)
            q, r = self.kernel("givens_qr")(a)
            np.testing.assert_allclose(q @ r, a, atol=0)
        self.assertEqual(self.kernel("cholesky")(np.zeros((0, 0))).shape, (0, 0))
        self.assertEqual(self.kernel("qr_least_squares")(np.zeros((4, 0)), np.ones(4)).shape, (0,))
        w, v, sweeps = self.kernel("svd_jacobi")(np.zeros((4, 3)), tol=0)
        self.assertEqual(sweeps, 1)
        np.testing.assert_array_equal(w, np.zeros((4, 3)))
        np.testing.assert_array_equal(v, np.eye(3))

    def test_noncontiguous_inputs_and_transposed_triangular_solve(self):
        a = self.rng.standard_normal((16, 16))[::2, ::2]
        spd = a @ a.T + 8 * np.eye(8)
        l = self.kernel("cholesky")(spd)
        b = self.rng.standard_normal(16)[::2]
        saved = b.copy()
        x = self.kernel("forward_substitution")(l=l, b=b)
        np.testing.assert_allclose(l @ x, b, atol=1e-12)
        x = self.kernel("back_substitution")(u=l, b=b, transposed=True)
        np.testing.assert_allclose(l.T @ x, b, atol=1e-12)
        x = self.kernel("back_substitution")(u=l.T, b=b)
        np.testing.assert_allclose(l.T @ x, b, atol=1e-12)
        np.testing.assert_array_equal(b, saved)
        q, r = self.kernel("householder_qr")(a)
        np.testing.assert_allclose(q @ r, a, atol=1e-12)

    def test_rhs_conversion_cannot_invalidate_captured_matrix_shape(self):
        for name in ("forward_substitution", "back_substitution"):
            with self.subTest(name=name):
                a = np.eye(2)

                class ReshapingScalar:
                    def __float__(self):
                        a.shape = (4,)
                        return 1.0

                result = self.kernel(name)(a, [ReshapingScalar(), 2.0])
                self.assertEqual(a.shape, (4,))
                np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_least_squares_residual_and_svd_reconstruction(self):
        a = self.rng.standard_normal((80, 9))
        b = self.rng.standard_normal(80)
        saved_a, saved_b = a.copy(), b.copy()
        x = self.kernel("qr_least_squares")(a=a, b=b)
        np.testing.assert_allclose(a.T @ (a @ x - b), np.zeros(9), atol=1e-11)
        w, v, sweeps = self.kernel("svd_jacobi")(a=a)
        np.testing.assert_allclose(w @ v.T, a, atol=2e-12)
        np.testing.assert_allclose(v.T @ v, np.eye(9), atol=2e-12)
        self.assertLess(sweeps, 60)
        np.testing.assert_array_equal(a, saved_a)
        np.testing.assert_array_equal(b, saved_b)

    def test_svd_layout_roundtrip_is_exact_without_sweeps(self):
        for m, n in ((17, 3), (9, 7), (10, 6), (10, 2), (11, 1), (4, 0), (0, 0)):
            with self.subTest(m=m, n=n):
                a = self.rng.standard_normal((m, n))
                if m and n:
                    a[0, 0] = float("nan")
                    a[-1, -1] = float("inf")
                w, v, sweeps = self.kernel("svd_jacobi")(a, max_sweeps=0)
                self.assertEqual(sweeps, 0)
                self.assertTrue(w.flags.c_contiguous)
                np.testing.assert_array_equal(w, a)
                np.testing.assert_array_equal(v, np.eye(n))

    def test_finite_extreme_scales_reconstruct_without_squared_norm_overflow(self):
        base = self.rng.standard_normal((12, 5))
        expected = self.rng.standard_normal(5)
        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale):
                a = base * scale
                q, r = self.kernel("householder_qr")(a, reduced=True)
                np.testing.assert_allclose(q @ (r / scale), base, atol=1e-12)
                np.testing.assert_allclose(q.T @ q, np.eye(5), atol=1e-12)
                actual = self.kernel("qr_least_squares")(a, (base @ expected) * scale)
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_keyword_validation_and_tagged_errors(self):
        with self.assertRaisesRegex(RuntimeError, "^notpd:"):
            self.kernel("cholesky")(a=np.array([[1., 2.], [2., 1.]]))
        for name in ("forward_substitution", "back_substitution"):
            with self.assertRaisesRegex(RuntimeError, "^singular:"):
                self.kernel(name)(np.zeros((2, 2)), np.ones(2))
            with self.assertRaisesRegex(ValueError, "rhs must match"):
                self.kernel(name)(np.eye(2), np.ones(3))
        with self.assertRaisesRegex(ValueError, "square"):
            self.kernel("plu")(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "at least as many rows"):
            self.kernel("qr_least_squares")(np.ones((2, 3)), np.ones(2))
        with self.assertRaises(OverflowError):
            self.kernel("svd_jacobi")(np.eye(2), max_sweeps=-1)

    def test_symmetry_handles_relative_tolerance_and_nonfinite_values(self):
        symmetry = self.kernel("is_symmetric")
        self.assertTrue(symmetry(a=np.array([[1., 1e6], [1e6 + 1., 2.]])))
        self.assertFalse(symmetry(np.array([[1., 2.], [2.01, 1.]])))
        self.assertFalse(symmetry(np.array([[float("nan")]])))
        self.assertTrue(symmetry(np.array([[1., float("inf")], [float("inf"), 1.]])))
        self.assertFalse(symmetry(np.array([[1., float("inf")], [1., 1.]])))
        self.assertFalse(symmetry(np.ones((3, 2))))

    def test_packed_plu_solve_and_determinant_preserve_pivot_signs(self):
        for perm in ([0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 3, 0], [2, 0, 1, 3]):
            with self.subTest(perm=perm):
                a = np.diag(np.array([2., 3., 5., 7.]))[perm]
                b = self.rng.standard_normal(4)
                saved_a, saved_b = a.copy(), b.copy()
                with _accel.disabled():
                    slow_x, slow_det = plu_solve(a, b), determinant(a)
                x = plu_solve(a, b)
                np.testing.assert_allclose(a @ x, b, atol=1e-12)
                np.testing.assert_allclose(x, slow_x, atol=1e-12)
                self.assertEqual(determinant(a), slow_det)
                np.testing.assert_array_equal(a, saved_a)
                np.testing.assert_array_equal(b, saved_b)
        for scale in (0., 1e-301):
            with self.assertRaisesRegex(SingularMatrixError, "singular to working precision"):
                plu_solve(np.diag(np.array([1., scale])), np.ones(2))
        self.assertEqual(determinant(np.array([[1., 2.], [2., 4.]])), 0.)
        self.assertEqual(determinant(np.zeros((0, 0))), 1.)
        self.assertEqual(plu_solve(np.zeros((0, 0)), np.zeros(0)).shape, (0,))
        with self.assertRaises(DimensionError):
            plu_solve(np.eye(3), np.ones(2))

    def test_compact_qr_solve_matches_factorization(self):
        for m, n in ((0, 0), (5, 0), (8, 8), (80, 9)):
            with self.subTest(m=m, n=n):
                a = self.rng.standard_normal((m, n))
                b = self.rng.standard_normal(m)
                with _accel.disabled():
                    slow = qr_solve(a, b)
                fast = qr_solve(a, b)
                np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=2e-12)
                np.testing.assert_allclose(a.T @ (a @ fast - b), np.zeros(n), atol=2e-11)
        with self.assertRaises(SingularMatrixError):
            qr_solve(np.zeros((3, 2)), np.ones(3))
        with self.assertRaises(DimensionError):
            qr_solve(np.ones((3, 2)), np.ones(2))

    def test_solve_and_determinant_avoid_dense_factor_copies(self):
        a = self.rng.standard_normal((256, 256)) + 256 * np.eye(256)
        b = self.rng.standard_normal(256)
        tall = self.rng.standard_normal((2048, 8))
        rhs = self.rng.standard_normal(2048)
        for name, operation, limit in (
            ("plu_solve", lambda: plu_solve(a, b), 2 * a.nbytes),
            ("determinant", lambda: determinant(a), 2 * a.nbytes),
            ("qr_solve", lambda: qr_solve(tall, rhs), 3 * tall.nbytes),
        ):
            with self.subTest(name=name):
                already_tracing = tracemalloc.is_tracing()
                if not already_tracing:
                    tracemalloc.start()
                try:
                    tracemalloc.reset_peak()
                    before = tracemalloc.get_traced_memory()[0]
                    operation()
                    peak = tracemalloc.get_traced_memory()[1] - before
                finally:
                    if not already_tracing:
                        tracemalloc.stop()
                self.assertLess(peak, limit)


if __name__ == "__main__":
    unittest.main()
