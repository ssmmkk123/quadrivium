"""Exercise the native boundary directly, including layouts wrappers may copy."""

from __future__ import annotations

import unittest

from quadrivium import numeric as np

from quadrivium import _accel

rs = _accel._rs


def layouts(array):
    """Every layout the array core can hand to a compiled kernel.

    Its arrays are always element-aligned -- the allocator returns aligned
    storage and views only ever offset by whole elements -- so the byte-offset
    buffers a foreign array library can build are not representable. What is
    left is what the native boundary has to get right anyway: column-major
    order, negative strides, gapped views, and read-only inputs.
    """
    rows, cols = array.shape
    gapped = np.empty((rows * 2, cols * 3))
    gapped[::2, ::3] = array
    readonly = array.copy()
    readonly.flags.writeable = False
    return (array.copy(), np.asfortranarray(array), array.T.copy().T,
            array[::-1, ::-1].copy()[::-1, ::-1], gapped[::2, ::3], readonly)


@unittest.skipUnless(rs is not None, "compiled backend not built")
class TestNativeArrayLayouts(unittest.TestCase):
    def test_matrix_products_preserve_logical_layout(self):
        a = np.arange(1., 13.).reshape(4, 3)
        b = np.array([[2., -1.], [4., 3.], [-2., 5.]])
        for left in layouts(a):
            for right in layouts(b):
                with self.subTest(left=left.strides, right=right.strides):
                    np.testing.assert_allclose(rs.matmul(left, right), a @ b)

    def test_factorizations_preserve_logical_layout(self):
        a = np.array([[4., 3., -2.], [1., 7., 5.], [6., -4., 8.]])
        for value in layouts(a):
            with self.subTest(strides=value.strides, aligned=value.flags.aligned):
                q, r = rs.householder_qr(value)
                np.testing.assert_allclose(q @ r, a, atol=1e-12)
                np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-12)
                perm, lu = rs.plu(value)
                l = np.tril(lu, -1) + np.eye(3)
                np.testing.assert_allclose(l @ np.triu(lu), a[perm], atol=1e-12)
                np.testing.assert_array_equal(value, a)

    def test_triangular_solves_preserve_logical_layout(self):
        l = np.array([[4., 0., 0.], [1., 7., 0.], [6., -4., 8.]])
        x = np.array([2., -3., 1.])
        for value in layouts(l):
            with self.subTest(strides=value.strides, aligned=value.flags.aligned):
                np.testing.assert_allclose(rs.forward_substitution(value, l @ x), x)
                np.testing.assert_allclose(rs.back_substitution(value, l.T @ x,
                                                                transposed=True), x)
        for value in layouts(l.T):
            np.testing.assert_allclose(rs.back_substitution(value, l.T @ x), x)

    def test_read_only_vectors_handle_gapped_and_negative_strides(self):
        x = np.array([0.25, 0.5, 1., 2.])
        gapped = np.empty(x.size * 3)
        gapped[::3] = x
        for value in (gapped[::3], x[::-1].copy()[::-1]):
            for name in ("gamma", "log_gamma", "erf", "erfc"):
                np.testing.assert_array_equal(getattr(rs, name)(value), getattr(rs, name)(x))
        z = x.astype(complex)
        spread = np.empty(z.size * 2, dtype=complex)
        spread[::2] = z
        raw = spread[::2]
        np.testing.assert_allclose(rs.fft(raw), np.fft.fft(z), atol=1e-12)
        np.testing.assert_allclose(rs.ifft(raw), np.fft.ifft(z), atol=1e-12)

    def test_empty_products_and_factorizations(self):
        for m, k, n in ((0, 3, 2), (2, 0, 3), (2, 3, 0)):
            np.testing.assert_array_equal(rs.matmul(np.zeros((m, k)), np.zeros((k, n))),
                                          np.zeros((m, n)))
        q, r = rs.householder_qr(np.empty((0, 3)))
        self.assertEqual(q.shape, (0, 0))
        self.assertEqual(r.shape, (0, 3))

    def test_in_place_kernels_reject_fortran_layout_before_mutating(self):
        def hessenberg_h(a):
            return rs.hessenberg_qr_iterate(a)

        def hessenberg_v(a):
            return rs.hessenberg_qr_iterate(np.eye(3), a)

        def poisson(a):
            return rs.sor_poisson(a, np.zeros((3, 3)), 1., 1., 1.)

        def cavity_psi(a):
            return rs.lid_driven_cavity(a, np.zeros((3, 3)), 0.5, 0.1, 0.01, 1e-6, 1)

        def cavity_w(a):
            return rs.lid_driven_cavity(np.zeros((3, 3)), a, 0.5, 0.1, 0.01, 1e-6, 1)

        for call in (hessenberg_h, hessenberg_v, poisson, cavity_psi, cavity_w):
            value = np.asfortranarray(np.arange(9.).reshape(3, 3))
            original = value.copy()
            with self.subTest(kernel=call.__name__):
                with self.assertRaisesRegex(ValueError, "C-contiguous"):
                    call(value)
                np.testing.assert_array_equal(value, original)


@unittest.skipUnless(rs is not None, "compiled backend not built")
class TestNativeValidation(unittest.TestCase):
    def test_triangular_inputs_raise_value_error_instead_of_panicking(self):
        for name in ("forward_substitution", "back_substitution"):
            for a, b in ((np.ones((3, 2)), np.ones(3)),
                         (np.eye(3), np.ones(2)), (np.eye(3), np.ones(4))):
                with self.subTest(kernel=name, shape=a.shape, length=b.size):
                    with self.assertRaises(ValueError):
                        getattr(rs, name)(a, b)

    def test_jacobi_requires_square_input(self):
        for shape in ((3, 2), (2, 3), (1, 0)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "square"):
                    rs.jacobi_eigen(np.ones(shape))

    def test_thomas_requires_exact_diagonal_lengths(self):
        for n, sub_size, sup_size in ((3, 1, 2), (3, 2, 1), (3, 3, 2),
                                      (1, 1, 0), (0, 1, 0)):
            with self.subTest(n=n, sub=sub_size, sup=sup_size):
                with self.assertRaises(ValueError):
                    rs.thomas(np.ones(sub_size), np.ones(n), np.ones(sup_size), np.ones(n))
        np.testing.assert_array_equal(rs.thomas(np.empty(0), np.empty(0),
                                                np.empty(0), np.empty(0)), np.empty(0))

    def test_symmetry_matches_numpy_with_nonfinite_entries_and_relative_tolerance(self):
        cases = [np.array([[np.nan]]), np.array([[np.inf]]),
                 np.array([[1., np.inf], [np.inf, 2.]]),
                 np.array([[1., np.inf], [-np.inf, 2.]]),
                 np.array([[1., np.inf], [2., 2.]]),
                 np.array([[1., np.nan], [np.nan, 2.]]),
                 np.array([[1., 1.00001000005], [1., 1.]])]
        for a in cases:
            with self.subTest(matrix=a):
                self.assertEqual(rs.is_symmetric(a, 0., 1e-5),
                                 bool(np.allclose(a, a.T, atol=0., rtol=1e-5)))

    def test_malformed_runge_kutta_tableaux_fail_before_callback(self):
        base = dict(f=lambda t, y: self.fail("malformed tableau evaluated callback"),
                    c=[0.], a_flat=[0.], b_hi=[1.], b_lo=[1.], order=1.,
                    t0=0., tf=1., y0=np.ones(1), rtol=1e-6, atol=1e-8,
                    h0=None, max_step=np.inf, min_step=1e-14, max_steps=1)
        for change in ({"c": []}, {"a_flat": []}, {"b_lo": []}, {"b_hi": []},
                       {"order": 0.}, {"order": np.nan}, {"c": [np.inf]}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    rs.adaptive_rk(**{**base, **change})

    def test_poisson_does_not_report_nan_updates_as_converged(self):
        u = np.zeros((3, 3))
        f = np.zeros_like(u)
        f[1, 1] = np.nan
        iterations, converged, residuals = rs.sor_poisson(u, f, 1., 1., 1., max_iter=3)
        self.assertEqual(iterations, 3)
        self.assertFalse(converged)
        self.assertTrue(np.isnan(residuals).all())

    def test_compact_least_squares_shape_checks(self):
        for a, b in ((np.ones((2, 3)), np.ones(2)),
                     (np.ones((3, 2)), np.ones(2))):
            with self.assertRaises(ValueError):
                rs.qr_least_squares(a, b)
        self.assertEqual(rs.qr_least_squares(np.empty((3, 0)), np.ones(3)).size, 0)

    def test_compact_least_squares_layouts_and_residual(self):
        rng = np.random.default_rng(90)
        a = rng.normal(size=(80, 5))
        b = rng.normal(size=80)
        expected = np.linalg.lstsq(a, b, rcond=None)[0]
        for value in layouts(a):
            result = rs.qr_least_squares(value, b)
            np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(a.T @ (a @ result - b), np.zeros(5), atol=1e-12)
            np.testing.assert_array_equal(value, a)


@unittest.skipUnless(rs is not None, "compiled backend not built")
class TestNativeAdaptiveStepSafety(unittest.TestCase):
    def _solve(self, **changes):
        controls = dict(f=lambda t, y: np.ones_like(y), c=[0.], a_flat=[0.],
                        b_hi=[1.], b_lo=[1.], order=1., t0=0., tf=1.,
                        y0=np.ones(1), rtol=1e-6, atol=1e-8, h0=None,
                        max_step=np.inf, min_step=1e-14, max_steps=100)
        return rs.adaptive_rk(**{**controls, **changes})

    def test_invalid_controls_fail_before_iterations(self):
        for change in ({"t0": np.nan}, {"tf": np.inf}, {"y0": np.array([np.nan])},
                       {"rtol": -1.}, {"atol": np.inf}, {"rtol": 0., "atol": 0.},
                       {"h0": 0.}, {"h0": np.inf}, {"max_step": 0.},
                       {"max_step": np.nan}, {"min_step": 0.},
                       {"min_step": np.inf}, {"min_step": 1., "max_step": 0.1},
                       {"max_steps": 0}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self._solve(**change)

    def test_max_step_applies_to_first_step_in_both_directions(self):
        for t0, tf in ((0., 1.), (1., 0.)):
            ts, ys, *_ = self._solve(t0=t0, tf=tf, h0=1., max_step=0.1)
            self.assertLessEqual(np.abs(np.diff(ts)).max(), 0.1 + 1e-15)
            self.assertEqual(ts[-1], tf)
            np.testing.assert_allclose(ys[-1], [1. + tf - t0], atol=1e-12)

    def test_excess_error_at_minimum_step_raises(self):
        with self.assertRaisesRegex(RuntimeError, "stepsize:"):
            self._solve(b_lo=[0.], h0=0.01, min_step=0.01)

    def test_nonfinite_error_raises(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "stepsize:"):
                    self._solve(f=lambda t, y: np.full_like(y, value))

    def test_step_must_advance_floating_point_time(self):
        with self.assertRaisesRegex(RuntimeError, "stepsize:"):
            self._solve(t0=1e16, tf=1e16 + 2., h0=0.1)

    def test_successful_final_step_does_not_check_unused_next_step(self):
        ts, ys, *_ = self._solve(tf=1e-5, h0=1e-5, min_step=1e-3)
        self.assertEqual(ts[-1], 1e-5)
        np.testing.assert_allclose(ys[-1], [1.00001], atol=1e-12)
