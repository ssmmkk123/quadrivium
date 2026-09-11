"""Native ODE storage and callback ownership regression tests."""

import unittest
import weakref

from quadrivium import _accel, numeric as np
from quadrivium.ode import solve_ivp
from quadrivium.ode.explicit import BUTCHER_TABLEAUX


class TestCAdaptiveRK(unittest.TestCase):
    def setUp(self):
        with _accel.enabled():
            self.kernel = _accel.kernel("adaptive_rk")
        if self.kernel is None:
            self.skipTest("native adaptive RK kernel is unavailable")

    def call(self, rhs, **changes):
        c, rows, hi, lo, order = BUTCHER_TABLEAUX["dormand_prince"]
        a = np.zeros((len(c), len(c)))
        for i, row in enumerate(rows):
            a[i, :len(row)] = row
        args = dict(f=rhs, c=c, a_flat=a.ravel(), b_hi=hi, b_lo=lo,
                    order=float(order), t0=0.0, tf=1.0, y0=np.array([1.0, 2.0]),
                    rtol=1e-8, atol=1e-10, h0=None, max_step=np.inf,
                    min_step=1e-14, max_steps=1000000)
        args.update(changes)
        return self.kernel(**args)

    def test_retained_callback_arguments_and_views_remain_stable(self):
        for retain_view in (False, True):
            retained, expected = [], []

            def rhs(t, y):
                retained.append(y[:] if retain_view else y)
                expected.append(y.copy())
                return -y

            with self.subTest(retain_view=retain_view):
                ts, ys, dys, accepted, rejected, calls = self.call(rhs)
                self.assertEqual(len(retained), calls)
                for actual, snapshot in zip(retained, expected):
                    np.testing.assert_array_equal(actual, snapshot)
                np.testing.assert_allclose(ys[-1], [np.exp(-1), 2 * np.exp(-1)], rtol=2e-8)

    def test_callback_weakrefs_are_not_reused_for_later_states(self):
        refs = []

        def rhs(t, y):
            refs.append(weakref.ref(y))
            return -y

        self.call(rhs)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_callback_can_change_argument_shape(self):
        shapes = []

        def rhs(t, y):
            shapes.append(y.shape)
            y.shape = (2, 1)
            return -y

        result = self.call(rhs)
        self.assertTrue(all(shape == (2,) for shape in shapes))
        np.testing.assert_allclose(result[1][-1], [np.exp(-1), 2 * np.exp(-1)], rtol=2e-8)

    def test_callback_inputs_do_not_alias_user_arrays(self):
        source = np.array([1.0, 2.0])
        c, rows, hi, lo, order = BUTCHER_TABLEAUX["dormand_prince"]
        copied_c = c.copy()

        def rhs(t, y):
            source[:] = 999.0
            copied_c[:] = 999.0
            return -y

        result = self.call(rhs, y0=source, c=copied_c)
        np.testing.assert_allclose(result[1][-1], [np.exp(-1), 2 * np.exp(-1)], rtol=2e-8)

    def test_trajectory_grows_and_returns_exact_owning_shapes(self):
        slope = np.array([1.0, -2.0])
        ts, ys, dys, accepted, rejected, calls = self.call(
            lambda t, y: slope, max_step=1 / 1024, h0=1 / 1024)
        self.assertEqual(accepted, 1024)
        self.assertEqual(rejected, 0)
        self.assertEqual(calls, 7 * accepted + 1)
        self.assertEqual(ts.shape, (1025,))
        self.assertEqual(ys.shape, (1025, 2))
        self.assertEqual(dys.shape, ys.shape)
        for array in (ts, ys, dys):
            self.assertIsNone(array.base)
        np.testing.assert_allclose(ys[-1], [2.0, 0.0], atol=1e-13)
        np.testing.assert_allclose(dys, np.broadcast_to(slope, dys.shape))

    def test_scalar_and_strided_callback_outputs(self):
        scalar = self.call(lambda t, y: -float(y[0]), y0=np.array([1.0]))
        np.testing.assert_allclose(scalar[1][-1], [np.exp(-1)], rtol=2e-8)
        buffer = np.zeros(4)

        def rhs(t, y):
            buffer[::2] = -y
            return buffer[::2]

        strided = self.call(rhs)
        np.testing.assert_allclose(strided[1][-1], [np.exp(-1), 2 * np.exp(-1)], rtol=2e-8)

    def test_callback_exception_is_preserved(self):
        failure = LookupError("callback failure")

        def rhs(t, y):
            raise failure

        with self.assertRaises(LookupError) as caught:
            self.call(rhs)
        self.assertIs(caught.exception, failure)

    def test_native_boundary_rejects_invalid_tableaux_and_results(self):
        invalid = ({"c": []}, {"a_flat": [0.0]}, {"b_hi": []},
                   {"b_lo": [0.0]}, {"order": 0.0}, {"order": np.inf},
                   {"c": [np.nan] * 7}, {"y0": []}, {"y0": [np.inf]},
                   {"rtol": -1.0}, {"max_steps": 0})
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(lambda t, y: -y, **changes)
        with self.assertRaisesRegex(ValueError, "components"):
            self.call(lambda t, y: [1.0])
        for span in ((0.0, 1.0), (1.0, 1.0)):
            with self.subTest(span=span), self.assertRaisesRegex(RuntimeError, "stepsize:"):
                self.call(lambda t, y: [np.nan, 0.0], t0=span[0], tf=span[1])

    def test_all_tableaux_match_reference_for_rejected_backward_steps(self):
        def rhs(t, y):
            return np.array([y[1], -y[0]])

        for method in BUTCHER_TABLEAUX:
            with self.subTest(method=method):
                with _accel.disabled():
                    reference = solve_ivp(rhs, (2.0, 0.0), [1.0, 0.0],
                                          method=method, h0=1.0)
                with _accel.enabled():
                    native = solve_ivp(rhs, (2.0, 0.0), [1.0, 0.0],
                                       method=method, h0=1.0)
                self.assertEqual(native.n_steps, reference.n_steps)
                self.assertEqual(native.n_rejected, reference.n_rejected)
                self.assertGreater(native.n_rejected, 0)
                np.testing.assert_allclose(native.t, reference.t, atol=2e-9, rtol=2e-9)
                np.testing.assert_allclose(native.y, reference.y, atol=2e-9, rtol=2e-9)


if __name__ == "__main__":
    unittest.main()
