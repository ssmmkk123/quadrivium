"""Adaptive integration must report incomplete or invalid solves honestly."""

import unittest

import numpy as np

from quadrivium import _accel
from quadrivium.core.exceptions import DimensionError, StepSizeError
from quadrivium.ode import solve_ivp


class TestAdaptiveReliability(unittest.TestCase):
    def test_step_limit_returns_partial_solution(self):
        for context in (_accel.disabled, _accel.enabled):
            for span in ((0.0, 1.0), (1.0, 0.0)):
                with self.subTest(backend=context.__name__, span=span), context():
                    sol = solve_ivp(lambda t, y: -y, span, [1.0],
                                    max_steps=1, h0=0.01, dense_output=True)
                    self.assertFalse(sol.success)
                    self.assertIn("maximum number of steps", sol.message)
                    self.assertEqual(sol.n_steps, 1)
                    self.assertNotEqual(sol.t[-1], span[-1])
                    self.assertTrue(np.isfinite(sol(sol.t[-1])).all())

    def test_max_step_is_respected_from_first_step(self):
        for context in (_accel.disabled, _accel.enabled):
            for span in ((0.0, 1.0), (1.0, 0.0)):
                with self.subTest(backend=context.__name__, span=span), context():
                    sol = solve_ivp(lambda t, y: np.ones_like(y), span, [0.0],
                                    h0=0.5, max_step=0.02)
                    self.assertTrue(sol.success)
                    self.assertLessEqual(np.abs(np.diff(sol.t)).max(), 0.02 + 1e-15)
                    self.assertAlmostEqual(sol.y[-1, 0], span[1] - span[0], places=12)

    def test_bad_controls_are_rejected_on_both_paths(self):
        bad_controls = (
            {"rtol": -1}, {"rtol": np.nan}, {"atol": np.inf},
            {"atol": 0, "rtol": 0}, {"max_step": 0}, {"max_step": np.nan},
            {"min_step": 0}, {"min_step": np.inf},
            {"min_step": 1, "max_step": 0.1}, {"h0": 0}, {"h0": np.inf},
            {"max_steps": 0}, {"max_steps": -1},
        )
        for context in (_accel.disabled, _accel.enabled):
            with context():
                for options in bad_controls:
                    with self.subTest(backend=context.__name__, options=options):
                        with self.assertRaises(ValueError):
                            solve_ivp(lambda t, y: -y, (0, 1), [1.0], **options)
                with self.assertRaises(TypeError):
                    solve_ivp(lambda t, y: -y, (0, 1), [1.0], max_steps=1.5)
                for span, y in (((0, np.inf), [1]), ((0, 1), []), ((0, 1), [np.nan])):
                    with self.assertRaises(ValueError):
                        solve_ivp(lambda t, y: -y, span, y)

    def test_nonfinite_and_mismatched_rhs_fail_promptly(self):
        for context in (_accel.disabled, _accel.enabled):
            with context():
                with self.assertRaises(StepSizeError):
                    solve_ivp(lambda t, y: [np.nan], (0, 1), [1.0])
                with self.assertRaises(DimensionError):
                    solve_ivp(lambda t, y: [1.0], (0, 1), [1.0, 2.0])

    def test_unsatisfied_tolerance_at_minimum_step_fails(self):
        for context in (_accel.disabled, _accel.enabled):
            with context(), self.assertRaises(StepSizeError):
                solve_ivp(lambda t, y: 1000 * y, (0, 1), [1.0],
                          h0=0.1, min_step=0.1, rtol=1e-12, atol=1e-12)

    def test_time_must_actually_advance(self):
        for context in (_accel.disabled, _accel.enabled):
            with context(), self.assertRaises(StepSizeError):
                solve_ivp(lambda t, y: y, (1, 2), [1.0],
                          max_step=1e-20, min_step=1e-25)

    def test_dense_output_for_backward_and_zero_intervals(self):
        for context in (_accel.disabled, _accel.enabled):
            with context():
                backward = solve_ivp(lambda t, y: -y, (1, 0), [np.exp(-1)],
                                     dense_output=True)
                self.assertTrue(backward.success)
                np.testing.assert_allclose(backward([0, 0.5, 1])[:, 0],
                                           np.exp(-np.array([0, 0.5, 1])), atol=1e-5)
                zero = solve_ivp(lambda t, y: -y, (1, 1), [2.0], dense_output=True)
                self.assertTrue(zero.success)
                self.assertEqual(zero.n_steps, 0)
                np.testing.assert_array_equal(zero(1), [2.0])

    def test_final_interval_shorter_than_minimum_step(self):
        for context in (_accel.disabled, _accel.enabled):
            with context():
                sol = solve_ivp(lambda t, y: np.ones_like(y), (0, 1e-16), [0.0])
                self.assertTrue(sol.success)
                self.assertEqual(sol.t[-1], 1e-16)
                self.assertAlmostEqual(sol.y[-1, 0] / 1e-16, 1, places=12)

    def test_callback_can_reuse_its_output_buffer(self):
        for context in (_accel.disabled, _accel.enabled):
            buffer = np.empty(2)

            def rhs(t, y):
                buffer[:] = -y
                return buffer

            with context():
                sol = solve_ivp(rhs, (0, 1), [1, 2])
            np.testing.assert_allclose(sol.y[-1], np.exp(-1) * np.array([1, 2]), rtol=2e-8)


if __name__ == "__main__":
    unittest.main()
