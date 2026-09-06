"""Numerical and concurrency regressions for bounded-workspace operations."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from quadrivium import numeric as np

from quadrivium import _accel
from quadrivium.core.exceptions import DimensionError, SingularMatrixError
from quadrivium.linalg import pseudoinverse, qr_least_squares, ridge_regression, tikhonov
from quadrivium.stochastic import kernel_density


class TestBackendIsolation(unittest.TestCase):
    def test_overlapping_threads_keep_independent_backends(self):
        barrier = Barrier(2, timeout=5)

        def worker(context, expected):
            with context():
                barrier.wait()
                observed = (_accel.available(), _accel.kernel("probe"))
                barrier.wait()
            return observed == expected

        probe = lambda: None
        with patch.object(_accel, "_rs", SimpleNamespace(probe=probe)):
            with ThreadPoolExecutor(max_workers=2) as pool:
                off = pool.submit(worker, _accel.disabled, (False, None))
                on = pool.submit(worker, _accel.enabled, (True, probe))
                self.assertTrue(off.result(timeout=10))
                self.assertTrue(on.result(timeout=10))

    def test_async_tasks_do_not_change_one_another(self):
        async def run():
            async def worker(context, expected):
                with context():
                    await asyncio.sleep(0)
                    self.assertEqual(_accel.available(), expected)
            with _accel.enabled():
                await asyncio.gather(worker(_accel.disabled, False),
                                     worker(_accel.enabled, True))
                self.assertTrue(_accel.available())

        with patch.object(_accel, "_rs", SimpleNamespace()):
            asyncio.run(run())

    def test_nested_exception_restores_outer_context(self):
        before = _accel.available()
        with patch.object(_accel, "_rs", SimpleNamespace()):
            with _accel.disabled():
                with self.assertRaisesRegex(RuntimeError, "test"):
                    with _accel.enabled():
                        self.assertTrue(_accel.available())
                        raise RuntimeError("test")
                self.assertFalse(_accel.available())
        self.assertEqual(_accel.available(), before)


class TestCompactLeastSquares(unittest.TestCase):
    def test_residual_is_orthogonal_and_inputs_are_unchanged(self):
        rng = np.random.default_rng(70)
        for context in (_accel.disabled, _accel.enabled):
            for m, n in ((1, 1), (17, 17), (3000, 5)):
                with self.subTest(backend=context.__name__, shape=(m, n)):
                    # Exercise strided inputs on both paths.
                    A = rng.normal(size=(m * 2, n * 2))[::2, ::2]
                    b = rng.normal(size=m * 2)[::2]
                    saved_A, saved_b = A.copy(), b.copy()
                    with context():
                        x = qr_least_squares(A, b)
                    expected = np.linalg.lstsq(A, b, rcond=None)[0]
                    np.testing.assert_allclose(x, expected, atol=2e-12, rtol=2e-11)
                    np.testing.assert_allclose(A.T @ (A @ x - b), 0, atol=2e-10)
                    np.testing.assert_array_equal(A, saved_A)
                    np.testing.assert_array_equal(b, saved_b)

    def test_scaled_and_ill_conditioned_problems(self):
        A = np.vander(np.linspace(0, 1, 40), 9)
        expected = np.arange(9, dtype=float)
        for context in (_accel.disabled, _accel.enabled):
            with context():
                for scale in (1.0, 1e-140):
                    x = qr_least_squares(A * scale, (A @ expected) * scale)
                    np.testing.assert_allclose(x, expected, rtol=2e-8, atol=2e-8)

    def test_empty_shapes_and_invalid_systems(self):
        for context in (_accel.disabled, _accel.enabled):
            with context():
                for m in (0, 4):
                    self.assertEqual(qr_least_squares(np.empty((m, 0)), np.zeros(m)).shape, (0,))
                with self.assertRaises(DimensionError):
                    qr_least_squares(np.ones((2, 3)), np.ones(2))
                with self.assertRaises(DimensionError):
                    qr_least_squares(np.ones((4, 2)), np.ones(2))
                with self.assertRaises(SingularMatrixError):
                    qr_least_squares(np.zeros((4, 2)), np.ones(4))

    def test_pseudoinverse_identities_and_regularization(self):
        rng = np.random.default_rng(90)
        A = rng.normal(size=(30, 8))
        A[:, -1] = A[:, 0] + A[:, 1]
        inverse = pseudoinverse(A)
        np.testing.assert_allclose(A @ inverse @ A, A, atol=2e-13)
        np.testing.assert_allclose(inverse @ A @ inverse, inverse, atol=2e-13)
        b = rng.normal(size=30)
        expected = np.linalg.solve(A.T @ A + 0.7 * np.eye(8), A.T @ b)
        np.testing.assert_allclose(ridge_regression(A, b, 0.7), expected)
        np.testing.assert_allclose(tikhonov(A, b, 0.7), expected)


class TestTiledDensity(unittest.TestCase):
    def test_all_kernels_match_direct_sum_across_tile_boundaries(self):
        rng = np.random.default_rng(123)
        for count in (9, 137):
            x = rng.normal(size=count * 2)[::2]
            points = np.linspace(-3, 3, 78)[::2]
            saved = x.copy()
            u = (points[:, None] - x) / 0.4
            values = {
                "gaussian": np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi),
                "epanechnikov": np.where(abs(u) <= 1, 0.75 * (1 - u**2), 0),
                "uniform": np.where(abs(u) <= 1, 0.5, 0),
                "triangular": np.maximum(1 - abs(u), 0),
            }
            with patch("quadrivium.stochastic.stats._KDE_TILE_ELEMENTS", 64):
                for kernel, K in values.items():
                    with self.subTest(count=count, kernel=kernel):
                        pts, density = kernel_density(x, points, 0.4, kernel)
                        np.testing.assert_array_equal(pts, points)
                        np.testing.assert_allclose(density, K.mean(axis=1) / 0.4,
                                                   rtol=3e-15, atol=1e-16)
            np.testing.assert_array_equal(x, saved)

    def test_singleton_and_constant_samples_are_finite_and_normalized(self):
        for x in ([2.0], [2.0] * 20):
            points = np.linspace(-8, 12, 2001)
            _, density = kernel_density(x, points)
            self.assertTrue(np.isfinite(density).all())
            integral = np.sum((density[:-1] + density[1:]) * np.diff(points) / 2)
            self.assertAlmostEqual(integral, 1.0, places=10)

    def test_invalid_inputs_and_distant_points(self):
        for x in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                kernel_density(x)
        for bandwidth in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                kernel_density([0.0], bandwidth=bandwidth)
        with self.assertRaises(ValueError):
            kernel_density([0.0], [np.nan])
        with self.assertRaises(ValueError):
            kernel_density([0.0], kernel="unknown")
        self.assertEqual(kernel_density([0.0], [])[1].size, 0)
        for kernel in ("gaussian", "epanechnikov", "uniform", "triangular"):
            with np.errstate(over="raise", invalid="raise"):
                _, density = kernel_density([-1e308], [1e308], 0.1, kernel)
            np.testing.assert_array_equal(density, [0.0])


if __name__ == "__main__":
    unittest.main()
