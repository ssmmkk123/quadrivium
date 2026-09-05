"""The compiled backend must be indistinguishable from the Python one.

Every kernel in the Rust extension shadows a pure-Python implementation that
stays in the tree. These tests pin the contract between them: for the same
input, both paths must agree to within floating-point noise, and both must
raise the same exception types. If the extension is not built, the comparisons
are skipped but the pure-Python path is still exercised by the rest of the
suite.
"""

from __future__ import annotations

import unittest

import numpy as np

from quadrivium import _accel
from quadrivium.core.exceptions import SingularMatrixError
from quadrivium.linalg import (back_substitution, cholesky, forward_substitution,
                               householder_qr, jacobi_eigen, plu_decomposition,
                               qr_algorithm, solve, thomas)
from quadrivium.ode import solve_ivp
from quadrivium.pde import lid_driven_cavity, poisson_2d_iterative
from quadrivium.special import erf, erfc, gamma, log_gamma
from quadrivium.transforms import fft, ifft

HAVE_ACCEL = _accel.available()
skip_no_accel = unittest.skipUnless(HAVE_ACCEL, "compiled backend not built")


def both_backends(fn):
    """Return ``(python_result, active_result)`` for a zero-argument callable."""
    with _accel.disabled():
        slow = fn()
    fast = fn()
    return slow, fast


class TestBackendSwitching(unittest.TestCase):
    def test_backend_reports_a_known_name(self):
        self.assertIn(_accel.backend(), ("rust", "python"))

    def test_disabled_forces_python(self):
        with _accel.disabled():
            self.assertEqual(_accel.backend(), "python")
            self.assertFalse(_accel.available())

    def test_backend_restored_after_context(self):
        before = _accel.backend()
        with _accel.disabled():
            pass
        self.assertEqual(_accel.backend(), before)

    def test_disabled_restores_even_on_error(self):
        before = _accel.backend()
        with self.assertRaises(ValueError):
            with _accel.disabled():
                raise ValueError("boom")
        self.assertEqual(_accel.backend(), before)


@skip_no_accel
class TestLinalgEquivalence(unittest.TestCase):
    """Sizes straddle the 64-wide blocking threshold in the Rust kernels."""

    SIZES = (1, 2, 7, 63, 64, 65, 130)

    def setUp(self):
        self.rng = np.random.default_rng(20240904)

    def _spd(self, n):
        a = self.rng.standard_normal((n, n))
        return a @ a.T + n * np.eye(n)

    def test_cholesky(self):
        for n in self.SIZES:
            with self.subTest(n=n):
                s = self._spd(n)
                slow, fast = both_backends(lambda: cholesky(s))
                np.testing.assert_allclose(slow, fast, rtol=1e-10, atol=1e-12)

    def test_cholesky_rejects_indefinite_on_both_paths(self):
        bad = np.array([[1.0, 2.0], [2.0, 1.0]])
        with _accel.disabled():
            self.assertRaises(SingularMatrixError, cholesky, bad)
        self.assertRaises(SingularMatrixError, cholesky, bad)

    def test_plu(self):
        for n in self.SIZES:
            with self.subTest(n=n):
                a = self.rng.standard_normal((n, n))
                slow, fast = both_backends(lambda: plu_decomposition(a))
                for got, want in zip(fast, slow):
                    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)
                p, l, u = fast
                np.testing.assert_allclose(p @ a, l @ u, rtol=1e-9, atol=1e-11)

    def test_householder_qr(self):
        for m, n in ((5, 5), (64, 64), (80, 40), (40, 80), (129, 65)):
            with self.subTest(m=m, n=n):
                a = self.rng.standard_normal((m, n))
                slow, fast = both_backends(lambda: householder_qr(a))
                for got, want in zip(fast, slow):
                    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)
                q, r = fast
                np.testing.assert_allclose(q @ r, a, rtol=1e-9, atol=1e-11)

    def test_triangular_solves(self):
        for n in self.SIZES:
            with self.subTest(n=n):
                s = self._spd(n)
                b = self.rng.standard_normal(n)
                lo, up = np.tril(s), np.triu(s)
                slow, fast = both_backends(lambda: forward_substitution(lo, b))
                np.testing.assert_allclose(slow, fast, rtol=1e-9, atol=1e-11)
                slow, fast = both_backends(lambda: back_substitution(up, b))
                np.testing.assert_allclose(slow, fast, rtol=1e-9, atol=1e-11)

    def test_triangular_solve_zero_diagonal_raises_on_both_paths(self):
        lo = np.array([[1.0, 0.0], [1.0, 0.0]])
        b = np.array([1.0, 1.0])
        with _accel.disabled():
            self.assertRaises(SingularMatrixError, forward_substitution, lo, b)
        self.assertRaises(SingularMatrixError, forward_substitution, lo, b)

    def test_solve(self):
        for n in self.SIZES:
            with self.subTest(n=n):
                s = self._spd(n)
                b = self.rng.standard_normal(n)
                slow, fast = both_backends(lambda: solve(s, b))
                np.testing.assert_allclose(slow, fast, rtol=1e-9, atol=1e-11)

    def test_jacobi_eigen(self):
        # Eigenvector signs are not unique, so the spectrum is compared
        # directly and the vectors through the defining residual.
        for n in (2, 9, 40):
            with self.subTest(n=n):
                a = self.rng.standard_normal((n, n))
                a = a + a.T
                slow, fast = both_backends(lambda: jacobi_eigen(a))
                np.testing.assert_allclose(
                    fast.eigenvalues, slow.eigenvalues, rtol=1e-9, atol=1e-11
                )
                v, w = fast.eigenvectors, fast.eigenvalues
                np.testing.assert_allclose(a @ v, v * w, rtol=1e-8, atol=1e-10)


@skip_no_accel
class TestGridSolverEquivalence(unittest.TestCase):
    """Kernels that run a whole iteration rather than a single step.

    These do not merely have to land on the same answer: they drive their own
    convergence tests, so a backend that took a different number of sweeps
    would be reporting different diagnostics for the same problem. Iteration
    counts are therefore compared alongside the fields.
    """

    def test_qr_algorithm_spectrum_and_vectors(self):
        rng = np.random.default_rng(5)
        for n in (2, 3, 9, 25):
            with self.subTest(n=n):
                a = rng.standard_normal((n, n))
                a = a + a.T
                slow, fast = both_backends(lambda: qr_algorithm(a))
                self.assertEqual(fast.iterations, slow.iterations)
                self.assertEqual(fast.converged, slow.converged)
                np.testing.assert_allclose(fast.eigenvalues, slow.eigenvalues,
                                           rtol=1e-9, atol=1e-11)

    def test_qr_algorithm_accumulates_the_same_basis(self):
        rng = np.random.default_rng(6)
        a = rng.standard_normal((8, 8))
        a = a + a.T
        slow, fast = both_backends(lambda: qr_algorithm(a, compute_vectors=True))
        np.testing.assert_allclose(fast.eigenvectors, slow.eigenvectors,
                                   rtol=1e-9, atol=1e-11)
        v = fast.eigenvectors
        # V stays orthogonal, and diagonalizes A when the iteration converged.
        np.testing.assert_allclose(v.T @ v, np.eye(8), rtol=1e-9, atol=1e-11)
        if fast.converged:
            np.testing.assert_allclose(a @ v, v * fast.eigenvalues,
                                       rtol=1e-7, atol=1e-9)

    def test_sor_and_gauss_seidel_sweeps_agree(self):
        src = lambda x, y: np.sin(np.pi * x) * y
        for method in ("sor", "gauss_seidel"):
            for n in (5, 12):
                with self.subTest(method=method, n=n):
                    slow, fast = both_backends(
                        lambda: poisson_2d_iterative(src, (0, 1), (0, 1), nx=n, ny=n,
                                                     method=method))
                    self.assertEqual(fast.iterations, slow.iterations)
                    self.assertEqual(fast.converged, slow.converged)
                    np.testing.assert_allclose(fast.u, slow.u, rtol=1e-9, atol=1e-11)

    def test_jacobi_poisson_has_no_kernel_and_still_matches(self):
        # The Jacobi branch is array-expressible and deliberately has no
        # compiled twin; enabling the backend must not perturb it.
        src = lambda x, y: 1.0
        slow, fast = both_backends(
            lambda: poisson_2d_iterative(src, (0, 1), (0, 1), nx=8, ny=8,
                                         method="jacobi", max_iter=200))
        np.testing.assert_allclose(fast.u, slow.u, rtol=1e-12, atol=1e-14)

    def test_thomas_matches_and_stays_exact(self):
        rng = np.random.default_rng(8)
        for n in (1, 2, 3, 64, 2000):
            with self.subTest(n=n):
                # Diagonally dominant, so the pivot-free recurrence is stable.
                diag = rng.uniform(3.0, 4.0, n)
                sub = rng.uniform(-1.0, 1.0, max(n - 1, 0))
                sup = rng.uniform(-1.0, 1.0, max(n - 1, 0))
                rhs = rng.standard_normal(n)
                slow, fast = both_backends(lambda: thomas(sub, diag, sup, rhs))
                np.testing.assert_array_equal(fast, slow)
                a = np.diag(diag)
                if n > 1:
                    a += np.diag(sub, -1) + np.diag(sup, 1)
                np.testing.assert_allclose(a @ fast, rhs, rtol=1e-9, atol=1e-11)

    def test_thomas_zero_pivot_raises_on_both_paths(self):
        sub, diag = np.array([1.0]), np.array([0.0, 1.0])
        sup, rhs = np.array([1.0]), np.array([1.0, 1.0])
        with _accel.disabled():
            self.assertRaises(SingularMatrixError, thomas, sub, diag, sup, rhs)
        self.assertRaises(SingularMatrixError, thomas, sub, diag, sup, rhs)

    def test_thomas_does_not_consume_its_inputs(self):
        # The kernel copies before eliminating; a caller reusing the arrays
        # (every implicit PDE step does) must see them unchanged.
        diag = np.array([4.0, 4.0, 4.0])
        sub, sup, rhs = np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 2.0, 3.0])
        keep = [v.copy() for v in (sub, diag, sup, rhs)]
        thomas(sub, diag, sup, rhs)
        for got, want in zip((sub, diag, sup, rhs), keep):
            np.testing.assert_array_equal(got, want)

    def test_lid_driven_cavity_fields_agree(self):
        for n, re in ((7, 10.0), (13, 100.0)):
            with self.subTest(n=n, re=re):
                slow, fast = both_backends(
                    lambda: lid_driven_cavity(re=re, n=n, max_iter=300))
                for got, want in zip(fast, slow):
                    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)


@skip_no_accel
class TestTransformEquivalence(unittest.TestCase):
    # Powers of two, smooth composites, and primes -- one per FFT code path.
    LENGTHS = (1, 2, 5, 16, 60, 97, 128, 1000, 1024)

    def setUp(self):
        self.rng = np.random.default_rng(11)

    def test_fft_matches_python_backend(self):
        for n in self.LENGTHS:
            with self.subTest(n=n):
                x = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
                slow, fast = both_backends(lambda: fft(x))
                np.testing.assert_allclose(slow, fast, rtol=1e-8, atol=1e-10)

    def test_fft_matches_numpy(self):
        for n in self.LENGTHS:
            with self.subTest(n=n):
                x = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
                np.testing.assert_allclose(fft(x), np.fft.fft(x), rtol=1e-8, atol=1e-10)

    def test_ifft_round_trip(self):
        for n in self.LENGTHS:
            with self.subTest(n=n):
                x = self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)
                np.testing.assert_allclose(ifft(fft(x)), x, rtol=1e-8, atol=1e-10)


@skip_no_accel
class TestSpecialEquivalence(unittest.TestCase):
    def test_array_valued_functions_agree(self):
        grids = {
            gamma: np.linspace(0.1, 15.0, 257),
            log_gamma: np.linspace(0.1, 60.0, 257),
            erf: np.linspace(-6.0, 6.0, 257),
            erfc: np.linspace(-6.0, 6.0, 257),
        }
        for fn, x in grids.items():
            with self.subTest(fn=fn.__name__):
                slow, fast = both_backends(lambda: fn(x))
                np.testing.assert_allclose(slow, fast, rtol=1e-10, atol=1e-12)

    def test_scalar_input_still_returns_a_float(self):
        for fn in (gamma, log_gamma, erf, erfc):
            with self.subTest(fn=fn.__name__):
                value = fn(2.5)
                self.assertIsInstance(value, float)
                with _accel.disabled():
                    self.assertAlmostEqual(value, fn(2.5), places=10)

    def test_negative_and_pole_arguments(self):
        x = np.array([-2.5, -1.5, -0.5, 0.0, -3.0])
        slow, fast = both_backends(lambda: gamma(x))
        np.testing.assert_allclose(slow, fast, rtol=1e-10, atol=1e-12)
        self.assertTrue(np.isinf(fast[3]) and np.isinf(fast[4]))


@skip_no_accel
class TestOdeEquivalence(unittest.TestCase):
    """The RK driver is a faithful port, so agreement here is exact.

    Both backends assemble the stages in the same order with the same
    arithmetic, so they take identical steps -- not merely steps that converge
    to the same answer. Anything less than bit equality is a port error.
    """

    TABLEAUX = ("dormand_prince", "rkf45", "cash_karp", "bogacki_shampine")

    @staticmethod
    def _decay(t, y):
        return -2.0 * y

    @staticmethod
    def _van_der_pol(t, y):
        return np.array([y[1], 3.0 * (1.0 - y[0] ** 2) * y[1] - y[0]])

    @staticmethod
    def _lorenz(t, y):
        return np.array([
            10.0 * (y[1] - y[0]),
            y[0] * (28.0 - y[2]) - y[1],
            y[0] * y[1] - 8.0 / 3.0 * y[2],
        ])

    def _problems(self):
        yield "decay", self._decay, (0.0, 5.0), [1.0]
        yield "van_der_pol", self._van_der_pol, (0.0, 20.0), [2.0, 0.0]
        yield "lorenz", self._lorenz, (0.0, 10.0), [1.0, 1.0, 1.0]

    def test_every_tableau_steps_identically(self):
        for tableau in self.TABLEAUX:
            for name, f, span, y0 in self._problems():
                with self.subTest(tableau=tableau, problem=name):
                    with _accel.disabled():
                        slow = solve_ivp(f, span, y0, method=tableau)
                    fast = solve_ivp(f, span, y0, method=tableau)
                    self.assertEqual(slow.n_accepted, fast.n_accepted)
                    self.assertEqual(slow.n_rejected, fast.n_rejected)
                    self.assertEqual(slow.n_rhs_evals, fast.n_rhs_evals)
                    np.testing.assert_array_equal(slow.t, fast.t)
                    np.testing.assert_array_equal(slow.y, fast.y)
                    np.testing.assert_array_equal(slow.dydt, fast.dydt)

    def test_backwards_integration(self):
        with _accel.disabled():
            slow = solve_ivp(self._decay, (5.0, 0.0), [1.0])
        fast = solve_ivp(self._decay, (5.0, 0.0), [1.0])
        np.testing.assert_array_equal(slow.t, fast.t)
        np.testing.assert_array_equal(slow.y, fast.y)

    def test_scalar_rhs_and_list_rhs_are_accepted(self):
        # The kernel coerces whatever the callback returns the same way
        # ``as_vector`` does on the Python path.
        for rhs in (lambda t, y: -2.0 * y[0], lambda t, y: [-2.0 * y[0]]):
            with self.subTest(rhs=rhs):
                with _accel.disabled():
                    slow = solve_ivp(rhs, (0.0, 3.0), [1.0])
                fast = solve_ivp(rhs, (0.0, 3.0), [1.0])
                np.testing.assert_allclose(slow.y[-1], fast.y[-1], rtol=1e-12)

    def test_exception_from_the_callback_propagates(self):
        sentinel = RuntimeError("rhs exploded")

        def bad(t, y):
            raise sentinel

        with self.assertRaises(RuntimeError) as caught:
            solve_ivp(bad, (0.0, 1.0), [1.0])
        self.assertIs(caught.exception, sentinel)

    def test_dense_output_agrees(self):
        q = np.linspace(0.0, 5.0, 37)
        with _accel.disabled():
            slow = solve_ivp(self._decay, (0.0, 5.0), [1.0], dense_output=True)
        fast = solve_ivp(self._decay, (0.0, 5.0), [1.0], dense_output=True)
        np.testing.assert_allclose(slow.interpolant(q), fast.interpolant(q), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
