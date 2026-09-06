"""Tests for core utilities and linear algebra."""

import unittest

import array as array_module

from quadrivium import numeric as np

from quadrivium.core import (EPS, CountedFunction, absolute_error,
                            as_vector, condition_number,
                            is_diagonally_dominant, is_positive_definite,
                            machine_epsilon, matrix_norm, norm,
                            numerical_gradient, numerical_hessian,
                            numerical_jacobian, relative_error)
from quadrivium.core.exceptions import DimensionError, SingularMatrixError
from quadrivium.linalg import *


class TestCoreUtils(unittest.TestCase):
    def test_machine_epsilon(self):
        self.assertAlmostEqual(machine_epsilon(), EPS)
        self.assertTrue(1.0 + EPS != 1.0)
        self.assertTrue(1.0 + EPS / 2 == 1.0)

    def test_norms(self):
        v = np.array([3.0, -4.0])
        self.assertAlmostEqual(norm(v, 1), 7.0)
        self.assertAlmostEqual(norm(v, 2), 5.0)
        self.assertAlmostEqual(norm(v, np.inf), 4.0)
        A = np.array([[1.0, -2.0], [-3.0, 4.0]])
        self.assertAlmostEqual(matrix_norm(A, 1), 6.0)
        self.assertAlmostEqual(matrix_norm(A, np.inf), 7.0)
        self.assertAlmostEqual(matrix_norm(A, "fro"), np.sqrt(30))

    def test_condition_number(self):
        self.assertAlmostEqual(condition_number(np.eye(3)), 1.0)
        hilbert = np.array([[1 / (i + j + 1) for j in range(5)] for i in range(5)])
        self.assertGreater(condition_number(hilbert), 1e5)

    def test_errors_and_predicates(self):
        self.assertAlmostEqual(absolute_error([1.0, 2.0], [1.0, 2.5]), 0.5)
        self.assertAlmostEqual(relative_error([1.0, 2.0], [1.0, 2.0]), 0.0)
        self.assertTrue(is_positive_definite(np.array([[2.0, 1.0], [1.0, 2.0]])))
        self.assertFalse(is_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]])))
        self.assertTrue(is_diagonally_dominant(np.array([[5.0, 1.0], [1.0, 5.0]])))

    def test_numerical_derivatives(self):
        f = lambda v: v[0] ** 2 * v[1] + np.sin(v[1])
        x = np.array([2.0, 0.5])
        g = numerical_gradient(f, x)
        np.testing.assert_allclose(g, [2 * 2 * 0.5, 4 + np.cos(0.5)], atol=1e-6)
        H = numerical_hessian(f, x)
        np.testing.assert_allclose(H, [[1.0, 4.0], [4.0, -np.sin(0.5)]], atol=1e-4)
        J = numerical_jacobian(lambda v: np.array([v[0] * v[1], v[0] + v[1]]), x)
        np.testing.assert_allclose(J, [[0.5, 2.0], [1.0, 1.0]], atol=1e-6)


    def test_as_vector_shapes_and_aliasing(self):
        # The fast path returns the argument itself; every other input is
        # coerced. Which of those share memory with the input is part of the
        # contract, because callers rely on `as_vector(x).copy()` being the
        # only thing that detaches.
        flat = np.arange(4.0)
        self.assertIs(as_vector(flat), flat)
        for value, want in ((3.0, [3.0]), ([1, 2], [1.0, 2.0]),
                            (np.zeros((2, 3)), [0.0] * 6),
                            (np.array(5.0), [5.0]),
                            (array_module.array("f", [1.0] * 3), [1.0] * 3)):
            got = as_vector(value)
            self.assertEqual(got.ndim, 1)
            self.assertEqual(got.dtype, np.float64)
            np.testing.assert_array_equal(got, want)
        strided = np.arange(6.0)[::2]
        self.assertFalse(np.shares_memory(as_vector(strided), strided))
        view = as_vector(flat)
        view[0] = 9.0
        self.assertEqual(flat[0], 9.0)

    def test_counted_function_call_shapes(self):
        self.assertEqual(CountedFunction(lambda: 7)(), 7)
        self.assertEqual(CountedFunction(lambda a: a + 1)(2), 3)
        self.assertEqual(CountedFunction(lambda a, b: a + b)(2, 3), 5)
        self.assertEqual(CountedFunction(lambda k: k * 2)(k=4), 8)
        self.assertEqual(CountedFunction(lambda a, k=0: a + k)(1, k=2), 3)
        self.assertIsNone(CountedFunction(lambda a: a)(None))
        counted = CountedFunction(lambda a: a)
        for _ in range(5):
            counted(1)
        self.assertEqual(counted.calls, 5)
        counted.reset()
        self.assertEqual(counted.calls, 0)


class TestDirectSolvers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.A = rng.random((6, 6)) + 6 * np.eye(6)
        self.b = rng.random(6)
        self.x = np.linalg.solve(self.A, self.b)
        self.S = self.A @ self.A.T + np.eye(6)

    def test_triangular_solves(self):
        L = np.tril(self.A)
        np.testing.assert_allclose(forward_substitution(L, self.b),
                                   np.linalg.solve(L, self.b))
        U = np.triu(self.A)
        np.testing.assert_allclose(back_substitution(U, self.b),
                                   np.linalg.solve(U, self.b))

    def test_gauss_all_pivoting(self):
        for p in ("none", "partial", "scaled", "complete"):
            np.testing.assert_allclose(gauss_elimination(self.A, self.b, p),
                                       self.x, atol=1e-10)

    def test_factorizations_reconstruct(self):
        L, U = lu_decomposition(self.A)
        np.testing.assert_allclose(L @ U, self.A, atol=1e-12)
        P, L, U = plu_decomposition(self.A)
        np.testing.assert_allclose(P @ self.A, L @ U, atol=1e-12)
        P, L, U, Q = lu_complete_pivot(self.A)
        np.testing.assert_allclose(P @ self.A @ Q, L @ U, atol=1e-12)
        Lc = cholesky(self.S)
        np.testing.assert_allclose(Lc @ Lc.T, self.S, atol=1e-10)
        Ld, d = ldl_decomposition(self.S)
        np.testing.assert_allclose(Ld @ np.diag(d) @ Ld.T, self.S, atol=1e-10)

    def test_qr_variants(self):
        for f in (gram_schmidt_qr, modified_gram_schmidt_qr, householder_qr,
                  givens_qr):
            Q, R = f(self.A)
            np.testing.assert_allclose(Q @ R, self.A, atol=1e-10)
            np.testing.assert_allclose(Q.T @ Q, np.eye(6), atol=1e-8)

    def test_solvers_agree(self):
        for f in (plu_solve, gauss_jordan, qr_solve, solve):
            np.testing.assert_allclose(f(self.A, self.b), self.x, atol=1e-10)
        np.testing.assert_allclose(cholesky_solve(self.S, self.b),
                                   np.linalg.solve(self.S, self.b), atol=1e-10)
        np.testing.assert_allclose(ldl_solve(self.S, self.b),
                                   np.linalg.solve(self.S, self.b), atol=1e-10)

    def test_structured(self):
        T = (np.diag([4.0] * 5) + np.diag([-1.0] * 4, 1) + np.diag([-1.0] * 4, -1))
        d = np.arange(5.0)
        np.testing.assert_allclose(
            thomas(np.diag(T, -1), np.diag(T), np.diag(T, 1), d),
            np.linalg.solve(T, d), atol=1e-12)
        np.testing.assert_allclose(banded_solve(T, d, 1, 1),
                                   np.linalg.solve(T, d), atol=1e-12)

    def test_utilities(self):
        np.testing.assert_allclose(inverse(self.A), np.linalg.inv(self.A), atol=1e-10)
        self.assertAlmostEqual(determinant(self.A), np.linalg.det(self.A), places=6)
        self.assertEqual(rank(np.array([[1.0, 2.0], [2.0, 4.0]])), 1)
        N = nullspace(np.array([[1.0, 2.0], [2.0, 4.0]]))
        np.testing.assert_allclose(np.array([[1.0, 2.0], [2.0, 4.0]]) @ N, 0, atol=1e-12)

    def test_singular_matrix_raises(self):
        with self.assertRaises(SingularMatrixError):
            plu_solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(DimensionError):
            forward_substitution(np.eye(3), np.ones(2))


class TestEigen(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        S = rng.random((5, 5))
        self.S = S + S.T + 5 * np.eye(5)
        self.eigs = np.sort(np.linalg.eigvalsh(self.S))

    def test_vector_iterations(self):
        self.assertAlmostEqual(power_iteration(self.S).eigenvalues[0],
                               self.eigs[-1], places=6)
        r = inverse_power_iteration(self.S, sigma=self.eigs[0] - 0.1)
        self.assertAlmostEqual(r.eigenvalues[0], self.eigs[0], places=6)
        r = rayleigh_quotient_iteration(self.S, x0=np.ones(5))
        self.assertLess(np.min(np.abs(self.eigs - r.eigenvalues[0])), 1e-8)

    def test_qr_and_jacobi(self):
        np.testing.assert_allclose(np.sort(qr_algorithm(self.S).eigenvalues),
                                   self.eigs, atol=1e-8)
        np.testing.assert_allclose(shifted_qr_algorithm(self.S).eigenvalues,
                                   self.eigs, atol=1e-8)
        res = jacobi_eigen(self.S)
        np.testing.assert_allclose(res.eigenvalues, self.eigs, atol=1e-10)
        np.testing.assert_allclose(self.S @ res.eigenvectors,
                                   res.eigenvectors * res.eigenvalues, atol=1e-9)

    def test_krylov(self):
        a, b, Q = lanczos(self.S)
        T = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(T)), self.eigs, atol=1e-8)
        np.testing.assert_allclose(bisection_eigenvalues(a, b), self.eigs, atol=1e-8)

    def test_svd(self):
        rng = np.random.default_rng(2)
        M = rng.random((6, 4))
        ref = np.linalg.svd(M, compute_uv=False)
        for f in (svd_jacobi, svd_golub_kahan):
            U, s, Vt = f(M)
            np.testing.assert_allclose(np.sort(s)[::-1], ref, atol=1e-9)
            np.testing.assert_allclose(U @ np.diag(s) @ Vt, M, atol=1e-8)

    def test_matrix_functions(self):
        A = np.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(
            matrix_exponential(A),
            [[np.cos(1), np.sin(1)], [-np.sin(1), np.cos(1)]], atol=1e-12)
        B = np.array([[4.0, 1.0], [1.0, 3.0]])
        R = matrix_function(B, np.sqrt)
        np.testing.assert_allclose(R @ R, B, atol=1e-10)

    def test_gershgorin_bounds_spectrum(self):
        centers, radii = gershgorin_disks(self.S)
        self.assertTrue(np.all(self.eigs >= np.min(centers - radii) - 1e-9))
        self.assertTrue(np.all(self.eigs <= np.max(centers + radii) + 1e-9))


class TestIterativeSolvers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        n = 30
        A = rng.random((n, n))
        self.A = (A + A.T) / 2 + n * np.eye(n)
        self.b = rng.random(n)
        self.x = np.linalg.solve(self.A, self.b)

    def test_all_solvers_converge(self):
        for f in (jacobi_iteration, gauss_seidel, ssor, richardson,
                  chebyshev_iteration, steepest_descent, conjugate_gradient,
                  preconditioned_cg, minres, gmres, bicg, bicgstab, cgs, cgnr,
                  lsqr):
            with self.subTest(solver=f.__name__):
                res = f(self.A, self.b)
                self.assertTrue(res.converged, res.message)
                np.testing.assert_allclose(res.x, self.x, atol=1e-7)

    def test_divergence_reported_not_raised(self):
        rng = np.random.default_rng(3)
        A = rng.random((20, 20))
        A = A @ A.T + 20 * np.eye(20)   # SPD but not diagonally dominant
        res = jacobi_iteration(A, rng.random(20), max_iter=200)
        self.assertFalse(res.converged)
        self.assertIn("diverged", res.message)

    def test_optimal_sor_beats_gauss_seidel(self):
        T = (np.diag([2.0] * 10) + np.diag([-1.0] * 9, 1) + np.diag([-1.0] * 9, -1))
        b = np.ones(10)
        omega = optimal_sor_omega(T)
        self.assertTrue(1.0 < omega < 2.0)
        self.assertLess(sor(T, b, omega).iterations, gauss_seidel(T, b).iterations)

    def test_preconditioners(self):
        for M in (jacobi_preconditioner(self.A), ssor_preconditioner(self.A)):
            res = preconditioned_cg(self.A, self.b, M=M)
            np.testing.assert_allclose(res.x, self.x, atol=1e-7)


class TestLeastSquaresAndSparse(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.A = rng.random((20, 4))
        self.b = rng.random(20)
        self.x = np.linalg.lstsq(self.A, self.b, rcond=None)[0]

    def test_least_squares_methods_agree(self):
        for f in (normal_equations, qr_least_squares, svd_least_squares,
                  lsqr_least_squares):
            np.testing.assert_allclose(f(self.A, self.b), self.x, atol=1e-8)

    def test_pseudoinverse(self):
        np.testing.assert_allclose(pseudoinverse(self.A), np.linalg.pinv(self.A),
                                   atol=1e-10)

    def test_constrained_and_nonnegative(self):
        C = np.array([[1.0, 1.0, 1.0, 1.0]])
        x = constrained_least_squares(self.A, self.b, C, np.array([1.0]))
        self.assertAlmostEqual(float((C @ x)[0]), 1.0, places=10)
        rng = np.random.default_rng(4)
        An = np.abs(rng.random((10, 3)))
        bn = An @ np.array([1.0, 0.0, 2.0])
        xn = nonnegative_least_squares(An, bn)
        self.assertTrue(np.all(xn >= 0))
        np.testing.assert_allclose(xn, [1.0, 0.0, 2.0], atol=1e-6)

    def test_sparse_formats(self):
        M = np.zeros((6, 6))
        M[0, 0], M[1, 2], M[3, 1], M[5, 5], M[2, 4] = 2.0, 3.0, -1.0, 4.0, 1.5
        v = np.arange(6.0)
        csr = from_dense(M)
        coo = from_dense(M, fmt="coo")
        np.testing.assert_allclose(csr @ v, M @ v)
        np.testing.assert_allclose(coo @ v, M @ v)
        np.testing.assert_allclose(csr.todense(), M)
        np.testing.assert_allclose(csr.rmatvec(v), M.T @ v)
        self.assertEqual(csr.nnz, 5)

    def test_diags_and_bandwidth(self):
        T = (np.diag([2.0] * 5) + np.diag([-1.0] * 4, 1) + np.diag([-1.0] * 4, -1))
        D = diags([[-1.0] * 4, [2.0] * 5, [-1.0] * 4], [-1, 0, 1])
        np.testing.assert_allclose(D.todense(), T)
        self.assertEqual(bandwidth(T), (1, 1))

    def test_reverse_cuthill_mckee_reduces_bandwidth(self):
        B = np.eye(8)
        B[0, 7] = B[7, 0] = 1.0
        B[0, 1] = B[1, 0] = 1.0
        p = reverse_cuthill_mckee(B)
        self.assertLess(bandwidth(B[np.ix_(p, p)])[0], bandwidth(B)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
