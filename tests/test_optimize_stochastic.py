"""Tests for optimization, transforms and stochastic methods."""

import math
import unittest

import numpy as np

from quadrivium.optimize import *
from quadrivium.stochastic import *
from quadrivium.transforms import *


def rosenbrock(v):
    return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2


def rosenbrock_grad(v):
    return np.array([-2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
                     200 * (v[1] - v[0] ** 2)])


def rosenbrock_hess(v):
    return np.array([[2 - 400 * (v[1] - 3 * v[0] ** 2), -400 * v[0]],
                     [-400 * v[0], 200.0]])


class TestScalarOptimization(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: (x - 2.7) ** 2

    def test_all_methods_find_minimum(self):
        for m in (golden_section, ternary_search, brent_minimize,
                  parabolic_interpolation):
            with self.subTest(method=m.__name__):
                self.assertAlmostEqual(m(self.f, 0.0, 6.0).x, 2.7, places=6)
        self.assertAlmostEqual(fibonacci_search(self.f, 0.0, 6.0, 50).x, 2.7, places=6)

    def test_brent_more_efficient_than_golden(self):
        rb = brent_minimize(self.f, 0, 6, tol=1e-12)
        rg = golden_section(self.f, 0, 6, tol=1e-12)
        self.assertLess(rb.function_calls, rg.function_calls)

    def test_newton_1d(self):
        r = newton_minimize_1d(self.f, 0.0, lambda x: 2 * (x - 2.7), lambda x: 2.0)
        self.assertAlmostEqual(r.x, 2.7, places=12)

    def test_bracket_minimum(self):
        a, b, c = bracket_minimum(self.f, 0.0, 1.0)
        self.assertLess(a, 2.7)
        self.assertGreater(c, 2.7)


class TestLineSearches(unittest.TestCase):
    def setUp(self):
        self.x0 = np.array([-1.2, 1.0])
        self.d = -rosenbrock_grad(self.x0)

    def test_all_decrease_the_objective(self):
        alphas = {
            "backtracking": backtracking(rosenbrock, self.x0, self.d,
                                         rosenbrock_grad(self.x0)),
            "goldstein": goldstein(rosenbrock, self.x0, self.d,
                                   rosenbrock_grad(self.x0)),
            "wolfe": wolfe(rosenbrock, rosenbrock_grad, self.x0, self.d),
            "strong_wolfe": strong_wolfe(rosenbrock, rosenbrock_grad, self.x0, self.d),
        }
        for name, a in alphas.items():
            with self.subTest(search=name):
                self.assertLess(rosenbrock(self.x0 + a * self.d), rosenbrock(self.x0))

    def test_strong_wolfe_curvature_condition(self):
        a = strong_wolfe(rosenbrock, rosenbrock_grad, self.x0, self.d, c2=0.9)
        g_new = rosenbrock_grad(self.x0 + a * self.d)
        self.assertLessEqual(abs(g_new @ self.d),
                             0.9 * abs(rosenbrock_grad(self.x0) @ self.d) + 1e-12)

    def test_interpolating_zoom_is_exact_for_quadratics(self):
        Q = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([1.0, -1.0])
        f = lambda v: 0.5 * v @ Q @ v - b @ v
        g = lambda v: Q @ v - b
        x = np.array([0.3, -0.4])
        d = -g(x)
        a_star = -(g(x) @ d) / (d @ Q @ d)
        self.assertAlmostEqual(strong_wolfe(f, g, x, d, c2=0.1), a_star, places=12)


class TestGradientMethods(unittest.TestCase):
    def test_rosenbrock_converges(self):
        methods = [("cg_fr", conjugate_gradient_fr), ("cg_pr", conjugate_gradient_pr),
                   ("cg_hs", conjugate_gradient_hs),
                   ("barzilai_borwein", barzilai_borwein)]
        for name, m in methods:
            with self.subTest(method=name):
                r = m(rosenbrock, [-1.2, 1.0], rosenbrock_grad, tol=1e-8)
                np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-5)

    def test_cg_finite_termination_on_quadratic(self):
        Q = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([1.0, -1.0])
        f = lambda v: 0.5 * v @ Q @ v - b @ v
        g = lambda v: Q @ v - b
        for variant in ("fr", "pr", "hs"):
            r = nonlinear_cg(f, [0.0, 0.0], g, variant, tol=1e-12, max_iter=200)
            self.assertTrue(r.converged)
            self.assertLessEqual(r.iterations, 4)     # n = 2 in exact arithmetic

    def test_barzilai_borwein_robustness(self):
        for x0 in ([-1.2, 1.0], [3.0, -4.0], [0.0, 0.0]):
            for variant in (1, 2):
                r = barzilai_borwein(rosenbrock, x0, rosenbrock_grad, tol=1e-8,
                                     variant=variant)
                np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-5)

    def test_barzilai_borwein_beats_steepest_descent(self):
        rb = barzilai_borwein(rosenbrock, [-1.2, 1.0], rosenbrock_grad, tol=1e-8)
        rg = gradient_descent(rosenbrock, [-1.2, 1.0], rosenbrock_grad,
                              line_search=True, tol=1e-8, max_iter=50000)
        self.assertLess(rb.iterations, rg.iterations / 10)

    def test_adaptive_rules_on_quadratic(self):
        Q = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([1.0, -1.0])
        f = lambda v: 0.5 * v @ Q @ v - b @ v
        g = lambda v: Q @ v - b
        x = np.linalg.solve(Q, b)
        for name, run in (("gradient_descent", lambda: gradient_descent(f, [0.0, 0.0], g, lr=0.2, tol=1e-10, max_iter=20000)),
                          ("momentum", lambda: momentum(f, [0.0, 0.0], g, lr=0.05, tol=1e-10, max_iter=20000)),
                          ("nesterov", lambda: nesterov(f, [0.0, 0.0], g, lr=0.05, tol=1e-10, max_iter=20000)),
                          ("adagrad", lambda: adagrad(f, [0.0, 0.0], g, lr=0.5, tol=1e-10, max_iter=50000)),
                          ("rmsprop", lambda: rmsprop(f, [0.0, 0.0], g, lr=0.05, tol=1e-10, max_iter=50000)),
                          ("adam", lambda: adam(f, [0.0, 0.0], g, lr=0.05, tol=1e-10, max_iter=50000))):
            with self.subTest(method=name):
                np.testing.assert_allclose(run().x, x, atol=1e-6)


class TestQuasiNewton(unittest.TestCase):
    def test_all_methods_on_rosenbrock(self):
        methods = [("newton", lambda: newton_method(rosenbrock, [-1.2, 1.0], rosenbrock_grad, rosenbrock_hess, tol=1e-10)),
                   ("modified_newton", lambda: modified_newton(rosenbrock, [-1.2, 1.0], rosenbrock_grad, rosenbrock_hess, tol=1e-10)),
                   ("bfgs", lambda: bfgs(rosenbrock, [-1.2, 1.0], rosenbrock_grad, tol=1e-10)),
                   ("dfp", lambda: dfp(rosenbrock, [-1.2, 1.0], rosenbrock_grad, tol=1e-10)),
                   ("sr1", lambda: sr1(rosenbrock, [-1.2, 1.0], rosenbrock_grad, tol=1e-10)),
                   ("broyden", lambda: broyden_class(rosenbrock, [-1.2, 1.0], rosenbrock_grad, tol=1e-10)),
                   ("lbfgs", lambda: lbfgs(rosenbrock, [-1.2, 1.0], rosenbrock_grad, tol=1e-10))]
        for name, run in methods:
            with self.subTest(method=name):
                np.testing.assert_allclose(run().x, [1.0, 1.0], atol=1e-6)

    def test_newton_quadratic_convergence(self):
        r = newton_method(rosenbrock, [0.9, 0.8], rosenbrock_grad, rosenbrock_hess,
                          tol=1e-12)
        self.assertLessEqual(r.iterations, 8)

    def test_lbfgs_scales(self):
        n = 200
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        A = A @ A.T / n + np.eye(n)
        b = rng.random(n)
        f = lambda v: 0.5 * v @ A @ v - b @ v
        g = lambda v: A @ v - b
        r = lbfgs(f, np.zeros(n), g, m=10, tol=1e-8, max_iter=2000)
        np.testing.assert_allclose(r.x, np.linalg.solve(A, b), atol=1e-6)

    def test_standard_test_functions(self):
        beale = lambda v: ((1.5 - v[0] + v[0] * v[1]) ** 2
                           + (2.25 - v[0] + v[0] * v[1] ** 2) ** 2
                           + (2.625 - v[0] + v[0] * v[1] ** 3) ** 2)
        np.testing.assert_allclose(bfgs(beale, [1.0, 1.0], tol=1e-10).x,
                                   [3.0, 0.5], atol=1e-4)
        him = lambda v: (v[0] ** 2 + v[1] - 11) ** 2 + (v[0] + v[1] ** 2 - 7) ** 2
        self.assertLess(lbfgs(him, [0.0, 0.0], tol=1e-10).fun, 1e-12)

    def test_numerical_gradient_fallback(self):
        np.testing.assert_allclose(bfgs(rosenbrock, [-1.2, 1.0], tol=1e-6).x,
                                   [1.0, 1.0], atol=1e-3)


class TestTrustRegionAndLeastSquares(unittest.TestCase):
    def test_subproblem_solvers_respect_radius(self):
        g = np.array([1.0, 2.0])
        B = np.array([[2.0, 0.0], [0.0, 1.0]])
        for solver in (cauchy_point, dogleg, steihaug_cg):
            for delta in (0.1, 1.0, 10.0):
                with self.subTest(solver=solver.__name__, delta=delta):
                    self.assertLessEqual(np.linalg.norm(solver(g, B, delta)),
                                         delta * (1 + 1e-9))

    def test_steihaug_handles_negative_curvature(self):
        B = np.array([[1.0, 0.0], [0.0, -1.0]])
        p = steihaug_cg(np.array([0.1, 0.1]), B, 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(p)), 1.0, places=9)

    def test_trust_region_on_rosenbrock(self):
        for sub in ("dogleg", "steihaug"):
            with self.subTest(subproblem=sub):
                r = trust_region(rosenbrock, [-1.2, 1.0], rosenbrock_grad,
                                 rosenbrock_hess, tol=1e-10, subproblem=sub)
                np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-5)

    def test_escapes_saddle_point(self):
        f = lambda v: v[0] ** 2 - v[1] ** 2 + 0.1 * v[1] ** 4
        g = lambda v: np.array([2 * v[0], -2 * v[1] + 0.4 * v[1] ** 3])
        h = lambda v: np.array([[2.0, 0.0], [0.0, -2 + 1.2 * v[1] ** 2]])
        for sub in ("dogleg", "steihaug", "cauchy"):
            for x0 in ([0.5, 0.1], [0.5, -0.3], [0.1, 0.05]):
                with self.subTest(subproblem=sub, x0=tuple(x0)):
                    r = trust_region(f, x0, g, h, tol=1e-9, max_iter=3000,
                                     subproblem=sub)
                    self.assertAlmostEqual(abs(r.x[1]), np.sqrt(5), places=4)

    def test_nonlinear_least_squares(self):
        rng = np.random.default_rng(0)
        xd = np.linspace(0, 2, 40)
        true = [2.5, -1.3]
        yd = true[0] * np.exp(true[1] * xd) + 0.01 * rng.standard_normal(40)
        resid = lambda p: p[0] * np.exp(p[1] * xd) - yd
        for method in ("lm", "gn"):
            with self.subTest(method=method):
                r = nonlinear_least_squares(resid, [1.0, -0.5], method=method)
                np.testing.assert_allclose(r.x, true, atol=0.05)

    def test_curve_fit_reports_uncertainties(self):
        rng = np.random.default_rng(0)
        xd = np.linspace(0, 2, 40)
        yd = 2.5 * np.exp(-1.3 * xd) + 0.01 * rng.standard_normal(40)
        r = curve_fit(lambda x, a, b: a * np.exp(b * x), xd, yd, [1.0, -0.5])
        np.testing.assert_allclose(r.x, [2.5, -1.3], atol=0.05)
        self.assertTrue(np.all(r.std_errors < 0.05))


class TestDerivativeFree(unittest.TestCase):
    def test_all_methods_on_rosenbrock(self):
        methods = [("nelder_mead", lambda: nelder_mead(rosenbrock, [-1.2, 1.0], tol=1e-12, max_iter=10000)),
                   ("powell", lambda: powell(rosenbrock, [-1.2, 1.0], tol=1e-13)),
                   ("hooke_jeeves", lambda: hooke_jeeves(rosenbrock, [-1.2, 1.0], step=0.5, tol=1e-10)),
                   ("compass_search", lambda: compass_search(rosenbrock, [-1.2, 1.0], step=0.5, tol=1e-10))]
        for name, run in methods:
            with self.subTest(method=name):
                np.testing.assert_allclose(run().x, [1.0, 1.0], atol=1e-4)

    def test_nonsmooth_objective(self):
        f = lambda v: abs(v[0] - 1) + abs(v[1] - 2) + 0.5 * abs(v[0] + v[1] - 3)
        for run in (lambda: nelder_mead(f, [0.0, 0.0], tol=1e-12, max_iter=20000),
                    lambda: hooke_jeeves(f, [0.0, 0.0], step=1.0, tol=1e-12),
                    lambda: compass_search(f, [0.0, 0.0], step=1.0, tol=1e-12)):
            self.assertLess(run().fun, 1e-5)

    def test_higher_dimension(self):
        sphere = lambda v: float(np.sum((np.asarray(v) - np.arange(1, 6)) ** 2))
        for run in (lambda: nelder_mead(sphere, np.zeros(5), tol=1e-12, max_iter=20000),
                    lambda: powell(sphere, np.zeros(5), tol=1e-13),
                    lambda: coordinate_descent(sphere, np.zeros(5), tol=1e-12)):
            np.testing.assert_allclose(run().x, np.arange(1, 6), atol=1e-5)


class TestGlobalOptimization(unittest.TestCase):
    @staticmethod
    def rastrigin(v):
        v = np.asarray(v)
        return 10 * len(v) + np.sum(v**2 - 10 * np.cos(2 * np.pi * v))

    @staticmethod
    def ackley(v):
        v = np.asarray(v)
        n = len(v)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(v**2) / n))
                - np.exp(np.sum(np.cos(2 * np.pi * v)) / n) + 20 + np.e)

    def test_population_methods_find_global_optimum(self):
        b3 = [(-5.12, 5.12)] * 3
        runs = [("particle_swarm", lambda: particle_swarm(self.rastrigin, b3, 40, 300, rng=0)),
                ("differential_evolution", lambda: differential_evolution(self.rastrigin, b3, 30, max_iter=300, rng=0)),
                ("cma_es", lambda: cma_es(self.rastrigin, [2.0, 2.0, 2.0], sigma0=2.0, pop_size=50, max_iter=2000, rng=0)),
                ("basin_hopping", lambda: basin_hopping(self.rastrigin, [4.0, 4.0, 4.0], 60, step=1.5, rng=0)),
                ("dual_annealing", lambda: dual_annealing_lite(self.rastrigin, b3, 20000, rng=0))]
        for name, run in runs:
            with self.subTest(method=name):
                self.assertLess(run().fun, 1e-4)

    def test_ackley(self):
        b2 = [(-32.0, 32.0)] * 2
        for run in (lambda: differential_evolution(self.ackley, b2, 30, max_iter=300, rng=1),
                    lambda: particle_swarm(self.ackley, b2, 40, 300, rng=1),
                    lambda: cma_es(self.ackley, [10.0, 10.0], sigma0=8.0, max_iter=2000, rng=1)):
            self.assertLess(run().fun, 1e-3)

    def test_beats_random_search(self):
        b3 = [(-5.12, 5.12)] * 3
        baseline = random_search(self.rastrigin, b3, 20000, rng=0).fun
        self.assertLess(differential_evolution(self.rastrigin, b3, 30, max_iter=300,
                                               rng=0).fun, baseline)

    def test_cma_es_on_ill_conditioned_problem(self):
        def elli(v):
            v = np.asarray(v)
            n = len(v)
            return float(np.sum((1e6 ** (np.arange(n) / (n - 1))) * v**2))
        r = cma_es(elli, np.ones(10), sigma0=1.0, max_iter=3000, rng=0)
        self.assertLess(r.fun, 1e-8)


class TestConstrainedOptimization(unittest.TestCase):
    def setUp(self):
        self.f = lambda v: v[0] ** 2 + v[1] ** 2
        self.eq = lambda v: np.array([v[0] + v[1] - 1.0])
        self.ineq = lambda v: np.array([2.0 - v[0] - v[1]])

    def test_equality_constraint(self):
        for run in (lambda: penalty_method(self.f, [2.0, -1.0], eq=self.eq),
                    lambda: augmented_lagrangian(self.f, [2.0, -1.0], eq=self.eq),
                    lambda: sqp(self.f, [2.0, -1.0], eq=self.eq)):
            np.testing.assert_allclose(run().x, [0.5, 0.5], atol=1e-6)

    def test_inequality_constraint(self):
        for run in (lambda: penalty_method(self.f, [0.0, 0.0], ineq=self.ineq),
                    lambda: augmented_lagrangian(self.f, [0.0, 0.0], ineq=self.ineq),
                    lambda: sqp(self.f, [0.0, 0.0], ineq=self.ineq),
                    lambda: barrier_method(self.f, [3.0, 3.0], ineq=self.ineq)):
            np.testing.assert_allclose(run().x, [1.0, 1.0], atol=1e-4)

    def test_barrier_requires_feasible_start(self):
        with self.assertRaises(ValueError):
            barrier_method(self.f, [0.0, 0.0], ineq=self.ineq)

    def test_mixed_constraints(self):
        f = lambda v: (v[0] - 2) ** 2 + (v[1] - 1) ** 2
        for x0 in ([0.0, 0.0], [1.0, 1.0], [3.0, -1.0]):
            r = sqp(f, x0, eq=lambda v: np.array([v[0] + v[1] - 2.0]),
                    ineq=lambda v: np.array([v[0] - 1.5]))
            np.testing.assert_allclose(r.x, [1.5, 0.5], atol=1e-6)

    def test_projections(self):
        np.testing.assert_allclose(project_box([1.0, -3.0, 5.0], -1, 2), [1, -1, 2])
        p = project_simplex([0.5, 0.3, 0.9])
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)
        self.assertTrue(np.all(p >= 0))
        np.testing.assert_allclose(project_ball([3.0, 4.0], 1.0), [0.6, 0.8])

    def test_projected_gradient_stays_feasible(self):
        q = lambda v: float(np.sum((v - np.array([0.8, 0.1, 0.4])) ** 2))
        r = projected_gradient(q, [1 / 3, 1 / 3, 1 / 3], project_simplex, lr=1.0,
                               tol=1e-12)
        self.assertAlmostEqual(float(r.x.sum()), 1.0, places=10)
        self.assertTrue(np.all(r.x >= -1e-12))

    def test_quadratic_programming(self):
        G = np.array([[2.0, 0.0], [0.0, 2.0]])
        c = np.array([-2.0, -5.0])
        x, lam = solve_qp(G, c, np.array([[1.0, 1.0]]), np.array([3.0]))
        np.testing.assert_allclose(x, [0.75, 2.25], atol=1e-12)
        A = np.array([[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0], [-1.0, 0.0], [0.0, -1.0]])
        b = np.array([2.0, 6.0, 2.0, 0.0, 0.0])
        for start in ([2.0, 0.0], [0.0, 0.0], None, [10.0, 10.0]):
            with self.subTest(start=start):
                x0 = None if start is None else np.array(start)
                r = active_set_qp(G, c, A, b, x0=x0)
                np.testing.assert_allclose(r.x, [1.4, 1.7], atol=1e-8)


class TestProximalMethods(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.m, self.n, self.k = 50, 100, 5
        self.A = rng.standard_normal((self.m, self.n)) / np.sqrt(self.m)
        self.x_true = np.zeros(self.n)
        self.idx = rng.choice(self.n, self.k, replace=False)
        self.x_true[self.idx] = rng.standard_normal(self.k) * 2 + 1
        self.b = self.A @ self.x_true + 0.01 * rng.standard_normal(self.m)
        self.lam = 0.05

    def test_prox_operators(self):
        np.testing.assert_allclose(soft_threshold([3.0, -0.5, 0.2], 1.0), [2.0, 0, 0])
        np.testing.assert_allclose(prox_l2([3.0, 4.0], 1.0), [2.4, 3.2])
        np.testing.assert_allclose(prox_l2([0.6, 0.8], 1.0), [0.0, 0.0])
        np.testing.assert_allclose(prox_nonneg([-1.0, 2.0]), [0, 2])

    def test_lasso_solvers_agree(self):
        objs = []
        for run in (lambda: lasso(self.A, self.b, self.lam, accelerate=False, tol=1e-14, max_iter=200000),
                    lambda: lasso(self.A, self.b, self.lam, tol=1e-14, max_iter=200000),
                    lambda: admm_lasso(self.A, self.b, self.lam, tol=1e-14, max_iter=200000)):
            r = run()
            objs.append(r.fun)
            support = set(np.flatnonzero(np.abs(r.x) > 1e-6))
            self.assertTrue(set(self.idx) <= support)
        self.assertLess(max(objs) - min(objs), 1e-10)

    def test_fista_accelerates(self):
        star = lasso(self.A, self.b, self.lam, tol=1e-14, max_iter=200000).fun
        for budget in (20, 50, 100):
            gap_ista = lasso(self.A, self.b, self.lam, accelerate=False, tol=0.0,
                             max_iter=budget).fun - star
            gap_fista = lasso(self.A, self.b, self.lam, accelerate=True, tol=0.0,
                              max_iter=budget).fun - star
            self.assertLess(gap_fista, gap_ista)

    def test_ridge_is_dense_lasso_is_sparse(self):
        dense = ridge(self.A, self.b, 1.0)
        sparse = lasso(self.A, self.b, self.lam, tol=1e-12)
        self.assertGreater(np.sum(np.abs(dense.x) > 1e-6), 50)
        self.assertLess(np.sum(np.abs(sparse.x) > 1e-6), 15)

    def test_admm_and_douglas_rachford(self):
        a = np.array([3.0, -0.5, 0.2])
        r = admm(lambda v, t: (v + t * a) / (1 + t),
                 lambda v, t: soft_threshold(v, t), np.zeros(3), rho=1.0)
        np.testing.assert_allclose(r.x, soft_threshold(a, 1.0), atol=1e-6)
        d = douglas_rachford(lambda v, t: prox_box(v, -1, 1),
                             lambda v, t: prox_nonneg(v), np.array([5.0, -5.0, 0.3]))
        self.assertTrue(np.all(d.x >= -1e-9) and np.all(d.x <= 1 + 1e-9))


class TestLinearProgramming(unittest.TestCase):
    def test_simplex_and_interior_point_agree(self):
        c = np.array([-3.0, -5.0])
        A = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
        b = np.array([4.0, 12.0, 18.0])
        for method in ("simplex", "interior_point"):
            with self.subTest(method=method):
                r = linprog(c, A, b, method=method)
                np.testing.assert_allclose(r.x, [2.0, 6.0], atol=1e-5)

    def test_random_problems_agree(self):
        rng = np.random.default_rng(0)
        agree = 0
        for _ in range(15):
            c = rng.standard_normal(4)
            A = np.abs(rng.standard_normal((5, 4))) + 0.1
            b = np.abs(rng.standard_normal(5)) * 5 + 1
            s = simplex(c, A, b)
            i = interior_point_lp(c, A, b, max_iter=300)
            if s.converged and abs(s.fun - i.fun) < 1e-4:
                agree += 1
        self.assertGreaterEqual(agree, 14)

    def test_infeasible_and_unbounded_detection(self):
        r = simplex(np.array([1.0, 1.0]), A_eq=np.array([[1.0, 1.0]]),
                    b_eq=np.array([-5.0]))
        self.assertFalse(r.converged)
        self.assertIn("infeasible", r.message)
        r = simplex(np.array([-1.0, 0.0]), A_ub=np.array([[0.0, 1.0]]),
                    b_ub=np.array([1.0]))
        self.assertIn("unbounded", r.message)

    def test_equality_constrained_lp(self):
        c = np.array([2.0, 3.0, 1.0])
        Aeq = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
        beq = np.array([10.0, 2.0])
        r = simplex(c, A_eq=Aeq, b_eq=beq)
        np.testing.assert_allclose(Aeq @ r.x, beq, atol=1e-9)
        self.assertTrue(np.all(r.x >= -1e-9))

    def test_assignment_matches_brute_force(self):
        import itertools
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(2, 6))
            M = rng.integers(0, 20, (n, n)).astype(float)
            _, _, total = assignment_problem(M)
            best = min(sum(M[i, p[i]] for i in range(n))
                       for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(total, best, places=9)


class TestTransforms(unittest.TestCase):
    def test_fft_matches_reference_all_lengths(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 31, 32, 60, 64, 97, 128):
            with self.subTest(n=n):
                x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
                np.testing.assert_allclose(fft(x), np.fft.fft(x), atol=1e-9)
                np.testing.assert_allclose(ifft(fft(x)), x, atol=1e-10)

    def test_algorithm_variants_agree(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(64)
        np.testing.assert_allclose(dft(x), np.fft.fft(x), atol=1e-10)
        np.testing.assert_allclose(fft_radix2(x), np.fft.fft(x), atol=1e-10)
        np.testing.assert_allclose(fft_bluestein(x), np.fft.fft(x), atol=1e-10)
        for n in (12, 60, 100):
            y = rng.standard_normal(n)
            np.testing.assert_allclose(fft_mixed_radix(y), np.fft.fft(y), atol=1e-9)

    def test_real_transforms_round_trip(self):
        rng = np.random.default_rng(0)
        for n in (7, 8, 15, 16, 31, 32, 64, 65):
            with self.subTest(n=n):
                x = rng.standard_normal(n)
                r = rfft(x)
                np.testing.assert_allclose(r, np.fft.rfft(x), atol=1e-10)
                np.testing.assert_allclose(irfft(r, n), x, atol=1e-10)

    def test_dct_dst_definitions_and_inverses(self):
        rng = np.random.default_rng(0)
        N = 16
        x = rng.standard_normal(N)
        n = np.arange(N)
        refs = {
            1: np.array([x[0] + (-1) ** k * x[-1]
                         + 2 * np.sum(x[1:-1] * np.cos(np.pi * np.arange(1, N - 1) * k / (N - 1)))
                         for k in range(N)]),
            2: np.array([2 * np.sum(x * np.cos(np.pi * k * (2 * n + 1) / (2 * N)))
                         for k in range(N)]),
            3: np.array([x[0] + 2 * np.sum(x[1:] * np.cos(np.pi * np.arange(1, N) * (2 * k + 1) / (2 * N)))
                         for k in range(N)]),
            4: np.array([2 * np.sum(x * np.cos(np.pi * (2 * k + 1) * (2 * n + 1) / (4 * N)))
                         for k in range(N)]),
        }
        for kind, ref in refs.items():
            with self.subTest(dct=kind):
                np.testing.assert_allclose(dct(x, kind), ref, atol=1e-10)
                np.testing.assert_allclose(idct(dct(x, kind), kind), x, atol=1e-12)
        s1 = np.array([2 * np.sum(x * np.sin(np.pi * (n + 1) * (k + 1) / (N + 1)))
                       for k in range(N)])
        np.testing.assert_allclose(dst(x, 1), s1, atol=1e-10)

    def test_parseval(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(64)
        self.assertAlmostEqual(float(np.sum(y**2)),
                               float(np.sum(np.abs(fft(y)) ** 2) / 64), places=10)

    def test_convolution(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(40)
        b = rng.standard_normal(15)
        for mode in ("full", "same", "valid"):
            with self.subTest(mode=mode):
                ref = np.convolve(a, b, mode)
                np.testing.assert_allclose(convolve(a, b, mode), ref, atol=1e-10)
                np.testing.assert_allclose(convolve_fft(a, b, mode), ref, atol=1e-10)

    def test_spectral_estimation(self):
        rng = np.random.default_rng(0)
        fs = 1000.0
        dt = 1 / fs
        n = 4096
        t = np.arange(n) * dt
        sig = (2 * np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
               + 0.1 * rng.standard_normal(n))
        f, p = power_spectrum(sig, dt)
        # Parseval: integrating the PSD recovers the variance
        self.assertAlmostEqual(float(np.trapezoid(p, f)) / float(np.var(sig)), 1.0,
                               places=1)
        for target in (50, 120):
            i = int(np.argmin(np.abs(f - target)))
            self.assertGreater(p[max(i - 2, 0):i + 3].max() / np.median(p), 1000)

    def test_welch_reduces_variance(self):
        rng = np.random.default_rng(0)
        sig = rng.standard_normal(4096)
        _, praw = periodogram(sig)
        _, pw = welch(sig, 512, 0.5)
        self.assertLess(np.var(np.log(pw[10:] + 1e-20)),
                        np.var(np.log(praw[10:] + 1e-20)))

    def test_hilbert_envelope(self):
        t = np.linspace(0, 4, 4000)
        env = 1 + 0.5 * np.cos(2 * np.pi * 3 * t)
        am = env * np.cos(2 * np.pi * 100 * t)
        e = np.abs(hilbert(am))
        self.assertLess(np.max(np.abs(e[200:-200] - env[200:-200])), 0.05)

    def test_resample_exact_for_tone(self):
        up = resample(np.sin(2 * np.pi * np.arange(64) / 64), 128)
        np.testing.assert_allclose(up, np.sin(2 * np.pi * np.arange(128) / 128),
                                   atol=1e-10)


class TestGenerators(unittest.TestCase):
    def test_uniformity(self):
        for name, g in (("LCG", LCG(12345)), ("ParkMiller", ParkMiller(1)),
                        ("XorShift", XorShift()),
                        ("MersenneTwister", MersenneTwister(5489))):
            with self.subTest(generator=name):
                u = g.random(20000)
                self.assertLess(abs(u.mean() - 0.5), 0.02)
                self.assertLess(abs(u.var() - 1 / 12), 0.005)
                self.assertTrue(np.all((u >= 0) & (u < 1)))

    def test_chi_square_uniformity(self):
        for g in (LCG(7), MersenneTwister(42), XorShift(999)):
            u = g.random(50000)
            counts, _ = np.histogram(u, bins=50, range=(0, 1))
            chi2 = np.sum((counts - 1000) ** 2 / 1000)
            self.assertLess(chi2, 100)      # 49 dof, 5% critical value 66.3

    def test_mt19937_reference_stream(self):
        mt = MersenneTwister(5489)
        self.assertEqual([mt.next_int() for _ in range(3)],
                         [3499211612, 581869302, 3890346734])

    def test_van_der_corput(self):
        np.testing.assert_allclose(
            van_der_corput(8, 2),
            [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625])

    def test_low_discrepancy_beats_random(self):
        def star_discrepancy(pts):
            worst = 0.0
            for t in np.linspace(0.05, 0.95, 40):
                frac = np.mean(np.all(pts <= t, axis=1))
                worst = max(worst, abs(frac - t ** pts.shape[1]))
            return worst
        rng = np.random.default_rng(0)
        self.assertLess(star_discrepancy(halton(500, 2)),
                        star_discrepancy(rng.random((500, 2))))

    def test_spectral_test_detects_randu(self):
        randu = spectral_test(LCG(1, a=65539, c=0, m=2**31), 1500, dim=3)
        good = spectral_test(MersenneTwister(1), 1500, dim=3)
        self.assertTrue(randu["lattice_detected"])
        self.assertFalse(good["lattice_detected"])
        np.testing.assert_array_equal(randu["coefficients"], [9, -6, 1])


class TestSampling(unittest.TestCase):
    def test_normal_generators(self):
        for f in (box_muller, marsaglia_polar):
            with self.subTest(method=f.__name__):
                z = f(200000, rng=0)
                self.assertLess(abs(z.mean()), 0.02)
                self.assertLess(abs(z.std() - 1), 0.02)
                kurt = np.mean((z - z.mean()) ** 4) / z.std() ** 4
                self.assertLess(abs(kurt - 3), 0.1)

    def test_inverse_transform(self):
        x = inverse_transform(lambda u: -np.log(1 - u) / 2.0, 100000, rng=0)
        self.assertLess(abs(x.mean() - 0.5), 0.01)

    def test_rejection_and_ratio_of_uniforms(self):
        pdf = lambda t: np.exp(-t * t / 2) / np.sqrt(2 * np.pi)
        s, acc = rejection_sampling(pdf, lambda r: r.uniform(-5, 5),
                                    lambda t: 0.1, M=4.0, n=20000, rng=0)
        self.assertLess(abs(s.mean()), 0.05)
        self.assertLess(abs(s.std() - 1), 0.05)
        s2, _ = ratio_of_uniforms(lambda t: np.exp(-t * t / 2), 20000,
                                  v_range=(-2, 2), rng=0)
        self.assertLess(abs(s2.std() - 1), 0.05)

    def test_adaptive_rejection(self):
        z = adaptive_rejection(lambda t: -t * t / 2, [-1.0, 0.0, 1.0], 30000,
                               rng=0, dlog_pdf=lambda t: -t)
        self.assertLess(abs(z.mean()), 0.03)
        self.assertLess(abs(z.std() - 1), 0.03)
        g = adaptive_rejection(lambda t: 2 * np.log(t) - t, [0.5, 2.0, 5.0], 20000,
                               domain=(0, np.inf), rng=0,
                               dlog_pdf=lambda t: 2 / t - 1)
        self.assertLess(abs(g.mean() - 3), 0.1)

    def test_adaptive_rejection_rejects_bad_start(self):
        with self.assertRaises(ValueError):
            adaptive_rejection(lambda t: -t * t / 2, [1.0, 2.0, 3.0], 100, rng=0,
                               dlog_pdf=lambda t: -t)

    def test_discrete_sampling(self):
        p = np.array([0.1, 0.5, 0.2, 0.2])
        for method in ("alias", "inverse"):
            with self.subTest(method=method):
                d = sample_discrete(p, 200000, rng=0, method=method)
                freq = np.bincount(d, minlength=4) / 200000
                self.assertLess(np.max(np.abs(freq - p)), 0.01)

    def test_multivariate_normal(self):
        mean = np.array([1.0, -2.0])
        cov = np.array([[2.0, 0.8], [0.8, 1.0]])
        mv = multivariate_normal(mean, cov, 200000, rng=0)
        np.testing.assert_allclose(mv.mean(axis=0), mean, atol=0.02)
        np.testing.assert_allclose(np.cov(mv.T), cov, atol=0.03)

    def test_singular_covariance(self):
        s = multivariate_normal([0, 0], np.array([[1.0, 1.0], [1.0, 1.0]]), 10000,
                                rng=0)
        self.assertLess(np.max(np.abs(s[:, 0] - s[:, 1])), 1e-8)

    def test_bootstrap_and_jackknife(self):
        rng = np.random.default_rng(0)
        data = rng.normal(5, 2, 200)
        analytic = data.std(ddof=1) / np.sqrt(200)
        bs = bootstrap(data, np.mean, 5000, rng=0)
        self.assertLess(abs(bs["std_error"] - analytic), 0.02)
        jk = jackknife(data, np.mean)
        self.assertAlmostEqual(jk["std_error"], analytic, places=10)

    def test_permutation_test(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 60)
        b = rng.normal(1.0, 1, 60)
        c = rng.normal(0, 1, 60)
        self.assertLess(permutation_test(a, b, rng=0)["p_value"], 0.01)
        self.assertGreater(permutation_test(a, c, rng=0)["p_value"], 0.05)


class TestMCMC(unittest.TestCase):
    def setUp(self):
        self.lt = lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2))
        self.gl = lambda x: -np.asarray(x, dtype=float)

    @staticmethod
    def col(chain):
        a = np.asarray(chain)
        return a[:, 0] if a.ndim > 1 else a

    def test_samplers_target_standard_normal(self):
        runs = [("random_walk", lambda: random_walk_metropolis(self.lt, [0.0], step=2.4, n=40000, burn=2000, rng=0)),
                ("hmc", lambda: hamiltonian_mc(self.lt, self.gl, [0.0], step=0.15, n_leapfrog=20, n=20000, burn=1000, rng=0)),
                ("slice", lambda: slice_sampler(self.lt, 0.0, w=2.0, n=20000, burn=1000, rng=0)),
                ("nuts_lite", lambda: nuts_lite(self.lt, self.gl, [0.0], step=0.2, n=8000, burn=500, rng=0))]
        for name, run in runs:
            with self.subTest(sampler=name):
                x = self.col(run())
                self.assertLess(abs(x.mean()), 0.07)
                self.assertLess(abs(x.std() - 1), 0.07)

    def test_hmc_more_efficient_than_random_walk(self):
        rw = self.col(random_walk_metropolis(self.lt, [0.0], step=2.4, n=20000,
                                             burn=1000, rng=0))
        hm = self.col(hamiltonian_mc(self.lt, self.gl, [0.0], step=0.15,
                                     n_leapfrog=20, n=10000, burn=500, rng=0))
        self.assertGreater(effective_sample_size(hm) / len(hm),
                           effective_sample_size(rw) / len(rw))

    def test_correlated_target(self):
        S = np.array([[1.0, 0.8], [0.8, 1.0]])
        Si = np.linalg.inv(S)
        h = hamiltonian_mc(lambda x: -0.5 * float(np.asarray(x) @ Si @ np.asarray(x)),
                           lambda x: -Si @ np.asarray(x, dtype=float),
                           [0.0, 0.0], step=0.15, n_leapfrog=25, n=20000, burn=1000,
                           rng=0)
        np.testing.assert_allclose(np.cov(np.asarray(h).T), S, atol=0.06)

    def test_gibbs(self):
        g = gibbs_sampler([lambda x, r: r.normal(0.8 * x[1], 0.6),
                           lambda x, r: r.normal(0.8 * x[0], 0.6)],
                          [0.0, 0.0], n=40000, burn=1000, rng=0)
        self.assertLess(abs(np.corrcoef(g.T)[0, 1] - 0.8), 0.02)

    def test_gelman_rubin_diagnostic(self):
        chains = np.array([self.col(random_walk_metropolis(self.lt, [x0], step=2.4,
                                                           n=5000, burn=500, rng=i))
                           for i, x0 in enumerate([-2.0, 0.0, 2.0, 1.0])])
        self.assertLess(gelman_rubin(chains), 1.02)
        stuck = np.array([np.full(5000, v) + 1e-6 * np.random.default_rng(i).standard_normal(5000)
                          for i, v in enumerate([-3.0, 0.0, 3.0, 6.0])])
        self.assertGreater(gelman_rubin(stuck), 2.0)

    def test_parallel_tempering_crosses_modes(self):
        bimodal = lambda x: float(np.logaddexp(-0.5 * (np.asarray(x)[0] - 5) ** 2,
                                               -0.5 * (np.asarray(x)[0] + 5) ** 2))
        pt = parallel_tempering(bimodal, [5.0], temperatures=(1.0, 3.0, 9.0, 27.0),
                                step=1.5, n=20000, burn=2000, rng=0)
        self.assertTrue((pt < 0).any() and (pt > 0).any())


class TestStatistics(unittest.TestCase):
    def test_descriptive(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 2, 5000)
        d = describe(x)
        self.assertLess(abs(d["mean"] - 5), 0.1)
        self.assertLess(abs(d["std"] - 2), 0.1)
        self.assertLess(abs(d["skewness"]), 0.1)

    def test_welford_stable(self):
        big = np.array([1e9 + 1, 1e9 + 2, 1e9 + 3, 1e9 + 4])
        self.assertAlmostEqual(welford_mean_var(big)[1], np.var(big, ddof=1),
                               places=9)

    def test_regression(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 3))
        y = 1.5 + X @ np.array([2.0, -1.0, 0.5]) + 0.3 * rng.normal(size=200)
        r = linear_regression(X, y)
        np.testing.assert_allclose(r["coefficients"], [1.5, 2, -1, 0.5], atol=0.1)
        self.assertGreater(r["r_squared"], 0.95)

    def test_logistic_regression_consistency(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50000, 2))
        p = 1 / (1 + np.exp(-(0.5 + 1.5 * X[:, 0] - 2 * X[:, 1])))
        y = (rng.random(50000) < p).astype(float)
        r = logistic_regression(X, y)
        np.testing.assert_allclose(r["coefficients"], [0.5, 1.5, -2.0], atol=0.05)

    def test_pca(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(0, 1, (1000, 2)) @ np.array([[3.0, 1.0], [0.0, 0.5]])
        p = pca(Z)
        self.assertAlmostEqual(float(p["explained_variance_ratio"].sum()), 1.0,
                               places=12)
        self.assertGreater(p["explained_variance_ratio"][0], 0.8)

    def test_hypothesis_tests(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 80)
        b = rng.normal(0.8, 1, 80)
        c = rng.normal(0, 1, 80)
        self.assertLess(t_test(a, b)["p_value"], 0.001)
        self.assertGreater(t_test(a, c)["p_value"], 0.05)
        self.assertLess(t_test(a, b, equal_var=False)["p_value"], 0.001)
        self.assertLess(chi_square_test(np.array([50.0, 10, 10, 10, 20]))["p_value"],
                        0.001)
        self.assertGreater(chi_square_test(np.array([20.0, 18, 22, 19, 21]))["p_value"],
                           0.05)
        self.assertLess(anova_one_way(a, b, rng.normal(2, 1, 80))["p_value"], 0.001)
        self.assertGreater(anova_one_way(a, c, rng.normal(0, 1, 80))["p_value"], 0.05)

    def test_ks_test_calibration(self):
        from quadrivium.special import erf
        cdf = lambda t: 0.5 * (1 + erf(t / np.sqrt(2)))
        ps = np.array([ks_test(np.random.default_rng(i).normal(0, 1, 100), cdf)["p_value"]
                       for i in range(300)])
        self.assertGreater(ps.mean(), 0.42)      # uniform under the null
        self.assertLess(ps.mean(), 0.58)
        self.assertLess(np.mean(ps < 0.05), 0.12)
        rng = np.random.default_rng(0)
        self.assertLess(ks_test(rng.normal(0, 1, 80) + 3, cdf)["p_value"], 0.001)

    def test_confidence_intervals(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 2, 5000)
        for method in ("t", "normal", "bootstrap"):
            with self.subTest(method=method):
                lo, hi = confidence_interval(x, method=method)
                self.assertLess(lo, 5.0)
                self.assertGreater(hi, 5.0)

    def test_kernel_density_normalization(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 2, 5000)
        for kernel in ("gaussian", "epanechnikov", "uniform", "triangular"):
            with self.subTest(kernel=kernel):
                pts, dens = kernel_density(x, kernel=kernel)
                self.assertAlmostEqual(float(np.trapezoid(dens, pts)), 1.0, places=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
