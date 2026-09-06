"""Tests for ODE and PDE solvers."""

import unittest

from quadrivium import numeric as np

from quadrivium.core.exceptions import DomainError
from quadrivium.ode import *
from quadrivium.pde import *

# y' = -2y + t, y(0) = 1
LINEAR_RHS = lambda t, y: -2 * y + t
LINEAR_SOL = lambda t: 1.25 * np.exp(-2 * t) + t / 2 - 0.25
# stiff: y' = -1000(y - cos t) - sin t, y(0) = 1 -> y = cos t
STIFF_RHS = lambda t, y: -1000.0 * (y - np.cos(t)) - np.sin(t)


class TestExplicitODE(unittest.TestCase):
    def test_convergence_orders(self):
        cases = [(euler, 1), (heun, 2), (midpoint_method, 2), (ralston, 2),
                 (rk3, 3), (rk4, 4), (rk38, 4)]
        for method, order in cases:
            with self.subTest(method=method.__name__):
                e1 = abs(method(LINEAR_RHS, (0, 2), [1.0], 40).y[-1, 0] - LINEAR_SOL(2))
                e2 = abs(method(LINEAR_RHS, (0, 2), [1.0], 80).y[-1, 0] - LINEAR_SOL(2))
                self.assertGreater(e1 / e2, 2**order * 0.6)
                self.assertLess(e1 / e2, 2**order * 1.8)

    def test_adaptive_methods(self):
        for tableau in BUTCHER_TABLEAUX:
            with self.subTest(tableau=tableau):
                r = adaptive_rk(LINEAR_RHS, (0, 2), [1.0], tableau,
                                rtol=1e-10, atol=1e-12)
                self.assertLess(abs(r.y[-1, 0] - LINEAR_SOL(2)), 1e-8)
                self.assertGreater(r.n_accepted, 0)

    def test_harmonic_oscillator_system(self):
        f = lambda t, y: np.array([y[1], -y[0]])
        r = dormand_prince(f, (0, 20), [1.0, 0.0], rtol=1e-12, atol=1e-14)
        self.assertLess(abs(r.y[-1, 0] - np.cos(20)), 1e-9)

    def test_dense_output(self):
        r = dormand_prince(LINEAR_RHS, (0, 2), [1.0], rtol=1e-10, atol=1e-12,
                           dense_output=True)
        q = np.linspace(0, 2, 50)
        self.assertLess(np.max(np.abs(r(q)[:, 0] - LINEAR_SOL(q))), 1e-5)

    def test_generic_tableau_reproduces_rk4(self):
        A = [[], [0.5], [0, 0.5], [0, 0, 1]]
        b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]
        c = [0, 0.5, 0.5, 1]
        rg = rk_general(LINEAR_RHS, (0, 2), [1.0], A, b, c, 50)
        rr = rk4(LINEAR_RHS, (0, 2), [1.0], 50)
        np.testing.assert_allclose(rg.y, rr.y, atol=1e-14)


class TestImplicitODE(unittest.TestCase):
    def test_convergence_orders(self):
        cases = [("backward_euler", lambda n: backward_euler(LINEAR_RHS, (0, 2), [1.0], n), 1),
                 ("trapezoidal", lambda n: trapezoidal(LINEAR_RHS, (0, 2), [1.0], n), 2),
                 ("implicit_midpoint", lambda n: implicit_midpoint(LINEAR_RHS, (0, 2), [1.0], n), 2),
                 ("sdirk", lambda n: sdirk(LINEAR_RHS, (0, 2), [1.0], n), 3),
                 ("esdirk", lambda n: esdirk(LINEAR_RHS, (0, 2), [1.0], n), 3),
                 ("tr_bdf2", lambda n: tr_bdf2(LINEAR_RHS, (0, 2), [1.0], n), 2),
                 ("gauss_irk2", lambda n: gauss_legendre_irk(LINEAR_RHS, (0, 2), [1.0], n, stages=2), 4),
                 ("radau_iia3", lambda n: radau_iia(LINEAR_RHS, (0, 2), [1.0], n, stages=3), 5),
                 ("lobatto_iiic", lambda n: lobatto_iiic(LINEAR_RHS, (0, 2), [1.0], n), 4),
                 ("rosenbrock", lambda n: rosenbrock(LINEAR_RHS, (0, 2), [1.0], n), 2)]
        for name, run, order in cases:
            with self.subTest(method=name):
                e1 = abs(run(10).y[-1, 0] - LINEAR_SOL(2))
                e2 = abs(run(20).y[-1, 0] - LINEAR_SOL(2))
                self.assertLess(e2, e1)
                if e2 > 1e-13:
                    self.assertGreater(e1 / e2, 2**order * 0.5)

    def test_bdf_orders(self):
        # BDF1..BDF6 each attain their formal order; BDF7+ is unstable and
        # correctly refused.
        for order in range(1, 7):
            with self.subTest(order=order):
                e = [abs(bdf(LINEAR_RHS, (0, 2), [1.0], n, order=order).y[-1, 0]
                         - LINEAR_SOL(2)) for n in (80, 160)]
                self.assertGreater(e[0] / e[1], 2**order * 0.7)
        with self.assertRaises(ValueError):
            bdf(LINEAR_RHS, (0, 2), [1.0], 40, order=7)

    def test_stiff_problem(self):
        for name, run in (("backward_euler", lambda: backward_euler(STIFF_RHS, (0, 1), [1.0], 50)),
                          ("trapezoidal", lambda: trapezoidal(STIFF_RHS, (0, 1), [1.0], 50)),
                          ("radau_iia", lambda: radau_iia(STIFF_RHS, (0, 1), [1.0], 50, stages=3)),
                          ("rosenbrock", lambda: rosenbrock(STIFF_RHS, (0, 1), [1.0], 50)),
                          ("bdf2", lambda: bdf(STIFF_RHS, (0, 1), [1.0], 50, order=2)),
                          ("tr_bdf2", lambda: tr_bdf2(STIFF_RHS, (0, 1), [1.0], 50))):
            with self.subTest(method=name):
                self.assertLess(abs(run().y[-1, 0] - np.cos(1)), 1e-2)

    def test_explicit_method_fails_on_stiff_problem(self):
        r = rk4(STIFF_RHS, (0, 1), [1.0], 50)
        self.assertTrue(not np.isfinite(r.y[-1, 0])
                        or abs(r.y[-1, 0] - np.cos(1)) > 1.0)

    def test_l_stability(self):
        decay = lambda t, y: -1e6 * y
        for run in (lambda: backward_euler(decay, (0, 1), [1.0], 10),
                    lambda: radau_iia(decay, (0, 1), [1.0], 10),
                    lambda: tr_bdf2(decay, (0, 1), [1.0], 10)):
            self.assertLess(abs(run().y[-1, 0]), 1e-6)


class TestMultistep(unittest.TestCase):
    def test_adams_bashforth_orders(self):
        for order in range(1, 7):
            e1 = abs(adams_bashforth(LINEAR_RHS, (0, 2), [1.0], 40, order).y[-1, 0]
                     - LINEAR_SOL(2))
            e2 = abs(adams_bashforth(LINEAR_RHS, (0, 2), [1.0], 80, order).y[-1, 0]
                     - LINEAR_SOL(2))
            if e2 > 1e-13:
                self.assertGreater(e1 / e2, 2**order * 0.6)

    def test_adams_moulton_orders(self):
        for order in range(1, 6):
            e1 = abs(adams_moulton(LINEAR_RHS, (0, 2), [1.0], 40, order).y[-1, 0]
                     - LINEAR_SOL(2))
            e2 = abs(adams_moulton(LINEAR_RHS, (0, 2), [1.0], 80, order).y[-1, 0]
                     - LINEAR_SOL(2))
            if e2 > 1e-13:
                self.assertGreater(e1 / e2, 2**order * 0.6)

    def test_predictor_corrector(self):
        r = predictor_corrector(LINEAR_RHS, (0, 2), [1.0], 80, 4)
        self.assertLess(abs(r.y[-1, 0] - LINEAR_SOL(2)), 1e-7)

    def test_variable_step_adams_is_efficient(self):
        r = variable_step_adams(LINEAR_RHS, (0, 2), [1.0], rtol=1e-10, atol=1e-12)
        self.assertLess(abs(r.y[-1, 0] - LINEAR_SOL(2)), 1e-8)
        self.assertLess(r.n_accepted, 500)


class TestSymplectic(unittest.TestCase):
    def setUp(self):
        self.dHdq = lambda q: q
        self.dHdp = lambda p: p
        self.energy = lambda y: 0.5 * (y[0] ** 2 + y[1] ** 2)

    def test_orders(self):
        cases = [(symplectic_euler, 1), (leapfrog, 2), (position_verlet, 2),
                 (ruth3, 3), (yoshida4, 4), (forest_ruth, 4), (pefrl, 4)]
        for method, order in cases:
            with self.subTest(method=method.__name__):
                e = [abs(method(self.dHdq, self.dHdp, (0, 10), [1.0], [0.0], n)
                         .y[-1, 0] - np.cos(10)) for n in (200, 400)]
                self.assertGreater(e[0] / e[1], 2**order * 0.55)

    def test_energy_bounded_over_long_times(self):
        for method in (leapfrog, yoshida4, pefrl):
            r = method(self.dHdq, self.dHdp, (0, 200), [1.0], [0.0], 20000)
            self.assertLess(energy_drift(r, self.energy), 1e-2)

    def test_kepler_conserves_invariants(self):
        force = lambda q: q / np.linalg.norm(q) ** 3
        r = yoshida4(force, lambda p: p, (0, 100), [1.0, 0.0], [0.0, 1.0], 20000)
        E = lambda y: 0.5 * (y[2] ** 2 + y[3] ** 2) - 1.0 / np.hypot(y[0], y[1])
        L = lambda y: y[0] * y[3] - y[1] * y[2]
        self.assertLess(energy_drift(r, E), 1e-6)
        self.assertLess(max(abs(L(y) - L(r.y[0])) for y in r.y), 1e-10)

    def test_velocity_verlet(self):
        r = velocity_verlet(lambda q: -q, (0, 20), [1.0], [0.0], 2000)
        self.assertLess(abs(r.y[-1, 0] - np.cos(20)), 1e-3)


class TestExponentialIntegrators(unittest.TestCase):
    def test_phi_functions(self):
        np.testing.assert_allclose(phi_function(np.zeros((3, 3)), 1), np.eye(3))
        np.testing.assert_allclose(phi_function(np.zeros((3, 3)), 2), np.eye(3) / 2)
        z = 2.5
        self.assertAlmostEqual(phi_function(np.array([[z]]), 1)[0, 0],
                               (np.exp(z) - 1) / z, places=13)

    def test_phi_stable_for_large_negative_argument(self):
        z = -40.0
        self.assertAlmostEqual(phi_function(np.array([[z]]), 1)[0, 0],
                               (np.exp(z) - 1) / z, places=14)

    def test_exact_for_linear_problems(self):
        A = np.array([[-100.0, 1.0], [0.0, -1.0]])
        y0 = np.array([1.0, 1.0])
        from quadrivium.linalg import matrix_exponential
        exact = matrix_exponential(A) @ y0
        for m in (exponential_euler, etd_rk2, etd_rk4):
            r = m(A, lambda t, y: np.zeros(2), (0, 1), y0, 10)
            np.testing.assert_allclose(r.y[-1], exact, atol=1e-10)

    def test_magnus_preserves_norm(self):
        At = lambda t: np.array([[0.0, t], [-t, 0.0]])
        r = magnus_second_order(At, (0, 2), [1.0, 0.0], 200)
        norms = [np.linalg.norm(y) for y in r.y]
        self.assertLess(max(norms) - min(norms), 1e-12)

    def test_krylov_expm(self):
        from quadrivium.linalg import matrix_exponential
        rng = np.random.default_rng(0)
        M = rng.random((40, 40)) - 0.5
        v = rng.random(40)
        np.testing.assert_allclose(krylov_expm_multiply(M, v, 0.1, 30),
                                   matrix_exponential(0.1 * M) @ v, atol=1e-8)


class TestBVP(unittest.TestCase):
    def setUp(self):
        self.exact = lambda t: np.sinh(t) / np.sinh(1)

    def test_linear_bvp_methods(self):
        s = shooting(lambda t, y, yp: y, (0, 1), 0.0, 1.0, n=200)
        self.assertLess(np.max(np.abs(s.y[:, 0] - self.exact(s.t))), 1e-8)
        l = linear_shooting(lambda t: 0.0, lambda t: 1.0, lambda t: 0.0,
                            (0, 1), 0.0, 1.0, 200)
        self.assertLess(np.max(np.abs(l.y[:, 0] - self.exact(l.t))), 1e-9)
        fd = finite_difference_bvp(lambda t: 0.0, lambda t: 1.0, lambda t: 0.0,
                                   (0, 1), 0.0, 1.0, 200)
        self.assertLess(np.max(np.abs(fd.y[:, 0] - self.exact(fd.t))), 1e-5)
        m = multiple_shooting(lambda t, y, yp: y, (0, 1), 0.0, 1.0, 4, 200)
        self.assertLess(np.max(np.abs(m.y[:, 0] - self.exact(m.t))), 1e-6)

    def test_spectral_collocation(self):
        c = collocation_bvp(lambda t, y, yp: y, (0, 1), 0.0, 1.0, 16)
        self.assertLess(np.max(np.abs(c.y[:, 0] - self.exact(c.t))), 1e-10)

    def test_galerkin_second_order(self):
        errs = []
        for n in (25, 50, 100):
            g = galerkin_bvp(lambda t: 0.0, lambda t: 1.0, lambda t: 0.0,
                             (0, 1), 0.0, 1.0, n)
            errs.append(np.max(np.abs(g.y[:, 0] - self.exact(g.t))))
        self.assertGreater(errs[0] / errs[1], 3.5)
        self.assertGreater(errs[1] / errs[2], 3.5)

    def test_nonlinear_bvp(self):
        # y'' = 2 y^3 on [1,2], y(1)=1, y(2)=1/2 -> y = 1/t
        nl = nonlinear_fd_bvp(lambda t, y, yp: 2 * y**3, (1, 2), 1.0, 0.5, 200)
        self.assertLess(np.max(np.abs(nl.y[:, 0] - 1 / nl.t)), 1e-5)
        nc = collocation_bvp(lambda t, y, yp: 2 * y**3, (1, 2), 1.0, 0.5, 20)
        self.assertLess(np.max(np.abs(nc.y[:, 0] - 1 / nc.t)), 1e-10)

    def test_sturm_liouville_eigenvalues(self):
        vals, t, modes = sturm_liouville(lambda t: 1.0, lambda t: 0.0,
                                         lambda t: 1.0, (0, np.pi), 400, 5)
        np.testing.assert_allclose(vals, [1, 4, 9, 16, 25], rtol=1e-3)


class TestParabolicPDE(unittest.TestCase):
    def setUp(self):
        self.u0 = lambda x: np.sin(np.pi * x)
        self.exact = lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
        self.T = 0.1

    def test_schemes_agree_with_analytic_solution(self):
        cases = [("ftcs", lambda: heat_ftcs(self.u0, 1.0, (0, 1), (0, self.T), 40, 4000)),
                 ("btcs", lambda: heat_btcs(self.u0, 1.0, (0, 1), (0, self.T), 40, 400)),
                 ("crank_nicolson", lambda: heat_crank_nicolson(self.u0, 1.0, (0, 1), (0, self.T), 40, 400))]
        for name, run in cases:
            with self.subTest(scheme=name):
                s = run()
                self.assertLess(np.max(np.abs(s.final - self.exact(s.x, self.T))), 1e-3)

    def test_ftcs_stability_guard(self):
        with self.assertRaises(DomainError):
            heat_ftcs(self.u0, 1.0, (0, 1), (0, self.T), 40, 100)
        blown = heat_ftcs(self.u0, 1.0, (0, 1), (0, self.T), 40, 100,
                          check_stability=False)
        self.assertGreater(np.max(np.abs(blown.final)), 1e3)

    def test_crank_nicolson_second_order_in_time(self):
        grid = np.linspace(0, 1, 201)
        e = [np.max(np.abs(heat_crank_nicolson(self.u0, 1.0, (0, 1), (0, self.T),
                                               200, nt).final - self.exact(grid, self.T)))
             for nt in (25, 50)]
        self.assertGreater(e[0] / e[1], 3.0)

    def test_adi_2d(self):
        u0 = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        T = 0.05
        s = heat_2d_adi(u0, 1.0, (0, 1), (0, 1), (0, T), 40, 40, 50)
        X, Y = np.meshgrid(s.x, s.y, indexing="ij")
        exact = np.exp(-2 * np.pi**2 * T) * np.sin(np.pi * X) * np.sin(np.pi * Y)
        self.assertLess(np.max(np.abs(s.final - exact)), 1e-3)

    def test_method_of_lines(self):
        def rhs(t, u, x, dx):
            du = np.zeros_like(u)
            du[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
            return du
        s = method_of_lines(self.u0, rhs, (0, 1), (0, self.T), 40, "radau", n=50)
        self.assertLess(np.max(np.abs(s.final - self.exact(s.x, self.T))), 1e-3)

    def test_upwind_monotone_at_high_peclet(self):
        g = lambda x: 1.0 if x < 0.1 else 0.0
        up = advection_diffusion(g, 1.0, 1e-4, (0, 1), (0, 0.4), 200, 2000,
                                 bc=(1.0, 0.0), upwind=True).final
        ce = advection_diffusion(g, 1.0, 1e-4, (0, 1), (0, 0.4), 200, 2000,
                                 bc=(1.0, 0.0), upwind=False).final
        self.assertGreaterEqual(up.min(), -1e-9)
        self.assertLessEqual(up.max(), 1 + 1e-9)
        self.assertTrue(ce.max() > 1.01 or ce.min() < -0.01)


class TestHyperbolicPDE(unittest.TestCase):
    def test_wave_equation(self):
        c, T = 1.0, 0.5
        s = wave_explicit(lambda x: np.sin(np.pi * x), lambda x: 0.0, c,
                          (0, 1), (0, T), 200, 400)
        exact = np.sin(np.pi * s.x) * np.cos(np.pi * c * T)
        self.assertLess(np.max(np.abs(s.final - exact)), 1e-4)

    def test_cfl_guard(self):
        with self.assertRaises(DomainError):
            wave_explicit(lambda x: np.sin(np.pi * x), lambda x: 0.0, 1.0,
                          (0, 1), (0, 0.5), 400, 100)

    def test_scheme_accuracy_ordering(self):
        u0 = lambda x: np.sin(2 * np.pi * x)
        errs = {}
        for name, run in (("upwind", advection_upwind),
                          ("lax_friedrichs", lax_friedrichs),
                          ("lax_wendroff", lax_wendroff),
                          ("maccormack", maccormack)):
            s = run(u0, 1.0, (0, 1), (0, 1.0), 200, 400)
            errs[name] = np.sqrt(np.mean((s.final - u0(s.x)) ** 2))
        self.assertLess(errs["lax_wendroff"], errs["upwind"])
        self.assertLess(errs["maccormack"], errs["lax_friedrichs"])

    def test_godunov_theorem_and_tvd_limiters(self):
        sq = lambda x: 1.0 if 0.3 <= x <= 0.6 else 0.0
        lw = lax_wendroff(sq, 1.0, (0, 1), (0, 1.0), 200, 400).final
        self.assertLess(lw.min(), -1e-3)          # second order + linear oscillates
        for lim in ("minmod", "superbee", "van_leer", "van_albada", "mc", "koren",
                    "ospre"):
            with self.subTest(limiter=lim):
                s = tvd_scheme(sq, 1.0, (0, 1), (0, 1.0), 200, 400, lim)
                self.assertGreaterEqual(s.final.min(), -1e-3)
                self.assertLessEqual(s.final.max(), 1 + 1e-3)

    def test_burgers_shock_speed(self):
        u0 = lambda x: 1.0 if x < 0.5 else 0.0     # Rankine-Hugoniot speed 0.5
        for run in (godunov_burgers, lax_friedrichs_burgers):
            s = run(u0, (0, 2), (0, 0.8), 400, 800)
            pos = s.x[np.argmin(np.diff(s.final))]
            self.assertLess(abs(pos - 0.9), 0.02)

    def test_burgers_conserves_mass(self):
        s = godunov_burgers(lambda x: 0.5 + 0.5 * np.sin(2 * np.pi * x),
                            (0, 1), (0, 0.5), 400, 1000)
        self.assertAlmostEqual(float(s.u[0].mean()), float(s.final.mean()), places=12)


class TestEllipticPDE(unittest.TestCase):
    def setUp(self):
        self.u = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        self.f = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def test_stencil_orders(self):
        e5, e9 = [], []
        for n in (8, 16, 32):
            g = np.linspace(0, 1, n + 1)
            X, Y = np.meshgrid(g, g, indexing="ij")
            e5.append(np.max(np.abs(poisson_2d_direct(self.f, (0, 1), (0, 1), n, n,
                                                      0.0, 5).u - self.u(X, Y))))
            e9.append(np.max(np.abs(poisson_2d_direct(self.f, (0, 1), (0, 1), n, n,
                                                      0.0, 9).u - self.u(X, Y))))
        self.assertGreater(e5[0] / e5[1], 3.5)      # second order
        self.assertGreater(e9[0] / e9[1], 12.0)     # fourth order
        self.assertLess(e9[-1], e5[-1] / 100)

    def test_harmonic_function_exact(self):
        h = lambda x, y: x * x - y * y
        for stencil in (5, 9):
            s = poisson_2d_direct(lambda x, y: 0.0, (0, 1), (0, 1), 20, 20,
                                  bc=h, stencil=stencil)
            X, Y = np.meshgrid(s.x, s.y, indexing="ij")
            self.assertLess(np.max(np.abs(s.u - h(X, Y))), 1e-12)

    def test_iterative_solvers(self):
        n = 24
        g = np.linspace(0, 1, n + 1)
        X, Y = np.meshgrid(g, g, indexing="ij")
        for method in ("jacobi", "gauss_seidel", "sor", "cg"):
            with self.subTest(method=method):
                s = poisson_2d_iterative(self.f, (0, 1), (0, 1), n, n, 0.0,
                                         method=method, tol=1e-10, max_iter=30000)
                self.assertLess(np.max(np.abs(s.u - self.u(X, Y))), 5e-3)

    def test_sor_faster_than_gauss_seidel(self):
        n = 24
        it_sor = poisson_2d_iterative(self.f, (0, 1), (0, 1), n, n, 0.0,
                                      method="sor", tol=1e-10).iterations
        it_gs = poisson_2d_iterative(self.f, (0, 1), (0, 1), n, n, 0.0,
                                     method="gauss_seidel", tol=1e-10).iterations
        self.assertLess(it_sor, it_gs / 3)

    def test_fft_poisson_matches_direct(self):
        n = 32
        g = np.linspace(0, 1, n + 1)
        X, Y = np.meshgrid(g, g, indexing="ij")
        sf = poisson_fft(self.f, (0, 1), (0, 1), n, n)
        sd = poisson_2d_direct(self.f, (0, 1), (0, 1), n, n, 0.0)
        np.testing.assert_allclose(sf.u, sd.u, atol=1e-8)

    def test_poisson_1d(self):
        s = poisson_1d(lambda x: -np.pi**2 * np.sin(np.pi * x), (0, 1), (0, 0), 200)
        self.assertLess(np.max(np.abs(s.u - np.sin(np.pi * s.x))), 1e-4)


class TestMultigrid(unittest.TestCase):
    def setUp(self):
        self.u = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        self.f = lambda x, y: -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def test_grid_transfer_operators(self):
        np.testing.assert_allclose(prolong(np.ones((9, 9))), 1.0)
        rc = restrict(np.ones((17, 17)))
        np.testing.assert_allclose(rc[1:-1, 1:-1], 1.0)

    def test_mesh_independent_convergence(self):
        counts = []
        for n in (32, 64, 128):
            s = multigrid_solve(self.f, (0, 1), (0, 1), n, tol=1e-10, max_cycles=50)
            self.assertTrue(s.converged)
            X, Y = np.meshgrid(s.x, s.y, indexing="ij")
            self.assertLess(np.max(np.abs(s.u - self.u(X, Y))), 5e-3)
            counts.append(s.iterations)
        self.assertLessEqual(max(counts) - min(counts), 2)

    def test_convergence_factor(self):
        s = multigrid_solve(self.f, (0, 1), (0, 1), 64, tol=1e-12, max_cycles=20)
        rates = [s.residuals[i + 1] / s.residuals[i] for i in range(4)]
        self.assertLess(np.mean(rates), 0.3)

    def test_w_cycle(self):
        s = multigrid_solve(self.f, (0, 1), (0, 1), 64, tol=1e-10, cycle="w")
        self.assertTrue(s.converged)

    def test_full_multigrid_single_pass(self):
        h = 1.0 / 64
        g = np.linspace(0, 1, 65)
        X, Y = np.meshgrid(g, g, indexing="ij")
        u = full_multigrid(self.f(X, Y), h, cycles=1)
        self.assertLess(np.max(np.abs(u - self.u(X, Y))), 1e-3)

    def test_requires_power_of_two(self):
        with self.assertRaises(ValueError):
            multigrid_solve(self.f, (0, 1), (0, 1), 30)


class TestFEMandFVM(unittest.TestCase):
    def test_p1_nodally_exact_in_1d(self):
        # -u'' = 2, u(0)=u(1)=0 -> u = x(1-x), reproduced exactly at the nodes
        s = fem_1d_linear(lambda x: 2.0, (0, 1), (0.0, 0.0), 10)
        self.assertLess(np.max(np.abs(s.u - s.x * (1 - s.x))), 1e-14)

    def test_p2_reproduces_cubic(self):
        s = fem_1d_quadratic(lambda x: 6 * x, (0, 1), (0.0, 0.0), 5)
        self.assertLess(np.max(np.abs(s.u - (s.x - s.x**3))), 1e-13)

    def test_fem_1d_convergence(self):
        f = lambda x: np.pi**2 * np.sin(np.pi * x)
        exact = lambda x: np.sin(np.pi * x)
        errs = [np.max(np.abs(fem_1d_linear(f, (0, 1), (0.0, 0.0), n).u
                              - exact(np.linspace(0, 1, n + 1)))) for n in (20, 40, 80)]
        self.assertGreater(errs[0] / errs[1], 8)
        self.assertGreater(errs[1] / errs[2], 8)

    def test_mass_and_stiffness_properties(self):
        K, M = fem_1d_mass_stiffness(np.linspace(0, 1, 11))
        np.testing.assert_allclose(K @ np.ones(11), 0, atol=1e-12)
        self.assertAlmostEqual(float(M.sum()), 1.0, places=12)

    def test_fem_2d_convergence(self):
        f2 = lambda x, y: 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        u2 = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
        errs = []
        for n in (8, 16, 32):
            s = fem_2d_triangular(f2, n=n)
            P = np.column_stack([s.grids[0], s.grids[1]])
            errs.append(np.max(np.abs(s.u - u2(P[:, 0], P[:, 1]))))
        self.assertGreater(errs[0] / errs[1], 3.5)
        self.assertGreater(errs[1] / errs[2], 3.5)

    def test_fem_2d_harmonic_exact(self):
        h = lambda x, y: x * x - y * y
        s = fem_2d_triangular(lambda x, y: 0.0, n=8, bc=h)
        P = np.column_stack([s.grids[0], s.grids[1]])
        self.assertLess(np.max(np.abs(s.u - h(P[:, 0], P[:, 1]))), 1e-12)

    def test_fvm_conservation_and_monotonicity(self):
        sq = lambda x: 1.0 if 0.3 <= x <= 0.6 else 0.0
        flux = lambda u: 1.0 * u
        speed = lambda u: np.ones_like(u)
        for nf in ("rusanov", "hll", "lax_friedrichs"):
            with self.subTest(flux=nf):
                s = fvm_1d_conservation(sq, flux, speed, (0, 1), (0, 1.0), 200, 400,
                                        numerical_flux=nf)
                self.assertAlmostEqual(float(s.u[0].mean()), float(s.final.mean()),
                                       places=12)
                self.assertGreaterEqual(s.final.min(), -1e-9)

    def test_muscl_sharper_than_first_order(self):
        sq = lambda x: 1.0 if 0.3 <= x <= 0.6 else 0.0
        flux = lambda u: 1.0 * u
        speed = lambda u: np.ones_like(u)
        s1 = fvm_1d_conservation(sq, flux, speed, (0, 1), (0, 1.0), 200, 800)
        s2 = fvm_muscl(sq, flux, speed, (0, 1), (0, 1.0), 200, 800, "mc")
        ref = np.array([sq(xi) for xi in s1.x])
        self.assertLess(np.mean(np.abs(s2.final - ref)),
                        np.mean(np.abs(s1.final - ref)) / 2)


class TestSpectralPDE(unittest.TestCase):
    def test_fourier_heat_exact(self):
        u0 = lambda x: np.sin(x) + 0.5 * np.sin(3 * x)
        s = fourier_heat(u0, 1.0, 2 * np.pi, 64, (0, 0.5), 50)
        exact = np.exp(-0.5) * np.sin(s.x) + 0.5 * np.exp(-4.5) * np.sin(3 * s.x)
        self.assertLess(np.max(np.abs(s.final - exact)), 1e-13)

    def test_fourier_advection_no_dispersion(self):
        u0 = lambda x: np.sin(x) + 0.5 * np.sin(3 * x)
        s = fourier_advection(u0, 1.0, 2 * np.pi, 64, (0, 10.0), 50)
        exact = np.sin(s.x - 10.0) + 0.5 * np.sin(3 * (s.x - 10.0))
        self.assertLess(np.max(np.abs(s.final - exact)), 1e-12)

    def test_chebyshev_spectral_convergence(self):
        f = lambda x: -np.pi**2 * np.sin(np.pi * x)
        errs = []
        for n in (8, 16, 32):
            s = chebyshev_poisson_1d(f, (-1, 1), (0.0, 0.0), n)
            errs.append(np.max(np.abs(s.u - np.sin(np.pi * s.x))))
        self.assertLess(errs[1], errs[0] / 100)     # exponential convergence
        self.assertLess(errs[2], 1e-12)

    def test_spectral_burgers_conserves_mass(self):
        s = spectral_burgers(np.sin, 0.05, 2 * np.pi, 128, (0, 1.0), 2000)
        self.assertLess(abs(float(s.final.mean())), 1e-10)

    def test_kuramoto_sivashinsky_bounded(self):
        s = kuramoto_sivashinsky(lambda x: np.cos(x / 16) * (1 + np.sin(x / 16)),
                                 32 * np.pi, 128, (0, 30), 3000)
        self.assertTrue(np.all(np.isfinite(s.final)))
        self.assertLess(np.max(np.abs(s.final)), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
