"""Tests for the extended functionality: SDEs, wavelets, matrix functions,
advanced ODE solvers, high-resolution PDE schemes, and the new special
functions, quadrature rules and optimizers.

The emphasis throughout is on *properties* that pin the answer down
independently of any table of constants: exact identities, convergence orders,
conservation laws, and published benchmarks.
"""

import math
import unittest

from quadrivium import numeric as np

from quadrivium.approx import aaa, chebyshev_economization
from quadrivium.integrate import (cauchy_principal_value, filon,
                                 hadamard_finite_part, monte_carlo,
                                 sparse_grid_quadrature)
from quadrivium.interpolate import (bezier, bezier_derivative, bezier_subdivide,
                                   de_casteljau, nurbs, nurbs_circle,
                                   open_uniform_knots, trilinear)
from quadrivium.linalg import (condition_estimate, cur_decomposition,
                              discrete_lyapunov, generalized_eigh,
                              interpolative_decomposition, lobpcg, logm,
                              lyapunov, nystrom_approximation, polar_decomposition,
                              qz_decomposition, qz_eigenvalues,
                              randomized_range_finder, randomized_svd,
                              schur, schur_eigenvalues, signm, sqrtm,
                              subspace_iteration, sylvester)
from quadrivium.ode import (dae_index1_bdf, dde_method_of_steps, detect_stiffness,
                           dormand_prince, find_events, gragg_bulirsch_stoer,
                           mass_matrix_ode, modified_midpoint, rk4, rk_nystrom,
                           solve_ivp, solve_ivp_events, stormer_cowell)
from quadrivium.optimize import lbfgsb, newton_cg
from quadrivium.pde import (lid_driven_cavity, navier_stokes_2d,
                           poisson_neumann, poisson_periodic_fft, ssp_rk3,
                           vorticity_streamfunction, weno5_reconstruct,
                           weno_burgers, weno_conservation_law)
from quadrivium.rootfind import anderson_acceleration, newton_krylov
from quadrivium.special import (associated_legendre, bessel_i0, bessel_i1,
                               bessel_in, bessel_k0, bessel_k1, bessel_kn,
                               dawson, erfcx, expint_n, fresnel_c, fresnel_s,
                               hyp1f1, hyp2f1, lambert_w, logistic, logit,
                               polygamma, spherical_bessel_j, spherical_bessel_y,
                               spherical_harmonic, trigamma, zeta)
from quadrivium.stochastic import (brownian_bridge, brownian_path,
                                  cox_ingersoll_ross, euler_maruyama,
                                  geometric_brownian_motion, gillespie_ssa,
                                  milstein, ornstein_uhlenbeck, stochastic_heun,
                                  stochastic_rk, tau_leaping)
from quadrivium.transforms import (cwt, dwt, dwt2, goertzel, idwt, idwt2, iswt,
                                  scale_to_frequency, swt, wavedec,
                                  wavelet_denoise, wavelet_energy, waverec,
                                  window)


class TestStochasticDifferentialEquations(unittest.TestCase):
    def test_strong_order_of_the_schemes(self):
        """Milstein is strong order 1; Euler-Maruyama only 1/2.

        The gap is the whole reason Milstein exists: it keeps the Ito term
        b b' (dW^2 - dt)/2 that Euler-Maruyama discards.
        """
        mu, sig, x0, T = 1.5, 0.6, 1.0, 1.0
        a = lambda x, t: mu * x
        b = lambda x, t: sig * x
        db = lambda x, t: sig * np.ones_like(x)
        exact = lambda wt: x0 * np.exp((mu - 0.5 * sig**2) * T + sig * wt)

        def strong(solver, n, paths=400, **kw):
            rng = np.random.default_rng(0)
            dt = T / n
            total = 0.0
            for _ in range(paths):
                dW = np.sqrt(dt) * rng.standard_normal((n, 1))
                y = solver(a, b, (0, T), [x0], n=n, dW=dW, **kw).y[-1][0]
                total += abs(y - exact(dW.sum()))
            return total / paths

        e_m = [strong(milstein, n, db=db) for n in (32, 128)]
        order_m = math.log2(e_m[0] / e_m[1]) / 2.0
        self.assertGreater(order_m, 0.85)
        e_rk = [strong(stochastic_rk, n) for n in (32, 128)]
        self.assertGreater(math.log2(e_rk[0] / e_rk[1]) / 2.0, 0.85)

    def test_heun_converges_to_the_stratonovich_solution(self):
        """Not a defect: stochastic Heun solves a different equation.

        For multiplicative noise the Ito and Stratonovich solutions differ by
        the drift correction b b'/2, so Heun does not converge to the Ito
        answer at all -- it converges to the Stratonovich one.
        """
        mu, sig, x0, T = 1.5, 0.6, 1.0, 1.0
        a = lambda x, t: mu * x
        b = lambda x, t: sig * x
        strat = lambda wt: x0 * np.exp(mu * T + sig * wt)
        rng = np.random.default_rng(1)
        errs = []
        for n in (32, 128):
            dt = T / n
            total = 0.0
            for _ in range(300):
                dW = np.sqrt(dt) * rng.standard_normal((n, 1))
                y = stochastic_heun(a, b, (0, T), [x0], n=n, dW=dW).y[-1][0]
                total += abs(y - strat(dW.sum()))
            errs.append(total / 300)
        self.assertGreater(math.log2(errs[0] / errs[1]) / 2.0, 0.85)

    def test_exact_samplers_have_no_discretization_error(self):
        rng = np.random.default_rng(7)
        sol = ornstein_uhlenbeck([0.0], theta=2.0, mu=1.0, sigma=0.5,
                                 t_span=(0, 40), n=80000, rng=rng)
        tail = sol.y[800:, 0]
        self.assertAlmostEqual(float(tail.mean()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(tail.var()), 0.25 / 4.0, delta=0.01)
        for seed in range(20):        # GBM stays positive by construction
            sol = geometric_brownian_motion([1.0], 0.5, 0.3, (0, 1), n=50,
                                            rng=np.random.default_rng(seed))
            self.assertTrue(np.all(np.asarray(sol.y) > 0))

    def test_cir_stays_non_negative(self):
        sol = cox_ingersoll_ross([0.05], theta=2.0, mu=0.04, sigma=0.5,
                                 t_span=(0, 5), n=2000,
                                 rng=np.random.default_rng(4))
        self.assertTrue(np.all(np.asarray(sol.y) >= 0.0))

    def test_gillespie_reproduces_the_exact_master_equation(self):
        """A pure death process has X(t) ~ Binomial(X0, e^{-kt}) exactly."""
        k, x0, T = 0.8, 40, 1.5
        prop = lambda x, t: np.array([k * x[0]])
        S = np.array([[-1.0]])
        finals = np.array([gillespie_ssa(prop, S, [x0], (0, T),
                                         rng=np.random.default_rng(s))[1][-1, 0]
                           for s in range(1500)])
        p = math.exp(-k * T)
        self.assertAlmostEqual(float(finals.mean()), x0 * p, delta=0.35)
        self.assertAlmostEqual(float(finals.var()), x0 * p * (1 - p), delta=1.2)

    def test_tau_leaping_matches_the_exact_algorithm(self):
        k, x0, T = 0.8, 40, 1.5
        prop = lambda x, t: np.array([k * x[0]])
        S = np.array([[-1.0]])
        finals = [tau_leaping(prop, S, [x0], (0, T), tau=0.01,
                              rng=np.random.default_rng(s))[1][-1, 0]
                  for s in range(1500)]
        self.assertAlmostEqual(float(np.mean(finals)), x0 * math.exp(-k * T),
                               delta=0.4)

    def test_brownian_path_and_bridge(self):
        rng = np.random.default_rng(3)
        t, W = brownian_path((0, 1), n=4000, dim=2, rng=rng)
        self.assertTrue(np.allclose(W[0], 0.0))
        self.assertAlmostEqual(float(np.var(W[-1])), 1.0, delta=0.9)
        t, B = brownian_bridge((0, 2), 1.0, -3.0, n=500, dim=1, rng=rng)
        self.assertAlmostEqual(float(B[0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(B[-1, 0]), -3.0, places=12)


class TestWavelets(unittest.TestCase):
    WAVELETS = ["haar", "db2", "db3", "db4", "db6", "db8", "sym4", "coif1", "coif2"]

    def test_perfect_reconstruction(self):
        """The defining property: analysis followed by synthesis is the identity."""
        rng = np.random.default_rng(0)
        for wav in self.WAVELETS:
            for n in (64, 128, 300):
                x = rng.standard_normal(n)
                self.assertLess(np.abs(idwt(*dwt(x, wav), wav, n) - x).max(), 1e-10)
                for level in (1, 3, 5):
                    rec = waverec(wavedec(x, wav, level), wav, n)
                    self.assertLess(np.abs(rec - x).max(), 1e-9, msg=f"{wav} L{level}")

    def test_vanishing_moments(self):
        """dbN annihilates polynomials of degree below N -- this is what fixes
        the filter coefficients, so it is the sharpest check on them."""
        t = np.linspace(0, 1, 256)
        for wav, n_vanishing in [("db2", 2), ("db3", 3), ("db4", 4)]:
            for deg in range(n_vanishing):
                detail = dwt(t**deg, wav)[1]
                self.assertLess(np.abs(detail[4:-4]).max(), 1e-9,
                                msg=f"{wav} should annihilate t^{deg}")

    def test_parseval(self):
        rng = np.random.default_rng(1)
        for wav in ("haar", "db4", "coif2"):
            x = rng.standard_normal(512)
            bands = wavelet_energy(wavedec(x, wav, 4))
            self.assertAlmostEqual(float(bands.sum()) / float(np.sum(x**2)), 1.0,
                                   places=9)

    def test_stationary_transform_is_shift_invariant(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(256)
        a = swt(x, "db2", 3)
        b = swt(np.roll(x, 9), "db2", 3)
        for ca, cb in zip(a, b):
            self.assertLess(np.abs(np.roll(np.asarray(ca), 9) - np.asarray(cb)).max(),
                            1e-12)
        self.assertLess(np.abs(iswt(a, "db2") - x).max(), 1e-10)

    def test_two_dimensional_transform(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((32, 48))
        cA, det = dwt2(X, "db4")
        self.assertLess(np.abs(idwt2(cA, det, "db4", X.shape) - X).max(), 1e-9)

    def test_denoising_reduces_error(self):
        t = np.linspace(0, 1, 1024)
        clean = np.sin(4 * np.pi * t) + np.where(t > 0.5, 1.0, 0.0)
        noisy = clean + 0.3 * np.random.default_rng(5).standard_normal(1024)
        before = np.sqrt(np.mean((noisy - clean) ** 2))
        after = np.sqrt(np.mean((wavelet_denoise(noisy, "db4", level=5) - clean) ** 2))
        self.assertLess(after, before / 2.0)

    def test_cwt_recovers_frequency(self):
        fs = 256.0
        t = np.arange(1024) / fs
        scales = np.arange(2, 80.0)
        for f0 in (10.0, 20.0, 40.0):
            W = cwt(np.sin(2 * np.pi * f0 * t), scales, dt=1 / fs)
            peak = scales[int(np.argmax(np.abs(W).mean(axis=1)))]
            self.assertAlmostEqual(float(scale_to_frequency(peak, 1 / fs)[0]), f0,
                                   delta=0.06 * f0)

    def test_goertzel_matches_the_dft_bin(self):
        x = np.random.default_rng(6).standard_normal(200)
        for k in (0, 1, 7, 33, 99):
            self.assertLess(abs(goertzel(x, k) - np.fft.fft(x)[k]), 1e-9)

    def test_new_windows(self):
        for kind in ("kaiser", "flattop", "blackman_harris", "nuttall",
                     "gaussian", "welch", "cosine", "lanczos", "bohman", "parzen"):
            w = window(64, kind)
            self.assertEqual(w.size, 64)
            self.assertTrue(np.allclose(w, w[::-1]), msg=f"{kind} not symmetric")
        for beta in (5.0, 8.6, 14.0):
            self.assertLess(np.abs(window(128, "kaiser", beta=beta)
                                   - np.kaiser(128, beta)).max(), 1e-12)
        # A periodic window is the symmetric one of length n+1, minus its last point.
        self.assertTrue(np.allclose(window(8, "hann", sym=False), np.hanning(9)[:8]))


class TestMatrixFunctions(unittest.TestCase):
    def test_sqrtm_and_logm(self):
        from quadrivium.linalg import matrix_exponential

        rng = np.random.default_rng(0)
        for n in (2, 5, 12):
            A = rng.standard_normal((n, n))
            A = A @ A.T + n * np.eye(n)
            X = sqrtm(A)
            self.assertLess(np.abs(X @ X - A).max() / np.abs(A).max(), 1e-12)
            B = 0.3 * rng.standard_normal((n, n))
            self.assertLess(np.abs(logm(matrix_exponential(B)) - B).max(), 1e-11)
        rot = np.array([[0.0, 1.0], [-1.0, 0.0]])   # needs a complex root
        self.assertLess(np.abs(sqrtm(rot) @ sqrtm(rot) - rot).max(), 1e-12)

    def test_signm_squares_to_the_identity(self):
        rng = np.random.default_rng(1)
        Q = np.linalg.qr(rng.standard_normal((4, 4)))[0]
        A = Q @ np.diag([2.0, -3.0, 5.0, -1.0]) @ Q.T
        S = signm(A)
        self.assertLess(np.abs(S @ S - np.eye(4)).max(), 1e-10)
        self.assertEqual(int(np.sum(np.linalg.eigvals(S).real > 0)), 2)

    def test_sylvester_and_lyapunov(self):
        rng = np.random.default_rng(2)
        for n, m in [(4, 3), (6, 6), (8, 5)]:
            A = rng.standard_normal((n, n))
            B = rng.standard_normal((m, m))
            C = rng.standard_normal((n, m))
            X = sylvester(A, B, C)
            self.assertLess(np.abs(A @ X + X @ B - C).max(), 1e-9)
        A = rng.standard_normal((6, 6)) - 3.5 * np.eye(6)      # stable
        Q = rng.standard_normal((6, 6))
        Q = Q @ Q.T
        X = lyapunov(A, Q)
        self.assertLess(np.abs(A @ X + X @ A.T + Q).max(), 1e-10)
        self.assertGreater(float(np.linalg.eigvalsh(X).min()), -1e-10)
        Ad = 0.6 * np.linalg.qr(rng.standard_normal((5, 5)))[0]
        Qd = rng.standard_normal((5, 5))
        Qd = Qd @ Qd.T
        Xd = discrete_lyapunov(Ad, Qd)
        self.assertLess(np.abs(Ad @ Xd @ Ad.T - Xd + Qd).max(), 1e-11)

    def test_condition_estimate(self):
        rng = np.random.default_rng(3)
        for n in (5, 20, 60):
            A = rng.standard_normal((n, n))
            est = condition_estimate(A)
            true = np.linalg.cond(A, 1)
            self.assertGreater(est / true, 0.3)     # Hager's bound is one-sided
            self.assertLess(est / true, 1.0 + 1e-9)

    def test_randomized_svd_matches_the_optimal_truncation(self):
        rng = np.random.default_rng(4)
        U, _, V = np.linalg.svd(rng.standard_normal((300, 200)), full_matrices=False)
        s = np.exp(-np.arange(200) / 12.0)
        A = U @ np.diag(s) @ V
        for k in (5, 15, 30):
            Ur, sr, Vr = randomized_svd(A, k, rng=rng)
            err = np.linalg.norm(A - Ur @ np.diag(sr) @ Vr, 2)
            self.assertLess(err / s[k], 1.05)       # Eckart-Young is the floor

    def test_low_rank_factorizations(self):
        rng = np.random.default_rng(5)
        A = rng.standard_normal((100, 60))
        cols, Z = interpolative_decomposition(A, 30)
        self.assertEqual(len(set(cols.tolist())), 30)
        self.assertLess(np.linalg.norm(A[:, cols] @ Z - A) / np.linalg.norm(A), 0.8)
        c, r, U = cur_decomposition(A, 40)
        self.assertLess(np.linalg.norm(A[:, c] @ U @ A[r, :] - A) / np.linalg.norm(A),
                        0.8)
        S = rng.standard_normal((60, 60))
        S = S @ S.T
        N = nystrom_approximation(S, 40, rng=rng)
        self.assertGreater(float(np.linalg.eigvalsh(N).min()), -1e-8)

    def test_subspace_solvers(self):
        rng = np.random.default_rng(6)
        n = 50
        B = rng.standard_normal((n, n))
        S = B @ B.T + n * np.eye(n)
        ev = np.sort(np.linalg.eigvalsh(S))
        for k in (1, 3):
            res = subspace_iteration(S, k, rng=rng, max_iter=4000)
            got = np.sort(np.asarray(res.eigenvalues))[::-1][:k]
            self.assertLess(np.abs(got - ev[::-1][:k]).max(), 1e-8)
        m = 60
        L = (np.diag(2.0 * np.ones(m)) + np.diag(-np.ones(m - 1), 1)
             + np.diag(-np.ones(m - 1), -1))
        res = lobpcg(L, 4, rng=rng, max_iter=600)
        exact = np.sort(np.linalg.eigvalsh(L))[:4]
        self.assertLess(np.abs(np.sort(np.asarray(res.eigenvalues))[:4] - exact).max(),
                        1e-9)

    def test_generalized_eigenproblem_and_qz(self):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((8, 8))
        A = A + A.T
        B = rng.standard_normal((8, 8))
        B = B @ B.T + 8 * np.eye(8)
        res = generalized_eigh(A, B)
        w = np.asarray(res.eigenvalues)
        X = np.asarray(res.eigenvectors)
        self.assertLess(np.abs(A @ X - B @ X * w).max(), 1e-10)
        self.assertLess(np.abs(X.T @ B @ X - np.eye(8)).max(), 1e-10)
        for n in (2, 3, 6, 12):
            A = rng.standard_normal((n, n))
            B = rng.standard_normal((n, n))
            Q, Z, S, T = qz_decomposition(A, B)
            self.assertLess(np.abs(Q.T @ A @ Z - S).max(), 1e-9)
            self.assertLess(np.abs(Q.T @ B @ Z - T).max(), 1e-10)
            got = qz_eigenvalues(S, T)
            exp = np.linalg.eigvals(np.linalg.solve(B, A))
            d = np.abs(got[:, None] - exp[None, :])
            self.assertLess(max(d.min(axis=1).max(), d.min(axis=0).max()), 1e-8)

    def test_schur_handles_complex_pairs_and_polar_is_rectangular(self):
        rng = np.random.default_rng(8)
        for n in (2, 3, 6, 15):
            A = rng.standard_normal((n, n))
            Q, T = schur(A)
            self.assertLess(np.abs(Q @ T @ Q.T - A).max(), 1e-10)
            self.assertLess(np.abs(Q.T @ Q - np.eye(n)).max(), 1e-12)
            got = np.sort_complex(schur_eigenvalues(T))
            self.assertLess(np.abs(got - np.sort_complex(np.linalg.eigvals(A))).max(),
                            1e-8)
        S = rng.standard_normal((10, 10))
        S = S + S.T
        self.assertLess(np.abs(np.diag(schur(S)[1], -1)).max(), 1e-12)
        for shape in [(5, 3), (4, 4), (3, 5)]:   # rectangular polar decomposition
            A = rng.standard_normal(shape)
            R, P = polar_decomposition(A)
            self.assertLess(np.abs(R @ P - A).max(), 1e-12)
            self.assertLess(np.abs(P - P.T).max(), 1e-12)
            Pl, Rl = polar_decomposition(A, side="left")
            self.assertLess(np.abs(Pl @ Rl - A).max(), 1e-12)


class TestAdvancedODE(unittest.TestCase):
    F = staticmethod(lambda t, y: -y + np.sin(t))
    EXACT = staticmethod(lambda t: (np.sin(t) - np.cos(t) + np.exp(-t)) / 2)

    def test_extrapolation_beats_a_fixed_order_method(self):
        """Bulirsch-Stoer reaches tight tolerances far more cheaply, because
        each extrapolation column gains two orders."""
        gbs = gragg_bulirsch_stoer(self.F, (0, 10), [0.0], rtol=1e-12, atol=1e-14)
        dp = dormand_prince(self.F, (0, 10), [0.0], rtol=1e-12, atol=1e-14)
        self.assertLess(abs(gbs.y[-1][0] - self.EXACT(10.0)), 1e-11)
        self.assertLess(gbs.n_rhs_evals, dp.n_rhs_evals / 3)

    def test_modified_midpoint_has_an_even_error_expansion(self):
        errs = [abs(modified_midpoint(self.F, 0.0, np.array([0.0]), 1.0, n)[0]
                    - self.EXACT(1.0)) for n in (8, 16, 32)]
        self.assertGreater(errs[1] / errs[2], 3.2)     # -> 4 for an h^2 expansion

    def test_event_location_is_as_accurate_as_the_solver(self):
        g = 9.81
        rhs = lambda t, y: np.array([y[1], -g])
        sol, te, ye = solve_ivp_events(rhs, (0, 3), [10.0, 0.0],
                                       events=lambda t, y: y[0], terminal=True,
                                       rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(float(te[0][0]), math.sqrt(20.0 / g), places=11)
        self.assertAlmostEqual(float(ye[0][0][1]), -g * math.sqrt(20.0 / g), places=9)
        self.assertLessEqual(float(sol.t[-1]), float(te[0][0]) + 1e-12)

    def test_directional_events(self):
        osc = lambda t, y: np.array([y[1], -y[0]])
        sol = dormand_prince(osc, (0, 20), [1.0, 0.0], rtol=1e-12, atol=1e-14)
        allt, _ = find_events(sol, lambda t, y: y[0])
        up, _ = find_events(sol, lambda t, y: y[0], direction=1)
        down, _ = find_events(sol, lambda t, y: y[0], direction=-1)
        self.assertEqual(len(allt), len(up) + len(down))
        expected = np.array([(k + 0.5) * np.pi for k in range(len(allt))])
        self.assertLess(np.abs(allt - expected).max(), 1e-10)

    def test_nystrom_is_fourth_order(self):
        errs = [abs(rk_nystrom(lambda t, y, dy: -y, (0, 10), [1.0], [0.0],
                               n=n).y[-1][0] - math.cos(10.0))
                for n in (40, 80, 160)]
        self.assertGreater(errs[0] / errs[1], 10.0)
        self.assertGreater(errs[1] / errs[2], 12.0)

    def test_stormer_cowell_is_fourth_order(self):
        errs = [abs(stormer_cowell(lambda t, y: -y, (0, 10), [1.0], [0.0],
                                   n=n).y[-1][0] - math.cos(10.0))
                for n in (160, 320, 640)]
        self.assertGreater(errs[0] / errs[1], 12.0)
        self.assertGreater(errs[1] / errs[2], 13.0)

    def test_dae_enforces_the_constraint_exactly(self):
        """The point of a DAE solver: g = 0 holds at every step, not just to
        the accuracy of an ODE approximation."""
        sol = dae_index1_bdf(lambda t, y, z: np.array([-y[0] + z[0]]),
                             lambda t, y, z: np.array([y[0]**2 + z[0]**2 - 1.0]),
                             (0, 2), [0.6], [0.8], n=800, order=2)
        g = sol.y[:, 0] ** 2 + sol.z[:, 0] ** 2 - 1.0
        self.assertLess(np.abs(g).max(), 1e-10)

    def test_singular_mass_matrix(self):
        M = np.array([[1.0, 0.0], [0.0, 0.0]])      # the second row is algebraic
        sol = mass_matrix_ode(M, lambda t, y: np.array([y[1], y[0] - np.sin(t)]),
                              (0, 4), [0.0, 0.0], n=1200)
        resid = max(abs(sol.y[i, 0] - math.sin(sol.t[i])) for i in range(len(sol.t)))
        self.assertLess(resid, 1e-10)

    def test_delay_equation(self):
        def exact(t):
            if t <= 1:
                return 1 - t
            if t <= 2:
                return 1 - t + (t - 1)**2 / 2
            if t <= 3:
                return 1 - t + (t - 1)**2 / 2 - (t - 2)**3 / 6
            return 1 - t + (t - 1)**2 / 2 - (t - 2)**3 / 6 + (t - 3)**4 / 24

        sol = dde_method_of_steps(lambda t, y, lags: -lags[0],
                                  lambda t: np.array([1.0]), [1.0], (0, 4),
                                  rtol=1e-12, atol=1e-14)
        err = max(abs(sol.y[j, 0] - exact(sol.t[j])) for j in range(len(sol.t)))
        self.assertLess(err, 1e-7)

    def test_stiffness_detection(self):
        stiff, ratio = detect_stiffness(
            lambda t, y: np.array([y[1], 1000 * (1 - y[0]**2) * y[1] - y[0]]),
            0.0, np.array([2.0, 0.0]))
        self.assertTrue(stiff)
        self.assertGreater(ratio, 1e5)
        mild, _ = detect_stiffness(lambda t, y: np.array([y[1], -y[0]]),
                                   0.0, np.array([1.0, 0.0]))
        self.assertFalse(mild)

    def test_dense_output_is_cubic_not_linear(self):
        """The stored slopes lift interpolation from O(h^2) to O(h^4)."""
        sol = solve_ivp(self.F, (0, 4), [0.0], rtol=1e-10, atol=1e-12)
        q = np.linspace(0, 4, 401)
        self.assertLess(np.abs(sol(q)[:, 0] - self.EXACT(q)).max(), 1e-6)
        errs = []
        for n in (25, 50, 100):
            s = rk4(self.F, (0, 4), [0.0], n=n)
            qq = np.linspace(0, 4, 997)
            errs.append(np.abs(s(qq)[:, 0] - self.EXACT(qq)).max())
        self.assertGreater(errs[0] / errs[1], 8.0)     # 16 for O(h^4)
        self.assertGreater(errs[1] / errs[2], 8.0)


class TestHighResolutionPDE(unittest.TestCase):
    def test_weno5_high_order_on_smooth_data(self):
        errs = []
        for nx in (40, 80, 160):
            s = weno_conservation_law(lambda x: np.sin(x), lambda v: v,
                                      lambda v: np.ones_like(v),
                                      (0, 2 * np.pi), (0, 1.0), nx=nx, cfl=0.2)
            errs.append(np.max(np.abs(np.asarray(s.final) - np.sin(s.x - 1.0))))
        self.assertGreater(math.log2(errs[0] / errs[1]), 4.0)
        self.assertGreater(math.log2(errs[1] / errs[2]), 4.0)

    def test_weno_captures_a_square_wave_without_oscillation(self):
        s = weno_conservation_law(lambda x: np.where((x > 2.0) & (x < 4.0), 1.0, 0.0),
                                  lambda v: v, lambda v: np.ones_like(v),
                                  (0, 2 * np.pi), (0, 2 * np.pi), nx=400, cfl=0.4)
        u = np.asarray(s.final)
        self.assertLess(np.abs(np.diff(np.concatenate([u, u[:1]]))).sum(), 2.02)
        self.assertLess(u.max() - 1.0, 5e-3)
        self.assertGreater(u.min(), -5e-3)

    def test_burgers_shock_speed(self):
        s = weno_burgers(lambda x: np.where(x < np.pi, 1.0, 0.0),
                         (0, 2 * np.pi), (0, 1.0), nx=400)
        u = np.asarray(s.final)
        x = np.asarray(s.x)
        # Periodic data has two structures: the shock moving right from x = pi
        # at the Rankine-Hugoniot speed (u_l + u_r)/2 = 1/2, and a rarefaction
        # fan spreading from the 0 -> 1 jump at the wrap-around.  Both cross
        # u = 0.5, so the search has to be restricted to the shock.
        band = (x > 2.5) & (x < 5.0)
        idx = int(np.argmin(np.abs(u[band] - 0.5)))
        self.assertAlmostEqual(float(x[band][idx]), np.pi + 0.5, delta=0.05)
        # The shock must stay sharp: no more than a couple of transition cells.
        transition = int(np.sum((u[band] > 0.05) & (u[band] < 0.95)))
        self.assertLessEqual(transition, 4)

    def test_ssp_rk3_is_third_order(self):
        errs = []
        for n in (10, 20, 40):
            dt = 1.0 / n
            u = np.array([1.0])
            for _ in range(n):
                u = ssp_rk3(lambda v: -v, u, dt)
            errs.append(abs(u[0] - math.exp(-1.0)))
        self.assertGreater(errs[0] / errs[1], 6.0)
        self.assertGreater(errs[1] / errs[2], 6.0)

    def test_projection_is_divergence_free_and_second_order(self):
        nu = 0.05
        errs = []
        for n in (32, 64):
            s = navier_stokes_2d(lambda X, Y: np.cos(X) * np.sin(Y),
                                 lambda X, Y: -np.sin(X) * np.cos(Y),
                                 nu, (0, 1.0), nx=n, ny=n, nt=400)
            X, Y = np.meshgrid(s.x, s.y, indexing="xy")
            decay = math.exp(-2 * nu)
            u, v = np.asarray(s.u)[-1]
            errs.append(np.abs(u - np.cos(X) * np.sin(Y) * decay).max())
            dx = s.x[1] - s.x[0]
            div = ((np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
                   + (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx))
            self.assertLess(np.abs(div).max(), 1e-12)
        self.assertGreater(errs[0] / errs[1], 3.0)

    def test_vorticity_streamfunction_is_spectrally_exact(self):
        nu = 0.05
        s = vorticity_streamfunction(lambda X, Y: 2 * np.cos(X) * np.cos(Y),
                                     nu, (0, 1.0), nx=64, ny=64, nt=200)
        X, Y = np.meshgrid(s.x, s.y, indexing="xy")
        exact = 2 * np.cos(X) * np.cos(Y) * math.exp(-2 * nu)
        self.assertLess(np.abs(np.asarray(s.u)[-1] - exact).max(), 1e-12)

    def test_lid_driven_cavity_matches_the_ghia_benchmark(self):
        """Ghia, Ghia & Shin (1982) is the standard reference solution."""
        psi, w, x = lid_driven_cavity(re=100.0, n=65, tol=1e-7, max_iter=40000)
        i = np.unravel_index(int(np.argmin(psi)), psi.shape)
        self.assertAlmostEqual(float(x[i[1]]), 0.6172, delta=0.03)
        self.assertAlmostEqual(float(x[i[0]]), 0.7344, delta=0.03)
        self.assertAlmostEqual(float(psi.min()), -0.1034, delta=0.005)

    def test_neumann_poisson_is_second_order(self):
        """The ghost-point boundary keeps the whole scheme second order; a
        one-sided difference there would drop it to first."""
        f = lambda x, y: -2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)
        errs = []
        for n in (20, 40, 80):
            s = poisson_neumann(f, (0, 1), (0, 1), n, n)
            X, Y = np.meshgrid(s.x, s.y, indexing="ij")
            u = np.asarray(s.u)
            ue = np.cos(np.pi * X) * np.cos(np.pi * Y)
            errs.append(np.abs((u - u.mean()) - (ue - ue.mean())).max())
        self.assertGreater(errs[0] / errs[1], 3.8)
        self.assertGreater(errs[1] / errs[2], 3.8)
        direct = poisson_neumann(f, (0, 1), (0, 1), 40, 40, method="dct")
        sor = poisson_neumann(f, (0, 1), (0, 1), 40, 40, method="sor", tol=1e-12)
        self.assertLess(np.abs(np.asarray(direct.u) - np.asarray(sor.u)).max(), 1e-9)

    def test_periodic_poisson_symbols(self):
        n, L = 64, 2 * np.pi
        x = np.linspace(0, L, n, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="xy")
        p = poisson_periodic_fft(-2 * np.sin(X) * np.sin(Y), L / n, L / n)
        self.assertLess(np.abs(p - np.sin(X) * np.sin(Y)).max(), 5e-3)

    def test_weno_weights_recover_the_linear_ones_on_smooth_data(self):
        x = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        u = np.sin(x)
        rec = weno5_reconstruct(u)
        ideal = np.interp(x + 0.5 * (x[1] - x[0]), x, u, period=2 * np.pi)
        self.assertLess(np.abs(rec - ideal).max(), 1e-3)


class TestNewSpecialFunctions(unittest.TestCase):
    def test_lambert_w_solves_its_defining_equation(self):
        for x in [-0.36, -0.3, -0.1, 0.0, 0.5, 1.0, math.e, 10.0, 1e3, 1e10]:
            w = lambert_w(x)
            self.assertLess(abs(w * math.exp(w) - x) / max(abs(x), 1e-3), 1e-12)
        self.assertAlmostEqual(lambert_w(math.e), 1.0, places=14)
        self.assertAlmostEqual(lambert_w(1.0), 0.5671432904097839, places=14)
        for x in [-0.36, -0.2, -0.01, -1e-6]:
            w = lambert_w(x, branch=-1)
            self.assertLess(w, -1.0)
            self.assertLess(abs(w * math.exp(w) - x) / abs(x), 1e-11)

    def test_dawson_satisfies_its_differential_equation(self):
        """F' = 1 - 2xF pins the function down without any reference table."""
        for x in [0.05, 0.5, 1.0, 3.0, 5.0, 12.0, 40.0]:
            h = 1e-5
            d = (dawson(x + h) - dawson(x - h)) / (2 * h)
            self.assertLess(abs(d - (1 - 2 * x * dawson(x))), 1e-8)
        self.assertAlmostEqual(dawson(1.0), 0.5380795069127684, places=14)
        self.assertAlmostEqual(dawson(-2.5), -dawson(2.5), places=15)

    def test_erfcx_survives_where_erfc_underflows(self):
        for x in (0.1, 1.0, 3.0):
            self.assertAlmostEqual(erfcx(x) / (math.exp(x * x) * math.erfc(x)), 1.0,
                                   places=12)
        self.assertGreater(erfcx(1e300), 0.0)          # erfc(1e300) is 0
        self.assertAlmostEqual(1e5 * math.sqrt(math.pi) * erfcx(1e5), 1.0, places=8)

    def test_fresnel_integrals_by_their_derivatives(self):
        for x in [0.3, 1.0, 2.0, 3.0, 5.0, 12.0]:
            h = 1e-5
            ds = (fresnel_s(x + h) - fresnel_s(x - h)) / (2 * h)
            dc = (fresnel_c(x + h) - fresnel_c(x - h)) / (2 * h)
            self.assertLess(abs(ds - math.sin(math.pi * x * x / 2)), 1e-6)
            self.assertLess(abs(dc - math.cos(math.pi * x * x / 2)), 1e-6)
        self.assertAlmostEqual(fresnel_s(1.0), 0.4382591473903548, places=12)
        self.assertAlmostEqual(fresnel_c(1.0), 0.7798934003768228, places=12)

    def test_polygamma_recurrence_and_closed_forms(self):
        for n in (1, 2, 3, 4):
            for x in [0.7, 1.5, 3.0, 25.0]:
                lhs = polygamma(n, x + 1) - polygamma(n, x)
                rhs = ((-1.0) ** n) * math.factorial(n) / x ** (n + 1)
                self.assertAlmostEqual(lhs / rhs, 1.0, places=12)
        self.assertAlmostEqual(trigamma(1.0), math.pi**2 / 6, places=13)
        self.assertAlmostEqual(trigamma(0.5), math.pi**2 / 2, places=12)
        self.assertAlmostEqual(polygamma(2, 1.0), -2 * 1.2020569031595943, places=13)
        self.assertAlmostEqual(polygamma(3, 1.0), math.pi**4 / 15, places=11)

    def test_modified_bessel_wronskian(self):
        """I0 K1 + I1 K0 = 1/x is an exact identity, independent of any table."""
        for x in [1e-3, 0.05, 0.5, 5.0, 14.9, 15.1, 60.0, 300.0]:
            w = bessel_i0(x) * bessel_k1(x) + bessel_i1(x) * bessel_k0(x)
            self.assertAlmostEqual(w * x, 1.0, places=9)

    def test_bessel_in_and_kn_recurrences(self):
        for x in [0.5, 3.0, 10.0, 25.0]:
            for n in (1, 3, 7, 12):
                lhs = bessel_in(n - 1, x) - bessel_in(n + 1, x)
                self.assertAlmostEqual(lhs / (2 * n / x * bessel_in(n, x)), 1.0,
                                       places=11)
        for x in [0.5, 3.0, 10.0]:
            for n in (1, 3, 6):
                lhs = bessel_kn(n + 1, x) - bessel_kn(n - 1, x)
                self.assertAlmostEqual(lhs / (2 * n / x * bessel_kn(n, x)), 1.0,
                                       places=11)
        self.assertAlmostEqual(bessel_in(-3, 2.0), bessel_in(3, 2.0), places=14)

    def test_spherical_bessel_wronskian_and_closed_forms(self):
        for n in (0, 1, 2, 5, 10):
            for x in [0.5, 2.0, 5.0, 30.0]:
                h = 1e-6
                dj = (spherical_bessel_j(n, x + h) - spherical_bessel_j(n, x - h)) / (2 * h)
                dy = (spherical_bessel_y(n, x + h) - spherical_bessel_y(n, x - h)) / (2 * h)
                w = spherical_bessel_j(n, x) * dy - dj * spherical_bessel_y(n, x)
                self.assertAlmostEqual(w * x * x, 1.0, places=5)
        for x in (1.0, 5.0, 20.0, 40.0):
            closed = (3 / x**3 - 1 / x) * math.sin(x) - 3 * math.cos(x) / x**2
            self.assertAlmostEqual(spherical_bessel_j(2, x), closed, places=13)

    def test_spherical_harmonics_are_orthonormal(self):
        from quadrivium.approx import gauss_legendre_nodes

        xs, ws = gauss_legendre_nodes(60)

        def inner(l1, m, l2):
            th = np.arccos(xs)
            v = np.array([np.conj(spherical_harmonic(l1, m, t, 0.0))
                          * spherical_harmonic(l2, m, t, 0.0) for t in th])
            return float(np.real(2 * np.pi * np.sum(ws * v)))

        for l, m in [(0, 0), (2, 1), (4, 3)]:
            self.assertAlmostEqual(inner(l, m, l), 1.0, places=11)
        self.assertLess(abs(inner(1, 0, 3)), 1e-12)
        self.assertLess(abs(spherical_harmonic(3, -2, 0.7, 1.1)
                            - np.conj(spherical_harmonic(3, 2, 0.7, 1.1))), 1e-14)
        for l in range(5):
            for x in np.linspace(-1, 1, 7):
                self.assertAlmostEqual(
                    associated_legendre(l, 0, x),
                    float(np.polynomial.legendre.Legendre.basis(l)(x)), places=13)

    def test_hypergeometric_closed_forms(self):
        for z in [-100.0, -20.0, -1.0, 0.5, 20.0, 60.0]:
            self.assertAlmostEqual(hyp1f1(1, 1, z) / math.exp(z), 1.0, places=13)
            self.assertAlmostEqual(hyp1f1(2.5, 2.5, z) / math.exp(z), 1.0, places=13)
        for z in [-60.0, -5.0, 0.5, 20.0]:
            self.assertAlmostEqual(hyp1f1(1, 2, z) / ((math.exp(z) - 1) / z), 1.0,
                                   places=13)
        for x in [0.3, 1.0, 4.0]:
            ref = math.sqrt(math.pi) * math.erf(x) / (2 * x)
            self.assertAlmostEqual(hyp1f1(0.5, 1.5, -x * x) / ref, 1.0, places=13)
        for z in [-0.9, -0.4, 0.2, 0.95]:
            self.assertAlmostEqual(hyp2f1(1, 1, 2, z) / (-math.log(1 - z) / z), 1.0,
                                   places=12)
            self.assertAlmostEqual(hyp2f1(2.0, 3, 3, z) / (1 - z) ** -2.0, 1.0,
                                   places=12)

    def test_expint_recurrence(self):
        for n in (1, 2, 4, 8):
            for x in [0.3, 1.0, 3.0, 10.0]:
                lhs = n * expint_n(n + 1, x)
                self.assertAlmostEqual(lhs / (math.exp(-x) - x * expint_n(n, x)), 1.0,
                                       places=11)
        self.assertAlmostEqual(expint_n(1, 1.0), 0.21938393439552027, places=13)

    def test_zeta_off_the_convergence_half_plane(self):
        from quadrivium.special import gamma

        for s in (0.01, 0.25, 0.6, 0.99, 2.3, -0.5, -7.3):
            lhs = zeta(s)
            rhs = (2.0**s * math.pi ** (s - 1.0) * math.sin(0.5 * math.pi * s)
                   * gamma(1.0 - s) * zeta(1.0 - s))
            self.assertAlmostEqual(lhs / rhs, 1.0, places=12)
        self.assertAlmostEqual(zeta(-1.0), -1.0 / 12.0, places=14)

    def test_logistic_avoids_overflow(self):
        self.assertEqual(float(logistic(-800.0)), 0.0)
        self.assertEqual(float(logistic(800.0)), 1.0)
        p = np.array([1e-12, 0.01, 0.5, 0.99, 1 - 1e-12])
        self.assertLess(np.abs(logistic(logit(p)) - p).max(), 1e-15)


class TestNewQuadratureAndApproximation(unittest.TestCase):
    def test_filon_beats_gauss_on_oscillatory_integrals(self):
        """Filon's accuracy *improves* with frequency; ordinary quadrature's
        collapses, because it needs points per oscillation."""
        from quadrivium.integrate import gauss_legendre

        exact = lambda w: (math.exp(1) * (math.sin(w) - w * math.cos(w)) + w) / (1 + w * w)
        for w in (100.0, 1000.0, 10000.0):
            fv = filon(np.exp, 0, 1, w, n=200, kind="sin").value
            self.assertLess(abs(fv - exact(w)), 1e-9)
        gv = gauss_legendre(lambda x: np.exp(x) * np.sin(1000.0 * x), 0, 1, 200).value
        self.assertGreater(abs(gv - exact(1000.0)), 1e-3)

    def test_cauchy_principal_value(self):
        self.assertAlmostEqual(cauchy_principal_value(lambda x: 1.0, -1, 1, 0.0).value,
                               0.0, places=12)
        self.assertAlmostEqual(cauchy_principal_value(lambda x: x, 0, 3, 1.0).value,
                               3 + math.log(2), places=10)

    def test_hadamard_finite_part(self):
        self.assertAlmostEqual(hadamard_finite_part(lambda x: 1.0, -1, 1, 0.0).value,
                               -2.0, places=8)

    def test_sparse_grid_beats_the_tensor_product(self):
        for d in (4, 6):
            exact = (math.e - 1) ** d
            r = sparse_grid_quadrature(lambda p: np.exp(np.sum(p)), [0.0] * d,
                                       [1.0] * d, level=5)
            self.assertLess(abs(r.value - exact) / exact, 1e-4)
            self.assertLess(r.subintervals, 17 ** d)      # the tensor count

    def test_aaa_rational_approximation(self):
        """A rational approximant recovers a rational function exactly and
        handles poles that no polynomial can."""
        pts = np.linspace(-1, 1, 600)
        r, zj, _, _ = aaa(lambda z: 1 / (1 + 25 * z * z), pts, tol=1e-13)
        test = np.linspace(-0.99, 0.99, 501)
        self.assertLess(np.max(np.abs(np.asarray(r(test)) - 1 / (1 + 25 * test**2))),
                        1e-12)
        self.assertLessEqual(len(zj), 6)
        pts = np.linspace(-1.5, 1.5, 800)
        r, _, _, _ = aaa(np.tan, pts, tol=1e-13)
        test = np.linspace(-1.4, 1.4, 401)
        self.assertLess(np.max(np.abs(np.asarray(r(test)) - np.tan(test))), 1e-10)

    def test_chebyshev_economization_beats_truncation(self):
        c = np.array([1 / math.factorial(k) for k in range(11)])
        xs = np.linspace(-1, 1, 1001)
        for n in (3, 4, 5):
            eco = np.max(np.abs(np.polyval(chebyshev_economization(c, n), xs)
                                - np.exp(xs)))
            trunc = np.max(np.abs(np.polyval(c[:n + 1][::-1], xs) - np.exp(xs)))
            self.assertLess(eco, trunc / 5.0)


class TestGeometryAndOptimization(unittest.TestCase):
    def test_bezier_curves(self):
        from math import comb

        P = np.array([[0.0, 0], [1, 2], [3, -1], [4, 1]])
        c = bezier(P)
        self.assertTrue(np.allclose(c(0.0), P[0]))
        self.assertTrue(np.allclose(c(1.0), P[-1]))
        t = np.linspace(0, 1, 11)
        bern = np.array([sum(comb(3, i) * (1 - tt)**(3 - i) * tt**i * P[i]
                             for i in range(4)) for tt in t])
        self.assertLess(np.abs(np.asarray(c(t)) - bern).max(), 1e-14)
        dc = bezier(bezier_derivative(P))
        h = 1e-6
        num = (np.asarray(c(0.4 + h)) - np.asarray(c(0.4 - h))) / (2 * h)
        self.assertLess(np.abs(np.asarray(dc(0.4)) - num).max(), 1e-7)
        left, right = bezier_subdivide(P, 0.37)
        s = np.linspace(0, 1, 20)
        self.assertLess(np.abs(np.asarray(bezier(left)(s))
                               - np.asarray(c(0.37 * s))).max(), 1e-13)
        self.assertLess(np.abs(np.asarray(bezier(right)(s))
                               - np.asarray(c(0.37 + 0.63 * s))).max(), 1e-13)

    def test_nurbs_represents_a_circle_exactly(self):
        """No polynomial spline can do this -- it is why CAD uses rational forms."""
        circ = nurbs_circle(2.0, (1.0, -0.5))
        pts = np.asarray(circ(np.linspace(0, 1, 401)))
        r = np.linalg.norm(pts - np.array([1.0, -0.5]), axis=1)
        self.assertLess(np.abs(r - 2.0).max(), 1e-13)

    def test_clamped_nurbs_interpolates_its_end_points(self):
        P = np.array([[0.0, 0], [1, 3], [3, 3], [4, 0], [6, 2]])
        knots = open_uniform_knots(5, 3)
        c = nurbs(P, np.ones(5), knots, 3)
        self.assertTrue(np.allclose(c(0.0), P[0]))
        self.assertTrue(np.allclose(c(1.0), P[-1]))
        heavy = nurbs(P, np.array([1.0, 4.0, 1, 1, 1]), knots, 3)
        s = np.linspace(0, 1, 300)
        d_plain = np.min(np.linalg.norm(np.asarray(c(s)) - P[1], axis=1))
        d_heavy = np.min(np.linalg.norm(np.asarray(heavy(s)) - P[1], axis=1))
        self.assertLess(d_heavy, d_plain)

    def test_trilinear_matches_the_bilinear_calling_convention(self):
        gx, gy, gz = (np.linspace(0, 1, 5), np.linspace(0, 1, 6), np.linspace(0, 1, 7))
        V = gx[:, None, None] + 2 * gy[None, :, None] + 3 * gz[None, None, :]
        f = trilinear(gx, gy, gz, V)
        self.assertAlmostEqual(f(0.3, 0.4, 0.5), 0.3 + 0.8 + 1.5, places=12)
        self.assertTrue(np.allclose(f([0.1, 0.3], [0.2, 0.4], [0.3, 0.5]),
                                    [1.4, 2.6]))

    def test_anderson_acceleration(self):
        """On a linear fixed point Anderson with m >= n is essentially exact,
        where Picard crawls."""
        A = np.array([[0.0, 0.45, 0.4], [0.45, 0.0, 0.45], [0.4, 0.45, 0.0]])
        b = np.array([1.0, 2.0, 3.0])
        exact = np.linalg.solve(np.eye(3) - A, b)
        fast = anderson_acceleration(lambda x: A @ x + b, np.zeros(3), m=5, tol=1e-12)
        slow = anderson_acceleration(lambda x: A @ x + b, np.zeros(3), m=0, tol=1e-12)
        self.assertTrue(fast.converged)
        self.assertLess(np.linalg.norm(fast.root - exact), 1e-11)
        self.assertLess(fast.iterations, slow.iterations / 10)

    def test_newton_krylov_scales_and_accepts_a_preconditioner(self):
        def make(n, lam=1.0):
            h = 1.0 / (n + 1)

            def F(u):
                u = np.asarray(u)
                lap = (-np.concatenate([[0.0], u[:-1]]) + 2 * u
                       - np.concatenate([u[1:], [0.0]])) / h**2
                return lap - lam * np.exp(u)
            return F

        from quadrivium.rootfind import newton_system

        F = make(20)
        a = newton_krylov(F, np.zeros(20), tol=1e-12)
        b = newton_system(F, np.zeros(20), tol=1e-12)
        self.assertTrue(a.converged)
        self.assertLess(np.abs(a.root - b.root).max(), 1e-10)
        n = 400
        h = 1.0 / (n + 1)
        k = np.arange(1, n + 1)
        lam_k = (2 - 2 * np.cos(np.pi * k / (n + 1))) / h**2

        def precond(v):
            e = np.concatenate([[0.0], v, [0.0], -v[::-1]])
            c = -np.imag(np.fft.fft(e))[1:n + 1] / lam_k
            e2 = np.concatenate([[0.0], c, [0.0], -c[::-1]])
            return -np.imag(np.fft.fft(e2))[1:n + 1] / (n + 1)

        pre = newton_krylov(make(n), np.zeros(n), tol=1e-8, precond=precond)
        self.assertTrue(pre.converged)
        self.assertLess(pre.function_calls, 200)     # thousands without one

    def test_newton_cg(self):
        rosen = lambda x: float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

        def rgrad(x):
            g = np.zeros_like(x)
            g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
            g[1:] += 200 * (x[1:] - x[:-1]**2)
            return g

        for n in (2, 10, 100):
            x0 = np.full(n, -1.2)
            x0[1::2] = 1.0
            r = newton_cg(rosen, x0, rgrad, tol=1e-8, max_iter=2000)
            self.assertTrue(r.converged, msg=f"n={n}")
            self.assertLess(np.abs(r.x - 1.0).max(), 1e-6)
        # Negative curvature: a plain Newton step would head for the saddle.
        r = newton_cg(lambda v: v[0]**2 - v[1]**2 + v[1]**4, np.array([0.5, 0.01]),
                      lambda v: np.array([2 * v[0], -2 * v[1] + 4 * v[1]**3]),
                      tol=1e-10)
        self.assertAlmostEqual(r.fun, -0.25, places=8)

    def test_lbfgsb_respects_bounds(self):
        r = lbfgsb(lambda v: (v[0] - 3)**2 + (v[1] + 2)**2, np.zeros(2),
                   lambda v: np.array([2 * (v[0] - 3), 2 * (v[1] + 2)]),
                   bounds=[(0.0, 1.0), (0.0, 1.0)], tol=1e-12)
        self.assertTrue(np.allclose(r.x, [1.0, 0.0], atol=1e-8))
        self.assertAlmostEqual(r.fun, 8.0, places=10)
        rosen = lambda x: float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))
        rg = lambda x: np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
                                 200 * (x[1] - x[0]**2)])
        r = lbfgsb(rosen, np.array([-1.2, 1.0]), rg, tol=1e-10)
        self.assertLess(r.fun, 1e-14)
        r = lbfgsb(rosen, np.array([-1.2, 1.0]), rg,
                   bounds=[(-2.0, 0.5), (-2.0, 2.0)], tol=1e-10)
        self.assertTrue(np.allclose(r.x, [0.5, 0.25], atol=1e-6))


class TestRandomnessConventions(unittest.TestCase):
    """``rng=`` accepts whatever ``np.random.default_rng`` accepts.

    Every documented call takes an integer seed, a ``Generator``, or ``None``.
    Passing the seed straight through to a generator method instead of
    normalising it first raises ``AttributeError``, which is easy to miss
    because ``None`` and a ``Generator`` both work.
    """

    def _cases(self):
        rng = np.random.default_rng(7)
        low_rank = rng.standard_normal((40, 6)) @ rng.standard_normal((6, 30))
        spd = np.diag(np.arange(1.0, 13.0))
        drift, diffusion = (lambda x, t: -x), (lambda x, t: 0.2)
        return {
            "randomized_svd": lambda r: randomized_svd(low_rank, 4, rng=r)[1],
            "randomized_range_finder":
                lambda r: randomized_range_finder(low_rank, 6, rng=r),
            "nystrom_approximation": lambda r: nystrom_approximation(spd, 3, rng=r),
            "cur_decomposition": lambda r: cur_decomposition(low_rank, 3, rng=r)[1],
            "subspace_iteration":
                lambda r: subspace_iteration(spd, 2, rng=r).eigenvalues,
            "lobpcg": lambda r: lobpcg(spd, 2, rng=r).eigenvalues,
            "brownian_path": lambda r: brownian_path((0, 1), n=16, rng=r)[1],
            "euler_maruyama":
                lambda r: euler_maruyama(drift, diffusion, (0, 1), [1.0], n=16, rng=r).y,
            "milstein":
                lambda r: milstein(drift, diffusion, (0, 1), [1.0], n=16, rng=r).y,
            "monte_carlo": lambda r: monte_carlo(lambda x: x * x, 0, 1, n=500, rng=r).value,
        }

    def test_integer_seed_is_accepted_and_reproducible(self):
        for name, call in self._cases().items():
            with self.subTest(routine=name):
                first = np.asarray(call(0), dtype=float)
                second = np.asarray(call(0), dtype=float)
                self.assertTrue(np.array_equal(first, second))

    def test_generator_and_none_also_work(self):
        for name, call in self._cases().items():
            with self.subTest(routine=name):
                seeded = np.asarray(call(np.random.default_rng(3)), dtype=float)
                self.assertTrue(np.all(np.isfinite(seeded)))
                self.assertTrue(np.all(np.isfinite(np.asarray(call(None), dtype=float))))

    def test_the_same_seed_means_the_same_stream(self):
        """An integer seed must reach the generator, not merely be accepted."""
        by_int = randomized_svd(np.diag(np.arange(1.0, 9.0)), 3, rng=5)[1]
        by_gen = randomized_svd(np.diag(np.arange(1.0, 9.0)), 3,
                                rng=np.random.default_rng(5))[1]
        self.assertTrue(np.array_equal(by_int, by_gen))


if __name__ == "__main__":
    unittest.main(verbosity=2)
