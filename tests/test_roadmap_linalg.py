"""Independent identities and bounded-storage regressions for solver infrastructure."""
import concurrent.futures

import numpy as oracle
import pytest

from quadrivium import numeric as np
from quadrivium.linalg import (
    LinearOperator, aslinearoperator, LUFactor, CholeskyFactor, QRFactor,
    COOMatrix, CSRMatrix, CSCMatrix, from_dense, identity_sparse, spmm,
    conjugate_gradient, gmres, cgnr, lsqr, ilu0, incomplete_cholesky,
    ilu_preconditioner, ichol_preconditioner, sparse_triangular_solve,
)
from quadrivium.optimize import least_squares, curve_fit, bfgs, adam, minimize


def close(actual, expected, atol=1e-10):
    oracle.testing.assert_allclose(oracle.asarray(actual), expected, rtol=1e-9, atol=atol)


def test_operator_composition_adjoint_and_shapes():
    a = oracle.array([[1 + 2j, 3], [-2j, 4], [2, 1j]])
    b = oracle.array([[1, 2j, 3], [2, 0, 1]])
    A, B = aslinearoperator(a), aslinearoperator(b)
    x = np.array([1 - 1j, 2, 3j])
    close((A @ B) @ x, (a @ b) @ oracle.asarray(x))
    close(A.H @ x, a.conj().T @ oracle.asarray(x))
    close(A.T @ x, a.T @ oracle.asarray(x))
    close((2j * A + A) @ np.ones(2), (2j * a + a) @ oracle.ones(2))
    close(A @ np.eye(2), a)
    with pytest.raises(ValueError):
        A @ np.ones(4)
    with pytest.raises(NotImplementedError):
        LinearOperator((2, 2), lambda v: v).rmatvec(np.ones(2))


@pytest.mark.parametrize("complex_data", [False, True])
def test_reusable_factors_multiple_rhs_transpose_and_thread_safety(complex_data):
    rng = oracle.random.default_rng(820)
    a = rng.normal(size=(9, 9)) + 4 * oracle.eye(9)
    b = rng.normal(size=(9, 4))
    if complex_data:
        a = a + 1j * rng.normal(size=(9, 9))
        b = b + 1j * rng.normal(size=(9, 4))
    original = a.copy()
    factor = LUFactor(a)
    expected = oracle.linalg.solve(a, b)
    close(factor.solve(b), expected)
    out = np.empty(b.shape, dtype=complex if complex_data else float)
    assert factor.solve(b, out=out) is out
    close(out, expected)
    close(factor.solve(b, trans="T"), oracle.linalg.solve(a.T, b))
    close(factor.solve(b, trans="H"), oracle.linalg.solve(a.conj().T, b))
    sign, logabs = factor.slogdet()
    expected_sign, expected_log = oracle.linalg.slogdet(a)
    close(sign, expected_sign)
    close(logabs, expected_log)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        results = list(pool.map(factor.solve, [b[:, i] for i in range(4)]))
    for j, result in enumerate(results):
        close(result, expected[:, j])
    close(a, original)
    spd = a.conj().T @ a + oracle.eye(9)
    chol = CholeskyFactor(spd)
    close(chol.solve(b), oracle.linalg.solve(spd, b))
    close(chol.logdet(), oracle.linalg.slogdet(spd)[1])
    tall = oracle.vstack([a, a[:3]])
    tall_b = rng.normal(size=(12, 4))
    qr = QRFactor(tall)
    close(qr.solve(tall_b), oracle.linalg.lstsq(tall, tall_b, rcond=None)[0])
    assert qr.packed.shape == tall.shape
    assert not hasattr(qr, "Q")


def test_sparse_products_duplicates_empty_rows_formats():
    a = COOMatrix([0, 0, 0, 2], [1, 1, 2, 0], [2, -1, 4, 3], (4, 3))
    b = COOMatrix([0, 1, 2, 2], [1, 2, 0, 0], [2, 3, 4, -1], (3, 4))
    expected = oracle.asarray(a.todense()) @ oracle.asarray(b.todense())
    for left in (a, a.tocsr(), a.tocsc()):
        for right in (b, b.tocsr(), b.tocsc()):
            product = left @ right
            assert isinstance(product, CSRMatrix)
            close(product.todense(), expected)
    close(spmm(oracle.ones((2, 4)), a), oracle.ones((2, 4)) @ oracle.asarray(a.todense()))
    close(a.T.T.todense(), a.todense())
    close(a.sum_duplicates().todense(), a.todense())


def test_sparse_preconditioners_never_densify(monkeypatch):
    n = 1500
    rows = oracle.concatenate([oracle.arange(n), oracle.arange(n - 1), oracle.arange(1, n)])
    cols = oracle.concatenate([oracle.arange(n), oracle.arange(1, n), oracle.arange(n - 1)])
    data = oracle.concatenate([oracle.full(n, 4.), oracle.full(2 * n - 2, -1.)])
    A = COOMatrix(rows, cols, data, (n, n)).tocsc()
    monkeypatch.setattr(CSRMatrix, "todense", lambda _: pytest.fail("dense conversion"))
    monkeypatch.setattr(CSCMatrix, "todense", lambda _: pytest.fail("dense conversion"))
    L, U = ilu0(A)
    C = incomplete_cholesky(A)
    assert L.nnz + U.nnz <= 4 * n
    assert C.nnz <= 2 * n
    wanted = np.linspace(0, 1, n)
    b = A @ wanted
    for preconditioner in (ilu_preconditioner(A), ichol_preconditioner(A)):
        close(preconditioner(b), wanted, atol=2e-12)
    product = A @ identity_sparse(n)
    assert product.nnz == A.nnz
    close(product @ wanted, b)


@pytest.mark.parametrize("solver", [conjugate_gradient, gmres])
def test_krylov_public_operator(solver):
    diag = np.arange(1, 41, dtype=float)
    A = LinearOperator((40, 40), lambda x: diag * x, lambda x: diag * x)
    result = solver(A, diag, tol=1e-10)
    assert result.converged
    close(result.x, oracle.ones(40), atol=1e-8)


@pytest.mark.parametrize("solver", [cgnr, lsqr])
def test_rectangular_operator_inconsistent_least_squares(solver):
    rng = oracle.random.default_rng(91)
    a = rng.normal(size=(20, 6))
    b = rng.normal(size=20)
    A = aslinearoperator(a)
    result = solver(A, b)
    assert result.converged
    close(result.x, oracle.linalg.lstsq(a, b, rcond=None)[0])
    zero = solver(A, oracle.zeros(20))
    assert zero.converged and zero.iterations == 0
    close(zero.x, oracle.zeros(6))


def test_lsqr_damping_matches_augmented_system():
    rng = oracle.random.default_rng(903)
    a, b = rng.normal(size=(17, 5)), rng.normal(size=17)
    result = lsqr(aslinearoperator(a), b, damp=0.7)
    expected = oracle.linalg.solve(a.T @ a + 0.49 * oracle.eye(5), a.T @ b)
    assert result.converged
    close(result.x, expected)


def test_robust_fit_rejects_outlier_and_honors_bounds():
    samples = np.array([1.9, 2.0, 2.1, 100.0])
    result = least_squares(lambda x: x[0] - samples, [0.], loss="huber")
    assert result.converged
    close(result.x, [2 + 1 / 3], atol=1e-6)
    bounded = least_squares(lambda x: np.array([x[0] - 2, 2 * x[1] - 6]),
                            [0., 0.], bounds=([0, 0], [1, 5]), store_history=False)
    assert bounded.converged and bounded.history == []
    close(bounded.x, [1, 3])
    close(bounded.active_mask, [1, 0])
    assert bounded.optimality < 1e-7


def test_sparse_operator_jacobian_and_scaling():
    diag = np.linspace(1, 3, 300)
    desired = np.linspace(1, 2, 300)
    matrix = COOMatrix(np.arange(300), np.arange(300), diag, (300, 300)).tocsr()
    operator = aslinearoperator(matrix)
    for jac in (lambda x: matrix, lambda x: operator):
        result = least_squares(lambda x: diag * (x - desired), np.zeros(300),
                               jac=jac, x_scale="jac", store_history=False)
        assert result.converged
        close(result.x, desired, atol=1e-8)


def test_colored_finite_differences_and_domain_bounds():
    n = 80
    calls = 0
    def residual(x):
        nonlocal calls
        calls += 1
        assert np.all(x >= 0)
        return x * x - 4
    result = least_squares(residual, np.ones(n), bounds=(0, np.inf),
                           jac_sparsity=identity_sparse(n), jac="3-point")
    assert result.converged
    close(result.x, oracle.full(n, 2.), atol=1e-8)
    assert calls < 60  # grouped columns avoid O(n) residual evaluations per Jacobian


def test_weighted_curve_fit_covariance_controls():
    x = np.linspace(0, 1, 20)
    y = 2 * x + 1
    model = lambda t, a, b: a * t + b
    fit = curve_fit(model, x, y, [0., 0.], sigma=np.linspace(1, 2, 20),
                    jac=lambda p: np.column_stack((x, np.ones(20))),
                    bounds=([0, 0], [4, 4]), absolute_sigma=True)
    assert fit.converged
    close(fit.x, [2, 1], atol=1e-8)
    assert fit.covariance is not None
    robust = curve_fit(model, x, y, [0., 0.], loss="soft_l1")
    assert robust.converged and robust.covariance is None


def test_optimizer_histories_callbacks_and_nested_dispatch():
    objective = lambda x: float(x @ x)
    grad = lambda x: 2 * x
    full = bfgs(objective, [3., -2.], grad)
    empty = minimize(objective, [3., -2.], grad_f=grad, store_history=False)
    assert full.history and empty.history == []
    close(full.x, empty.x)
    seen = []
    def callback(x):
        seen.append(x.copy())
        x[:] = 12345  # callback cannot mutate the iterate
        return len(seen) == 3
    stopped = adam(objective, [3., -2.], grad, callback=callback, store_history=False)
    assert stopped.message == "stopped by callback" and not stopped.converged
    assert len(seen) == 3 and stopped.history == []
    assert np.max(np.abs(stopped.x)) < 3
    sparse = adam(objective, [3., -2.], grad, max_iter=20, history_stride=5)
    assert len(sparse.history) == 5


def test_sparse_real_matrix_accepts_complex_vectors():
    a = from_dense([[2., 1.], [0., 3.], [1., -2.]])
    x = np.array([1 + 2j, 3 - 1j])
    for matrix in (a, a.tocsc(), a.tocoo()):
        close(matrix @ x, oracle.asarray(a.todense()) @ oracle.asarray(x))
        close(aslinearoperator(matrix).H @ np.array([1j, 2., 3.]),
              oracle.asarray(a.todense()).T @ oracle.array([1j, 2., 3.]))


def test_factor_and_solver_validation():
    with pytest.raises(Exception, match="rank deficient"):
        QRFactor([[1, 2], [2, 4], [3, 6]])
    with pytest.raises(ValueError, match="symmetric"):
        incomplete_cholesky(from_dense([[2, 1], [0, 2]]))
    with pytest.raises(ValueError, match="restart"):
        gmres(identity_sparse(2), [1, 1], restart=0)
    with pytest.raises(ValueError, match="outside bounds"):
        least_squares(lambda x: x, [2.], bounds=(0, 1))


def test_krylov_history_retention_does_not_change_stopping():
    diagonal = np.linspace(1, 40, 40)
    A = LinearOperator((40, 40), lambda x: diagonal * x)
    full = conjugate_gradient(A, np.ones(40), tol=1e-12)
    empty = conjugate_gradient(A, np.ones(40), tol=1e-12, store_history=False)
    decimated = conjugate_gradient(A, np.ones(40), tol=1e-12, history_stride=7)
    assert empty.iterations == decimated.iterations == full.iterations
    assert len(empty.residuals) == 1
    assert len(decimated.residuals) < len(full.residuals)
    close(empty.x, full.x)
    close(decimated.residual, full.residual)
    calls = []
    def callback(x):
        calls.append(x.copy())
        return len(calls) == 3
    stopped = gmres(A, np.ones(40), callback=callback, store_history=False)
    assert stopped.message == "stopped by callback" and not stopped.converged
    close(stopped.x, calls[-1])
    assert np.linalg.norm(stopped.x) > 0


def test_sparse_ssor_matches_factor_identity():
    from quadrivium.linalg import ssor_preconditioner
    a = oracle.array([[4., -1, 0], [-1, 4, -1], [0, -1, 4.]])
    omega = 1.3
    d = oracle.diag(oracle.diag(a))
    lower, upper = d / omega + oracle.tril(a, -1), d / omega + oracle.triu(a, 1)
    matrix = omega / (2 - omega) * lower @ oracle.linalg.solve(d, upper)
    b = oracle.array([1., 2., 3.])
    for A in (a, from_dense(a)):
        close(ssor_preconditioner(A, omega)(b), oracle.linalg.solve(matrix, b))


def test_root_histories_and_callbacks_preserve_iterates():
    from quadrivium.rootfind import bisection, newton_system, anderson_acceleration, continuation
    f = lambda x: x * x - 2
    full = bisection(f, 0, 2)
    limited = bisection(f, 0, 2, store_history=False)
    sampled = bisection(f, 0, 2, history_stride=4)
    assert limited.history == [] and len(sampled.history) < len(full.history)
    assert limited.root == sampled.root == full.root
    seen = []
    stopped = newton_system(lambda x: x * x - 2, [3., 4.],
                            callback=lambda x: seen.append(x.copy()) or True,
                            store_history=False)
    assert stopped.message == "stopped by callback" and stopped.history == []
    close(stopped.root, seen[-1])
    close(stopped.f_root, oracle.asarray(stopped.root) ** 2 - 2)
    accelerated = anderson_acceleration(lambda x: 0.5 * x + 1, [0., 0.], store_history=False)
    assert accelerated.converged and accelerated.history == []
    path = continuation(lambda x: x * x - 2, [1.], steps=4, store_history=False)
    assert path.converged and path.history == []


def test_history_policies_are_isolated_across_threads():
    from quadrivium.rootfind import bisection
    def solve(store):
        return bisection(lambda x: x * x - 2, 0, 2, store_history=store)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(solve, [True, False]))
    assert results[0].history and not results[1].history
    close(results[0].root, results[1].root)


def test_high_level_complex_factorizations_preserve_imaginary_parts():
    from quadrivium import accel
    from quadrivium.linalg import cholesky, cholesky_solve, householder_qr, jacobi_eigen, svd_jacobi, polar_decomposition
    rng = oracle.random.default_rng(285)
    a = rng.normal(size=(7, 5)) + 1j * rng.normal(size=(7, 5))
    hermitian = a.conj().T @ a + oracle.eye(5)
    b = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    L = cholesky(hermitian)
    close(L @ np.conjugate(L.T), hermitian)
    upper = cholesky(hermitian, lower=False)
    close(np.conjugate(upper.T) @ upper, hermitian)
    close(cholesky_solve(hermitian, b), oracle.linalg.solve(hermitian, b))
    Q, R = householder_qr(a)
    close(Q @ R, a)
    close(np.conjugate(Q.T) @ Q, oracle.eye(5))
    for disabled in (False, True):
        context = accel.disabled() if disabled else accel.enabled()
        with context:
            eig = jacobi_eigen(hermitian)
            assert eig.converged
            close(hermitian @ oracle.asarray(eig.eigenvectors),
                  oracle.asarray(eig.eigenvectors) * oracle.asarray(eig.eigenvalues), atol=2e-10)
            for matrix in (a, a.T):
                U, S, Vh = svd_jacobi(matrix)
                close((U * S) @ Vh, matrix)
                close(S, oracle.linalg.svd(matrix, compute_uv=False))
    limited = jacobi_eigen(hermitian, max_sweeps=0)
    assert not limited.converged
    Q, P = polar_decomposition(a)
    close(Q @ P, a)
    close(P, oracle.asarray(P).conj().T)


def test_high_level_batches_match_individual_numpy_problems():
    import quadrivium as qd
    from quadrivium.linalg import solve, cholesky, householder_qr, svd_jacobi
    rng = oracle.random.default_rng(107)
    a = rng.normal(size=(2, 3, 4, 4)) + 1j * rng.normal(size=(2, 3, 4, 4))
    a = a @ a.conj().swapaxes(-1, -2) + oracle.eye(4)
    b = rng.normal(size=(4, 2))
    for method in ("auto", "lu", "plu", "qr", "cholesky"):
        close(solve(a, b, method=method), oracle.linalg.solve(a, b))
    close(qd.solve(a, b), oracle.linalg.solve(a, b))
    factor = cholesky(a)
    close(factor @ np.conjugate(np.swapaxes(factor, -1, -2)), a)
    Q, R = householder_qr(a)
    close(Q @ R, a)
    U, S, Vh = svd_jacobi(a)
    close((U * S[..., None, :]) @ Vh, a)
    with pytest.raises(ValueError, match="unknown"):
        solve(a, b, method="invalid")
    # Custom sweep budgets take the explicit per-matrix path without integer truncation.
    matrices = np.array([[[3, 1], [0, 2]], [[2, 1], [1, 4]]], dtype=int)
    U, S, Vh = svd_jacobi(matrices, max_sweeps=50)
    close((U * S[..., None, :]) @ Vh, matrices)
