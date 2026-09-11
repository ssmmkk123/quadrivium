"""Native grid kernels preserve in-place and allocation contracts."""

import tracemalloc

import pytest

from quadrivium import _accel, numeric as np
from quadrivium.linalg import thomas
from quadrivium.pde import lid_driven_cavity


pytestmark = pytest.mark.skipif(not _accel.available(), reason="native backend unavailable")


def kernel(name):
    return _accel.kernel(name)


@pytest.mark.parametrize("n", [0, 1, 7])
def test_jacobi_eigen_accepts_readonly_strided_input(n):
    rng = np.random.default_rng(890 + n)
    matrix = rng.standard_normal((n, n))
    matrix = (matrix + matrix.T)[::-1, ::-1]
    original = matrix.copy()
    matrix.flags.writeable = False
    values, vectors, sweeps, converged = kernel("jacobi_eigen")(matrix)
    np.testing.assert_array_equal(matrix, original)
    np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-10)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(n), atol=1e-12)
    assert converged
    assert 1 <= sweeps <= 100


def test_hessenberg_iteration_retains_similarity_and_mutates_original_storage():
    h = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [0.0, 3.0, 6.0]])
    original = h.copy()
    v = np.eye(3)
    h_view, v_view = h[:], v[:]
    iterations, converged = kernel("hessenberg_qr_iterate")(h, v, max_iter=3000)
    assert converged
    assert iterations > 0
    np.testing.assert_allclose(v @ h @ v.T, original, atol=1e-10)
    np.testing.assert_allclose(v.T @ v, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(h, h_view)
    np.testing.assert_array_equal(v, v_view)


@pytest.mark.parametrize("name", ["hessenberg_qr_iterate", "lid_driven_cavity"])
def test_overlapping_mutable_outputs_fail_before_any_write(name):
    storage = np.arange(12.0)
    a, b = storage[:9].reshape(3, 3), storage[3:].reshape(3, 3)
    original = storage.copy()
    args = (a, b) if name == "hessenberg_qr_iterate" else (a, b, 0.5, 0.01, 0.001, 1e-6, 10)
    with pytest.raises(ValueError, match="overlap"):
        kernel(name)(*args)
    np.testing.assert_array_equal(storage, original)


@pytest.mark.parametrize("name", ["hessenberg_qr_iterate", "sor_poisson", "lid_driven_cavity", "thomas_batch"])
@pytest.mark.parametrize("layout", ["strided", "readonly", "integer"])
def test_mutable_buffers_reject_invalid_storage(name, layout):
    output = np.ones((3, 3))
    if layout == "strided":
        output = output.T
    elif layout == "readonly":
        output.flags.writeable = False
    else:
        output = output.astype(int)
    original = output.copy()
    if name == "hessenberg_qr_iterate":
        args = (output,)
    elif name == "sor_poisson":
        args = (output, np.zeros((3, 3)), 1.0, 0.25, 1.0)
    elif name == "lid_driven_cavity":
        args = (output, np.zeros((3, 3)), 0.5, 0.01, 0.001, 1e-6, 10)
    else:
        args = (np.ones(2), np.full(3, 4.0), np.ones(2), output)
    with pytest.raises((TypeError, ValueError)):
        kernel(name)(*args)
    np.testing.assert_array_equal(output, original)


def test_sor_overlapping_source_is_snapshotted():
    u = np.arange(25.0).reshape(5, 5) / 25.0
    reference = u.copy()
    expected = kernel("sor_poisson")(reference, u.copy(), 1.2, 0.01, 1.1, max_iter=17)
    actual = kernel("sor_poisson")(u, u, 1.2, 0.01, 1.1, max_iter=17)
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(u, reference)
    np.testing.assert_array_equal(actual[2], expected[2])


def test_sor_huge_iteration_limit_does_not_preallocate_history():
    u = np.zeros((5, 5))
    source = np.zeros_like(u)
    tracemalloc.start()
    try:
        iterations, converged, residuals = kernel("sor_poisson")(
            u, source, 1.0, 0.01, 1.0, max_iter=10**12
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert iterations == 1
    assert converged
    assert residuals.shape == (1,)
    assert float(residuals[0]) == 0.0
    assert peak < 1024 * 1024


def test_sor_history_grows_beyond_initial_chunk_and_retains_every_sweep():
    u = np.zeros((3, 3))
    iterations, converged, residuals = kernel("sor_poisson")(
        u, u.copy(), 1.0, 0.25, 1.0, tol=0.0, max_iter=1025
    )
    assert iterations == 1025
    assert not converged
    np.testing.assert_array_equal(residuals, np.zeros(1025))


@pytest.mark.parametrize("iterations", [0, 1, 2, 9])
def test_cavity_swapped_buffers_reach_the_caller_on_even_and_odd_iterations(iterations):
    with _accel.disabled():
        expected = lid_driven_cavity(n=5, max_iter=iterations)
    actual = lid_driven_cavity(n=5, max_iter=iterations)
    for want, got in zip(expected, actual):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("empty_shape", [(0, 0), (0, 3), (3, 0)])
def test_thomas_batch_empty_systems(empty_shape):
    n = empty_shape[0]
    rhs = np.zeros(empty_shape)
    assert kernel("thomas_batch")(
        np.zeros(max(0, n - 1)), np.zeros(n), np.zeros(max(0, n - 1)), rhs
    ) is None
    assert rhs.shape == empty_shape


def test_thomas_batch_late_singular_pivot_leaves_rhs_untouched():
    rhs = np.arange(6.0).reshape(3, 2)
    original = rhs.copy()
    with pytest.raises(RuntimeError, match="singular:zero pivot"):
        kernel("thomas_batch")([1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0], rhs)
    np.testing.assert_array_equal(rhs, original)


def test_thomas_batch_snapshots_coefficients_that_alias_rhs():
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sub = rhs.ravel()[:2]
    sup = rhs.ravel()[3:5]
    diag = np.full(3, 12.0)
    expected = thomas(sub.copy(), diag, sup.copy(), rhs.copy())
    kernel("thomas_batch")(sub, diag, sup, rhs)
    np.testing.assert_array_equal(rhs, expected)


def test_thomas_preserves_all_inputs_and_handles_strided_vectors():
    sub = np.array([-1.0, 99.0, -2.0, 99.0])[::2]
    diag = np.array([8.0, 7.0, 6.0])[::-1]
    sup = np.array([-2.0, -1.0])[::-1]
    rhs = np.array([1.0, 2.0, 3.0])[::-1]
    inputs = (sub, diag, sup, rhs)
    originals = tuple(a.copy() for a in inputs)
    for a in inputs:
        a.flags.writeable = False
    actual = kernel("thomas")(*inputs)
    matrix = np.diag(diag) + np.diag(sub, -1) + np.diag(sup, 1)
    np.testing.assert_allclose(matrix @ actual, rhs, atol=1e-12)
    for a, original in zip(inputs, originals):
        np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("name", ["jacobi_eigen", "hessenberg_qr_iterate", "sor_poisson", "lid_driven_cavity"])
def test_negative_iteration_budget_is_rejected_before_mutation(name):
    output = np.ones((3, 3))
    original = output.copy()
    if name == "jacobi_eigen":
        args, kwargs = (output,), {"max_sweeps": -1}
    elif name == "hessenberg_qr_iterate":
        args, kwargs = (output,), {"max_iter": -1}
    elif name == "sor_poisson":
        args, kwargs = (output, output, 1.0, 0.25, 1.0), {"max_iter": -1}
    else:
        args, kwargs = (output, output.copy(), 0.5, 0.01, 0.001, 1e-6, -1), {}
    with pytest.raises(OverflowError):
        kernel(name)(*args, **kwargs)
    np.testing.assert_array_equal(output, original)


class _MutatingFloat:
    def __init__(self, mutate, value=1.0):
        self.mutate = mutate
        self.value = value

    def __float__(self):
        self.mutate()
        return self.value


@pytest.mark.parametrize("operand", ["sub", "sup"])
def test_thomas_rhs_conversion_cannot_invalidate_earlier_vector_metadata(operand):
    sub, sup = np.ones(1), np.ones(1)
    target = sub if operand == "sub" else sup
    value = _MutatingFloat(lambda: setattr(target, "shape", ()))
    with pytest.raises(ValueError, match="argument conversion"):
        kernel("thomas")(sub, [4.0, 4.0], sup, [value, 2.0])


@pytest.mark.parametrize("conversion", ["diag", "sup"])
def test_thomas_batch_conversion_revalidates_earlier_input_before_shape_access(conversion):
    sub, rhs = np.ones(1), np.ones((2, 2))
    original = rhs.copy()
    value = _MutatingFloat(lambda: setattr(sub, "shape", ()))
    diag = [value, 4.0] if conversion == "diag" else [4.0, 4.0]
    sup = [value] if conversion == "sup" else [1.0]
    with pytest.raises(ValueError, match="argument conversion"):
        kernel("thomas_batch")(sub, diag, sup, rhs)
    np.testing.assert_array_equal(rhs, original)


@pytest.mark.parametrize("mutation", ["shape", "readonly"])
def test_sor_source_conversion_revalidates_mutable_grid(mutation):
    u = np.ones((3, 3))
    if mutation == "shape":
        mutate = lambda: setattr(u, "shape", (9,))
    else:
        mutate = lambda: setattr(u.flags, "writeable", False)
    source = [[_MutatingFloat(mutate), 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    with pytest.raises(ValueError, match="argument conversion"):
        kernel("sor_poisson")(u, source, 1.0, 0.25, 1.0)
    np.testing.assert_array_equal(u.ravel(), np.ones(9))
