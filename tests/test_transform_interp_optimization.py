"""Accuracy, tie conventions and workspace checks for batched interpolation."""

import tracemalloc

import numpy as reference
import pytest

from quadrivium import numeric as np
from quadrivium.interpolate import multivariate as interp


def _grid_reference(grids, values, queries, nearest=False):
    out = []
    for pt in queries:
        if nearest:
            index = tuple(reference.argmin(reference.abs(g - pt[d]))
                          for d, g in enumerate(grids))
            out.append(values[index])
            continue
        lo = [int(reference.clip(reference.searchsorted(g, pt[d]) - 1,
                                 0, g.size - 2))
              for d, g in enumerate(grids)]
        t = [(pt[d] - g[lo[d]]) / (g[lo[d] + 1] - g[lo[d]])
             for d, g in enumerate(grids)]
        total = 0.0
        for corner in range(2 ** len(grids)):
            weight = 1.0
            indices = []
            for d in range(len(grids)):
                bit = (corner >> d) & 1
                weight *= t[d] if bit else 1 - t[d]
                indices.append(lo[d] + bit)
            total += weight * values[tuple(indices)]
        out.append(total)
    return reference.array(out)


@pytest.mark.parametrize("ndim", [1, 2, 3, 5])
@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_grid_batches_match_corner_definition_and_preserve_inputs(ndim, method):
    rng = reference.random.default_rng(812)
    grids = [reference.array([-2.0, -0.4, 0.1, 2.0]) for _ in range(ndim)]
    values = rng.normal(size=(4,) * ndim)
    source = rng.uniform(-3, 3, size=(82, ndim * 2))
    queries = source[::2, ::2]
    queries[0] = -0.4
    queries[1] = 2.0
    before = queries.copy()
    queries.flags.writeable = False
    f = interp.regular_grid_interpolator(grids, values, method)
    expected = _grid_reference(grids, values, queries, method == "nearest")
    reference.testing.assert_allclose(f(queries), expected, rtol=2e-13, atol=2e-13)
    reference.testing.assert_allclose(f(queries[0]), expected[0], atol=2e-13)
    reference.testing.assert_array_equal(queries, before)
    assert f(reference.empty((0, ndim))).shape == (0,)


@pytest.mark.parametrize("grid", [
    [-2.0, 0.0, 2.0], [2.0, -2.0, 0.0], [-2.0, -2.0, 0.0, 2.0],
    [3.0], [-1.7e308, -1.6e308], [-1.0, reference.nan, 2.0],
    [-reference.inf, 0.0, reference.inf],
])
def test_nearest_grid_keeps_first_ties_and_nonfinite_argmin_convention(grid):
    grid = reference.array(grid)
    queries = reference.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [3.0],
                               [1.7e308], [reference.nan], [reference.inf]])
    values = reference.arange(grid.size, dtype=float)
    with reference.errstate(over="ignore", invalid="ignore"):
        expected = _grid_reference([grid], values, queries, nearest=True)
    result = interp.regular_grid_interpolator([grid], values, "nearest")(queries)
    reference.testing.assert_array_equal(result, expected)


def test_nearest_grid_observes_grid_mutations_between_evaluations():
    grid = np.array([-2.0, 0.0, 2.0])
    values = np.array([5.0, 6.0, 7.0])
    f = interp.regular_grid_interpolator([grid], values, "nearest")
    assert f([1.75]) == 7.0
    grid[:] = [2.0, -2.0, 0.0]
    assert f([1.75]) == 5.0


def test_trilinear_broadcasts_and_extrapolates_multiaffine_fields():
    g = reference.array([-2.0, 0.0, 1.0, 3.0])
    x, y, z = reference.meshgrid(g, g, g, indexing="ij")
    field = lambda a, b, c: 2 + a - 3 * b + a * b - b * c + 0.5 * a * b * c
    f = interp.trilinear(g, g, g, field(x, y, z))
    a = reference.array([-3.0, 0.3, 2.0])[:, None]
    b = reference.array([-1.2, 0.0, 1.0, 3.5])[None, :]
    result = f(a, b, 0.7)
    assert result.shape == (3, 4)
    reference.testing.assert_allclose(result, field(a, b, 0.7), atol=2e-14)
    assert isinstance(f(0.0, 1.0, 2.0), float)
    assert f(reference.empty(0), 1.0, 2.0).shape == (0,)


def test_grid_large_query_workspace_is_bounded_and_crosses_blocks():
    g = np.array([-1.0, 0.0, 1.0])
    values = g[:, None, None] + 2 * g[None, :, None] - g[None, None, :]
    q = np.linspace(-1.2, 1.2, 140000)
    queries = np.column_stack([q, 0.5 * q, -q])
    f = interp.regular_grid_interpolator([g, g, g], values)
    tracemalloc.start()
    try:
        result = f(queries)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16_000_000
    np.testing.assert_allclose(result, 3 * q, rtol=1e-13, atol=1e-13)


def test_scattered_nearest_keeps_ties_and_accurate_distances_at_large_offsets():
    points = reference.array([[1e12, 1e12], [1e12 + 0.01, 1e12],
                              [1e12, 1e12 + 0.01], [1e12, 1e12]])
    values = reference.array([3.0, 7.0, 11.0, 100.0])
    query = reference.array([[1e12, 1e12], [1e12 + 0.009, 1e12],
                             [1e12, 1e12 + 0.009]])
    f = interp.nearest_neighbor(points, values)
    reference.testing.assert_array_equal(f(query), [3.0, 7.0, 11.0])
    assert f(query[0]) == 3.0
    assert f(reference.empty((0, 2))).shape == (0,)
    tied = interp.nearest_neighbor([[-1.0, 0.0], [1.0, 0.0]], [4.0, 9.0])
    assert tied([0.0, 0.0]) == 4.0


def test_scattered_nearest_bounds_distance_workspace():
    rng = np.random.default_rng(821)
    points = rng.normal(size=(1000, 3))
    queries = rng.normal(size=(2000, 3))
    values = np.arange(1000, dtype=float)
    f = interp.nearest_neighbor(points, values)
    tracemalloc.start()
    try:
        result = f(queries)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 5_000_000  # A full coordinate tensor previously used 64 MB.
    for i in (0, 86, 87, 1000, 1999):
        expected = np.argmin(np.linalg.norm(points - queries[i], axis=1))
        assert result[i] == values[expected]


@pytest.mark.parametrize("power,tol", [(2.0, 1e-12), (0.0, 1e-12),
                                      (1.5, 0.01), (2.0, 0.0)])
def test_idw_batches_preserve_exact_hits_duplicates_and_power(power, tol):
    rng = reference.random.default_rng(412)
    points = rng.normal(size=(91, 3))
    points[1] = points[0]
    values = rng.normal(size=91)
    queries = rng.normal(size=(1031, 3))
    queries[0] = points[0]
    queries[1000] = points[10] + 0.001
    expected = []
    with reference.errstate(divide="ignore", invalid="ignore"):
        for query in queries:
            d = reference.linalg.norm(points - query, axis=1)
            hits = reference.flatnonzero(d < tol)
            if hits.size:
                expected.append(values[hits[0]])
            else:
                w = 1.0 / d ** power
                expected.append(w @ values / w.sum())
    f = interp.inverse_distance_weighting(points, values, power=power, tol=tol)
    reference.testing.assert_allclose(f(queries), expected, rtol=2e-13,
                                      atol=2e-13, equal_nan=True)
    assert f(reference.empty((0, 3))).shape == (0,)


@pytest.mark.parametrize("kernel", ["gaussian", "multiquadric", "thin_plate"])
def test_rbf_batches_match_weights_at_nodes_and_large_offset_queries(kernel):
    rng = np.random.default_rng(981)
    points = 1e9 + rng.normal(size=(64, 3))
    values = rng.normal(size=64)
    queries = 1e9 + rng.normal(size=(1701, 3))
    queries[:64] = points
    f = interp.rbf_interpolation(points, values, kernel=kernel, smooth=0.01)
    r = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2)
    expected = interp._RBF_KERNELS[kernel](r, 1.0) @ f.weights
    np.testing.assert_allclose(f(queries), expected, rtol=2e-12, atol=2e-12)
    assert f(np.empty((0, 3))).shape == (0,)


def test_rbf_evaluation_bounds_distance_workspace():
    rng = np.random.default_rng(931)
    points = rng.normal(size=(96, 3))
    f = interp.rbf_interpolation(points, rng.normal(size=96), kernel="gaussian")
    queries = rng.normal(size=(10000, 3))
    tracemalloc.start()
    try:
        result = f(queries)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.shape == (10000,)
    assert peak < 5_000_000


def _kriging_reference(points, values, queries, model):
    def gamma(d):
        if model == "spherical":
            out = np.where(d < 1, 1.5 * d - 0.5 * d ** 3, 1.0)
        elif model == "exponential":
            out = 1.0 - np.exp(-3 * d)
        else:
            out = 1.0 - np.exp(-3 * d ** 2)
        return np.where(d == 0, 0.0, out)
    n = points.shape[0]
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = gamma(np.linalg.norm(points[:, None] - points[None, :], axis=2))
    A[n, n] = 0.0
    estimates, variances = [], []
    for q in queries:
        g = gamma(np.linalg.norm(points - q, axis=1))
        b = np.append(g, 1.0)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        estimates.append(w[:n] @ values)
        variances.append(w[:n] @ g + w[n])
    return np.array(estimates), np.array(variances)


@pytest.mark.parametrize("model", ["spherical", "exponential", "gaussian"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_kriging_multiple_rhs_matches_pointwise_solves_and_singular_fallback(model, duplicate):
    rng = np.random.default_rng(47)
    points = rng.normal(size=(12, 2))
    values = rng.normal(size=12)
    if duplicate:
        points[1] = points[0]
    queries = rng.normal(size=(19, 2))
    queries[:12] = points
    expected = _kriging_reference(points, values, queries, model)
    f = interp.kriging(points, values, model=model)
    for actual, wanted in zip(f(queries), expected):
        np.testing.assert_allclose(actual, wanted, rtol=2e-10, atol=2e-10)
    scalar = f(queries[0])
    assert all(isinstance(v, float) for v in scalar)
    assert all(v.shape == (0,) for v in f(np.empty((0, 2))))


def test_kriging_reuses_one_factorization_for_a_query_batch(monkeypatch):
    rng = np.random.default_rng(39)
    points = rng.normal(size=(32, 2))
    values = rng.normal(size=32)
    queries = rng.normal(size=(127, 2))
    f = interp.kriging(points, values)
    solve = np.linalg.solve
    calls = []
    def counted_solve(a, b):
        calls.append(b.shape)
        return solve(a, b)
    monkeypatch.setattr(np.linalg, "solve", counted_solve)
    estimate, variance = f(queries)
    assert calls == [(33, 127)]
    assert np.all(np.isfinite(estimate))
    assert np.all(variance >= -1e-12)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 17, 64])
@pytest.mark.parametrize("fallback", [False, True])
def test_half_length_dct2_matches_cosine_definition(n, fallback):
    from contextlib import nullcontext
    from quadrivium import _accel
    from quadrivium.transforms import dct, idct
    source = np.random.default_rng(931).normal(size=2 * n)
    x = source[::2]
    before = x.copy()
    x.flags.writeable = False
    if n:
        k = reference.arange(n)[:, None]
        j = reference.arange(n)[None, :]
        expected = 2 * reference.cos(reference.pi * k * (2 * j + 1) / (2 * n)) @ reference.asarray(x)
    else:
        expected = reference.empty(0)
    with _accel.disabled() if fallback else nullcontext():
        result = dct(x, kind=2)
        reference.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
        if n:
            np.testing.assert_allclose(idct(result, kind=2), x, atol=2e-12)
            reference.testing.assert_allclose(dct(x, kind=2, norm=True),
                                              expected / reference.sqrt(2 * n), atol=2e-12)
    np.testing.assert_array_equal(x, before)
    assert result.flags.owndata
    assert not np.shares_memory(result, source)


def test_dct2_long_transform_preserves_energy_and_reduces_workspace():
    from quadrivium.transforms import dct, idct
    n = 65536
    x = np.random.default_rng(193).normal(size=n)
    tracemalloc.start()
    try:
        coefficients = dct(x, kind=2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The old 4n extension required over 10 MB with the compiled FFT and
    # over 14 MB with the Python FFT; the 2n extension fits below 8 MB on both.
    assert peak < 8_000_000
    energy = coefficients[0] ** 2 / (4 * n) + np.sum(coefficients[1:] ** 2) / (2 * n)
    np.testing.assert_allclose(energy, x @ x, rtol=2e-12)
    np.testing.assert_allclose(idct(coefficients, kind=2), x, atol=2e-11)
