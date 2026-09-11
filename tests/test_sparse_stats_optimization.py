"""Accuracy and bounded-workspace regressions for sparse/statistical summaries."""

import tracemalloc
from unittest.mock import patch

import pytest

from quadrivium import numeric as np
from quadrivium.linalg import sparse
from quadrivium.stochastic import stats


def _diagonal_reference(matrix):
    """Pairwise summation of each row's diagonal entries, in storage order."""
    result = np.zeros(min(matrix.shape))
    for row in range(result.size):
        start, stop = matrix.indptr[row:row + 2]
        result[row] = np.sum(matrix.data[start:stop][matrix.indices[start:stop] == row])
    return result


def _strided(values):
    result = np.empty(values.size * 2, dtype=values.dtype)
    result[::2] = values
    return result[::2]


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (5, 0), (13, 7), (7, 13)])
@pytest.mark.parametrize("strided", [False, True])
def test_tiled_diagonal_unsorted_duplicates_empty_rows_and_rectangles(shape, strided):
    rng = np.random.default_rng(456)
    counts = rng.integers(0, 17, size=shape[0]) if shape[1] else np.zeros(shape[0], dtype=int)
    if counts.size:
        counts[::3] = 0
    indptr = np.concatenate([np.array([0], dtype=np.intp), np.cumsum(counts)])
    count = int(indptr[-1])
    indices = rng.integers(0, shape[1], size=count) if count else np.empty(0, dtype=np.intp)
    data = rng.normal(size=count)
    if strided:
        indptr, indices, data = map(_strided, (indptr, indices, data))
    matrix = sparse.CSRMatrix(indptr, indices, data, shape)
    expected = _diagonal_reference(matrix)
    saved = data.copy()
    with (patch.object(sparse, "_DIAGONAL_TILE_ROWS", 3),
          patch.object(sparse, "_DIAGONAL_TILE_ELEMENTS", 7)):
        np.testing.assert_array_equal(matrix.diagonal(), expected)
    np.testing.assert_array_equal(data, saved)


@pytest.mark.parametrize("tile_elements", [7, 1024])
def test_diagonal_preserves_pairwise_cancellation_and_signed_zero(tile_elements):
    values = np.array([1e16, 1., -1e16, 1., 1e16, 1., -1e16, 1.,
                       -0., 3., 1e16, -1e16, 4.])
    matrix = sparse.CSRMatrix([0, 8, 9, 9, 13],
                              [0] * 8 + [1] + [3] * 4, values, (4, 4))
    expected = _diagonal_reference(matrix)
    with patch.object(sparse, "_DIAGONAL_TILE_ELEMENTS", tile_elements):
        actual = matrix.diagonal()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    # No cached structural assumption may survive mutations of public arrays.
    matrix.indices[0] = 2
    matrix.data[-1] = 6.
    np.testing.assert_array_equal(matrix.diagonal(), _diagonal_reference(matrix))


def test_diagonal_preserves_nonfinite_entries_and_ignores_off_diagonal_nan():
    matrix = sparse.CSRMatrix([0, 2, 4, 6, 7], [0, 3, 1, 1, 2, 2, 0],
                              [np.inf, np.nan, np.inf, -np.inf, np.nan, 1., np.nan],
                              (4, 4))
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(matrix.diagonal(), _diagonal_reference(matrix))


def test_diagonal_workspace_does_not_scale_with_stored_entries():
    n, width = 80_000, 20
    indptr = np.arange(0, n * width + 1, width, dtype=np.intp)
    # Each row has exactly one diagonal entry, including wraparound rows.
    indices = (np.arange(n)[:, None] + np.arange(width)).ravel() % n
    matrix = sparse.CSRMatrix(indptr, indices, np.ones(n * width), (n, n))
    tracemalloc.start()
    try:
        result = matrix.diagonal()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(result, np.ones(n))
    # A full nnz-sized integer expansion alone would require 12.8 MB.
    assert peak < result.nbytes + 8 * 1024 * 1024


@pytest.mark.parametrize("sample", [
    [3.], [2., 2., 2., 2.], [1e12 + i / 8 for i in range(37)],
    [-1e8, -1., 0., 0., 1., 1e8],
    [np.nan, 1., 2.], [-np.inf, 0., np.inf],
])
def test_describe_matches_individual_statistics(sample):
    sample = _strided(np.array(sample, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        actual = stats.describe(sample)
        expected = {
            "n": sample.size,
            "mean": stats.mean(sample),
            "variance": stats.variance(sample),
            "std": float(np.std(sample, ddof=1)) if sample.size > 1 else 0.,
            "min": float(sample.min()),
            "max": float(sample.max()),
            "q1": float(stats.quantile(sample, .25)),
            "median": stats.median(sample),
            "q3": float(stats.quantile(sample, .75)),
            "skewness": stats.skewness(sample),
            "kurtosis": stats.kurtosis(sample),
        }
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal([actual[name]], [expected[name]])


def test_describe_preserves_input_and_rejects_empty_samples():
    sample = np.random.default_rng(123).normal(size=101)[::-2]
    saved = sample.copy()
    stats.describe(sample)
    np.testing.assert_array_equal(sample, saved)
    with pytest.raises(ValueError):
        stats.describe([])
