"""Sparse operations must preserve sparsity, dimensions and duplicate semantics."""

from unittest.mock import patch

import numpy as np
import pytest

from quadrivium.core.exceptions import DimensionError
from quadrivium.linalg.sparse import (
    COOMatrix, CSCMatrix, CSRMatrix, DIAMatrix, bandwidth, diags, from_dense,
    identity_sparse, reverse_cuthill_mckee, sparse_solve, sparsity,
)


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
def test_identity_and_statistics_never_allocate_dense(fmt):
    with patch("quadrivium.linalg.sparse.np.eye", side_effect=AssertionError("dense identity")):
        matrix = identity_sparse(100_000, fmt=fmt)
    with patch.object(matrix, "todense", side_effect=AssertionError("densification")):
        assert bandwidth(matrix) == (0, 0)
        assert sparsity(matrix) == 0.99999
    assert matrix.nnz == 100_000
    np.testing.assert_array_equal(matrix @ np.ones(100_000), np.ones(100_000))


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
def test_wide_todense_allocates_only_the_result(fmt):
    matrix = COOMatrix([0, 1, 1], [199_999, 0, 0], [3., 4., -1.], (2, 200_000))
    if fmt == "csr":
        matrix = matrix.tocsr()
    elif fmt == "csc":
        transposed = matrix.transpose().tocsr()
        matrix = CSCMatrix(transposed.indptr, transposed.indices, transposed.data,
                           matrix.shape)
    with (patch("quadrivium.linalg.sparse.np.eye", side_effect=AssertionError("dense identity")),
          patch.object(matrix, "matvec", side_effect=AssertionError("basis products"))):
        result = matrix.todense()
    assert result.shape == (2, 200_000)
    assert np.count_nonzero(result) == 2
    assert result[0, -1] == result[1, 0] == 3.


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (5, 7)])
@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
def test_empty_storage_and_multiple_rhs(shape, fmt):
    matrix = from_dense(np.zeros(shape), fmt=fmt)
    np.testing.assert_array_equal(matrix.todense(), np.zeros(shape))
    np.testing.assert_array_equal(matrix @ np.ones(shape[1]), np.zeros(shape[0]))
    assert (matrix @ np.empty((shape[1], 0))).shape == (shape[0], 0)
    assert (matrix @ np.ones((shape[1], 2))).shape == (shape[0], 2)
    assert bandwidth(matrix) == (0, 0)
    assert sparsity(matrix) == 1.
    assert matrix.density() == 0.
    if hasattr(matrix, "rmatvec"):
        np.testing.assert_array_equal(matrix.rmatvec(np.ones(shape[0])),
                                      np.zeros(shape[1]))


def test_csr_segments_duplicates_and_transpose_product():
    # Leading, consecutive interior and trailing empty rows exercise reduceat.
    matrix = CSRMatrix([0, 0, 3, 3, 3, 5, 5], [2, 1, 1, 0, 3],
                       [3., 2., -1., 4., 5.], (6, 4))
    dense = np.zeros((6, 4))
    dense[1, 1], dense[1, 2], dense[4, 0], dense[4, 3] = 1., 3., 4., 5.
    rhs = np.arange(24.).reshape(4, 6)[:, ::2]
    np.testing.assert_allclose(matrix @ rhs, dense @ rhs)
    np.testing.assert_allclose(matrix @ rhs[:, 1], dense @ rhs[:, 1])
    np.testing.assert_allclose(matrix.rmatvec(np.arange(6.)), dense.T @ np.arange(6.))
    np.testing.assert_array_equal(matrix.diagonal(), np.diag(dense))


def test_csc_unsorted_duplicate_rows_and_empty_columns():
    matrix = CSCMatrix([0, 0, 3, 3, 5, 5], [2, 0, 2, 1, 0],
                       [4., 3., -2., 2., 7.], (3, 5))
    dense = np.array([[0., 3., 0., 7., 0.], [0., 0., 0., 2., 0.],
                      [0., 2., 0., 0., 0.]])
    np.testing.assert_array_equal(matrix.todense(), dense)
    np.testing.assert_allclose(matrix @ np.arange(5.), dense @ np.arange(5.))
    np.testing.assert_allclose(matrix.rmatvec(np.arange(3.)), dense.T @ np.arange(3.))


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc", "dia"])
def test_statistics_use_summed_duplicates_and_ignore_explicit_zeros(fmt):
    rows, cols = [0, 0, 1, 2, 3, 3, 3], [3, 3, 1, 0, 3, 3, 3]
    values = [9., -9., 2., 0., 1e16, -1e16, 1.]
    coo = COOMatrix(rows, cols, values, (4, 4))
    if fmt == "coo":
        matrix = coo
    elif fmt == "csr":
        matrix = coo.tocsr()
    elif fmt == "csc":
        transposed = coo.transpose().tocsr()
        matrix = CSCMatrix(transposed.indptr, transposed.indices, transposed.data, (4, 4))
    else:
        matrix = DIAMatrix([[0., 0., 0., 9.], [0., 0., 0., -9.],
                            [0., 2., 0., 1.]], [3, 3, 0], (4, 4))
    dense = matrix.todense()
    with patch.object(matrix, "todense", side_effect=AssertionError("densification")):
        assert bandwidth(matrix) == bandwidth(dense) == (0, 0)
        assert sparsity(matrix) == sparsity(dense) == 0.875
        np.testing.assert_array_equal(reverse_cuthill_mckee(matrix),
                                      reverse_cuthill_mckee(dense))


def test_dia_padding_missing_tail_and_out_of_range_offsets():
    matrix = DIAMatrix([[99., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                        [1., 1., 1.]], [1, -1, 10, -10], (4, 5))
    dense = np.zeros((4, 5))
    dense[0, 1], dense[1, 2] = 2., 3.
    dense[1, 0], dense[2, 1], dense[3, 2] = 4., 5., 6.
    np.testing.assert_array_equal(matrix.todense(), dense)
    np.testing.assert_allclose(matrix @ np.arange(5.), dense @ np.arange(5.))
    assert matrix.nnz == 5
    assert bandwidth(matrix) == (1, 1)


def test_reordering_sparse_disconnected_graph_and_zero_size():
    matrix = identity_sparse(10_000)
    with patch.object(matrix, "todense", side_effect=AssertionError("densification")):
        np.testing.assert_array_equal(reverse_cuthill_mckee(matrix),
                                      np.arange(9999, -1, -1))
    assert reverse_cuthill_mckee(identity_sparse(0)).shape == (0,)
    with pytest.raises(DimensionError):
        reverse_cuthill_mckee(COOMatrix([], [], [], (2, 3)))


def test_sparse_reordering_matches_dense_connected_graph():
    rng = np.random.default_rng(12)
    upper = np.triu(rng.random((80, 80)) < .035, 1)
    graph = upper | upper.T
    csr = from_dense(graph)
    np.testing.assert_array_equal(reverse_cuthill_mckee(csr), reverse_cuthill_mckee(graph))
    assert sorted(reverse_cuthill_mckee(csr)) == list(range(80))


@pytest.mark.parametrize("factory", [
    lambda: COOMatrix([-1], [0], [1.], (2, 2)),
    lambda: COOMatrix([0], [2], [1.], (2, 2)),
    lambda: COOMatrix([.5], [0], [1.], (2, 2)),
    lambda: COOMatrix([[0]], [0], [1.], (2, 2)),
    lambda: COOMatrix([0], [0], [[1.]], (2, 2)),
    lambda: COOMatrix([], [], [], (-1, 2)),
    lambda: COOMatrix([], [], [], (2.5, 2)),
    lambda: CSRMatrix([1, 1], [], [], (1, 1)),
    lambda: CSRMatrix([0, 2], [0], [1.], (1, 1)),
    lambda: CSRMatrix([0, 2, 1, 2], [0, 0], [1., 2.], (3, 1)),
    lambda: CSRMatrix([0, 0], [0], [], (1, 1)),
    lambda: CSRMatrix([0, 1], [1], [1.], (1, 1)),
    lambda: CSCMatrix([0], [], [], (2, 1)),
    lambda: CSCMatrix([0, 1], [-1], [1.], (2, 1)),
    lambda: DIAMatrix([[1., 2.]], [0, 1], (2, 2)),
    lambda: diags([[1., 2.]], [0, 1]),
    lambda: from_dense([1., 2.]),
    lambda: from_dense(np.eye(2), fmt="typo"),
    lambda: from_dense(np.eye(2), tol=-1.),
    lambda: identity_sparse(2, fmt="typo"),
])
def test_invalid_structure_rejected_before_indexing(factory):
    with pytest.raises((ValueError, DimensionError)):
        factory()


@pytest.mark.parametrize("matrix", [
    COOMatrix([0], [0], [1.], (2, 3)),
    CSRMatrix([0, 1, 1], [0], [1.], (2, 3)),
    CSCMatrix([0, 1, 1, 1], [0], [1.], (2, 3)),
    diags([[1., 2.]], [0], shape=(2, 3)),
])
def test_all_sparse_products_check_dimensions(matrix):
    with pytest.raises(DimensionError):
        matrix.matvec(np.ones(4))
    with pytest.raises(DimensionError):
        matrix @ np.ones((4, 2))
    with pytest.raises(DimensionError):
        matrix @ np.ones((3, 2, 1))
    if hasattr(matrix, "rmatvec"):
        with pytest.raises(DimensionError):
            matrix.rmatvec(np.ones(3))


def test_csr_conjugate_gradient_residual_without_dense_conversion():
    n = 250
    rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
    values = np.concatenate([np.full(n, 4.), np.full(2 * (n - 1), -1.)])
    matrix = COOMatrix(rows, cols, values, (n, n)).tocsr()
    exact = np.sin(np.arange(n) / 7.)
    rhs = matrix @ exact
    with patch.object(matrix, "todense", side_effect=AssertionError("densification")):
        result = sparse_solve(matrix, rhs, tol=1e-10)
    np.testing.assert_allclose(result.x, exact, atol=1e-9)
    assert np.linalg.norm(matrix @ result.x - rhs) < 1e-8
