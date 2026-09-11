"""Regression coverage for allocation-free native segmented reductions."""

import sys

import pytest

from quadrivium import numeric as qp

np = pytest.importorskip("numpy")


def _native_layouts(values):
    source = qp.array(values)
    spread = qp.empty(tuple(dim * 2 for dim in source.shape), dtype=source.dtype)
    spread[::2, ::2, ::2] = source
    readonly = source.copy()
    readonly.flags.writeable = False
    return (source, qp.asfortranarray(source), source.transpose(2, 0, 1),
            source[::-1, :, ::-1], spread[::2, ::2, ::2], readonly)


@pytest.mark.parametrize("operation", ["add", "multiply", "minimum", "maximum"])
@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_segmented_reductions_match_numpy_across_dtypes_axes_and_layouts(operation, dtype):
    values = (np.arange(120).reshape(5, 4, 6) % 5 - 2).astype(dtype)
    if dtype is complex:
        values += 1j * (np.arange(120).reshape(5, 4, 6) % 3 - 1)
    for source in _native_layouts(values):
        expected_source = np.asarray(memoryview(source)).copy()
        for axis in (0, 1, 2, -1):
            length = source.shape[axis]
            # Unsorted and repeated starts represent a single element, while
            # the final segment always extends to the end of the axis.
            starts = [length - 1, 0, 0, length - 2, 1]
            actual = getattr(qp, operation).reduceat(source, starts, axis=axis)
            expected = getattr(np, operation).reduceat(expected_source, starts, axis=axis)
            assert actual.shape == expected.shape
            assert str(actual.dtype) == str(expected.dtype)
            np.testing.assert_array_equal(np.asarray(memoryview(actual)), expected)
            np.testing.assert_array_equal(np.asarray(memoryview(source)), expected_source)


def test_segmented_sum_retains_pairwise_rounding_for_long_strided_runs():
    source = qp.array(([1e16, 1., -1e16, 3.] * 200)).reshape(400, 2)
    starts = [0, 129, 260, 399, 10]
    for values in (source, source[::-1], qp.asfortranarray(source)):
        result = qp.add.reduceat(values, starts, axis=0)
        for segment, start in enumerate(starts):
            end = starts[segment + 1] if segment + 1 < len(starts) else values.shape[0]
            end = max(start + 1, end)
            qp.testing.assert_array_equal(result[segment], qp.sum(values[start:end], axis=0))


@pytest.mark.parametrize("operation", ["add", "multiply", "minimum", "maximum"])
def test_negative_segment_indices_are_consistent_between_fast_and_strided_paths(operation):
    # Quadrivium historically accepts negative starts. Preserve that contract
    # and normalize the next start too, including on the contiguous sum path.
    values = qp.arange(1., 11.)
    spread = qp.zeros(20)
    spread[::2] = values
    starts = [0, -3, -3, -8, -1]
    expected = getattr(np, operation).reduceat(np.arange(1., 11.), [0, 7, 7, 2, 9])
    for source in (values, spread[::2], values[::-1].copy()[::-1]):
        result = getattr(qp, operation).reduceat(source, starts)
        np.testing.assert_array_equal(np.asarray(memoryview(result)), expected)


def test_segment_indices_may_be_strided_and_read_only():
    values = qp.arange(36.).reshape(12, 3)
    storage = qp.array([0, 99, 5, 99, 5, 99, 9, 99])
    starts = storage[::2]
    starts.flags.writeable = False
    before = storage.copy()
    result = qp.add.reduceat(values, starts, axis=0)
    expected = np.add.reduceat(np.arange(36.).reshape(12, 3), [0, 5, 5, 9], axis=0)
    np.testing.assert_array_equal(np.asarray(memoryview(result)), expected)
    qp.testing.assert_array_equal(storage, before)


@pytest.mark.parametrize("operation", ["add", "multiply", "minimum", "maximum"])
def test_segmented_reductions_support_broadcast_zero_strides_and_empty_shapes(operation):
    values = qp.broadcast_to(qp.array([2., -1., 3.]), (8, 3))
    result = getattr(qp, operation).reduceat(values, [0, 3, 3, 6], axis=0)
    expected = getattr(np, operation).reduceat(np.broadcast_to([2., -1., 3.], (8, 3)),
                                              [0, 3, 3, 6], axis=0)
    np.testing.assert_array_equal(np.asarray(memoryview(result)), expected)
    for shape, axis, starts, expected_shape in (
        ((0, 3), 0, [], (0, 3)),
        ((0, 3), 1, [0, 2], (0, 2)),
        ((2, 0, 3), 0, [0, 1], (2, 0, 3)),
        ((2, 3), 1, [], (2, 0)),
    ):
        assert getattr(qp, operation).reduceat(qp.empty(shape), starts, axis=axis).shape == expected_shape


@pytest.mark.parametrize("operation", ["add", "multiply", "minimum", "maximum"])
def test_segmented_reductions_reject_invalid_indices_even_without_output_elements(operation):
    reducer = getattr(qp, operation).reduceat
    for values, axis in ((qp.arange(5.), 0), (qp.arange(10.)[::2], 0),
                         (qp.zeros((0, 5)), 1)):
        for starts in ([5], [-6], [0, 5], [0, -6]):
            with pytest.raises(IndexError):
                reducer(values, starts, axis=axis)
    with pytest.raises(IndexError):
        reducer(qp.empty((0, 2)), [0], axis=0)
    with pytest.raises(ValueError):
        reducer(qp.array(2.), [0])
    for axis in (-3, 2):
        with pytest.raises(ValueError):
            reducer(qp.ones((2, 2)), [0], axis=axis)


@pytest.mark.parametrize("operation", ["add", "multiply"])
@pytest.mark.parametrize("dtype", [bool, int, float, complex])
def test_ufunc_accumulation_preserves_dtypes_axes_and_layouts(operation, dtype):
    values = (np.arange(120).reshape(5, 4, 6) % 5 - 2).astype(dtype)
    if dtype is complex:
        values += .5j
    for source in _native_layouts(values):
        expected_source = np.asarray(memoryview(source)).copy()
        # The wrapper historically normalizes axes modulo ndim, and bool
        # accumulations retain bool rather than promoting to int64.
        for axis in (0, 1, 2, -1, -5, 5):
            result = getattr(qp, operation).accumulate(source, axis=axis)
            expected = getattr(np, operation).accumulate(expected_source,
                                                         axis=axis % source.ndim,
                                                         dtype=expected_source.dtype)
            assert str(result.dtype) == str(expected.dtype)
            np.testing.assert_array_equal(np.asarray(memoryview(result)), expected)
            np.testing.assert_array_equal(np.asarray(memoryview(source)), expected_source)


@pytest.mark.parametrize("operation, cumulative", [("add", "cumsum"), ("multiply", "cumprod")])
def test_cumulative_first_element_keeps_signed_zero_and_complex_infinities(operation, cumulative):
    for raw in ([-0., -0., 2.], [complex(-0., -0.), complex(2., 1.)],
                [complex(float("inf"), 2.)], [complex(2., float("inf"))],
                [complex(float("inf"), float("inf"))]):
        source = qp.array(raw)
        with np.errstate(invalid="ignore"):
            expected = getattr(np, operation).accumulate(np.array(raw))
        for result in (getattr(qp, operation).accumulate(source), getattr(qp, cumulative)(source)):
            actual = np.asarray(memoryview(result))
            np.testing.assert_array_equal(actual, expected)
            # Equality does not distinguish +0 from -0.
            np.testing.assert_array_equal(np.signbit(actual.real), np.signbit(expected.real))
            if np.iscomplexobj(expected):
                np.testing.assert_array_equal(np.signbit(actual.imag), np.signbit(expected.imag))


@pytest.mark.parametrize("operation, cumulative", [("add", "cumsum"), ("multiply", "cumprod")])
def test_integer_cumulative_overflow_wraps(operation, cumulative):
    raw = np.array([2**63 - 1, 2, -3, -2**63, -1, 17], dtype=np.int64)
    expected = getattr(np, operation).accumulate(raw)
    values = qp.array(raw)
    for result in (getattr(qp, operation).accumulate(values), getattr(qp, cumulative)(values)):
        np.testing.assert_array_equal(np.asarray(memoryview(result)), expected)


@pytest.mark.parametrize("operation", ["add", "multiply"])
def test_accumulation_handles_overlapping_strided_output_and_readonly_inputs(operation):
    values = qp.arange(1., 25.).reshape(4, 6)
    original = values.copy()
    out = values[:, ::-1]
    expected = getattr(np, operation).accumulate(np.asarray(memoryview(original)), axis=0)
    result = getattr(qp, operation).accumulate(values, out=out)
    assert result is out
    np.testing.assert_array_equal(np.asarray(memoryview(out)), expected)

    original.flags.writeable = False
    detached = getattr(qp, operation).accumulate(original)
    np.testing.assert_array_equal(np.asarray(memoryview(detached)), expected)
    with pytest.raises(ValueError):
        getattr(qp, operation).accumulate(original, out=original)


@pytest.mark.parametrize("operation", ["add", "multiply"])
def test_accumulation_retains_empty_shapes_and_scalar_validation(operation):
    for shape in ((0,), (0, 3), (3, 0), (2, 0, 3)):
        for axis in range(len(shape)):
            assert getattr(qp, operation).accumulate(qp.empty(shape), axis=axis).shape == shape
    with pytest.raises(ValueError):
        getattr(qp, operation).accumulate(qp.array(2.))
    with pytest.raises(TypeError):
        getattr(qp, operation).accumulate(qp.arange(3.), axis=None)


def test_empty_segment_index_array_does_not_multiply_huge_unreduced_dimensions():
    # Zero segments imply no work, even when removing the empty reduction axis
    # would leave a dimension product larger than a platform index can hold.
    huge = 1 << (sys.maxsize.bit_length() // 2 + 1)
    shape = (0, huge, huge)
    values = qp.empty(shape)
    assert qp.add.reduceat(values, [], axis=0).shape == shape
