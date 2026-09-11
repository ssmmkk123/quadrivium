"""Stable Euclidean norms across exponent limits and strided reduction axes."""

import math

import numpy as reference
import pytest

from quadrivium import numeric as np


def _assert_norm(actual, expected):
    # Zero absolute tolerance matters: returning zero for a tiny, nonzero
    # norm is a numerical failure even though ordinary allclose accepts it.
    assert math.isclose(float(actual), expected, rel_tol=3e-15, abs_tol=0.0)


@pytest.mark.parametrize("scale", [5e-324, 1e-300, 1e-200, 1e-160, 1e-155,
                                   1.0, 1e150, 1e200, 1e300])
@pytest.mark.parametrize("order", [None, 2])
def test_real_euclidean_norm_preserves_finite_tiny_and_huge_values(scale, order):
    values = [3 * scale, 4 * scale]
    expected = math.hypot(*values)
    _assert_norm(np.linalg.norm(np.array(values), ord=order), expected)


@pytest.mark.parametrize("scale", [5e-324, 1e-300, 1e-200, 1e-160, 1.0, 1e200, 1e300])
def test_complex_euclidean_norm_scales_real_and_imaginary_components(scale):
    values = np.array([complex(3 * scale, 4 * scale), 0j])
    expected = math.hypot(3 * scale, 4 * scale)
    _assert_norm(np.linalg.norm(values), expected)
    _assert_norm(np.linalg.norm(values, 2), expected)


@pytest.mark.parametrize("scale", [1e-300, 1e-160, 1.0, 1e200, 1e300])
@pytest.mark.parametrize("order", [None, "fro"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_frobenius_norm_uses_stable_euclidean_reduction(scale, order, complex_values):
    values = reference.array([[3.0, 4.0], [0.0, 12.0]]) * scale
    if complex_values:
        values = values + 1j * values
    components = [component for z in values.ravel()
                  for component in (float(z.real), float(z.imag))]
    _assert_norm(np.linalg.norm(np.asarray(values), ord=order), math.hypot(*components))


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
def test_axis_norms_preserve_shape_and_strided_extreme_values(axis, keepdims, complex_values):
    base = reference.array([[3e-300, 4e-300, 12e-300],
                            [3e-160, 4e-160, 12e-160],
                            [3e200, 4e200, 12e200],
                            [3e300, 4e300, 12e300]])
    if complex_values:
        base = base + 1j * base
    values = np.asarray(base)[::-1, ::-1]
    values.flags.writeable = False
    before = values.copy()
    raw = reference.asarray(values)
    real_axis = axis % 2
    slices = raw.T if real_axis == 0 else raw
    expected = reference.array([math.hypot(*[part for z in row
                                             for part in (float(z.real), float(z.imag))])
                                for row in slices])
    if keepdims:
        expected = reference.expand_dims(expected, axis=real_axis)
    result = np.linalg.norm(values, axis=axis, keepdims=keepdims)
    assert result.shape == expected.shape
    for actual, wanted in zip(result.ravel(), expected.ravel()):
        _assert_norm(actual, float(wanted))
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("complex_values", [False, True])
def test_norm_handles_many_subnormal_squared_terms(complex_values):
    values = np.full(4096, 1e-160)
    if complex_values:
        values = values + 1j * values
    expected = math.hypot(*([1e-160] * (8192 if complex_values else 4096)))
    _assert_norm(np.linalg.norm(values), expected)


@pytest.mark.parametrize("values,expected", [
    ([0.0, -0.0], 0.0), ([reference.inf, 1.0], reference.inf),
    ([-reference.inf, 1.0], reference.inf),
    ([reference.nan, 1.0], reference.nan),
    ([reference.inf, reference.nan], reference.nan),
    ([reference.nan, -reference.inf], reference.nan),
    ([1.7e308, 1.7e308], reference.inf),
    ([complex(reference.inf, 0.0), 1j], reference.inf),
    ([complex(reference.inf, reference.nan), 1j], reference.nan),
    ([complex(reference.nan, reference.inf), 1j], reference.nan),
])
def test_euclidean_norm_preserves_nonfinite_and_true_overflow_behavior(values, expected):
    result = np.linalg.norm(np.array(values))
    if math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected
        if expected == 0:
            assert math.copysign(1.0, result) == 1.0


@pytest.mark.parametrize("shape,axis", [((3, 0), 1), ((0, 3), 0),
                                       ((2, 0, 4), 1), ((0, 3), 1)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", [float, complex])
def test_empty_norm_axes_return_initialized_zeros(shape, axis, keepdims, dtype):
    result = np.linalg.norm(np.empty(shape, dtype=dtype), axis=axis, keepdims=keepdims)
    expected = reference.linalg.norm(reference.empty(shape, dtype=dtype),
                                     axis=axis, keepdims=keepdims)
    assert result.shape == expected.shape
    reference.testing.assert_array_equal(result, expected)
    assert np.linalg.norm(np.empty(0, dtype=dtype)) == 0.0
