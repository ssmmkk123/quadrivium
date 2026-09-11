"""Independent FFT references and concurrent twiddle-cache regression tests."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as reference
import pytest

from quadrivium import _accel, numeric as np


def transform(api, inverse):
    name = "ifft" if inverse else "fft"
    if api == "native":
        function = _accel.kernel(name)
        if function is None:
            pytest.skip("native acceleration backend unavailable")
        return function
    return getattr(np.fft, name)


@pytest.mark.parametrize("api", ["core", "native"])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("n", [1, 2, 3, 5, 32, 45, 97, 257, 1000, 1024, 10000, 30030])
def test_native_fft_matches_numpy_for_negative_stride_readonly_input(api, inverse, n):
    rng = reference.random.default_rng(600 + n)
    raw = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
    source = np.array(raw)[::-2]
    source.flags.writeable = False
    original = reference.asarray(source).copy()
    actual = transform(api, inverse)(source)
    expected = (reference.fft.ifft if inverse else reference.fft.fft)(original)
    reference.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    reference.testing.assert_array_equal(source, original)
    assert actual.flags.owndata
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("inverse", [False, True])
def test_fft_strided_multi_axis_traversal_matches_numpy(axis, inverse):
    rng = reference.random.default_rng(71)
    raw = rng.standard_normal((12, 18, 20)) + 1j * rng.standard_normal((12, 18, 20))
    source = np.array(raw)[::-2, ::3, ::-2]
    source.flags.writeable = False
    expected = (reference.fft.ifft if inverse else reference.fft.fft)(
        reference.asarray(source), axis=axis
    )
    actual = transform("core", inverse)(source, axis=axis)
    reference.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("api", ["core", "native"])
def test_parallel_transforms_pin_cache_entries_during_eviction(api):
    # More simultaneous lengths than cache slots; both signs compete for the
    # same cache while power-of-two and Bluestein transforms release the GIL.
    lengths = [64, 81, 97, 128, 243, 257, 1024, 2048, 4096, 8192, 16384, 65536]
    start = Barrier(len(lengths))
    forward, inverse = transform(api, False), transform(api, True)

    def run(n):
        rng = reference.random.default_rng(n)
        raw = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        source = np.array(raw)
        expected_forward = reference.fft.fft(raw)
        expected_inverse = reference.fft.ifft(raw)
        start.wait(timeout=20)
        for _ in range(4):
            reference.testing.assert_allclose(
                forward(source), expected_forward, rtol=1e-10, atol=1e-10
            )
            reference.testing.assert_allclose(
                inverse(source), expected_inverse, rtol=1e-10, atol=1e-10
            )

    with ThreadPoolExecutor(max_workers=len(lengths)) as pool:
        list(pool.map(run, lengths))


def test_fft_larger_than_cache_slot_limit_uses_transient_twiddles():
    rng = reference.random.default_rng(442)
    raw = rng.standard_normal(131072) + 1j * rng.standard_normal(131072)
    source = np.array(raw)
    actual = transform("native", False)(source)
    reference.testing.assert_allclose(actual, reference.fft.fft(raw), rtol=1e-10, atol=1e-9)
