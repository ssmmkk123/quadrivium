"""Numerical and allocation regressions for transform hot paths.

Timing measurements belong in benchmarks; these tests verify the independent
transform definitions, boundary conventions and bounded workspace instead.
"""

import tracemalloc

import numpy as np
import pytest

from quadrivium import _accel
from quadrivium.transforms import fourier, signal, wavelet


@pytest.mark.parametrize("n", [0, 1, 2, 8, 1024, 8192])
@pytest.mark.parametrize("inverse", [False, True])
def test_vectorized_radix2_handles_complex_strided_readonly_input(n, inverse):
    rng = np.random.default_rng(73)
    source = rng.normal(size=2 * n) + 1j * rng.normal(size=2 * n)
    x = source[::2]
    before = x.copy()
    x.flags.writeable = False
    result = fourier.fft_radix2(x, inverse=inverse)
    reference = (np.fft.ifft(x) if inverse else np.fft.fft(x)) if n else x
    np.testing.assert_allclose(result, reference, rtol=1e-11, atol=1e-10)
    np.testing.assert_array_equal(x, before)
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize("n", [6, 12, 20, 45, 105, 202, 1536, 2560])
def test_mixed_radix_forward_and_inverse_use_consistent_signs(n):
    rng = np.random.default_rng(91)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    np.testing.assert_allclose(fourier.fft_mixed_radix(x), np.fft.fft(x), atol=1e-9)
    np.testing.assert_allclose(fourier.fft_mixed_radix(x, inverse=True),
                               np.fft.ifft(x), atol=1e-10)


def test_real_fft_does_not_retain_redundant_spectrum():
    x = np.arange(1024.0)
    with _accel.disabled():
        result = fourier.rfft(x)
    assert result.flags.owndata
    np.testing.assert_allclose(result, np.fft.rfft(x), atol=1e-9)


@pytest.mark.parametrize("n,m", [(64, 3), (3, 64), (6, 4), (1, 13)])
@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_short_kernel_convolution_preserves_mode_alignment(n, m, mode):
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=n), rng.normal(size=m)
    expected = np.convolve(a, b)
    if mode == "same":
        # Quadrivium's existing same convention returns len(a), even if b
        # is longer (NumPy instead returns max(len(a), len(b))).
        expected = expected[(m - 1) // 2 : (m - 1) // 2 + n]
    elif mode == "valid":
        expected = expected[min(n, m) - 1 : max(n, m)]
    np.testing.assert_allclose(signal.convolve(a, b, mode), expected, atol=1e-12)
    np.testing.assert_allclose(signal.convolve_fft(a, b, mode), expected, atol=1e-11)


@pytest.mark.parametrize("function", [signal.convolve, signal.convolve_fft])
@pytest.mark.parametrize("a,b", [([], [1.0]), ([1.0], []), ([], [])])
def test_convolution_rejects_empty_signals(function, a, b):
    with pytest.raises(ValueError, match="nonempty"):
        function(a, b)


def test_direct_convolution_bounds_workspace_for_long_signals():
    x = np.ones(262144)
    kernel = np.ones(8)
    tracemalloc.start()
    try:
        result = signal.convolve(x, kernel)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < result.nbytes + 150_000
    np.testing.assert_array_equal(result[:7], np.arange(1, 8))
    np.testing.assert_array_equal(result[7:-7], 8.0)
    np.testing.assert_array_equal(result[-7:], np.arange(7, 0, -1))


@pytest.mark.parametrize("segment", [15, 16])
def test_streamed_spectral_estimates_match_independent_periodograms(segment):
    x = np.random.default_rng(11).normal(size=151)
    dt = 0.125
    overlap = 0.375
    step = int(segment * (1 - overlap))
    starts = np.arange(0, x.size - segment + 1, step)
    w = np.hanning(segment)
    references = []
    for start in starts:
        part = x[start : start + segment]
        p = np.abs(np.fft.rfft((part - part.mean()) * w)) ** 2 * dt / np.sum(w**2)
        p[1 : -1 if segment % 2 == 0 else None] *= 2
        references.append(p)
    references = np.array(references).T
    times, frequencies, spectrum = signal.spectrogram(x, segment, overlap, dt)
    f_welch, average = signal.welch(x, segment, overlap, dt)
    np.testing.assert_allclose(times, (starts + segment / 2) * dt)
    np.testing.assert_allclose(frequencies, np.fft.rfftfreq(segment, dt))
    np.testing.assert_allclose(f_welch, frequencies)
    np.testing.assert_allclose(spectrum, references, atol=1e-12)
    np.testing.assert_allclose(average, references.mean(axis=1), atol=1e-12)


def test_spectrogram_without_a_complete_frame_has_explicit_empty_time_axis():
    times, frequencies, spectrum = signal.spectrogram([1, 2, 3], segment=16)
    assert times.shape == (0,)
    assert spectrum.shape == (9, 0)
    np.testing.assert_allclose(frequencies, np.fft.rfftfreq(16))


@pytest.mark.parametrize("function", [signal.welch, signal.spectrogram])
@pytest.mark.parametrize("kwargs", [{"segment": 0}, {"segment": 2.5},
                                   {"overlap": -0.1}, {"overlap": 1},
                                   {"overlap": np.nan}, {"dt": 0}, {"dt": np.inf}])
def test_spectral_estimation_rejects_invalid_sampling(function, kwargs):
    with pytest.raises(ValueError):
        function(np.arange(256.0), **kwargs)


@pytest.mark.parametrize("n,num,k", [(8, 16, 4), (16, 8, 4), (8, 15, 4),
                                   (15, 8, 4), (7, 8, 3), (8, 7, 3),
                                   (1, 8, 0), (8, 1, 0), (8, 8, 4)])
def test_resample_preserves_tone_amplitude_at_nyquist_and_odd_boundaries(n, num, k):
    x = np.cos(2 * np.pi * k * np.arange(n) / n)
    expected = np.cos(2 * np.pi * k * np.arange(num) / num)
    np.testing.assert_allclose(signal.resample(x, num), expected, atol=1e-11)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 31])
@pytest.mark.parametrize("name", ["haar", "db4", "db8", "coif2"])
def test_short_wavelet_filter_matches_periodic_definition(n, name):
    x = np.random.default_rng(27).normal(size=n)
    padded = np.append(x, x[-1]) if n % 2 else x
    _, _, h, g = wavelet.wavelet_filters(name)
    indices = (2 * np.arange(padded.size // 2)[:, None] + np.arange(h.size)) % padded.size
    a, d = wavelet.dwt(x, name)
    np.testing.assert_allclose(a, padded[indices] @ h, atol=1e-12)
    np.testing.assert_allclose(d, padded[indices] @ g, atol=1e-12)
    np.testing.assert_allclose(wavelet.idwt(a, d, name, length=n), x, atol=1e-10)
    assert a.flags.owndata and d.flags.owndata


def test_long_custom_wavelet_filter_wraps_all_taps_and_uses_transpose_synthesis():
    rng = np.random.default_rng(37)
    x = rng.normal(size=14)
    h = rng.normal(size=40)
    _, _, h, g = wavelet.wavelet_filters(h)
    indices = (2 * np.arange(7)[:, None] + np.arange(h.size)) % x.size
    a, d = wavelet.dwt(x, h)
    np.testing.assert_allclose(a, x[indices] @ h, atol=1e-12)
    np.testing.assert_allclose(d, x[indices] @ g, atol=1e-12)
    expected = np.zeros(x.size)
    for j in range(a.size):
        np.add.at(expected, indices[j], a[j] * h + d[j] * g)
    np.testing.assert_allclose(wavelet.idwt(a, d, h), expected, atol=1e-11)
    assert a.flags.owndata and d.flags.owndata


def test_stationary_wavelet_dilation_matches_periodic_tap_definition():
    x = np.random.default_rng(17).normal(size=31)
    _, _, h, g = wavelet.wavelet_filters("db4")
    approximation = x
    details = []
    for j in range(9):
        details.append(sum(gk * np.roll(approximation, -(k * 2**j))
                           for k, gk in enumerate(g)))
        approximation = sum(hk * np.roll(approximation, -(k * 2**j))
                            for k, hk in enumerate(h))
    result = wavelet.swt(x, "db4", 9)
    for actual, expected in zip(result, [approximation] + details[::-1]):
        np.testing.assert_allclose(actual, expected, atol=1e-11)
    np.testing.assert_allclose(wavelet.iswt(result, "db4"), x, atol=1e-10)


def test_stationary_wavelet_workspace_does_not_grow_with_dilated_filter_length():
    x = np.random.default_rng(18).normal(size=128)
    wavelet.swt(x, "db4", 1)  # Initialize FFT state outside the allocation check.
    tracemalloc.start()
    try:
        result = wavelet.swt(x, "db4", 16)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512_000  # Previously over 4 MiB for mostly-zero filters.
    np.testing.assert_allclose(wavelet.iswt(result, "db4"), x, atol=1e-9)
