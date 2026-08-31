"""Signal processing built on the transforms: convolution, correlation,
filtering and spectral estimation."""

from __future__ import annotations

import numpy as np

from ..core.utils import as_vector
from .fourier import fft, ifft, next_power_of_two, rfft

__all__ = [
    "convolve",
    "convolve_fft",
    "correlate",
    "autocorrelation",
    "cross_correlation",
    "deconvolve",
    "power_spectrum",
    "periodogram",
    "welch",
    "window",
    "spectrogram",
    "hilbert",
    "resample",
    "moving_average",
    "savitzky_golay_filter",
    "lowpass_filter",
    "zero_pad",
]


def zero_pad(x, n: int):
    """Zero pad (or truncate) a signal to length ``n``."""
    x = as_vector(x)
    out = np.zeros(n)
    m = min(n, x.size)
    out[:m] = x[:m]
    return out


def convolve(a, b, mode: str = "full"):
    """Direct convolution, ``O(nm)``.

    Use :func:`convolve_fft` when both signals are long.
    """
    a, b = as_vector(a), as_vector(b)
    n, m = a.size, b.size
    out = np.zeros(n + m - 1)
    for i in range(n):
        out[i : i + m] += a[i] * b
    if mode == "full":
        return out
    if mode == "same":
        start = (m - 1) // 2
        return out[start : start + n]
    if mode == "valid":
        if n < m:
            a, b, n, m = b, a, m, n
        return out[m - 1 : n]
    raise ValueError("mode must be 'full', 'same' or 'valid'")


def convolve_fft(a, b, mode: str = "full"):
    """Convolution by the FFT: ``O((n+m) log(n+m))``.

    Both signals are zero padded to at least ``n+m-1`` so the circular
    convolution the FFT computes equals the linear one.
    """
    a, b = as_vector(a), as_vector(b)
    n, m = a.size, b.size
    size = next_power_of_two(n + m - 1)
    A = fft(zero_pad(a, size))
    B = fft(zero_pad(b, size))
    full = np.real(ifft(A * B))[: n + m - 1]
    if mode == "full":
        return full
    if mode == "same":
        start = (m - 1) // 2
        return full[start : start + n]
    if mode == "valid":
        if n < m:
            n, m = m, n
        return full[m - 1 : n]
    raise ValueError("mode must be 'full', 'same' or 'valid'")


def correlate(a, b, mode: str = "full"):
    """Cross-correlation ``sum a[n+k] b[n]`` (convolution with a reversed kernel)."""
    return convolve(a, as_vector(b)[::-1], mode)


def cross_correlation(a, b, normalize: bool = False):
    """Full cross-correlation with the corresponding lag vector."""
    a, b = as_vector(a), as_vector(b)
    c = correlate(a, b, "full")
    lags = np.arange(-(b.size - 1), a.size)
    if normalize:
        denom = np.sqrt(float(a @ a) * float(b @ b))
        c = c / denom if denom > 0 else c
    return lags, c


def autocorrelation(x, normalize: bool = True, max_lag=None):
    """Autocorrelation of a signal (mean removed), for lags ``0..max_lag``."""
    x = as_vector(x) - np.mean(x)
    n = x.size
    size = next_power_of_two(2 * n)
    X = fft(zero_pad(x, size))
    r = np.real(ifft(X * np.conj(X)))[:n]
    if normalize and r[0] != 0:
        r = r / r[0]
    return r if max_lag is None else r[: max_lag + 1]


def deconvolve(y, h, regularization: float = 1e-8):
    """Wiener-regularized deconvolution: recover ``x`` from ``y = x * h``.

    Deconvolution is ill-posed -- the transfer function's small values amplify
    noise -- so the inverse filter is damped by ``regularization``.
    """
    y, h = as_vector(y), as_vector(h)
    size = next_power_of_two(y.size + h.size)
    Y = fft(zero_pad(y, size))
    H = fft(zero_pad(h, size))
    X = Y * np.conj(H) / (np.abs(H) ** 2 + regularization)
    return np.real(ifft(X))[: y.size - h.size + 1]


_COSINE_WINDOWS = {
    # Generalized cosine windows, given as their expansion coefficients a_k in
    # w[i] = sum_k (-1)^k a_k cos(2 pi k i / (N-1)).
    "hann": (0.5, 0.5),
    "hamming": (0.54, 0.46),
    "blackman": (0.42, 0.5, 0.08),
    "blackman_harris": (0.35875, 0.48829, 0.14128, 0.01168),
    "nuttall": (0.355768, 0.487396, 0.144232, 0.012604),
    "blackman_nuttall": (0.3635819, 0.4891775, 0.1365995, 0.0106411),
    "flattop": (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368),
}


def window(n: int, kind: str = "hann", sym: bool = True, beta: float = 8.6,
           alpha: float = 0.5):
    """Window functions for spectral analysis.

    Available: rectangular/boxcar, hann, hamming, blackman, blackman_harris,
    nuttall, blackman_nuttall, flattop, bartlett, triangular, tukey, kaiser,
    gaussian, welch, cosine/sine, lanczos, bohman, parzen.

    ``sym=True`` gives the symmetric window used for filter design.  ``sym=False``
    gives the *periodic* (DFT-even) window, which is what spectral estimation
    wants: a symmetric window repeats its endpoint when the segment is treated
    as one period, and that discontinuity leaks energy across the spectrum.

    ``beta`` shapes the Kaiser window (larger is more sidelobe suppression at
    the cost of a wider main lobe); ``alpha`` is the Tukey taper fraction and
    the Gaussian width.
    """
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    # A periodic window of length n is the symmetric window of length n+1 with
    # its last sample dropped.
    m = n if sym else n + 1
    k = np.arange(m)
    denom = m - 1

    if kind in ("rectangular", "boxcar"):
        w = np.ones(m)
    elif kind in _COSINE_WINDOWS:
        a = _COSINE_WINDOWS[kind]
        w = np.zeros(m)
        for j, aj in enumerate(a):
            w += ((-1.0) ** j) * aj * np.cos(2 * np.pi * j * k / denom)
    elif kind in ("bartlett", "triangular"):
        w = 1.0 - np.abs((k - denom / 2) / (denom / 2))
    elif kind == "tukey":
        if alpha <= 0.0:
            w = np.ones(m)
        elif alpha >= 1.0:
            w = 0.5 - 0.5 * np.cos(2 * np.pi * k / denom)
        else:
            w = np.ones(m)
            edge = int(np.floor(alpha * denom / 2))
            t = np.arange(edge + 1)
            ramp = 0.5 * (1 + np.cos(np.pi * (2 * t / (alpha * denom) - 1)))
            w[: edge + 1] = ramp
            w[m - edge - 1:] = ramp[::-1]
    elif kind == "kaiser":
        from ..special.functions import bessel_i0

        r = 2.0 * k / denom - 1.0
        w = np.array([bessel_i0(beta * np.sqrt(max(1.0 - ri * ri, 0.0)))
                      for ri in r]) / bessel_i0(beta)
    elif kind == "gaussian":
        w = np.exp(-0.5 * ((k - denom / 2) / (alpha * denom / 2)) ** 2)
    elif kind == "welch":
        w = 1.0 - ((k - denom / 2) / (denom / 2)) ** 2
    elif kind in ("cosine", "sine"):
        w = np.sin(np.pi * k / denom)
    elif kind == "lanczos":
        r = 2.0 * k / denom - 1.0
        w = np.sinc(r)
    elif kind == "bohman":
        r = np.abs(2.0 * k / denom - 1.0)
        w = (1.0 - r) * np.cos(np.pi * r) + np.sin(np.pi * r) / np.pi
    elif kind == "parzen":
        r = np.abs(2.0 * k / denom - 1.0)
        w = np.where(r <= 0.5, 1.0 - 6 * r**2 + 6 * r**3, 2.0 * (1.0 - r) ** 3)
    else:
        raise ValueError(f"unknown window {kind!r}")
    return w[:n]


def power_spectrum(x, dt: float = 1.0, win: str = "hann", detrend: bool = True):
    """One-sided power spectral density estimate.

    Returns ``(frequencies, psd)``, normalized so that integrating the PSD
    over frequency recovers the signal variance.
    """
    x = as_vector(x)
    n = x.size
    if detrend:
        x = x - np.mean(x)
    w = window(n, win)
    # one-sided PSD scaling: integrating the result over frequency recovers
    # the signal variance (Parseval)
    scale = dt / np.sum(w**2)
    X = rfft(x * w)
    psd = scale * np.abs(X) ** 2
    if n % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    freqs = np.arange(psd.size) / (n * dt)
    return freqs, psd


def periodogram(x, dt: float = 1.0):
    """Raw (unwindowed) periodogram."""
    return power_spectrum(x, dt, win="rectangular")


def welch(x, segment: int = 256, overlap: float = 0.5, dt: float = 1.0,
          win: str = "hann"):
    """Welch's method: average periodograms of overlapping segments.

    Trades frequency resolution for a large reduction in estimator variance.
    """
    x = as_vector(x)
    step = max(int(segment * (1 - overlap)), 1)
    segs = [x[i : i + segment] for i in range(0, x.size - segment + 1, step)]
    if not segs:
        return power_spectrum(x, dt, win)
    acc = None
    for s in segs:
        f, p = power_spectrum(s, dt, win)
        acc = p if acc is None else acc + p
    return f, acc / len(segs)


def spectrogram(x, segment: int = 128, overlap: float = 0.5, dt: float = 1.0,
                win: str = "hann"):
    """Short-time Fourier transform magnitudes.

    Returns ``(times, frequencies, S)`` with ``S`` of shape ``(n_freq, n_time)``.
    """
    x = as_vector(x)
    step = max(int(segment * (1 - overlap)), 1)
    starts = list(range(0, x.size - segment + 1, step))
    cols = []
    for i in starts:
        f, p = power_spectrum(x[i : i + segment], dt, win)
        cols.append(p)
    times = (np.array(starts) + segment / 2) * dt
    return times, f, np.array(cols).T


def hilbert(x):
    """Analytic signal via the Hilbert transform.

    ``abs`` of the result is the envelope and ``angle`` the instantaneous phase.
    """
    x = as_vector(x)
    n = x.size
    X = fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return ifft(X * h)


def resample(x, num: int):
    """Band-limited resampling by truncating or zero padding the spectrum."""
    x = as_vector(x)
    n = x.size
    X = fft(x)
    Y = np.zeros(num, dtype=complex)
    m = min(n, num)
    half = m // 2
    Y[: half + 1] = X[: half + 1]
    if half > 0:
        Y[-half:] = X[-half:]
    return np.real(ifft(Y)) * (num / n)


def moving_average(x, window_size: int = 5, mode: str = "same"):
    """Simple moving average (a boxcar FIR filter)."""
    x = as_vector(x)
    kernel = np.ones(window_size) / window_size
    return convolve(x, kernel, mode)


def savitzky_golay_filter(x, window_size: int = 11, poly_order: int = 3):
    """Savitzky-Golay smoothing: local least squares polynomial fitting.

    Preserves peak heights and widths far better than a moving average.
    """
    from ..diff.finite import savitzky_golay_derivative

    return savitzky_golay_derivative(x, window_size, poly_order, order=0, dx=1.0)


def lowpass_filter(x, cutoff: float, dt: float = 1.0, order: int = 4,
                   kind: str = "brickwall"):
    """Low-pass filter in the frequency domain.

    ``'brickwall'`` zeroes everything above the cutoff; ``'butterworth'``
    applies a smooth roll-off that avoids ringing.
    """
    x = as_vector(x)
    n = x.size
    X = fft(x)
    freqs = np.abs(np.fft.fftfreq(n, d=dt))
    if kind == "brickwall":
        H = (freqs <= cutoff).astype(float)
    elif kind == "butterworth":
        H = 1.0 / np.sqrt(1.0 + (freqs / cutoff) ** (2 * order))
    else:
        raise ValueError("kind must be 'brickwall' or 'butterworth'")
    return np.real(ifft(X * H))
