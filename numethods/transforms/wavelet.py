"""Wavelet transforms: discrete, stationary and continuous.

Where the Fourier transform trades all time information for perfect frequency
resolution, wavelets keep both at a fixed trade-off set by scale.  That is what
makes them the tool for transient and non-stationary signals -- a step edge is
a handful of large coefficients here and an infinite Fourier tail there.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DimensionError
from ..core.utils import as_vector

__all__ = [
    "WAVELET_FILTERS",
    "wavelet_filters",
    "dwt",
    "idwt",
    "wavedec",
    "waverec",
    "dwt2",
    "idwt2",
    "swt",
    "iswt",
    "wavelet_denoise",
    "universal_threshold",
    "soft_threshold_array",
    "hard_threshold_array",
    "cwt",
    "scale_to_frequency",
    "morlet",
    "mexican_hat",
    "goertzel",
    "wavelet_energy",
]

# Orthogonal scaling-filter coefficients (low-pass analysis, 'h').  Daubechies
# 'dbN' has N vanishing moments and 2N taps; symN and coifN are the
# least-asymmetric and Coiflet families.
WAVELET_FILTERS = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db1": [0.7071067811865476, 0.7071067811865476],
    "db2": [0.48296291314469025, 0.836516303737469,
            0.22414386804185735, -0.12940952255092145],
    "db3": [0.3326705529509569, 0.8068915093133388, 0.4598775021193313,
            -0.13501102001039084, -0.08544127388224149, 0.035226291882100656],
    "db4": [0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
            -0.02798376941698385, -0.18703481171888114, 0.030841381835986965,
            0.032883011666982945, -0.010597401784997278],
    "db5": [0.160102397974125, 0.6038292697974729, 0.7243085284385744,
            0.13842814590110342, -0.24229488706619015, -0.03224486958502952,
            0.07757149384006515, -0.006241490213011705, -0.012580751999015526,
            0.003335725285001549],
    "db6": [0.11154074335008017, 0.4946238903983854, 0.7511339080215775,
            0.3152503517092432, -0.22626469396516913, -0.12976686756709563,
            0.09750160558707936, 0.02752286553001629, -0.031582039318031156,
            0.0005538422009938016, 0.004777257511010651, -0.001077301085308479],
    "db8": [0.05441584224308161, 0.3128715909144659, 0.6756307362980128,
            0.5853546836548691, -0.015829105256023893, -0.28401554296242809,
            0.00047248457399797254, 0.128747426620186, -0.017369301002022108,
            -0.044088253931064719, 0.013981027917015516, 0.0087460940470156547,
            -0.00487035299301066, -0.000391740373376942, 0.00067544940599855677,
            -0.00011747678400228192],
    "sym2": [0.48296291314469025, 0.836516303737469,
             0.22414386804185735, -0.12940952255092145],
    "sym4": [0.032223100604042702, -0.012603967262037833, -0.099219543576847229,
             0.29785779560527736, 0.80373875180591614, 0.49761866763201545,
             -0.029635527645999409, -0.075765714789273325],
    "coif1": [-0.015655728135465, -0.072732619512854, 0.384864846864203,
              0.852572020212255, 0.337897662457809, -0.072732619512854],
    "coif2": [-0.000720549445365, -0.001823208870703, 0.005611434819394,
              0.023680171946334, -0.059434418646457, -0.076488599078311,
              0.417005184423784, 0.812723635449569, 0.386110066823092,
              -0.067372554721963, -0.041464936781959, 0.016387336463522],
}


def wavelet_filters(wavelet="haar"):
    """Return the four filters ``(dec_lo, dec_hi, rec_lo, rec_hi)``.

    The high-pass filter is the *quadrature mirror* of the low-pass one --
    reversed with alternating signs -- which is what makes the pair split the
    spectrum in half without losing information.  ``wavelet`` may also be given
    as an explicit list of scaling coefficients.
    """
    if isinstance(wavelet, str):
        key = wavelet.lower()
        if key not in WAVELET_FILTERS:
            raise ValueError(f"unknown wavelet {wavelet!r}; available: "
                             f"{sorted(WAVELET_FILTERS)}")
        h = np.asarray(WAVELET_FILTERS[key], dtype=float)
    else:
        h = as_vector(wavelet)
    n = h.size
    g = h[::-1].copy()
    g[1::2] *= -1.0                     # QMF: g[k] = (-1)^k h[n-1-k]
    return h[::-1].copy(), g[::-1].copy(), h.copy(), g.copy()


def _fft_filter(h, n):
    """Spectrum of ``h`` wrapped -- not truncated -- to length ``n``.

    ``np.fft.fft(h, n)`` pads when ``n > h.size`` but *truncates* when
    ``n < h.size``.  Truncation is wrong for circular filtering: the taps past
    the end must alias back around, which is what ``h[k] -> h[k mod n]``
    below does.  This is what keeps deep decomposition levels correct, where
    the band gets shorter than the filter.
    """
    h = np.asarray(h, dtype=float)
    if h.size <= n:
        return np.fft.fft(h, n)
    wrapped = np.zeros(n)
    for start in range(0, h.size, n):
        chunk = h[start:start + n]
        wrapped[:chunk.size] += chunk
    return np.fft.fft(wrapped)


def _circ_correlate(x, h):
    """Circular correlation ``y[i] = sum_k h[k] x[(i+k) mod n]``, via the FFT."""
    n = x.size
    return np.real(np.fft.ifft(np.fft.fft(x) * np.conj(_fft_filter(h, n))))


def _circ_convolve(x, h):
    """Circular convolution ``y[i] = sum_j x[j] h[(i-j) mod n]``, via the FFT."""
    n = x.size
    return np.real(np.fft.ifft(np.fft.fft(x) * _fft_filter(h, n)))


def dwt(x, wavelet="haar"):
    """One level of the periodic discrete wavelet transform.

    Returns ``(approx, detail)``, each of length ``n // 2``.  The analysis
    operator is ``cA[j] = sum_k h[k] x[2j+k]`` with circular indexing, whose
    rows are orthonormal when ``h`` is an orthogonal scaling filter -- which is
    what lets :func:`idwt` be the exact transpose rather than an approximate
    inverse filter.  Odd-length input is extended by one sample.
    """
    x = as_vector(x)
    _, _, h, g = wavelet_filters(wavelet)
    if x.size % 2:
        x = np.concatenate([x, x[-1:]])
    return _circ_correlate(x, h)[::2], _circ_correlate(x, g)[::2]


def idwt(cA, cD, wavelet="haar", length=None):
    """Invert one level of :func:`dwt`.

    Exactly the transpose of the analysis operator: upsample by two, then
    circularly convolve with the same filters and add.  For an orthogonal
    wavelet the transpose *is* the inverse, so reconstruction is exact to
    round-off with no boundary correction.
    """
    cA, cD = as_vector(cA), as_vector(cD)
    if cA.size != cD.size:
        raise DimensionError("approximation and detail must have equal length")
    _, _, h, g = wavelet_filters(wavelet)
    n = 2 * cA.size
    up_a = np.zeros(n)
    up_d = np.zeros(n)
    up_a[::2] = cA
    up_d[::2] = cD
    out = _circ_convolve(up_a, h) + _circ_convolve(up_d, g)
    return out if length is None else out[:length]


def wavedec(x, wavelet="haar", level=None):
    """Multi-level decomposition.

    Returns ``[cA_n, cD_n, ..., cD_1]`` -- the coarsest approximation followed
    by the detail bands from coarse to fine, the ordering :func:`waverec`
    expects.
    """
    x = as_vector(x)
    max_level = int(np.floor(np.log2(max(x.size, 1)))) if x.size > 1 else 0
    level = max_level if level is None else min(level, max_level)
    coeffs = []
    a = x
    for _ in range(level):
        a, d = dwt(a, wavelet)
        coeffs.append(d)
    return [a] + coeffs[::-1]


def waverec(coeffs, wavelet="haar", length=None):
    """Rebuild a signal from :func:`wavedec` coefficients."""
    a = as_vector(coeffs[0])
    for d in coeffs[1:]:
        d = as_vector(d)
        if a.size > d.size:
            a = a[: d.size]
        elif a.size < d.size:
            d = d[: a.size]
        a = idwt(a, d, wavelet)
    return a if length is None else a[:length]


def dwt2(X, wavelet="haar"):
    """One level of the 2-D (separable) DWT.

    Returns ``(cA, (cH, cV, cD))``: approximation plus the horizontal, vertical
    and diagonal detail bands, obtained by transforming rows then columns.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    rows_a, rows_d = [], []
    for r in X:
        a, d = dwt(r, wavelet)
        rows_a.append(a)
        rows_d.append(d)
    A, D = np.array(rows_a), np.array(rows_d)
    cA, cH = [], []
    for j in range(A.shape[1]):
        a, d = dwt(A[:, j], wavelet)
        cA.append(a)
        cH.append(d)
    cV, cD = [], []
    for j in range(D.shape[1]):
        a, d = dwt(D[:, j], wavelet)
        cV.append(a)
        cD.append(d)
    return np.array(cA).T, (np.array(cH).T, np.array(cV).T, np.array(cD).T)


def idwt2(cA, details, wavelet="haar", shape=None):
    """Invert one level of :func:`dwt2`."""
    cH, cV, cD = details
    cA, cH, cV, cD = (np.atleast_2d(np.asarray(c, dtype=float))
                      for c in (cA, cH, cV, cD))
    A = np.array([idwt(cA[:, j], cH[:, j], wavelet) for j in range(cA.shape[1])]).T
    D = np.array([idwt(cV[:, j], cD[:, j], wavelet) for j in range(cV.shape[1])]).T
    out = np.array([idwt(A[i], D[i], wavelet) for i in range(A.shape[0])])
    return out if shape is None else out[: shape[0], : shape[1]]


def swt(x, wavelet="haar", level=1):
    """Stationary (undecimated) wavelet transform, "a trous" algorithm.

    Skips the downsampling and dilates the filters instead, so the transform is
    shift-invariant: translating the input translates the coefficients.  The
    ordinary DWT is not, which is why denoising with it can leave visible
    artefacts that depend on where the signal happens to start.  The price is
    redundancy -- every level keeps the full length.
    """
    x = as_vector(x)
    _, _, h, g = wavelet_filters(wavelet)
    a = x.copy()
    out = []
    for j in range(level):
        step = 2 ** j
        lo = _upsample_filter(h, step)
        hi = _upsample_filter(g, step)
        out.append(_circ_correlate(a, hi))
        a = _circ_correlate(a, lo)
    return [a] + out[::-1]


def _upsample_filter(h, step):
    """Insert ``step - 1`` zeros between taps (the 'a trous' dilation)."""
    if step == 1:
        return h
    out = np.zeros((h.size - 1) * step + 1)
    out[::step] = h
    return out


def iswt(coeffs, wavelet="haar"):
    """Invert :func:`swt` by averaging the two reconstruction phases."""
    a = as_vector(coeffs[0])
    details = list(coeffs[1:])[::-1]
    _, _, h, g = wavelet_filters(wavelet)
    for j, d in reversed(list(enumerate(details))):
        step = 2 ** j
        lo = _upsample_filter(h, step)
        hi = _upsample_filter(g, step)
        # The undecimated frame has redundancy 2 at every level -- |H|^2 + |G|^2
        # is 2, not 1 -- so the adjoint must be halved to invert it.
        a = 0.5 * (_circ_convolve(a, lo) + _circ_convolve(as_vector(d), hi))
    return a


def universal_threshold(detail, n=None):
    """Donoho-Johnstone universal threshold ``sigma sqrt(2 ln n)``.

    ``sigma`` is estimated from the *median absolute deviation* of the finest
    detail band, divided by 0.6745 to make it consistent for Gaussian noise.
    The median is used rather than the standard deviation precisely because a
    few large signal coefficients would inflate the latter and under-threshold
    everything else.
    """
    d = as_vector(detail)
    n = d.size if n is None else n
    sigma = float(np.median(np.abs(d))) / 0.6745
    return sigma * np.sqrt(2.0 * np.log(max(n, 2)))


def soft_threshold_array(x, t):
    """Soft (shrinkage) threshold: ``sign(x) max(|x| - t, 0)``."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def hard_threshold_array(x, t):
    """Hard threshold: keep coefficients above ``t``, zero the rest."""
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) > t, x, 0.0)


def wavelet_denoise(x, wavelet="db4", level=None, threshold=None,
                    mode: str = "soft"):
    """Denoise by thresholding wavelet detail coefficients.

    The approximation band is left untouched -- it holds the signal's coarse
    structure, and thresholding it would remove the signal rather than the
    noise.
    """
    x = as_vector(x)
    coeffs = wavedec(x, wavelet, level)
    if threshold is None:
        threshold = universal_threshold(coeffs[-1], x.size)
    shrink = soft_threshold_array if mode == "soft" else hard_threshold_array
    out = [coeffs[0]] + [shrink(c, threshold) for c in coeffs[1:]]
    return waverec(out, wavelet, x.size)


def morlet(t, w0: float = 6.0):
    """Complex Morlet wavelet, normalized to unit energy.

    The correction term ``exp(-w0^2/2)`` enforces the admissibility condition
    (zero mean); it is negligible for ``w0 >= 6``, which is why that is the
    conventional choice.
    """
    t = np.asarray(t, dtype=float)
    return (np.pi ** -0.25) * (np.exp(1j * w0 * t) - np.exp(-0.5 * w0 * w0)) \
        * np.exp(-0.5 * t * t)


def mexican_hat(t, sigma: float = 1.0):
    """Ricker ("Mexican hat") wavelet: the second derivative of a Gaussian."""
    t = np.asarray(t, dtype=float) / sigma
    c = 2.0 / (np.sqrt(3.0 * sigma) * np.pi**0.25)
    return c * (1.0 - t * t) * np.exp(-0.5 * t * t)


def cwt(x, scales, wavelet=morlet, dt: float = 1.0, **kwargs):
    """Continuous wavelet transform by FFT convolution.

    ``scales`` are in *samples*, the usual convention: a scale ``s`` picks out
    the Fourier frequency ``f = fc / (s * dt)``, where ``fc`` is the wavelet's
    centre frequency (``w0 / 2 pi``, so ~0.955 for the default Morlet).
    :func:`scale_to_frequency` does that conversion.

    Returns an array of shape ``(len(scales), len(x))``, normalized by
    ``1/sqrt(s)`` so that power is comparable across scales.
    """
    x = as_vector(x)
    scales = np.atleast_1d(np.asarray(scales, dtype=float))
    n = x.size
    out = np.zeros((scales.size, n), dtype=complex)
    Xf = np.fft.fft(x)
    for i, s in enumerate(scales):
        m = min(int(10 * s) | 1, 4 * n + 1)          # odd-length support
        t = (np.arange(m) - (m - 1) // 2) / s
        psi = np.conj(np.asarray(wavelet(t, **kwargs), dtype=complex)) / np.sqrt(s)
        pad = np.zeros(n, dtype=complex)
        take = min(m, n)
        pad[:take] = psi[:take]
        conv = np.fft.ifft(Xf * np.fft.fft(pad))
        # Undo the shift introduced by placing the wavelet centre at index 0.
        out[i] = np.roll(conv, -((take - 1) // 2)) * dt
    return out


def scale_to_frequency(scales, dt: float = 1.0, w0: float = 6.0):
    """Fourier frequency corresponding to each Morlet CWT scale."""
    scales = np.atleast_1d(np.asarray(scales, dtype=float))
    return (w0 / (2.0 * np.pi)) / (scales * dt)


def wavelet_energy(coeffs):
    """Energy in each band of a :func:`wavedec` result.

    For an orthogonal wavelet these sum to the signal energy -- Parseval's
    theorem holds band by band, which is what makes the decomposition an
    energy budget rather than just a filter bank.
    """
    return np.array([float(np.sum(np.asarray(c, dtype=float) ** 2))
                     for c in coeffs])


def goertzel(x, k: int):
    """Single DFT bin by the Goertzel algorithm.

    Computes ``X[k]`` in ``O(n)`` with two real multiplies per sample, against
    ``O(n log n)`` for a whole FFT.  When only a handful of bins are wanted --
    tone detection, DTMF decoding -- this is the cheaper route.
    """
    x = as_vector(x)
    n = x.size
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)
    s1 = s2 = 0.0
    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2, s1 = s1, s0
    return complex(s1 * np.cos(w) - s2, s1 * np.sin(w)) * np.exp(0j)
