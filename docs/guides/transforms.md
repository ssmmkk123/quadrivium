# Transforms

```python
from quadrivium.transforms import fft, wavedec, welch, cwt
import quadrivium as qd          # qd.fft, qd.convolve, qd.power_spectrum, ...
```

57 routines: the discrete Fourier family and its fast algorithms, signal
processing built on them, and wavelets. Full signatures are in the
[`transforms` reference](../api/transforms.md).

## The Fourier family

```pycon
>>> import numpy as np
>>> import quadrivium as qd
>>> x = np.array([1.0, 2.0, 3.0, 4.0])
>>> X = qd.fft(x)
>>> float(np.max(np.abs(qd.ifft(X) - x))) < 1e-14
True
>>> round(float(np.real(X[0])), 12)                 # the DC term is the sum
10.0

```

`fft` chooses its algorithm from the length: radix-2 Cooley-Tukey for a power
of two, mixed-radix when the length factorizes, and Bluestein's chirp-z
otherwise — so a prime length is still `O(n log n)`, not `O(n²)`:

```pycon
>>> from quadrivium.transforms import dft, fft_bluestein
>>> y = np.random.default_rng(0).standard_normal(97)      # 97 is prime
>>> float(np.max(np.abs(qd.fft(y) - dft(y)))) < 1e-10
True

```

| Transform | Function | For |
| --- | --- | --- |
| complex DFT | `fft`, `ifft` | general |
| definition, `O(n²)` | `dft`, `idft`, `dft_matrix` | teaching, checking |
| real input | `rfft`, `irfft` | half the work, half the storage |
| 2-D | `fft2`, `ifft2` | images, 2-D PDEs |
| cosine (4 kinds) | `dct`, `idct` | compression, Chebyshev methods |
| sine (2 kinds) | `dst` | Dirichlet boundary problems |
| Hartley | `hartley` | real-to-real, no complex arithmetic |
| one frequency only | `goertzel` | `O(n)` per bin, cheaper than a whole FFT |

`fftfreq` gives the frequency of each bin and `fftshift` reorders them to put
zero in the middle — both needed for any plot to mean anything.

Parseval's identity holds to machine precision, which is the standard check
that a transform is correctly normalized:

```pycon
>>> n = 64
>>> sig = np.random.default_rng(1).standard_normal(n)
>>> lhs = float(np.sum(sig**2))
>>> rhs = float(np.sum(np.abs(qd.fft(sig))**2) / n)
>>> abs(lhs - rhs) < 1e-10
True

```

## Convolution and correlation

Direct convolution costs `O(nm)`; through the FFT it is `O(n log n)`, and the
two agree:

```pycon
>>> from quadrivium.transforms import convolve, convolve_fft
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([0.0, 1.0, 0.5])
>>> float(np.max(np.abs(convolve(a, b) - convolve_fft(a, b)))) < 1e-12
True
>>> convolve(a, b).round(10).tolist()
[0.0, 1.0, 2.5, 4.0, 1.5]

```

`correlate`, `cross_correlation`, and `autocorrelation` are the correlation
versions; `deconvolve` inverts a convolution with Wiener regularization, which
is what keeps it from amplifying noise at the frequencies where the kernel is
small.

## Spectral estimation

A raw periodogram is noisy and does not become less so as the record
lengthens — its variance is independent of `n`. Welch's method averages
overlapping windowed segments and trades resolution for that variance:

```pycon
>>> from quadrivium.transforms import periodogram, welch
>>> fs = 512.0
>>> t = np.arange(2048) / fs
>>> sig = np.sin(2*np.pi*50*t) + 0.5*np.random.default_rng(0).standard_normal(t.size)
>>> f_w, p_w = welch(sig, segment=256, overlap=0.5, dt=1/fs)
>>> peak = float(f_w[int(np.argmax(p_w))])
>>> abs(peak - 50.0) < 2.0
True

```

`window(n, kind)` provides 18 window functions in symmetric and periodic
forms — Hann, Hamming, Blackman, Kaiser, flat-top, Blackman-Harris, Nuttall,
Bohman, Parzen and the rest. The choice is a trade between the width of the
main lobe (resolution) and the height of the side lobes (leakage):
rectangular resolves closest but leaks worst, flat-top has the most accurate
amplitude, Kaiser is tunable through `beta`.

`spectrogram` computes the short-time transform for a signal whose content
changes, and `hilbert` gives the analytic signal, from which instantaneous
amplitude and phase follow.

## Filtering and resampling

```pycon
>>> from quadrivium.transforms import lowpass_filter, resample, moving_average
>>> clean = np.sin(2*np.pi*3*t)
>>> noisy = clean + 0.3*np.sin(2*np.pi*120*t)
>>> filtered = lowpass_filter(noisy, cutoff=20.0, dt=1/fs)
>>> float(np.max(np.abs(filtered - clean))) < 0.1
True

```

`resample` changes the sample count through the frequency domain, which is
exact for a band-limited signal; `moving_average` and `savitzky_golay_filter`
smooth in the time domain, the latter preserving peak heights that a moving
average flattens.

## Wavelets

A Fourier transform says which frequencies are present; a wavelet transform
says which are present *when*. `wavedec` runs the multi-level decomposition and
`waverec` inverts it exactly:

```pycon
>>> sig = np.random.default_rng(0).standard_normal(256)
>>> coeffs = qd.wavedec(sig, "db4", level=4)
>>> rec = qd.waverec(coeffs, "db4", length=sig.size)
>>> float(np.max(np.abs(rec - sig))) < 1e-10
True

```

Twelve filter names covering the orthogonal families are available —
`WAVELET_FILTERS` lists them:

>>> from quadrivium.transforms import WAVELET_FILTERS
>>> list(WAVELET_FILTERS)
['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db8', 'sym2', 'sym4', 'coif1', 'coif2']

Their defining property is the number of vanishing moments: a Daubechies-N
wavelet annihilates every polynomial of degree below N, so its detail
coefficients on smooth data are zero:

```pycon
>>> from quadrivium.transforms import dwt
>>> ramp = np.arange(64, dtype=float)         # degree 1
>>> cA, cD = dwt(ramp, "db4")
>>> float(np.max(np.abs(cD[:-3]))) < 1e-10    # away from the periodic wrap
True

```

That property is what makes wavelets good at denoising: the signal is
concentrated in a few large coefficients while noise spreads across all of
them, so thresholding removes mostly noise.

```pycon
>>> from quadrivium.transforms import wavelet_denoise
>>> t2 = np.linspace(0, 1, 512)
>>> pure = np.sin(2*np.pi*5*t2)
>>> noisy2 = pure + 0.2*np.random.default_rng(0).standard_normal(t2.size)
>>> clean2 = wavelet_denoise(noisy2, wavelet="db4")
>>> float(np.std(clean2 - pure)) < float(np.std(noisy2 - pure))
True

```

`universal_threshold` computes Donoho and Johnstone's `σ√(2 log n)`; `swt` and
`iswt` are the shift-invariant (undecimated) transform, which avoids the
artefacts that decimation introduces at edges; `dwt2` and `idwt2` handle
images; `cwt` is the continuous transform with Morlet and Ricker wavelets, and
`scale_to_frequency` converts its scales to frequencies.

## Pitfalls

- **Sampling below the Nyquist rate aliases, invisibly.** A component above
  `fs/2` appears as a lower frequency, and no transform can undo it.
- **A finite record leaks.** Unless the signal is exactly periodic in the
  window, energy spreads across bins. Apply a window.
- **Zero padding does not add information.** It interpolates the spectrum,
  making it look smoother without improving resolution.
- **The periodogram's variance does not decrease with more data.** Use `welch`.
- **DWT levels are limited by the length.** Level `k` needs at least
  `2ᵏ · (filter length)` samples; beyond that the coefficients are boundary
  artefacts.

## See also

- [`transforms` API reference](../api/transforms.md) — every signature.
- [PDE guide](pde.md) — spectral solvers built on these transforms.
- [Approximation guide](approx.md) — Fourier series from the coefficient side.
- `examples/05_pde_and_transforms.py` — a runnable tour.
