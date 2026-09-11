# Transforms and signal processing

A transform changes how a sampled signal is represented. The Fourier transform
resolves oscillations across a record; short-time transforms and wavelets add
localization; filtering and resampling change the signal itself. Useful analysis
starts by recording the sampling interval, the signal units, and what happens
outside the finite record.

The [transforms reference](../api/transforms.md) documents Fourier routines,
spectral estimators, wavelets, and stateful signal-processing interfaces.
The examples below use `quadrivium.numeric` arrays throughout.

## Choose the output you need

| Question | Interface | Output to interpret |
| --- | --- | --- |
| Which frequency bins are present? | `fft`, `rfft` | Complex coefficients with magnitude and phase |
| How is power distributed over frequency? | `power_spectrum`, `welch` | Frequency and power spectral density |
| How does spectral power change over time? | `spectrogram` | Times, frequencies, nonnegative power array |
| Can I edit coefficients and reconstruct? | `stft`, `istft` | Complex coefficients plus synthesis metadata |
| Can I process successive data chunks? | `FIRFilter`, `IIRFilter`, `SOSFilter` | Filtered samples and retained filter state |
| Can I change sample rate? | `resample_poly`, `PolyphaseResampler` | Offline or causal rational resampling |
| Can I localize transient structure? | `wavedec`, `cwt` | Scale-dependent coefficients |
| Do I need only one DFT bin? | `goertzel` | One coefficient without a complete spectrum |

Magnitude, amplitude, power, and power spectral density have different units.
A graph is interpretable only when its normalization matches the quantity named
on the axis.

## Fourier conventions and reconstruction

The forward transform uses
`X[k] = sum(x[j] * exp(-2*pi*i*j*k/n))`.
The inverse includes division by `n`. Thus `X[0]` is the sample sum, and the
mean is `X[0]/n`.

```pycon
>>> from quadrivium import numeric as np
>>> from quadrivium import transforms as tr
>>> samples = np.array([1.0, 2.0, 3.0, 4.0])
>>> spectrum = tr.fft(samples)
>>> np.allclose(tr.ifft(spectrum), samples, atol=1e-12)
True
>>> round(float(spectrum[0].real), 8)
10.0

```

`fft` handles arbitrary positive lengths. Radix-2, mixed-radix, and Bluestein
implementations are also exposed for studying algorithm choices. `dft` and
`dft_matrix` evaluate the direct definition at quadratic cost and are useful
for small independent checks, not large records.

```pycon
>>> short = np.random.default_rng(5).standard_normal(17)
>>> float(np.max(np.abs(tr.fft(short) - tr.dft(short)))) < 1e-10
True
>>> energy_time = float(np.sum(short**2))
>>> energy_frequency = float(np.sum(np.abs(tr.fft(short))**2) / short.size)
>>> abs(energy_time - energy_frequency) < 1e-10
True

```

The last comparison is Parseval's identity. It checks normalization across the
whole signal rather than a single coefficient. A round-trip check alone can
miss a shared scaling error in a forward/inverse pair.

### Real transforms and odd lengths

For real input, negative-frequency coefficients are conjugates of their
positive-frequency partners. `rfft` retains `n//2 + 1` coefficients. Its input
is real; use `fft` to preserve a complex-valued signal.

```pycon
>>> odd = np.array([0.0, 1.0, -1.0, 2.0, 0.5])
>>> half = tr.rfft(odd)
>>> half.shape
(3,)
>>> np.allclose(tr.irfft(half, n=odd.size), odd, atol=1e-12)
True

```

Always pass the original `n` when reconstructing an odd-length real signal.
Without it, `irfft` infers the even length `2*(len(half)-1)`, which cannot
identify whether the original signal was odd or even.

`fft2` and `ifft2` transform matrices along both spatial directions. For
broader axis and batch operations, consult the array namespace in the
[numeric guide](numeric.md); do not assume that the educational transform
functions accept every keyword from another array library.

## Attach physical frequencies and amplitude

If samples are spaced by `dt`, their sample rate is `fs=1/dt`. The bin spacing
is `fs/n`; the Nyquist frequency for a real signal is `fs/2`. A two-sided
spectrum uses `fftfreq(n, d=dt)`. Apply `fftshift` consistently to both the
frequencies and the coefficients if you want negative frequencies on the left.

For an even-length, unwindowed real record, a one-sided amplitude spectrum is
`abs(rfft(x))/n` with the **interior** bins doubled. DC and Nyquist bins are
not doubled.

```pycon
>>> fs = 128.0
>>> n = 256
>>> time = np.arange(n) / fs
>>> signal = 1.5 * np.sin(2*np.pi*8*time) + 0.4 * np.cos(2*np.pi*23*time)
>>> coefficients = tr.rfft(signal)
>>> frequency = np.arange(coefficients.size) * fs / n
>>> amplitude = np.abs(coefficients) / n
>>> amplitude[1:-1] *= 2.0
>>> round(float(amplitude[16]), 6), round(float(amplitude[46]), 6)
(1.5, 0.4)
>>> float(frequency[16]), float(frequency[46])
(8.0, 23.0)

```

Both tones complete an integer number of cycles in this record, so their energy
lands on exact bins. With an odd-length record there is no Nyquist singleton:
double all non-DC bins. Windowed amplitude estimates also need a correction for
the window's coherent gain; PSD scaling is a different correction.

<figure markdown="span">
  ![Two-tone signals sampled at 64 and 16 Hz, with correctly normalized amplitude spectra](../assets/figures/transforms-sampling-spectrum.svg#only-light)
  ![Two-tone signals sampled at 64 and 16 Hz, with correctly normalized amplitude spectra](../assets/figures/transforms-sampling-spectrum-dark.svg#only-dark)
  <figcaption>The graph samples a 5 Hz sine of amplitude 1 and an 11 Hz sine of amplitude 0.4. At 64 Hz sampling, both tones are resolved. At 16 Hz, the 11 Hz component aliases to a negative 5 Hz sine and partially cancels the first tone, leaving amplitude 0.6. Correct amplitude normalization exposes this information loss; the FFT cannot recover the missing distinction.</figcaption>
</figure>

## Leakage, windows, and spectral estimates

A finite record is treated as a repeated segment by the DFT. If its ends do
not join smoothly, power spreads into neighboring bins. A window reduces that
boundary mismatch while broadening peaks. A longer observation interval can
improve separation of nearby tones; zero-padding only samples the same finite
record's spectrum more densely.

`window(n, kind, sym=True)` creates symmetric windows by default.
`kind` includes Hann, Hamming, Blackman, Kaiser, and other families; the
parameter names are listed in the reference. Symmetric windows are useful
for filter design; periodic variants can be selected for spectral analysis.

`periodogram` computes an unwindowed, mean-detrended power estimate.
`power_spectrum` adds window selection and a detrending option.
`welch` averages overlapping windowed segment estimates. It trades segment
frequency resolution for lower estimator variability.

```pycon
>>> noisy = signal + 0.3 * np.random.default_rng(4).standard_normal(n)
>>> frequencies, density = tr.welch(noisy, segment=128, overlap=0.5, dt=1/fs)
>>> frequencies.shape == density.shape
True
>>> abs(float(frequencies[int(np.argmax(density))]) - 8.0) < 1.1
True

```

The PSD has units of signal-units squared per frequency-unit. Integrating it
over frequency estimates power after the routine's detrending and window
normalization. A single noisy bin is not an amplitude estimate for an isolated
tone. Use segment size, window, and the number of averages together when
comparing two spectra.

For `welch`, `overlap` is a fraction, not a sample count. It averages complete
segments; if the input is shorter than a segment, it falls back to a full-record
power spectrum. `spectrogram` behaves differently for a short record: it
returns an empty time axis because there are no complete frames.

## Reconstructable time-frequency analysis

The older `spectrogram` returns `(times, frequencies, S)` with
`S.shape == (frequency_bins, frames)`. Its entries are segment power estimates;
phase is unavailable, so it is not input to an inverse transform.

`stft` returns an `STFTResult` containing complex coefficients, frequencies,
times, window, hop, FFT length, original length, and boundary-padding metadata.
It takes sample rate `fs`, whereas `spectrogram` and `welch` take sample
interval `dt`.

```pycon
>>> analysis = tr.stft(signal, segment=64, hop=16, fs=fs)
>>> analysis.coefficients.shape[0]
33
>>> recovered = tr.istft(analysis)
>>> recovered.shape == signal.shape
True
>>> float(np.max(np.abs(recovered - signal))) < 1e-12
True
>>> modified = analysis.coefficients.copy()
>>> modified[analysis.frequencies > 16.0, :] = 0.0
>>> filtered = tr.istft(analysis, coefficients=modified)
>>> filtered.shape
(256,)

```

STFT inversion uses normalized overlap-add. Default boundary padding and a
final partial frame preserve original samples. Edited coefficients must retain
the original shape. Windows and hops that leave any original sample uncovered
are rejected during inversion; a plausible spectrogram is not enough to
establish invertibility.

Short windows localize events in time but blur frequency. Long windows resolve
nearby frequencies while averaging over temporal change. Complex input or a
complex window uses a full two-sided spectrum. Retain the `STFTResult` rather
than saving only coefficient magnitudes when reconstruction is a requirement.

## Convolution and stateful filtering

Full linear convolution of lengths `n` and `m` has length `n+m-1`.
`convolve_fft` pads internally to avoid circular wraparound; a raw product of
same-length FFTs would instead produce circular convolution.

```pycon
>>> a = np.array([1.0, 2.0, 3.0])
>>> b = np.array([0.0, 1.0, 0.5])
>>> tr.convolve(a, b).round(8).tolist()
[0.0, 1.0, 2.5, 4.0, 1.5]
>>> np.allclose(tr.convolve(a, b), tr.convolve_fft(a, b), atol=1e-12)
True

```

`correlate` and `cross_correlation` measure alignment rather than convolution.
Check lag ordering before interpreting a delay. `deconvolve` uses
regularization because division by a small transfer coefficient can amplify
noise dramatically; its regularization level changes the reconstructed signal.

A causal filter can process arbitrary successive chunk lengths without
retaining old inputs. Keep the same filter object, or restore its state into
an identically configured object.

```pycon
>>> taps = np.array([0.25, 0.5, 0.25])
>>> stream = tr.FIRFilter(taps)
>>> chunked = np.concatenate([stream.process(a[:1]), stream.process(a[1:])])
>>> np.allclose(chunked, tr.convolve(a, taps)[:a.size])
True
>>> iir = tr.IIRFilter([0.3], [1.0, -0.7])
>>> first = iir.process(a[:2])
>>> restored = tr.IIRFilter([0.3], [1.0, -0.7], state=iir.state)
>>> np.allclose(restored.process(a[2:]), iir.process(a[2:]))
True

```

`IIRFilter` coefficients use ascending powers of `z**-1`; `a[0]` must be
nonzero and is normalized internally. `SOSFilter` stores biquad rows
`[b0, b1, b2, a0, a1, a2]`. State getters return copies, and `reset()` clears
state. Starting a new filter for every chunk inserts a new initial-condition
transient at every chunk boundary.

`firwin` designs low-pass or high-pass windowed-sinc filters; high-pass designs
require an odd tap count. `butterworth_sos` designs low-pass biquads.
Their `cutoff` and `fs` must use the same units, with `0 < cutoff < fs/2`.
The returned design is an input to the corresponding filter object.

## Resampling and boundary assumptions

`resample` is a Fourier-based change in sample count and therefore inherits
periodic-record assumptions. `resample_poly(x, up, down)` uses a finite-support
FIR and zero extension outside the record. Its output has
`ceil(len(x)*up/down)` samples. Boundary transients are consequences of that
extension, not evidence that the interior sample-rate conversion failed.

```pycon
>>> converted = tr.resample_poly(np.ones(12), 2, 3)
>>> converted.shape
(8,)

```

`PolyphaseResampler` is causal and carries FIR overlap between chunks.
Its `delay` is the group delay in input-sample units. It cannot use future
samples to remove that delay as an offline routine can. Feed explicit zeros to
flush a desired tail, and account for delay before comparing with an offline
resample. Persist its state together with the ratio and filter configuration.

Aliasing occurs when different continuous frequencies produce the same sampled
sequence. A digital transform cannot recover information already lost at
sampling. Downsampling also requires adequate low-pass filtering; simply
selecting every kth sample is not a general resampler.

## Wavelets and localized structure

`dwt` returns one approximation band and one detail band; `wavedec` repeats the
split and returns `[cA_level, cD_level, ..., cD_1]`. The implemented discrete
wavelet transforms use periodic indexing. An odd input length is extended by
one repeated sample, so pass the original length during reconstruction.

```pycon
>>> wave = np.random.default_rng(7).standard_normal(65)
>>> bands = tr.wavedec(wave, wavelet="db4", level=3)
>>> reconstructed = tr.waverec(bands, wavelet="db4", length=wave.size)
>>> float(np.max(np.abs(reconstructed - wave))) < 1e-10
True

```

Available names include Haar, several Daubechies filters, Symlets, and Coiflets;
`WAVELET_FILTERS` records the implemented choices. Longer filters can represent
smooth structure with fewer significant detail coefficients, but affect a
wider neighborhood near record boundaries. Deep levels are mathematically
supported with wrapped filtering; their coefficients increasingly summarize
global and boundary behavior rather than a well-localized transient.

`wavelet_denoise` thresholds detail coefficients. The default threshold is
estimated from the data; it is not a universal guarantee that narrow peaks or
small physical events survive. Soft thresholding shrinks retained coefficients,
whereas hard thresholding leaves them unchanged above the cutoff. Compare
against representative signals with known transients before choosing a level.

`dwt2`/`idwt2` handle images, `swt`/`iswt` avoid downsampling, and `cwt`
computes coefficients across specified scales. `scale_to_frequency` translates
Morlet scales using the chosen `dt` and `w0`; scale is not itself frequency.
Interpret coefficients near the record ends cautiously because the wavelet
support extends beyond the available data.

## Cosine and sine transforms

`dct` provides types 1–4, with an unnormalized convention by default;
`idct(..., kind=...)` supplies the matching inverse scaling.
The `norm=True` option applies a simple scale and is not a drop-in replacement
for another library's orthonormal DCT convention. Use the documented default
forward/inverse pair when testing reconstruction.

```pycon
>>> cosine = tr.dct(samples, kind=2)
>>> np.allclose(tr.idct(cosine, kind=2), samples, atol=1e-12)
True

```

`dst` supplies sine-transform types 1 and 2. These transforms are useful for
boundary-conditioned PDEs and approximation, but their endpoint conventions
must match the spatial grid. See [PDEs](pde.md), [approximation](approx.md),
and [scientific workflows](workflows.md) for applications.
