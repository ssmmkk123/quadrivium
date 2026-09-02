"""Figures for the transforms guide."""

from __future__ import annotations

import numpy as np

from quadrivium.transforms import (dwt, fft, fftfreq, fftshift, periodogram,
                                   spectrogram, wavedec, wavelet_denoise, welch,
                                   window)

from . import figure
from figstyle import annotate, finish, ordinal, sequential_cmap

GUIDE = "guides/transforms.md"


@figure("transforms-signal-spectrum", GUIDE,
        "A noisy two-tone signal and the spectrum that finds both tones",
        size=(7.0, 3.6))
def signal_spectrum(fig, scheme):
    fs, n = 512.0, 1024
    t = np.arange(n) / fs
    signal = (1.0 * np.sin(2 * np.pi * 40 * t) + 0.4 * np.sin(2 * np.pi * 120 * t)
              + 0.5 * np.random.default_rng(0).standard_normal(n))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t[:256], signal[:256], color=scheme.series[0], linewidth=1.1)
    finish(ax1, scheme, title="Half a second of the signal", xlabel="time (s)",
           ylabel="amplitude", grid="both")

    ax2 = fig.add_subplot(1, 2, 2)
    spectrum = np.abs(fft(signal)) * 2 / n
    freqs = fftfreq(n, 1 / fs)
    half = n // 2
    ax2.plot(freqs[:half], spectrum[:half], color=scheme.series[0], linewidth=1.2)
    for frequency, amplitude in ((40, 1.0), (120, 0.4)):
        index = int(np.argmin(np.abs(freqs[:half] - frequency)))
        ax2.scatter([freqs[index]], [spectrum[index]], s=36, zorder=3,
                    color=scheme.series[1], edgecolors=scheme.surface,
                    linewidths=1.0)
        ax2.text(freqs[index] + 6, spectrum[index],
                 f"{frequency} Hz, amplitude {amplitude:g}", fontsize=8,
                 color=scheme.secondary, va="center")
    finish(ax2, scheme, title="Its magnitude spectrum", xlabel="frequency (Hz)",
           ylabel="amplitude", grid="both")
    ax2.set_xlim(0, 256)
    fig.tight_layout()


@figure("transforms-windows", GUIDE,
        "Four window functions and the leakage each one trades for resolution",
        size=(7.0, 3.6))
def windows(fig, scheme):
    n = 128
    kinds = ["boxcar", "hann", "blackman", "kaiser"]
    labels = {"boxcar": "rectangular", "hann": "Hann", "blackman": "Blackman",
              "kaiser": r"Kaiser, $\beta$ = 8.6"}

    ax1 = fig.add_subplot(1, 2, 1)
    for i, kind in enumerate(kinds):
        w = window(n, kind=kind)
        ax1.plot(np.arange(n), w, color=scheme.series[i], label=labels[kind])
    finish(ax1, scheme, title="In time", xlabel="sample", ylabel="weight",
           grid="both", legend=True, legend_kw=dict(loc="lower center",
                                                    fontsize=7.5))
    ax1.set_ylim(-0.05, 1.35)

    ax2 = fig.add_subplot(1, 2, 2)
    pad = 4096
    for i, kind in enumerate(kinds):
        w = np.zeros(pad)
        w[:n] = window(n, kind=kind)
        response = np.abs(fftshift(fft(w)))
        response = response / response.max()
        bins = fftshift(fftfreq(pad, 1 / n))
        ax2.plot(bins, 20 * np.log10(np.maximum(response, 1e-8)),
                 color=scheme.series[i], linewidth=1.2, label=labels[kind])
    finish(ax2, scheme, title="In frequency (decibels)",
           xlabel="frequency (bins)", ylabel="response (dB)", grid="both",
           legend=True, legend_kw=dict(loc="upper right", fontsize=7.5))
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-140, 8)
    fig.tight_layout()


@figure("transforms-periodogram-welch", GUIDE,
        "A raw periodogram against Welch's averaged estimate", size=(7.0, 3.6))
def periodogram_welch(fig, scheme):
    fs, n = 512.0, 4096
    t = np.arange(n) / fs
    signal = (np.sin(2 * np.pi * 50 * t)
              + 0.5 * np.random.default_rng(0).standard_normal(n))
    f_raw, p_raw = periodogram(signal, dt=1 / fs)
    f_welch, p_welch = welch(signal, segment=256, overlap=0.5, dt=1 / fs)

    ax = fig.add_subplot()
    ax.semilogy(f_raw, np.maximum(p_raw, 1e-8), color=scheme.series[1],
                linewidth=0.8, label="periodogram — variance independent of n")
    ax.semilogy(f_welch, np.maximum(p_welch, 1e-8), color=scheme.series[0],
                linewidth=1.8, label="Welch, 256-sample segments, 50% overlap")
    finish(ax, scheme,
           title="Averaging trades frequency resolution for a readable estimate",
           xlabel="frequency (Hz)", ylabel="power spectral density", grid="both",
           legend=True, legend_kw=dict(loc="lower center", fontsize=8))
    ax.set_xlim(0, 256)
    peak = float(f_welch[int(np.argmax(p_welch))])
    annotate(ax, f"the tone, at {peak:.0f} Hz", (peak, float(np.max(p_welch))),
             (peak + 40, float(np.max(p_welch)) * 2.5), scheme, ha="left")


@figure("transforms-spectrogram", GUIDE,
        "A chirp: one transform for the whole record says nothing about when",
        size=(7.0, 3.6))
def spectrogram_figure(fig, scheme):
    fs, n = 1024.0, 4096
    t = np.arange(n) / fs
    chirp = np.sin(2 * np.pi * (20 + 180 * t / t[-1]) * t)
    chirp[n // 2:] += 0.7 * np.sin(2 * np.pi * 300 * t[n // 2:])

    ax1 = fig.add_subplot(1, 2, 1)
    spectrum = np.abs(fft(chirp)) * 2 / n
    freqs = fftfreq(n, 1 / fs)
    ax1.plot(freqs[:n // 2], spectrum[:n // 2], color=scheme.series[0],
             linewidth=1.0)
    finish(ax1, scheme, title="The whole-record spectrum",
           xlabel="frequency (Hz)", ylabel="amplitude", grid="both")
    ax1.set_xlim(0, 450)

    ax2 = fig.add_subplot(1, 2, 2)
    times, freqs2, S = spectrogram(chirp, segment=256, overlap=0.75, dt=1 / fs)
    half = freqs2.size // 2
    image = ax2.pcolormesh(times, freqs2[:half],
                           20 * np.log10(np.maximum(S[:half], 1e-6)),
                           cmap=sequential_cmap(scheme), shading="auto",
                           rasterized=True)
    ax2.set_ylim(0, 450)
    ax2.grid(False)
    bar = fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.03)
    bar.ax.tick_params(labelsize=7, color=scheme.muted, labelcolor=scheme.muted)
    bar.outline.set_edgecolor(scheme.axis)
    bar.set_label("dB", fontsize=7, color=scheme.muted)
    finish(ax2, scheme, title="The short-time transform", xlabel="time (s)",
           ylabel="frequency (Hz)", grid=None)
    fig.tight_layout()


@figure("transforms-wavelet-denoise", GUIDE,
        "Wavelet thresholding: the signal is in a few coefficients, the noise "
        "is in all of them", size=(7.0, 4.0))
def wavelet_denoise_figure(fig, scheme):
    rng = np.random.default_rng(0)
    n = 512
    t = np.linspace(0, 1, n)
    pure = np.sin(2 * np.pi * 5 * t) + 0.7 * np.where(t > 0.6, 1.0, 0.0)
    noisy = pure + 0.25 * rng.standard_normal(n)
    cleaned = wavelet_denoise(noisy, wavelet="db4")

    layout = fig.add_gridspec(2, 1, height_ratios=(1.35, 1.0), hspace=0.42)
    ax1 = fig.add_subplot(layout[0])
    ax1.plot(t, noisy, color=scheme.muted, linewidth=0.9, label="noisy")
    ax1.plot(t, pure, color=scheme.secondary, linewidth=1.3,
             linestyle=(0, (5, 2)), label="signal")
    ax1.plot(t, cleaned, color=scheme.series[0], linewidth=1.6,
             label=f"denoised — residual std "
                   f"{float(np.std(cleaned - pure)):.3f} against "
                   f"{float(np.std(noisy - pure)):.3f}")
    finish(ax1, scheme, title="A step and a sine under noise", xlabel="t",
           ylabel="value", grid="both", legend=True,
           legend_kw=dict(loc="upper left", fontsize=7.5, ncol=3))
    ax1.set_ylim(-2.2, 3.4)

    ax2 = fig.add_subplot(layout[1])
    coefficients = wavedec(noisy, "db4", level=4)
    offset = 0
    colours = ordinal(scheme, len(coefficients))
    for level, block in enumerate(coefficients):
        block = np.asarray(block, dtype=float)
        index = np.arange(offset, offset + block.size)
        ax2.bar(index, np.abs(block), width=1.0, color=colours[level])
        ax2.text(offset + block.size / 2, 3.0,
                 "approx" if level == 0 else f"detail {len(coefficients) - level}",
                 fontsize=7, ha="center", color=scheme.secondary)
        offset += block.size
    ax2.axhline(0.25 * np.sqrt(2 * np.log(n)), color=scheme.series[1],
                linewidth=1.1, linestyle=(0, (4, 3)),
                label=r"universal threshold $\sigma\sqrt{2\log n}$")
    finish(ax2, scheme, title="Magnitude of every coefficient, by level",
           xlabel="coefficient", ylabel="|coefficient|", grid="y", legend=True,
           legend_kw=dict(loc="upper right", fontsize=7.5))
    ax2.set_ylim(0, 4.2)


@figure("transforms-aliasing", GUIDE,
        "Sampling below the Nyquist rate: a high frequency arrives as a low "
        "one", size=(7.0, 3.6))
def aliasing(fig, scheme):
    fs = 10.0
    true_frequency, alias_frequency = 9.0, 1.0
    t = np.linspace(0, 2, 2000)
    samples = np.arange(0, 2 + 1e-9, 1 / fs)

    ax = fig.add_subplot()
    ax.plot(t, np.sin(2 * np.pi * true_frequency * t), color=scheme.series[1],
            linewidth=1.0, label=f"{true_frequency:g} Hz — the real signal")
    ax.plot(t, np.sin(2 * np.pi * alias_frequency * t), color=scheme.series[0],
            linewidth=1.8, label=f"{alias_frequency:g} Hz — what the samples say")
    ax.scatter(samples, np.sin(2 * np.pi * true_frequency * samples), s=34,
               zorder=3, color=scheme.ink, edgecolors=scheme.surface,
               linewidths=1.0, label=f"samples at {fs:g} Hz")
    finish(ax, scheme,
           title=f"Both curves pass through every sample — "
                 f"{true_frequency:g} Hz is above the {fs / 2:g} Hz Nyquist limit",
           xlabel="time (s)", ylabel="amplitude", grid="both", legend=True,
           legend_kw=dict(loc="upper center", ncol=3, fontsize=8))
    ax.set_ylim(-1.5, 2.0)


@figure("transforms-vanishing-moments", GUIDE,
        "A Daubechies-N wavelet annihilates polynomials of degree below N",
        size=(7.0, 3.4))
def vanishing_moments(fig, scheme):
    n = 128
    x = np.linspace(0, 1, n)
    signals = [("constant", np.ones(n)), ("linear", x),
               ("quadratic", x ** 2), ("cubic", x ** 3)]
    wavelets = ["haar", "db2", "db3", "db4"]

    ax = fig.add_subplot()
    width = 0.2
    for i, (name, signal) in enumerate(signals):
        magnitudes = []
        for wavelet in wavelets:
            _, detail = dwt(signal, wavelet)
            # The last few coefficients straddle the periodic wrap, where the
            # signal is not a polynomial at all; the interior ones are the test.
            magnitudes.append(max(float(np.max(np.abs(detail[:-4]))), 1e-18))
        ax.bar(np.arange(len(wavelets)) + (i - 1.5) * width, magnitudes,
               width=width, color=scheme.series[i], label=name)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(wavelets)))
    ax.set_xticklabels([f"{w} ({k + 1} vanishing moment"
                        f"{'s' if k else ''})"
                        for k, w in enumerate(wavelets)], fontsize=8)
    finish(ax, scheme,
           title="Largest interior detail coefficient, by wavelet and by signal",
           ylabel="|detail coefficient|", grid="y", legend=True,
           legend_kw=dict(loc="lower left", ncol=4, fontsize=8))
    ax.set_ylim(1e-18, 1e3)
