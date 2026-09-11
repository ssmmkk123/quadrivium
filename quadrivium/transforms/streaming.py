"""Causal filters, polyphase resampling, and reconstructable time-frequency transforms."""
from __future__ import annotations
import math
import operator
from dataclasses import dataclass
from .. import numeric as np
from .signal import window as make_window

__all__ = ["IIRFilter", "FIRFilter", "SOSFilter", "lfilter", "sosfilt", "firwin",
           "butterworth_sos", "PolyphaseResampler", "resample_poly", "STFTResult", "stft", "istft"]


def _vector(x):
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError("expected a one-dimensional signal")
    return a.astype(complex if np.iscomplexobj(a) else float, copy=False)


class IIRFilter:
    """Direct-form II transposed filter; process chunks without retaining inputs.

    ``state`` can be copied or restored between chunks. Complex coefficients and
    signals are supported. Coefficients use ascending powers of z**-1.
    """
    def __init__(self, b, a=(1.0,), *, state=None):
        b, a = _vector(b), _vector(a)
        if not a.size or not b.size or a[0] == 0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("finite nonempty coefficients and nonzero a[0] required")
        dtype = complex if np.iscomplexobj(a) or np.iscomplexobj(b) else float
        n = max(a.size, b.size)
        self.b, self.a = np.zeros(n, dtype=dtype), np.zeros(n, dtype=dtype)
        self.b[:b.size], self.a[:a.size] = b / a[0], a / a[0]
        self._state = np.zeros(n - 1, dtype=dtype)
        if state is not None:
            self.state = state

    @property
    def state(self):
        return self._state.copy()

    @state.setter
    def state(self, state):
        state = _vector(state)
        if state.shape != self._state.shape:
            raise ValueError("filter state has the wrong shape")
        self._state = state.astype(np.result_type(state, self.a), copy=True)

    def reset(self):
        self._state.fill(0)
        return self

    def process(self, x):
        x = _vector(x)
        dtype = np.result_type(x, self.a, self._state)
        if self._state.dtype != dtype:
            self._state = self._state.astype(dtype)
        out = np.empty(x.shape, dtype=dtype)
        state, n = self._state, self._state.size
        for i in range(x.size):
            value = self.b[0] * x[i] + (state[0] if n else 0)
            if n > 1:
                state[:-1] = state[1:] + self.b[1:-1] * x[i] - self.a[1:-1] * value
            if n:
                state[-1] = self.b[-1] * x[i] - self.a[-1] * value
            out[i] = value
        return out

    __call__ = process


class FIRFilter(IIRFilter):
    def __init__(self, taps, *, state=None):
        super().__init__(taps, state=state)


class SOSFilter:
    """Cascade of biquads, with rows [b0,b1,b2,a0,a1,a2]."""
    def __init__(self, sos, *, state=None):
        coefficients = np.asarray(sos)
        if coefficients.ndim != 2 or coefficients.shape[1] != 6 or not coefficients.shape[0]:
            raise ValueError("sos must have shape (sections, 6)")
        self.sections = [IIRFilter(row[:3], row[3:]) for row in coefficients]
        if state is not None:
            self.state = state

    @property
    def state(self):
        return np.array([s.state for s in self.sections])

    @state.setter
    def state(self, value):
        value = np.asarray(value)
        if value.shape != (len(self.sections), 2):
            raise ValueError("sos state shape must be (sections, 2)")
        for section, row in zip(self.sections, value):
            section.state = row

    def reset(self):
        for section in self.sections:
            section.reset()
        return self

    def process(self, x):
        for section in self.sections:
            x = section.process(x)
        return x

    __call__ = process


def lfilter(b, a, x, zi=None):
    """Filter once; with ``zi``, return ``(output, final_state)``."""
    filt = IIRFilter(b, a, state=zi)
    result = filt.process(x)
    return result if zi is None else (result, filt.state)


def sosfilt(sos, x, zi=None):
    """Filter a signal through a cascade of second-order sections.

    ``sos`` has shape ``(sections, 6)`` with each row ordered as
    ``[b0, b1, b2, a0, a1, a2]``. With ``zi=None``, sections start from
    zero state and only the filtered signal is returned. Supplying ``zi``
    with shape ``(sections, 2)`` returns ``(signal, final_state)`` so the
    state can be passed to a subsequent chunk.
    """
    filt = SOSFilter(sos, state=zi)
    result = filt.process(x)
    return result if zi is None else (result, filt.state)


def firwin(numtaps, cutoff, *, fs=2.0, window="hamming", pass_zero=True):
    """Windowed-sinc low/high-pass FIR design, normalized at DC/Nyquist."""
    numtaps = operator.index(numtaps)
    if numtaps < 1 or not math.isfinite(fs) or not 0 < cutoff < fs / 2:
        raise ValueError("require positive taps and 0 < cutoff < fs/2")
    if not pass_zero and numtaps % 2 == 0:
        raise ValueError("a high-pass FIR requires an odd tap count")
    n = np.arange(numtaps) - (numtaps - 1) / 2
    taps = 2 * cutoff / fs * np.sinc(2 * cutoff / fs * n)
    taps *= make_window(numtaps, window)
    if not pass_zero:
        taps = -taps
        taps[numtaps // 2] += 1
        taps /= float(np.sum(taps * (-1.0) ** np.arange(numtaps))) * (-1) ** (numtaps // 2)
    else:
        taps /= np.sum(taps)
    return taps


def butterworth_sos(order, cutoff, *, fs=2.0):
    """Digital Butterworth low-pass using prewarped bilinear poles and biquads."""
    order = operator.index(order)
    if order < 1 or not math.isfinite(fs) or not 0 < cutoff < fs / 2:
        raise ValueError("require order >= 1 and 0 < cutoff < fs/2")
    warped = 2 * fs * math.tan(math.pi * cutoff / fs)
    rows = []
    for k in range(order):
        angle = math.pi * (2 * k + 1 + order) / (2 * order)
        pole = warped * complex(math.cos(angle), math.sin(angle))
        z = (2 * fs + pole) / (2 * fs - pole)
        if abs(z.imag) < 1e-12:
            gain = (1 - z.real) / 2
            rows.append([gain, gain, 0, 1, -z.real, 0])
        elif z.imag > 0:
            a1, a2 = -2 * z.real, abs(z) ** 2
            gain = (1 + a1 + a2) / 4
            rows.append([gain, 2 * gain, gain, 1, a1, a2])
    return np.array(rows)


def _resampling_filter(up, down, taps):
    up, down = operator.index(up), operator.index(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be positive integers")
    divisor = math.gcd(up, down)
    up, down = up // divisor, down // divisor
    if taps is None:
        rate = max(up, down)
        taps = np.array([1.0]) if rate == 1 else firwin(20 * rate + 1, 1.0 / rate, window="kaiser")
    taps = _vector(taps)
    if not taps.size or not np.all(np.isfinite(taps)):
        raise ValueError("resampling filter must be nonempty and finite")
    return up, down, taps * up


def resample_poly(x, up, down, *, taps=None):
    """Zero-phase rational resampling with finite-support polyphase FIR filtering.

    The signal is zero outside its extent. Output length is ceil(len(x)*up/down).
    Custom taps are interpreted at the expanded sample rate with unity DC gain.
    """
    x = _vector(x)
    up, down, h = _resampling_filter(up, down, taps)
    out = np.zeros((x.size * up + down - 1) // down, dtype=np.result_type(x, h))
    center = (h.size - 1) // 2
    for k in range(out.size):
        t = center + k * down
        j0 = max(0, (t - h.size + up) // up)
        j1 = min(x.size - 1, t // up)
        if j1 >= j0:
            indices = np.arange(j0, j1 + 1)
            out[k] = np.dot(x[j0:j1 + 1], h[t - indices * up])
    return out


class PolyphaseResampler:
    """Causal rational resampling retaining only FIR overlap between chunks.

    This streaming form includes the FIR group delay (``delay`` in input samples).
    Unlike ``resample_poly``, no future samples are assumed. Append zeros explicitly
    to flush the tail. State is a portable dictionary of counters and overlap.
    """
    def __init__(self, up, down, *, taps=None):
        self.up, self.down, self.taps = _resampling_filter(up, down, taps)
        self.delay = (self.taps.size - 1) / (2 * self.up)
        self.reset()

    def reset(self):
        self._tail = np.zeros(0)
        self._received = self._emitted = 0
        return self

    @property
    def state(self):
        return {"received": self._received, "emitted": self._emitted,
                "tail_real": np.real(self._tail).tolist(), "tail_imag": np.imag(self._tail).tolist()}

    @state.setter
    def state(self, value):
        received, emitted = operator.index(value["received"]), operator.index(value["emitted"])
        real = np.asarray(value["tail_real"])
        imag = np.asarray(value["tail_imag"])
        if (real.ndim != 1 or real.shape != imag.shape or np.iscomplexobj(real) or
                np.iscomplexobj(imag) or not np.all(np.isfinite(real)) or
                not np.all(np.isfinite(imag))):
            raise ValueError("resampler overlap must contain matching finite real vectors")
        required = min(received, (self.taps.size + self.up - 1) // self.up)
        if (received < 0 or emitted != (received * self.up + self.down - 1) // self.down or
                real.size != required):
            raise ValueError("invalid resampler counters or incomplete overlap history")
        tail = real + 1j * imag
        self._received, self._emitted, self._tail = received, emitted, tail

    def process(self, x):
        x = _vector(x)
        start = self._received - self._tail.size
        buffer = np.concatenate((self._tail, x))
        received = self._received + x.size
        target = (received * self.up + self.down - 1) // self.down
        out = np.zeros(target - self._emitted, dtype=np.result_type(buffer, self.taps))
        for k in range(self._emitted, target):
            t = k * self.down
            j0 = max(start, 0, (t - self.taps.size + self.up) // self.up)
            j1 = min(received - 1, t // self.up)
            if j1 >= j0:
                indices = np.arange(j0, j1 + 1)
                out[k - self._emitted] = np.dot(buffer[j0 - start:j1 - start + 1], self.taps[t - indices * self.up])
        keep = min(buffer.size, (self.taps.size + self.up - 1) // self.up)
        self._tail = buffer[-keep:].copy() if keep else buffer[:0].copy()
        self._received, self._emitted = received, target
        return out

    __call__ = process


@dataclass
class STFTResult:
    coefficients: object
    frequencies: object
    times: object
    window: object
    hop: int
    nfft: int
    length: int
    padding: int
    onesided: bool
    fs: float

    def __iter__(self):
        yield self.frequencies
        yield self.times
        yield self.coefficients


def stft(x, *, segment=256, hop=None, nfft=None, window="hann", fs=1.0, boundary=True):
    """Complex STFT coefficients plus metadata for exact overlap-add inversion.

    Coefficients have shape (frequency, frame). Boundary padding and a final
    partial frame are included by default to preserve every input sample.
    Complex windows use the full spectrum even for a real input signal.
    """
    x = _vector(x)
    segment = operator.index(segment)
    hop = max(1, segment // 2) if hop is None else operator.index(hop)
    nfft = segment if nfft is None else operator.index(nfft)
    if segment < 1 or not 1 <= hop <= segment or nfft < segment or not math.isfinite(fs) or fs <= 0:
        raise ValueError("invalid STFT segment, hop, nfft, or sampling frequency")
    win = make_window(segment, window) if isinstance(window, str) else _vector(window)
    if win.shape != (segment,) or not np.all(np.isfinite(win)):
        raise ValueError("window must be finite with segment entries")
    pad = segment // 2 if boundary else 0
    length = x.size + 2 * pad
    frames = max(1, 1 + math.ceil(max(length - segment, 0) / hop)) if x.size else 0
    onesided = not (np.iscomplexobj(x) or np.iscomplexobj(win))
    coefficients = np.empty((nfft // 2 + 1 if onesided else nfft, frames), dtype=complex)
    # One reusable frame avoids materializing a padded or strided frame matrix.
    frame = np.zeros(nfft, dtype=np.result_type(x.dtype, win.dtype))
    for k in range(frames):
        frame.fill(0)
        first = k * hop - pad
        lo, hi = max(0, first), min(x.size, first + segment)
        if hi > lo:
            frame[lo - first:hi - first] = x[lo:hi]
        frame[:segment] *= win
        spectrum = np.fft.fft(frame)
        coefficients[:, k] = spectrum[:coefficients.shape[0]]
    frequencies = np.fft.rfftfreq(nfft, 1 / fs) if onesided else np.fft.fftfreq(nfft, 1 / fs)
    times = (np.arange(frames) * hop - pad + segment / 2) / fs
    return STFTResult(coefficients, frequencies, times, win.copy(), hop, nfft, x.size, pad, onesided, fs)


def istft(result, *, coefficients=None):
    """Reconstruct an STFTResult, optionally with edited complex coefficients.

    Raises if the window/hop leaves any original sample with zero overlap weight.
    """
    if not isinstance(result, STFTResult):
        raise TypeError("istft requires the STFTResult containing synthesis metadata")
    z = np.asarray(result.coefficients if coefficients is None else coefficients, dtype=complex)
    if z.shape != result.coefficients.shape:
        raise ValueError("edited coefficients must preserve STFT shape")
    segment, frames = result.window.size, z.shape[1]
    size = max(result.length + 2 * result.padding, (frames - 1) * result.hop + segment)
    out = np.zeros(size, dtype=float if result.onesided else complex)
    weights = np.zeros(size)
    for k in range(frames):
        frame = np.fft.irfft(z[:, k], n=result.nfft) if result.onesided else np.fft.ifft(z[:, k])
        start = k * result.hop
        out[start:start + segment] += frame[:segment] * np.conj(result.window)
        weights[start:start + segment] += np.abs(result.window) ** 2
    sl = slice(result.padding, result.padding + result.length)
    if np.any(weights[sl] <= 1e-30):
        raise ValueError("window and hop do not satisfy nonzero overlap-add coverage")
    return out[sl] / weights[sl]
