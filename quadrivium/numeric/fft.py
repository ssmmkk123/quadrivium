"""Discrete transforms, mirroring ``numpy.fft`` for what the library uses."""

from __future__ import annotations

from .. import _qnp as _c

fft = _c.fft.fft
ifft = _c.fft.ifft

__all__ = ["fft", "ifft", "fft2", "ifft2", "rfft", "irfft", "fftfreq",
           "rfftfreq", "fftshift", "ifftshift"]


def fft2(a, s=None, axes=(-2, -1)):
    first = fft(a, None if s is None else s[0], axes[0])
    return fft(first, None if s is None else s[1], axes[1])


def ifft2(a, s=None, axes=(-2, -1)):
    first = ifft(a, None if s is None else s[0], axes[0])
    return ifft(first, None if s is None else s[1], axes[1])


def rfft(a, n=None, axis=-1):
    """Transform of real input, keeping the non-redundant half."""
    values = _c.asarray(a)
    length = values.shape[axis] if n is None else n
    full = fft(values, n, axis)
    keep = length // 2 + 1
    key = [slice(None)] * full.ndim
    key[axis] = slice(0, keep)
    return full[tuple(key)]


def irfft(a, n=None, axis=-1):
    """Inverse of :func:`rfft`, rebuilding the conjugate-symmetric half."""
    values = _c.asarray(a, dtype=complex)
    half = values.shape[axis]
    length = 2 * (half - 1) if n is None else n
    key = [slice(None)] * values.ndim
    key[axis] = slice(1, length - half + 1)
    mirror = values[tuple(key)]
    reverse = [slice(None)] * values.ndim
    reverse[axis] = slice(None, None, -1)
    mirror = _c.conjugate(mirror[tuple(reverse)])
    full = _c.concatenate([values, mirror], axis=axis)
    return _c.real(ifft(full, length, axis))


def fftfreq(n, d=1.0):
    """Sample frequencies, negative half last, as NumPy orders them."""
    out = _c.empty(n, dtype=_c.float64)
    positive = (n - 1) // 2 + 1
    out[:positive] = _c.arange(0, positive, dtype=_c.float64)
    out[positive:] = _c.arange(-(n // 2), 0, dtype=_c.float64)
    # NumPy multiplies by the reciprocal rather than dividing; matching that
    # keeps the last bit of every frequency identical.
    return out * (1.0 / (n * d))


def rfftfreq(n, d=1.0):
    return _c.arange(0, n // 2 + 1, dtype=_c.float64) * (1.0 / (n * d))


def fftshift(x, axes=None):
    values = _c.asarray(x)
    axes = range(values.ndim) if axes is None else (
        (axes,) if isinstance(axes, int) else axes)
    for axis in axes:
        values = _c.roll(values, values.shape[axis] // 2, axis=axis)
    return values


def ifftshift(x, axes=None):
    values = _c.asarray(x)
    axes = range(values.ndim) if axes is None else (
        (axes,) if isinstance(axes, int) else axes)
    for axis in axes:
        values = _c.roll(values, -(values.shape[axis] // 2), axis=axis)
    return values
