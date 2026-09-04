"""Discrete transforms: the DFT and its fast algorithms.

The direct DFT costs ``O(n^2)``; the Cooley-Tukey factorization brings that to
``O(n log n)``, and Bluestein's algorithm extends the speedup to lengths with
no small factors.
"""

from __future__ import annotations

import numpy as np

from .. import _accel

from ..core.utils import as_vector

__all__ = [
    "dft",
    "idft",
    "dft_matrix",
    "fft",
    "ifft",
    "fft_radix2",
    "fft_bluestein",
    "fft_mixed_radix",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "fftfreq",
    "fftshift",
    "dct",
    "idct",
    "dst",
    "hartley",
    "next_power_of_two",
    "bit_reverse_permutation",
]


def next_power_of_two(n: int) -> int:
    """Smallest power of two at least ``n``."""
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def dft_matrix(n: int, inverse: bool = False):
    """Explicit DFT matrix ``W`` with ``W[j,k] = exp(-2 pi i j k / n)``."""
    j, k = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sign = 1.0 if inverse else -1.0
    W = np.exp(sign * 2j * np.pi * j * k / n)
    return W / n if inverse else W


def dft(x):
    """Direct discrete Fourier transform, ``O(n^2)``.

    Kept for reference and for testing the fast algorithms against.
    """
    x = np.asarray(x, dtype=complex)
    n = x.size
    return dft_matrix(n) @ x


def idft(X):
    """Direct inverse discrete Fourier transform."""
    X = np.asarray(X, dtype=complex)
    n = X.size
    return dft_matrix(n, inverse=True) @ X


def bit_reverse_permutation(n: int):
    """Bit-reversal permutation used by the iterative radix-2 FFT."""
    bits = int(n).bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=int)
    for i in range(bits):
        rev |= ((idx >> i) & 1) << (bits - 1 - i)
    return rev


def fft_radix2(x, inverse: bool = False):
    """Iterative radix-2 Cooley-Tukey FFT. Length must be a power of two.

    Works in place over a bit-reversed copy, doubling the transform size each
    stage -- the classic decimation-in-time formulation.
    """
    a = np.asarray(x, dtype=complex).copy()
    n = a.size
    if n & (n - 1):
        raise ValueError(f"radix-2 FFT needs a power-of-two length, got {n}")
    if n == 1:
        return a
    a = a[bit_reverse_permutation(n)]
    sign = 1.0 if inverse else -1.0
    size = 2
    while size <= n:
        half = size // 2
        w = np.exp(sign * 2j * np.pi * np.arange(half) / size)
        for start in range(0, n, size):
            # `even` must be a copy: assigning to a[start:start+half] below
            # would otherwise overwrite it before the second line reads it.
            even = a[start : start + half].copy()
            odd = a[start + half : start + size] * w
            a[start : start + half] = even + odd
            a[start + half : start + size] = even - odd
        size *= 2
    return a / n if inverse else a


def fft_bluestein(x, inverse: bool = False):
    """Bluestein's chirp-z algorithm: an FFT for any length.

    Rewrites the DFT as a convolution, which is then evaluated with a
    power-of-two FFT -- so prime lengths still cost ``O(n log n)``.
    """
    a = np.asarray(x, dtype=complex)
    n = a.size
    if n <= 1:
        return a.copy()
    sign = 1.0 if inverse else -1.0
    m = next_power_of_two(2 * n - 1)
    k = np.arange(n)
    chirp = np.exp(sign * 1j * np.pi * k * k / n)
    A = np.zeros(m, dtype=complex)
    A[:n] = a * chirp
    B = np.zeros(m, dtype=complex)
    B[:n] = np.conj(chirp)
    B[m - n + 1 :] = np.conj(chirp[1:][::-1])
    conv = fft_radix2(fft_radix2(A) * fft_radix2(B), inverse=True)
    out = conv[:n] * chirp
    return out / n if inverse else out


def fft_mixed_radix(x, inverse: bool = False):
    """Recursive mixed-radix FFT: splits on the smallest prime factor.

    Falls back to Bluestein when the remaining length is prime.
    """
    a = np.asarray(x, dtype=complex)
    n = a.size
    if n <= 1:
        return a.copy()
    # find the smallest prime factor
    p = None
    for q in range(2, int(np.sqrt(n)) + 1):
        if n % q == 0:
            p = q
            break
    if p is None:
        return fft_bluestein(a, inverse)
    m = n // p
    sign = 1.0 if inverse else -1.0
    # decimation in time by p
    sub = np.array([fft_mixed_radix(a[r::p], inverse=False) for r in range(p)])
    out = np.zeros(n, dtype=complex)
    for k in range(n):
        acc = 0.0
        for r in range(p):
            twiddle = np.exp(sign * 2j * np.pi * r * k / n)
            acc += twiddle * sub[r, k % m]
        out[k] = acc
    return out / n if inverse else out


def fft(x):
    """Fast Fourier transform for any length.

    Dispatches to the radix-2 algorithm when possible and to Bluestein's
    otherwise, so no length is penalized with ``O(n^2)`` work.
    """
    x = np.asarray(x, dtype=complex)
    n = x.size
    if n == 0:
        return x.copy()
    fast = _accel.kernel("fft")
    if fast is not None:
        return fast(np.ascontiguousarray(x))
    if n & (n - 1) == 0:
        return fft_radix2(x)
    return fft_bluestein(x)


def ifft(X):
    """Inverse fast Fourier transform for any length."""
    X = np.asarray(X, dtype=complex)
    n = X.size
    if n == 0:
        return X.copy()
    fast = _accel.kernel("ifft")
    if fast is not None:
        return fast(np.ascontiguousarray(X))
    if n & (n - 1) == 0:
        return fft_radix2(X, inverse=True)
    return fft_bluestein(X, inverse=True)


def rfft(x):
    """FFT of real input, returning the non-redundant half spectrum.

    A real signal has a Hermitian spectrum, so only ``n//2 + 1`` values carry
    information.
    """
    x = np.asarray(x, dtype=float)
    return fft(x)[: x.size // 2 + 1]


def irfft(X, n=None):
    """Inverse of :func:`rfft` for a real signal of length ``n``."""
    X = np.asarray(X, dtype=complex)
    if n is None:
        n = 2 * (X.size - 1)
    full = np.zeros(n, dtype=complex)
    full[: X.size] = X
    # rebuild the redundant half by Hermitian symmetry
    full[X.size :] = np.conj(X[1 : n - X.size + 1][::-1])
    return np.real(ifft(full))


def fft2(A):
    """Two-dimensional FFT: transform the rows, then the columns."""
    A = np.asarray(A, dtype=complex)
    out = np.array([fft(row) for row in A])
    return np.array([fft(col) for col in out.T]).T


def ifft2(A):
    """Two-dimensional inverse FFT."""
    A = np.asarray(A, dtype=complex)
    out = np.array([ifft(row) for row in A])
    return np.array([ifft(col) for col in out.T]).T


def fftfreq(n: int, d: float = 1.0):
    """Frequencies matching the output ordering of :func:`fft`."""
    val = 1.0 / (n * d)
    results = np.empty(n, dtype=int)
    half = (n - 1) // 2 + 1
    results[:half] = np.arange(half)
    results[half:] = np.arange(-(n // 2), 0)
    return results * val


def fftshift(x):
    """Shift the zero frequency to the centre of the spectrum."""
    x = np.asarray(x)
    n = x.shape[0]
    return np.roll(x, n // 2, axis=0)


def dct(x, kind: int = 2, norm: bool = False):
    """Discrete cosine transform, types 1-4 (unnormalized conventions).

    Types 1 and 2 are evaluated through the FFT of a symmetric extension;
    types 3 and 4 are formed directly. ``idct`` supplies the matching inverse
    scaling.
    """
    x = as_vector(x)
    n = x.size
    if kind == 1:
        ext = np.concatenate([x, x[-2:0:-1]])
        out = np.real(fft(ext))[:n]
    elif kind == 2:
        ext = np.zeros(4 * n)
        ext[1::2][:n] = x            # odd samples carry the signal
        ext[2 * n + 1 :: 2] = x[::-1]
        out = np.real(fft(ext))[:n]
    elif kind == 3:
        k = np.arange(n)
        out = np.array([x[0] / 2 + np.sum(x[1:] * np.cos(np.pi * np.arange(1, n)
                                                         * (2 * j + 1) / (2 * n)))
                        for j in k]) * 2
    elif kind == 4:
        j = np.arange(n)
        out = np.array([2 * np.sum(x * np.cos(np.pi * (2 * j + 1) * (2 * m + 1)
                                              / (4 * n))) for m in range(n)])
    else:
        raise ValueError("kind must be 1, 2, 3 or 4")
    if norm:
        out = out / np.sqrt(2 * n)
    return out


def idct(X, kind: int = 2):
    """Inverse discrete cosine transform.

    With the unnormalized conventions used by :func:`dct`, type 2 and type 3
    are inverses of each other up to ``1/(2N)``, while types 1 and 4 are their
    own inverses up to ``1/(2(N-1))`` and ``1/(2N)`` respectively.
    """
    X = as_vector(X)
    n = X.size
    if kind == 1:
        return dct(X, 1) / (2 * (n - 1))
    if kind == 2:
        return dct(X, 3) / (2 * n)
    if kind == 3:
        return dct(X, 2) / (2 * n)
    if kind == 4:
        return dct(X, 4) / (2 * n)
    raise ValueError("kind must be 1, 2, 3 or 4")


def dst(x, kind: int = 1):
    """Discrete sine transform, types 1 and 2."""
    x = as_vector(x)
    n = x.size
    if kind == 1:
        # odd extension of length 2(n+1); its FFT is purely imaginary
        ext = np.zeros(2 * n + 2, dtype=complex)
        ext[1 : n + 1] = x
        ext[n + 2 :] = -x[::-1]
        return -np.imag(fft(ext))[1 : n + 1]
    if kind == 2:
        j = np.arange(n)
        return np.array([2 * np.sum(x * np.sin(np.pi * (2 * j + 1) * (m + 1)
                                               / (2 * n))) for m in range(n)])
    raise ValueError("kind must be 1 or 2")


def hartley(x):
    """Discrete Hartley transform: a real-valued relative of the DFT."""
    X = fft(np.asarray(x, dtype=float))
    return np.real(X) - np.imag(X)
