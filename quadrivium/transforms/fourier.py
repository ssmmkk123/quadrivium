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
    sign = 1.0 if inverse else -1.0
    # The outer product of the index ranges is the exponent table directly;
    # meshgrid would first materialize two full n-by-n index arrays for it.
    idx = np.arange(n)
    W = np.outer(idx, idx) * (sign * 2j * np.pi / n)
    np.exp(W, out=W)
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
    a = np.atleast_1d(np.asarray(x, dtype=complex))
    if a.ndim != 1:
        raise ValueError("FFT input must be one-dimensional")
    n = a.size
    if n & (n - 1):
        raise ValueError(f"radix-2 FFT needs a power-of-two length, got {n}")
    if n <= 1:
        return a.copy()
    a = a[bit_reverse_permutation(n)]
    sign = 1.0 if inverse else -1.0
    # All stages draw from one twiddle table. Reuse one work array instead
    # of allocating a pair for each of the n-1 butterfly groups.
    roots = np.arange(n // 2, dtype=complex)
    roots *= sign * 2j * np.pi / n
    np.exp(roots, out=roots)
    odd_work = np.empty(n // 2, dtype=complex)
    size = 2
    while size <= n:
        half = size // 2
        blocks = a.reshape(-1, size)
        odd = odd_work.reshape(-1, half)
        np.multiply(blocks[:, half:], roots[:: n // size], out=odd)
        # Write the odd half first, while the original even half is intact.
        np.subtract(blocks[:, :half], odd, out=blocks[:, half:])
        np.add(blocks[:, :half], odd, out=blocks[:, :half])
        size *= 2
    if inverse:
        a /= n
    return a


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
    """Recursive mixed-radix FFT with a radix-2 base case.

    Small odd factors are split first to leave power-of-two subtransforms.
    Falls back to Bluestein when the remaining length is prime.
    """
    a = np.asarray(x, dtype=complex)
    n = a.size
    if n <= 1:
        return a.copy()
    if inverse:
        # Conjugation keeps every recursive subtransform in the same forward
        # convention; changing only the final twiddle signs is not an inverse.
        return np.conj(fft_mixed_radix(np.conj(a))) / n
    if n & (n - 1) == 0:
        return fft_radix2(a)
    # Peel common odd radices before 2: e.g. 3 * 512 needs three radix-2
    # transforms, rather than a recursion tree of 512 length-3 transforms.
    p = next((q for q in (3, 5, 7) if n % q == 0 and n != q), None)
    if p is None:
        for q in range(2, int(np.sqrt(n)) + 1):
            if n % q == 0:
                p = q
                break
    if p is None:
        return fft_bluestein(a, inverse)
    m = n // p
    # decimation in time by p
    out = np.zeros(n, dtype=complex)
    angles = np.arange(n, dtype=complex)
    angles *= -2j * np.pi / n
    twiddle = np.empty(n, dtype=complex)
    for r in range(p):
        sub = fft_mixed_radix(a[r::p])
        np.multiply(angles, r, out=twiddle)
        np.exp(twiddle, out=twiddle)
        twiddle.reshape(p, m)[:] *= sub
        out += twiddle
    return out


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
    # Do not retain the redundant half through the result's base array.
    return fft(x)[: x.size // 2 + 1].copy()


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
        # cos(pi k (2j+1) / 2n) = Re[ e^{i pi k /2n} e^{i 2 pi k j / 2n} ], so
        # pre-twiddling the input turns the sum into one length-2n inverse
        # DFT -- O(n log n) instead of the O(n^2) the definition spells out.
        w = np.zeros(2 * n, dtype=complex)
        c = np.full(n, 2.0)
        c[0] = 1.0
        w[:n] = c * x * np.exp(1j * np.pi * np.arange(n) / (2 * n))
        out = np.real(ifft(w))[:n] * (2 * n)
    elif kind == 4:
        # Same idea with a half-sample shift on both indices, which leaves a
        # twiddle on the output as well as the input.
        v = np.zeros(2 * n, dtype=complex)
        v[:n] = x * np.exp(1j * np.pi * np.arange(n) / (2 * n))
        m = np.arange(n)
        out = np.real(2 * np.exp(1j * np.pi * (2 * m + 1) / (4 * n))
                      * (ifft(v)[:n] * (2 * n)))
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
        # The imaginary counterpart of DCT-III: one length-2n inverse DFT of
        # the zero-padded signal, with the half-sample shift applied as an
        # output twiddle. O(n log n) rather than the definition's O(n^2).
        v = np.zeros(2 * n, dtype=complex)
        v[:n] = x
        m = np.arange(n)
        return np.imag(2 * np.exp(1j * np.pi * (m + 1) / (2 * n))
                       * (ifft(v) * (2 * n))[1:n + 1])
    raise ValueError("kind must be 1 or 2")


def hartley(x):
    """Discrete Hartley transform: a real-valued relative of the DFT."""
    X = fft(np.asarray(x, dtype=float))
    return np.real(X) - np.imag(X)
