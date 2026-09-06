"""Pseudorandom and low-discrepancy number generators.

Implemented explicitly so the mechanics -- and the failure modes of the older
designs -- are visible rather than hidden behind a library call.
"""

from __future__ import annotations

from .. import numeric as np

__all__ = [
    "LCG",
    "ParkMiller",
    "XorShift",
    "MersenneTwister",
    "middle_square",
    "halton",
    "sobol",
    "latin_hypercube_sample",
    "van_der_corput",
    "spectral_test",
    "spectral_test_2d",
]


class LCG:
    """Linear congruential generator ``x <- (a x + c) mod m``.

    The default constants are those of ``glibc``. LCGs are fast but their
    points fall on a small number of hyperplanes -- see :func:`spectral_test_2d`.
    """

    def __init__(self, seed: int = 1, a: int = 1103515245, c: int = 12345,
                 m: int = 2**31):
        self.a, self.c, self.m = int(a), int(c), int(m)
        self.state = int(seed) % self.m

    def next_int(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random(self, size=None):
        """Uniform samples on ``[0, 1)``."""
        if size is None:
            return self.next_int() / self.m
        n = int(np.prod(size)) if np.ndim(size) else int(size)
        out = np.array([self.next_int() / self.m for _ in range(n)])
        return out.reshape(size) if np.ndim(size) else out


class ParkMiller(LCG):
    """Park-Miller minimal standard generator: ``a = 16807``, ``m = 2^31 - 1``."""

    def __init__(self, seed: int = 1):
        super().__init__(seed=seed or 1, a=16807, c=0, m=2**31 - 1)


class XorShift:
    """Marsaglia's xorshift generator: fast, long period, tiny state."""

    def __init__(self, seed: int = 88172645463325252):
        self.state = int(seed) & 0xFFFFFFFFFFFFFFFF or 88172645463325252

    def next_int(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x
        return x

    def random(self, size=None):
        if size is None:
            return self.next_int() / 2**64
        n = int(np.prod(size)) if np.ndim(size) else int(size)
        out = np.array([self.next_int() / 2**64 for _ in range(n)])
        return out.reshape(size) if np.ndim(size) else out


class MersenneTwister:
    """MT19937: period ``2^19937 - 1`` and 623-dimensional equidistribution.

    The standard general-purpose generator for simulation (though not
    cryptographically secure).
    """

    N, M = 624, 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int = 5489):
        self.mt = [0] * self.N
        self.index = self.N + 1
        self.seed(seed)

    def seed(self, s: int) -> None:
        self.mt[0] = int(s) & 0xFFFFFFFF
        for i in range(1, self.N):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.index = self.N

    def _generate(self) -> None:
        for i in range(self.N):
            y = ((self.mt[i] & self.UPPER_MASK)
                 | (self.mt[(i + 1) % self.N] & self.LOWER_MASK))
            nxt = self.mt[(i + self.M) % self.N] ^ (y >> 1)
            if y % 2:
                nxt ^= self.MATRIX_A
            self.mt[i] = nxt
        self.index = 0

    def next_int(self) -> int:
        if self.index >= self.N:
            self._generate()
        y = self.mt[self.index]
        self.index += 1
        # tempering
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def random(self, size=None):
        if size is None:
            return self.next_int() / 2**32
        n = int(np.prod(size)) if np.ndim(size) else int(size)
        out = np.array([self.next_int() / 2**32 for _ in range(n)])
        return out.reshape(size) if np.ndim(size) else out


def middle_square(seed: int, n: int, digits: int = 4):
    """Von Neumann's middle square method.

    Included as a cautionary example: it degenerates to zero or short cycles
    very quickly and must not be used for real work.
    """
    out = []
    x = int(seed)
    mod = 10**digits
    for _ in range(n):
        x = (x * x // 10 ** (digits // 2)) % mod
        out.append(x / mod)
    return np.array(out)


def van_der_corput(n: int, base: int = 2):
    """First ``n`` elements of the van der Corput low-discrepancy sequence."""
    # One base-b digit per pass over the whole index range, rather than one
    # index at a time: the digit count is log_b(n), so this is a handful of
    # vector operations however long the sequence is.
    out = np.zeros(n)
    k = np.arange(1, n + 1, dtype=np.int64)
    weight = 1.0 / base
    while k.any():
        k, rem = np.divmod(k, base)
        out += rem * weight
        weight /= base
    return out


def halton(n: int, dim: int = 1, skip: int = 1):
    """Halton low-discrepancy sequence."""
    from ..integrate.monte_carlo import halton_sequence

    return halton_sequence(n, dim, skip)


def sobol(n: int, dim: int = 1):
    """Sobol low-discrepancy sequence."""
    from ..integrate.monte_carlo import sobol_sequence

    return sobol_sequence(n, dim)


def latin_hypercube_sample(n: int, dim: int = 1, rng=None):
    """Latin hypercube sample."""
    from ..integrate.monte_carlo import latin_hypercube

    return latin_hypercube(n, dim, rng)


def spectral_test(generator, n: int = 2000, dim: int = 3, max_coeff: int = 10,
                  tol: float = 1e-6):
    """Search for a short dual-lattice vector in consecutive ``dim``-tuples.

    Successive outputs of a linear congruential generator satisfy an exact
    integer recurrence, so some small integer vector ``a`` makes
    ``a . (x_i, ..., x_{i+dim-1})`` an integer for *every* tuple. This searches
    the small integer vectors and reports the one whose combination stays
    closest to an integer.

    Returns a dict with the best ``coefficients``, the worst-case
    ``deviation`` from an integer, and ``lattice_detected``. RANDU
    (``a = 65539``, ``m = 2^31``) is found immediately with ``(9, -6, 1)``,
    since its triples lie on just 15 planes; a good generator leaves every
    small vector uniformly spread, giving a deviation near ``0.5``.
    """
    import itertools

    u = np.array([generator.random() for _ in range(n + dim)])
    pts = np.column_stack([u[i : i + n] for i in range(dim)])
    best = {"coefficients": None, "deviation": 0.5, "lattice_detected": False}
    rng_coeffs = range(-max_coeff, max_coeff + 1)
    for a in itertools.product(rng_coeffs, repeat=dim):
        if all(c == 0 for c in a):
            continue
        if next(c for c in a if c != 0) < 0:
            continue                      # skip sign-flipped duplicates
        z = np.mod(pts @ np.array(a, dtype=float), 1.0)
        dev = float(np.max(np.minimum(z, 1.0 - z)))
        if dev < best["deviation"]:
            best = {"coefficients": np.array(a), "deviation": dev,
                    "lattice_detected": dev < tol}
        if best["lattice_detected"]:
            break
    return best


def spectral_test_2d(generator, n: int = 5000, **kwargs):
    """Two-dimensional dual-lattice search (see :func:`spectral_test`)."""
    return spectral_test(generator, n=n, dim=2, **kwargs)
