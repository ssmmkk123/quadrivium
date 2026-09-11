"""Stateful randomized digital nets and reproducible child random streams.

Direction numbers are constructed from primitive binary polynomials. The first
dimensions use the package's historical initial directions; later dimensions
use deterministic odd initial values, rather than substituting another sequence.
They are not the optimized Joe--Kuo direction tables. Scrambling applies a random
invertible lower-triangular binary matrix followed by a digital shift.
"""
from __future__ import annotations

import hashlib
import math
import operator
from functools import lru_cache

from .. import numeric as np
from ..core import QuadratureResult

__all__ = ["Sobol", "spawn_rngs", "randomized_qmc"]


def _child_seed(seed, index):
    payload = f"quadrivium-stream-v1:{operator.index(seed)}:{index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "little")


def spawn_rngs(seed, n):
    """Return ``n`` reproducible child streams independent of scheduling order."""
    n = operator.index(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return [np.random.default_rng(_child_seed(seed, i)) for i in range(n)]


def _multiply(a, b, polynomial):
    top = 1 << (polynomial.bit_length() - 1)
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & top:
            a ^= polynomial
    return result


def _power(a, n, polynomial):
    result = 1
    while n:
        if n & 1:
            result = _multiply(result, a, polynomial)
        a = _multiply(a, a, polynomial)
        n >>= 1
    return result


def _prime_factors(n):
    factors, p = [], 2
    while p * p <= n:
        if n % p == 0:
            factors.append(p)
            while n % p == 0:
                n //= p
        p += 1
    if n > 1:
        factors.append(n)
    return factors


@lru_cache(maxsize=8)
def _polynomials(n):
    result, degree = [], 1
    while len(result) < n:
        period = (1 << degree) - 1
        factors = _prime_factors(period)
        for polynomial in range((1 << degree) | 1, 1 << (degree + 1), 2):
            x = 2 if degree > 1 else 1
            if (_power(x, period, polynomial) == 1 and
                    all(_power(x, period // p, polynomial) != 1 for p in factors)):
                result.append(polynomial)
                if len(result) == n:
                    break
        degree += 1
    return tuple(result)


class Sobol:
    """Incremental Sobol engine with bounded state and optional LMS scrambling.

    ``random_base2(m)`` draws 2**m points and enforces a power-of-two cumulative
    sample count. ``random(n)`` permits arbitrary sizes but loses that balance
    guarantee. ``state`` is a JSON-serializable checkpoint. Up to 1024 dimensions
    and 52 direction bits are supported; memory is O(dim * bits).
    """
    def __init__(self, dim, *, scramble=True, seed=0, bits=30):
        self.dim, self.bits = operator.index(dim), operator.index(bits)
        if not 1 <= self.dim <= 1024 or not 1 <= self.bits <= 52:
            raise ValueError("require 1 <= dim <= 1024 and 1 <= bits <= 52")
        self.scramble, self.seed = bool(scramble), operator.index(seed)
        self.num_generated = 0
        directions = []
        initial = [[1], [1, 3], [1, 1, 5], [1, 3, 1], [1, 1, 3, 7]]
        polys = _polynomials(self.dim - 1) if self.dim > 1 else ()
        for dimension in range(self.dim):
            if dimension == 0:
                row = [1 << (self.bits - 1 - k) for k in range(self.bits)]
            else:
                poly = polys[dimension - 1]
                degree = poly.bit_length() - 1
                row = []
                for k in range(min(degree, self.bits)):
                    if dimension <= len(initial) and k < len(initial[dimension - 1]):
                        m = initial[dimension - 1][k]
                    else:
                        m = (_child_seed(dimension, k) % (1 << (k + 1))) | 1
                    row.append(m << (self.bits - k - 1))
                for k in range(degree, self.bits):
                    value = row[k - degree] ^ (row[k - degree] >> degree)
                    for j in range(1, degree):
                        if (poly >> (degree - j)) & 1:
                            value ^= row[k - j]
                    row.append(value)
            directions.append(row)
        self._shift = [0] * self.dim
        if scramble:
            rng = np.random.default_rng(seed)
            for d, row in enumerate(directions):
                self._shift[d] = int(rng.integers(0, 1 << self.bits))
                # Each output digit depends on itself and preceding digits.
                masks = [(int(rng.integers(0, 1 << k)) << (self.bits - k)) |
                         (1 << (self.bits - k - 1)) for k in range(self.bits)]
                directions[d] = [sum(((bin(v & mask).count("1") & 1) <<
                                      (self.bits - k - 1))
                                     for k, mask in enumerate(masks)) for v in row]
        self._directions = directions

    @property
    def state(self):
        return {"version": 1, "dim": self.dim, "bits": self.bits,
                "scramble": self.scramble, "seed": self.seed,
                "num_generated": self.num_generated}

    @classmethod
    def from_state(cls, state):
        if state.get("version") != 1:
            raise ValueError("unsupported Sobol state version")
        engine = cls(state["dim"], bits=state["bits"],
                     scramble=state["scramble"], seed=state["seed"])
        return engine.fast_forward(state["num_generated"])

    def reset(self):
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        n = operator.index(n)
        if n < 0 or self.num_generated + n > 1 << self.bits:
            raise ValueError("Sobol position exceeds available direction bits")
        self.num_generated += n
        return self

    def random(self, n=1):
        n = operator.index(n)
        if n < 0 or self.num_generated + n > 1 << self.bits:
            raise ValueError("Sobol sample count exceeds available direction bits")
        out = np.empty((n, self.dim))
        index = self.num_generated
        gray = index ^ (index >> 1)
        state = self._shift.copy()
        for d, directions in enumerate(self._directions):
            for bit in range(self.bits):
                if gray & (1 << bit):
                    state[d] ^= directions[bit]
        scale = 1.0 / (1 << self.bits)
        for i in range(n):
            for d in range(self.dim):
                out[i, d] = state[d] * scale
            index += 1
            if i + 1 < n:
                bit = (index & -index).bit_length() - 1
                for d in range(self.dim):
                    state[d] ^= self._directions[d][bit]
        self.num_generated += n
        return out

    def random_base2(self, m):
        m = operator.index(m)
        if m < 0:
            raise ValueError("m must be nonnegative")
        n = 1 << m
        total = self.num_generated + n
        if total & (total - 1):
            raise ValueError("cumulative Sobol sample count must be a power of two")
        return self.random(n)

    def spawn(self, n):
        n = operator.index(n)
        if n < 0:
            raise ValueError("n must be nonnegative")
        return [Sobol(self.dim, bits=self.bits, scramble=True,
                      seed=_child_seed(self.seed, i)) for i in range(n)]


def randomized_qmc(f, lows, highs, *, m=10, replicates=8, seed=0, batch_size=256):
    """Integrate using independent scrambled nets; estimate replicate standard error.

    Integrands accept a scalar in one dimension or a point in higher dimensions.
    Points and values are consumed in bounded batches; only replicate estimates
    are retained. Error is a sampling estimate, not a certified bound.
    """
    lows, highs = np.asarray(lows, dtype=float).ravel(), np.asarray(highs, dtype=float).ravel()
    m, replicates, batch_size = map(operator.index, (m, replicates, batch_size))
    if (lows.shape != highs.shape or not lows.size or not np.all(highs > lows) or
            not np.all(np.isfinite(lows)) or not np.all(np.isfinite(highs))):
        raise ValueError("bounds must have matching nonempty shapes with highs > lows")
    if m < 0 or m > 52 or replicates < 2 or batch_size < 1:
        raise ValueError("require 0 <= m <= 52, replicates >= 2, batch_size >= 1")
    widths = highs - lows
    volume, n = float(np.prod(widths)), 1 << m
    if not np.all(np.isfinite(widths)) or not math.isfinite(volume) or volume <= 0:
        raise ValueError("integration volume must be representable in float64")
    estimates = []
    for r in range(replicates):
        engine = Sobol(lows.size, seed=_child_seed(seed, r), bits=max(m, 30))
        mean, count = 0.0, 0
        for start in range(0, n, batch_size):
            points = lows + engine.random(min(batch_size, n - start)) * (highs - lows)
            for p in points:
                value = float(f(p[0] if lows.size == 1 else p))
                if not math.isfinite(value):
                    raise ValueError("randomized QMC requires finite integrand values")
                count += 1
                mean += value / count - mean / count
        estimate = volume * mean
        if not math.isfinite(estimate):
            raise ValueError("integral estimate is not representable in float64")
        estimates.append(estimate)
    values = np.array(estimates)
    mean = math.fsum(value / replicates for value in estimates)
    scale = max(abs(value) for value in estimates)
    error = (scale * math.sqrt(math.fsum((value / scale - mean / scale)**2
                                       for value in estimates) / (replicates-1) / replicates)
             if scale else 0.0)
    if not math.isfinite(error):
        raise ValueError("sampling error is not representable in float64")
    result = QuadratureResult(mean, error,
                              n * replicates, replicates, True, "randomized_qmc")
    result.replicates = values
    return result
