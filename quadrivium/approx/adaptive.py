"""Piecewise, tolerance-driven Chebyshev representations of scalar functions."""
from __future__ import annotations
from dataclasses import dataclass
import math
import operator
from .. import numeric as np
from ..diff.spectral import chebyshev_coefficients, clenshaw

__all__ = ["ChebyshevApproximation", "chebfun"]


@dataclass
class _Piece:
    a: float
    b: float
    coefficients: object
    error: float

    def __call__(self, x):
        return clenshaw(self.coefficients, (2 * x - self.a - self.b) / (self.b - self.a))


def _differentiate(c):
    n = len(c) - 1
    if n < 1:
        return np.zeros(1)
    d = np.zeros(n)
    d[n - 1] = 2 * n * c[n]
    if n > 1:
        d[n - 2] = 2 * (n - 1) * c[n - 1]
    for k in range(n - 3, -1, -1):
        d[k] = d[k + 2] + 2 * (k + 1) * c[k + 1]
    d[0] *= 0.5
    return d


def _antiderivative(c):
    b = np.zeros(len(c) + 1)
    b[1] = c[0]
    if len(c) > 1:
        b[2] = c[1] / 4
    for k in range(2, len(c)):
        b[k + 1] += c[k] / (2 * (k + 1))
        b[k - 1] -= c[k] / (2 * (k - 1))
    return b


class ChebyshevApproximation:
    """Adaptive scalar approximation with evaluation, derivatives, integrals, roots.

    Degree doubling is checked using both coefficient tails and independent
    off-grid samples. Difficult pieces are bisected. ``converged`` reports
    whether the requested sample-based error target was reached; the estimate
    is not a proof for arbitrary functions between samples.
    """
    def __init__(self, f, domain=(-1.0, 1.0), *, atol=1e-12, rtol=1e-10,
                 max_degree=128, max_pieces=256):
        a, b = map(float, domain)
        max_degree, max_pieces = operator.index(max_degree), operator.index(max_pieces)
        if not math.isfinite(a + b) or b <= a or atol < 0 or rtol < 0 or not math.isfinite(atol + rtol) or atol + rtol <= 0 or max_degree < 16 or max_pieces < 1:
            raise ValueError("invalid domain, tolerances, or approximation budgets")
        self.domain, self.function_calls = (a, b), 0
        self.converged = True
        self.pieces = []
        def counted(x):
            value = float(f(float(x)))
            self.function_calls += 1
            if not math.isfinite(value):
                raise ValueError("approximation requires finite scalar function values")
            return value
        pending = [(a, b)]
        while pending:
            left, right = pending.pop()
            degree = 16
            while True:
                coefficients = chebyshev_coefficients(counted, degree, left, right)
                scale = float(np.sum(np.abs(coefficients)))
                threshold = atol + rtol * scale
                tail = float(np.sum(np.abs(coefficients[-max(4, degree // 8):])))
                # Non-Lobatto probes catch common aliases of oscillatory functions.
                probes = left + (right - left) * (np.arange(13) + 0.3819660112501051) / 13
                estimate = clenshaw(coefficients, (2 * probes - left - right) / (right - left))
                error = max(tail, max(abs(counted(x) - float(y)) for x, y in zip(probes, estimate)))
                if error <= threshold or degree >= max_degree:
                    break
                degree = min(degree * 2, max_degree)
            if error > threshold and len(self.pieces) + len(pending) + 1 < max_pieces and (left + right) / 2 not in (left, right):
                mid = (left + right) / 2
                pending.extend([(mid, right), (left, mid)])
                continue
            self.converged &= error <= threshold
            cutoff = len(coefficients)
            budget, removed = threshold * 0.1, 0.0
            while cutoff > 1 and removed + abs(float(coefficients[cutoff - 1])) <= budget:
                removed += abs(float(coefficients[cutoff - 1]))
                cutoff -= 1
            self.pieces.append(_Piece(left, right, coefficients[:cutoff].copy(), error + removed))
        self.pieces.sort(key=lambda p: p.a)
        self.error_estimate = max(p.error for p in self.pieces)

    @classmethod
    def _from_pieces(cls, pieces, domain):
        obj = cls.__new__(cls)
        obj.pieces, obj.domain = pieces, domain
        obj.converged, obj.function_calls, obj.error_estimate = True, 0, None
        return obj

    def __call__(self, x):
        values = np.asarray(x, dtype=float)
        if np.any(values < self.domain[0]) or np.any(values > self.domain[1]) or not np.all(np.isfinite(values)):
            raise ValueError("evaluation points must lie in the approximation domain")
        flat, out = values.ravel(), np.empty(values.size)
        edges = np.array([p.b for p in self.pieces])
        indices = np.minimum(np.searchsorted(edges, flat), len(self.pieces) - 1)
        for i, piece in enumerate(self.pieces):
            mask = indices == i
            if np.any(mask):
                out[mask] = piece(flat[mask])
        return float(out[0]) if values.ndim == 0 else out.reshape(values.shape)

    def derivative(self, order=1):
        order = operator.index(order)
        if order < 0:
            raise ValueError("derivative order must be nonnegative")
        pieces = []
        for piece in self.pieces:
            c = piece.coefficients.copy()
            for _ in range(order):
                c = _differentiate(c) * (2 / (piece.b - piece.a))
            pieces.append(_Piece(piece.a, piece.b, c, math.nan))
        result = self._from_pieces(pieces, self.domain)
        result.converged = self.converged
        return result

    def integrate(self, a=None, b=None):
        a = self.domain[0] if a is None else float(a)
        b = self.domain[1] if b is None else float(b)
        sign = 1
        if b < a:
            a, b, sign = b, a, -1
        if a < self.domain[0] or b > self.domain[1] or not math.isfinite(a + b):
            raise ValueError("integration limits must lie in the approximation domain")
        terms = []
        for piece in self.pieces:
            left, right = max(a, piece.a), min(b, piece.b)
            if right > left:
                c = _antiderivative(piece.coefficients)
                tl = (2 * left - piece.a - piece.b) / (piece.b - piece.a)
                tr = (2 * right - piece.a - piece.b) / (piece.b - piece.a)
                terms.append((piece.b - piece.a) / 2 * float(clenshaw(c, tr) - clenshaw(c, tl)))
        return sign * math.fsum(terms)

    def roots(self, tol=1e-8):
        """Real roots of the represented pieces, deduplicated at shared boundaries."""
        if tol <= 0:
            raise ValueError("tol must be positive")
        roots = []
        for piece in self.pieces:
            c = piece.coefficients
            n = len(c) - 1
            if n == 0:
                if c[0] == 0:
                    raise ValueError("an identically zero piece has infinitely many roots")
                continue
            if n == 1:
                values = [-c[0] / c[1]]
            else:
                colleague = np.zeros((n, n))
                for k in range(n - 1):
                    colleague[k, k + 1] = 0.5
                    colleague[k + 1, k] = 0.5
                colleague[1, 0] = 1.0
                colleague[:, -1] -= c[:-1] / (2 * c[-1])
                values = np.linalg.eigvals(colleague)
            for value in values:
                z = complex(value)
                if abs(z.imag) <= tol and -1 - tol <= z.real <= 1 + tol:
                    t = min(1.0, max(-1.0, z.real))
                    roots.append((piece.a + piece.b) / 2 + (piece.b - piece.a) / 2 * t)
        unique = []
        for x in sorted(roots):
            if not unique or x - unique[-1] > tol * max(1, abs(x)):
                unique.append(x)
        return np.array(unique)


def chebfun(f, domain=(-1.0, 1.0), **kwargs):
    """Construct a tolerance-driven piecewise Chebyshev approximation."""
    return ChebyshevApproximation(f, domain, **kwargs)
