"""Shared-node adaptive integration of vector and complex-valued functions."""
from __future__ import annotations
import heapq
import math
import operator
from .. import numeric as np
from ..core import QuadratureResult
from .adaptive import _GK15_NODES, _GK15_W, _GK7_W

__all__ = ["quad_vec"]


def quad_vec(f, a, b, *, epsabs=1e-10, epsrel=1e-8, norm="max", limit=1000, points=None):
    """Adaptive Gauss--Kronrod integration preserving the integrand's array shape.

    ``norm='max'`` controls each component using epsabs + epsrel*abs(integral).
    ``norm='2'`` controls the Euclidean error norm. epsabs may be an array under
    componentwise control. ``limit`` bounds retained panels. Infinite limits
    are mapped to a finite interval. Error estimates are not certified bounds.
    """
    a, b, limit = float(a), float(b), operator.index(limit)
    abstol = np.asarray(epsabs, dtype=float)
    if (np.any(abstol < 0) or not np.all(np.isfinite(abstol)) or epsrel < 0 or
            not math.isfinite(epsrel) or (not np.any(abstol > 0) and epsrel < 1e-14) or limit < 1):
        raise ValueError("invalid absolute/relative tolerance or panel limit")
    if norm not in ("max", "2") or math.isnan(a) or math.isnan(b):
        raise ValueError("invalid norm or integration limits")
    sign = 1
    if b < a:
        a, b, sign = b, a, -1
    original = f
    if not math.isfinite(a) or not math.isfinite(b):
        if points:
            raise ValueError("breakpoints require finite integration limits")
        if a == -math.inf and b == math.inf:
            def f(t):
                return np.asarray(original(t / (1 - t * t))) * (1 + t * t) / (1 - t * t) ** 2
            a, b = -1.0, 1.0
        elif math.isfinite(a) and b == math.inf:
            start = a
            def f(t):
                return np.asarray(original(start + t / (1 - t))) / (1 - t) ** 2
            a, b = 0.0, 1.0
        elif a == -math.inf and math.isfinite(b):
            end = b
            def f(t):
                return np.asarray(original(end - t / (1 - t))) / (1 - t) ** 2
            a, b = 0.0, 1.0
        else:
            raise ValueError("invalid infinite integration limits")
    calls, shape = 0, None
    def panel(left, right):
        nonlocal calls, shape
        center, half = (left + right) / 2, (right - left) / 2
        values = []
        for node in _GK15_NODES:
            v = np.asarray(f(center + half * node))
            if shape is None:
                shape = v.shape
            if v.shape != shape or not np.all(np.isfinite(v)):
                raise ValueError("integrand must return finite arrays of a fixed shape")
            values.append(v.copy())
            calls += 1
        values = np.array(values)
        expanded = (15,) + (1,) * len(shape)
        kronrod = half * np.sum(values * _GK15_W.reshape(expanded), axis=0)
        gauss = half * np.sum(values[1::2] * _GK7_W.reshape((7,) + (1,) * len(shape)), axis=0)
        error = np.abs(kronrod - gauss)
        floor = 50 * 2.220446049250313e-16 * abs(half) * np.sum(np.abs(values) * _GK15_W.reshape(expanded), axis=0)
        return kronrod, np.maximum(error, floor)
    breaks = sorted(set(float(x) for x in (() if points is None else points)))
    if any(not a < x < b for x in breaks) or len(breaks) + 1 > limit:
        raise ValueError("breakpoints must lie inside limits and fit the panel budget")
    edges = [a] + breaks + [b]
    queue, counter = [], 0
    total = errors = None
    for left, right in zip(edges[:-1], edges[1:]):
        value, error = panel(left, right)
        total = value.copy() if total is None else total + value
        errors = error.copy() if errors is None else errors + error
        heapq.heappush(queue, (-float(np.max(error)), counter, left, right, value, error))
        counter += 1
    try:
        abstol = np.broadcast_to(abstol, shape)
    except ValueError as exc:
        raise ValueError("epsabs must broadcast to integrand shape") from exc
    def satisfied():
        if norm == "max":
            return bool(np.all(errors <= abstol + epsrel * np.abs(total)))
        return float(np.linalg.norm(errors.ravel())) <= float(np.linalg.norm(abstol.ravel())) + epsrel * float(np.linalg.norm(total.ravel()))
    while not satisfied() and len(queue) < limit:
        _, _, left, right, old, old_error = heapq.heappop(queue)
        mid = (left + right) / 2
        if mid == left or mid == right:
            heapq.heappush(queue, (0, counter, left, right, old, old_error))
            break
        v1, e1 = panel(left, mid)
        v2, e2 = panel(mid, right)
        total += v1 + v2 - old
        errors = np.maximum(0, errors + e1 + e2 - old_error)
        for l, r, v, e in ((left, mid, v1, e1), (mid, right, v2, e2)):
            heapq.heappush(queue, (-float(np.max(e)), counter, l, r, v, e))
            counter += 1
    value = sign * total
    value = value.item() if value.ndim == 0 else value
    result = QuadratureResult(value, errors.item() if errors.ndim == 0 else errors,
                              calls, len(queue), satisfied(), "quad_vec")
    result.message = "converged" if result.converged else "panel budget or floating-point interval resolution exhausted"
    return result
