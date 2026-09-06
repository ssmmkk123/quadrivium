"""Automatic differentiation: exact derivatives without step sizes.

Forward mode uses dual numbers and costs one pass per input; reverse mode
records a tape and yields the whole gradient in one pass, which is what makes
it the right choice for scalar objectives in many variables.
"""

from __future__ import annotations

import math
from typing import Callable

from .. import numeric as np

__all__ = [
    "Dual",
    "HyperDual",
    "Variable",
    "derivative",
    "forward_gradient",
    "forward_jacobian",
    "gradient",
    "jacobian",
    "hessian",
    "hessian_vector_product",
    "second_derivative_ad",
    "value_and_grad",
    "taylor_coefficients",
]


class Dual:
    """Dual number ``a + b eps`` with ``eps^2 = 0``, for forward-mode AD.

    Arithmetic on duals propagates the derivative exactly alongside the value.
    """

    __slots__ = ("value", "deriv")

    def __init__(self, value, deriv=0.0):
        self.value = float(value)
        self.deriv = float(deriv)

    # -- arithmetic ---------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.value + o.value, self.deriv + o.deriv)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.deriv)

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.value - o.value, self.deriv - o.deriv)

    def __rsub__(self, o):
        return Dual(o) - self

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.value * o.value, self.deriv * o.value + self.value * o.deriv)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.value / o.value,
                    (self.deriv * o.value - self.value * o.deriv) / (o.value**2))

    def __rtruediv__(self, o):
        return Dual(o) / self

    def __pow__(self, p):
        if isinstance(p, Dual):
            v = self.value**p.value
            return Dual(v, v * (p.deriv * math.log(self.value)
                                + p.value * self.deriv / self.value))
        return Dual(self.value**p, p * self.value ** (p - 1) * self.deriv)

    def __rpow__(self, base):
        v = base**self.value
        return Dual(v, v * math.log(base) * self.deriv)

    # -- comparisons (compare on value, as usual for AD types) --------------
    def __lt__(self, o):
        return self.value < (o.value if isinstance(o, Dual) else o)

    def __le__(self, o):
        return self.value <= (o.value if isinstance(o, Dual) else o)

    def __gt__(self, o):
        return self.value > (o.value if isinstance(o, Dual) else o)

    def __ge__(self, o):
        return self.value >= (o.value if isinstance(o, Dual) else o)

    def __eq__(self, o):
        return self.value == (o.value if isinstance(o, Dual) else o)

    def __abs__(self):
        return Dual(abs(self.value), self.deriv * (1.0 if self.value >= 0 else -1.0))

    def __float__(self):
        return self.value

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"Dual({self.value:.6g}, {self.deriv:+.6g}eps)"

    # -- elementary functions ----------------------------------------------
    def sin(self):
        return Dual(math.sin(self.value), math.cos(self.value) * self.deriv)

    def cos(self):
        return Dual(math.cos(self.value), -math.sin(self.value) * self.deriv)

    def tan(self):
        return Dual(math.tan(self.value), self.deriv / math.cos(self.value) ** 2)

    def exp(self):
        e = math.exp(self.value)
        return Dual(e, e * self.deriv)

    def log(self):
        return Dual(math.log(self.value), self.deriv / self.value)

    def sqrt(self):
        s = math.sqrt(self.value)
        return Dual(s, self.deriv / (2 * s))

    def sinh(self):
        return Dual(math.sinh(self.value), math.cosh(self.value) * self.deriv)

    def cosh(self):
        return Dual(math.cosh(self.value), math.sinh(self.value) * self.deriv)

    def tanh(self):
        t = math.tanh(self.value)
        return Dual(t, (1 - t * t) * self.deriv)

    def arctan(self):
        return Dual(math.atan(self.value), self.deriv / (1 + self.value**2))

    def arcsin(self):
        return Dual(math.asin(self.value), self.deriv / math.sqrt(1 - self.value**2))

    def arccos(self):
        return Dual(math.acos(self.value), -self.deriv / math.sqrt(1 - self.value**2))


def derivative(f: Callable, x: float) -> float:
    """Exact first derivative of a scalar function by forward-mode AD."""
    return f(Dual(x, 1.0)).deriv


def forward_gradient(f: Callable, x) -> np.ndarray:
    """Gradient by forward mode: one pass per input variable."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = x.size
    g = np.empty(n)
    for i in range(n):
        duals = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        g[i] = f(duals).deriv
    return g


def forward_jacobian(F: Callable, x) -> np.ndarray:
    """Jacobian by forward mode."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = x.size
    cols = []
    for i in range(n):
        duals = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        out = F(duals)
        cols.append(np.array([o.deriv if isinstance(o, Dual) else 0.0 for o in out]))
    return np.column_stack(cols)


class _Tape:
    """Records elementary operations for reverse-mode accumulation."""

    def __init__(self):
        self.nodes = []

    def push(self, parents, grads):
        self.nodes.append((parents, grads))
        return len(self.nodes) - 1


class Variable:
    """Reverse-mode AD variable; call :meth:`backward` then read ``.grad``."""

    __slots__ = ("value", "grad", "_parents", "_grads")

    def __init__(self, value, parents=(), grads=()):
        self.value = float(value)
        self.grad = 0.0
        self._parents = parents
        self._grads = grads

    def _wrap(self, o):
        return o if isinstance(o, Variable) else Variable(o)

    def __add__(self, o):
        o = self._wrap(o)
        return Variable(self.value + o.value, (self, o), (1.0, 1.0))

    __radd__ = __add__

    def __neg__(self):
        return Variable(-self.value, (self,), (-1.0,))

    def __sub__(self, o):
        o = self._wrap(o)
        return Variable(self.value - o.value, (self, o), (1.0, -1.0))

    def __rsub__(self, o):
        return self._wrap(o) - self

    def __mul__(self, o):
        o = self._wrap(o)
        return Variable(self.value * o.value, (self, o), (o.value, self.value))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._wrap(o)
        return Variable(self.value / o.value, (self, o),
                        (1.0 / o.value, -self.value / o.value**2))

    def __rtruediv__(self, o):
        return self._wrap(o) / self

    def __pow__(self, p):
        if isinstance(p, Variable):
            v = self.value**p.value
            return Variable(v, (self, p),
                            (p.value * self.value ** (p.value - 1), v * math.log(self.value)))
        return Variable(self.value**p, (self,), (p * self.value ** (p - 1),))

    def __rpow__(self, base):
        v = base**self.value
        return Variable(v, (self,), (v * math.log(base),))

    def __lt__(self, o):
        return self.value < (o.value if isinstance(o, Variable) else o)

    def __gt__(self, o):
        return self.value > (o.value if isinstance(o, Variable) else o)

    def __float__(self):
        return self.value

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"Variable({self.value:.6g}, grad={self.grad:.6g})"

    def sin(self):
        return Variable(math.sin(self.value), (self,), (math.cos(self.value),))

    def cos(self):
        return Variable(math.cos(self.value), (self,), (-math.sin(self.value),))

    def tan(self):
        return Variable(math.tan(self.value), (self,), (1.0 / math.cos(self.value) ** 2,))

    def exp(self):
        e = math.exp(self.value)
        return Variable(e, (self,), (e,))

    def log(self):
        return Variable(math.log(self.value), (self,), (1.0 / self.value,))

    def sqrt(self):
        s = math.sqrt(self.value)
        return Variable(s, (self,), (0.5 / s,))

    def tanh(self):
        t = math.tanh(self.value)
        return Variable(t, (self,), (1 - t * t,))

    def sinh(self):
        return Variable(math.sinh(self.value), (self,), (math.cosh(self.value),))

    def cosh(self):
        return Variable(math.cosh(self.value), (self,), (math.sinh(self.value),))

    def arctan(self):
        return Variable(math.atan(self.value), (self,), (1.0 / (1 + self.value**2),))

    def backward(self, seed: float = 1.0) -> None:
        """Accumulate gradients through the recorded graph (reverse sweep)."""
        topo, seen = [], set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node._parents:
                visit(p)
            topo.append(node)

        visit(self)
        for node in topo:
            node.grad = 0.0
        self.grad = seed
        for node in reversed(topo):
            for parent, g in zip(node._parents, node._grads):
                parent.grad += node.grad * g


def gradient(f: Callable, x) -> np.ndarray:
    """Gradient by reverse mode: one sweep for all partials."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    vs = [Variable(v) for v in x]
    out = f(vs)
    out.backward()
    return np.array([v.grad for v in vs])


def value_and_grad(f: Callable, x):
    """Both the value and the gradient from a single reverse sweep."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    vs = [Variable(v) for v in x]
    out = f(vs)
    out.backward()
    return out.value, np.array([v.grad for v in vs])


def jacobian(F: Callable, x) -> np.ndarray:
    """Jacobian by reverse mode: one sweep per output component."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    rows = []
    m = len(F([Variable(v) for v in x]))
    for i in range(m):
        vs = [Variable(v) for v in x]
        out = F(vs)[i]
        out.backward()
        rows.append([v.grad for v in vs])
    return np.array(rows)


class HyperDual:
    """Hyper-dual number ``a + b e1 + c e2 + d e1 e2`` with ``e1^2 = e2^2 = 0``.

    The ``e1 e2`` component carries an exact second derivative, so Hessians come
    out to machine precision with no step size to choose.
    """

    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a, b=0.0, c=0.0, d=0.0):
        self.a, self.b, self.c, self.d = float(a), float(b), float(c), float(d)

    def _wrap(self, o):
        return o if isinstance(o, HyperDual) else HyperDual(o)

    def _chain(self, fv, fp, fpp):
        """Lift a scalar function given its value and first two derivatives."""
        return HyperDual(fv, self.b * fp, self.c * fp,
                         self.d * fp + self.b * self.c * fpp)

    def __add__(self, o):
        o = self._wrap(o)
        return HyperDual(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    __radd__ = __add__

    def __neg__(self):
        return HyperDual(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, o):
        o = self._wrap(o)
        return HyperDual(self.a - o.a, self.b - o.b, self.c - o.c, self.d - o.d)

    def __rsub__(self, o):
        return self._wrap(o) - self

    def __mul__(self, o):
        o = self._wrap(o)
        return HyperDual(
            self.a * o.a,
            self.a * o.b + self.b * o.a,
            self.a * o.c + self.c * o.a,
            self.a * o.d + self.b * o.c + self.c * o.b + self.d * o.a,
        )

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._wrap(o)
        return self * o._chain(1.0 / o.a, -1.0 / o.a**2, 2.0 / o.a**3)

    def __rtruediv__(self, o):
        return self._wrap(o) / self

    def __pow__(self, p):
        return self._chain(self.a**p, p * self.a ** (p - 1),
                           p * (p - 1) * self.a ** (p - 2))

    def __rpow__(self, base):
        v = base**self.a
        lb = math.log(base)
        return self._chain(v, v * lb, v * lb * lb)

    def __float__(self):
        return self.a

    def __lt__(self, o):
        return self.a < (o.a if isinstance(o, HyperDual) else o)

    def __gt__(self, o):
        return self.a > (o.a if isinstance(o, HyperDual) else o)

    def __repr__(self):  # pragma: no cover - cosmetic
        return f"HyperDual({self.a:.6g}, {self.b:+.4g}e1, {self.c:+.4g}e2, {self.d:+.4g}e1e2)"

    def sin(self):
        return self._chain(math.sin(self.a), math.cos(self.a), -math.sin(self.a))

    def cos(self):
        return self._chain(math.cos(self.a), -math.sin(self.a), -math.cos(self.a))

    def tan(self):
        t = math.tan(self.a)
        return self._chain(t, 1 + t * t, 2 * t * (1 + t * t))

    def exp(self):
        e = math.exp(self.a)
        return self._chain(e, e, e)

    def log(self):
        return self._chain(math.log(self.a), 1.0 / self.a, -1.0 / self.a**2)

    def sqrt(self):
        s = math.sqrt(self.a)
        return self._chain(s, 0.5 / s, -0.25 / (s * self.a))

    def tanh(self):
        t = math.tanh(self.a)
        return self._chain(t, 1 - t * t, -2 * t * (1 - t * t))

    def sinh(self):
        return self._chain(math.sinh(self.a), math.cosh(self.a), math.sinh(self.a))

    def cosh(self):
        return self._chain(math.cosh(self.a), math.sinh(self.a), math.cosh(self.a))

    def arctan(self):
        return self._chain(math.atan(self.a), 1.0 / (1 + self.a**2),
                           -2 * self.a / (1 + self.a**2) ** 2)


def second_derivative_ad(f: Callable, x: float) -> float:
    """Exact second derivative of a scalar function via hyper-dual arithmetic."""
    return f(HyperDual(x, 1.0, 1.0, 0.0)).d


def hessian(f: Callable, x) -> np.ndarray:
    """Exact Hessian by hyper-dual (forward-over-forward) differentiation.

    Perturbs component ``i`` along ``e1`` and ``j`` along ``e2``; the ``e1 e2``
    coefficient of the result is exactly the mixed partial.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = x.size
    H = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            args = [HyperDual(x[k], 1.0 if k == i else 0.0,
                              1.0 if k == j else 0.0, 0.0) for k in range(n)]
            H[i, j] = H[j, i] = f(args).d
    return H


def hessian_vector_product(f: Callable, x, v) -> np.ndarray:
    """Hessian-vector product ``H v`` without forming ``H``."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    n = x.size
    # directional second derivative: e1 along v, e2 along each coordinate
    out = np.empty(n)
    for j in range(n):
        args = [HyperDual(x[k], v[k], 1.0 if k == j else 0.0, 0.0) for k in range(n)]
        out[j] = f(args).d
    return out


def taylor_coefficients(f: Callable, x: float, order: int = 5, h: float = 0.1):
    """Taylor coefficients of ``f`` about ``x`` up to ``order``.

    One symmetric Fornberg stencil supplies the weights for every derivative
    order at once, so the whole series costs a single set of evaluations.

    ``h`` trades truncation against round-off: the stencil spans
    ``x +/- order*h``, so reduce it for functions with nearby singularities and
    raise it if high-order coefficients look noisy.
    """
    from math import factorial

    from .finite import fornberg_weights

    m = max(order, 2)
    nodes = x + h * np.arange(-m, m + 1, dtype=float)
    vals = np.array([float(f(t)) for t in nodes])
    W = fornberg_weights(float(x), nodes, order)
    return np.array([float(W[:, k] @ vals) / factorial(k) for k in range(order + 1)])
