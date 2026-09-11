"""Array-aware differentiation with reverse pullbacks and forward tangents.

Each tape node represents a whole array operation. JVP and VJP never assemble
Jacobians. Supported primitives include broadcasting arithmetic, indexing,
reductions, reshape/transpose, matrix multiplication and smooth elementary
functions. Mutation and complex differentiation are intentionally rejected.
"""
from __future__ import annotations
from .. import numeric as np

__all__ = ["Tensor", "jvp", "vjp", "array_gradient", "array_value_and_grad"]


def _unbroadcast(value, shape):
    value = np.asarray(value, dtype=float)
    while value.ndim > len(shape):
        value = value.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(axis=axis, keepdims=True)
    return value.reshape(shape)


def _tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)


class Tensor:
    """Differentiable real array; ``data`` owns a snapshot of the input."""
    __array_priority__ = 1000

    def __init__(self, data, *, requires_grad=True, tangent=None, _parents=()):
        if np.iscomplexobj(data):
            raise TypeError("Tensor differentiation currently requires real arrays")
        self._data = np.array(data, dtype=float, copy=True)
        self._data.flags.writeable = False
        self.requires_grad = bool(requires_grad)
        self.parents = tuple(_parents) if self.requires_grad else ()
        self.grad = None
        self.tangent = None if tangent is None else np.array(tangent, dtype=float, copy=True)
        if self.tangent is not None and self.tangent.shape != self._data.shape:
            raise ValueError("tangent shape must equal data shape")

    @property
    def data(self):
        """A read-only view of the immutable forward value."""
        return self._data.reshape(self._data.shape)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def size(self):
        return self._data.size

    def __len__(self):
        return len(self._data)

    def __float__(self):
        raise TypeError("converting a Tensor to float would discard its derivative; use .data")

    def __repr__(self):
        return f"Tensor({self._data!r}, requires_grad={self.requires_grad})"

    def _unary(self, value, derivative):
        parents = ((self, lambda g: g * derivative),) if self.requires_grad else ()
        tangent = None if self.tangent is None else self.tangent * derivative
        return Tensor(value, requires_grad=self.requires_grad, tangent=tangent, _parents=parents)

    def _binary(self, other, op, da, db):
        other = _tensor(other)
        value = op(self._data, other._data)
        left, right = da(self._data, other._data), db(self._data, other._data)
        parents = []
        if self.requires_grad:
            parents.append((self, lambda g: _unbroadcast(g * left, self.shape)))
        if other.requires_grad:
            parents.append((other, lambda g: _unbroadcast(g * right, other.shape)))
        tangent = None
        if self.tangent is not None or other.tangent is not None:
            tangent = ((0 if self.tangent is None else self.tangent * left) +
                       (0 if other.tangent is None else other.tangent * right))
            tangent = np.broadcast_to(np.asarray(tangent), np.shape(value)).copy()
        return Tensor(value, requires_grad=bool(parents), tangent=tangent, _parents=parents)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b, lambda a, b: 1, lambda a, b: 1)
    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b, lambda a, b: 1, lambda a, b: -1)

    def __rsub__(self, other):
        return _tensor(other) - self

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b, lambda a, b: b, lambda a, b: a)
    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b, lambda a, b: 1 / b, lambda a, b: -a / (b * b))

    def __rtruediv__(self, other):
        return _tensor(other) / self

    def __neg__(self):
        return self._unary(-self._data, -1)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            power = float(other)
            derivative = np.zeros_like(self._data) if power == 0 else power * self._data ** (power - 1)
            return self._unary(self._data ** power, derivative)
        return (self.log() * other).exp()

    def __rpow__(self, other):
        return (_tensor(other).log() * self).exp()

    def __abs__(self):
        return self._unary(np.abs(self._data), np.sign(self._data))

    def exp(self):
        value = np.exp(self._data)
        return self._unary(value, value)

    def expm1(self):
        return self._unary(np.expm1(self._data), np.exp(self._data))

    def log(self):
        return self._unary(np.log(self._data), 1 / self._data)

    def log1p(self):
        return self._unary(np.log1p(self._data), 1 / (1 + self._data))

    def sin(self):
        return self._unary(np.sin(self._data), np.cos(self._data))

    def cos(self):
        return self._unary(np.cos(self._data), -np.sin(self._data))

    def tan(self):
        return self._unary(np.tan(self._data), 1 / np.cos(self._data) ** 2)

    def tanh(self):
        value = np.tanh(self._data)
        return self._unary(value, 1 - value * value)

    def sqrt(self):
        return self._unary(np.sqrt(self._data), 0.5 / np.sqrt(self._data))

    def sum(self, axis=None, keepdims=False, **kwargs):
        if kwargs:
            raise TypeError("Tensor.sum supports only axis and keepdims")
        value = self._data.sum(axis=axis, keepdims=keepdims)
        axes = tuple(range(self.ndim)) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
        axes = tuple(a % self.ndim for a in axes) if self.ndim else ()
        def pullback(g):
            if not keepdims:
                shape = list(self.shape)
                for a in axes:
                    shape[a] = 1
                g = g.reshape(shape)
            return np.broadcast_to(g, self.shape).copy()
        parents = ((self, pullback),) if self.requires_grad else ()
        tangent = None if self.tangent is None else self.tangent.sum(axis=axis, keepdims=keepdims)
        return Tensor(value, requires_grad=self.requires_grad, tangent=tangent, _parents=parents)

    def mean(self, axis=None, keepdims=False, **kwargs):
        axes = tuple(range(self.ndim)) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
        count = 1
        for a in axes:
            count *= self.shape[a]
        return self.sum(axis=axis, keepdims=keepdims, **kwargs) / count

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        parents = ((self, lambda g: g.reshape(self.shape)),) if self.requires_grad else ()
        tangent = None if self.tangent is None else self.tangent.reshape(shape)
        return Tensor(self._data.reshape(shape), requires_grad=self.requires_grad, tangent=tangent, _parents=parents)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes or axes == (None,):
            axes = tuple(reversed(range(self.ndim)))
        axes = tuple(a % self.ndim if -self.ndim <= a < self.ndim else a for a in axes)
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError("axes must be a permutation of dimensions")
        inverse = tuple(axes.index(i) for i in range(self.ndim))
        parents = ((self, lambda g: g.transpose(inverse)),) if self.requires_grad else ()
        tangent = None if self.tangent is None else self.tangent.transpose(axes)
        return Tensor(self._data.transpose(axes), requires_grad=self.requires_grad, tangent=tangent, _parents=parents)

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, key):
        # The pullback must use the same indices even if the caller edits them.
        def snapshot(index):
            if isinstance(index, (tuple, list)):
                return type(index)(snapshot(v) for v in index)
            if hasattr(index, "shape") and hasattr(index, "copy"):
                return index.copy()
            return index
        key = snapshot(key)
        def pullback(g):
            out = np.zeros(self.shape)
            np.add.at(out, key, g)
            return out
        parents = ((self, pullback),) if self.requires_grad else ()
        tangent = None if self.tangent is None else self.tangent[key]
        return Tensor(self._data[key], requires_grad=self.requires_grad, tangent=tangent, _parents=parents)

    def __matmul__(self, other):
        other = _tensor(other)
        a, b = self._data, other._data
        value = a @ b
        aa = a.reshape(1, -1) if a.ndim == 1 else a
        bb = b.reshape(-1, 1) if b.ndim == 1 else b
        fullshape = np.broadcast_shapes(aa.shape[:-2], bb.shape[:-2]) + (aa.shape[-2], bb.shape[-1])
        def transpose_last(x):
            return np.swapaxes(x, -1, -2)
        def left(g):
            raw = g.reshape(fullshape) @ transpose_last(bb)
            return _unbroadcast(raw, aa.shape).reshape(a.shape)
        def right(g):
            raw = transpose_last(aa) @ g.reshape(fullshape)
            return _unbroadcast(raw, bb.shape).reshape(b.shape)
        parents = []
        if self.requires_grad:
            parents.append((self, left))
        if other.requires_grad:
            parents.append((other, right))
        tangent = None
        if self.tangent is not None or other.tangent is not None:
            tangent = ((0 if self.tangent is None else self.tangent @ b) +
                       (0 if other.tangent is None else a @ other.tangent))
        return Tensor(value, requires_grad=bool(parents), tangent=tangent, _parents=parents)

    def __rmatmul__(self, other):
        return _tensor(other) @ self

    def backward(self, cotangent=None):
        """Accumulate a VJP into leaf ``grad`` arrays, resetting previous gradients."""
        if cotangent is None:
            if self.size != 1:
                raise ValueError("a nonscalar output requires a cotangent")
            cotangent = np.ones(self.shape)
        cotangent = np.asarray(cotangent, dtype=float)
        if cotangent.shape != self.shape:
            raise ValueError("cotangent shape must equal output shape")
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p, _ in node.parents)
        for node in order:
            node.grad = None
        self.grad = cotangent.copy()
        for node in reversed(order):
            if node.grad is None:
                continue
            for parent, pullback in node.parents:
                contribution = pullback(node.grad)
                parent.grad = contribution if parent.grad is None else parent.grad + contribution
        return self

    def __quadrivium_function__(self, name, *args, **kwargs):
        if kwargs.get("out") is not None:
            raise TypeError("differentiable operations do not support mutation/out")
        kwargs.pop("out", None)
        unary = {"exp", "expm1", "log", "log1p", "sin", "cos", "tan", "tanh", "sqrt"}
        if name in unary:
            return getattr(_tensor(args[0]), name)()
        if name in ("absolute", "abs"):
            return abs(_tensor(args[0]))
        if name == "negative":
            return -_tensor(args[0])
        if name == "square":
            return _tensor(args[0]) ** 2
        if name in ("sum", "mean", "reshape", "transpose"):
            return getattr(_tensor(args[0]), name)(*args[1:], **kwargs)
        if name == "matmul":
            return _tensor(args[0]) @ args[1]
        return NotImplemented


def jvp(f, x, tangent):
    """Return ``(f(x), J(x) @ tangent)`` in one forward pass."""
    point = Tensor(x, requires_grad=False, tangent=tangent)
    result = _tensor(f(point))
    return result.data.copy(), np.zeros(result.shape) if result.tangent is None else result.tangent.copy()


def vjp(f, x, cotangent=None):
    """Return value and a pullback, or value and J.T @ cotangent if supplied."""
    point = Tensor(x)
    result = _tensor(f(point))
    def pullback(seed):
        result.backward(seed)
        return np.zeros(point.shape) if point.grad is None else point.grad.copy()
    return result.data.copy(), pullback if cotangent is None else pullback(cotangent)


def array_value_and_grad(f, x):
    """Return a scalar objective value and its gradient with respect to an array.

    ``f`` receives a differentiable ``Tensor`` and must use supported real
    arithmetic or tensor operations. Its result must contain exactly one
    element; other output sizes raise ``ValueError``. The returned pair is
    ``(float_value, gradient_array)``, with the gradient shaped like ``x``.
    Inputs disconnected from the objective receive zero derivatives.
    """
    point = Tensor(x)
    result = _tensor(f(point))
    if result.size != 1:
        raise ValueError("gradient requires a scalar-valued objective")
    result.backward()
    return float(result.data), np.zeros(point.shape) if point.grad is None else point.grad.copy()


def array_gradient(f, x):
    """Return the gradient of a scalar-valued objective with array input.

    This is the gradient component of ``array_value_and_grad(f, x)`` and
    has the same requirements: ``f`` must accept a ``Tensor``, use supported
    differentiable operations, and return exactly one element. The gradient
    has the shape of ``x``.
    """
    return array_value_and_grad(f, x)[1]
