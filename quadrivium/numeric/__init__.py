"""The array namespace the rest of the package computes with.

`quadrivium.numeric` is a drop-in replacement for the subset of NumPy the
library used to depend on: the same names, the same call signatures, the same
semantics, backed by the compiled core in :mod:`quadrivium._qnp` rather than by
NumPy.  Import it the way the package does::

    >>> from quadrivium import numeric as np
    >>> np.linalg.solve(np.eye(2), np.array([1.0, 2.0]))
    array([1., 2.])

Arrays are strided and N-dimensional over ``bool``, ``int64``, ``float32``,
``float64``, ``complex64`` and ``complex128``. Python floating-point input
defaults to double precision; explicit single precision halves array storage.  Broadcasting, the indexing grammar, views, the buffer protocol and
NumPy's pairwise summation are all reproduced, so results are not merely close
to what NumPy produced: sums are bit-for-bit identical, and a seeded
:func:`random.default_rng` yields exactly the same stream.

Objects that export a PEP 3118 buffer -- a real NumPy array among them --
convert on the way in, so callers who have NumPy may keep passing NumPy arrays.
"""

from __future__ import annotations

import builtins
import math as _math
from contextlib import contextmanager

from .. import _qnp as _c
from . import _formatting
from . import linalg, fft, random, polynomial, testing

_c.set_printer(lambda array, is_repr: _formatting.format_array(array, bool(is_repr)))

ndarray = _c.ndarray
dtype = _c.dtype
float32 = _c.float32
complex64 = _c.complex64
single = float32
csingle = complex64
float64 = _c.float64
complex128 = _c.complex128
int64 = _c.int64
bool_ = _c.bool
intp = _c.int64
float_ = float64
complex_ = complex128
double = float64
cdouble = complex128

#: Abstract dtype groups, kept so ``isinstance(x, np.integer)`` still works.
integer = (int, _c.int64.type)
floating = (float,)
number = (int, float, complex)
inexact = (float, complex)

pi = _math.pi
e = _math.e
euler_gamma = 0.5772156649015329
inf = _math.inf
Inf = inf
infty = inf
nan = _math.nan
NaN = nan
newaxis = None

LinAlgError = _c.LinAlgError

# --- straight from the compiled core -------------------------------------
array = _c.array
asarray = _c.asarray
ascontiguousarray = _c.ascontiguousarray
asanyarray = _c.asarray
zeros = _c.zeros
ones = _c.ones
empty = _c.empty
full = _c.full
zeros_like = _c.zeros_like
ones_like = _c.ones_like
empty_like = _c.empty_like
full_like = _c.full_like
arange = _c.arange
linspace = _c.linspace
eye = _c.eye
reshape = _c.reshape
transpose = _c.transpose
broadcast_to = _c.broadcast_to
concatenate = _c.concatenate
roll = _c.roll
repeat = _c.repeat
diff = _c.diff
outer = _c.outer
where = _c.where
copy = _c.copy
_interp_c = _c.interp


def interp(x, xp, fp, left=None, right=None, period=None):
    """Piecewise linear interpolation; `period` wraps the sample points."""
    if period is None and left is None and right is None:
        return _interp_c(x, xp, fp)
    points = asarray(xp, dtype=float64)
    values = asarray(fp, dtype=float64)
    query = asarray(x, dtype=float64)
    if period is not None:
        period = float(period)
        query = remainder(query, period)
        points = remainder(points, period)
        order = argsort(points)
        points = take(points, order)
        values = take(values, order)
        # One wrapped point at each end makes the interval periodic.
        points = concatenate([points[-1:] - period, points, points[:1] + period])
        values = concatenate([values[-1:], values, values[:1]])
        return _interp_c(query, points, values)
    result = _interp_c(query, points, values)
    if left is not None:
        result = where(query < points[0], asarray(float(left), dtype=float64), result)
    if right is not None:
        result = where(query > points[-1], asarray(float(right), dtype=float64), result)
    return result


shares_memory = _c.shares_memory
may_share_memory = _c.shares_memory
nonzero = _c.nonzero
flatnonzero = _c.flatnonzero
take = _c.take
sort = _c.sort
argsort = _c.argsort
lexsort = _c.lexsort
searchsorted = _c.searchsorted
bincount = _c.bincount
clip = _c.clip
quantile = _c.quantile
sum = _c.sum
prod = _c.prod
amax = _c.amax
amin = _c.amin
max = _c.amax
min = _c.amin
any = _c.any
all = _c.all
argmax = _c.argmax
argmin = _c.argmin
mean = _c.mean
count_nonzero = _c.count_nonzero
var = _c.var
std = _c.std
cumsum = _c.cumsum
cumprod = _c.cumprod
matmul = _c.matmul
round = _c.round
around = _c.round
sqrt = _c.sqrt
exp = _c.exp
log = _c.log
log2 = _c.log2
log10 = _c.log10
log1p = _c.log1p
expm1 = _c.expm1
sin = _c.sin
cos = _c.cos
tan = _c.tan
arcsin = _c.arcsin
arccos = _c.arccos
arctan = _c.arctan
sinh = _c.sinh
cosh = _c.cosh
tanh = _c.tanh
arcsinh = _c.arcsinh
arccosh = _c.arccosh
arctanh = _c.arctanh
sign = _c.sign
floor = _c.floor
ceil = _c.ceil
trunc = _c.trunc
rint = _c.rint
fix = _c.trunc
square = _c.square
reciprocal = _c.reciprocal
conj = _c.conjugate
conjugate = _c.conjugate
real = _c.real
imag = _c.imag
angle = _c.angle
isfinite = _c.isfinite
isnan = _c.isnan
isinf = _c.isinf
logical_not = _c.logical_not
invert = _c.invert
signbit = _c.signbit
abs = _c.absolute
absolute = _c.absolute
negative = _c.negative
hypot = _c.hypot
arctan2 = _c.arctan2
copysign = _c.copysign
equal = _c.equal
not_equal = _c.not_equal
less = _c.less
less_equal = _c.less_equal
greater = _c.greater
greater_equal = _c.greater_equal
logical_and = _c.logical_and
logical_or = _c.logical_or
bitwise_and = _c.bitwise_and
bitwise_or = _c.bitwise_or
left_shift = _c.left_shift
right_shift = _c.right_shift
float_power = _c.power
mod = _c.remainder
remainder = _c.remainder
floor_divide = _c.floor_divide
true_divide = _c.divide


class _UFunc:
    """A callable with the reduction attributes NumPy's ufuncs carry.

    Only the handful the library uses are provided -- ``at``, ``reduce``,
    ``accumulate``, ``reduceat`` and ``outer`` -- each delegating to the
    compiled core.
    """

    __slots__ = ("_call", "__name__", "_reduce_kind", "_identity")

    def __init__(self, call, name, reduce_kind=None, identity=None):
        self._call = call
        self.__name__ = name
        self._reduce_kind = reduce_kind
        self._identity = identity

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __repr__(self):
        return f"<ufunc {self.__name__!r}>"

    def reduce(self, a, axis=0, out=None, keepdims=False):
        if self._reduce_kind is None:
            raise NotImplementedError(f"{self.__name__}.reduce")
        return self._reduce_kind(a, axis=axis, out=out, keepdims=keepdims)

    def accumulate(self, a, axis=0, out=None):
        values = asarray(a)
        if values.ndim == 0:
            raise ValueError("accumulate needs at least one dimension")
        if self.__name__ in ("add", "multiply") and values.dtype != bool_:
            # Keep the existing axis normalization and output assignment. The
            # native kernels preserve prefix order without Python slice and
            # ufunc allocations at every step; bool keeps its original dtype.
            kernel = _c.cumsum if self.__name__ == "add" else _c.cumprod
            result = kernel(values, axis=axis % values.ndim)
        else:
            result = values.astype(values.dtype, copy=True)
            moved = result if axis in (0, -values.ndim) else result.transpose(
                _moveaxis_perm(values.ndim, axis))
            for i in builtins.range(1, moved.shape[0]):
                moved[i] = self._call(moved[i - 1], moved[i])
        if out is not None:
            out[...] = result
            return out
        return result

    def reduceat(self, a, indices, axis=0):
        if self._reduce_kind is None:
            raise NotImplementedError(f"{self.__name__}.reduceat")
        return _c.reduceat(a, indices, axis, _REDUCE_KINDS[self.__name__])

    def at(self, a, indices, b=None):
        if self.__name__ != "add":
            raise NotImplementedError(f"{self.__name__}.at")
        _c.add_at(a, indices, 0 if b is None else b)

    def outer(self, a, b):
        left = asarray(a)
        right = asarray(b)
        return self._call(left.reshape(left.shape + (1,) * right.ndim), right)


def _moveaxis_perm(ndim, axis):
    axis = axis % ndim
    return (axis,) + tuple(i for i in builtins.range(ndim) if i != axis)


def _take_slice(values, axis, start, stop):
    key = [slice(None)] * values.ndim
    key[axis] = slice(start, stop)
    return values[tuple(key)]


#: Reduction codes the compiled core understands, by ufunc name.
_REDUCE_KINDS = {"add": 0, "multiply": 1, "maximum": 2, "minimum": 3}

add = _UFunc(_c.add, "add", sum)
subtract = _UFunc(_c.subtract, "subtract")
multiply = _UFunc(_c.multiply, "multiply", prod)
divide = _UFunc(_c.divide, "divide")
power = _UFunc(_c.power, "power")
maximum = _UFunc(_c.maximum, "maximum", amax)
minimum = _UFunc(_c.minimum, "minimum", amin)
bitwise_xor = _UFunc(_c.bitwise_xor, "bitwise_xor")
dot = _UFunc(_c.matmul, "dot")


# --- shape helpers --------------------------------------------------------

def ndim(a):
    return a.ndim if isinstance(a, ndarray) else asarray(a).ndim


def shape(a):
    return a.shape if isinstance(a, ndarray) else asarray(a).shape


def size(a, axis=None):
    values = asarray(a)
    return values.size if axis is None else values.shape[axis]


def atleast_1d(*arys):
    out = []
    for a in arys:
        values = asarray(a)
        out.append(values.reshape(1) if values.ndim == 0 else values)
    return out[0] if len(out) == 1 else out


def atleast_2d(*arys):
    out = []
    for a in arys:
        values = asarray(a)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values.reshape(1, values.shape[0])
        out.append(values)
    return out[0] if len(out) == 1 else out


def atleast_3d(*arys):
    out = []
    for a in arys:
        values = atleast_2d(asarray(a))
        if values.ndim == 2:
            values = values.reshape(values.shape + (1,))
        out.append(values)
    return out[0] if len(out) == 1 else out


def asfortranarray(a, dtype=None):
    """Column-major copy, built as a transposed row-major one."""
    values = asarray(a, dtype=dtype)
    if values.ndim < 2:
        return ascontiguousarray(values)
    return ascontiguousarray(values.T).T


def ravel(a):
    return asarray(a).ravel()


def squeeze(a, axis=None):
    return asarray(a).squeeze(axis)


def expand_dims(a, axis):
    values = asarray(a)
    axis = axis % (values.ndim + 1)
    return values.reshape(values.shape[:axis] + (1,) + values.shape[axis:])


def moveaxis(a, source, destination):
    values = asarray(a)
    order = [i for i in builtins.range(values.ndim) if i != source % values.ndim]
    order.insert(destination % values.ndim, source % values.ndim)
    return values.transpose(tuple(order))


def swapaxes(a, axis1, axis2):
    values = asarray(a)
    order = list(builtins.range(values.ndim))
    order[axis1], order[axis2] = order[axis2], order[axis1]
    return values.transpose(tuple(order))


def hstack(tup):
    arrays = [atleast_1d(a) for a in tup]
    axis = 0 if arrays and arrays[0].ndim == 1 else 1
    return concatenate(arrays, axis=axis)


def vstack(tup):
    return concatenate([atleast_2d(a) for a in tup], axis=0)


def column_stack(tup):
    columns = []
    for a in tup:
        values = asarray(a)
        columns.append(values.reshape(values.shape[0], 1) if values.ndim == 1 else values)
    return concatenate(columns, axis=1)


def stack(arrays, axis=0):
    expanded = [expand_dims(asarray(a), axis) for a in arrays]
    return concatenate(expanded, axis=axis)


def block(arrays):
    if not isinstance(arrays, (list, tuple)):
        return asarray(arrays)
    if arrays and isinstance(arrays[0], (list, tuple)):
        return concatenate([concatenate([atleast_2d(b) for b in row], axis=1)
                            for row in arrays], axis=0)
    return concatenate([atleast_2d(b) for b in arrays], axis=1)


def append(arr, values, axis=None):
    if axis is None:
        return concatenate([asarray(arr).ravel(), asarray(values).ravel()])
    return concatenate([asarray(arr), asarray(values)], axis=axis)


def tile(a, reps):
    values = asarray(a)
    if isinstance(reps, int):
        reps = (reps,)
    reps = tuple(reps)
    while values.ndim < len(reps):
        values = values.reshape((1,) + values.shape)
    reps = (1,) * (values.ndim - len(reps)) + reps
    for axis, count in enumerate(reps):
        if count != 1:
            values = concatenate([values] * count, axis=axis)
    return values


def delete(arr, obj, axis=None):
    values = asarray(arr)
    if axis is None:
        values = values.ravel()
        axis = 0
    length = values.shape[axis]
    if isinstance(obj, slice):
        drop = set(builtins.range(*obj.indices(length)))
    elif isinstance(obj, (int, _c.int64.type)):
        drop = {int(obj) % length}
    else:
        drop = {int(i) % length for i in asarray(obj).tolist()}
    keep = array([i for i in builtins.range(length) if i not in drop], dtype=int64)
    return take(values, keep, axis=axis)


def insert(arr, obj, values, axis=None):
    source = asarray(arr)
    if axis is None:
        source = source.ravel()
        axis = 0
    position = int(obj)
    head = _take_slice(source, axis, 0, position)
    tail = _take_slice(source, axis, position, source.shape[axis])
    middle = atleast_1d(asarray(values))
    if source.ndim > 1:
        middle = middle.reshape(
            tuple(1 if i == axis else source.shape[i] for i in builtins.range(source.ndim)))
    return concatenate([head, middle, tail], axis=axis)


def flip(m, axis=None):
    values = asarray(m)
    axes = builtins.range(values.ndim) if axis is None else (axis,)
    key = [slice(None)] * values.ndim
    for ax in axes:
        key[ax % values.ndim] = slice(None, None, -1)
    return values[tuple(key)]


def pad(array_in, pad_width, mode="constant", constant_values=0):
    values = asarray(array_in)
    if isinstance(pad_width, int):
        widths = [(pad_width, pad_width)] * values.ndim
    elif isinstance(pad_width[0], int):
        widths = [tuple(pad_width)] * values.ndim
    else:
        widths = [tuple(w) for w in pad_width]
    for axis, (before, after) in enumerate(widths):
        if before == 0 and after == 0:
            continue
        n = values.shape[axis]
        if mode == "constant":
            head_shape = list(values.shape)
            head_shape[axis] = before
            tail_shape = list(values.shape)
            tail_shape[axis] = after
            head = full(tuple(head_shape), constant_values, dtype=values.dtype)
            tail = full(tuple(tail_shape), constant_values, dtype=values.dtype)
        elif mode == "edge":
            head = concatenate([_take_slice(values, axis, 0, 1)] * before, axis=axis) \
                if before else _take_slice(values, axis, 0, 0)
            tail = concatenate([_take_slice(values, axis, n - 1, n)] * after, axis=axis) \
                if after else _take_slice(values, axis, 0, 0)
        elif mode == "wrap":
            head = _take_slice(values, axis, n - before, n) if before else _take_slice(values, axis, 0, 0)
            tail = _take_slice(values, axis, 0, after) if after else _take_slice(values, axis, 0, 0)
        elif mode in ("reflect", "symmetric"):
            offset = 1 if mode == "reflect" else 0
            head = flip(_take_slice(values, axis, offset, before + offset), axis=axis) \
                if before else _take_slice(values, axis, 0, 0)
            tail = flip(_take_slice(values, axis, n - after - offset, n - offset), axis=axis) \
                if after else _take_slice(values, axis, 0, 0)
        else:
            raise NotImplementedError(f"pad mode {mode!r}")
        values = concatenate([head, values, tail], axis=axis)
    return values


def broadcast_shapes(*shapes):
    result = []
    for s in shapes:
        s = (s,) if isinstance(s, int) else tuple(s)
        result = _pair_broadcast(tuple(result), s)
    return tuple(result)


def _pair_broadcast(a, b):
    nd = builtins.max(len(a), len(b))
    a = (1,) * (nd - len(a)) + a
    b = (1,) * (nd - len(b)) + b
    out = []
    for x, y in zip(a, b):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"shape mismatch: {a} and {b} are not broadcastable")
    return tuple(out)


def broadcast_arrays(*arrays):
    values = [asarray(a) for a in arrays]
    target = broadcast_shapes(*[v.shape for v in values])
    return [broadcast_to(v, target) for v in values]


# --- construction helpers -------------------------------------------------

def identity(n, dtype=None):
    return eye(n, dtype=dtype)


def diagonal(a, offset=0):
    values = asarray(a)
    if values.ndim != 2:
        raise ValueError("diagonal expects a 2-dimensional array")
    rows, cols = values.shape
    flat = values.reshape(-1) if values.flags["C_CONTIGUOUS"] else values.ravel()
    if offset >= 0:
        count = builtins.max(0, builtins.min(rows, cols - offset))
        start = offset
    else:
        count = builtins.max(0, builtins.min(rows + offset, cols))
        start = -offset * cols
    if count == 0:
        return zeros(0, dtype=values.dtype)
    return flat[start:start + (count - 1) * (cols + 1) + 1:cols + 1]


def diag(v, k=0):
    values = asarray(v)
    if values.ndim == 2:
        return diagonal(values, k)
    n = values.shape[0] + builtins.abs(k)
    out = zeros((n, n), dtype=values.dtype)
    flat = out.reshape(-1)
    start = k if k >= 0 else -k * n
    flat[start:start + (values.shape[0] - 1) * (n + 1) + 1:n + 1] = values
    return out


def diagflat(v, k=0):
    return diag(asarray(v).ravel(), k)


def trace(a, offset=0):
    return diagonal(asarray(a), offset).sum()


def _row_col(n, m):
    rows = arange(n).reshape(n, 1)
    cols = arange(m).reshape(1, m)
    return rows, cols


def tri(n, m=None, k=0, dtype=None):
    m = n if m is None else m
    rows, cols = _row_col(n, m)
    mask = (cols - rows) <= k
    return mask.astype(float64 if dtype is None else dtype)


def triu(m, k=0):
    values = asarray(m)
    rows, cols = _row_col(values.shape[-2], values.shape[-1])
    return where(cols - rows >= k, values, zeros((), dtype=values.dtype))


def tril(m, k=0):
    values = asarray(m)
    rows, cols = _row_col(values.shape[-2], values.shape[-1])
    return where(cols - rows <= k, values, zeros((), dtype=values.dtype))


def tril_indices(n, k=0, m=None):
    m = n if m is None else m
    rows, cols = _row_col(n, m)
    return nonzero((cols - rows) <= k)


def triu_indices(n, k=0, m=None):
    m = n if m is None else m
    rows, cols = _row_col(n, m)
    return nonzero((cols - rows) >= k)


def vander(x, N=None, increasing=False):
    values = asarray(x, dtype=float64)
    n = values.shape[0]
    N = n if N is None else N
    out = empty((n, N), dtype=values.dtype)
    powers = builtins.range(N) if increasing else builtins.range(N - 1, -1, -1)
    for column, power_value in enumerate(powers):
        out[:, column] = values ** power_value
    return out


def meshgrid(*xi, indexing="xy"):
    arrays = [asarray(x) for x in xi]
    nd = len(arrays)
    out = []
    for i, values in enumerate(arrays):
        shape_i = [1] * nd
        shape_i[i] = values.shape[0]
        out.append(values.reshape(tuple(shape_i)))
    target = broadcast_shapes(*[o.shape for o in out])
    out = [broadcast_to(o, target).copy() for o in out]
    if indexing == "xy" and nd >= 2:
        out = [swapaxes(o, 0, 1) for o in out]
    return out


def ix_(*args):
    out = []
    nd = len(args)
    for i, values in enumerate(args):
        values = asarray(values)
        if values.dtype == bool_:
            values = flatnonzero(values)
        shape_i = [1] * nd
        shape_i[i] = values.size
        out.append(values.reshape(tuple(shape_i)).astype(int64))
    return tuple(out)


def indices(dimensions):
    dimensions = tuple(dimensions)
    nd = len(dimensions)
    out = empty((nd,) + dimensions, dtype=int64)
    for i, n in enumerate(dimensions):
        shape_i = [1] * nd
        shape_i[i] = n
        out[i] = broadcast_to(arange(n).reshape(tuple(shape_i)), dimensions)
    return out


def ndindex(*shape_args):
    if len(shape_args) == 1 and not isinstance(shape_args[0], int):
        shape_args = tuple(shape_args[0])

    def walk(prefix, rest):
        if not rest:
            yield prefix
            return
        for i in builtins.range(rest[0]):
            yield from walk(prefix + (i,), rest[1:])
    return walk((), tuple(shape_args))


def unravel_index(indices_in, shape_in):
    flat = asarray(indices_in, dtype=int64)
    out = []
    for dim in reversed(tuple(shape_in)):
        out.append(flat % dim)
        flat = flat // dim
    return tuple(reversed(out))


def ravel_multi_index(multi_index, dims):
    total = zeros((), dtype=int64)
    stride = 1
    for coord, dim in zip(reversed(list(multi_index)), reversed(tuple(dims))):
        total = total + asarray(coord, dtype=int64) * stride
        stride *= dim
    return total


def kron(a, b):
    left = atleast_2d(asarray(a))
    right = atleast_2d(asarray(b))
    m, n = left.shape
    p, q = right.shape
    out = empty((m * p, n * q), dtype=_result_dtype(left, right))
    for i in builtins.range(m):
        for j in builtins.range(n):
            out[i * p:(i + 1) * p, j * q:(j + 1) * q] = left[i, j] * right
    return out


def _result_dtype(*arrays):
    result = bool_
    for a in arrays:
        dt = asarray(a).dtype
        if result == dt or dt == bool_:
            continue
        if result == bool_:
            result = dt
        elif complex128 in (result, dt):
            result = complex128
        elif complex64 in (result, dt):
            result = complex128 if float64 in (result, dt) or int64 in (result, dt) else complex64
        elif float64 in (result, dt) or int64 in (result, dt):
            result = float64
        else:
            result = float32
    return result


def result_type(*items):
    arrays = []
    for item in items:
        if isinstance(item, (dtype, type)) or isinstance(item, str):
            arrays.append(zeros((), dtype=item))
        else:
            arrays.append(asarray(item))
    return _result_dtype(*arrays)


def promote_types(a, b):
    return result_type(a, b)


# --- predicates and comparisons -------------------------------------------

def isscalar(x):
    return isinstance(x, (int, float, complex, bool, str, bytes)) and not isinstance(x, ndarray)


def iscomplexobj(x):
    return asarray(x).dtype.kind == "c"


def isrealobj(x):
    return not iscomplexobj(x)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    x, y = asarray(a), asarray(b)
    if x.dtype == complex128 or y.dtype == complex128:
        close = absolute(x - y) <= (atol + rtol * absolute(y))
    else:
        close = absolute(x - y) <= (atol + rtol * absolute(y))
    finite = isfinite(x) & isfinite(y)
    close = close & finite
    both_inf = (isinf(x) | isinf(y)) & (x == y)
    result = close | both_inf
    if equal_nan:
        result = result | (isnan(x) & isnan(y))
    return result


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return bool(all(isclose(a, b, rtol, atol, equal_nan)))


def array_equal(a1, a2):
    x, y = asarray(a1), asarray(a2)
    if x.shape != y.shape:
        return False
    return bool(all(x == y))


def array_equiv(a1, a2):
    try:
        x, y = broadcast_arrays(asarray(a1), asarray(a2))
    except ValueError:
        return False
    return bool(all(x == y))


# --- statistics -----------------------------------------------------------

def median(a, axis=None):
    return quantile(a, 0.5) if axis is None else _apply_axis(
        lambda v: quantile(v, 0.5), a, axis)


def percentile(a, q, axis=None, method="linear"):
    fraction = asarray(q, dtype=float64) / 100.0
    return quantile(a, fraction, method=method)


def _apply_axis(fn, a, axis):
    values = asarray(a)
    axis = axis % values.ndim
    moved = moveaxis(values, axis, 0)
    return array([fn(moved[i]) for i in builtins.range(moved.shape[0])])


def cov(m, y=None, ddof=None, rowvar=True):
    x = atleast_2d(asarray(m, dtype=float64))
    if y is not None:
        x = concatenate([x, atleast_2d(asarray(y, dtype=float64))], axis=0)
    if not rowvar:
        x = x.T
    ddof = 1 if ddof is None else ddof
    n = x.shape[1]
    centred = x - mean(x, axis=1).reshape(x.shape[0], 1)
    return (centred @ centred.T) / (n - ddof)


def corrcoef(m, y=None, rowvar=True):
    c = cov(m, y, rowvar=rowvar)
    d = sqrt(diagonal(c))
    return c / outer(d, d)


def histogram(a, bins=10, range=None, density=False, weights=None):
    values = asarray(a, dtype=float64).ravel()
    if isinstance(bins, int):
        lo, hi = (float(values.min()), float(values.max())) if range is None else \
            (float(range[0]), float(range[1]))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = linspace(lo, hi, bins + 1)
    else:
        edges = asarray(bins, dtype=float64)
    nbins = edges.size - 1
    if range is not None and not isinstance(bins, int):
        pass
    index = searchsorted(edges, values, side="right") - 1
    index = clip(index, 0, nbins - 1)
    inside = (values >= edges[0]) & (values <= edges[nbins])
    index = index[inside]
    if weights is not None:
        w = asarray(weights, dtype=float64).ravel()[inside]
        counts = bincount(index, weights=w, minlength=nbins)
    else:
        counts = bincount(index, minlength=nbins)
    counts = counts[:nbins]
    if density:
        widths = diff(edges)
        counts = counts / (counts.sum() * widths)
    return counts, edges


def gradient(f, *varargs, axis=None):
    values = asarray(f, dtype=float64)
    axes = builtins.range(values.ndim) if axis is None else (
        (axis,) if isinstance(axis, int) else tuple(axis))
    spacings = list(varargs) if varargs else [1.0] * len(tuple(axes))
    if len(spacings) == 1 and len(tuple(axes)) > 1:
        spacings = spacings * len(tuple(axes))
    out = []
    for k, ax in enumerate(axes):
        h = spacings[k] if k < len(spacings) else 1.0
        moved = moveaxis(values, ax, 0)
        n = moved.shape[0]
        result = empty_like(moved)
        if isinstance(h, (int, float)):
            step = float(h)
            if n > 2:
                result[1:-1] = (moved[2:] - moved[:-2]) / (2.0 * step)
            if n >= 2:
                result[0] = (moved[1] - moved[0]) / step
                result[-1] = (moved[-1] - moved[-2]) / step
        else:
            coords = asarray(h, dtype=float64)
            if n > 2:
                dx = (coords[2:] - coords[:-2])
                result[1:-1] = (moved[2:] - moved[:-2]) / _align(dx, moved.ndim)
            if n >= 2:
                result[0] = (moved[1] - moved[0]) / (coords[1] - coords[0])
                result[-1] = (moved[-1] - moved[-2]) / (coords[-1] - coords[-2])
        out.append(moveaxis(result, 0, ax))
    return out[0] if len(out) == 1 else out


def _align(vector, ndim):
    return vector.reshape((vector.shape[0],) + (1,) * (ndim - 1))


def unique(ar, return_index=False, return_counts=False):
    values = asarray(ar).ravel()
    order = argsort(values)
    ordered = take(values, order)
    if ordered.size == 0:
        result = ordered
        first = order
        counts = zeros(0, dtype=int64)
    else:
        keep = concatenate([array([True]), ordered[1:] != ordered[:-1]])
        result = ordered[keep]
        first = order[keep]
        positions = flatnonzero(keep)
        edges = concatenate([positions, array([ordered.size], dtype=int64)])
        counts = diff(edges)
    if return_index and return_counts:
        return result, first, counts
    if return_index:
        return result, first
    if return_counts:
        return result, counts
    return result


def setdiff1d(ar1, ar2):
    left = unique(ar1)
    right = unique(ar2)
    if right.size == 0:
        return left
    position = searchsorted(right, left)
    position = clip(position, 0, right.size - 1)
    return left[take(right, position) != left]


def intersect1d(ar1, ar2):
    left = unique(ar1)
    right = unique(ar2)
    if right.size == 0 or left.size == 0:
        return left[:0]
    position = clip(searchsorted(right, left), 0, right.size - 1)
    return left[take(right, position) == left]


def union1d(ar1, ar2):
    return unique(concatenate([asarray(ar1).ravel(), asarray(ar2).ravel()]))


def in1d(ar1, ar2):
    left = asarray(ar1).ravel()
    right = unique(ar2)
    if right.size == 0:
        return zeros(left.shape, dtype=bool_)
    position = clip(searchsorted(right, left), 0, right.size - 1)
    return take(right, position) == left


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    values = asarray(arr)
    axis = axis % values.ndim
    moved = moveaxis(values, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1]) if moved.ndim > 1 else moved.reshape(1, -1)
    results = [asarray(func1d(flat[i], *args, **kwargs))
               for i in builtins.range(flat.shape[0])]
    stacked = stack(results, axis=0)
    outer_shape = moved.shape[:-1]
    result = stacked.reshape(outer_shape + stacked.shape[1:])
    if stacked.ndim == 1:
        return result
    return moveaxis(result, -1, axis) if result.ndim > len(outer_shape) else result


def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None, **ignored):
    """Set how arrays are rendered, with NumPy's option names.

    `precision` is the number of digits after the point, `suppress` keeps small
    values in fixed point instead of switching the whole array to exponential
    notation, `threshold` is the size past which an array is summarised,
    `edgeitems` how many items each end of a summary shows, and `linewidth`
    where rows wrap.
    """
    updates = {"precision": precision, "threshold": threshold,
               "edgeitems": edgeitems, "linewidth": linewidth,
               "suppress": suppress}
    for key, value in updates.items():
        if value is not None:
            _formatting.OPTIONS[key] = value


def get_printoptions():
    """The current print options, as a copy."""
    return dict(_formatting.OPTIONS)


@contextmanager
def printoptions(**kwargs):
    """Apply print options for the duration of a `with` block."""
    saved = dict(_formatting.OPTIONS)
    set_printoptions(**kwargs)
    try:
        yield get_printoptions()
    finally:
        _formatting.OPTIONS.update(saved)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Composite trapezoidal rule along one axis."""
    values = asarray(y, dtype=float64)
    moved = moveaxis(values, axis, 0) if values.ndim > 1 else values
    if x is None:
        widths = dx
        total = (moved[1:] + moved[:-1]).sum(axis=0) * (widths / 2.0)
    else:
        coords = asarray(x, dtype=float64)
        widths = diff(coords)
        if moved.ndim > 1:
            widths = widths.reshape((widths.shape[0],) + (1,) * (moved.ndim - 1))
        total = ((moved[1:] + moved[:-1]) * widths / 2.0).sum(axis=0)
    return total


trapz = trapezoid


def logaddexp(x1, x2):
    """``log(exp(x1) + exp(x2))``, computed without overflowing."""
    a = asarray(x1, dtype=float64)
    b = asarray(x2, dtype=float64)
    hi = maximum(a, b)
    lo = minimum(a, b)
    finite = isfinite(hi)
    return where(finite, hi + log1p(exp(lo - where(finite, hi, zeros((), dtype=float64)))), hi)


def _bessel_i0(x):
    """Modified Bessel function of the first kind, order zero.

    The ascending series is exact to the last bit for the arguments a Kaiser
    window uses; the asymptotic form covers the tail so the function is defined
    everywhere.
    """
    ax = asarray(absolute(asarray(x, dtype=float64)), dtype=float64)
    quarter = asarray((ax / 2.0) ** 2, dtype=float64)
    total = ones(ax.shape, dtype=float64)
    term = ones(ax.shape, dtype=float64)
    for k in range(1, 64):
        term = term * quarter / float(k * k)
        total = total + term
    large = maximum(ax, 20.0)
    u = 1.0 / (8.0 * large)
    asymptotic = (exp(large) / sqrt(2.0 * pi * large)) * (
        1.0 + u * (1.0 + u * (4.5 + u * (37.5 + u * 459.375))))
    return where(ax < 20.0, total, asymptotic)


def hanning(M):
    """Symmetric Hann window of length `M`."""
    if M < 1:
        return zeros(0, dtype=float64)
    if M == 1:
        return ones(1, dtype=float64)
    n = arange(0, M, dtype=float64)
    return 0.5 - 0.5 * cos(2.0 * pi * n / (M - 1))


def hamming(M):
    if M < 1:
        return zeros(0, dtype=float64)
    if M == 1:
        return ones(1, dtype=float64)
    n = arange(0, M, dtype=float64)
    return 0.54 - 0.46 * cos(2.0 * pi * n / (M - 1))


def blackman(M):
    if M < 1:
        return zeros(0, dtype=float64)
    if M == 1:
        return ones(1, dtype=float64)
    n = arange(0, M, dtype=float64)
    return (0.42 - 0.5 * cos(2.0 * pi * n / (M - 1))
            + 0.08 * cos(4.0 * pi * n / (M - 1)))


def bartlett(M):
    if M < 1:
        return zeros(0, dtype=float64)
    if M == 1:
        return ones(1, dtype=float64)
    n = arange(0, M, dtype=float64)
    return where(n <= (M - 1) / 2.0, 2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))


def kaiser(M, beta):
    if M < 1:
        return zeros(0, dtype=float64)
    if M == 1:
        return ones(1, dtype=float64)
    n = arange(0, M, dtype=float64) - (M - 1) / 2.0
    ratio = 2.0 * n / (M - 1)
    return _bessel_i0(beta * sqrt(1.0 - ratio * ratio)) / _bessel_i0(
        asarray(float(beta), dtype=float64))


def sort_complex(a):
    """Sorted by real part, then imaginary, as ``numpy.sort_complex`` does."""
    values = asarray(a)
    if values.dtype != complex128:
        values = values.astype(complex128)
    order = lexsort((imag(values), real(values)))
    return take(values, order)


def sinc(x):
    values = asarray(x, dtype=float64)
    scaled = pi * values
    return where(values == 0, ones((), dtype=float64), sin(scaled) / where(values == 0, ones((), dtype=float64), scaled))


def divmod_(x, y):
    return floor_divide(x, y), remainder(x, y)


divmod = divmod_
inner = _c.matmul
vdot = lambda a, b: builtins.sum(conjugate(asarray(a).ravel()) * asarray(b).ravel())


# --- numeric environment --------------------------------------------------

class _FloatInfo:
    """The fields of ``np.finfo(float)`` the library reads."""

    eps = 2.220446049250313e-16
    epsneg = 1.1102230246251565e-16
    tiny = 2.2250738585072014e-308
    smallest_normal = 2.2250738585072014e-308
    max = 1.7976931348623157e308
    min = -1.7976931348623157e308
    resolution = 1e-15
    precision = 15
    bits = 64

    def __repr__(self):
        return "finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)"


class _IntInfo:
    max = 2 ** 63 - 1
    min = -(2 ** 63)
    bits = 64


class _SingleInfo(_FloatInfo):
    eps = 2.0 ** -23
    epsneg = 2.0 ** -24
    tiny = smallest_normal = 2.0 ** -126
    max = (2.0 - 2.0 ** -23) * 2.0 ** 127
    min = -max
    resolution = 1e-6
    precision = 6
    bits = 32

    def __repr__(self):
        return "finfo(dtype=float32, eps=1.1920928955078125e-07)"


def finfo(_dtype=None):
    dt = dtype(float64 if _dtype is None else _dtype)
    if dt.kind not in ("f", "c"):
        raise ValueError("finfo requires a floating-point dtype")
    return _SingleInfo() if dt in (float32, complex64) else _FloatInfo()


def iinfo(_dtype=None):
    return _IntInfo()


@contextmanager
def errstate(**kwargs):
    """Accepted for compatibility; the compiled core never raises or warns.

    NumPy used these to silence warnings from overflow and division by zero.
    The C loops here simply produce the IEEE result, so there is nothing to
    silence -- but callers keep their ``with`` blocks.
    """
    yield


def seterr(**kwargs):
    return {"divide": "ignore", "over": "ignore", "under": "ignore", "invalid": "ignore"}


def geterr():
    return seterr()


def polyval(p, x):
    return polynomial.polyval(p, x)


def polyfit(x, y, deg, rcond=None):
    return polynomial.polyfit(x, y, deg, rcond)


poly = polynomial.poly
roots = polynomial.roots
polymul = polynomial.polymul
polydiv = polynomial.polydiv
polyadd = polynomial.polyadd
polysub = polynomial.polysub
polyint = polynomial.polyint
polyder = polynomial.polyder
trim_zeros = polynomial.trim_zeros
convolve = polynomial.convolve

frombuffer = _c.frombuffer
from ._io import save, load, open_memmap, flush


def _function_dispatch(name, native):
    """Allow differentiable array objects to own public numeric operations."""
    def call(*args, **kwargs):
        for value in args:
            if type(value) is ndarray:
                continue
            hook = getattr(value, "__quadrivium_function__", None)
            if hook is not None:
                result = hook(name, *args, **kwargs)
                if result is not NotImplemented:
                    return result
        return native(*args, **kwargs)
    call.__name__ = name
    call.__doc__ = native.__doc__
    return call


for _name in ("exp", "log", "log1p", "expm1", "sin", "cos", "tan", "tanh",
              "sqrt", "square", "absolute", "negative", "sum", "mean",
              "reshape", "transpose", "matmul"):
    globals()[_name] = _function_dispatch(_name, globals()[_name])
abs = absolute

__all__ = [name for name in dir() if not name.startswith("_")]
