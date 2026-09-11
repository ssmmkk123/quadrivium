"""Broadcast batch iteration without materializing broadcast operands."""
from itertools import product


def indices(shape):
    return product(*(range(n) for n in shape))


def broadcast_shape(*shapes):
    result = ()
    for shape in shapes:
        n = max(len(result), len(shape))
        left = (1,) * (n - len(result)) + result
        right = (1,) * (n - len(shape)) + tuple(shape)
        dims = []
        for a, b in zip(left, right):
            if a != b and a != 1 and b != 1:
                raise ValueError("batch dimensions do not broadcast")
            dims.append(b if a == 1 else a)
        result = tuple(dims)
    return result


def batch_index(index, shape):
    if not shape:
        return ()
    return tuple(0 if n == 1 else i for i, n in zip(index[-len(shape):], shape))
