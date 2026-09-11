"""Bounded residual retention for iterative linear solvers."""
from contextvars import ContextVar
from functools import wraps
from inspect import Parameter, signature
from operator import index

from ..core.types import IterationResult

_policy = ContextVar("quadrivium_linear_history", default=None)


class _Stopped(Exception):
    def __init__(self, x, history, iteration):
        self.x, self.history, self.iteration = x, history, iteration


class ResidualHistory(list):
    """Keep current/first residual for stopping tests independently of samples."""
    def __init__(self, initial=()):
        super().__init__()
        self.policy = _policy.get() or {"store": True, "stride": 1, "callback": None}
        self.count, self.first, self.last = 0, None, None
        self.saved_count = 0
        for value in initial:
            self.append(value)

    def append(self, value, *, x=None, iteration=None):
        self.count += 1
        if self.first is None:
            self.first = value
        self.last = value
        if not self.policy["store"]:
            self.clear()
            super().append(value)
            self.saved_count = self.count
        elif self.count == 1 or (self.count - 1) % self.policy["stride"] == 0:
            super().append(value)
            self.saved_count = self.count
        callback = self.policy["callback"]
        if callback is not None and x is not None:
            point = x() if callable(x) else x
            try:
                stop = callback(point.copy())
            except StopIteration:
                stop = True
            if stop:
                raise _Stopped(point.copy(), self,
                               self.count if iteration is None else iteration)

    def __getitem__(self, key):
        if key == -1:
            return self.last
        return super().__getitem__(key)

    def finish(self):
        if self.last is not None and self.saved_count != self.count:
            super().append(self.last)
        return list(self)


def monitor(function):
    sig = signature(function)
    parameters = list(sig.parameters.values())
    position = next((i for i, p in enumerate(parameters) if p.kind == Parameter.VAR_KEYWORD), len(parameters))
    parameters[position:position] = [
        Parameter("store_history", Parameter.KEYWORD_ONLY, default=True),
        Parameter("history_stride", Parameter.KEYWORD_ONLY, default=1),
        Parameter("callback", Parameter.KEYWORD_ONLY, default=None),
    ]

    @wraps(function)
    def wrapped(*args, **kwargs):
        store = kwargs.pop("store_history", True)
        stride = index(kwargs.pop("history_stride", 1))
        callback = kwargs.pop("callback", None)
        if stride < 1:
            raise ValueError("history_stride must be positive")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        token = _policy.set({"store": bool(store), "stride": stride, "callback": callback})
        try:
            result = function(*args, **kwargs)
        except _Stopped as stop:
            result = IterationResult(stop.x, stop.iteration, False, stop.history,
                                     function.__name__, "stopped by callback")
        finally:
            _policy.reset(token)
        if isinstance(result.residuals, ResidualHistory):
            result.residuals = result.residuals.finish()
        return result

    wrapped.__signature__ = sig.replace(parameters=parameters)
    wrapped.__doc__ = (function.__doc__ or "") + "\n\n    store_history=False retains only the latest residual; history_stride keeps\n    every Nth residual plus the first and last. callback(x) receives a private\n    iterate snapshot; return True or raise StopIteration to stop.\n"
    return wrapped
