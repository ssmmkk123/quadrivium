"""Shared optional optimizer histories and iteration callbacks.

Policies are context-local, so concurrent tasks can select different retention.
Nested optimizers inherit the memory policy; only the outer history invokes the
callback. Copies are made only for retained snapshots or callback arguments.
"""
from contextvars import ContextVar
from functools import wraps
from inspect import Parameter, getdoc, signature
from operator import index

from ..core.types import OptimizeResult

_policy = ContextVar("quadrivium_optimization_history", default=None)


def _snapshot(value):
    return value.copy() if hasattr(value, "copy") else value


class _Stopped(Exception):
    def __init__(self, x, history):
        self.x, self.history = _snapshot(x), history


class History(list):
    def __init__(self, initial=()):
        super().__init__()
        self.policy = _policy.get()
        self.steps = 0
        if self.policy is not None and self.policy["owner"] is None:
            self.policy["owner"] = id(self)
        self.owner = self.policy is None or self.policy["owner"] == id(self)
        if self.policy is None or (self.owner and self.policy["store"]):
            self.extend(_snapshot(value) for value in initial)

    def append(self, value, *, x=None):
        self.steps += 1
        policy = self.policy
        if policy is None or (self.owner and policy["store"] and self.steps % policy["stride"] == 0):
            super().append(_snapshot(value))
        if policy is not None and self.owner and policy["callback"] is not None:
            point = value if x is None else x
            try:
                stop = policy["callback"](_snapshot(point))
            except StopIteration:
                stop = True
            if stop:
                raise _Stopped(point, self)


def monitor(function):
    """Add keyword-only retention/callback controls without changing defaults."""
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
        inherited = _policy.get()
        explicit = any(name in kwargs for name in ("store_history", "history_stride", "callback"))
        store = kwargs.pop("store_history", True)
        stride = kwargs.pop("history_stride", 1)
        callback = kwargs.pop("callback", None)
        # An outer policy applies across delegating aliases and inner solvers.
        if inherited is not None and not explicit:
            return function(*args, **kwargs)
        stride = index(stride)
        if stride < 1:
            raise ValueError("history_stride must be a positive integer")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        token = _policy.set({"store": bool(store), "stride": stride,
                             "callback": callback, "owner": None})
        try:
            result = function(*args, **kwargs)
            if isinstance(getattr(result, "history", None), History):
                result.history = list(result.history)
            return result
        except _Stopped as stopped:
            result = OptimizeResult(stopped.x, iterations=stopped.history.steps,
                                    converged=False, method=function.__name__,
                                    history=list(stopped.history), message="stopped by callback")
            # Report the objective where the public routine accepts one. Other
            # algorithms (proximal operators, LP) leave the optional cost unset.
            bound = sig.bind_partial(*args, **kwargs).arguments
            objective = bound.get("f")
            if callable(objective):
                result.fun = float(objective(stopped.x))
            return result
        finally:
            _policy.reset(token)

    wrapped.__signature__ = sig.replace(parameters=parameters)
    wrapped.__doc__ = (getdoc(function) or "") + (
        "\n\nOptional controls: store_history=False disables snapshots; history_stride\n"
        "retains every Nth snapshot; callback(x) receives a private iterate copy.\n"
        "Return True or raise StopIteration from the callback to stop.\n")
    return wrapped
