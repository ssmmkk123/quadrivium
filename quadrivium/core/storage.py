"""Bounded trajectory recording and portable solver checkpoints.

Output controls select what is retained *during* integration. Callbacks receive
an independent state copy, so retaining or editing it cannot corrupt a solver.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import inspect
import json
import math
import operator

from .. import numeric as np

__all__ = ["SolverCheckpoint", "OutputRecorder", "resume_ode", "resume_pde"]
_OPTIONS = ContextVar("quadrivium_output_options", default={})
_KEYS = ("save_at", "save_every", "final_only", "callback")


@dataclass
class SolverCheckpoint:
    """JSON-serializable restart state; model functions are supplied on restart.

    ``metadata`` holds step size/order/history when an integrator needs them.
    This stores numerical state, never executable Python objects or pickle data.
    """
    t: float
    y: object
    method: str = ""
    metadata: dict = field(default_factory=dict)
    version: int = 1

    def to_dict(self):
        return {"version": self.version, "t": float(self.t),
                "y": np.asarray(self.y).tolist(), "method": self.method,
                "metadata": self.metadata}

    def to_json(self):
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_json(cls, value):
        d = json.loads(value)
        if d.get("version") != 1:
            raise ValueError("unsupported checkpoint version")
        return cls(float(d["t"]), np.asarray(d["y"], dtype=float),
                   str(d["method"]), dict(d.get("metadata", {})))


class OutputRecorder:
    """Record selected states using O(state size + requested output size) memory.

    ``save_at`` is ordered in integration direction, and uses cubic Hermite
    interpolation when endpoint slopes are supplied (linear otherwise).
    ``save_every`` retains the initial state, every kth step, and the endpoint.
    ``final_only`` retains one state. A callback is called on accepted states;
    return True to request termination, and None/False to continue.
    """
    def __init__(self, t_span, *, save_at=None, save_every=1,
                 final_only=False, callback=None):
        self.t0, self.tf = map(float, t_span)
        self.direction = 1 if self.tf >= self.t0 else -1
        self.every = operator.index(save_every)
        if self.every < 1:
            raise ValueError("save_every must be a positive integer")
        if final_only and save_at is not None:
            raise ValueError("final_only and save_at are mutually exclusive")
        if save_at is not None and self.every != 1:
            raise ValueError("save_at and save_every are mutually exclusive")
        self.requested = None if save_at is None else [float(t) for t in save_at]
        if self.requested is not None:
            if not self.requested:
                raise ValueError("save_at cannot be empty")
            for i, t in enumerate(self.requested):
                if (not math.isfinite(t) or (t-self.t0)*self.direction < 0
                        or (self.tf-t)*self.direction < 0
                        or (i and (t-self.requested[i-1])*self.direction <= 0)):
                    raise ValueError("save_at must be finite, strictly ordered, and inside t_span")
        self.final_only, self.callback = bool(final_only), callback
        self.times, self.states, self.slopes = [], [], []
        self.previous = None
        self.steps, self.cursor = -1, 0
        self.stopped = False

    @classmethod
    def current(cls, t_span, **overrides):
        options = dict(_OPTIONS.get())
        options.update(overrides)
        return cls(t_span, **options)

    def _save(self, t, y, slope=None):
        self.times.append(float(t))
        self.states.append(np.asarray(y).copy())
        self.slopes.append(None if slope is None else np.asarray(slope).copy())

    def append(self, t, y, slope=None):
        t, y = float(t), np.asarray(y)
        self.steps += 1
        if self.callback is not None:
            if self.callback(t, y.copy()) is True:
                self.stopped = True
        if self.final_only:
            self.times.clear(); self.states.clear(); self.slopes.clear()
            self._save(t, y, slope)
        elif self.requested is None:
            if self.steps % self.every == 0:
                self._save(t, y, slope)
        else:
            while self.cursor < len(self.requested):
                tq = self.requested[self.cursor]
                if (tq-t)*self.direction > 8*math.ulp(max(abs(t), 1.0)):
                    break
                if self.previous is None or tq == t:
                    value, derivative = y, slope
                else:
                    ta, ya, da = self.previous
                    h = t-ta
                    s = (tq-ta)/h
                    if da is None or slope is None:
                        value, derivative = (1-s)*ya+s*y, None
                    else:
                        value = ((2*s**3-3*s*s+1)*ya + (s**3-2*s*s+s)*h*da
                                 + (-2*s**3+3*s*s)*y + (s**3-s*s)*h*slope)
                        derivative = ((6*s*s-6*s)*ya/h + (3*s*s-4*s+1)*da
                                      + (-6*s*s+6*s)*y/h + (3*s*s-2*s)*slope)
                self._save(tq, value, derivative)
                self.cursor += 1
        self.previous = (t, y.copy(), None if slope is None else np.asarray(slope).copy())
        return not self.stopped

    def finish(self):
        if self.previous is None:
            raise ValueError("no solver states recorded")
        t, y, d = self.previous
        if self.requested is None and (not self.times or self.times[-1] != t):
            self._save(t, y, d)
        if not self.times:  # interrupted before the first requested output
            self._save(t, y, d)
        slopes = None if any(d is None for d in self.slopes) else np.array(self.slopes)
        return np.array(self.times), np.array(self.states), slopes

    def checkpoint(self, method="", metadata=None):
        t, y, _ = self.previous
        return SolverCheckpoint(t, y.copy(), method, metadata or {})


class TimeGrid:
    """A uniform time grid that does not allocate one value per integration step."""
    def __init__(self, start, stop, count):
        self.start, self.stop = float(start), float(stop)
        self.count = operator.index(count)
        if self.count < 2:
            raise ValueError("at least one time step is required")
        self.step = (self.stop-self.start)/(self.count-1)
    def __len__(self):
        return self.count
    def __getitem__(self, i):
        if isinstance(i, slice):
            return np.array([self[j] for j in range(*i.indices(self.count))])
        i = operator.index(i)
        if i < 0:
            i += self.count
        if i < 0 or i >= self.count:
            raise IndexError(i)
        return self.stop if i == self.count-1 else self.start+i*self.step
    def __iter__(self):
        return (self[i] for i in range(self.count))


class Trajectory:
    """Write-only indexed trajectory sink for existing time-stepping loops."""
    def __init__(self, times):
        self.grid = times
        self.recorder = OutputRecorder.current((times[0], times[-1]))
        self.recent = []
    def __setitem__(self, index, value):
        if isinstance(index, slice):
            for i, state in zip(range(*index.indices(len(self.grid))), value):
                self[i] = state
            return
        self.recorder.append(self.grid[index], value)
        self.recent.append((int(index), np.asarray(value).copy()))
        if len(self.recent) > 8:
            self.recent.pop(0)
    def __getitem__(self, index):
        if isinstance(index, slice):
            return np.array([self[i] for i in range(*index.indices(len(self.grid)))])
        if index < 0:
            index += len(self.grid)
        for k, state in self.recent:
            if k == index:
                return state
        raise IndexError("only the eight latest working states are retained")


def output_control(function):
    """Expose common keyword-only output controls on a time-dependent solver."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        options = dict(_OPTIONS.get())
        for key in _KEYS:
            if key in kwargs:
                options[key] = kwargs.pop(key)
        token = _OPTIONS.set(options)
        try:
            return function(*args, **kwargs)
        finally:
            _OPTIONS.reset(token)
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    insertion = next((i for i,p in enumerate(parameters)
                      if p.kind == p.VAR_KEYWORD), len(parameters))
    for key, default in zip(_KEYS, (None, 1, False, None)):
        if key not in signature.parameters:
            parameters.insert(insertion, inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY,
                                                          default=default))
            insertion += 1
    wrapped.__signature__ = signature.replace(parameters=parameters)
    return wrapped


def pde_solution(u, grids=(), t=None, method="", *args, **kwargs):
    from .types import PDESolution
    if isinstance(u, Trajectory):
        ts, states, _ = u.recorder.finish()
        result = PDESolution(states, grids, ts, method, *args, **kwargs)
        metadata = {"dt": u.grid[1]-u.grid[0],
                    "recent": [state.tolist() for _,state in u.recent]}
        result.checkpoint = u.recorder.checkpoint(method, metadata)
        if u.recorder.stopped:
            result.converged = False
            result.message = "callback stopped"
        result._final_state = result.checkpoint.y
        return result
    return PDESolution(u, grids, t, method, *args, **kwargs)


def resume_ode(f, checkpoint, t_final, *, solver=None, **kwargs):
    """Restart an ODE, supplying its model again instead of serializing callables.

    Adaptive BDF/Radau restore step/order/history. Other methods restart from
    the saved state, with their usual self-starting initialization; this is not
    a bitwise continuation of multistep history. ``solver`` can be an adapter
    with signature ``solver(f, t_span, y0, **options)`` for split Hamiltonian or
    exponential models that need extra model arguments. Delay equations restore
    their recorded delay window; supply ``history=`` for earlier times if needed.
    """
    from .. import ode
    if isinstance(checkpoint, str):
        checkpoint = SolverCheckpoint.from_json(checkpoint)
    method = checkpoint.method
    span = (checkpoint.t,t_final)
    if solver is not None:
        return solver(f,span,checkpoint.y,**kwargs)
    if method=="dde_method_of_steps":
        from .types import ODESolution
        original_history = kwargs.pop("history",None)
        segments = [ODESolution(np.array(seg["t"]),np.array(seg["y"]),
                    dydt=None if seg["dydt"] is None else np.array(seg["dydt"]))
                    for seg in checkpoint.metadata["segments"]]
        def history(t):
            for segment in segments:
                if segment.t[0]<=t<=segment.t[-1]:
                    return segment(t)
            if original_history is not None:
                return original_history(t)
            raise ValueError("restart needs history(t) before the recorded delay window")
        return ode.dde_method_of_steps(f,history,checkpoint.metadata["delays"],span,**kwargs)
    adaptive = method in ode.BUTCHER_TABLEAUX or method in {
        "bdf_adaptive","radau_adaptive","gragg_bulirsch_stoer"}
    if method in {"bdf_adaptive","radau_adaptive"}:
        kwargs.setdefault("checkpoint",checkpoint)
    elif adaptive and "h_next" in checkpoint.metadata:
        kwargs.setdefault("h0",checkpoint.metadata["h_next"])
    if not adaptive and "n" not in kwargs:
        dt = checkpoint.metadata.get("dt",checkpoint.metadata.get("h_next"))
        if dt is not None and dt!=0:
            kwargs["n"] = max(1,int(round(abs((t_final-checkpoint.t)/dt))))
    if method=="gragg_bulirsch_stoer":
        return ode.gragg_bulirsch_stoer(f,span,checkpoint.y,**kwargs)
    if method in {"rk_nystrom","stormer_cowell"}:
        return getattr(ode,method)(f,span,checkpoint.y,
                                  checkpoint.metadata["velocity"],**kwargs)
    if method.startswith("dae_index1_bdf"):
        size = checkpoint.metadata["differential_size"]
        g = kwargs.pop("g")
        kwargs.setdefault("order",checkpoint.metadata["order"])
        return ode.dae_index1_bdf(f,g,span,checkpoint.y[:size],checkpoint.y[size:],**kwargs)
    if method=="mass_matrix_ode":
        return ode.mass_matrix_ode(kwargs.pop("M"),f,span,checkpoint.y,**kwargs)
    for prefix in ("bdf","adams_bashforth","adams_moulton","predictor_corrector"):
        suffix = method[len(prefix):] if method.startswith(prefix) else ""
        if suffix.isdigit():
            kwargs.setdefault("order",int(suffix))
            return getattr(ode,prefix)(f,span,checkpoint.y,**kwargs)
    if method.startswith("variable_adams"):
        kwargs.pop("n",None)
        kwargs.setdefault("order",int(method[len("variable_adams"):]))
        return ode.variable_step_adams(f,span,checkpoint.y,**kwargs)
    known = set(ode.BUTCHER_TABLEAUX)|{"bdf_adaptive","radau_adaptive","euler",
             "heun","midpoint","ralston","rk3","rk4","rk38",
             "backward_euler","trapezoidal","implicit_midpoint"}
    if method not in known:
        raise ValueError("this model requires an explicit solver(f,t_span,y0,**options) restart adapter")
    kwargs.setdefault("method",method)
    return ode.solve_ivp(f,span,checkpoint.y,**kwargs)


def resume_pde(solver, checkpoint, t_final, **kwargs):
    """Restart a time-dependent PDE with its model/grid options resupplied.

    Wave and leapfrog solvers restore their two-level recurrence exactly when
    the new interval contains an integer number of previous time steps. Other
    first-order-in-time methods restart directly from the saved field.
    """
    if isinstance(checkpoint, str):
        checkpoint = SolverCheckpoint.from_json(checkpoint)
    signature = inspect.signature(solver)
    if "nt" in signature.parameters and "nt" not in kwargs:
        dt = abs(float(checkpoint.metadata.get("dt",t_final-checkpoint.t)))
        count = abs(t_final-checkpoint.t)/dt
        kwargs["nt"] = max(1,int(round(count)) if abs(count-round(count))<1e-10 else int(math.ceil(count)))
    if "wave" in checkpoint.method or "leapfrog" in checkpoint.method:
        if "checkpoint" not in signature.parameters:
            raise ValueError("the supplied solver does not support recurrence checkpoints")
        kwargs["checkpoint"] = checkpoint
        if "v0" in signature.parameters:
            kwargs.setdefault("v0",np.zeros_like(checkpoint.y))
    if "v0" in signature.parameters and checkpoint.method=="navier_stokes_projection":
        return solver(checkpoint.y[0],checkpoint.y[1],t_span=(checkpoint.t,t_final),**kwargs)
    return solver(checkpoint.y,t_span=(checkpoint.t,t_final),**kwargs)


def ode_solution(t, y, method="", *args, **kwargs):
    """Finalize a rolling ODE trajectory without materializing discarded states."""
    from .types import ODESolution
    if isinstance(y, Trajectory):
        ts, states, slopes = y.recorder.finish()
        result = ODESolution(ts, states, method, *args, **kwargs)
        if result.dydt is None:
            result.dydt = slopes
        result.n_steps = result.n_accepted = y.recorder.steps
        if y.recorder.stopped:
            result.success = False
            result.message = "callback stopped"
        result.checkpoint = y.recorder.checkpoint(method, {
            "dt": y.grid[1]-y.grid[0],
            "recent": [state.tolist() for _,state in y.recent]})
        result._final_state = result.checkpoint.y
        return result
    return ODESolution(t, y, method, *args, **kwargs)
