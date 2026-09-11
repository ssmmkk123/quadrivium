"""Adaptive BDF and Radau with reusable simplified-Newton linear solves."""
from __future__ import annotations
import math
import operator
from .. import numeric as np
from ..core.utils import as_vector, numerical_jacobian
from ..core.types import ODESolution
from ..core.exceptions import StepSizeError
from ..core.storage import OutputRecorder, output_control

__all__ = ["bdf_adaptive", "radau_adaptive", "BandedJacobian"]


class BandedJacobian:
    """Compact Jacobian: ``bands[upper+i-j, j] == J[i,j]``.

    Matrix-vector products use O(n*(lower+upper+1)) storage/work; stiff solvers
    use restarted GMRES without forming a dense matrix.
    """
    def __init__(self, bands, lower, upper):
        self.bands = np.asarray(bands, dtype=float).copy()
        self.lower, self.upper = operator.index(lower), operator.index(upper)
        if min(self.lower, self.upper) < 0 or self.bands.ndim != 2:
            raise ValueError("invalid banded Jacobian")
        if self.bands.shape[0] != self.lower+self.upper+1:
            raise ValueError("band row count must equal lower + upper + 1")
        self.shape = (self.bands.shape[1],)*2
    def matvec(self, v):
        v = as_vector(v)
        n = self.shape[0]
        if v.size != n:
            raise ValueError("Jacobian/vector shape mismatch")
        out = np.zeros(n)
        for d in range(-self.upper, self.lower+1):
            j0, j1 = max(0, -d), min(n, n-d)
            out[j0+d:j1+d] += self.bands[self.upper+d, j0:j1]*v[j0:j1]
        return out
    def __matmul__(self, v):
        return self.matvec(v)


class _NewtonFailure(Exception):
    pass


class _Linearization:
    def __init__(self, rhs, jac, dimension, preconditioner=None):
        self.rhs, self.jac, self.n = rhs, jac, dimension
        self.preconditioner = preconditioner
        self.J, self.generation = None, 0
        self.factors = {}
        self.njev = self.nlu = self.nlinear = 0

    def refresh(self, t, y):
        if self.J is not None and self.jac is not None and (not callable(self.jac) or hasattr(self.jac, "matvec")):
            return
        if self.jac is None:
            self.J = numerical_jacobian(lambda z: self.rhs(t, z), y)
        elif callable(self.jac) and not hasattr(self.jac, "matvec"):
            self.J = self.jac(t, y)
        else:
            self.J = self.jac
        if getattr(self.J, "shape", None) != (self.n, self.n):
            raise ValueError("Jacobian must have shape (dimension, dimension)")
        self.njev += 1
        self.generation += 1
        self.factors.clear()

    def solve(self, h, A, residual):
        """Solve (I - h A ⊗ J) delta = residual, caching a bounded factor set."""
        s = len(A)
        key = (float(h), tuple(float(a) for row in A for a in row))
        self.nlinear += 1
        if isinstance(self.J, np.ndarray):
            if key not in self.factors:
                from ..linalg import lu_factor
                matrix = np.eye(s*self.n)
                for i in range(s):
                    for j in range(s):
                        matrix[i*self.n:(i+1)*self.n, j*self.n:(j+1)*self.n] -= h*A[i][j]*self.J
                if len(self.factors) >= 4:
                    self.factors.pop(next(iter(self.factors)))
                self.factors[key] = lu_factor(matrix)
                self.nlu += 1
            return self.factors[key].solve(residual)
        from ..linalg.iterative import gmres
        matvec = self.J.matvec if hasattr(self.J, "matvec") else lambda v: self.J @ v
        def apply(v):
            blocks = v.reshape((s, self.n))
            products = [matvec(blocks[j]) for j in range(s)]
            out = blocks.copy()
            for i in range(s):
                for j in range(s):
                    out[i] -= h*A[i][j]*products[j]
            return out.ravel()
        preconditioner = self.preconditioner
        if preconditioner is not None and s != 1:
            base = preconditioner
            preconditioner = lambda v: np.concatenate([base(w) for w in v.reshape((s,self.n))])
        result = gmres(apply, residual, tol=1e-10, restart=min(40, s*self.n),
                       max_iter=max(100, 4*s*self.n), M=preconditioner)
        if not result.converged:
            raise _NewtonFailure("linear iteration failed")
        return result.x


def _newton(residual, guess, linearization, h, A, scale, max_iter):
    z = guess.copy()
    for _ in range(max_iter):
        r = residual(z)
        if not np.all(np.isfinite(r)):
            raise _NewtonFailure("nonfinite Newton residual")
        if float(np.max(np.abs(r)/scale)) < 0.03:
            return z
        try:
            dz = linearization.solve(h, A, -r)
        except np.linalg.LinAlgError as exc:
            raise _NewtonFailure(str(exc)) from exc
        norm = float(np.linalg.norm(r))
        factor = 1.0
        for _ in range(8):
            candidate = z + factor*dz
            if float(np.linalg.norm(residual(candidate))) < norm:
                break
            factor *= 0.5
        else:
            raise _NewtonFailure("Newton damping failed")
        z = candidate
    raise _NewtonFailure("Newton iteration limit reached")


def _bdf_step(rhs, history, t_new, order, lin, scale, max_iter):
    order = min(order, len(history))
    h = t_new-history[0][0]
    nodes = [0.0] + [(t-t_new)/h for t,_ in history[:order]]
    system = np.array([[x**k for x in nodes] for k in range(order+1)])
    rhs_weights = np.zeros(order+1)
    rhs_weights[1] = 1.0
    weights = np.linalg.solve(system, rhs_weights)
    gamma = h/weights[0]
    base = np.zeros_like(history[0][1])
    for w,(_,y) in zip(weights[1:], history):
        base -= w/weights[0]*y
    if len(history) > 1:
        guess = history[0][1]+h/(history[0][0]-history[1][0])*(history[0][1]-history[1][1])
    else:
        guess = history[0][1]+h*rhs(history[0][0], history[0][1])
    value = _newton(lambda z: z-base-gamma*rhs(t_new,z), guess, lin,
                    gamma, [[1.0]], scale, max_iter)
    # Leading local truncation coefficient for this actual nonuniform stencil.
    coefficient = gamma*math.prod(t_new-t for t,_ in history[:order])
    propagation = -float(weights[1]/weights[0])
    return value,coefficient,propagation


def _radau_step(rhs, t, y, h, lin, scale, max_iter):
    root = math.sqrt(6.0)
    A = [[(88-7*root)/360, (296-169*root)/1800, (-2+3*root)/225],
         [(296+169*root)/1800, (88+7*root)/360, (-2-3*root)/225],
         [(16-root)/36, (16+root)/36, 1/9]]
    c = [(4-root)/10, (4+root)/10, 1.0]
    n = y.size
    f0 = rhs(t,y)
    guess = np.array([y+h*ci*f0 for ci in c]).ravel()
    def residual(flat):
        stages = flat.reshape((3,n))
        values = [rhs(t+ci*h, stages[i]) for i,ci in enumerate(c)]
        return np.array([stages[i]-y-h*sum(A[i][j]*values[j] for j in range(3))
                         for i in range(3)]).ravel()
    stages = _newton(residual, guess, lin, h, A, np.tile(scale,3), max_iter)
    return stages.reshape((3,n))[-1].copy()


def _adaptive_stiff(f, t_span, y0, method, rtol, atol, jac, h0, max_step,
                    min_step, max_steps, max_order, newton_max_iter,
                    preconditioner, checkpoint):
    y = as_vector(y0).copy()
    t0, tf = map(float, t_span)
    if y.size == 0 or not np.all(np.isfinite(y)) or not all(map(math.isfinite,(t0,tf))):
        raise ValueError("finite time interval and nonempty finite y0 required")
    rt = np.broadcast_to(np.asarray(rtol,float), y.shape)
    at = np.broadcast_to(np.asarray(atol,float), y.shape)
    if (np.any(rt<0) or np.any(at<0) or not np.all(np.isfinite(rt))
            or not np.all(np.isfinite(at)) or np.any(rt+at<=0)):
        raise ValueError("invalid component tolerances")
    max_order = operator.index(max_order)
    max_steps = operator.index(max_steps)
    newton_max_iter = operator.index(newton_max_iter)
    if not 1 <= max_order <= 5 or max_steps < 1 or newton_max_iter < 1:
        raise ValueError("max_order must be 1..5; iteration limits must be positive")
    if (max_step <= 0 or math.isnan(max_step) or min_step<=0
            or not math.isfinite(min_step) or max_step<min_step):
        raise ValueError("invalid step bounds")
    if h0 is not None and (not math.isfinite(h0) or h0==0):
        raise ValueError("h0 must be finite and nonzero")
    direction = 1 if tf>=t0 else -1
    h = direction*min(max_step, max(min_step, abs(h0) if h0 is not None else abs(tf-t0)/100))
    history, order = [(t0,y.copy())], 1 if method=="bdf_adaptive" else 5
    if checkpoint is not None:
        if checkpoint.method != method or float(checkpoint.t) != t0:
            raise ValueError("checkpoint method/time mismatch")
        if not np.array_equal(np.asarray(checkpoint.y,float),y):
            raise ValueError("checkpoint state and supplied y0 differ")
        metadata = checkpoint.metadata
        restart_h = float(metadata.get("h_next",h))
        if not math.isfinite(restart_h) or restart_h==0:
            raise ValueError("checkpoint next step must be finite and nonzero")
        h = direction*min(max_step,max(min_step,abs(restart_h)))
        if method=="bdf_adaptive":
            history = [(float(t),np.asarray(v,float)) for t,v in metadata.get("history",[(t0,y.tolist())])]
            if (not history or history[0][0]!=t0 or
                    not np.array_equal(history[0][1],y) or
                    any(not math.isfinite(ti) or yi.shape!=y.shape or
                        not np.all(np.isfinite(yi)) for ti,yi in history)):
                raise ValueError("checkpoint history has inconsistent time/state/shape")
            differences = [history[i][0]-history[i+1][0] for i in range(len(history)-1)]
            if differences and (any(d==0 for d in differences) or
                                any(d*differences[0]<=0 for d in differences)):
                raise ValueError("checkpoint history times must be strictly ordered")
            stored_order = operator.index(metadata.get("order",1))
            if not 1<=stored_order<=5:
                raise ValueError("invalid checkpoint order")
            order = min(max_order,stored_order,len(history))
            if len(history)>1 and direction*(history[0][0]-history[1][0])<0:
                history,order = history[:1],1
    calls = 0
    def rhs(t, state):
        nonlocal calls
        calls += 1
        value = as_vector(f(t,state))
        if value.shape != y.shape:
            raise ValueError("right-hand-side shape mismatch")
        return value
    lin = _Linearization(rhs,jac,y.size,preconditioner)
    recorder = OutputRecorder.current((t0,tf))
    t = t0
    recorder.append(t,y,rhs(t,y))
    accepted = rejected = failures = good_steps = 0
    max_used_order = order
    refresh_step = -8
    for _ in range(max_steps):
        if direction*(tf-t)<=0 or recorder.stopped:
            break
        h = direction*min(abs(h),abs(tf-t))
        if t+h==t or abs(h)<min_step and abs(tf-t)>min_step:
            raise StepSizeError("adaptive stiff step size underflow")
        scale = at+rt*np.maximum(np.abs(y),1e-15)
        if lin.J is None or accepted-refresh_step>=8:
            lin.refresh(t,y)
            refresh_step = accepted
        try:
            if method=="bdf_adaptive":
                coarse,coarse_coefficient,_ = _bdf_step(rhs,history,t+h,order,lin,scale,newton_max_iter)
                mid,mid_coefficient,_ = _bdf_step(rhs,history,t+h/2,order,lin,scale,newton_max_iter)
                fine,end_coefficient,propagation = _bdf_step(rhs,[(t+h/2,mid)]+history,t+h,order,lin,scale,newton_max_iter)
                fine_coefficient = end_coefficient+propagation*mid_coefficient
                difference = abs(fine_coefficient-coarse_coefficient)
                # Half steps share old history, so the Richardson denominator
                # 2**order-1 does not apply. Use the actual stencil coefficients
                # and retain at least the raw difference as a conservative floor.
                error_factor = max(1.0,abs(fine_coefficient)/difference) if difference else 1.0
            else:
                coarse = _radau_step(rhs,t,y,h,lin,scale,newton_max_iter)
                mid = _radau_step(rhs,t,y,h/2,lin,scale,newton_max_iter)
                fine = _radau_step(rhs,t+h/2,mid,h/2,lin,scale,newton_max_iter)
                error_factor = 1/(2**order-1)
            scale = at+rt*np.maximum(np.abs(y),np.abs(fine))
            error = float(np.sqrt(np.mean(((fine-coarse)*error_factor/scale)**2)))
            if not math.isfinite(error):
                raise _NewtonFailure("nonfinite local error")
        except _NewtonFailure:
            rejected += 1
            failures += 1
            good_steps = 0
            h *= 0.25
            if method=="bdf_adaptive":
                order = max(1,order-1)
            lin.refresh(t,y)
            refresh_step = accepted
            if abs(h)<min_step:
                raise StepSizeError("Newton convergence requires a step below min_step")
            continue
        if error<=1:
            t += h
            y = fine
            accepted += 1
            good_steps += 1
            recorder.append(t,y,rhs(t,y))
            history.insert(0,(t,y.copy()))
            del history[max_order+1:]
            max_used_order = max(order,max_used_order)
            if (method=="bdf_adaptive" and good_steps>=order+2
                    and order<max_order and len(history)>order):
                order += 1
                good_steps = 0
            factor = min(2.0,max(0.5,0.9*max(error,1e-12)**(-1/(order+1))))
            # Quantization retains factors on steady portions of a trajectory.
            if 0.85<factor<1.2:
                factor = 1.0
        else:
            rejected += 1
            good_steps = 0
            factor = max(0.2,0.8*error**(-1/(order+1)))
            if method=="bdf_adaptive" and error>8:
                order=max(1,order-1)
        h = direction*min(max_step,abs(h)*factor)
    ts,ys,dys = recorder.finish()
    success = direction*(tf-t)<=0
    result = ODESolution(ts,ys,method,accepted+rejected,accepted,rejected,calls,
                         success,"completed" if success else "callback stopped" if recorder.stopped
                         else "maximum number of steps reached",dydt=dys)
    result.n_jac_evals,result.n_factorizations = lin.njev,lin.nlu
    result.n_linear_solves,result.n_newton_failures = lin.nlinear,failures
    result.order,result.max_order_used = order,max_used_order
    result.checkpoint = recorder.checkpoint(method,{
        "h_next":h,"order":order,
        "history":[[ti,yi.tolist()] for ti,yi in history]})
    result._final_state = result.checkpoint.y
    return result


@output_control
def bdf_adaptive(f, t_span, y0, rtol=1e-6, atol=1e-9, jac=None, h0=None,
                 max_step=np.inf, min_step=1e-14, max_steps=100000,
                 max_order=5, newton_max_iter=12, preconditioner=None,
                 checkpoint=None):
    """Variable-step, variable-order BDF1--5 with step-doubling error estimates.

    Dense, CSR, LinearOperator, and BandedJacobian Jacobians are accepted.
    Dense systems reuse LU factors; sparse/operator systems use restarted GMRES.
    Newton failures reject the step and refresh the Jacobian. Optional
    ``preconditioner(v)`` approximately solves the Newton system.
    """
    return _adaptive_stiff(f,t_span,y0,"bdf_adaptive",rtol,atol,jac,h0,max_step,
                           min_step,max_steps,max_order,newton_max_iter,
                           preconditioner,checkpoint)


@output_control
def radau_adaptive(f, t_span, y0, rtol=1e-6, atol=1e-9, jac=None, h0=None,
                   max_step=np.inf, min_step=1e-14, max_steps=100000,
                   newton_max_iter=12, preconditioner=None, checkpoint=None):
    """Adaptive fifth-order, three-stage Radau IIA with step-doubling control.

    Uses the same component tolerances, structured Jacobians, output controls,
    restart checkpoints, and Newton failure recovery as :func:`bdf_adaptive`.
    """
    return _adaptive_stiff(f,t_span,y0,"radau_adaptive",rtol,atol,jac,h0,max_step,
                           min_step,max_steps,5,newton_max_iter,preconditioner,
                           checkpoint)
