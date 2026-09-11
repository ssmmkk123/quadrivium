"""Forward variational equations and checkpointed continuous adjoints."""
from __future__ import annotations
from dataclasses import dataclass
import operator
from .. import numeric as np
from ..core.utils import as_vector, numerical_jacobian, numerical_gradient
from .explicit import solve_ivp

__all__ = ["SensitivitySolution", "AdjointResult", "solve_ivp_sensitivities",
           "adjoint_sensitivity"]


@dataclass
class SensitivitySolution:
    """State trajectory and derivatives with respect to parameters and y0."""
    solution: object
    sensitivities: object
    initial_sensitivities: object = None
    sensitivity_final: object = None
    initial_sensitivity_final: object = None
    augmented_checkpoint: object = None
    @property
    def t(self):
        return self.solution.t
    @property
    def y(self):
        return self.solution.y
    @property
    def y_final(self):
        return self.solution.y_final
    @property
    def success(self):
        return self.solution.success


@dataclass
class AdjointResult:
    """Terminal objective, parameter gradient, and initial-state gradient."""
    value: float
    gradient: object
    initial_gradient: object
    y_final: object
    checkpoints: int
    recomputed_steps: int
    success: bool = True


def _jacobians(f, t, y, parameters, jac_y, jac_p):
    Jy = (np.asarray(jac_y(t,y,parameters),float) if jac_y is not None else
          numerical_jacobian(lambda z: f(t,z,parameters),y))
    Jp = (np.asarray(jac_p(t,y,parameters),float) if jac_p is not None else
          numerical_jacobian(lambda p: f(t,y,p),parameters))
    if Jy.shape!=(y.size,y.size) or Jp.shape!=(y.size,parameters.size):
        raise ValueError("sensitivity Jacobian shape mismatch")
    return Jy,Jp


def solve_ivp_sensitivities(f, t_span, y0, parameters, *, jac_y=None, jac_p=None,
                            initial=False, initial_sensitivity=None,
                            method="dormand_prince", **kwargs):
    """Integrate y'=f(t,y,p) together with parameter/initial-state sensitivities.

    Analytic ``jac_y`` and ``jac_p`` are optional; finite differences otherwise
    approximate only the local RHS derivatives. ``initial_sensitivity`` gives
    dy0/dp when the initial condition itself depends on parameters. Output
    controls apply to the augmented system. Component tolerances may be supplied
    for that complete augmented state. ``sensitivity_final`` preserves endpoint
    derivatives even when save_at omits the endpoint. ``augmented_checkpoint``
    is the underlying variational-system state, not a state-only IVP checkpoint.
    """
    y0,p = as_vector(y0),as_vector(parameters)
    n,m = y0.size,p.size
    S0 = np.zeros((n,m)) if initial_sensitivity is None else np.asarray(initial_sensitivity,float)
    if S0.shape!=(n,m):
        raise ValueError("initial_sensitivity must have shape (state, parameters)")
    columns = m+n if initial else m
    S = np.concatenate([S0,np.eye(n)],axis=1) if initial else S0
    z0 = np.concatenate([y0,S.ravel()])
    def rhs(t,z):
        y,S = z[:n],z[n:].reshape((n,columns))
        Jy,Jp = _jacobians(f,t,y,p,jac_y,jac_p)
        dS = Jy@S
        dS[:,:m] += Jp
        return np.concatenate([as_vector(f(t,y,p)),dS.ravel()])
    solution = solve_ivp(rhs,t_span,z0,method=method,**kwargs)
    sensitivity = solution.y[:,n:].reshape((len(solution.t),n,columns)).copy()
    final_sensitivity = solution.y_final[n:].reshape((n,columns)).copy()
    augmented_checkpoint = solution.checkpoint
    # The integrator checkpoint belongs to the complete variational system;
    # exposing it as a state-only checkpoint would silently restart the wrong IVP.
    solution.checkpoint = None
    solution.y = solution.y[:,:n].copy()
    if hasattr(solution, "_final_state"):
        solution._final_state = solution._final_state[:n].copy()
    if solution.dydt is not None:
        solution.dydt = solution.dydt[:,:n].copy()
    if solution.interpolant is not None:
        interpolant = solution.interpolant
        solution.interpolant = lambda t: interpolant(t)[:,:n]
    return SensitivitySolution(solution,sensitivity[:,:,:m],
                               sensitivity[:,:,m:] if initial else None,
                               final_sensitivity[:,:m],
                               final_sensitivity[:,m:] if initial else None,
                               augmented_checkpoint)


def adjoint_sensitivity(f, t_span, y0, parameters, terminal, *, terminal_y=None,
                        terminal_p=None, jac_y=None, jac_p=None, checkpoints=16,
                        method="dormand_prince", rtol=1e-8, atol=1e-10,
                        max_step=np.inf):
    """Gradient of terminal(y(tf), p) via checkpointed continuous adjoints.

    Store ``checkpoints+1`` states, then replay one segment at a time during the
    backward solve. Memory is O(checkpoints*state + longest segment trajectory),
    rather than O(full forward trajectory). The model must be deterministic.
    This differentiates the continuous IVP, not adaptive step-size decisions.
    Terminal gradients and RHS Jacobians default to local finite differences.
    """
    count = operator.index(checkpoints)
    if count<1:
        raise ValueError("checkpoints must be positive")
    p,y = as_vector(parameters),as_vector(y0).copy()
    n,m = y.size,p.size
    times = np.linspace(float(t_span[0]),float(t_span[1]),count+1)
    saved = [y.copy()]
    options = dict(method=method,rtol=rtol,atol=atol,max_step=max_step)
    rhs = lambda t,z: f(t,z,p)
    for i in range(count):
        segment = solve_ivp(rhs,(times[i],times[i+1]),y,final_only=True,**options)
        if not segment.success:
            raise RuntimeError("forward sensitivity integration failed")
        y = segment.y_final.copy()
        saved.append(y)
    value = float(terminal(y,p))
    lam = (as_vector(terminal_y(y,p)) if terminal_y is not None else
           numerical_gradient(lambda z: terminal(z,p),y))
    gradient = (as_vector(terminal_p(y,p)) if terminal_p is not None else
                numerical_gradient(lambda q: terminal(y,q),p))
    if lam.size!=n or gradient.size!=m:
        raise ValueError("terminal derivative shape mismatch")
    augmented = np.concatenate([lam,gradient])
    recomputed = 0
    for i in range(count-1,-1,-1):
        replay = solve_ivp(rhs,(times[i],times[i+1]),saved[i],**options)
        if not replay.success:
            raise RuntimeError("checkpoint replay failed")
        recomputed += replay.n_steps
        def adjoint_rhs(t,z):
            yt = replay(t)
            Jy,Jp = _jacobians(f,t,yt,p,jac_y,jac_p)
            return np.concatenate([-Jy.T@z[:n],-Jp.T@z[:n]])
        backward = solve_ivp(adjoint_rhs,(times[i+1],times[i]),augmented,
                             final_only=True,**options)
        if not backward.success:
            raise RuntimeError("adjoint integration failed")
        augmented = backward.y_final.copy()
    return AdjointResult(value,augmented[n:],augmented[:n],y,count+1,recomputed)
