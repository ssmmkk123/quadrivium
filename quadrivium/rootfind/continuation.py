"""Pseudo-arclength continuation through folds of nonlinear equilibrium branches."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import operator
from .. import numeric as np
from ..core import numerical_jacobian

__all__ = ["ContinuationResult", "pseudo_arclength"]


@dataclass
class ContinuationResult:
    x: object
    parameters: object
    tangents: object
    residuals: object
    converged: bool
    message: str
    function_calls: int
    bifurcations: list = field(default_factory=list)
    eigenvalues: object = None
    stable: object = None


def pseudo_arclength(f, x0, parameter0, *, jac=None, parameter_derivative=None,
                     ds=0.05, max_steps=100, direction=1, tol=1e-9,
                     min_step=1e-6, max_step=0.2, max_newton=12,
                     stability=False, callback=None):
    """Track ``f(x, parameter)=0`` with a bordered predictor-corrector solve.

    ``jac`` supplies dF/dx and ``parameter_derivative`` supplies dF/dparameter;
    finite differences are used otherwise. Tangent orientation is continuous
    through folds. Step sizes adapt to corrector work. ``stability=True`` treats
    F as a dynamical-system RHS and classifies Re(eigenvalue(dF/dx)) < 0.
    Fold and stability-change locations are interpolated candidates, not certified
    bifurcation points. A callback returning True stops after an accepted point.
    """
    max_steps, max_newton = operator.index(max_steps), operator.index(max_newton)
    if (max_steps < 0 or max_newton < 1 or not 0 < min_step <= ds <= max_step or
            not all(math.isfinite(v) for v in (ds, min_step, max_step, tol, parameter0)) or tol <= 0 or direction not in (-1, 1)):
        raise ValueError("invalid continuation steps, direction, or tolerances")
    x = np.asarray(x0, dtype=float).ravel().copy()
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a nonempty finite vector")
    calls = 0
    def F(z):
        nonlocal calls
        calls += 1
        result = np.asarray(f(z[:-1], float(z[-1])), dtype=float).ravel()
        if result.shape != x.shape or not np.all(np.isfinite(result)):
            raise ValueError("residual must be a finite vector matching x0")
        return result
    def augmented_jac(z):
        J = numerical_jacobian(lambda y: F(np.concatenate((y, z[-1:]))), z[:-1]) if jac is None else np.asarray(jac(z[:-1], float(z[-1])), dtype=float)
        if J.shape != (x.size, x.size):
            raise ValueError("Jacobian must have shape (n,n)")
        if parameter_derivative is None:
            h = 6e-6 * max(1.0, abs(float(z[-1])))
            plus, minus = z.copy(), z.copy()
            plus[-1] += h
            minus[-1] -= h
            dp = (F(plus) - F(minus)) / (2 * h)
        else:
            dp = np.asarray(parameter_derivative(z[:-1], float(z[-1])), dtype=float).ravel()
        if dp.shape != x.shape:
            raise ValueError("parameter derivative must have shape (n,)")
        return np.column_stack((J, dp))
    z = np.concatenate((x, [float(parameter0)]))
    # Correct the starting equilibrium at the user's fixed parameter.
    for _ in range(max_newton):
        residual = F(z)
        if float(np.linalg.norm(residual)) <= tol:
            break
        z[:-1] -= np.linalg.solve(augmented_jac(z)[:, :-1], residual)
    if float(np.linalg.norm(F(z))) > tol:
        raise ValueError("initial equilibrium did not converge at parameter0")
    J = augmented_jac(z)
    _, _, vh = np.linalg.svd(J, full_matrices=True)
    tangent = vh[-1].copy()
    tangent /= np.linalg.norm(tangent)
    if tangent[-1] * direction < 0:
        tangent = -tangent
    states, tangents, residuals = [z.copy()], [tangent.copy()], [float(np.linalg.norm(F(z)))]
    eigenvalues, bifurcations = [], []
    if stability:
        eigenvalues.append(np.linalg.eigvals(J[:, :-1]))
    converged, message, step = True, "requested branch steps completed", ds
    for _ in range(max_steps):
        accepted = False
        while step >= min_step:
            prediction = z + step * tangent
            trial = prediction.copy()
            for iteration in range(max_newton):
                value = np.concatenate((F(trial), [np.dot(tangent, trial - prediction)]))
                norm0 = float(np.linalg.norm(value))
                if norm0 <= tol:
                    accepted = True
                    break
                bordered = np.vstack((augmented_jac(trial), tangent))
                try:
                    correction = np.linalg.solve(bordered, -value)
                except np.linalg.LinAlgError:
                    break
                factor = 1.0
                for _backtrack in range(12):
                    candidate = trial + factor * correction
                    new = np.concatenate((F(candidate), [np.dot(tangent, candidate - prediction)]))
                    if float(np.linalg.norm(new)) < norm0:
                        trial = candidate
                        break
                    factor *= 0.5
                else:
                    break
            if not accepted:
                # The last permitted Newton correction may already have met
                # the target, without another loop entry to inspect it.
                final_value = np.concatenate((F(trial), [np.dot(tangent, trial - prediction)]))
                accepted = float(np.linalg.norm(final_value)) <= tol
            if accepted:
                break
            step *= 0.5
        if not accepted:
            converged, message = False, "corrector failed at minimum branch step"
            break
        J = augmented_jac(trial)
        bordered = np.vstack((J, tangent))
        rhs = np.zeros(x.size + 1)
        rhs[-1] = 1
        try:
            new_tangent = np.linalg.solve(bordered, rhs)
        except np.linalg.LinAlgError:
            converged, message = False, "branch tangent is singular; possible branch point"
            break
        new_tangent /= np.linalg.norm(new_tangent)
        if np.dot(new_tangent, tangent) < 0:
            new_tangent = -new_tangent
        if float(tangent[-1] * new_tangent[-1]) < 0:
            weight = abs(float(tangent[-1])) / (abs(float(tangent[-1])) + abs(float(new_tangent[-1])))
            point = z + weight * (trial - z)
            bifurcations.append({"type": "fold", "step": len(states), "x": point[:-1].copy(),
                                 "parameter": float(point[-1]), "estimated": True})
        if stability:
            eig = np.linalg.eigvals(J[:, :-1])
            old_max = float(np.max(np.real(eigenvalues[-1])))
            new_max = float(np.max(np.real(eig)))
            if old_max * new_max < 0:
                index = int(np.argmax(np.real(eig)))
                kind = "hopf_candidate" if abs(complex(eig[index]).imag) > math.sqrt(tol) else "stability_change"
                bifurcations.append({"type": kind, "step": len(states), "x": trial[:-1].copy(),
                                     "parameter": float(trial[-1]), "estimated": True})
            eigenvalues.append(eig)
        z, tangent = trial, new_tangent
        states.append(z.copy())
        tangents.append(tangent.copy())
        residuals.append(float(np.linalg.norm(F(z))))
        if callback is not None and callback(z[:-1].copy(), float(z[-1])):
            message = "stopped by callback"
            break
        if iteration <= 3:
            step = min(max_step, step * 1.35)
        elif iteration >= 8:
            step = max(min_step, step * 0.7)
    states = np.array(states)
    eig = np.array(eigenvalues) if stability else None
    return ContinuationResult(states[:, :-1], states[:, -1], np.array(tangents),
                              np.array(residuals), converged, message, calls, bifurcations,
                              eig, np.max(np.real(eig), axis=1) < 0 if stability else None)
