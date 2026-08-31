"""Newton and quasi-Newton methods for unconstrained minimization.

Quasi-Newton methods build curvature information from successive gradients, so
they get near-Newton convergence without ever forming a Hessian.
"""

from __future__ import annotations

import numpy as np

from ..core.types import OptimizeResult
from ..core.utils import (CountedFunction, as_vector, numerical_gradient,
                          numerical_hessian)
from .linesearch import strong_wolfe

__all__ = [
    "newton_method",
    "modified_newton",
    "bfgs",
    "dfp",
    "sr1",
    "broyden_class",
    "lbfgs",
    "quasi_newton",
    "newton_cg",
    "lbfgsb",
]


def _prep(f, grad_f):
    fc = CountedFunction(f)
    g = (lambda x: as_vector(grad_f(x))) if grad_f is not None else \
        (lambda x: numerical_gradient(fc, x))
    return fc, g


def _stalled(alpha, x_new, x):
    return alpha <= 1e-16 or np.all(x_new == x)


def newton_method(f, x0, grad_f=None, hess_f=None, tol: float = 1e-10,
                  max_iter: int = 200, line_search: bool = True):
    """Newton's method: solve ``H p = -g`` each step. Quadratic convergence."""
    fc, g = _prep(f, grad_f)
    x = as_vector(x0).copy()
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        gk = g(x)
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, "newton", history, "converged")
        H = np.atleast_2d(hess_f(x)) if hess_f is not None else numerical_hessian(fc, x)
        try:
            p = np.linalg.solve(H, -gk)
        except np.linalg.LinAlgError:
            p = -gk
        if gk @ p > 0:
            p = -gk           # fall back to descent if the Hessian is indefinite
        alpha = strong_wolfe(fc, g, x, p) if line_search else 1.0
        x_new = x + alpha * p
        if _stalled(alpha, x_new, x):
            return OptimizeResult(x, float(fc(x)), gk, H, k, np.linalg.norm(gk) < tol,
                                  fc.calls, k, "newton", history,
                                  "line search stalled at floating-point precision")
        x = x_new
        history.append(x.copy())
    return OptimizeResult(x, float(fc(x)), g(x), None, max_iter, False, fc.calls,
                          max_iter, "newton", history, "maximum iterations reached")


def modified_newton(f, x0, grad_f=None, hess_f=None, tol: float = 1e-10,
                    max_iter: int = 200, beta: float = 1e-3):
    """Newton with a Hessian modification that forces positive definiteness.

    Adds increasing multiples of the identity until Cholesky succeeds, which
    guarantees a descent direction even in non-convex regions.
    """
    fc, g = _prep(f, grad_f)
    x = as_vector(x0).copy()
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        gk = g(x)
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, "modified_newton", history, "converged")
        H = np.atleast_2d(hess_f(x)) if hess_f is not None else numerical_hessian(fc, x)
        n = H.shape[0]
        tau = 0.0 if np.min(np.diag(H)) > 0 else -np.min(np.diag(H)) + beta
        for _ in range(60):
            try:
                L = np.linalg.cholesky(H + tau * np.eye(n))
                break
            except np.linalg.LinAlgError:
                tau = max(2 * tau, beta)
        else:
            L = np.eye(n)
        p = np.linalg.solve(L.T, np.linalg.solve(L, -gk))
        alpha = strong_wolfe(fc, g, x, p)
        x_new = x + alpha * p
        if _stalled(alpha, x_new, x):
            return OptimizeResult(x, float(fc(x)), gk, H, k, np.linalg.norm(gk) < tol,
                                  fc.calls, k, "modified_newton", history,
                                  "line search stalled at floating-point precision")
        x = x_new
        history.append(x.copy())
    return OptimizeResult(x, float(fc(x)), g(x), None, max_iter, False, fc.calls,
                          max_iter, "modified_newton", history,
                          "maximum iterations reached")


def _quasi_newton_driver(f, x0, grad_f, update, name, tol, max_iter, H0=None,
                         c2: float = 0.9):
    """Shared loop for the inverse-Hessian quasi-Newton updates."""
    fc, g = _prep(f, grad_f)
    x = as_vector(x0).copy()
    n = x.size
    H = np.eye(n) if H0 is None else np.array(H0, dtype=float)
    gk = g(x)
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, H, k - 1, True, fc.calls, k,
                                  name, history, "converged")
        p = -H @ gk
        if gk @ p >= 0:
            H = np.eye(n)      # reset if the approximation lost definiteness
            p = -gk
        alpha = strong_wolfe(fc, g, x, p, alpha0=1.0, c2=c2)
        x_new = x + alpha * p
        if _stalled(alpha, x_new, x):
            return OptimizeResult(x, float(fc(x)), gk, H, k, np.linalg.norm(gk) < tol,
                                  fc.calls, k, name, history,
                                  "line search stalled: further progress is limited "
                                  "by the floating-point precision of f")
        g_new = g(x_new)
        s = x_new - x
        y = g_new - gk
        H = update(H, s, y)
        x, gk = x_new, g_new
        history.append(x.copy())
    return OptimizeResult(x, float(fc(x)), gk, H, max_iter, False, fc.calls,
                          max_iter, name, history, "maximum iterations reached")


def bfgs(f, x0, grad_f=None, tol: float = 1e-10, max_iter: int = 1000, H0=None):
    """BFGS: the standard quasi-Newton method.

    Updates the inverse Hessian directly and keeps it positive definite whenever
    the curvature condition ``s'y > 0`` holds -- which the Wolfe line search
    guarantees.
    """
    def update(H, s, y):
        sy = float(s @ y)
        if sy <= 1e-12:
            return H                     # skip the update rather than corrupt it
        rho = 1.0 / sy
        n = H.shape[0]
        I = np.eye(n)
        V = I - rho * np.outer(s, y)
        return V @ H @ V.T + rho * np.outer(s, s)

    return _quasi_newton_driver(f, x0, grad_f, update, "bfgs", tol, max_iter, H0)


def dfp(f, x0, grad_f=None, tol: float = 1e-10, max_iter: int = 1000, H0=None,
        c2: float = 0.1):
    """Davidon-Fletcher-Powell: the original quasi-Newton update.

    DFP is markedly more sensitive to line search accuracy than BFGS -- with a
    loose curvature condition it can stall on Rosenbrock -- so the default
    ``c2`` here is tighter than the BFGS default.
    """
    def update(H, s, y):
        sy = float(s @ y)
        if sy <= 1e-12:
            return H
        Hy = H @ y
        return H + np.outer(s, s) / sy - np.outer(Hy, Hy) / float(y @ Hy)

    return _quasi_newton_driver(f, x0, grad_f, update, "dfp", tol, max_iter, H0, c2)


def sr1(f, x0, grad_f=None, tol: float = 1e-10, max_iter: int = 1000, H0=None,
        r: float = 1e-8, c2: float = 0.1):
    """Symmetric rank-one update.

    Not guaranteed positive definite, but often a better Hessian approximation
    than BFGS -- useful inside trust region methods.
    """
    def update(H, s, y):
        v = s - H @ y
        denom = float(v @ y)
        if abs(denom) < r * np.linalg.norm(v) * np.linalg.norm(y):
            return H                    # skip near-singular updates
        return H + np.outer(v, v) / denom

    return _quasi_newton_driver(f, x0, grad_f, update, "sr1", tol, max_iter, H0, c2)


def broyden_class(f, x0, grad_f=None, phi: float = 0.5, tol: float = 1e-10,
                  max_iter: int = 1000, H0=None, c2: float = 0.1):
    """Broyden family interpolating DFP (``phi=1``) and BFGS (``phi=0``)."""
    def update(H, s, y):
        sy = float(s @ y)
        if sy <= 1e-12:
            return H
        Hy = H @ y
        yHy = float(y @ Hy)
        dfp_term = H + np.outer(s, s) / sy - np.outer(Hy, Hy) / yHy
        rho = 1.0 / sy
        I = np.eye(H.shape[0])
        V = I - rho * np.outer(s, y)
        bfgs_term = V @ H @ V.T + rho * np.outer(s, s)
        return phi * dfp_term + (1 - phi) * bfgs_term

    return _quasi_newton_driver(f, x0, grad_f, update, f"broyden_phi{phi}", tol,
                                max_iter, H0, c2)


def lbfgs(f, x0, grad_f=None, m: int = 10, tol: float = 1e-10, max_iter: int = 1000):
    """Limited-memory BFGS.

    Stores only the last ``m`` correction pairs and applies the inverse Hessian
    by the two-loop recursion, so memory is ``O(mn)`` instead of ``O(n^2)`` --
    the reason L-BFGS is the default for large problems.
    """
    fc, g = _prep(f, grad_f)
    x = as_vector(x0).copy()
    gk = g(x)
    S, Y, rho = [], [], []
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, "lbfgs", history, "converged")
        # two-loop recursion for p = -H_k g_k
        q = gk.copy()
        alphas = []
        for i in range(len(S) - 1, -1, -1):
            a = rho[i] * float(S[i] @ q)
            alphas.append(a)
            q = q - a * Y[i]
        if S:
            gamma = float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
            q = gamma * q
        for i, a in zip(range(len(S)), reversed(alphas)):
            b = rho[i] * float(Y[i] @ q)
            q = q + (a - b) * S[i]
        p = -q
        if gk @ p >= 0:
            p = -gk
        alpha = strong_wolfe(fc, g, x, p, alpha0=1.0, c2=0.9)
        x_new = x + alpha * p
        if _stalled(alpha, x_new, x):
            return OptimizeResult(x, float(fc(x)), gk, None, k,
                                  np.linalg.norm(gk) < tol, fc.calls, k, "lbfgs",
                                  history, "line search stalled at floating-point "
                                  "precision")
        g_new = g(x_new)
        s = x_new - x
        y = g_new - gk
        sy = float(s @ y)
        if sy > 1e-12:
            S.append(s)
            Y.append(y)
            rho.append(1.0 / sy)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
                rho.pop(0)
        x, gk = x_new, g_new
        history.append(x.copy())
    return OptimizeResult(x, float(fc(x)), gk, None, max_iter, False, fc.calls,
                          max_iter, "lbfgs", history, "maximum iterations reached")


def quasi_newton(f, x0, grad_f=None, method: str = "bfgs", **kwargs):
    """Dispatch to a quasi-Newton method by name."""
    return {"bfgs": bfgs, "dfp": dfp, "sr1": sr1, "lbfgs": lbfgs,
            "broyden": broyden_class}[method](f, x0, grad_f, **kwargs)


def newton_cg(f, x0, grad_f=None, hess_vec=None, tol: float = 1e-8,
              max_iter: int = 500, cg_max: int = None, forcing: str = "superlinear"):
    """Truncated (Hessian-free) Newton with conjugate-gradient inner solves.

    Solves the Newton system ``H p = -g`` approximately with CG, terminating
    early on two conditions.  The first is the *forcing sequence*: an accurate
    Newton direction is worthless far from the solution, so the inner
    tolerance is tied to the current gradient norm and tightens automatically
    as convergence sets in.  The second is negative curvature: if CG meets a
    direction with ``d'Hd <= 0`` the quadratic model is unbounded there, and
    the iterate is truncated to the current direction -- which is still a
    descent direction, so the method keeps working on non-convex problems
    where plain Newton would step toward a saddle.

    Only Hessian-*vector* products are needed; supply ``hess_vec(x, v)`` for
    large problems, or let it be approximated by a directional difference of
    gradients.
    """
    fc = CountedFunction(f)
    x = as_vector(x0).astype(float)
    n = x.size
    gf = grad_f if grad_f is not None else (lambda v: numerical_gradient(fc, v))
    cg_max = n if cg_max is None else cg_max

    def hv(xx, v, g):
        if hess_vec is not None:
            return as_vector(hess_vec(xx, v))
        nv = float(np.linalg.norm(v))
        if nv == 0.0:
            return np.zeros_like(v)
        h = np.sqrt(np.finfo(float).eps) * (1.0 + float(np.linalg.norm(xx))) / nv
        return (as_vector(gf(xx + h * v)) - g) / h

    history = []
    for k in range(1, max_iter + 1):
        g = as_vector(gf(x))
        gnorm = float(np.linalg.norm(g))
        history.append(float(fc(x)))
        if gnorm < tol:
            return OptimizeResult(x, float(fc(x)), g, None, k - 1, True,
                                  fc.calls, k, "newton_cg", history, "converged")
        # Forcing sequence: linear -> min(0.5, sqrt(g)); superlinear -> min(0.5, g).
        eta = min(0.5, gnorm) if forcing == "superlinear" else min(0.5, np.sqrt(gnorm))
        p = np.zeros(n)
        r = g.copy()
        d = -r
        took_step = False
        for _ in range(cg_max):
            Hd = hv(x, d, g)
            curv = float(d @ Hd)
            if curv <= 1e-14 * float(d @ d):
                # Negative curvature: stop here rather than follow the model
                # into a direction where it decreases without bound.  On the
                # very first inner step there is no accumulated direction yet,
                # and d = -g is then the sensible fallback.
                if not took_step:
                    p = d
                break
            alpha = float(r @ r) / curv
            p = p + alpha * d
            took_step = True
            r_new = r + alpha * Hd
            if float(np.linalg.norm(r_new)) < eta * gnorm:
                r = r_new
                break
            beta = float(r_new @ r_new) / float(r @ r)
            d = -r_new + beta * d
            r = r_new
        if float(g @ p) >= 0.0:          # safeguard: fall back to steepest descent
            p = -g
        res = strong_wolfe(fc, gf, x, p)
        alpha = res if np.isscalar(res) else res[0]
        x = x + alpha * p
    return OptimizeResult(x, float(fc(x)), as_vector(gf(x)), None, max_iter, False,
                          fc.calls, max_iter, "newton_cg", history,
                          "maximum iterations reached")


def lbfgsb(f, x0, grad_f=None, bounds=None, m: int = 10, tol: float = 1e-8,
           max_iter: int = 1000):
    """L-BFGS with simple bound constraints, by the projected-gradient approach.

    At each step the variables at their bounds with a gradient pushing further
    out are *frozen* (the active set); the L-BFGS direction is computed on the
    remaining free variables and the result projected back into the box.
    Freezing them matters: a curvature pair built from a step that the
    projection then truncated corrupts the Hessian approximation, so the update
    is skipped when the step was clipped.

    ``bounds`` is a sequence of ``(lo, hi)`` pairs; use ``None`` for an
    unbounded side.  With ``bounds=None`` this is plain L-BFGS.
    """
    fc = CountedFunction(f)
    x = as_vector(x0).astype(float)
    n = x.size
    gf = grad_f if grad_f is not None else (lambda v: numerical_gradient(fc, v))
    if bounds is None:
        lo = np.full(n, -np.inf)
        hi = np.full(n, np.inf)
    else:
        lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], float)
        hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], float)
    x = np.clip(x, lo, hi)
    S, Y, rho = [], [], []
    history = []
    g = as_vector(gf(x))
    for k in range(1, max_iter + 1):
        history.append(float(fc(x)))
        # Projected gradient: the true optimality measure under bounds.
        pg = np.where(((x <= lo) & (g > 0)) | ((x >= hi) & (g < 0)), 0.0, g)
        if float(np.linalg.norm(pg)) < tol:
            return OptimizeResult(x, float(fc(x)), g, None, k - 1, True, fc.calls,
                                  k, "lbfgsb", history, "converged")
        free = ~(((x <= lo) & (g > 0)) | ((x >= hi) & (g < 0)))
        q = pg.copy()
        alphas = []
        for si, yi, ri in zip(reversed(S), reversed(Y), reversed(rho)):
            a = ri * float(si @ q)
            alphas.append(a)
            q = q - a * yi
        if S:
            gamma = float(S[-1] @ Y[-1]) / float(Y[-1] @ Y[-1])
            q = gamma * q
        for si, yi, ri, a in zip(S, Y, rho, reversed(alphas)):
            b = ri * float(yi @ q)
            q = q + si * (a - b)
        d = -q
        d = np.where(free, d, 0.0)
        if float(pg @ d) >= 0.0:
            d = -pg
        alpha = 1.0
        f0 = float(fc(x))
        slope = float(pg @ d)
        for _ in range(60):              # projected backtracking
            x_try = np.clip(x + alpha * d, lo, hi)
            if float(fc(x_try)) <= f0 + 1e-4 * float(pg @ (x_try - x)):
                break
            alpha *= 0.5
        else:
            x_try = np.clip(x + 1e-8 * d, lo, hi)
        s = x_try - x
        g_new = as_vector(gf(x_try))
        y = g_new - g
        sy = float(s @ y)
        clipped = not np.allclose(x_try, x + alpha * d)
        if sy > 1e-12 * float(np.linalg.norm(s)) * float(np.linalg.norm(y)) \
                and not clipped:
            S.append(s)
            Y.append(y)
            rho.append(1.0 / sy)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
                rho.pop(0)
        if float(np.linalg.norm(x_try - x)) < 1e-16 * max(1.0, float(np.linalg.norm(x))):
            return OptimizeResult(x, f0, g, None, k, True, fc.calls, k, "lbfgsb",
                                  history, "step size underflow; likely converged")
        x, g = x_try, g_new
    return OptimizeResult(x, float(fc(x)), g, None, max_iter, False, fc.calls,
                          max_iter, "lbfgsb", history, "maximum iterations reached")
