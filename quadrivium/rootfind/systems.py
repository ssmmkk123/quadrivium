"""Root finding for systems ``F(x) = 0`` in several variables."""

from __future__ import annotations

from ._history import History, monitor

from .. import numeric as np

from ..core.types import RootResult
from ..core.utils import CountedFunction, as_vector, numerical_jacobian
from ..linalg.direct import plu_solve

__all__ = [
    "newton_system",
    "damped_newton_system",
    "broyden_good",
    "broyden_bad",
    "secant_system",
    "fixed_point_system",
    "nonlinear_gauss_seidel",
    "continuation",
    "homotopy",
    "trust_region_dogleg_root",
    "anderson_acceleration",
    "newton_krylov",
]


def _solve_step(J, r):
    """Solve ``J s = -r``, falling back to minimum-norm least squares.

    A rank-deficient Jacobian is common far from the solution, so the fallback
    keeps the iteration alive rather than aborting.
    """
    try:
        return plu_solve(J, -r)
    except Exception:
        return np.linalg.lstsq(J, -r, rcond=None)[0]


def newton_system(F, x0, jac=None, tol: float = 1e-12, max_iter: int = 100):
    """Newton's method for systems: solve ``J(x) s = -F(x)`` and step."""
    Fc = CountedFunction(F)
    x = as_vector(x0).copy()
    history = History([x])
    for k in range(1, max_iter + 1):
        r = as_vector(Fc(x))
        if np.linalg.norm(r, np.inf) < tol:
            return RootResult(x, r, k - 1, True, Fc.calls, "newton_system", history, "converged")
        J = np.atleast_2d(jac(x)) if jac is not None else numerical_jacobian(Fc, x)
        s = _solve_step(J, r)
        x = x + s
        history.append(x)
        if np.linalg.norm(s, np.inf) < tol * max(1.0, np.linalg.norm(x, np.inf)):
            return RootResult(x, as_vector(Fc(x)), k, True, Fc.calls, "newton_system",
                              history, "converged")
    return RootResult(x, as_vector(Fc(x)), max_iter, False, Fc.calls, "newton_system",
                      history, "maximum iterations reached")


def damped_newton_system(F, x0, jac=None, tol: float = 1e-12, max_iter: int = 100,
                         max_backtracks: int = 30):
    """Newton with an Armijo line search on ``||F||^2`` -- far more robust globally."""
    Fc = CountedFunction(F)
    x = as_vector(x0).copy()
    history = History([x])
    for k in range(1, max_iter + 1):
        r = as_vector(Fc(x))
        nr = np.linalg.norm(r)
        if np.linalg.norm(r, np.inf) < tol:
            return RootResult(x, r, k - 1, True, Fc.calls, "damped_newton", history, "converged")
        J = np.atleast_2d(jac(x)) if jac is not None else numerical_jacobian(Fc, x)
        s = _solve_step(J, r)
        lam = 1.0
        for _ in range(max_backtracks):
            x_new = x + lam * s
            if np.linalg.norm(as_vector(Fc(x_new))) < (1 - 1e-4 * lam) * nr:
                break
            lam *= 0.5
        else:
            x_new = x + lam * s
        x = x_new
        history.append(x)
        if np.linalg.norm(lam * s, np.inf) < tol * max(1.0, np.linalg.norm(x, np.inf)):
            return RootResult(x, as_vector(Fc(x)), k, True, Fc.calls, "damped_newton",
                              history, "converged")
    return RootResult(x, as_vector(Fc(x)), max_iter, False, Fc.calls, "damped_newton",
                      history, "maximum iterations reached")


def broyden_good(F, x0, J0=None, tol: float = 1e-12, max_iter: int = 200):
    """Broyden's "good" method: rank-one secant update of the Jacobian."""
    Fc = CountedFunction(F)
    x = as_vector(x0).copy()
    n = x.size
    J = np.atleast_2d(J0).astype(float) if J0 is not None else numerical_jacobian(Fc, x)
    r = as_vector(Fc(x))
    history = History([x])
    for k in range(1, max_iter + 1):
        if np.linalg.norm(r, np.inf) < tol:
            return RootResult(x, r, k - 1, True, Fc.calls, "broyden_good", history, "converged")
        s = _solve_step(J, r)
        x_new = x + s
        r_new = as_vector(Fc(x_new))
        y = r_new - r
        denom = s @ s
        if denom > 1e-300:
            J = J + np.outer(y - J @ s, s) / denom
        x, r = x_new, r_new
        history.append(x)
        if np.linalg.norm(s, np.inf) < tol * max(1.0, np.linalg.norm(x, np.inf)):
            return RootResult(x, r, k, True, Fc.calls, "broyden_good", history, "converged")
    return RootResult(x, r, max_iter, False, Fc.calls, "broyden_good", history,
                      "maximum iterations reached")


def broyden_bad(F, x0, B0=None, tol: float = 1e-12, max_iter: int = 200):
    """Broyden's "bad" method: updates the inverse Jacobian directly (no solves)."""
    Fc = CountedFunction(F)
    x = as_vector(x0).copy()
    if B0 is not None:
        B = np.atleast_2d(B0).astype(float)
    else:
        J = numerical_jacobian(Fc, x)
        B = np.linalg.pinv(J)
    r = as_vector(Fc(x))
    history = History([x])
    for k in range(1, max_iter + 1):
        if np.linalg.norm(r, np.inf) < tol:
            return RootResult(x, r, k - 1, True, Fc.calls, "broyden_bad", history, "converged")
        s = -B @ r
        x_new = x + s
        r_new = as_vector(Fc(x_new))
        y = r_new - r
        denom = y @ y
        if denom > 1e-300:
            B = B + np.outer(s - B @ y, y) / denom
        x, r = x_new, r_new
        history.append(x)
        if np.linalg.norm(s, np.inf) < tol * max(1.0, np.linalg.norm(x, np.inf)):
            return RootResult(x, r, k, True, Fc.calls, "broyden_bad", history, "converged")
    return RootResult(x, r, max_iter, False, Fc.calls, "broyden_bad", history,
                      "maximum iterations reached")


def secant_system(F, x0, x1=None, tol: float = 1e-12, max_iter: int = 200):
    """Wolfe-Bittner sequential secant method.

    Keeps ``n+1`` points, fits the affine model interpolating ``F`` at all of
    them, and takes its root -- the direct generalization of the scalar secant
    method, needing no derivatives.
    """
    Fc = CountedFunction(F)
    x_start = as_vector(x0)
    n = x_start.size
    h = 1e-3 * np.maximum(np.abs(x_start), 1.0)
    pts = [x_start.copy()]
    if x1 is not None:
        pts.append(as_vector(x1).copy())
    while len(pts) < n + 1:
        i = len(pts) - 1
        p = x_start.copy()
        p[i % n] += h[i % n]
        pts.append(p)
    vals = [as_vector(Fc(p)) for p in pts]
    history = History([pts[0]])
    for k in range(1, max_iter + 1):
        best = int(np.argmin([np.linalg.norm(v) for v in vals]))
        if np.linalg.norm(vals[best], np.inf) < tol:
            return RootResult(pts[best], vals[best], k - 1, True, Fc.calls,
                              "secant_system", history, "converged")
        base = pts[0]
        dX = np.column_stack([p - base for p in pts[1:]])
        dF = np.column_stack([v - vals[0] for v in vals[1:]])
        try:
            step = dX @ np.linalg.solve(dF, vals[0])
        except np.linalg.LinAlgError:
            step = dX @ np.linalg.lstsq(dF, vals[0], rcond=None)[0]
        x_new = base - step
        if not np.all(np.isfinite(x_new)):
            break
        f_new = as_vector(Fc(x_new))
        pts = pts[1:] + [x_new]          # slide the window forward
        vals = vals[1:] + [f_new]
        history.append(x_new)
        if np.linalg.norm(step, np.inf) < tol * max(1.0, np.linalg.norm(x_new, np.inf)):
            return RootResult(x_new, f_new, k, True, Fc.calls, "secant_system",
                              history, "converged")
    best = int(np.argmin([np.linalg.norm(v) for v in vals]))
    return RootResult(pts[best], vals[best], max_iter,
                      np.linalg.norm(vals[best], np.inf) < 1e-8, Fc.calls,
                      "secant_system", history, "maximum iterations reached")


def fixed_point_system(G, x0, tol: float = 1e-12, max_iter: int = 1000,
                       relaxation: float = 1.0):
    """Vector fixed point iteration ``x <- G(x)`` with optional relaxation."""
    Gc = CountedFunction(G)
    x = as_vector(x0).copy()
    history = History([x])
    for k in range(1, max_iter + 1):
        gx = as_vector(Gc(x))
        x_new = (1 - relaxation) * x + relaxation * gx
        history.append(x_new)
        if not np.all(np.isfinite(x_new)):
            return RootResult(x, np.full_like(x, np.nan), k, False, Gc.calls,
                              "fixed_point_system", history, "iteration diverged")
        if np.linalg.norm(x_new - x, np.inf) < tol * max(1.0, np.linalg.norm(x_new, np.inf)):
            return RootResult(x_new, x_new - as_vector(Gc(x_new)), k, True, Gc.calls,
                              "fixed_point_system", history, "converged")
        x = x_new
    return RootResult(x, x - as_vector(Gc(x)), max_iter, False, Gc.calls,
                      "fixed_point_system", history, "maximum iterations reached")


def nonlinear_gauss_seidel(F, x0, tol: float = 1e-10, max_iter: int = 200,
                           inner_tol: float = 1e-12):
    """Solve one equation at a time for its own unknown, sweeping repeatedly."""
    from .scalar import brent, secant

    Fc = CountedFunction(F)
    x = as_vector(x0).copy()
    n = x.size
    history = History([x])
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            def fi(xi, i=i):
                y = x.copy()
                y[i] = xi
                return as_vector(Fc(y))[i]

            res = secant(fi, x[i], x[i] + 1e-3, tol=inner_tol, max_iter=50)
            if np.isfinite(res.root):
                x[i] = res.root
        history.append(x)
        if np.linalg.norm(x - x_old, np.inf) < tol * max(1.0, np.linalg.norm(x, np.inf)):
            return RootResult(x, as_vector(Fc(x)), k, True, Fc.calls,
                              "nonlinear_gauss_seidel", history, "converged")
    return RootResult(x, as_vector(Fc(x)), max_iter, False, Fc.calls,
                      "nonlinear_gauss_seidel", history, "maximum iterations reached")


def continuation(F, x0, steps: int = 10, tol: float = 1e-12, max_iter: int = 100):
    """Natural parameter continuation from ``F(x) - (1-t) F(x0)``.

    Solves a sequence of easier problems, using each solution to start the next.
    """
    x = as_vector(x0).copy()
    F0 = as_vector(F(x))
    path = History([x])
    for i in range(1, steps + 1):
        t = i / steps

        def Ft(y, t=t):
            return as_vector(F(y)) - (1 - t) * F0

        res = damped_newton_system(Ft, x, tol=tol, max_iter=max_iter)
        x = as_vector(res.root)
        path.append(x)
    r = as_vector(F(x))
    return RootResult(x, r, steps, np.linalg.norm(r, np.inf) < 1e-6, 0,
                      "continuation", path, "continuation path completed")


def homotopy(F, x0, steps: int = 20, tol: float = 1e-12):
    """Newton homotopy ``H(x,t) = F(x) - (1-t)F(x0)`` tracked by predictor-corrector."""
    return continuation(F, x0, steps=steps, tol=tol)


def trust_region_dogleg_root(F, x0, jac=None, tol: float = 1e-12, max_iter: int = 200,
                             delta0: float = 1.0, delta_max: float = 100.0):
    """Powell's dogleg trust region applied to ``min ||F(x)||^2``."""
    Fc = CountedFunction(F)
    x = as_vector(x0).copy()
    delta = delta0
    history = History([x])
    for k in range(1, max_iter + 1):
        r = as_vector(Fc(x))
        if np.linalg.norm(r, np.inf) < tol:
            return RootResult(x, r, k - 1, True, Fc.calls, "dogleg_root", history, "converged")
        J = np.atleast_2d(jac(x)) if jac is not None else numerical_jacobian(Fc, x)
        g = J.T @ r
        Jg = J @ g
        gg = g @ g
        # Cauchy point along the steepest descent direction
        t = gg / (Jg @ Jg) if (Jg @ Jg) > 1e-300 else 0.0
        p_u = -t * g
        try:
            p_b = _solve_step(J, r)
        except Exception:
            p_b = p_u
        if np.linalg.norm(p_b) <= delta:
            p = p_b
        elif np.linalg.norm(p_u) >= delta:
            p = delta * p_u / (np.linalg.norm(p_u) + 1e-300)
        else:
            d = p_b - p_u
            a = d @ d
            b = 2 * (p_u @ d)
            c = p_u @ p_u - delta**2
            tau = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a) if a > 1e-300 else 0.0
            p = p_u + tau * d
        pred = 0.5 * (r @ r) - 0.5 * np.linalg.norm(r + J @ p) ** 2
        r_new = as_vector(Fc(x + p))
        actual = 0.5 * (r @ r) - 0.5 * (r_new @ r_new)
        rho = actual / pred if abs(pred) > 1e-300 else -1.0
        if rho > 0.25:
            x = x + p
            history.append(x)
            if rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-10:
                delta = min(2 * delta, delta_max)
        else:
            delta *= 0.5
            if delta < 1e-14:
                break
    r = as_vector(Fc(x))
    return RootResult(x, r, max_iter, np.linalg.norm(r, np.inf) < 1e-8, Fc.calls,
                      "dogleg_root", history, "maximum iterations reached")


def anderson_acceleration(g, x0, m: int = 5, beta: float = 1.0,
                          tol: float = 1e-10, max_iter: int = 500,
                          reg: float = 1e-12):
    """Anderson acceleration of the fixed-point iteration ``x = g(x)``.

    Picture Picard iteration as producing a sequence of residuals
    ``f_k = g(x_k) - x_k``.  Anderson forms the next iterate from a *linear
    combination* of the last ``m`` steps, choosing the weights to minimize the
    combined residual.  That turns a linearly convergent fixed point into
    something close to a quasi-Newton method -- without ever forming a
    Jacobian.

    ``m=0`` reduces to plain Picard; ``beta`` mixes the accelerated point with
    the raw ``g`` value (damping, useful when the iteration is fragile).  The
    least-squares system is regularized because the difference matrix becomes
    rank deficient exactly as the iteration converges.
    """
    x = as_vector(x0).astype(float)
    gc = CountedFunction(lambda v: as_vector(g(v)))
    gx = gc(x)
    f = gx - x
    X_hist, F_hist = [], []
    history = History([float(np.linalg.norm(f))])
    for k in range(1, max_iter + 1):
        if np.linalg.norm(f) < tol:
            return RootResult(x, f, k - 1, True, gc.calls, "anderson", history,
                              "converged")
        if m == 0:
            x_new = x + beta * f
        else:
            X_hist.append(x.copy())
            F_hist.append(f.copy())
            if len(X_hist) > m + 1:
                X_hist.pop(0)
                F_hist.pop(0)
            if len(F_hist) == 1:
                x_new = x + beta * f
            else:
                dF = np.column_stack([F_hist[i + 1] - F_hist[i]
                                      for i in range(len(F_hist) - 1)])
                dX = np.column_stack([X_hist[i + 1] - X_hist[i]
                                      for i in range(len(X_hist) - 1)])
                # Regularized normal equations: dF loses rank as f -> 0, and an
                # unregularized solve then amplifies round-off without bound.
                A = dF.T @ dF + reg * np.eye(dF.shape[1]) * max(
                    1.0, float(np.linalg.norm(dF)) ** 2)
                gamma = np.linalg.solve(A, dF.T @ f)
                x_new = x + beta * f - (dX + beta * dF) @ gamma
        gx = gc(x_new)
        x, f = x_new, gx - x_new
        history.append(float(np.linalg.norm(f)), x=x)
        if not np.all(np.isfinite(x)):
            return RootResult(x, f, k, False, gc.calls, "anderson", history,
                              "iteration diverged")
    return RootResult(x, f, max_iter, False, gc.calls, "anderson", history,
                      "maximum iterations reached")


def newton_krylov(F, x0, tol: float = 1e-10, max_iter: int = 100,
                  inner_tol: float = 1e-3, inner_maxiter: int = 50,
                  inner_restarts: int = 40, eps: float = None,
                  line_search: bool = True, precond=None):
    """Jacobian-free Newton-Krylov (Newton-GMRES).

    Solves the Newton equation ``J s = -F`` with GMRES, supplying the
    Jacobian only through directional derivatives approximated as
    ``J v ~ (F(x + eps v) - F(x)) / eps``.  Nothing of size ``n^2`` is ever
    formed, which is what makes it the method of choice for the large systems
    coming out of PDE discretizations.

    The inner tolerance is *inexact by design* -- solving the Newton equation
    to high precision far from the solution is wasted work, since the Newton
    direction itself is only approximate there.

    ``precond`` applies an approximate inverse Jacobian ``M^-1`` on the left.
    Without one, GMRES converges at the rate set by the Jacobian's condition
    number, which for a PDE discretization grows with the mesh -- so the
    unpreconditioned method needs steadily more Krylov steps as the grid is
    refined, even though each is cheap.
    """
    from ..linalg.iterative import gmres

    fc = CountedFunction(lambda v: as_vector(F(v)))
    x = as_vector(x0).astype(float)
    Fx = fc(x)
    history = History([float(np.linalg.norm(Fx))])
    for k in range(1, max_iter + 1):
        normF = float(np.linalg.norm(Fx))
        if normF < tol:
            return RootResult(x, Fx, k - 1, True, fc.calls, "newton_krylov",
                              history, "converged")
        # Finite-difference step sized to balance truncation against round-off.
        h = eps if eps is not None else \
            np.sqrt(np.finfo(float).eps) * (1.0 + float(np.linalg.norm(x)))

        def jv(v):
            nv = float(np.linalg.norm(v))
            if nv == 0.0:
                return np.zeros_like(v)
            return (fc(x + (h / nv) * v) - Fx) * (nv / h)

        if precond is None:
            op, rhs = jv, -Fx
        else:                       # left preconditioning: M^-1 J s = -M^-1 F
            op = lambda v: as_vector(precond(jv(v)))
            rhs = -as_vector(precond(Fx))
        s = _gmres_matfree(op, rhs, inner_tol * float(np.linalg.norm(rhs)),
                           inner_maxiter, inner_restarts)
        alpha = 1.0
        if line_search:
            for _ in range(30):        # simple Armijo backtracking on ||F||
                trial = fc(x + alpha * s)
                if float(np.linalg.norm(trial)) < (1.0 - 1e-4 * alpha) * normF:
                    break
                alpha *= 0.5
            else:
                alpha = 1.0
        x = x + alpha * s
        Fx = fc(x)
        history.append(float(np.linalg.norm(Fx)), x=x)
    return RootResult(x, Fx, max_iter, False, fc.calls, "newton_krylov",
                      history, "maximum iterations reached")


def _gmres_matfree(matvec, b, tol, maxiter, restarts: int = 40):
    """Restarted matrix-free GMRES, given only a matrix-vector product.

    The restart loop is not optional here.  A single Krylov cycle of ``m``
    steps reduces the residual by whatever the ``m``-dimensional subspace
    allows and then stops; on a discretized Laplacian, where the condition
    number grows like ``n^2``, that leaves the residual essentially untouched
    for ``m << n``.  Restarting from the current iterate keeps making progress
    at fixed memory cost.
    """
    n = b.size
    x = np.zeros(n)
    r = b.copy()
    for _ in range(max(restarts, 1)):
        beta = float(np.linalg.norm(r))
        if beta <= tol or beta == 0.0:
            return x
        m = min(maxiter, n)
        Q = np.zeros((n, m + 1))
        H = np.zeros((m + 1, m))
        Q[:, 0] = r / beta
        k_used = m
        for k in range(m):
            w = matvec(Q[:, k])
            for i in range(k + 1):         # modified Gram-Schmidt
                H[i, k] = Q[:, i] @ w
                w = w - H[i, k] * Q[:, i]
            H[k + 1, k] = float(np.linalg.norm(w))
            if H[k + 1, k] < 1e-14:
                k_used = k + 1
                break
            Q[:, k + 1] = w / H[k + 1, k]
            e1 = np.zeros(k + 2)
            e1[0] = beta
            y, *_ = np.linalg.lstsq(H[: k + 2, : k + 1], e1, rcond=None)
            if float(np.linalg.norm(H[: k + 2, : k + 1] @ y - e1)) < tol:
                k_used = k + 1
                break
        e1 = np.zeros(k_used + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[: k_used + 1, :k_used], e1, rcond=None)
        x = x + Q[:, :k_used] @ y
        r = b - matvec(x)
    return x


# Common context-local retention and callback controls.
for _name in ['newton_system', 'damped_newton_system', 'broyden_good', 'broyden_bad', 'secant_system', 'fixed_point_system', 'nonlinear_gauss_seidel', 'continuation', 'trust_region_dogleg_root', 'anderson_acceleration', 'newton_krylov', 'homotopy']:
    globals()[_name] = monitor(globals()[_name])
