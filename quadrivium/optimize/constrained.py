"""Constrained optimization.

Constraints are handled by transforming the problem (penalty, barrier,
augmented Lagrangian), by projecting onto the feasible set, or by solving the
KKT conditions directly (SQP).
"""

from __future__ import annotations

from ._history import History, monitor

from .. import numeric as np

from ..core.types import OptimizeResult
from ..core.utils import (CountedFunction, as_vector, numerical_gradient,
                          numerical_jacobian)

__all__ = [
    "penalty_method",
    "barrier_method",
    "augmented_lagrangian",
    "projected_gradient",
    "sqp",
    "active_set_qp",
    "solve_qp",
    "project_box",
    "project_simplex",
    "project_ball",
    "kkt_residual",
]


def project_box(x, lower=None, upper=None):
    """Project onto a box ``[lower, upper]``."""
    x = as_vector(x)
    if lower is not None:
        x = np.maximum(x, as_vector(lower) if np.ndim(lower) else lower)
    if upper is not None:
        x = np.minimum(x, as_vector(upper) if np.ndim(upper) else upper)
    return x


def project_simplex(v, s: float = 1.0):
    """Euclidean projection onto the simplex ``{x >= 0, sum x = s}``.

    Uses the classical sort-and-threshold algorithm, exact in ``O(n log n)``.
    """
    v = as_vector(v)
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.flatnonzero(u * np.arange(1, n + 1) > (css - s))[-1]
    theta = (css[rho] - s) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def project_ball(x, radius: float = 1.0, center=None):
    """Project onto a Euclidean ball."""
    x = as_vector(x)
    c = np.zeros_like(x) if center is None else as_vector(center)
    d = x - c
    nd = np.linalg.norm(d)
    return x if nd <= radius else c + radius * d / nd


def kkt_residual(grad_f, x, eq_jac=None, ineq_jac=None, lam=None, mu=None,
                 eq=None, ineq=None):
    """Residual of the KKT conditions, for checking a candidate solution."""
    x = as_vector(x)
    r = as_vector(grad_f(x))
    if eq_jac is not None and lam is not None:
        r = r + np.atleast_2d(eq_jac(x)).T @ as_vector(lam)
    if ineq_jac is not None and mu is not None:
        r = r + np.atleast_2d(ineq_jac(x)).T @ as_vector(mu)
    parts = {"stationarity": float(np.linalg.norm(r, np.inf))}
    if eq is not None:
        parts["primal_eq"] = float(np.linalg.norm(as_vector(eq(x)), np.inf))
    if ineq is not None:
        g = as_vector(ineq(x))
        parts["primal_ineq"] = float(np.max(np.maximum(g, 0.0))) if g.size else 0.0
        if mu is not None:
            parts["complementarity"] = float(np.max(np.abs(as_vector(mu) * g)))
    return parts


def penalty_method(f, x0, eq=None, ineq=None, mu0: float = 1.0, growth: float = 10.0,
                   outer_iter: int = 20, tol: float = 1e-8, inner=None):
    """Quadratic penalty: minimize ``f + mu/2 (||h||^2 + ||max(g,0)||^2)``.

    Simple and robust, but the subproblems become ill-conditioned as ``mu``
    grows -- which is exactly what the augmented Lagrangian fixes.
    """
    from .quasinewton import bfgs

    inner = inner or (lambda g, x: bfgs(g, x, tol=1e-10, max_iter=2000))
    x = as_vector(x0).copy()
    mu = mu0
    history = History([x])
    for k in range(1, outer_iter + 1):
        def P(z, mu=mu):
            val = float(f(z))
            if eq is not None:
                h = as_vector(eq(z))
                val += 0.5 * mu * float(h @ h)
            if ineq is not None:
                g = np.maximum(as_vector(ineq(z)), 0.0)
                val += 0.5 * mu * float(g @ g)
            return val

        res = inner(P, x)
        x = as_vector(res.x)
        history.append(x)
        viol = 0.0
        if eq is not None:
            viol = max(viol, float(np.max(np.abs(as_vector(eq(x))))))
        if ineq is not None:
            gg = as_vector(ineq(x))
            viol = max(viol, float(np.max(np.maximum(gg, 0.0))) if gg.size else 0.0)
        if viol < tol:
            return OptimizeResult(x, float(f(x)), None, None, k, True, 0, 0,
                                  "penalty", history,
                                  f"constraints satisfied to {viol:.2e}")
        mu *= growth
    return OptimizeResult(x, float(f(x)), None, None, outer_iter, False, 0, 0,
                          "penalty", history,
                          f"maximum outer iterations; violation {viol:.2e}")


def _relaxed_log_barrier(g, mu: float, eps: float = 1e-10):
    """Log barrier extended quadratically past the boundary.

    ``-mu log(-g)`` is infinite outside the feasible set, and a numerical
    gradient of an infinite value is NaN. Replacing the log by its second-order
    Taylor expansion for ``-g < eps`` keeps the function finite, smooth and
    increasing outside, so the inner minimizer is always pushed back inside.
    """
    t = -np.asarray(g, dtype=float)
    safe = np.maximum(t, eps)
    inside = -mu * np.log(safe)
    d = (t - eps) / eps
    outside = -mu * (np.log(eps) + d - 0.5 * d * d)
    return float(np.sum(np.where(t > eps, inside, outside)))


def barrier_method(f, x0, ineq, mu0: float = 1.0, shrink: float = 0.2,
                   outer_iter: int = 30, tol: float = 1e-8, inner=None):
    """Logarithmic barrier (interior point) for ``g(x) <= 0``.

    Requires a strictly feasible starting point; the barrier keeps every
    iterate inside the feasible region as ``mu`` is driven to zero.
    """
    from .quasinewton import bfgs

    inner = inner or (lambda g, x: bfgs(g, x, tol=1e-10, max_iter=1000))
    x = as_vector(x0).copy()
    g0 = as_vector(ineq(x))
    if np.any(g0 >= 0):
        raise ValueError("barrier method needs a strictly feasible starting point "
                         f"(max constraint value {np.max(g0):.3e} >= 0)")
    mu = mu0
    history = History([x])
    m = g0.size
    for k in range(1, outer_iter + 1):
        def B(z, mu=mu):
            return float(f(z)) + _relaxed_log_barrier(as_vector(ineq(z)), mu)

        res = inner(B, x)
        x_new = as_vector(res.x)
        if np.all(as_vector(ineq(x_new)) < 0):
            x = x_new
        history.append(x)
        if mu * m < tol:
            return OptimizeResult(x, float(f(x)), None, None, k, True, 0, 0,
                                  "barrier", history,
                                  f"duality gap below {mu * m:.2e}")
        mu *= shrink
    return OptimizeResult(x, float(f(x)), None, None, outer_iter, False, 0, 0,
                          "barrier", history, "maximum outer iterations reached")


def augmented_lagrangian(f, x0, eq=None, ineq=None, mu0: float = 10.0,
                         growth: float = 5.0, outer_iter: int = 50,
                         tol: float = 1e-10, inner=None):
    """Augmented Lagrangian (method of multipliers).

    Adds explicit multiplier estimates to the penalty, so the constraints are
    satisfied exactly at a finite penalty -- no ill-conditioning blow-up.
    """
    from .quasinewton import bfgs

    inner = inner or (lambda g, x: bfgs(g, x, tol=1e-11, max_iter=2000))
    x = as_vector(x0).copy()
    n_eq = as_vector(eq(x)).size if eq is not None else 0
    n_in = as_vector(ineq(x)).size if ineq is not None else 0
    lam = np.zeros(n_eq)
    mult = np.zeros(n_in)
    mu = mu0
    history = History([x])
    for k in range(1, outer_iter + 1):
        def L(z, lam=lam, mult=mult, mu=mu):
            val = float(f(z))
            if eq is not None:
                h = as_vector(eq(z))
                val += float(lam @ h) + 0.5 * mu * float(h @ h)
            if ineq is not None:
                g = as_vector(ineq(z))
                t = np.maximum(g + mult / mu, 0.0)
                val += 0.5 * mu * float(t @ t) - 0.5 * float(mult @ mult) / mu
            return val

        res = inner(L, x)
        x = as_vector(res.x)
        history.append(x)
        viol = 0.0
        if eq is not None:
            h = as_vector(eq(x))
            lam = lam + mu * h                    # first-order multiplier update
            viol = max(viol, float(np.max(np.abs(h))))
        if ineq is not None:
            g = as_vector(ineq(x))
            mult = np.maximum(mult + mu * g, 0.0)
            viol = max(viol, float(np.max(np.maximum(g, 0.0))) if g.size else 0.0)
        if viol < tol:
            r = OptimizeResult(x, float(f(x)), None, None, k, True, 0, 0,
                               "augmented_lagrangian", history,
                               f"constraints satisfied to {viol:.2e}")
            r.multipliers_eq, r.multipliers_ineq = lam, mult
            return r
        mu = min(mu * growth, 1e12)
    r = OptimizeResult(x, float(f(x)), None, None, outer_iter, False, 0, 0,
                       "augmented_lagrangian", history,
                       f"maximum outer iterations; violation {viol:.2e}")
    r.multipliers_eq, r.multipliers_ineq = lam, mult
    return r


def projected_gradient(f, x0, projection, grad_f=None, lr: float = 0.01,
                       tol: float = 1e-10, max_iter: int = 20000,
                       line_search: bool = True):
    """Projected gradient descent for a simple feasible set.

    Each step takes a gradient step and projects back, so feasibility is
    maintained exactly whenever the projection is exact.
    """
    fc = CountedFunction(lambda v: float(f(v)))
    g = (lambda x: as_vector(grad_f(x))) if grad_f is not None else \
        (lambda x: numerical_gradient(fc, x))
    x = as_vector(projection(as_vector(x0)))
    history = History([x])
    for k in range(1, max_iter + 1):
        gk = g(x)
        step = lr
        if line_search:
            f0 = fc(x)
            step = lr
            for _ in range(50):                   # backtrack on the projected step
                cand = as_vector(projection(x - step * gk))
                if fc(cand) <= f0 - 1e-4 * float(gk @ (x - cand)):
                    break
                step *= 0.5
        x_new = as_vector(projection(x - step * gk))
        crit = np.linalg.norm(x_new - x) / max(step, 1e-16)
        x = x_new
        history.append(x)
        if crit < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k, True, fc.calls, k,
                                  "projected_gradient", history, "converged")
    return OptimizeResult(x, float(fc(x)), g(x), None, max_iter, False, fc.calls,
                          max_iter, "projected_gradient", history,
                          "maximum iterations reached")


def solve_qp(G, c, A_eq=None, b_eq=None):
    """Equality-constrained QP ``min 0.5 x'Gx + c'x`` s.t. ``A x = b`` via KKT."""
    G = np.atleast_2d(np.asarray(G, dtype=float))
    c = as_vector(c)
    n = G.shape[0]
    if A_eq is None or len(np.atleast_2d(A_eq)) == 0:
        return np.linalg.solve(G, -c)
    A = np.atleast_2d(np.asarray(A_eq, dtype=float))
    b = as_vector(b_eq)
    m = A.shape[0]
    KKT = np.block([[G, A.T], [A, np.zeros((m, m))]])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(KKT, rhs)
    return sol[:n], sol[n:]


def _find_feasible(A_in, b_in, A_eq, b_eq, x0, tol=1e-12, max_iter=20000):
    """Find a point satisfying ``A_in x <= b_in`` and ``A_eq x = b_eq``.

    Successive projection onto the most violated constraint (the
    Agmon-Motzkin-Schoenberg relaxation), which converges for any consistent
    system and needs no auxiliary solver.
    """
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(max_iter):
        worst, idx, kind = tol, -1, None
        for i in range(A_eq.shape[0]):
            v = abs(A_eq[i] @ x - b_eq[i])
            if v > worst:
                worst, idx, kind = v, i, "eq"
        for i in range(A_in.shape[0]):
            v = A_in[i] @ x - b_in[i]
            if v > worst:
                worst, idx, kind = v, i, "in"
        if idx < 0:
            return x, True
        a = A_eq[idx] if kind == "eq" else A_in[idx]
        rhs = b_eq[idx] if kind == "eq" else b_in[idx]
        aa = float(a @ a)
        if aa < 1e-300:
            return x, False
        x = x - ((a @ x - rhs) / aa) * a
    return x, False


def active_set_qp(G, c, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, x0=None,
                  max_iter: int = 200, tol: float = 1e-10):
    """Primal active set method for a convex QP.

    Solves ``min 0.5 x'Gx + c'x`` subject to ``A_ineq x <= b_ineq`` and
    ``A_eq x = b_eq`` by iterating on the working set of active constraints.

    The primal method must start inside the feasible region, so an infeasible
    (or omitted) ``x0`` is first repaired by successive projection.
    """
    G = np.atleast_2d(np.asarray(G, dtype=float))
    c = as_vector(c)
    n = G.shape[0]
    A_in = np.atleast_2d(np.asarray(A_ineq, dtype=float)) if A_ineq is not None else np.zeros((0, n))
    b_in = as_vector(b_ineq) if b_ineq is not None else np.zeros(0)
    A_e = np.atleast_2d(np.asarray(A_eq, dtype=float)) if A_eq is not None else np.zeros((0, n))
    b_e = as_vector(b_eq) if b_eq is not None else np.zeros(0)
    x = as_vector(x0).copy() if x0 is not None else np.zeros(n)
    infeasible = ((A_in.shape[0] and np.max(A_in @ x - b_in) > tol)
                  or (A_e.shape[0] and np.max(np.abs(A_e @ x - b_e)) > tol))
    if infeasible:
        x, ok = _find_feasible(A_in, b_in, A_e, b_e, x)
        if not ok:
            return OptimizeResult(x, np.inf, None, None, 0, False, 0, 0,
                                  "active_set_qp", [],
                                  "could not find a feasible point; the "
                                  "constraints may be inconsistent")
    working = [i for i in range(A_in.shape[0]) if abs(A_in[i] @ x - b_in[i]) < tol]
    for k in range(1, max_iter + 1):
        rows = [A_e[i] for i in range(A_e.shape[0])] + [A_in[i] for i in working]
        A_w = np.array(rows) if rows else np.zeros((0, n))
        g = G @ x + c
        # solve the equality-constrained subproblem for the step p
        if A_w.shape[0] == 0:
            try:
                p = np.linalg.solve(G, -g)
            except np.linalg.LinAlgError:
                p = np.linalg.lstsq(G, -g, rcond=None)[0]
            lam = np.zeros(0)
        else:
            m = A_w.shape[0]
            KKT = np.block([[G, A_w.T], [A_w, np.zeros((m, m))]])
            rhs = np.concatenate([-g, np.zeros(m)])
            try:
                sol = np.linalg.solve(KKT, rhs)
            except np.linalg.LinAlgError:
                sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            p, lam = sol[:n], sol[n:]
        if np.linalg.norm(p) < tol:
            mult_in = lam[A_e.shape[0]:]
            if mult_in.size == 0 or np.all(mult_in >= -tol):
                return OptimizeResult(x, 0.5 * float(x @ G @ x) + float(c @ x), g,
                                      G, k, True, 0, 0, "active_set_qp", [],
                                      "KKT conditions satisfied")
            j = int(np.argmin(mult_in))
            working.pop(j)                      # drop a constraint with mult < 0
            continue
        # ratio test over the inactive constraints
        alpha = 1.0
        blocking = None
        for i in range(A_in.shape[0]):
            if i in working:
                continue
            denom = A_in[i] @ p
            if denom > tol:
                ratio = (b_in[i] - A_in[i] @ x) / denom
                if ratio < alpha:
                    alpha, blocking = ratio, i
        x = x + alpha * p
        if blocking is not None:
            working.append(blocking)
    return OptimizeResult(x, 0.5 * float(x @ G @ x) + float(c @ x), None, None,
                          max_iter, False, 0, 0, "active_set_qp", [],
                          "maximum iterations reached")


def sqp(f, x0, eq=None, ineq=None, grad_f=None, tol: float = 1e-10,
        max_iter: int = 200, hess_update: str = "bfgs"):
    """Sequential quadratic programming.

    Each iteration solves a QP built from a quadratic model of the Lagrangian
    and linearized constraints; the Hessian is kept positive definite by a
    damped BFGS update (Powell's modification).
    """
    fc = CountedFunction(lambda v: float(f(v)))
    g = (lambda x: as_vector(grad_f(x))) if grad_f is not None else \
        (lambda x: numerical_gradient(fc, x))
    x = as_vector(x0).copy()
    n = x.size
    B = np.eye(n)
    history = History([x])
    lam = None
    for k in range(1, max_iter + 1):
        gk = g(x)
        rows, rhs = [], []
        if eq is not None:
            h = as_vector(eq(x))
            J = np.atleast_2d(numerical_jacobian(lambda z: as_vector(eq(z)), x))
            rows.append(J)
            rhs.append(-h)
        A_eq = np.vstack(rows) if rows else None
        b_eq = np.concatenate(rhs) if rhs else None
        A_in = b_in = None
        if ineq is not None:
            gi = as_vector(ineq(x))
            Ji = np.atleast_2d(numerical_jacobian(lambda z: as_vector(ineq(z)), x))
            A_in, b_in = Ji, -gi
        res = active_set_qp(B, gk, A_in, b_in, A_eq, b_eq, x0=np.zeros(n))
        p = as_vector(res.x)
        if np.linalg.norm(p) < tol:
            return OptimizeResult(x, float(fc(x)), gk, B, k, True, fc.calls, k,
                                  "sqp", history, "converged")
        # merit-function line search keeps the step from violating constraints
        def merit(z, rho=10.0):
            val = float(fc(z))
            if eq is not None:
                val += rho * float(np.sum(np.abs(as_vector(eq(z)))))
            if ineq is not None:
                val += rho * float(np.sum(np.maximum(as_vector(ineq(z)), 0.0)))
            return val

        alpha = 1.0
        m0 = merit(x)
        for _ in range(40):
            if merit(x + alpha * p) < m0:
                break
            alpha *= 0.5
        x_new = x + alpha * p
        s = x_new - x
        y = g(x_new) - gk
        sy = float(s @ y)
        Bs = B @ s
        sBs = float(s @ Bs)
        if sy < 0.2 * sBs:                       # Powell's damping
            theta = 0.8 * sBs / (sBs - sy) if abs(sBs - sy) > 1e-300 else 1.0
            y = theta * y + (1 - theta) * Bs
            sy = float(s @ y)
        if sy > 1e-12:
            B = B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy
        x = x_new
        history.append(x)
    return OptimizeResult(x, float(fc(x)), g(x), B, max_iter, False, fc.calls,
                          max_iter, "sqp", history, "maximum iterations reached")


# Apply a common context-local output policy to public iterative entry points.
for _name in ['penalty_method', 'barrier_method', 'augmented_lagrangian', 'projected_gradient', 'sqp']:
    globals()[_name] = monitor(globals()[_name])
