"""Trust region methods and nonlinear least squares.

Instead of a direction plus a step length, trust region methods pick a radius
in which the quadratic model is trusted and minimize the model there -- which
is what makes them reliable on non-convex problems.
"""

from __future__ import annotations

from ._history import History, monitor

from .. import numeric as np

from ..core.types import OptimizeResult
from ..core.utils import (CountedFunction, as_vector, numerical_gradient,
                          numerical_hessian, numerical_jacobian)

__all__ = [
    "trust_region",
    "cauchy_point",
    "dogleg",
    "steihaug_cg",
    "levenberg_marquardt",
    "gauss_newton",
    "nonlinear_least_squares",
    "curve_fit",
]


def cauchy_point(g, B, delta: float):
    """Cauchy point: the model minimizer along the steepest descent direction.

    Cheapest possible trust region step and enough for global convergence, but
    only linearly convergent -- it ignores the Newton direction entirely.
    """
    gn = np.linalg.norm(g)
    if gn < 1e-300:
        return np.zeros_like(g)
    gBg = float(g @ B @ g)
    p_s = -(delta / gn) * g
    if gBg <= 0:
        return p_s
    tau = min(1.0, gn**3 / (delta * gBg))
    return tau * p_s


def dogleg(g, B, delta: float):
    """Powell's dogleg step: interpolate between the Cauchy and Newton points.

    The Newton point is only used when ``B`` is positive definite. With an
    indefinite Hessian ``-B^-1 g`` can point *toward* a saddle rather than away
    from it, so in that case the step falls back to the Cauchy point, which
    always descends.
    """
    try:
        np.linalg.cholesky(B)            # positive definite?
        p_b = np.linalg.solve(B, -g)
    except np.linalg.LinAlgError:
        return cauchy_point(g, B, delta)
    if np.linalg.norm(p_b) <= delta:
        return p_b
    gBg = float(g @ B @ g)
    if gBg <= 0:
        return cauchy_point(g, B, delta)
    p_u = -(float(g @ g) / gBg) * g
    if np.linalg.norm(p_u) >= delta:
        return delta * p_u / np.linalg.norm(p_u)
    d = p_b - p_u
    a = float(d @ d)
    b = 2.0 * float(p_u @ d)
    c = float(p_u @ p_u) - delta**2
    disc = max(b * b - 4 * a * c, 0.0)
    tau = (-b + np.sqrt(disc)) / (2 * a) if a > 1e-300 else 0.0
    return p_u + tau * d


def steihaug_cg(g, B, delta: float, tol: float = 1e-10, max_iter: int = None):
    """Steihaug-Toint truncated CG: solves the trust region subproblem.

    Stops on negative curvature or at the boundary, so it needs only
    matrix-vector products and works for large problems.
    """
    n = g.size
    max_iter = max_iter or 2 * n
    z = np.zeros(n)
    r = g.copy()
    d = -g
    if np.linalg.norm(r) < tol:
        return z
    for _ in range(max_iter):
        dBd = float(d @ B @ d)
        if dBd <= 0:
            # negative curvature: go to the boundary along d
            a = float(d @ d)
            b = 2.0 * float(z @ d)
            c = float(z @ z) - delta**2
            tau = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
            return z + tau * d
        alpha = float(r @ r) / dBd
        z_next = z + alpha * d
        if np.linalg.norm(z_next) >= delta:
            a = float(d @ d)
            b = 2.0 * float(z @ d)
            c = float(z @ z) - delta**2
            tau = (-b + np.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
            return z + tau * d
        r_next = r + alpha * (B @ d)
        if np.linalg.norm(r_next) < tol:
            return z_next
        beta = float(r_next @ r_next) / float(r @ r)
        d = -r_next + beta * d
        z, r = z_next, r_next
    return z


def trust_region(f, x0, grad_f=None, hess_f=None, delta0: float = 1.0,
                 delta_max: float = 100.0, eta: float = 0.15, tol: float = 1e-10,
                 max_iter: int = 1000, subproblem: str = "dogleg"):
    """Trust region minimization with a selectable subproblem solver.

    ``subproblem`` is ``'dogleg'``, ``'cauchy'`` or ``'steihaug'``. The radius
    grows when the model predicts the actual reduction well and shrinks when it
    does not.

    The Cauchy point only minimizes along the steepest descent direction, so it
    guarantees global convergence but at a linear rate; ``dogleg`` and
    ``steihaug`` capture the Newton direction too and converge superlinearly.
    """
    fc = CountedFunction(f)
    g = (lambda x: as_vector(grad_f(x))) if grad_f is not None else \
        (lambda x: numerical_gradient(fc, x))
    hess = (lambda x: np.atleast_2d(hess_f(x))) if hess_f is not None else \
        (lambda x: numerical_hessian(fc, x))
    solvers = {"dogleg": dogleg, "cauchy": cauchy_point, "steihaug": steihaug_cg}
    solve_sub = solvers[subproblem]
    x = as_vector(x0).copy()
    delta = delta0
    history = History([x])
    for k in range(1, max_iter + 1):
        gk = g(x)
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, f"trust_region_{subproblem}", history, "converged")
        B = hess(x)
        p = solve_sub(gk, B, delta)
        pred = -(float(gk @ p) + 0.5 * float(p @ B @ p))
        actual = float(fc(x)) - float(fc(x + p))
        rho = actual / pred if abs(pred) > 1e-300 else -1.0
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < 1e-10:
            delta = min(2 * delta, delta_max)
        if rho > eta:
            x = x + p
            history.append(x)
        if delta < 1e-14:
            return OptimizeResult(x, float(fc(x)), gk, B, k,
                                  np.linalg.norm(gk) < tol, fc.calls, k,
                                  f"trust_region_{subproblem}", history,
                                  "trust region radius underflowed")
    return OptimizeResult(x, float(fc(x)), g(x), None, max_iter, False, fc.calls,
                          max_iter, f"trust_region_{subproblem}", history,
                          "maximum iterations reached")


def gauss_newton(residual, x0, jac=None, tol: float = 1e-10, max_iter: int = 200):
    """Gauss-Newton for least squares: drop the second-order residual term.

    Converges quadratically for zero-residual problems and linearly otherwise.
    """
    rc = CountedFunction(lambda x: as_vector(residual(x)))
    x = as_vector(x0).copy()
    history = History([x])
    for k in range(1, max_iter + 1):
        r = rc(x)
        J = np.atleast_2d(jac(x)) if jac is not None else numerical_jacobian(rc, x)
        g = J.T @ r
        if np.linalg.norm(g) < tol:
            return OptimizeResult(x, 0.5 * float(r @ r), g, None, k - 1, True,
                                  rc.calls, k, "gauss_newton", history, "converged")
        try:
            p = np.linalg.solve(J.T @ J, -g)
        except np.linalg.LinAlgError:
            p = np.linalg.lstsq(J, -r, rcond=None)[0]
        # simple backtracking keeps the step from overshooting
        lam = 1.0
        f0 = float(r @ r)
        for _ in range(40):
            if float(rc(x + lam * p) @ rc(x + lam * p)) < f0:
                break
            lam *= 0.5
        x = x + lam * p
        history.append(x)
        if np.linalg.norm(lam * p) < tol * max(1.0, np.linalg.norm(x)):
            r = rc(x)
            return OptimizeResult(x, 0.5 * float(r @ r), J.T @ r, None, k, True,
                                  rc.calls, k, "gauss_newton", history, "converged")
    r = rc(x)
    return OptimizeResult(x, 0.5 * float(r @ r), None, None, max_iter, False,
                          rc.calls, max_iter, "gauss_newton", history,
                          "maximum iterations reached")


def levenberg_marquardt(residual, x0, jac=None, lam0: float = 1e-3,
                        tol: float = 1e-12, max_iter: int = 500,
                        lam_up: float = 10.0, lam_down: float = 0.1):
    """Levenberg-Marquardt: interpolates Gauss-Newton and gradient descent.

    The damping ``lambda`` is raised when a step fails and lowered when it
    succeeds, which gives the robustness of descent far from the solution and
    the speed of Gauss-Newton near it.
    """
    rc = CountedFunction(lambda x: as_vector(residual(x)))
    x = as_vector(x0).copy()
    lam = lam0
    r = rc(x)
    cost = float(r @ r)
    history = History([x])
    for k in range(1, max_iter + 1):
        J = np.atleast_2d(jac(x)) if jac is not None else numerical_jacobian(rc, x)
        g = J.T @ r
        if np.linalg.norm(g) < tol:
            return OptimizeResult(x, 0.5 * cost, g, None, k - 1, True, rc.calls, k,
                                  "levenberg_marquardt", history, "converged")
        JTJ = J.T @ J
        # scale the damping by the diagonal (Marquardt's refinement)
        D = np.diag(np.maximum(np.diag(JTJ), 1e-12))
        try:
            p = np.linalg.solve(JTJ + lam * D, -g)
        except np.linalg.LinAlgError:
            lam *= lam_up
            continue
        r_new = rc(x + p)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            x = x + p
            r, cost = r_new, cost_new
            lam = max(lam * lam_down, 1e-14)
            history.append(x)
            if np.linalg.norm(p) < tol * max(1.0, np.linalg.norm(x)):
                return OptimizeResult(x, 0.5 * cost, J.T @ r, None, k, True,
                                      rc.calls, k, "levenberg_marquardt", history,
                                      "converged")
        else:
            lam *= lam_up
            if lam > 1e14:
                return OptimizeResult(x, 0.5 * cost, g, None, k, False, rc.calls, k,
                                      "levenberg_marquardt", history,
                                      "damping grew without improving the residual")
    return OptimizeResult(x, 0.5 * cost, None, None, max_iter, False, rc.calls,
                          max_iter, "levenberg_marquardt", history,
                          "maximum iterations reached")


def nonlinear_least_squares(residual, x0, jac=None, method: str = "lm", **kwargs):
    """Solve ``min 0.5 ||r(x)||^2`` by Levenberg-Marquardt or Gauss-Newton."""
    if method in ("trf", "robust") or any(k in kwargs for k in ("bounds", "loss", "x_scale", "jac_sparsity")):
        from .least_squares import least_squares
        return least_squares(residual, x0, jac=jac, **kwargs)
    return {"lm": levenberg_marquardt, "gn": gauss_newton}[method](
        residual, x0, jac, **kwargs)


def curve_fit(model, xdata, ydata, p0, jac=None, sigma=None, method: str = "lm",
              absolute_sigma=False, compute_covariance=True, **kwargs):
    """Fit ``model(x, *params)`` to data by nonlinear least squares.

    Returns the optimization result with ``covariance`` and ``std_errors``
    attached, estimated from the Jacobian at the solution.
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    if sigma is not None and (np.any(np.asarray(sigma) <= 0) or not np.all(np.isfinite(sigma))):
        raise ValueError("sigma must contain finite positive standard deviations")
    w = np.ones_like(ydata) if sigma is None else 1.0 / np.asarray(sigma, dtype=float)

    def residual(p):
        return (np.asarray(model(xdata, *p), dtype=float) - ydata) * w

    # The Jacobian callback retains the historical jac(params) convention.
    # Weight its rows exactly as the residuals are weighted.
    if jac is not None:
        from ..linalg.operators import LinearOperator, aslinearoperator
        def weighted_jac(p):
            J = aslinearoperator(jac(p))
            return LinearOperator(J.shape, lambda v: w * J.matvec(v),
                                  lambda v: J.rmatvec(w * v))
        if method in ("lm", "gn") and not any(k in kwargs for k in ("bounds", "loss", "x_scale", "jac_sparsity")):
            # Existing dense LM/GN expects a concrete Jacobian.
            use_jac = lambda p: np.asarray(jac(p)) * w[:, None]
        else:
            use_jac = weighted_jac
    else:
        use_jac = None
    res = nonlinear_least_squares(residual, p0, use_jac, method, **kwargs)
    if not compute_covariance or kwargs.get("loss", "linear") != "linear":
        res.covariance = res.std_errors = None
        return res
    J = numerical_jacobian(residual, res.x)
    m, n = J.shape
    dof = max(m - n, 1)
    s2 = 1.0 if absolute_sigma else 2.0 * res.fun / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        res.covariance = cov
        res.std_errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        res.covariance = None
        res.std_errors = None
    return res


# Apply a common context-local output policy to public iterative entry points.
for _name in ['trust_region', 'gauss_newton', 'levenberg_marquardt']:
    globals()[_name] = monitor(globals()[_name])
