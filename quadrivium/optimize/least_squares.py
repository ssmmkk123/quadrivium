"""Bounded robust least squares using matrix-free damped Gauss-Newton steps."""
from __future__ import annotations

from .. import numeric as np
from ..core.types import OptimizeResult
from ..core.exceptions import DimensionError
from ..linalg.operators import LinearOperator, aslinearoperator
from ..linalg.iterative import conjugate_gradient
from ..linalg.sparse import COOMatrix, _SparseBase

__all__ = ["least_squares"]


def _loss_values(r, loss, f_scale):
    z = (r / f_scale) ** 2
    if loss == "linear":
        rho, derivative = z, np.ones_like(z)
    elif loss == "huber":
        root = np.sqrt(np.maximum(z, 1.0))
        rho = np.where(z <= 1, z, 2 * root - 1)
        derivative = 1.0 / root
    elif loss == "soft_l1":
        root = np.sqrt(1 + z)
        # Rationalization avoids cancellation near a zero residual.
        rho, derivative = 2 * z / (root + 1), 1 / root
    elif loss == "cauchy":
        rho, derivative = np.log1p(z), 1 / (1 + z)
    else:
        raise ValueError("loss must be 'linear', 'huber', 'soft_l1', or 'cauchy'")
    return 0.5 * f_scale ** 2 * float(np.sum(rho)), derivative


def _finite_jac(fun, x, r, lower, upper, scheme, pattern=None):
    n, m = x.size, r.size
    if pattern is None:
        J = np.empty((m, n))
        support = None
        groups = [[j] for j in range(n)]
    else:
        if isinstance(pattern, _SparseBase):
            pat = pattern.tocsc()
            if pat.shape != (m, n):
                raise DimensionError("jac_sparsity has incompatible shape")
            support = [set(int(i) for i in pat.indices[int(pat.indptr[j]):int(pat.indptr[j + 1])])
                       for j in range(n)]
        else:
            pat = np.asarray(pattern)
            if pat.shape != (m, n):
                raise DimensionError("jac_sparsity has incompatible shape")
            support = [set(int(i) for i in np.flatnonzero(pat[:, j])) for j in range(n)]
        # Greedy coloring: columns sharing no residual row can be perturbed together.
        groups, occupied = [], []
        for j in range(n):
            for group, rows in zip(groups, occupied):
                if rows.isdisjoint(support[j]):
                    group.append(j)
                    rows.update(support[j])
                    break
            else:
                groups.append([j])
                occupied.append(set(support[j]))
        rr, cc, dd = [], [], []
    epsilon = np.finfo(float).eps ** (1 / 3 if scheme == "3-point" else 0.5)
    for group in groups:
        xp, xm = x.copy(), x.copy()
        steps = {}
        for j in group:
            h = epsilon * max(1.0, abs(float(x[j])))
            xp[j] = min(float(upper[j]), float(x[j]) + h)
            xm[j] = max(float(lower[j]), float(x[j]) - h) if scheme == "3-point" else x[j]
            if xp[j] == xm[j]:
                xm[j] = max(float(lower[j]), float(x[j]) - h)
            steps[j] = float(xp[j] - xm[j])
        rp = fun(xp)
        rm = fun(xm) if not np.array_equal(xm, x) else r
        for j in group:
            if support is None:
                J[:, j] = (rp - rm) / steps[j] if steps[j] else 0
            else:
                for i in sorted(support[j]):
                    rr.append(i)
                    cc.append(j)
                    dd.append(float((rp[i] - rm[i]) / steps[j]) if steps[j] else 0.0)
    return J if support is None else COOMatrix(rr, cc, dd, (m, n)).tocsr()


def least_squares(fun, x0, jac="2-point", bounds=(-np.inf, np.inf),
                  loss="linear", f_scale=1.0, x_scale=1.0, ftol=1e-10,
                  xtol=1e-10, gtol=1e-8, max_iter=500, max_nfev=None,
                  jac_sparsity=None, store_history=True, history_stride=1,
                  callback=None, tol=None):
    """Minimize robust residual cost subject to lower/upper parameter bounds.

    ``jac(x)`` may return a dense matrix, a sparse matrix or a LinearOperator
    with both forward and adjoint products. The normal equations are applied
    as products, never assembled, so sparse Jacobians remain sparse. Numerical
    Jacobians honor bounds; ``jac_sparsity`` enables grouped perturbations.
    ``x_scale='jac'`` estimates reciprocal column norms, or pass positive scales.

    Histories contain accepted parameter vectors. ``callback(x)`` receives an
    independent snapshot after an accepted step; returning True or raising
    StopIteration stops. The result exposes residuals, optimality and active_mask
    in addition to the usual OptimizeResult fields. Covariance is not computed.
    """
    x = np.array(x0, dtype=float, copy=True)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a nonempty finite vector")
    n = x.size
    if len(bounds) != 2:
        raise ValueError("bounds must be (lower, upper)")
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), x.shape)
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), x.shape)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise ValueError("bounds must be ordered and contain no NaNs")
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError("x0 is outside bounds")
    if tol is not None:
        ftol = xtol = gtol = tol
    if any(not np.isfinite(v) or v <= 0 for v in (f_scale, ftol, xtol, gtol)):
        raise ValueError("f_scale and tolerances must be finite and positive")
    if not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    if not isinstance(history_stride, int) or history_stride < 1:
        raise ValueError("history_stride must be a positive integer")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    max_nfev = max_nfev if max_nfev is not None else max(1, (max_iter + 1) * (2 * n + 2))
    if not isinstance(max_nfev, int) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    if jac is None:
        jac = "2-point"
    if not callable(jac) and jac not in ("2-point", "3-point"):
        raise ValueError("jac must be callable, '2-point', or '3-point'")
    by_jac = isinstance(x_scale, str) and x_scale == "jac"
    if isinstance(x_scale, str) and not by_jac:
        raise ValueError("x_scale must be positive scales or 'jac'")
    scale = np.ones(n) if by_jac else np.broadcast_to(np.asarray(x_scale, dtype=float), x.shape).copy()
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError("x_scale values must be finite and positive")
    calls, njev, accepted = 0, 0, 0
    m = None

    class EvaluationLimit(Exception):
        pass

    def residual_at(point):
        nonlocal calls, m
        if calls >= max_nfev:
            raise EvaluationLimit
        value = np.asarray(fun(point), dtype=float)
        calls += 1
        if value.ndim != 1 or not value.size or (m is not None and value.size != m):
            raise DimensionError("residual must be a nonempty vector of constant size")
        m = value.size
        return value

    r = residual_at(x)
    if not np.all(np.isfinite(r)):
        raise ValueError("residuals are nonfinite at x0")
    cost, weights = _loss_values(r, loss, f_scale)
    history = [x.copy()] if store_history else []
    converged, message, iterations = False, "maximum iterations reached", 0
    g, J = np.zeros(n), None
    lam = 1e-3
    for iteration in range(max_iter + 1):
        try:
            raw = jac(x) if callable(jac) else _finite_jac(residual_at, x, r, lower, upper, jac, jac_sparsity)
            J = aslinearoperator(raw)
            if J.shape != (r.size, n):
                raise DimensionError("Jacobian has incompatible shape")
            njev += 1
            g = J.rmatvec(weights * r)
            if not np.all(np.isfinite(g)):
                raise ValueError("Jacobian produced a nonfinite gradient")
            active = ((x <= lower) & (g > 0)) | ((x >= upper) & (g < 0)) | (lower == upper)
            if by_jac:
                if isinstance(raw, _SparseBase):
                    canonical = raw.sum_duplicates()
                    norms = np.sqrt(np.bincount(canonical.indices,
                                                weights=canonical.data ** 2, minlength=n))
                    scale = 1 / np.maximum(norms, 1e-12)
                elif not hasattr(raw, "matvec"):
                    scale = 1 / np.maximum(np.linalg.norm(np.asarray(raw), axis=0), 1e-12)
                else:
                    e = np.zeros(n)
                    for j in range(n):
                        e[j] = 1
                        col = J.matvec(e)
                        scale[j] = 1 / max(float(np.linalg.norm(col)), 1e-12)
                        e[j] = 0
            d = np.where(active, 0.0, scale)
            optimality = float(np.max(np.abs(d * g)))
            if optimality <= gtol:
                converged, message = True, "projected gradient tolerance reached"
                break
            if iteration == max_iter:
                break
            rhs = -d * g
            # A diagonal shift makes the free-variable operator positive definite.
            normal = LinearOperator((n, n), lambda v: d * J.rmatvec(weights * J.matvec(d * v)) + lam * v)
            inner = conjugate_gradient(normal, rhs, tol=min(1e-5, max(1e-12, 0.01 * optimality)),
                                       max_iter=max(20, min(4 * n, 1000)), store_history=False)
            candidate = np.clip(x + d * inner.x, lower, upper)
            p = candidate - x
            jp = J.matvec(p)
            predicted = -float(g @ p) - 0.5 * float(jp @ (weights * jp))
            rnew = residual_at(candidate)
            newcost = _loss_values(rnew, loss, f_scale)[0] if np.all(np.isfinite(rnew)) else np.inf
            iterations = iteration + 1
            if newcost < cost:
                oldcost = cost
                ratio = (cost - newcost) / predicted if predicted > 0 else 0.0
                x, r, cost = candidate, rnew, newcost
                _, weights = _loss_values(r, loss, f_scale)
                lam = max(1e-14, lam * (0.3 if ratio > 0.75 else 1.0))
                accepted += 1
                if store_history and accepted % history_stride == 0:
                    history.append(x.copy())
                if callback is not None:
                    try:
                        stop = callback(x.copy())
                    except StopIteration:
                        stop = True
                    if stop:
                        message = "stopped by callback"
                        break
                if float(np.linalg.norm(p / scale)) <= xtol * (xtol + float(np.linalg.norm(x / scale))):
                    converged, message = True, "step tolerance reached"
                    break
                if oldcost - cost <= ftol * max(oldcost, 1e-300) and ratio > 0.25:
                    converged, message = True, "cost tolerance reached"
                    break
            else:
                lam *= 10
                if lam > 1e20:
                    message = "damping grew without improving the cost"
                    break
        except EvaluationLimit:
            message = "maximum residual evaluations reached"
            break
    # Report a gradient at the returned point when the budget permits it.
    if J is not None:
        try:
            raw = jac(x) if callable(jac) else _finite_jac(residual_at, x, r, lower, upper, jac, jac_sparsity)
            J = aslinearoperator(raw)
            g = J.rmatvec(weights * r)
            njev += 1
        except EvaluationLimit:
            pass
    active_mask = np.where(x <= lower, -1, np.where(x >= upper, 1, 0))
    projected = np.where(((active_mask == -1) & (g > 0)) | ((active_mask == 1) & (g < 0)) | (lower == upper), 0.0, g)
    result = OptimizeResult(x, cost, g, None, iterations, converged, calls, njev,
                            "robust_least_squares", history, message)
    result.residuals, result.optimality = r, float(np.max(np.abs(projected * scale)))
    result.active_mask, result.jacobian = active_mask, J
    return result
