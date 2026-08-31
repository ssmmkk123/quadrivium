"""Proximal algorithms for composite objectives ``min f(x) + g(x)``.

``f`` is smooth and ``g`` is convex but possibly non-differentiable (an L1
penalty, an indicator of a constraint set). Proximal methods handle ``g``
through its proximal operator instead of a gradient, which is what makes
sparsity-inducing regularizers tractable.
"""

from __future__ import annotations

import numpy as np

from ..core.types import OptimizeResult
from ..core.utils import CountedFunction, as_vector, numerical_gradient

__all__ = [
    "prox_l1",
    "prox_l2",
    "prox_box",
    "prox_nonneg",
    "soft_threshold",
    "ista",
    "fista",
    "proximal_gradient",
    "admm",
    "admm_lasso",
    "douglas_rachford",
    "lasso",
    "ridge",
    "elastic_net",
]


def soft_threshold(x, t: float):
    """Soft thresholding ``sign(x) max(|x| - t, 0)``: the prox of the L1 norm."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def prox_l1(x, t: float):
    """Proximal operator of ``t ||x||_1``."""
    return soft_threshold(x, t)


def prox_l2(x, t: float):
    """Proximal operator of ``t ||x||_2`` (block soft thresholding)."""
    x = as_vector(x)
    nx = np.linalg.norm(x)
    return np.zeros_like(x) if nx <= t else (1.0 - t / nx) * x


def prox_box(x, lower=None, upper=None):
    """Proximal operator of the indicator of a box (that is, the projection)."""
    from .constrained import project_box

    return project_box(x, lower, upper)


def prox_nonneg(x, t: float = 0.0):
    """Proximal operator of the indicator of the non-negative orthant."""
    return np.maximum(as_vector(x), 0.0)


def proximal_gradient(grad_f, prox, x0, lr: float = 0.01, tol: float = 1e-10,
                      max_iter: int = 10000, f=None, g=None, accelerate: bool = False,
                      backtrack: bool = True):
    """Proximal gradient (forward-backward splitting).

    Alternates a gradient step on the smooth part with the proximal operator of
    the non-smooth part. With ``accelerate=True`` this is FISTA, which improves
    the rate from ``O(1/k)`` to ``O(1/k^2)``.
    """
    x = as_vector(x0).copy()
    y = x.copy()
    t_k = 1.0
    step = lr
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        gy = as_vector(grad_f(y))
        if backtrack and f is not None:
            fy = float(f(y))
            for _ in range(60):                  # backtracking on the step size
                x_new = as_vector(prox(y - step * gy, step))
                d = x_new - y
                if float(f(x_new)) <= fy + float(gy @ d) + float(d @ d) / (2 * step):
                    break
                step *= 0.5
        else:
            x_new = as_vector(prox(y - step * gy, step))
        if accelerate:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k))
            y = x_new + ((t_k - 1.0) / t_next) * (x_new - x)
            t_k = t_next
        else:
            y = x_new
        change = np.linalg.norm(x_new - x)
        x = x_new
        history.append(x.copy())
        if change < tol * max(1.0, np.linalg.norm(x)):
            obj = (float(f(x)) + float(g(x))) if (f and g) else np.nan
            return OptimizeResult(x, obj, None, None, k, True, 0, k,
                                  "fista" if accelerate else "ista", history,
                                  "converged")
    obj = (float(f(x)) + float(g(x))) if (f and g) else np.nan
    return OptimizeResult(x, obj, None, None, max_iter, False, 0, max_iter,
                          "fista" if accelerate else "ista", history,
                          "maximum iterations reached")


def ista(grad_f, prox, x0, lr: float = 0.01, **kwargs):
    """Iterative shrinkage-thresholding algorithm (unaccelerated)."""
    kwargs["accelerate"] = False
    return proximal_gradient(grad_f, prox, x0, lr, **kwargs)


def fista(grad_f, prox, x0, lr: float = 0.01, **kwargs):
    """FISTA: Nesterov-accelerated proximal gradient, ``O(1/k^2)``."""
    kwargs["accelerate"] = True
    return proximal_gradient(grad_f, prox, x0, lr, **kwargs)


def lasso(A, b, lam: float = 1.0, x0=None, tol: float = 1e-12,
          max_iter: int = 20000, accelerate: bool = True):
    """LASSO: ``min 0.5 ||Ax - b||^2 + lam ||x||_1``.

    Solved by proximal gradient with the Lipschitz step ``1/||A||_2^2``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = as_vector(b)
    n = A.shape[1]
    x0 = np.zeros(n) if x0 is None else as_vector(x0)
    L = float(np.linalg.norm(A, 2) ** 2)
    step = 1.0 / L if L > 0 else 1.0

    grad = lambda x: A.T @ (A @ x - b)
    prox = lambda z, t: soft_threshold(z, lam * t)
    f = lambda x: 0.5 * float(np.sum((A @ x - b) ** 2))
    g = lambda x: lam * float(np.sum(np.abs(x)))
    res = proximal_gradient(grad, prox, x0, step, tol, max_iter, f, g,
                            accelerate=accelerate, backtrack=False)
    res.method = "lasso_" + ("fista" if accelerate else "ista")
    return res


def ridge(A, b, lam: float = 1.0):
    """Ridge regression in closed form: ``(A'A + lam I)^-1 A'b``."""
    from ..linalg.lstsq import ridge_regression

    x = ridge_regression(A, b, lam)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = as_vector(b)
    obj = 0.5 * float(np.sum((A @ x - b) ** 2)) + 0.5 * lam * float(x @ x)
    return OptimizeResult(x, obj, None, None, 1, True, 0, 0, "ridge", [],
                          "closed-form solution")


def elastic_net(A, b, lam: float = 1.0, alpha: float = 0.5, tol: float = 1e-12,
                max_iter: int = 20000):
    """Elastic net: ``0.5||Ax-b||^2 + lam(alpha||x||_1 + (1-alpha)/2 ||x||^2)``.

    Blends LASSO's sparsity with ridge's stability under correlated columns.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = as_vector(b)
    n = A.shape[1]
    l2 = lam * (1 - alpha)
    L = float(np.linalg.norm(A, 2) ** 2) + l2
    step = 1.0 / L if L > 0 else 1.0
    grad = lambda x: A.T @ (A @ x - b) + l2 * x
    prox = lambda z, t: soft_threshold(z, lam * alpha * t)
    f = lambda x: 0.5 * float(np.sum((A @ x - b) ** 2)) + 0.5 * l2 * float(x @ x)
    g = lambda x: lam * alpha * float(np.sum(np.abs(x)))
    res = proximal_gradient(grad, prox, np.zeros(n), step, tol, max_iter, f, g,
                            accelerate=True, backtrack=False)
    res.method = "elastic_net"
    return res


def admm(prox_f, prox_g, x0, rho: float = 1.0, tol: float = 1e-10,
         max_iter: int = 5000, over_relax: float = 1.0):
    """Alternating direction method of multipliers for ``min f(x) + g(z)``, ``x = z``.

    Splits a hard problem into two easy proximal steps coupled by a dual
    variable; converges for any ``rho > 0`` when both parts are convex.
    """
    x = as_vector(x0).copy()
    z = x.copy()
    u = np.zeros_like(x)
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        x = as_vector(prox_f(z - u, 1.0 / rho))
        x_hat = over_relax * x + (1 - over_relax) * z
        z_old = z
        z = as_vector(prox_g(x_hat + u, 1.0 / rho))
        u = u + x_hat - z
        r_norm = np.linalg.norm(x - z)              # primal residual
        s_norm = np.linalg.norm(-rho * (z - z_old))  # dual residual
        history.append(x.copy())
        if r_norm < tol and s_norm < tol:
            return OptimizeResult(x, np.nan, None, None, k, True, 0, 0, "admm",
                                  history, "primal and dual residuals converged")
    return OptimizeResult(x, np.nan, None, None, max_iter, False, 0, 0, "admm",
                          history, "maximum iterations reached")


def admm_lasso(A, b, lam: float = 1.0, rho: float = 1.0, tol: float = 1e-10,
               max_iter: int = 5000):
    """LASSO by ADMM, with a cached factorization of ``A'A + rho I``."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = as_vector(b)
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.cholesky(AtA + rho * np.eye(n))
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)
    history = []
    for k in range(1, max_iter + 1):
        rhs = Atb + rho * (z - u)
        x = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        z_old = z
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
        history.append(z.copy())
        if (np.linalg.norm(x - z) < tol
                and np.linalg.norm(rho * (z - z_old)) < tol):
            obj = 0.5 * float(np.sum((A @ z - b) ** 2)) + lam * float(np.sum(np.abs(z)))
            return OptimizeResult(z, obj, None, None, k, True, 0, 0, "admm_lasso",
                                  history, "converged")
    obj = 0.5 * float(np.sum((A @ z - b) ** 2)) + lam * float(np.sum(np.abs(z)))
    return OptimizeResult(z, obj, None, None, max_iter, False, 0, 0, "admm_lasso",
                          history, "maximum iterations reached")


def douglas_rachford(prox_f, prox_g, x0, gamma: float = 1.0, tol: float = 1e-10,
                     max_iter: int = 5000):
    """Douglas-Rachford splitting for ``min f(x) + g(x)``.

    Handles two non-smooth terms, neither of which needs a gradient.
    """
    z = as_vector(x0).copy()
    history = []
    x = z.copy()
    for k in range(1, max_iter + 1):
        x = as_vector(prox_f(z, gamma))
        y = as_vector(prox_g(2 * x - z, gamma))
        z_new = z + (y - x)
        change = np.linalg.norm(z_new - z)
        z = z_new
        history.append(x.copy())
        if change < tol:
            return OptimizeResult(x, np.nan, None, None, k, True, 0, 0,
                                  "douglas_rachford", history, "converged")
    return OptimizeResult(x, np.nan, None, None, max_iter, False, 0, 0,
                          "douglas_rachford", history, "maximum iterations reached")
