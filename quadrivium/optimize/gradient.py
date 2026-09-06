"""Gradient-based unconstrained minimization.

From plain steepest descent through the nonlinear conjugate gradient methods
and the stochastic-gradient-style adaptive rules.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.types import OptimizeResult
from ..core.utils import CountedFunction, as_vector, numerical_gradient
from .linesearch import backtracking, strong_wolfe

__all__ = [
    "gradient_descent",
    "steepest_descent_opt",
    "momentum",
    "nesterov",
    "adagrad",
    "rmsprop",
    "adam",
    "conjugate_gradient_fr",
    "conjugate_gradient_pr",
    "conjugate_gradient_hs",
    "nonlinear_cg",
    "barzilai_borwein",
]


def _grad(grad_f, f):
    if grad_f is not None:
        return lambda x: as_vector(grad_f(x))
    return lambda x: numerical_gradient(f, x)


def _diverged(x) -> bool:
    """True once an iterate stops being finite.

    A fixed-step method with too large a step does not fail gracefully: it
    overflows, and every later iterate is NaN. Without this check the loop
    still runs to ``max_iter``, appending a NaN to ``history`` each time, and
    reports "maximum iterations reached" -- which reads like a tolerance that
    was nearly met rather than a step size that has to be reduced.
    """
    return not np.all(np.isfinite(x))


def gradient_descent(f, x0, grad_f=None, lr: float = 0.01, tol: float = 1e-8,
                     max_iter: int = 10000, line_search: bool = False):
    """Steepest descent, with a fixed step or an Armijo line search."""
    fc = CountedFunction(f)
    g = _grad(grad_f, fc)
    x = as_vector(x0).copy()
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        gk = g(x)
        gn = np.linalg.norm(gk)
        if gn < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, "gradient_descent", history, "converged")
        step = backtracking(fc, x, -gk, gk, alpha0=1.0) if line_search else lr
        x = x - step * gk
        history.append(x.copy())
        if _diverged(x):
            return OptimizeResult(x, float(fc(x)), gk, None, k, False, fc.calls,
                                  k, "gradient_descent", history,
                                  f"diverged: the iterate left the finite range at "
                                  f"step {k}; reduce lr (currently {lr:g}) or pass "
                                  f"line_search=True")
    return OptimizeResult(x, float(fc(x)), g(x), None, max_iter, False, fc.calls,
                          max_iter, "gradient_descent", history,
                          "maximum iterations reached")


def steepest_descent_opt(f, x0, grad_f=None, **kwargs):
    """Steepest descent with an exact line search at every step."""
    kwargs.setdefault("line_search", True)
    res = gradient_descent(f, x0, grad_f, **kwargs)
    res.method = "steepest_descent"
    return res


def _adaptive(f, x0, grad_f, update, name, lr, tol, max_iter):
    fc = CountedFunction(f)
    g = _grad(grad_f, fc)
    x = as_vector(x0).copy()
    state = {}
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        gk = g(x)
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, name, history, "converged")
        x = update(x, gk, state, lr, k)
        history.append(x.copy())
        if _diverged(x):
            return OptimizeResult(x, float(fc(x)), gk, None, k, False, fc.calls,
                                  k, name, history,
                                  f"diverged: the iterate left the finite range at "
                                  f"step {k}; reduce lr (currently {lr:g})")
    return OptimizeResult(x, float(fc(x)), g(x), None, max_iter, False, fc.calls,
                          max_iter, name, history, "maximum iterations reached")


def momentum(f, x0, grad_f=None, lr: float = 0.01, beta: float = 0.9,
             tol: float = 1e-8, max_iter: int = 10000):
    """Heavy-ball momentum: damps oscillation across narrow valleys."""
    def update(x, gk, st, lr, k):
        st["v"] = beta * st.get("v", np.zeros_like(x)) + gk
        return x - lr * st["v"]

    return _adaptive(f, x0, grad_f, update, "momentum", lr, tol, max_iter)


def nesterov(f, x0, grad_f=None, lr: float = 0.01, beta: float = 0.9,
             tol: float = 1e-8, max_iter: int = 10000):
    """Nesterov accelerated gradient: evaluates the gradient at the look-ahead point."""
    fc = CountedFunction(f)
    g = _grad(grad_f, fc)
    x = as_vector(x0).copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        gk = g(x)
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, "nesterov", history, "converged")
        g_look = g(x - lr * beta * v)
        v = beta * v + g_look
        x = x - lr * v
        history.append(x.copy())
    return OptimizeResult(x, float(fc(x)), g(x), None, max_iter, False, fc.calls,
                          max_iter, "nesterov", history, "maximum iterations reached")


def adagrad(f, x0, grad_f=None, lr: float = 0.1, eps: float = 1e-8,
            tol: float = 1e-8, max_iter: int = 10000):
    """AdaGrad: per-coordinate steps scaled by the accumulated gradient norm."""
    def update(x, gk, st, lr, k):
        st["s"] = st.get("s", np.zeros_like(x)) + gk * gk
        return x - lr * gk / (np.sqrt(st["s"]) + eps)

    return _adaptive(f, x0, grad_f, update, "adagrad", lr, tol, max_iter)


def rmsprop(f, x0, grad_f=None, lr: float = 0.01, beta: float = 0.9,
            eps: float = 1e-8, tol: float = 1e-8, max_iter: int = 10000):
    """RMSProp: an exponentially decaying window of squared gradients."""
    def update(x, gk, st, lr, k):
        st["s"] = beta * st.get("s", np.zeros_like(x)) + (1 - beta) * gk * gk
        return x - lr * gk / (np.sqrt(st["s"]) + eps)

    return _adaptive(f, x0, grad_f, update, "rmsprop", lr, tol, max_iter)


def adam(f, x0, grad_f=None, lr: float = 0.05, beta1: float = 0.9,
         beta2: float = 0.999, eps: float = 1e-8, tol: float = 1e-8,
         max_iter: int = 10000):
    """Adam: momentum plus RMSProp scaling, with bias correction."""
    def update(x, gk, st, lr, k):
        st["m"] = beta1 * st.get("m", np.zeros_like(x)) + (1 - beta1) * gk
        st["v"] = beta2 * st.get("v", np.zeros_like(x)) + (1 - beta2) * gk * gk
        m_hat = st["m"] / (1 - beta1**k)
        v_hat = st["v"] / (1 - beta2**k)
        return x - lr * m_hat / (np.sqrt(v_hat) + eps)

    return _adaptive(f, x0, grad_f, update, "adam", lr, tol, max_iter)


def nonlinear_cg(f, x0, grad_f=None, variant: str = "pr", tol: float = 1e-8,
                 max_iter: int = 2000, restart: int = None):
    """Nonlinear conjugate gradient with a strong Wolfe line search.

    ``variant`` selects the beta formula: ``'fr'`` (Fletcher-Reeves),
    ``'pr'`` (Polak-Ribiere, restarted at negative beta), or ``'hs'``
    (Hestenes-Stiefel). Automatic restarts keep the directions descent.
    """
    fc = CountedFunction(f)
    g = _grad(grad_f, fc)
    x = as_vector(x0).copy()
    n = x.size
    restart = restart or n
    gk = g(x)
    d = -gk
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, f"cg_{variant}", history, "converged")
        alpha = strong_wolfe(fc, g, x, d, alpha0=1.0, c2=0.1)
        x_new = x + alpha * d
        # Near the solution the Armijo test compares quantities below the
        # floating-point resolution of f, and the line search collapses to a
        # zero step. Report that honestly instead of spinning to max_iter.
        if alpha <= 1e-16 or np.all(x_new == x):
            return OptimizeResult(
                x, float(fc(x)), gk, None, k, np.linalg.norm(gk) < tol, fc.calls,
                k, f"cg_{variant}", history,
                "line search stalled: further progress is limited by the "
                f"floating-point precision of f (|grad| = {np.linalg.norm(gk):.3e})")
        g_new = g(x_new)
        if variant == "fr":
            beta = (g_new @ g_new) / (gk @ gk)
        elif variant == "pr":
            beta = max(0.0, (g_new @ (g_new - gk)) / (gk @ gk))
        elif variant == "hs":
            denom = d @ (g_new - gk)
            beta = (g_new @ (g_new - gk)) / denom if abs(denom) > 1e-300 else 0.0
        else:
            raise ValueError("variant must be 'fr', 'pr' or 'hs'")
        if k % restart == 0:
            beta = 0.0                      # periodic restart along -gradient
        d = -g_new + beta * d
        if d @ g_new > 0:                   # not a descent direction: reset
            d = -g_new
        x, gk = x_new, g_new
        history.append(x.copy())
    return OptimizeResult(x, float(fc(x)), gk, None, max_iter, False, fc.calls,
                          max_iter, f"cg_{variant}", history,
                          "maximum iterations reached")


def conjugate_gradient_fr(f, x0, grad_f=None, **kwargs):
    """Fletcher-Reeves nonlinear conjugate gradient."""
    return nonlinear_cg(f, x0, grad_f, "fr", **kwargs)


def conjugate_gradient_pr(f, x0, grad_f=None, **kwargs):
    """Polak-Ribiere nonlinear conjugate gradient (usually the most robust)."""
    return nonlinear_cg(f, x0, grad_f, "pr", **kwargs)


def conjugate_gradient_hs(f, x0, grad_f=None, **kwargs):
    """Hestenes-Stiefel nonlinear conjugate gradient."""
    return nonlinear_cg(f, x0, grad_f, "hs", **kwargs)


def barzilai_borwein(f, x0, grad_f=None, tol: float = 1e-8, max_iter: int = 5000,
                     variant: int = 1, memory: int = 10, sigma: float = 1e-4):
    """Barzilai-Borwein: a two-point step approximating the Newton scaling.

    Gradient-only cost with markedly better convergence than steepest descent.
    The raw iteration is non-monotone and can stall (it diverges on Rosenbrock),
    so Raydan's globalization is applied: a step is accepted when it improves on
    the *worst* of the last ``memory`` objective values, and is backtracked
    otherwise.
    """
    fc = CountedFunction(f)
    g = _grad(grad_f, fc)
    x = as_vector(x0).copy()
    gk = g(x)
    alpha = 1.0 / (np.linalg.norm(gk) + 1e-12)
    f_hist = [float(fc(x))]
    history = [x.copy()]
    for k in range(1, max_iter + 1):
        if np.linalg.norm(gk) < tol:
            return OptimizeResult(x, float(fc(x)), gk, None, k - 1, True, fc.calls,
                                  k, "barzilai_borwein", history, "converged")
        f_ref = max(f_hist[-memory:])
        step = alpha
        gg = float(gk @ gk)
        for _ in range(40):                      # non-monotone backtracking
            x_new = x - step * gk
            if float(fc(x_new)) <= f_ref - sigma * step * gg:
                break
            step *= 0.5
        x_new = x - step * gk
        g_new = g(x_new)
        s = x_new - x
        y = g_new - gk
        sy = float(s @ y)
        # A non-positive s'y means the local curvature estimate is unusable and
        # the BB ratio comes out negative. Clamping such a value would freeze
        # the iteration at a microscopic step, so fall back to a safe default.
        if not np.isfinite(sy) or sy <= 1e-300:
            alpha = min(1.0, 1.0 / (np.linalg.norm(g_new) + 1e-12))
        else:
            alpha = float(s @ s) / sy if variant == 1 else sy / float(y @ y)
            alpha = float(np.clip(alpha, 1e-10, 1e10))
        x, gk = x_new, g_new
        f_hist.append(float(fc(x)))
        history.append(x.copy())
    return OptimizeResult(x, float(fc(x)), gk, None, max_iter, False, fc.calls,
                          max_iter, "barzilai_borwein", history,
                          "maximum iterations reached")
