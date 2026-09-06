"""Numerical solution of stochastic differential equations.

Schemes for the Ito SDE ``dX = a(X, t) dt + b(X, t) dW`` together with the
driving Brownian machinery and exact-simulation algorithms for jump processes.

A word on what "order" means here, because it is not the deterministic notion.
*Strong* order measures pathwise accuracy, ``E|X_N - X(T)|``: it is what
matters when the individual trajectory is the object of interest.  *Weak* order
measures accuracy of distributions, ``|E g(X_N) - E g(X(T))|``: it is what
matters when only expectations are wanted, and it is typically higher.
Euler-Maruyama is strong order 1/2 but weak order 1 -- the gap is why halving
the step buys so little pathwise accuracy and so much in expectation.
"""

from __future__ import annotations

from .. import numeric as np

from ..core.types import ODESolution
from ..core.utils import as_vector

__all__ = [
    "brownian_path",
    "brownian_bridge",
    "euler_maruyama",
    "milstein",
    "implicit_milstein",
    "stochastic_heun",
    "stochastic_rk",
    "srk_strong_1_5",
    "tamed_euler",
    "geometric_brownian_motion",
    "ornstein_uhlenbeck",
    "cox_ingersoll_ross",
    "gillespie_ssa",
    "tau_leaping",
    "poisson_process",
    "strong_error",
    "weak_error",
]


def _rng(rng):
    return np.random.default_rng(rng)


def _drift_diffusion(a, b):
    """Normalize the coefficients to vector-valued callables of ``(x, t)``."""
    def av(x, t):
        return as_vector(a(x, t))

    def bv(x, t):
        # Same coercion as the drift above. `as_vector` returns an already
        # contiguous 1-D float array untouched, which is what every step of a
        # solve hands it.
        return as_vector(b(x, t))

    return av, bv


def brownian_path(t_span, n: int = 1000, dim: int = 1, rng=None):
    """Sample a standard Brownian path on a uniform grid.

    Returns ``(t, W)`` with ``W[0] = 0`` and ``W`` of shape ``(n+1, dim)``.
    """
    rng = _rng(rng)
    t = np.linspace(float(t_span[0]), float(t_span[1]), n + 1)
    dt = t[1] - t[0]
    dW = np.sqrt(dt) * rng.standard_normal((n, dim))
    W = np.vstack([np.zeros(dim), np.cumsum(dW, axis=0)])
    return t, W


def brownian_bridge(t_span, x0, x1, n: int = 1000, dim: int = 1, rng=None):
    """Brownian bridge pinned to ``x0`` at the start and ``x1`` at the end.

    Built by subtracting the linear drift that a free path would need to land
    on ``x1``: ``B(t) = W(t) - (t/T) W(T)`` shifted onto the two endpoints.
    """
    t, W = brownian_path(t_span, n, dim, rng)
    T = t[-1] - t[0]
    s = ((t - t[0]) / T)[:, None]
    x0 = np.broadcast_to(np.atleast_1d(np.asarray(x0, dtype=float)), (dim,))
    x1 = np.broadcast_to(np.atleast_1d(np.asarray(x1, dtype=float)), (dim,))
    B = W - s * W[-1]
    return t, B + (1 - s) * x0 + s * x1


def _sde_loop(step, a, b, t_span, x0, n, rng, name, dW=None):
    """Shared driver: allocate, walk the grid, package the result."""
    rng = _rng(rng)
    a, b = _drift_diffusion(a, b)
    x = as_vector(x0).astype(float)
    d = x.size
    t = np.linspace(float(t_span[0]), float(t_span[1]), n + 1)
    dt = t[1] - t[0]
    if dW is None:
        dW = np.sqrt(dt) * rng.standard_normal((n, d))
    else:
        dW = np.atleast_2d(np.asarray(dW, dtype=float)).reshape(n, d)
    X = np.empty((n + 1, d))
    X[0] = x
    for k in range(n):
        x = step(a, b, x, t[k], dt, dW[k], rng)
        X[k + 1] = x
    return ODESolution(t, X, name, n, n, 0, 0, True, "completed")


def euler_maruyama(a, b, t_span, x0, n: int = 1000, rng=None, dW=None):
    """Euler-Maruyama: ``X += a dt + b dW``.  Strong order 1/2, weak order 1.

    The direct analogue of forward Euler.  Its strong order is only 1/2 --
    *not* 1 -- because the Ito-Taylor expansion has a term ``b b' (dW^2 - dt)/2``
    of size ``dt`` that this scheme discards; :func:`milstein` keeps it.
    """
    def step(a, b, x, t, dt, dw, rng):
        return x + a(x, t) * dt + b(x, t) * dw

    return _sde_loop(step, a, b, t_span, x0, n, rng, "euler_maruyama", dW)


def milstein(a, b, t_span, x0, n: int = 1000, rng=None, db=None, dW=None):
    """Milstein scheme: strong order 1 for scalar noise.

    Adds the Ito correction ``0.5 b b' (dW^2 - dt)`` that Euler-Maruyama drops.
    ``db`` is the derivative ``b'(x, t)``; without it a central difference is
    used.  For genuinely multidimensional noise the scheme also needs Levy
    areas, so this implementation applies the diagonal-noise form.
    """
    def dbn(x, t):
        if db is not None:
            return as_vector(db(x, t))
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
        return (as_vector(b(x + h, t)) - as_vector(b(x - h, t))) / (2 * h)

    def step(a, b, x, t, dt, dw, rng):
        bx = b(x, t)
        return x + a(x, t) * dt + bx * dw + 0.5 * bx * dbn(x, t) * (dw * dw - dt)

    return _sde_loop(step, a, b, t_span, x0, n, rng, "milstein", dW)


def implicit_milstein(a, b, t_span, x0, n: int = 1000, rng=None, db=None,
                      theta: float = 0.5, tol: float = 1e-10, max_iter: int = 50):
    """Drift-implicit Milstein: stable on stiff SDEs.

    Only the *drift* is treated implicitly.  Making the diffusion implicit as
    well would be wrong, not merely awkward: ``dW`` can take either sign, so an
    implicit diffusion term does not define a contraction and the resulting
    scheme need not have a solution at all.
    """
    def dbn(x, t):
        if db is not None:
            return as_vector(db(x, t))
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
        return (as_vector(b(x + h, t)) - as_vector(b(x - h, t))) / (2 * h)

    def step(a, b, x, t, dt, dw, rng):
        bx = b(x, t)
        explicit = (x + (1 - theta) * a(x, t) * dt + bx * dw
                    + 0.5 * bx * dbn(x, t) * (dw * dw - dt))
        y = x + a(x, t) * dt + bx * dw          # Euler predictor
        for _ in range(max_iter):               # fixed-point on the drift
            y_new = explicit + theta * a(y, t + dt) * dt
            if np.max(np.abs(y_new - y)) < tol:
                return y_new
            y = y_new
        return y

    return _sde_loop(step, a, b, t_span, x0, n, rng, "implicit_milstein")


def stochastic_heun(a, b, t_span, x0, n: int = 1000, rng=None, dW=None):
    """Stochastic Heun (Stratonovich predictor-corrector).

    Converges to the *Stratonovich* solution, which differs from the Ito one by
    the drift correction ``0.5 b b'`` -- so for multiplicative noise this and
    :func:`euler_maruyama` are solving genuinely different equations.
    """
    def step(a, b, x, t, dt, dw, rng):
        ax, bx = a(x, t), b(x, t)
        xp = x + ax * dt + bx * dw
        return x + 0.5 * (ax + a(xp, t + dt)) * dt + 0.5 * (bx + b(xp, t + dt)) * dw

    return _sde_loop(step, a, b, t_span, x0, n, rng, "stochastic_heun", dW)


def stochastic_rk(a, b, t_span, x0, n: int = 1000, rng=None, dW=None):
    """Derivative-free Runge-Kutta of strong order 1 (Platen).

    Reproduces Milstein's correction using a supporting value instead of
    ``b'``, which makes it the practical choice when the derivative of the
    diffusion is unavailable.
    """
    def step(a, b, x, t, dt, dw, rng):
        sdt = np.sqrt(dt)
        bx = b(x, t)
        xs = x + a(x, t) * dt + bx * sdt          # supporting value
        return (x + a(x, t) * dt + bx * dw
                + 0.5 * (b(xs, t) - bx) * (dw * dw - dt) / sdt)

    return _sde_loop(step, a, b, t_span, x0, n, rng, "stochastic_rk", dW)


def srk_strong_1_5(a, b, t_span, x0, n: int = 1000, rng=None):
    """Strong order 1.5 Taylor scheme for additive noise.

    Restricted to ``b`` independent of ``x``.  With additive noise the only
    extra Ito integral needed is ``I_{1,0}``, which is jointly Gaussian with
    ``dW`` and can be sampled exactly -- the double integrals that block the
    general case never appear.
    """
    def step(a, b, x, t, dt, dw, rng):
        # (dW, I10) is bivariate normal with Cov = [[dt, dt^2/2],[dt^2/2, dt^3/3]].
        u1, u2 = rng.standard_normal(x.size), rng.standard_normal(x.size)
        dw = np.sqrt(dt) * u1
        i10 = 0.5 * dt ** 1.5 * (u1 + u2 / np.sqrt(3.0))
        ax, bx = a(x, t), b(x, t)
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
        da = (a(x + h, t) - a(x - h, t)) / (2 * h)          # a_x
        dat = (a(x, t + 1e-7) - a(x, t - 1e-7)) / 2e-7      # a_t
        d2a = (a(x + h, t) - 2 * ax + a(x - h, t)) / (h * h)
        return (x + ax * dt + bx * dw
                + da * bx * i10
                + 0.5 * dt * dt * (dat + ax * da + 0.5 * bx * bx * d2a))

    return _sde_loop(step, a, b, t_span, x0, n, rng, "srk_strong_1_5")


def tamed_euler(a, b, t_span, x0, n: int = 1000, rng=None):
    """Tamed Euler-Maruyama for super-linearly growing drift.

    Plain Euler-Maruyama *diverges in the strong sense* when the drift grows
    faster than linearly (Hutzenthaler-Jentzen-Kloeden): the moments blow up
    however small the step.  Scaling the drift by ``1/(1 + dt|a|)`` restores
    convergence while leaving the small-step limit unchanged.
    """
    def step(a, b, x, t, dt, dw, rng):
        ax = a(x, t)
        return x + ax * dt / (1.0 + dt * np.linalg.norm(ax)) + b(x, t) * dw

    return _sde_loop(step, a, b, t_span, x0, n, rng, "tamed_euler")


def geometric_brownian_motion(x0, mu: float, sigma: float, t_span, n: int = 1000,
                              rng=None, exact: bool = True):
    """Geometric Brownian motion ``dX = mu X dt + sigma X dW``.

    ``exact=True`` uses the closed form ``X0 exp((mu - sigma^2/2)t + sigma W)``,
    which is exact at every grid point -- there is no discretization error at
    all, only the sampling of ``W``.
    """
    if not exact:
        return euler_maruyama(lambda x, t: mu * x, lambda x, t: sigma * x,
                              t_span, x0, n, rng)
    t, W = brownian_path(t_span, n, 1, rng)
    x0 = as_vector(x0).astype(float)
    drift = (mu - 0.5 * sigma * sigma) * (t - t[0])
    X = x0[0] * np.exp(drift[:, None] + sigma * W)
    return ODESolution(t, X, "gbm_exact", n, n, 0, 0, True, "exact solution")


def ornstein_uhlenbeck(x0, theta: float, mu: float, sigma: float, t_span,
                       n: int = 1000, rng=None, exact: bool = True):
    """Ornstein-Uhlenbeck ``dX = theta (mu - X) dt + sigma dW``.

    ``exact=True`` samples the transition density directly: the process is
    Gaussian, so each step is drawn from its exact conditional law and the
    result carries no discretization error whatever the step size.
    """
    if not exact:
        return euler_maruyama(lambda x, t: theta * (mu - x),
                              lambda x, t: sigma * np.ones_like(x),
                              t_span, x0, n, rng)
    rng = _rng(rng)
    x = as_vector(x0).astype(float)
    t = np.linspace(float(t_span[0]), float(t_span[1]), n + 1)
    dt = t[1] - t[0]
    decay = np.exp(-theta * dt)
    sd = sigma * np.sqrt((1.0 - decay * decay) / (2.0 * theta)) if theta > 0 \
        else sigma * np.sqrt(dt)
    X = np.empty((n + 1, x.size))
    X[0] = x
    for k in range(n):
        x = mu + (x - mu) * decay + sd * rng.standard_normal(x.size)
        X[k + 1] = x
    return ODESolution(t, X, "ou_exact", n, n, 0, 0, True, "exact transitions")


def cox_ingersoll_ross(x0, theta: float, mu: float, sigma: float, t_span,
                       n: int = 1000, rng=None):
    """CIR process ``dX = theta (mu - X) dt + sigma sqrt(X) dW``.

    The square root makes naive Euler steps go negative, at which point the
    diffusion is undefined; the full-truncation fix evaluates the coefficients
    at ``max(X, 0)`` and is the standard remedy.  ``2 theta mu >= sigma^2``
    (the Feller condition) keeps the exact process strictly positive.
    """
    def step(a, b, x, t, dt, dw, rng):
        xp = np.maximum(x, 0.0)
        return x + theta * (mu - xp) * dt + sigma * np.sqrt(xp) * dw

    sol = _sde_loop(step, lambda x, t: theta * (mu - np.maximum(x, 0.0)),
                    lambda x, t: sigma * np.sqrt(np.maximum(x, 0.0)),
                    t_span, x0, n, rng, "cir")
    sol.y[:] = np.maximum(sol.y, 0.0)
    return sol


def poisson_process(rate: float, t_span, rng=None, max_events: int = 1000000):
    """Event times of a homogeneous Poisson process, by exponential gaps."""
    rng = _rng(rng)
    t0, tf = float(t_span[0]), float(t_span[1])
    times = []
    t = t0
    while len(times) < max_events:
        t += rng.exponential(1.0 / rate)
        if t > tf:
            break
        times.append(t)
    return np.array(times)


def gillespie_ssa(propensities, stoichiometry, x0, t_span, rng=None,
                  max_events: int = 1000000):
    """Gillespie's stochastic simulation algorithm for reaction networks.

    ``propensities(x, t)`` returns the reaction rates and ``stoichiometry`` is
    the ``(n_reactions, n_species)`` change matrix.  This is an *exact*
    simulation of the chemical master equation, not a discretization: the
    waiting time is drawn from its true exponential law and the reaction from
    its true categorical law, so no step-size error exists.  The cost is one
    step per individual reaction event.

    Returns ``(times, states)`` with one row per event.
    """
    rng = _rng(rng)
    S = np.atleast_2d(np.asarray(stoichiometry, dtype=float))
    x = as_vector(x0).astype(float)
    t0, tf = float(t_span[0]), float(t_span[1])
    t = t0
    times, states = [t], [x.copy()]
    for _ in range(max_events):
        rates = np.atleast_1d(np.asarray(propensities(x, t), dtype=float))
        total = float(np.sum(rates))
        if total <= 0.0:
            break                      # no reaction can fire; the state is absorbing
        t += rng.exponential(1.0 / total)
        if t > tf:
            break
        j = int(np.searchsorted(np.cumsum(rates), rng.random() * total))
        x = x + S[min(j, S.shape[0] - 1)]
        times.append(t)
        states.append(x.copy())
    return np.array(times), np.array(states)


def tau_leaping(propensities, stoichiometry, x0, t_span, tau: float = 0.01,
                rng=None):
    """Explicit tau-leaping: fire Poisson-many reactions per fixed step.

    Trades exactness for speed by holding the propensities fixed over ``tau``
    and drawing each reaction count from a Poisson law.  Species counts are
    clipped at zero, since a leap can otherwise overshoot into negative
    populations -- the well-known failure mode of the naive scheme.
    """
    rng = _rng(rng)
    S = np.atleast_2d(np.asarray(stoichiometry, dtype=float))
    x = as_vector(x0).astype(float)
    t0, tf = float(t_span[0]), float(t_span[1])
    n = int(np.ceil((tf - t0) / tau))
    times = np.linspace(t0, t0 + n * tau, n + 1)
    states = np.empty((n + 1, x.size))
    states[0] = x
    for k in range(n):
        rates = np.atleast_1d(np.asarray(propensities(x, times[k]), dtype=float))
        counts = rng.poisson(np.maximum(rates, 0.0) * tau)
        x = np.maximum(x + counts @ S, 0.0)
        states[k + 1] = x
    return times, states


def strong_error(solver, exact, t_span, x0, n: int = 500, paths: int = 200,
                 rng=None, **kwargs):
    """Mean pathwise error ``E|X_N - X(T)|`` at the final time.

    ``exact(t, W_T)`` must return the true endpoint for the *same* Brownian
    path, which is what makes this a strong (pathwise) measure rather than a
    weak one.
    """
    rng = _rng(rng)
    total = 0.0
    for _ in range(paths):
        dt = (float(t_span[1]) - float(t_span[0])) / n
        dW = np.sqrt(dt) * rng.standard_normal((n, np.size(as_vector(x0))))
        sol = solver(t_span=t_span, x0=x0, n=n, dW=dW, **kwargs)
        total += float(np.abs(sol.y[-1] - exact(t_span[1], np.sum(dW, axis=0))).max())
    return total / paths


def weak_error(solver, g, exact_mean: float, t_span, x0, n: int = 500,
               paths: int = 2000, rng=None, **kwargs):
    """Error in an expectation ``|E g(X_N) - E g(X(T))|``.

    Weak error is typically an order better than strong error, and is the
    relevant measure whenever only averages are wanted.
    """
    rng = _rng(rng)
    vals = [float(g(solver(t_span=t_span, x0=x0, n=n, rng=rng, **kwargs).y[-1]))
            for _ in range(paths)]
    return abs(float(np.mean(vals)) - exact_mean)
