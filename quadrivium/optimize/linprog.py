"""Linear programming.

The simplex method walks the vertices of the feasible polytope; interior point
methods cut through its interior. Both are provided, together with the
transportation and assignment special cases.
"""

from __future__ import annotations

from ._history import History, monitor

from .. import numeric as np

from ..core.types import OptimizeResult
from ..core.utils import as_vector

__all__ = [
    "simplex",
    "simplex_tableau",
    "two_phase_simplex",
    "big_m_simplex",
    "interior_point_lp",
    "linprog",
    "standard_form",
    "assignment_problem",
]


def standard_form(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    """Convert ``min c'x`` with mixed constraints to ``A x = b, x >= 0``.

    Slack variables are appended for the inequality rows; the caller keeps the
    original variable count to recover the solution.
    """
    c = as_vector(c)
    n = c.size
    rows, rhs = [], []
    n_slack = 0 if A_ub is None else np.atleast_2d(A_ub).shape[0]
    if A_ub is not None:
        A_ub = np.atleast_2d(np.asarray(A_ub, dtype=float))
        b_ub = as_vector(b_ub)
        S = np.eye(n_slack)
        rows.append(np.hstack([A_ub, S]))
        rhs.append(b_ub)
    if A_eq is not None:
        A_eq = np.atleast_2d(np.asarray(A_eq, dtype=float))
        rows.append(np.hstack([A_eq, np.zeros((A_eq.shape[0], n_slack))]))
        rhs.append(as_vector(b_eq))
    A = np.vstack(rows) if rows else np.zeros((0, n + n_slack))
    b = np.concatenate(rhs) if rhs else np.zeros(0)
    c_std = np.concatenate([c, np.zeros(n_slack)])
    # the simplex needs a non-negative right-hand side
    for i in range(A.shape[0]):
        if b[i] < 0:
            A[i] *= -1.0
            b[i] *= -1.0
    return c_std, A, b, n


def simplex_tableau(c, A, b, basis, max_iter: int = 10000, tol: float = 1e-10):
    """Run the primal simplex from a given basis on ``min c'x``, ``Ax = b, x >= 0``.

    Returns ``(x, basis, status)`` where status is ``'optimal'``, ``'unbounded'``
    or ``'max_iter'``. Bland's rule is used on degenerate pivots to guarantee
    termination.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float)).copy()
    b = as_vector(b).copy()
    c = as_vector(c)
    m, n = A.shape
    basis = list(basis)
    # canonical form with respect to the current basis
    B = A[:, basis]
    T = np.hstack([np.linalg.solve(B, A), np.linalg.solve(B, b).reshape(-1, 1)])
    for it in range(max_iter):
        cb = c[basis]
        # reduced costs
        z = cb @ T[:, :n]
        reduced = c - z
        degenerate = np.any(np.abs(T[:, n]) < tol)
        candidates = np.flatnonzero(reduced < -tol)
        if candidates.size == 0:
            x = np.zeros(n)
            x[basis] = T[:, n]
            return x, basis, "optimal"
        # Bland's rule when degenerate (guarantees no cycling), Dantzig otherwise
        j = int(candidates[0]) if degenerate else int(np.argmin(reduced))
        col = T[:, j]
        pos = np.flatnonzero(col > tol)
        if pos.size == 0:
            x = np.zeros(n)
            x[basis] = T[:, n]
            return x, basis, "unbounded"
        ratios = T[pos, n] / col[pos]
        best = np.min(ratios)
        ties = pos[np.abs(ratios - best) < tol]
        i = int(min(ties, key=lambda r: basis[r])) if degenerate else int(ties[0])
        # pivot
        T[i] /= T[i, j]
        for r in range(m):
            if r != i and abs(T[r, j]) > 0:
                T[r] -= T[r, j] * T[i]
        basis[i] = j
    x = np.zeros(n)
    x[basis] = T[:, n]
    return x, basis, "max_iter"


def two_phase_simplex(c, A, b, max_iter: int = 10000, tol: float = 1e-10):
    """Two-phase simplex for the standard form ``min c'x  s.t.  A x = b, x >= 0``.

    ``A x = b`` are *equalities*, not inequalities -- convert inequalities by
    adding slack variables first, or call :func:`simplex` / :func:`linprog`,
    which accept ``A_ub``/``b_ub`` and do that for you.  Phase 1 minimizes the
    sum of artificial variables to find a feasible basis; phase 2 optimizes
    from it.  Returns ``(x, basis, status)``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = as_vector(b)
    c = as_vector(c)
    m, n = A.shape
    # phase 1: minimize the sum of artificial variables
    A1 = np.hstack([A, np.eye(m)])
    c1 = np.concatenate([np.zeros(n), np.ones(m)])
    basis = list(range(n, n + m))
    x1, basis, status = simplex_tableau(c1, A1, b, basis, max_iter, tol)
    if status != "optimal" or float(c1 @ x1) > tol * max(1.0, float(np.sum(b))):
        return None, None, "infeasible"
    # drive any remaining artificial variables out of the basis
    for i, bi in enumerate(list(basis)):
        if bi >= n:
            B = A1[:, basis]
            row = np.linalg.solve(B, A1)[i, :n]
            j = np.flatnonzero(np.abs(row) > tol)
            if j.size:
                basis[i] = int(j[0])
    x, basis, status = simplex_tableau(c, A, b, basis, max_iter, tol)
    return x, basis, status


def big_m_simplex(c, A, b, M: float = 1e6, max_iter: int = 10000,
                  tol: float = 1e-10):
    """Big-M simplex for ``min c'x  s.t.  A x = b, x >= 0`` (equalities).

    Penalizes artificial variables in a single phase instead of solving a
    separate phase-1 problem.  Simpler than two-phase but numerically
    delicate -- a large ``M`` swamps the genuine costs, so the two-phase
    method is preferred in practice.  Returns ``(x, basis, status)``.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = as_vector(b)
    c = as_vector(c)
    m, n = A.shape
    A_big = np.hstack([A, np.eye(m)])
    c_big = np.concatenate([c, M * np.ones(m)])
    basis = list(range(n, n + m))
    x, basis, status = simplex_tableau(c_big, A_big, b, basis, max_iter, tol)
    if x is not None and np.any(x[n:] > 1e-6):
        return None, None, "infeasible"
    return (x[:n] if x is not None else None), basis, status


def simplex(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, max_iter: int = 10000):
    """Solve a linear program by the two-phase simplex method.

    Minimizes ``c'x`` subject to ``A_ub x <= b_ub``, ``A_eq x = b_eq``, ``x >= 0``.
    """
    c_std, A, b, n = standard_form(c, A_ub, b_ub, A_eq, b_eq)
    if A.shape[0] == 0:
        x = np.zeros(n)
        return OptimizeResult(x, 0.0, None, None, 0, True, 0, 0, "simplex", [],
                              "no constraints")
    x, basis, status = two_phase_simplex(c_std, A, b, max_iter)
    if status == "infeasible":
        return OptimizeResult(np.full(n, np.nan), np.nan, None, None, 0, False, 0,
                              0, "simplex", [], "problem is infeasible")
    if status == "unbounded":
        return OptimizeResult(x[:n], -np.inf, None, None, 0, False, 0, 0,
                              "simplex", [], "problem is unbounded")
    res = OptimizeResult(x[:n], float(as_vector(c) @ x[:n]), None, None, 0,
                         status == "optimal", 0, 0, "simplex", [], status)
    res.slack = x[n:]
    res.basis = basis
    return res


def interior_point_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                      tol: float = 1e-10, max_iter: int = 200,
                      sigma: float = 0.1):
    """Primal-dual interior point method with Mehrotra-style centering.

    Follows the central path from the interior, so the work is polynomial in
    the problem size rather than combinatorial in the vertices.
    """
    c_std, A, b, n_orig = standard_form(c, A_ub, b_ub, A_eq, b_eq)
    A = np.atleast_2d(A)
    m, n = A.shape
    if m == 0:
        return OptimizeResult(np.zeros(n_orig), 0.0, None, None, 0, True, 0, 0,
                              "interior_point", [], "no constraints")
    # strictly positive start
    x = np.ones(n)
    lam = np.zeros(m)
    s = np.ones(n)
    history = History()
    for k in range(1, max_iter + 1):
        r_p = A @ x - b
        r_d = A.T @ lam + s - c_std
        mu = float(x @ s) / n
        history.append(mu, x=x)
        if (np.linalg.norm(r_p) < tol and np.linalg.norm(r_d) < tol and mu < tol):
            break
        # Newton step on the perturbed KKT system
        #   A dx = -r_p,  A' dlam + ds = -r_d,  S dx + X ds = -XSe + sigma*mu*e
        # Eliminating dx and ds gives the normal equations
        #   (A D A') dlam = -r_p - A (D r_d - x + sigma*mu/s),   D = x/s
        D = x / s
        M = (A * D) @ A.T
        rhs = -r_p - A @ (D * r_d - x + sigma * mu / s)
        try:
            d_lam = np.linalg.solve(M + 1e-12 * np.eye(m), rhs)
        except np.linalg.LinAlgError:
            d_lam = np.linalg.lstsq(M, rhs, rcond=None)[0]
        d_s = -r_d - A.T @ d_lam
        d_x = -(x * s - sigma * mu) / s - D * d_s
        # step to the boundary, damped to stay strictly interior
        neg_x = d_x < 0
        neg_s = d_s < 0
        a_p = min(1.0, 0.99 * np.min(-x[neg_x] / d_x[neg_x])) if neg_x.any() else 1.0
        a_d = min(1.0, 0.99 * np.min(-s[neg_s] / d_s[neg_s])) if neg_s.any() else 1.0
        x = x + a_p * d_x
        lam = lam + a_d * d_lam
        s = s + a_d * d_s
        x = np.maximum(x, 1e-14)
        s = np.maximum(s, 1e-14)
    res = OptimizeResult(x[:n_orig], float(as_vector(c) @ x[:n_orig]), None, None,
                         k, mu < 1e-6, 0, 0, "interior_point_lp", history,
                         f"duality measure {mu:.2e}")
    res.dual = lam
    return res


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, method: str = "simplex",
            **kwargs):
    """Solve a linear program with the simplex or interior point method."""
    if method == "simplex":
        return simplex(c, A_ub, b_ub, A_eq, b_eq, **kwargs)
    if method in ("interior_point", "ipm"):
        return interior_point_lp(c, A_ub, b_ub, A_eq, b_eq, **kwargs)
    raise ValueError(f"unknown method {method!r}")


def assignment_problem(cost, maximize: bool = False):
    """Hungarian algorithm for the linear assignment problem.

    Returns ``(row_indices, col_indices, total_cost)`` for the optimal
    one-to-one matching in ``O(n^3)``.
    """
    C = np.array(cost, dtype=float)
    if maximize:
        C = C.max() - C
    n, m = C.shape
    size = max(n, m)
    M = np.full((size, size), C.max() + 1.0 if C.size else 0.0)
    M[:n, :m] = C
    # Jonker-Volgenant style shortest augmenting path
    INF = np.inf
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, INF)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = M[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    rows, cols = [], []
    for j in range(1, size + 1):
        if p[j] <= n and j <= m:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    order = np.argsort(rows)
    rows = np.array(rows)[order]
    cols = np.array(cols)[order]
    total = float(np.array(cost, dtype=float)[rows, cols].sum())
    return rows, cols, total


# Apply a common context-local output policy to public iterative entry points.
for _name in ['interior_point_lp']:
    globals()[_name] = monitor(globals()[_name])
