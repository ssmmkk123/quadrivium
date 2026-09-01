"""Matrix functions, matrix equations and randomized decompositions.

Three related themes that the classical factorizations leave uncovered:
evaluating ``f(A)`` for a matrix argument, solving equations whose unknown is a
matrix (Sylvester, Lyapunov, Riccati), and building low-rank approximations by
random projection.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import ConvergenceError, DimensionError, SingularMatrixError
from ..core.types import EigenResult
from ..core.utils import as_matrix, check_square, is_symmetric
from .direct import householder_qr, plu_solve
from .eigen import matrix_exponential, schur

__all__ = [
    "sqrtm",
    "logm",
    "signm",
    "matrix_sign_iteration",
    "sylvester",
    "lyapunov",
    "discrete_lyapunov",
    "care_newton",
    "kron",
    "vec",
    "unvec",
    "condition_estimate",
    "randomized_range_finder",
    "randomized_svd",
    "randomized_eigh",
    "nystrom_approximation",
    "interpolative_decomposition",
    "cur_decomposition",
    "subspace_iteration",
    "lobpcg",
    "generalized_eigh",
    "qz_decomposition",
    "qz_eigenvalues",
]


# --------------------------------------------------------------------------
# Matrix functions
# --------------------------------------------------------------------------
def sqrtm(A, tol: float = 1e-13, max_iter: int = 100):
    """Principal matrix square root ``X`` with ``X @ X == A``.

    Uses the Denman-Beavers iteration in its *scaled* form.  The unscaled
    iteration converges quadratically but can take hundreds of steps on a
    badly scaled matrix; the determinant-based scaling below equalizes the
    eigenvalue magnitudes each step and typically finishes in under ten.

    A matrix with a negative real eigenvalue has no real square root, and the
    result is then complex.
    """
    A = check_square(A)
    n = A.shape[0]
    if np.any(np.real(np.linalg.eigvals(A)) < 0) or np.iscomplexobj(A):
        Y = np.array(A, dtype=complex)
    else:
        Y = np.array(A, dtype=float)
    Z = np.eye(n, dtype=Y.dtype)
    for k in range(1, max_iter + 1):
        # Determinantal scaling mu = |det(Y) det(Z)|^{-1/2n}.  It has to equal 1
        # at the solution -- where Z = Y^-1, so det(Y)det(Z) = 1 -- otherwise
        # the scaling moves the fixed point and the iteration converges
        # smoothly to the wrong matrix.  Computed via slogdet to survive the
        # very large or small determinants that scaling exists to fix.
        sy, ly = np.linalg.slogdet(Y)
        sz, lz = np.linalg.slogdet(Z)
        g = 1.0 if sy == 0 or sz == 0 else float(np.exp(-(ly + lz) / (2.0 * n)))
        Yn = 0.5 * (g * Y + np.linalg.inv(g * Z))
        Zn = 0.5 * (g * Z + np.linalg.inv(g * Y))
        err = np.max(np.abs(Yn - Y))
        Y, Z = Yn, Zn
        if err < tol * max(1.0, np.max(np.abs(Y))):
            break
    else:
        raise ConvergenceError(f"sqrtm did not converge in {max_iter} iterations")
    if not np.iscomplexobj(A) and np.max(np.abs(np.imag(Y))) < 1e-12:
        Y = np.real(Y)
    return Y


def logm(A, tol: float = 1e-13, max_iter: int = 64):
    """Principal matrix logarithm by inverse scaling and squaring.

    Repeated square roots pull the spectrum toward 1, where the Pade
    approximant to ``log(I + X)`` is accurate; each root taken costs one
    doubling of the answer at the end.  Squaring *up* front instead would
    magnify rounding error, which is why the scaling runs in this direction.
    """
    A = check_square(A)
    n = A.shape[0]
    X = np.array(A, dtype=complex if np.iscomplexobj(A) else float)
    s = 0
    # The Mercator series needs a submultiplicative norm below 1 to converge;
    # the largest entry is not one, so use the spectral norm.
    while np.linalg.norm(X - np.eye(n), 2) > 0.25 and s < max_iter:
        X = sqrtm(X, tol)
        s += 1
    Y = X - np.eye(n)
    # Mercator series on the near-identity matrix; ||Y|| <= 1/4 makes it fast.
    term = np.array(Y)
    total = np.array(Y)
    for k in range(2, 200):
        term = term @ Y
        contrib = ((-1.0) ** (k - 1)) * term / k
        total = total + contrib
        if np.max(np.abs(contrib)) < tol * max(1.0, np.max(np.abs(total))):
            break
    out = (2.0 ** s) * total
    if not np.iscomplexobj(A) and np.max(np.abs(np.imag(out))) < 1e-11:
        out = np.real(out)
    return out


def matrix_sign_iteration(A, tol: float = 1e-13, max_iter: int = 100):
    """Matrix sign function by the scaled Newton iteration.

    ``sign(A)`` maps each eigenvalue to ``+-1`` by the sign of its real part,
    so it splits the spectrum across the imaginary axis.  Purely imaginary
    eigenvalues make it undefined, and the iteration will fail to converge.
    """
    A = check_square(A)
    n = A.shape[0]
    X = np.array(A, dtype=complex if np.iscomplexobj(A) else float)
    for _ in range(max_iter):
        Xi = np.linalg.inv(X)
        mu = (abs(np.linalg.det(Xi)) / abs(np.linalg.det(X))) ** (1.0 / (2 * n))
        Xn = 0.5 * (mu * X + Xi / mu)
        err = np.max(np.abs(Xn - X))
        X = Xn
        if err < tol * max(1.0, np.max(np.abs(X))):
            return X
    raise ConvergenceError(
        "matrix sign iteration did not converge: the matrix likely has an "
        "eigenvalue on the imaginary axis, where sign(z) is undefined"
    )


def signm(A, **kwargs):
    """Matrix sign function (alias for :func:`matrix_sign_iteration`)."""
    return matrix_sign_iteration(A, **kwargs)


# --------------------------------------------------------------------------
# Matrix equations
# --------------------------------------------------------------------------
def vec(A):
    """Column-major (Fortran) vectorization, the convention Kronecker identities use."""
    return np.asarray(A, dtype=float).reshape(-1, order="F")


def unvec(v, shape):
    """Inverse of :func:`vec`."""
    return np.asarray(v, dtype=float).reshape(shape, order="F")


def kron(A, B):
    """Kronecker product."""
    return np.kron(np.asarray(A, dtype=float), np.asarray(B, dtype=float))


def sylvester(A, B, C):
    """Solve the Sylvester equation ``A X + X B = C``.

    Uses the Bartels-Stewart algorithm: reduce ``A`` and ``B`` to real Schur
    form, solve the resulting quasi-triangular system one column block at a
    time, then transform back.  This costs ``O(n^3)``, against the ``O(n^6)``
    of forming the ``n^2``-by-``n^2`` Kronecker system directly -- the reason
    the algorithm exists.

    A solution exists and is unique iff ``A`` and ``-B`` share no eigenvalue.
    """
    A = check_square(A)
    B = check_square(B)
    C = as_matrix(C)
    m, n = A.shape[0], B.shape[0]
    if C.shape != (m, n):
        raise DimensionError(f"C must have shape {(m, n)}, got {C.shape}")
    ea = np.linalg.eigvals(A)
    eb = np.linalg.eigvals(B)
    if np.min(np.abs(ea[:, None] + eb[None, :])) < 1e-12 * max(
            1.0, np.max(np.abs(ea)) + np.max(np.abs(eb))):
        raise SingularMatrixError(
            "Sylvester equation is singular: A and -B share an eigenvalue"
        )
    # Reduce both sides to (complex) Schur form so the system is triangular.
    Ta, Ua = _complex_schur(A)
    Tb, Ub = _complex_schur(B)
    F = Ua.conj().T @ C @ Ub
    Y = np.zeros((m, n), dtype=complex)
    for j in range(n):
        rhs = F[:, j] - Y[:, :j] @ Tb[:j, j]
        M = Ta + Tb[j, j] * np.eye(m)
        Y[:, j] = np.linalg.solve(M, rhs)
    X = Ua @ Y @ Ub.conj().T
    if not (np.iscomplexobj(A) or np.iscomplexobj(B) or np.iscomplexobj(C)):
        X = np.real(X)
    return X


def _complex_schur(A):
    """Complex Schur form ``A = U T U*`` with ``T`` upper triangular.

    Built from the real Schur form by splitting each 2-by-2 block with the
    unitary that triangularizes it, so the back-substitution in
    :func:`sylvester` never has to special-case a block.
    """
    Q, T = schur(A)
    T = T.astype(complex)
    U = Q.astype(complex)
    n = T.shape[0]
    i = 0
    while i < n - 1:
        if abs(T[i + 1, i]) > 0.0:
            a, b, c, d = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
            lam = 0.5 * (a + d) + np.sqrt(complex(0.25 * (a - d) ** 2 + b * c))
            v = np.array([b, lam - a]) if abs(b) > abs(c) else np.array([lam - d, c])
            nv = np.linalg.norm(v)
            if nv > 0:
                v = v / nv
                G = np.array([[v[0], -np.conj(v[1])], [v[1], np.conj(v[0])]])
                T[i:i + 2, :] = G.conj().T @ T[i:i + 2, :]
                T[:, i:i + 2] = T[:, i:i + 2] @ G
                U[:, i:i + 2] = U[:, i:i + 2] @ G
            T[i + 1, i] = 0.0
            i += 2
        else:
            i += 1
    T = np.triu(T)
    return T, U


def lyapunov(A, Q):
    """Solve the continuous Lyapunov equation ``A X + X A' + Q = 0``.

    A special case of Sylvester with ``B = A'`` and ``C = -Q``.  For stable
    ``A`` and positive semidefinite ``Q`` the solution is the controllability
    Gramian and is itself positive semidefinite.
    """
    A = check_square(A)
    Q = as_matrix(Q)
    X = sylvester(A, A.conj().T, -Q)
    return 0.5 * (X + X.conj().T)     # symmetrize away the rounding asymmetry


def discrete_lyapunov(A, Q, tol: float = 1e-14, max_iter: int = 200):
    """Solve the discrete Lyapunov (Stein) equation ``A X A' - X + Q = 0``.

    Squaring ("doubling") iteration: ``X <- X + A X A'`` with ``A <- A A``
    doubles the number of terms of the series ``sum A^k Q (A')^k`` per step,
    so it converges in ``O(log(1/eps))`` steps rather than linearly.  Requires
    ``A`` to have spectral radius below 1.
    """
    A = check_square(A)
    Q = as_matrix(Q)
    if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
        raise ConvergenceError(
            "discrete Lyapunov requires spectral radius of A below 1"
        )
    X = np.array(Q, dtype=float)
    Ak = np.array(A, dtype=float)
    for _ in range(max_iter):
        Xn = X + Ak @ X @ Ak.T
        Ak = Ak @ Ak
        err = np.max(np.abs(Xn - X))
        X = Xn
        if err < tol * max(1.0, np.max(np.abs(X))):
            break
    return 0.5 * (X + X.T)


def care_newton(A, B, Q, R, X0=None, tol: float = 1e-12, max_iter: int = 100):
    """Continuous algebraic Riccati equation ``A'X + XA - XBR^-1B'X + Q = 0``.

    Newton-Kleinman iteration: each step solves one Lyapunov equation for the
    closed-loop matrix, and converges quadratically once inside the basin.
    Returns ``(X, K)`` with the stabilizing gain ``K = R^-1 B' X``.
    """
    A = check_square(A)
    B = as_matrix(B)
    Q = as_matrix(Q)
    R = as_matrix(np.atleast_2d(R))
    Rinv = np.linalg.inv(R)
    n = A.shape[0]
    X = np.zeros((n, n)) if X0 is None else as_matrix(X0)
    for k in range(1, max_iter + 1):
        K = Rinv @ B.T @ X
        Acl = A - B @ K
        if np.max(np.real(np.linalg.eigvals(Acl))) >= 0 and k == 1 and X0 is None:
            # A zero start is only admissible when A itself is stable; otherwise
            # shift onto a stabilizing initial guess.
            X = np.eye(n) * (1.0 + np.max(np.abs(A)))
            continue
        Xn = lyapunov(Acl.T, Q + K.T @ R @ K)
        err = np.max(np.abs(Xn - X))
        X = Xn
        if err < tol * max(1.0, np.max(np.abs(X))):
            return X, Rinv @ B.T @ X
    raise ConvergenceError(f"CARE Newton iteration did not converge in {max_iter} steps")


# --------------------------------------------------------------------------
# Condition estimation
# --------------------------------------------------------------------------
def condition_estimate(A, max_iter: int = 20):
    """Estimate the 1-norm condition number without forming the inverse.

    Hager's algorithm treats ``max ||A^-1 x||_1 / ||x||_1`` as a convex
    maximization over the unit ball and hill-climbs on the vertices.  It costs
    a handful of solves instead of the ``n`` needed to build ``A^-1``, and in
    practice lands within a factor of three of the true value.
    """
    A = check_square(A)
    n = A.shape[0]
    x = np.full(n, 1.0 / n)
    est = 0.0
    seen = set()
    for _ in range(max_iter):
        y = plu_solve(A, x)
        est = float(np.sum(np.abs(y)))
        xi = np.sign(y)
        xi[xi == 0] = 1.0
        z = plu_solve(A.T, xi)
        j = int(np.argmax(np.abs(z)))
        if abs(z[j]) <= float(z @ x) or j in seen:
            break                      # a local maximum of the convex problem
        seen.add(j)
        x = np.zeros(n)
        x[j] = 1.0
    return est * float(np.max(np.sum(np.abs(A), axis=0)))


# --------------------------------------------------------------------------
# Randomized methods
# --------------------------------------------------------------------------
def randomized_range_finder(A, size: int, power_iterations: int = 2, rng=None):
    """Orthonormal basis for an approximate range of ``A``, by random projection.

    ``A @ Omega`` with Gaussian ``Omega`` captures the dominant subspace with
    high probability.  The power iterations ``(A A')^q`` sharpen the spectral
    gap -- essential when the singular values decay slowly -- and the
    re-orthogonalization between them is what keeps that step from collapsing
    numerically onto the leading singular vector.
    """
    A = as_matrix(A)
    rng = np.random.default_rng(rng)
    Omega = rng.standard_normal((A.shape[1], size))
    Y = A @ Omega
    Q, _ = householder_qr(Y, reduced=True)
    for _ in range(power_iterations):
        Z, _ = householder_qr(A.T @ Q, reduced=True)
        Q, _ = householder_qr(A @ Z, reduced=True)
    return Q


def randomized_svd(A, k: int, oversampling: int = 10, power_iterations: int = 2,
                   rng=None):
    """Approximate truncated SVD by random projection (Halko-Martinsson-Tropp).

    Costs one pass over ``A`` per power iteration rather than a full ``O(mn^2)``
    factorization.  ``oversampling`` extra columns make the probability of
    missing part of the dominant subspace negligible; they are discarded at the
    end.
    """
    A = as_matrix(A)
    k = min(k, min(A.shape))
    Q = randomized_range_finder(A, min(k + oversampling, min(A.shape)),
                                power_iterations, rng)
    B = Q.T @ A                       # small: (k+p) by n
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], s[:k], Vt[:k]


def randomized_eigh(A, k: int, oversampling: int = 10, power_iterations: int = 2,
                    rng=None):
    """Approximate dominant eigenpairs of a symmetric matrix by random projection."""
    A = check_square(A)
    Q = randomized_range_finder(A, min(k + oversampling, A.shape[0]),
                                power_iterations, rng)
    T = Q.T @ A @ Q
    w, V = np.linalg.eigh(0.5 * (T + T.T))
    order = np.argsort(np.abs(w))[::-1][:k]
    return EigenResult(w[order], (Q @ V)[:, order], 1, True, "randomized_eigh")


def nystrom_approximation(A, k: int, rng=None):
    """Nystrom low-rank approximation of a symmetric positive semidefinite matrix.

    Cheaper than :func:`randomized_svd` -- one matrix product, no power
    iterations -- and, unlike a general low-rank truncation, the result stays
    positive semidefinite, which matters when it is used as a kernel or a
    preconditioner.
    """
    A = check_square(A)
    rng = np.random.default_rng(rng)
    n = A.shape[0]
    Omega = rng.standard_normal((n, min(k, n)))
    Q, _ = householder_qr(Omega, reduced=True)
    C = A @ Q
    W = Q.T @ C
    W = 0.5 * (W + W.T)
    # Pseudo-inverse square root keeps the result PSD even if W is singular.
    w, V = np.linalg.eigh(W)
    w = np.maximum(w, 0.0)
    inv_sqrt = np.zeros_like(w)
    nz = w > 1e-14 * max(w.max(), 1e-300)
    inv_sqrt[nz] = 1.0 / np.sqrt(w[nz])
    F = C @ (V * inv_sqrt) @ V.T
    return F @ F.T


def interpolative_decomposition(A, k: int):
    """Interpolative decomposition ``A ~ A[:, cols] @ Z``.

    Picks ``k`` actual columns of ``A`` by pivoted QR, so the basis vectors are
    real columns of the data rather than abstract singular vectors -- the point
    of the decomposition when interpretability matters.  Returns ``(cols, Z)``.
    """
    A = as_matrix(A)
    m, n = A.shape
    k = min(k, min(m, n))
    # Column-pivoted QR by greedy selection on the residual column norms.
    R = A.astype(float).copy()
    cols = []
    Qb = np.zeros((m, 0))
    for _ in range(k):
        norms = np.linalg.norm(R, axis=0)
        j = int(np.argmax(norms))
        if norms[j] <= 1e-300:
            break
        cols.append(j)
        q = R[:, j] / norms[j]
        Qb = np.hstack([Qb, q[:, None]])
        R = R - np.outer(q, q @ R)
    cols = np.array(cols, dtype=int)
    Z, *_ = np.linalg.lstsq(A[:, cols], A, rcond=None)
    return cols, Z


def cur_decomposition(A, k: int, rng=None):
    """CUR decomposition ``A ~ C U R`` from actual columns and rows.

    Both factors are sub-matrices of ``A``, so sparsity and non-negativity
    survive the approximation -- neither of which an SVD preserves.  Returns
    ``(cols, rows, U)``.
    """
    A = as_matrix(A)
    cols, _ = interpolative_decomposition(A, k)
    rows, _ = interpolative_decomposition(A.T, k)
    C, R = A[:, cols], A[rows, :]
    U = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
    return cols, rows, U


# --------------------------------------------------------------------------
# Subspace eigensolvers
# --------------------------------------------------------------------------
def subspace_iteration(A, k: int = 1, tol: float = 1e-10, max_iter: int = 1000,
                       V0=None, rng=None):
    """Orthogonal (simultaneous) iteration for the ``k`` dominant eigenpairs.

    The block generalization of the power method: re-orthonormalizing the whole
    block each step is what stops every column collapsing onto the same
    dominant eigenvector.
    """
    A = check_square(A)
    n = A.shape[0]
    rng = np.random.default_rng(rng)
    V = rng.standard_normal((n, k)) if V0 is None else as_matrix(V0)
    V, _ = householder_qr(V, reduced=True)
    lam = np.zeros(k)
    sym = is_symmetric(A)
    for it in range(1, max_iter + 1):
        V, _ = householder_qr(A @ V, reduced=True)
        T = V.T @ A @ V
        if sym:                        # rotate onto the Ritz basis
            lam_new, S = np.linalg.eigh(0.5 * (T + T.T))
            order = np.argsort(np.abs(lam_new))[::-1]
            lam_new, U = lam_new[order], (V @ S)[:, order]
        else:
            lam_new, U = np.diag(T).copy(), V
        # Converge on the true residual ||A u - lambda u||.  Testing the
        # *change* in lambda instead understates the error by a factor
        # 1/(1 - lam_{k+1}/lam_k), which for a small spectral gap means
        # declaring convergence several digits early.
        resid = float(np.max(np.linalg.norm(A @ U - U * lam_new, axis=0)))
        if resid < tol * max(1.0, float(np.max(np.abs(lam_new)))):
            return EigenResult(lam_new, U, it, True, "subspace_iteration")
        lam = lam_new
        V = U
    return EigenResult(lam, V, max_iter, False, "subspace_iteration")


def lobpcg(A, k: int = 1, B=None, X0=None, tol: float = 1e-10,
           max_iter: int = 500, precond=None, rng=None):
    """Locally optimal block preconditioned conjugate gradient.

    Finds the ``k`` *smallest* eigenvalues of a symmetric (generalized) problem
    ``A x = lambda B x``.  Each step minimizes the Rayleigh quotient over the
    span of the current block, its residual and the previous step -- the last
    of these is what supplies the conjugate-gradient acceleration and makes it
    much faster than plain inverse iteration without needing any factorization.
    """
    A = check_square(A)
    n = A.shape[0]
    rng = np.random.default_rng(rng)
    X = rng.standard_normal((n, k)) if X0 is None else as_matrix(X0)
    Bmul = (lambda V: V) if B is None else (lambda V: np.asarray(B) @ V)
    X = _b_orthonormalize(X, Bmul)
    P = np.zeros((n, 0))
    lam = np.full(k, np.inf)
    # Jacobi preconditioning by default: unpreconditioned LOBPCG converges at
    # the rate of the condition number, exactly like unpreconditioned CG.
    diag = np.diag(A).astype(float)
    inv_diag = np.where(np.abs(diag) > 1e-300, 1.0 / np.where(diag != 0, diag, 1.0), 1.0)
    inv_diag = np.abs(inv_diag)[:, None]
    for it in range(1, max_iter + 1):
        AX = A @ X
        T = X.T @ AX
        w, S = np.linalg.eigh(0.5 * (T + T.T))
        X = X @ S
        AX = AX @ S
        lam_new = w[:k]
        R = AX - Bmul(X) * lam_new
        resid = float(np.max(np.linalg.norm(R, axis=0)))
        if resid < tol * max(1.0, float(np.max(np.abs(lam_new)))):
            return EigenResult(lam_new, X, it, True, "lobpcg")
        W = precond(R) if precond is not None else R * inv_diag
        # Deflate X out of the search directions *before* orthonormalizing.
        # Left in, they make the Gram matrix of [X, W, P] nearly singular and
        # the Rayleigh-Ritz step then stalls well short of convergence -- the
        # classic LOBPCG breakdown.
        W = _b_orthonormalize(W - X @ (X.T @ Bmul(W)), Bmul)
        blocks = [X, W]
        if P.size:
            Pd = _b_orthonormalize(P - X @ (X.T @ Bmul(P)), Bmul)
            if Pd.size:
                blocks.append(Pd)
        S_basis = np.hstack(blocks)
        if S_basis.shape[1] < k:
            break
        Ts = S_basis.T @ (A @ S_basis)
        Gs = S_basis.T @ Bmul(S_basis)
        w, C = _reduced_eigh(0.5 * (Ts + Ts.T), 0.5 * (Gs + Gs.T))
        C = C[:, :k]
        Xn = S_basis @ C
        # The "locally optimal" direction: the part of the update that lies
        # outside the current block, which is what supplies the CG memory.
        P = Xn - X @ (X.T @ Bmul(Xn))
        X = _b_orthonormalize(Xn, Bmul)
        lam = lam_new
    return EigenResult(lam, X, max_iter, False, "lobpcg")


def _reduced_eigh(T, G):
    """Solve the small generalized problem ``T c = theta G c`` safely.

    The basis Gram matrix ``G`` can be near-singular, so drop its numerically
    null directions before reducing rather than inverting through it.
    """
    w, S = np.linalg.eigh(G)
    keep = w > 1e-12 * max(float(np.max(w)), 1e-300)
    Sk = S[:, keep] / np.sqrt(w[keep])
    theta, C = np.linalg.eigh(Sk.T @ T @ Sk)
    return theta, Sk @ C


def _b_orthonormalize(V, Bmul):
    """B-orthonormalize the columns of ``V``, dropping dependent ones."""
    V = np.asarray(V, dtype=float)
    G = V.T @ Bmul(V)
    w, S = np.linalg.eigh(0.5 * (G + G.T))
    keep = w > 1e-12 * max(float(np.max(w)), 1e-300)
    if not np.any(keep):
        return V[:, :0]
    return V @ (S[:, keep] / np.sqrt(w[keep]))


def generalized_eigh(A, B, tol: float = 1e-12):
    """Symmetric-definite generalized eigenproblem ``A x = lambda B x``.

    Reduces to a standard problem through the Cholesky factor of ``B``:
    ``A x = lambda B x`` becomes ``(L^-1 A L^-T) y = lambda y`` with ``y = L' x``.
    Forming ``B^-1 A`` instead would destroy the symmetry and with it the
    guarantee of real eigenvalues.  ``B`` must be positive definite.
    """
    A = check_square(A)
    B = check_square(B)
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError as exc:
        raise SingularMatrixError(
            "generalized_eigh requires B positive definite"
        ) from exc
    Linv = np.linalg.inv(L)
    C = Linv @ A @ Linv.T
    w, Y = np.linalg.eigh(0.5 * (C + C.T))
    X = Linv.T @ Y
    return EigenResult(w, X, 1, True, "generalized_eigh")


def qz_decomposition(A, B, tol: float = 1e-12, max_iter: int = 2000):
    """Generalized Schur (QZ) form of the pencil ``A - lambda B``.

    Returns ``(Q, Z, S, T)`` with ``Q' A Z = S`` and ``Q' B Z = T``, both
    upper triangular (quasi-triangular in the real case).  The generalized
    eigenvalues are the ratios ``S_ii / T_ii``; an infinite eigenvalue shows
    up as ``T_ii = 0``, which is exactly the case a plain ``B^-1 A`` reduction
    cannot represent.
    """
    A = check_square(A)
    B = check_square(B)
    if np.linalg.matrix_rank(B) < B.shape[0]:
        raise SingularMatrixError(
            "qz_decomposition requires a nonsingular B; a singular pencil has "
            "infinite eigenvalues that need the implicit QZ deflation"
        )
    # Right transform first: Schur-decompose B^-1 A, giving A Z = B Z S_m.
    Z, Sm = schur(np.linalg.solve(B, A))
    # Then the left transform is whatever triangularizes B Z.
    Q, T = householder_qr(B @ Z, reduced=False)
    # Q' A Z = Q' B Z S_m = T S_m, upper times quasi-upper, hence quasi-upper.
    return Q, Z, T @ Sm, T


def qz_eigenvalues(S, T, tol: float = 1e-12):
    """Generalized eigenvalues from a QZ pair ``(S, T)``.

    Reading ``diag(S) / diag(T)`` is correct only where ``S`` is genuinely
    triangular.  A real pencil with a complex-conjugate pair leaves a 2-by-2
    block, and there the eigenvalues come from ``det(S_blk - lambda T_blk) = 0``
    instead.  A zero diagonal entry of ``T`` is an infinite eigenvalue and is
    returned as ``inf``.

    The 2-by-2 blocks are identified by a *relative* test.  ``S`` is formed as
    a matrix product, so its subdiagonal carries round-off of order 1e-17 where
    the exact value is zero; an exact ``!= 0`` test reads every one of those as
    a spurious complex pair and returns nonsense.
    """
    S = check_square(S)
    T = check_square(T)
    n = S.shape[0]
    vals = np.zeros(n, dtype=complex)
    scale = float(np.max(np.abs(S))) if n else 1.0
    i = 0
    while i < n:
        if i + 1 < n and abs(S[i + 1, i]) > tol * max(
                abs(S[i, i]) + abs(S[i + 1, i + 1]), scale):
            a, b = S[i:i + 2, i:i + 2], T[i:i + 2, i:i + 2]
            # Roots of det(a - lambda b) = 0, a quadratic in lambda.
            q2 = b[0, 0] * b[1, 1] - b[0, 1] * b[1, 0]
            q1 = -(a[0, 0] * b[1, 1] + b[0, 0] * a[1, 1]
                   - a[0, 1] * b[1, 0] - b[0, 1] * a[1, 0])
            q0 = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
            if abs(q2) < 1e-300:
                vals[i] = np.inf if abs(q1) < 1e-300 else -q0 / q1
                vals[i + 1] = np.inf
            else:
                disc = np.sqrt(complex(q1 * q1 - 4 * q2 * q0))
                vals[i] = (-q1 + disc) / (2 * q2)
                vals[i + 1] = (-q1 - disc) / (2 * q2)
            i += 2
        else:
            vals[i] = np.inf if T[i, i] == 0.0 else S[i, i] / T[i, i]
            i += 1
    return vals
