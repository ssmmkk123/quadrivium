"""Linear algebra: factorizations, eigenvalues, and iterative solvers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numethods.core import condition_number
from numethods.linalg import *

np.set_printoptions(precision=6, suppress=True)


def banner(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


banner("1. Factorizations reconstruct the matrix exactly")
rng = np.random.default_rng(0)
A = rng.random((5, 5)) + 5 * np.eye(5)
b = rng.random(5)

P, L, U = plu_decomposition(A)
print(f"  PLU:        ||PA - LU||     = {np.max(np.abs(P @ A - L @ U)):.2e}")
Q, R = householder_qr(A)
print(f"  QR:         ||A - QR||      = {np.max(np.abs(Q @ R - A)):.2e}")
print(f"              ||Q'Q - I||     = {np.max(np.abs(Q.T @ Q - np.eye(5))):.2e}")
S = A @ A.T + np.eye(5)
Lc = cholesky(S)
print(f"  Cholesky:   ||S - LL'||     = {np.max(np.abs(Lc @ Lc.T - S)):.2e}")
H, Qh = hessenberg(A)
print(f"  Hessenberg: ||A - QHQ'||    = {np.max(np.abs(Qh @ H @ Qh.T - A)):.2e}")

banner("2. QR variants differ in orthogonality, not in the product")
ill = np.vander(np.linspace(1, 2, 8), 8)     # badly conditioned by construction
print(f"  condition number of the test matrix: {condition_number(ill):.2e}\n")
for f in (gram_schmidt_qr, modified_gram_schmidt_qr, householder_qr, givens_qr):
    Qv, Rv = f(ill)
    loss = np.max(np.abs(Qv.T @ Qv - np.eye(Qv.shape[1])))
    print(f"  {f.__name__:<26} ||Q'Q - I|| = {loss:.2e}")
print("\n  Classical Gram-Schmidt loses orthogonality; the others hold up.")

banner("3. Eigenvalues by five different routes")
Sym = rng.random((6, 6))
Sym = Sym + Sym.T + 6 * np.eye(6)
reference = np.sort(np.linalg.eigvalsh(Sym))
print(f"  reference (numpy):    {reference}")
print(f"  QR algorithm:         {np.sort(qr_algorithm(Sym).eigenvalues)}")
print(f"  shifted QR:           {shifted_qr_algorithm(Sym).eigenvalues}")
print(f"  Jacobi rotations:     {jacobi_eigen(Sym).eigenvalues}")
alpha, beta, _ = lanczos(Sym)
print(f"  Lanczos + bisection:  {bisection_eigenvalues(alpha, beta)}")
print(f"  power iteration:      {power_iteration(Sym).eigenvalues[0]:.6f} "
      f"(largest only)")

banner("4. Krylov solvers on a sparse Poisson matrix")
n = 120
T = (np.diag(2.0 * np.ones(n)) + np.diag(-np.ones(n - 1), 1)
     + np.diag(-np.ones(n - 1), -1))
rhs = np.ones(n)
exact = np.linalg.solve(T, rhs)
print(f"  {n}x{n} tridiagonal system, condition number {condition_number(T):.1f}\n")
print(f"  {'method':<22}{'iterations':>12}{'error':>14}")
for name, run in (("conjugate_gradient", lambda: conjugate_gradient(T, rhs, tol=1e-10)),
                  ("preconditioned_cg", lambda: preconditioned_cg(T, rhs, tol=1e-10)),
                  ("minres", lambda: minres(T, rhs, tol=1e-10)),
                  ("gmres (full)", lambda: gmres(T, rhs, tol=1e-10, restart=n, max_iter=100000)),
                  ("gmres(30) restarted", lambda: gmres(T, rhs, tol=1e-10, restart=30, max_iter=100000)),
                  ("bicgstab", lambda: bicgstab(T, rhs, tol=1e-10)),
                  ("gauss_seidel", lambda: gauss_seidel(T, rhs, tol=1e-10, max_iter=100000)),
                  ("sor (optimal omega)", lambda: sor(T, rhs, optimal_sor_omega(T), tol=1e-10, max_iter=100000))):
    r = run()
    print(f"  {name:<22}{r.iterations:>12}{np.max(np.abs(r.x - exact)):>14.2e}")
print("\n  CG needs O(sqrt(kappa)) iterations where Gauss-Seidel needs O(kappa),")
print("  and restarting GMRES saves memory at the cost of many more iterations.")

banner("5. Sparse storage")
sp = from_dense(T)
print(f"  dense:  {T.nbytes / 1024:.1f} KiB")
print(f"  CSR:    {sp.nnz} nonzeros, density {sp.density():.4%}")
print(f"  matvec agrees with dense: "
      f"{np.allclose(sp @ rhs, T @ rhs)}")
res = sparse_solve(sp, rhs, tol=1e-10)
print(f"  matrix-free CG on the sparse operator: error "
      f"{np.max(np.abs(res.x - exact)):.2e}")

banner("6. Least squares, four ways")
M = rng.random((30, 4))
y = rng.random(30)
ref = np.linalg.lstsq(M, y, rcond=None)[0]
for f in (normal_equations, qr_least_squares, svd_least_squares, lsqr_least_squares):
    print(f"  {f.__name__:<22} error vs reference = "
          f"{np.max(np.abs(f(M, y) - ref)):.2e}")
print(f"\n  R^2 = {residual_analysis(M, y, ref)['r_squared']:.4f}")
