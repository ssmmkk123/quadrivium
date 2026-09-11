# Linear algebra

Linear algebra begins with a representation and a question: solve a square
system, fit an overdetermined model, extract eigenpairs, or apply a matrix
function. The best algorithm depends on matrix structure, conditioning, size,
and whether entries are explicitly available.

Quadrivium exposes both named algorithms for study and practical interfaces
for repeated solves, sparse storage, and matrix-free iteration. This guide
shows how to choose among them and how to verify the result. Full signatures
are in the [linear algebra reference](../api/linalg.md).

```pycon
>>> import quadrivium as qd
>>> from quadrivium import numeric as np
>>> from quadrivium import linalg as la
>>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
>>> b = np.array([1.0, 2.0])
>>> x = la.solve(A, b)
>>> np.allclose(A @ x, b)
True

```

## Select a method from the problem

| Problem | Starting point | Main requirement or tradeoff |
| --- | --- | --- |
| Small or moderate square dense system | `solve` | Finite nonsingular matrix |
| Same matrix, many solves | `lu_factor`, `cholesky_factor` | Reuse the factor object |
| Symmetric positive definite system | Cholesky or conjugate gradient | Positive definiteness is essential |
| Symmetric indefinite system | Pivoted LU or `minres` | Unpivoted LDL can encounter zero pivots |
| Tridiagonal system | `thomas` | Pass three diagonal vectors; no pivoting |
| Overdetermined full-rank fit | `qr_least_squares`, `qr_factor` | Avoids forming normal equations |
| Rank-deficient or underdetermined fit | `svd_least_squares` | Singular-value cutoff defines effective rank |
| Sparse nonsymmetric system | `gmres`, `bicgstab` | Preconditioning usually matters |
| Matrix available only through products | `LinearOperator` and a Krylov solver | Supply the products the algorithm needs |
| Few dominant singular values | `randomized_svd` | Approximate answer; seed and rank matter |

Dense storage grows as the square of matrix size and a general dense
factorization grows roughly cubically in work. Sparsity only saves memory if
you retain sparse storage; constructing a dense matrix and converting it later
has already paid the dense allocation cost.

## What `solve(method="auto")` actually does

For a real two-dimensional matrix and one right-hand-side vector, `solve`
requires a square matrix and follows this order:

1. An exactly tridiagonal matrix of size greater than two uses Thomas elimination.
2. A matrix passing the symmetry test is tried with Cholesky.
3. If Cholesky is unsuitable, or the matrix is nonsymmetric, pivoted LU is used.

It does not send rectangular systems to least squares. Select a least-squares
routine explicitly. Valid explicit choices are `"lu"`, `"plu"`, `"gauss"`,
`"gauss_jordan"`, `"cholesky"`, `"ldl"`, and `"qr"`.
Call `thomas` directly to select that algorithm explicitly.

Complex matrices, batches, and multiple right-hand sides take the array layer's
dense path. That path supports `"auto"`, `"lu"`, `"plu"`, `"cholesky"`, and
`"qr"`, with square trailing matrix dimensions. It does not apply the same
real single-vector dispatch logic.

A tridiagonal matrix can be nonsingular and still have an unsuitable pivot for
Thomas elimination. If automatic selection encounters such a case, use
`method="plu"`. Structure does not remove the need for a numerically safe
factorization.

## Verify a solve with a scaled residual

For a computed vector `x`, form `r = A @ x - b`. An absolute residual is useful
only in relation to the input scale. A normwise scaled residual divides by
`||A|| ||x|| + ||b||`.

```pycon
>>> residual = A @ x - b
>>> scale = np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(b)
>>> float(np.linalg.norm(residual) / scale) < 1e-14
True

```

This check asks whether a small perturbation to the data could explain the
computed answer. It does not measure forward error directly. An ill-conditioned
matrix can amplify small perturbations substantially, so compare against a
known solution when validating an algorithm and inspect conditioning when
interpreting a real dataset.

<figure markdown="span">
  ![A small linear-system residual can coexist with a large solution error as conditioning worsens](../assets/figures/linalg-residual-sensitivity.svg#only-light)
  ![A small linear-system residual can coexist with a large solution error as conditioning worsens](../assets/figures/linalg-residual-sensitivity-dark.svg#only-dark)
  <figcaption>For A=diag(1,1/κ) with exact x=(1,1), add 10⁻¹⁰ to the second right-hand-side component and solve. The residual against the original right-hand side stays small while the solution error grows with κ. This demonstrates input sensitivity, not a failure of the linear solver.</figcaption>
</figure>

Use `core.condition_number` for small real square matrices or
`condition_estimate` for a 1-norm estimate. The latter still performs dense
linear algebra in this implementation; it is not a matrix-free estimator.

## Factorizations and repeated right-hand sides

`plu_decomposition(A)` returns `(P, L, U)` with `P @ A = L @ U`.
`cholesky(A)` returns a lower-triangular factor by default, with
`A = L @ L.T` for real data. Verify factorization identities with tolerances,
not exact equality.

```pycon
>>> P, L, U = la.plu_decomposition(A)
>>> np.allclose(P @ A, L @ U)
True
>>> C = la.cholesky(A)
>>> np.allclose(C @ C.T, A)
True

```

`lu_decomposition` and classical Doolittle/Crout algorithms expose unpivoted
factorization variants. Partial pivoting is the usual general-purpose choice;
`gauss_elimination` also offers `"none"`, `"scaled"`, and `"complete"` strategies.
Complete pivoting additionally permutes columns, so reconstruction and solution
reordering must use the returned column information.

For repeated work, use a factor object. Its construction snapshots and factors
the matrix, and `.solve` applies those factors to new right-hand sides.
Right-hand sides have shape `(n,)` or `(n, nrhs)`.

```pycon
>>> factor = la.lu_factor(A)
>>> B = np.array([[1.0, 0.0], [2.0, 1.0]])
>>> X = factor.solve(B)
>>> X.shape, bool(np.allclose(A @ X, B))
((2, 2), True)
>>> output = np.empty_like(B)
>>> factor.solve(B, out=output) is output
True

```

`LUFactor.solve(trans="N")` solves the original system; `"T"` and `"H"`
select transpose and conjugate-transpose systems. `CholeskyFactor` uses a
Hermitian positive definite factorization. `QRFactor` accepts rectangular
matrices with at least as many rows as columns and requires full column rank.
The objects support real or complex data but do not represent a batch of
independent factorizations.

## Least squares and regularization

Least squares solves `min ||A @ x - b||_2`. For an inconsistent system, a
nonzero residual is expected. Check that the residual is orthogonal to the
columns of `A`, and evaluate predictions on data that did not determine the fit.

```pycon
>>> design = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
>>> observations = np.array([1.0, 3.1, 4.9, 7.0])
>>> coefficients = la.qr_least_squares(design, observations)
>>> fit_residual = design @ coefficients - observations
>>> np.allclose(design.T @ fit_residual, [0.0, 0.0], atol=1e-12)
True

```

Normal equations form `A.T @ A`, which squares the 2-norm condition number.
They are useful for understanding the derivation but can lose substantially
more accuracy than QR. Householder QR is a practical dense default;
classical and modified Gram-Schmidt and Givens QR expose alternative
orthogonalization procedures.

`svd_least_squares(A, b, rcond=...)` discards singular values at or below a
cutoff relative to the largest singular value and returns a minimum-norm
solution for that effective rank. This threshold is a modeling decision when
small singular directions contain noise.

```pycon
>>> redundant = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
>>> minimum_norm = la.svd_least_squares(redundant, [2.0, 4.0, 6.0])
>>> np.allclose(minimum_norm, [1.0, 1.0])
True

```

`ridge_regression` penalizes coefficient magnitude; `tikhonov` allows a penalty
matrix; `truncated_svd` retains a chosen number of singular directions.
`weighted_least_squares` multiplies rows by square-root weights, so its objective
is `sum(weights * residual**2)`. Use nonnegative finite weights and meaningful
variable scaling. Other interfaces cover equality constraints, nonnegative
coefficients, and total least squares. See [approximation](approx.md) for the
statistical and modeling implications of fitting choices.

## Iterative solvers and preconditioners

Krylov methods return `IterationResult`, carrying `x`, `converged`, `iterations`,
`residuals`, and `message`. CG is appropriate for real symmetric positive
definite matrices. MINRES handles symmetric indefinite systems. GMRES handles
general systems by building an orthogonal basis; `restart` limits that basis
at a possible cost in convergence. BiCGSTAB uses less storage but can exhibit
irregular residual histories.

```pycon
>>> iterative = la.conjugate_gradient(A, b, tol=1e-12)
>>> iterative.converged and np.allclose(A @ iterative.x, b)
True
>>> iterative.residual < 1e-12
True

```

For CG, the stored residual is the Euclidean residual divided by `||b||_2`,
using one when `b` is zero. Recurrence-based residuals can drift from the true
residual because of rounding, so recompute `A @ x - b` before accepting an
important result. A small iteration count is useful only if the requested
accuracy and per-iteration cost are also comparable.

A preconditioner approximates a solve with a matrix that improves the
iteration's effective conditioning. In `preconditioned_cg`, a callable `M(v)`
returns the action of the **inverse** preconditioner. A matrix supplied as `M`
is solved against. The default uses a diagonal approximation when available.

```pycon
>>> inverse_diagonal = lambda v: v / np.diag(A)
>>> preconditioned = la.preconditioned_cg(A, b, M=inverse_diagonal, tol=1e-12)
>>> preconditioned.converged and np.allclose(A @ preconditioned.x, b)
True

```

CG requires a compatible positive definite preconditioner. A cheap
preconditioner is not automatically a good one: include setup time, application
cost, and memory in any comparison. Stationary Jacobi, Gauss-Seidel, SOR, and
SSOR are useful teaching methods and smoothers, but do not converge for every
matrix. Relaxation parameters must fit the problem.

## Sparse storage and matrix-free operators

`COOMatrix`, `CSRMatrix`, `CSCMatrix`, and `DIAMatrix` expose coordinate,
compressed-row, compressed-column, and diagonal storage. Use COO for assembly,
CSR for repeated row-oriented products, and diagonal construction when the
operator is naturally specified by bands.

```pycon
>>> sparse = la.from_dense(A, fmt="csr")
>>> np.allclose(sparse @ b, A @ b)
True
>>> sparse_result = la.sparse_solve(sparse, b, method="cg", tol=1e-12)
>>> sparse_result.converged
True

```

Sparse products, triangular solves, ordering helpers, and incomplete
factorizations are available. Sparse storage does not imply sparse direct
factorization for every method; read the selected routine's contract before
using it on a problem too large to densify.

A `LinearOperator` represents products without storing matrix entries. Declare
its shape and a `matvec`; algorithms needing transpose products also require
`rmatvec`, which applies the conjugate transpose. An optional `matmat` handles
multiple vectors efficiently.

```pycon
>>> operator = la.LinearOperator(A.shape, matvec=lambda v: A @ v,
...                              rmatvec=lambda v: A.T @ v)
>>> implicit = la.conjugate_gradient(operator, b, tol=1e-12)
>>> implicit.converged and np.allclose(implicit.x, x)
True

```

A matrix-free interface avoids materializing entries, but Krylov basis vectors
and retained histories still consume memory. See [workflows](workflows.md) for
callbacks and history controls on supported solvers.

## Eigenvalues, singular values, and matrix functions

Eigenvectors are defined up to sign or complex phase, and repeated eigenvalues
allow different bases of the same invariant subspace. Verify an eigenpair with
`A @ v - lambda*v`; do not compare eigenvector entries directly across methods.

```pycon
>>> eig = la.jacobi_eigen(A)
>>> values, vectors = eig
>>> np.allclose(A @ vectors, vectors * values, atol=1e-10)
True

```

Power iteration seeks a dominant eigenpair, inverse iteration targets an
eigenvalue near a shift, and Rayleigh quotient iteration refines a local guess.
Jacobi and symmetric eigensolvers exploit symmetry; Schur-based methods handle
general spectra. `lanczos` and `arnoldi` return reduced factorizations rather
than universally returning an `EigenResult`; inspect their documented outputs.

SVD represents `A = U @ diag(s) @ Vh`. It supports rank analysis, low-rank
compression, least squares, and polar decomposition. Randomized SVD trades an
approximate subspace for lower work; set `rng`, choose oversampling, and check
`||A - approximation||` or an independent product-based error estimate.

Matrix functions such as `matrix_exponential`, `sqrtm`, and `logm` apply a
function to a matrix, not elementwise to its entries. They have spectral-domain
restrictions and branch choices. Matrix-equation routines include Sylvester,
Lyapunov, and Riccati solvers. Always verify the equation and sign convention
stated in the API; the same name in another library may use a different sign.
For time evolution via matrix exponentials, continue with [ODE methods](ode.md).
