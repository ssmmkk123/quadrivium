"""Figures for the linear algebra guide."""

from __future__ import annotations

from quadrivium import numeric as np
from matplotlib.patches import Circle

from quadrivium.core import SingularMatrixError
from quadrivium.linalg import (back_substitution, bandwidth, conjugate_gradient,
                               forward_substitution, francis_qr, from_dense,
                               gauss_seidel, gershgorin_disks, givens_qr,
                               gmres, gram_schmidt_qr, householder_qr,
                               incomplete_cholesky, jacobi_preconditioner, kron,
                               modified_gram_schmidt_qr, normal_equations,
                               optimal_sor_omega, preconditioned_cg,
                               qr_least_squares, randomized_svd,
                               reverse_cuthill_mckee, sor, svd_jacobi,
                               svd_least_squares)

from . import figure
from figstyle import annotate, finish

GUIDE = "guides/linalg.md"


def poisson_2d(m: int) -> np.ndarray:
    """The 5-point Laplacian on an m x m grid, built with the library's `kron`."""
    tri = (np.diag(2.0 * np.ones(m)) + np.diag(-np.ones(m - 1), 1)
           + np.diag(-np.ones(m - 1), -1))
    eye = np.eye(m)
    return kron(eye, tri) + kron(tri, eye)


@figure("linalg-qr-orthogonality", GUIDE,
        "Loss of orthogonality in four QR algorithms as the matrix becomes "
        "nearly rank deficient", size=(7.0, 3.8))
def qr_orthogonality(fig, scheme):
    # The Lauchli matrix: three columns agreeing in their first row and
    # differing by eps everywhere else, so its condition number is about 1/eps.
    algorithms = [("classical Gram-Schmidt", gram_schmidt_qr),
                  ("modified Gram-Schmidt", modified_gram_schmidt_qr),
                  ("Householder", householder_qr),
                  ("Givens", givens_qr)]
    eps = np.logspace(-1, -8, 22)
    conds, losses = [], {name: [] for name, _ in algorithms}
    for e in eps:
        A = np.vstack([np.ones(3), e * np.eye(3)])
        # `condition_number` is square-only; for a tall matrix the ratio of the
        # extreme singular values is the same definition.
        singular = svd_jacobi(A)[1]
        conds.append(float(singular[0] / singular[-1]))
        for name, qr in algorithms:
            Q, _ = qr(A)
            k = Q.shape[1]
            losses[name].append(float(np.max(np.abs(Q.T @ Q - np.eye(k)))))

    ax = fig.add_subplot()
    floor = float(np.finfo(float).eps)
    for i, (name, _) in enumerate(algorithms):
        ax.loglog(conds, np.maximum(losses[name], floor / 4), label=name,
                  color=scheme.series[i],
                  marker="o" if i < 2 else None, markersize=3.5)
    ax.axhline(floor, color=scheme.muted, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_ylim(floor / 8, 20)
    annotate(ax, "machine epsilon", (conds[-1], floor), (conds[-1], floor / 5),
             scheme, arrow=False, ha="right")
    finish(ax, scheme,
           title="Orthogonality is not what QR computes — it is what QR keeps",
           xlabel="condition number of A", ylabel=r"$\max\,|Q^{\!\top}Q - I|$",
           grid="both", legend=True, legend_kw=dict(loc="upper left", ncol=2))


@figure("linalg-krylov-convergence", GUIDE,
        "Residual histories of Krylov and stationary iterations on a 2-D "
        "Poisson matrix", size=(7.0, 3.6))
def krylov_convergence(fig, scheme):
    A = poisson_2d(16)
    b = np.ones(A.shape[0])
    omega = float(optimal_sor_omega(A))
    runs = [("conjugate gradient", conjugate_gradient(A, b, tol=1e-12, max_iter=400)),
            ("GMRES", gmres(A, b, tol=1e-12, max_iter=400)),
            (f"SOR, $\\omega$ = {omega:.2f}", sor(A, b, omega=omega, tol=1e-12,
                                                 max_iter=9000)),
            ("Gauss-Seidel", gauss_seidel(A, b, tol=1e-12, max_iter=9000))]

    ax = fig.add_subplot()
    for i, (name, res) in enumerate(runs):
        ax.semilogy(np.arange(len(res.residuals)), res.residuals,
                    label=f"{name} — {res.iterations} iterations",
                    color=scheme.series[i])
    finish(ax, scheme,
           title=f"Krylov methods finish; stationary ones grind "
                 f"({A.shape[0]} unknowns)",
           xlabel="iteration", ylabel="relative residual", grid="both",
           legend=True, legend_kw=dict(loc="lower left"))
    ax.set_xlim(0, 260)
    ax.set_ylim(1e-13, 5)
    annotate(ax, "Gauss-Seidel needs "
                 f"{runs[3][1].iterations:,} — off the right of the plot",
             (255, 4e-4), (250, 3e-2), scheme, ha="right")


@figure("linalg-preconditioning", GUIDE,
        "Preconditioned conjugate gradient against plain CG on a badly scaled "
        "Poisson matrix", size=(7.0, 3.6))
def preconditioning(fig, scheme):
    # Poisson, then a diagonal rescaling spanning four orders of magnitude:
    # still symmetric positive definite, but with a condition number that comes
    # from the scaling rather than from the mesh -- exactly what a diagonal
    # preconditioner is for.
    K = poisson_2d(20)
    scale = np.logspace(0, 2, K.shape[0])
    A = (scale[:, None] * K) * scale[None, :]
    b = np.ones(A.shape[0])
    L = incomplete_cholesky(A)

    def ic_apply(r):
        return back_substitution(L.T, forward_substitution(L, r))

    runs = [("no preconditioner", conjugate_gradient(A, b, tol=1e-12, max_iter=4000)),
            ("Jacobi (diagonal)", preconditioned_cg(A, b, M=jacobi_preconditioner(A),
                                                    tol=1e-12, max_iter=4000)),
            ("incomplete Cholesky", preconditioned_cg(A, b, M=ic_apply, tol=1e-12,
                                                      max_iter=4000))]
    ax = fig.add_subplot()
    for i, (name, res) in enumerate(runs):
        ax.semilogy(np.arange(len(res.residuals)), res.residuals,
                    label=f"{name} — {res.iterations:,} iterations",
                    color=scheme.series[i])
    finish(ax, scheme,
           title="A preconditioner buys iterations, not accuracy",
           xlabel="iteration", ylabel="relative residual", grid="both",
           legend=True, legend_kw=dict(loc="lower left"))
    ax.set_xlim(0, 140)
    ax.set_ylim(1e-13, 5)
    annotate(ax, "plain CG carries on to "
                 f"{runs[0][1].iterations:,} iterations", (138, 4e-2), (136, 1.2),
             scheme, ha="right")


@figure("linalg-sparsity-rcm", GUIDE,
        "Sparsity pattern of a symmetric matrix before and after reverse "
        "Cuthill-McKee reordering", size=(7.0, 3.8))
def sparsity_rcm(fig, scheme):
    rng = np.random.default_rng(4)
    n = 120
    A = np.eye(n) * 4.0
    # A sparse symmetric matrix whose bandwidth is accidental: the entries were
    # placed at random, so nothing about the ordering reflects the structure.
    for _ in range(220):
        i, j = rng.integers(0, n, 2)
        if i != j:
            A[i, j] = A[j, i] = -1.0
    perm = reverse_cuthill_mckee(from_dense(A, fmt="csr"))
    B = A[np.ix_(perm, perm)]

    for k, (M, label) in enumerate([(A, "as given"), (B, "after RCM")]):
        ax = fig.add_subplot(1, 2, k + 1)
        rows, cols = np.nonzero(M)
        ax.scatter(cols, rows, s=1.6, color=scheme.series[0], linewidths=0,
                   rasterized=True)
        kl, ku = bandwidth(from_dense(M, fmt="csr"))
        ax.set_title(f"{label} — bandwidth {max(kl, ku)}", color=scheme.ink,
                     fontsize=9.5)
        ax.set_xlim(-2, n + 1)
        ax.set_ylim(n + 1, -2)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.set_ylabel("row")
        ax.set_xlabel("column")
    fig.suptitle("Reordering changes no entry of the matrix, only the bandwidth",
                 x=0.02, ha="left", fontsize=10, fontweight="bold",
                 color=scheme.ink)
    fig.subplots_adjust(top=0.84)


@figure("linalg-gershgorin", GUIDE,
        "Gershgorin disks bounding the eigenvalues of a nonsymmetric matrix",
        size=(7.0, 3.6))
def gershgorin(fig, scheme):
    A = np.array([[6.0, 2.0, 0.5, 0.0],
                  [-2.0, 6.0, 0.4, 0.3],
                  [0.5, 0.2, -3.0, 0.9],
                  [0.1, 0.4, 0.7, 10.0]])
    centres, radii = gershgorin_disks(A)
    values = francis_qr(A).eigenvalues

    ax = fig.add_subplot()
    for centre, radius in zip(centres, radii):
        ax.add_patch(Circle((float(centre), 0.0), float(radius),
                            facecolor=scheme.series[0], alpha=0.12,
                            edgecolor=scheme.series[0], linewidth=1.2))
    ax.scatter(centres, np.zeros_like(centres), s=26, zorder=3,
               color=scheme.series[0], marker="x", linewidths=1.6,
               label=r"disk centres $a_{ii}$")
    ax.scatter(np.real(values), np.imag(values), s=44, zorder=4,
               color=scheme.series[1], edgecolors=scheme.surface, linewidths=1.5,
               label="eigenvalues (Francis QR)")
    ax.axhline(0, color=scheme.axis, linewidth=0.8)
    finish(ax, scheme,
           title="Every eigenvalue lies in one of the disks — no solve required",
           xlabel="real part", ylabel="imaginary part", grid="both", legend=True,
           legend_kw=dict(loc="lower left"))
    ax.set_aspect("equal")
    ax.set_ylim(-4.6, 3.6)
    annotate(ax, "a complex pair, bounded\nby a radius from real arithmetic",
             (6.0, 2.0), (6.0, 3.2), scheme, ha="center", va="bottom")


@figure("linalg-randomized-svd", GUIDE,
        "Randomized SVD against the exact spectrum and the Eckart-Young "
        "optimum", size=(7.0, 3.4))
def randomized(fig, scheme):
    rng = np.random.default_rng(0)
    n, m, r = 200, 160, 40
    # A fast but not abrupt spectral decay: the regime a randomized method is
    # designed for, and the one where the Eckart-Young bound is worth checking.
    U = householder_qr(rng.standard_normal((n, r)))[0]
    V = householder_qr(rng.standard_normal((m, r)))[0]
    spectrum = 10.0 ** (-0.18 * np.arange(r))
    A = U @ np.diag(spectrum) @ V.T
    exact = svd_jacobi(A)[1]

    ax1 = fig.add_subplot(1, 2, 1)
    ranks = np.arange(1, 21)
    _, sampled, _ = randomized_svd(A, k=20, rng=0)
    ax1.semilogy(np.arange(1, r + 1), exact[:r], color=scheme.series[0],
                 label="exact (Jacobi SVD)")
    ax1.semilogy(ranks, sampled[:20], linestyle="none", marker="o", markersize=4,
                 color=scheme.series[1], markeredgecolor=scheme.surface,
                 markeredgewidth=0.8, label="randomized, k = 20")
    finish(ax1, scheme, title="Singular values", xlabel="index",
           ylabel="singular value", grid="both", legend=True)

    ax2 = fig.add_subplot(1, 2, 2)
    errors, optimum = [], []
    for k in ranks:
        Uk, sk, Vtk = randomized_svd(A, k=int(k), rng=0)
        errors.append(float(np.sqrt(np.sum((A - Uk @ np.diag(sk) @ Vtk) ** 2))))
        # Eckart-Young in the Frobenius norm: no rank-k matrix does better.
        optimum.append(float(np.sqrt(np.sum(exact[k:] ** 2))))
    ax2.semilogy(ranks, errors, color=scheme.series[1], marker="o", markersize=3.5,
                 label="randomized rank k")
    ax2.semilogy(ranks, optimum, color=scheme.series[0], linestyle=(0, (5, 2)),
                 label="best possible rank k")
    finish(ax2, scheme, title="Distance to the best rank-k matrix",
           xlabel="rank k", ylabel=r"$\|A - A_k\|_F$", grid="both", legend=True)
    fig.tight_layout()


@figure("linalg-least-squares-conditioning", GUIDE,
        "Normal equations against QR and SVD least squares as the Vandermonde "
        "system becomes ill-conditioned", size=(7.0, 3.6))
def least_squares_conditioning(fig, scheme):
    t = np.linspace(0, 1, 60)
    degrees = np.arange(2, 17)
    methods = [("normal equations", normal_equations),
               ("QR", qr_least_squares),
               ("SVD", svd_least_squares)]
    errors = {name: [] for name, _ in methods}
    conds, conds_squared = [], []
    for d in degrees:
        V = np.vander(t, d + 1)
        c_true = np.ones(d + 1)
        y = V @ c_true
        singular = svd_jacobi(V)[1]
        conds.append(float(singular[0] / singular[-1]))
        conds_squared.append(float((singular[0] / singular[-1]) ** 2))
        for name, solver in methods:
            try:
                c = solver(V, y)
            except SingularMatrixError:
                # Past degree 12 the Gram matrix is no longer numerically
                # positive definite and Cholesky refuses it: the normal
                # equations do not degrade quietly, they stop.
                errors[name].append(np.nan)
                continue
            errors[name].append(float(np.max(np.abs(c - c_true))) + 1e-18)

    ax1 = fig.add_subplot(1, 2, 1)
    for i, (name, _) in enumerate(methods):
        ax1.semilogy(degrees, errors[name], color=scheme.series[i], label=name,
                     marker="o", markersize=3)
    ax1.set_ylim(1e-16, 1e4)
    breaks = next((d for d, e in zip(degrees, errors["normal equations"])
                   if np.isnan(e)), None)
    if breaks is not None:
        ax1.axvline(breaks, color=scheme.muted, linewidth=0.9,
                    linestyle=(0, (4, 3)))
        annotate(ax1, "Cholesky refuses\n" r"$V^{\!\top}V$ past here",
                 (breaks, 3e2), (breaks - 0.5, 3e2), scheme, ha="right",
                 va="center")
    finish(ax1, scheme, title="Error in the recovered coefficients",
           xlabel="polynomial degree", ylabel="max coefficient error",
           grid="both", legend=True, legend_kw=dict(loc="lower right"))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.semilogy(degrees, conds_squared, color=scheme.series[0],
                 label=r"$\kappa(V)^2$, what the normal equations see")
    ax2.semilogy(degrees, conds, color=scheme.series[1],
                 label=r"$\kappa(V)$, what QR sees")
    ax2.axhline(1 / np.finfo(float).eps, color=scheme.muted, linewidth=0.9,
                linestyle=(0, (4, 3)))
    annotate(ax2, r"$1/\varepsilon$ — no digits left", (degrees[0], 6e15),
             (degrees[0], 6e15), scheme, arrow=False, ha="left", va="bottom")
    finish(ax2, scheme, title="Why: forming $V^{\\top}V$ squares the conditioning",
           xlabel="polynomial degree", ylabel="condition number", grid="both",
           legend=True, legend_kw=dict(loc="lower right"))
    fig.tight_layout()
