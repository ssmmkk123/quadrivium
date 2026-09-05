//! Symmetric and general eigenvalue kernels.

/// Tangent of the Jacobi rotation angle, guarded against overflow exactly as
/// the pure-Python `_safe_tangent` does, so both backends pick the same
/// rotation and converge along the same path.
#[inline]
fn safe_tangent(theta: f64) -> f64 {
    if theta == 0.0 {
        return 1.0;
    }
    if theta.abs() > 1e8 {
        return 1.0 / (2.0 * theta);
    }
    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix.
///
/// `d` holds the matrix on entry and the (diagonal) eigenvalues on exit; `v`
/// receives the eigenvectors as columns. Returns the sweep count reached.
///
/// Each rotation touches only rows and columns `p` and `q`, which is what makes
/// this `O(n^3)` per sweep. Forming the rotation as a dense matrix and
/// multiplying it through -- the obvious transcription of the textbook formula
/// -- costs `O(n^5)` per sweep instead.
pub fn jacobi_eigen(
    n: usize,
    d: &mut [f64],
    v: &mut [f64],
    tol: f64,
    max_sweeps: usize,
) -> (usize, bool) {
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    let mut sweep = 0usize;
    let mut converged = false;
    // Each rotation moves mass from the off-diagonal to the diagonal without
    // changing the Frobenius norm, so `off` falls monotonically until rounding
    // stops it near eps*||A||_F. A sweep that fails to reduce it has reached
    // that floor and the decomposition is as converged as double precision
    // allows -- which is the only outcome available whenever `tol` sits below
    // the floor, as it does for any matrix of norm much above one. Testing
    // `off < tol` alone burns every sweep and then reports failure on an exact
    // answer. This mirrors the pure-Python routine step for step.
    let mut prev_off = f64::INFINITY;
    while sweep < max_sweeps {
        sweep += 1;
        // Off-diagonal Frobenius norm, matching the Python convergence test.
        let mut off = 0.0;
        for i in 1..n {
            for j in 0..i {
                let a = d[i * n + j];
                off += a * a;
            }
        }
        off = (2.0 * off).sqrt();
        if off < tol || off >= prev_off {
            converged = true;
            break;
        }
        prev_off = off;
        for p in 0..n.saturating_sub(1) {
            for q in p + 1..n {
                let dpq = d[p * n + q];
                if dpq.abs() < 1e-300 {
                    continue;
                }
                let theta = (d[q * n + q] - d[p * n + p]) / (2.0 * dpq);
                let t = safe_tangent(theta);
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // D := J' D J, touching only columns p,q then rows p,q.
                // A symmetry-exploiting variant that writes both triangles in
                // one pass was tried and measured no faster: the cost here is
                // the strided column access, not the multiply count. This form
                // is kept because it mirrors the Python routine step for step.
                for i in 0..n {
                    let dip = d[i * n + p];
                    let diq = d[i * n + q];
                    d[i * n + p] = c * dip - s * diq;
                    d[i * n + q] = s * dip + c * diq;
                }
                for j in 0..n {
                    let dpj = d[p * n + j];
                    let dqj = d[q * n + j];
                    d[p * n + j] = c * dpj - s * dqj;
                    d[q * n + j] = s * dpj + c * dqj;
                }
                // V := V J
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    (sweep, converged)
}

/// Unshifted QR iteration on an upper Hessenberg matrix, in place.
///
/// `h` is a row-major `n x n` upper Hessenberg matrix, overwritten with the
/// (ideally quasi-triangular) iterate; `v`, when present, is post-multiplied
/// by the accumulated orthogonal factor so `A = V H V'` is maintained.
/// Returns the sweep count and whether the subdiagonal fell below `tol`.
///
/// A sweep is `n-1` Givens rotations applied to rows, then the same rotations
/// applied from the right. Hessenberg form is invariant under that step, so
/// each sweep is `O(n^2)`. The whole iteration lives here rather than in the
/// caller because the loop is thousands of `O(n^2)` sweeps: crossing back into
/// Python once per sweep would cost more than the arithmetic does.
pub fn hessenberg_qr_iterate(
    n: usize,
    h: &mut [f64],
    mut v: Option<&mut [f64]>,
    tol: f64,
    max_iter: usize,
) -> (usize, bool) {
    if n < 2 {
        return (0, true);
    }
    let mut cs = vec![0.0f64; n - 1];
    let mut sn = vec![0.0f64; n - 1];

    for iter in 1..=max_iter {
        // --- H = Q R: clear the subdiagonal, rotating rows (k, k+1).
        for k in 0..n - 1 {
            let a = h[k * n + k];
            let b = h[(k + 1) * n + k];
            let r = a.hypot(b);
            let (c, s) = if r == 0.0 { (1.0, 0.0) } else { (a / r, b / r) };
            cs[k] = c;
            sn[k] = s;
            if s != 0.0 {
                let (top, bot) = h.split_at_mut((k + 1) * n);
                let row0 = &mut top[k * n + k..k * n + n];
                let row1 = &mut bot[k..n];
                for (x, y) in row0.iter_mut().zip(row1.iter_mut()) {
                    let (u, w) = (*x, *y);
                    *x = c * u + s * w;
                    *y = c * w - s * u;
                }
            }
        }
        // --- H <- R Q: the same rotations from the right. R is upper
        // triangular, so rotation k only reaches row k+1 -- the entry that
        // restores the Hessenberg subdiagonal.
        for k in 0..n - 1 {
            let (c, s) = (cs[k], sn[k]);
            if s != 0.0 {
                for row in h.chunks_exact_mut(n).take(k + 2) {
                    let (u, w) = (row[k], row[k + 1]);
                    row[k] = c * u + s * w;
                    row[k + 1] = c * w - s * u;
                }
                if let Some(vv) = v.as_deref_mut() {
                    for row in vv.chunks_exact_mut(n) {
                        let (u, w) = (row[k], row[k + 1]);
                        row[k] = c * u + s * w;
                        row[k + 1] = c * w - s * u;
                    }
                }
            }
        }
        // A subdiagonal entry is negligible when it is small next to the two
        // diagonal entries it sits between. Comparing it against an unscaled
        // `tol` instead cannot succeed once ||A|| is large, since rounding
        // holds a converged subdiagonal near eps*||A||.
        let mut deflated = true;
        for k in 0..n - 1 {
            let e = h[(k + 1) * n + k].abs();
            let d = h[k * n + k].abs() + h[(k + 1) * n + k + 1].abs();
            if e > tol * (d + 1e-300) {
                deflated = false;
                break;
            }
        }
        if deflated {
            return (iter, true);
        }
    }
    (max_iter, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    /// A V = V diag(lambda) is the definition; check it directly.
    fn check_pairs(n: usize, a0: &[f64], vals: &[f64], vecs: &[f64], tol: f64, what: &str) {
        let mut err: f64 = 0.0;
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += a0[i * n + k] * vecs[k * n + j];
                }
                err = err.max((s - vals[j] * vecs[i * n + j]).abs());
            }
        }
        assert!(err < tol, "{what}: n={n} residual {err:e}");
    }

    #[test]
    fn jacobi_matches_definition() {
        for &n in &[2usize, 8, 40] {
            let a0 = sym(n, 11 + n as u64);
            let mut d = a0.clone();
            let mut v = vec![0.0f64; n * n];
            let (_, conv) = jacobi_eigen(n, &mut d, &mut v, 1e-12, 100);
            assert!(conv, "n={n} jacobi did not converge");
            let vals: Vec<f64> = (0..n).map(|i| d[i * n + i]).collect();
            check_pairs(n, &a0, &vals, &v, 1e-9, "jacobi");
        }
    }
}
