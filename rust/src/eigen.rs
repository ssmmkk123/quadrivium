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
        if off < tol {
            converged = true;
            break;
        }
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
