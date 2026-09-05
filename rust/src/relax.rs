//! Stationary relaxation sweeps on a 5-point Laplacian stencil.
//!
//! Lexicographic Gauss-Seidel and SOR are sequential by construction: the
//! update at `(i, j)` reads values already overwritten in the same sweep, so
//! there is no array expression that computes it. The pure-Python twin is a
//! doubly-nested loop over the grid, which is why this kernel exists -- and why
//! it runs the *entire* iteration rather than one sweep: at 40x40 a converging
//! solve is tens of thousands of sweeps, and a boundary crossing per sweep
//! would dominate the arithmetic.
//!
//! The arithmetic is ordered exactly as the Python is, so both backends walk
//! the same sequence of floating-point values and agree bit for bit.

/// Run SOR (`omega = 1` gives Gauss-Seidel) until the sup-norm change over a
/// sweep drops below `tol`.
///
/// `u` is a row-major `(nx+1) x (ny+1)` grid holding the boundary values, and
/// is updated in place; `f` is the source term on the same grid. Returns the
/// sweep count, whether it converged, and the per-sweep change history.
#[allow(clippy::too_many_arguments)]
pub fn sor_poisson(
    nx: usize,
    ny: usize,
    u: &mut [f64],
    f: &[f64],
    beta2: f64,
    dx2: f64,
    omega: f64,
    tol: f64,
    max_iter: usize,
) -> (usize, bool, Vec<f64>) {
    let stride = ny + 1;
    let denom = 2.0 * (1.0 + beta2);
    let mut residuals = Vec::with_capacity(max_iter.min(4096));
    for it in 1..=max_iter {
        let mut change = 0.0f64;
        for i in 1..nx {
            let row = i * stride;
            let up = row + stride;
            let down = row - stride;
            for j in 1..ny {
                let old = u[row + j];
                let new = (u[up + j] + u[down + j] + beta2 * (u[row + j + 1] + u[row + j - 1])
                    - dx2 * f[row + j])
                    / denom;
                let val = (1.0 - omega) * old + omega * new;
                u[row + j] = val;
                let d = (val - old).abs();
                if d > change || d.is_nan() {
                    change = d;
                }
            }
        }
        residuals.push(change);
        if change < tol {
            return (it, true, residuals);
        }
    }
    (max_iter, false, residuals)
}

/// Thomas algorithm for a tridiagonal system, matching the Python twin
/// operation for operation (including the two zero-pivot checks).
///
/// `sub` and `sup` have length `n-1`; `diag` and `rhs` have length `n` and are
/// consumed as scratch. Returns `None` on a zero pivot, which the caller turns
/// into the package's `SingularMatrixError`.
pub fn thomas(sub: &[f64], diag: &mut [f64], sup: &[f64], rhs: &mut [f64]) -> Option<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Some(Vec::new());
    }
    for i in 1..n {
        if diag[i - 1] == 0.0 {
            return None;
        }
        let m = sub[i - 1] / diag[i - 1];
        diag[i] -= m * sup[i - 1];
        rhs[i] -= m * rhs[i - 1];
    }
    if diag[n - 1] == 0.0 {
        return None;
    }
    let mut x = vec![0.0f64; n];
    x[n - 1] = rhs[n - 1] / diag[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = (rhs[i] - sup[i] * x[i + 1]) / diag[i];
    }
    Some(x)
}

/// Thomas algorithm for one tridiagonal matrix against many right-hand sides.
///
/// `rhs` is row-major `n x m`: column `k` is the `k`-th right-hand side, and
/// is overwritten with that system's solution. The elimination coefficients
/// depend only on the matrix, so the forward sweep divides once per row rather
/// than once per row and column -- and, more to the point, the caller crosses
/// the language boundary once instead of `m` times, which is where an
/// alternating-direction solver spends its time.
///
/// Returns `false` on a zero pivot, matching the single-system routine.
pub fn thomas_batch(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &mut [f64], m: usize) -> bool {
    let n = diag.len();
    if n == 0 || m == 0 {
        return true;
    }
    // `d[i]` is the eliminated diagonal, shared by every right-hand side.
    let mut d = vec![0.0f64; n];
    d[0] = diag[0];
    for i in 1..n {
        if d[i - 1] == 0.0 {
            return false;
        }
        let f = sub[i - 1] / d[i - 1];
        d[i] = diag[i] - f * sup[i - 1];
        let (prev, cur) = rhs.split_at_mut(i * m);
        let prev_row = &prev[(i - 1) * m..i * m];
        let cur_row = &mut cur[..m];
        for (x, &p) in cur_row.iter_mut().zip(prev_row) {
            *x -= f * p;
        }
    }
    if d[n - 1] == 0.0 {
        return false;
    }
    let inv = 1.0 / d[n - 1];
    for x in &mut rhs[(n - 1) * m..n * m] {
        *x *= inv;
    }
    for i in (0..n - 1).rev() {
        let inv = 1.0 / d[i];
        let s = sup[i];
        let (cur, next) = rhs.split_at_mut((i + 1) * m);
        let cur_row = &mut cur[i * m..];
        let next_row = &next[..m];
        for (x, &y) in cur_row.iter_mut().zip(next_row) {
            *x = (*x - s * y) * inv;
        }
    }
    true
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    /// Each column must match the single-system routine exactly.
    #[test]
    fn batch_matches_single_system() {
        let n = 9usize;
        let m = 4usize;
        let sub: Vec<f64> = (0..n - 1).map(|i| -1.0 - 0.01 * i as f64).collect();
        let diag: Vec<f64> = (0..n).map(|i| 4.0 + 0.1 * i as f64).collect();
        let sup: Vec<f64> = (0..n - 1).map(|i| -1.0 + 0.02 * i as f64).collect();
        let mut rhs = vec![0.0f64; n * m];
        for i in 0..n {
            for k in 0..m {
                rhs[i * m + k] = (i as f64 + 1.0) * (k as f64 + 2.0) % 7.0;
            }
        }
        let original = rhs.clone();
        assert!(thomas_batch(&sub, &diag, &sup, &mut rhs, m));
        for k in 0..m {
            let mut d = diag.clone();
            let mut b: Vec<f64> = (0..n).map(|i| original[i * m + k]).collect();
            let x = thomas(&sub, &mut d, &sup, &mut b).expect("single solve");
            for i in 0..n {
                assert!(
                    (x[i] - rhs[i * m + k]).abs() < 1e-13,
                    "column {k} row {i}: {} vs {}",
                    x[i],
                    rhs[i * m + k]
                );
            }
        }
    }

    /// A zero pivot is reported, not silently divided through.
    #[test]
    fn batch_reports_zero_pivot() {
        let mut rhs = vec![1.0f64; 3 * 2];
        assert!(!thomas_batch(
            &[1.0, 1.0],
            &[0.0, 1.0, 1.0],
            &[1.0, 1.0],
            &mut rhs,
            2
        ));
    }
}
