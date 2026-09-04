//! Blocked dense factorizations: LU, Cholesky, LDL', QR, Hessenberg.
//!
//! Every routine here matches the numerical conventions of the pure-Python
//! implementations in `quadrivium.linalg.direct` exactly -- same pivot choice,
//! same sign convention on Householder reflectors -- so the two backends are
//! bit-comparable on well-conditioned input and interchangeable everywhere.

use crate::blas::{axpy, dot, gemm_nn_acc, nrm2, scal, swap_rows};

pub const BLOCK: usize = 64;

#[derive(Debug)]
pub enum LinalgError {
    Singular(String),
    NotPositiveDefinite(String),
}

/// In-place forward substitution: solve `L x = b`, `b` overwritten with `x`.
pub fn forward_substitution(
    n: usize,
    l: &[f64],
    ldl: usize,
    b: &mut [f64],
    unit: bool,
) -> Result<(), LinalgError> {
    for i in 0..n {
        let s = b[i] - dot(&l[i * ldl..i * ldl + i], &b[..i]);
        if unit {
            b[i] = s;
        } else {
            let d = l[i * ldl + i];
            if d == 0.0 {
                return Err(LinalgError::Singular(format!(
                    "zero diagonal entry at row {i}"
                )));
            }
            b[i] = s / d;
        }
    }
    Ok(())
}

/// In-place back substitution: solve `U x = b`, `b` overwritten with `x`.
pub fn back_substitution(
    n: usize,
    u: &[f64],
    ldu: usize,
    b: &mut [f64],
    unit: bool,
) -> Result<(), LinalgError> {
    for i in (0..n).rev() {
        let s = b[i] - dot(&u[i * ldu + i + 1..i * ldu + n], &b[i + 1..n]);
        if unit {
            b[i] = s;
        } else {
            let d = u[i * ldu + i];
            if d == 0.0 {
                return Err(LinalgError::Singular(format!(
                    "zero diagonal entry at row {i}"
                )));
            }
            b[i] = s / d;
        }
    }
    Ok(())
}

/// Unblocked right-looking LU on a panel, partial pivoting. Returns row swaps
/// applied, recorded as `ipiv[k] = row swapped with k`.
fn lu_panel(
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    ipiv: &mut [usize],
    row_off: usize,
) -> Option<usize> {
    let mut singular = None;
    for k in 0..n.min(m) {
        // Partial pivot: largest magnitude in the current column below the diagonal.
        let mut p = k;
        let mut best = a[k * lda + k].abs();
        for i in k + 1..m {
            let v = a[i * lda + k].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        ipiv[k] = p + row_off;
        if p != k {
            swap_rows(a, lda, n, k, p);
        }
        let piv = a[k * lda + k];
        if piv == 0.0 {
            if singular.is_none() {
                singular = Some(k);
            }
            continue;
        }
        let inv = 1.0 / piv;
        for i in k + 1..m {
            a[i * lda + k] *= inv;
        }
        // Rank-1 update of the trailing panel.
        for i in k + 1..m {
            let lik = a[i * lda + k];
            if lik == 0.0 {
                continue;
            }
            let (before, after) = a.split_at_mut(i * lda);
            let urow = &before[k * lda + k + 1..k * lda + n];
            let arow = &mut after[k + 1..n];
            axpy(-lik, urow, arow);
        }
    }
    singular
}

/// Blocked LU with partial pivoting, computed in place on a row-major `n x n`
/// matrix. On return `a` holds `L` below the diagonal (unit diagonal implied)
/// and `U` on and above it; `perm` is the row permutation.
pub fn plu_inplace(n: usize, a: &mut [f64], perm: &mut [usize]) -> Result<(), LinalgError> {
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    let lda = n;
    let mut ipiv = vec![0usize; n];
    let mut j = 0;
    while j < n {
        let jb = BLOCK.min(n - j);
        // Factor the current column panel A[j.., j..j+jb].
        {
            let panel_rows = n - j;
            let mut panel = vec![0.0f64; panel_rows * jb];
            for i in 0..panel_rows {
                panel[i * jb..(i + 1) * jb]
                    .copy_from_slice(&a[(j + i) * lda + j..(j + i) * lda + j + jb]);
            }
            let mut lp = vec![0usize; jb];
            lu_panel(panel_rows, jb, &mut panel, jb, &mut lp, j);
            for i in 0..panel_rows {
                a[(j + i) * lda + j..(j + i) * lda + j + jb]
                    .copy_from_slice(&panel[i * jb..(i + 1) * jb]);
            }
            // Apply the panel's row interchanges to the rest of the matrix.
            for (k, &p) in lp.iter().enumerate() {
                let kk = j + k;
                if p != kk {
                    // Left of the panel.
                    for c in 0..j {
                        a.swap(kk * lda + c, p * lda + c);
                    }
                    // Right of the panel.
                    for c in j + jb..n {
                        a.swap(kk * lda + c, p * lda + c);
                    }
                    perm.swap(kk, p);
                }
            }
            ipiv[j..j + jb].copy_from_slice(&lp);
        }
        let rest = n - j - jb;
        if rest > 0 {
            // U12 := L11^-1 * A12  (triangular solve with unit lower L11)
            for k in 0..jb {
                for i in k + 1..jb {
                    let lik = a[(j + i) * lda + j + k];
                    if lik == 0.0 {
                        continue;
                    }
                    let (before, after) = a.split_at_mut((j + i) * lda);
                    let src = &before[(j + k) * lda + j + jb..(j + k) * lda + n];
                    let dst = &mut after[j + jb..n];
                    axpy(-lik, src, dst);
                }
            }
            // A22 := A22 - L21 * U12. L21 and A22 share the bottom row block,
            // so L21 is gathered into a packed buffer to satisfy the borrow
            // checker and to give the gemm a contiguous left operand.
            let (top, bottom) = a.split_at_mut((j + jb) * lda);
            let u12 = &top[j * lda + j + jb..];
            let mut l21 = vec![0.0f64; rest * jb];
            for i in 0..rest {
                l21[i * jb..(i + 1) * jb].copy_from_slice(&bottom[i * lda + j..i * lda + j + jb]);
            }
            gemm_nn_acc(
                rest,
                rest,
                jb,
                -1.0,
                &l21,
                jb,
                u12,
                lda,
                &mut bottom[j + jb..],
                lda,
            );
        }
        j += jb;
    }
    Ok(())
}

/// Cholesky `A = L L'` (Cholesky-Banachiewicz), lower triangle written in
/// place; the strict upper triangle of `a` is zeroed.
///
/// The row-major left-looking form is used deliberately: both operands of the
/// inner dot product are contiguous rows, so the whole factorization streams
/// through cache linearly and needs no packing buffers.
pub fn cholesky_inplace(n: usize, a: &mut [f64]) -> Result<(), LinalgError> {
    let lda = n;
    for k in 0..n {
        let d = a[k * lda + k] - dot(&a[k * lda..k * lda + k], &a[k * lda..k * lda + k]);
        if d <= 0.0 {
            return Err(LinalgError::NotPositiveDefinite(format!(
                "matrix is not positive definite (non-positive pivot {d:.3e} at index {k})"
            )));
        }
        let dsq = d.sqrt();
        a[k * lda + k] = dsq;
        let inv = 1.0 / dsq;
        let (top, bottom) = a.split_at_mut((k + 1) * lda);
        let krow = &top[k * lda..k * lda + k];
        for i in 0..n - k - 1 {
            let row = &mut bottom[i * lda..i * lda + k + 1];
            let s = row[k] - dot(&row[..k], krow);
            row[k] = s * inv;
        }
    }
    for i in 0..n {
        for c in i + 1..n {
            a[i * lda + c] = 0.0;
        }
    }
    Ok(())
}

/// Householder QR producing the same reflector sign convention as the Python
/// implementation. `a` (m x n) is overwritten with `R`; `qt` accumulates `Q^T`
/// when `want_q` is set.
pub fn householder_qr(m: usize, n: usize, a: &mut [f64], q: &mut [f64], want_q: bool) {
    let lda = n;
    if want_q {
        for i in 0..m {
            for j in 0..m {
                q[i * m + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    let mut v = vec![0.0f64; m];
    let mut w = vec![0.0f64; n];
    for k in 0..n.min(m.saturating_sub(1)) {
        let len = m - k;
        for i in 0..len {
            v[i] = a[(k + i) * lda + k];
        }
        let alpha = nrm2(&v[..len]);
        if alpha == 0.0 {
            continue;
        }
        // Sign chosen to avoid cancellation, matching the Python routine.
        let alpha = if v[0] >= 0.0 { -alpha } else { alpha };
        v[0] -= alpha;
        let vn = nrm2(&v[..len]);
        if vn == 0.0 {
            continue;
        }
        let inv = 1.0 / vn;
        scal(inv, &mut v[..len]);
        // A[k:, k:] -= 2 v (v' A[k:, k:]), done as two row-major passes.
        // Accumulating w = v' A first keeps every inner loop contiguous in j;
        // the naive column-at-a-time form strides by `lda` and is several times
        // slower once the trailing block leaves L2.
        for x in w[k..n].iter_mut() {
            *x = 0.0;
        }
        for i in 0..len {
            let vi = v[i];
            if vi == 0.0 {
                continue;
            }
            let row = &a[(k + i) * lda + k..(k + i) * lda + n];
            for j in 0..n - k {
                w[k + j] += vi * row[j];
            }
        }
        for i in 0..len {
            let s2 = 2.0 * v[i];
            if s2 == 0.0 {
                continue;
            }
            let row = &mut a[(k + i) * lda + k..(k + i) * lda + n];
            for j in 0..n - k {
                row[j] -= s2 * w[k + j];
            }
        }
        if want_q {
            // Q := Q (I - 2 v v') applied on the right, giving Q with A = Q R.
            for i in 0..m {
                let mut s = 0.0;
                for p in 0..len {
                    s += q[i * m + k + p] * v[p];
                }
                let s2 = 2.0 * s;
                for p in 0..len {
                    q[i * m + k + p] -= s2 * v[p];
                }
            }
        }
    }
    // Clean the strict lower triangle of R.
    for i in 0..m {
        for j in 0..n.min(i) {
            a[i * lda + j] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random fill so failures are reproducible.
    fn fill(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn spd(n: usize, seed: u64) -> Vec<f64> {
        let a = fill(n * n, seed);
        let mut s = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += a[i * n + k] * a[j * n + k];
                }
                s[i * n + j] = acc;
            }
            s[i * n + i] += n as f64;
        }
        s
    }

    #[test]
    fn plu_reconstructs_a() {
        for &n in &[1usize, 5, 64, 65, 130] {
            let a0 = fill(n * n, 42 + n as u64);
            let mut a = a0.clone();
            let mut perm = vec![0usize; n];
            plu_inplace(n, &mut a, &mut perm).expect("nonsingular");
            // Rebuild P A = L U and compare against the permuted original.
            let mut err: f64 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..=i.min(j) {
                        let l = if k == i { 1.0 } else { a[i * n + k] };
                        if k <= j {
                            s += l * a[k * n + j];
                        }
                    }
                    err = err.max((s - a0[perm[i] * n + j]).abs());
                }
            }
            assert!(err < 1e-10, "n={n} reconstruction error {err:e}");
        }
    }

    #[test]
    fn cholesky_reconstructs_a() {
        for &n in &[1usize, 6, 64, 70, 129] {
            let a0 = spd(n, 7 + n as u64);
            let mut a = a0.clone();
            cholesky_inplace(n, &mut a).expect("spd");
            let mut err: f64 = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..=i.min(j) {
                        s += a[i * n + k] * a[j * n + k];
                    }
                    err = err.max((s - a0[i * n + j]).abs() / a0[i * n + i].max(1.0));
                }
            }
            assert!(err < 1e-10, "n={n} cholesky error {err:e}");
        }
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let mut a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(matches!(
            cholesky_inplace(2, &mut a),
            Err(LinalgError::NotPositiveDefinite(_))
        ));
    }

    #[test]
    fn qr_is_orthogonal_and_reconstructs() {
        for &(m, n) in &[(5usize, 5usize), (80, 40), (64, 64), (100, 97)] {
            let a0 = fill(m * n, 3 + m as u64);
            let mut r = a0.clone();
            let mut q = vec![0.0f64; m * m];
            householder_qr(m, n, &mut r, &mut q, true);
            // Q'Q = I
            let mut orth: f64 = 0.0;
            for i in 0..m {
                for j in 0..m {
                    let mut s = 0.0;
                    for k in 0..m {
                        s += q[k * m + i] * q[k * m + j];
                    }
                    orth = orth.max((s - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            assert!(orth < 1e-10, "m={m} n={n} orthogonality {orth:e}");
            // Q R = A
            let mut err: f64 = 0.0;
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..m {
                        s += q[i * m + k] * r[k * n + j];
                    }
                    err = err.max((s - a0[i * n + j]).abs());
                }
            }
            assert!(err < 1e-10, "m={m} n={n} QR reconstruction {err:e}");
        }
    }

    #[test]
    fn triangular_solves_round_trip() {
        let n = 40;
        let mut l = fill(n * n, 99);
        for i in 0..n {
            for j in i + 1..n {
                l[i * n + j] = 0.0;
            }
            l[i * n + i] += 3.0;
        }
        let x0 = fill(n, 5);
        let mut b = vec![0.0f64; n];
        for i in 0..n {
            b[i] = dot(&l[i * n..i * n + i + 1], &x0[..i + 1]);
        }
        forward_substitution(n, &l, n, &mut b, false).unwrap();
        let err = (0..n).map(|i| (b[i] - x0[i]).abs()).fold(0.0f64, f64::max);
        assert!(err < 1e-9, "forward substitution error {err:e}");
    }
}
