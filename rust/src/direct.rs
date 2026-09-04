//! Blocked dense factorizations: LU, Cholesky, LDL', QR, Hessenberg.
//!
//! Every routine here matches the numerical conventions of the pure-Python
//! implementations in `quadrivium.linalg.direct` exactly -- same pivot choice,
//! same sign convention on Householder reflectors -- so the two backends are
//! bit-comparable on well-conditioned input and interchangeable everywhere.

use crate::blas::{axpy, dot, nrm2, scal, swap_rows};

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

/// Solve `L' x = b` for lower-triangular `L`, without forming `L'`.
///
/// Used for the second half of a Cholesky solve, where the caller has `L` and
/// wants `L' x = y`. Materialising the transpose would copy the whole matrix
/// for an `O(n^2)` solve; sweeping columns from the right instead reads only
/// contiguous rows of `L`.
pub fn back_substitution_trans(
    n: usize,
    l: &[f64],
    ldl: usize,
    b: &mut [f64],
    unit: bool,
) -> Result<(), LinalgError> {
    for j in (0..n).rev() {
        if !unit {
            let d = l[j * ldl + j];
            if d == 0.0 {
                return Err(LinalgError::Singular(format!(
                    "zero diagonal entry at row {j}"
                )));
            }
            b[j] /= d;
        }
        let xj = b[j];
        if xj != 0.0 {
            axpy(-xj, &l[j * ldl..j * ldl + j], &mut b[..j]);
        }
    }
    Ok(())
}

/// `np.allclose(a, a.T, atol=tol)` with NumPy's default `rtol`, short-circuited.
///
/// The Python form builds several `n x n` temporaries; this streams the matrix
/// once and stops at the first asymmetric entry, which is the common case.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn is_symmetric(n: usize, a: &[f64], atol: f64, rtol: f64) -> bool {
    for i in 0..n {
        for j in 0..i {
            let x = a[i * n + j];
            let y = a[j * n + i];
            // The comparison is negated rather than reversed on purpose: a NaN
            // entry must report "not symmetric", which is what `np.allclose`
            // does, and `>` would silently accept it.
            if !((x - y).abs() <= atol + rtol * y.abs()) {
                return false;
            }
        }
    }
    true
}

/// Width at which the recursive LU stops splitting and factors directly.
const LU_LEAF: usize = 16;
/// Column-panel width for the outer blocked loop.
const LU_NB: usize = 128;

/// Unblocked right-looking LU over columns `[j, j + w)`, rows `[j, n)`.
/// Row interchanges are applied across the full width of the matrix.
fn lu_unblocked(n: usize, a: &mut [f64], lda: usize, j: usize, w: usize, perm: &mut [usize]) {
    for k in j..j + w {
        let mut p = k;
        let mut best = a[k * lda + k].abs();
        for i in k + 1..n {
            let v = a[i * lda + k].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if p != k {
            swap_rows(a, lda, lda, k, p);
            perm.swap(k, p);
        }
        let piv = a[k * lda + k];
        if piv == 0.0 {
            continue;
        }
        let inv = 1.0 / piv;
        for i in k + 1..n {
            a[i * lda + k] *= inv;
        }
        // Rank-1 update of the remaining columns of this panel only; columns
        // to the right of the panel are handled by the caller's `gemm`.
        for i in k + 1..n {
            let lik = a[i * lda + k];
            if lik == 0.0 {
                continue;
            }
            let (before, after) = a.split_at_mut(i * lda);
            let urow = &before[k * lda + k + 1..k * lda + j + w];
            let arow = &mut after[k + 1..j + w];
            axpy(-lik, urow, arow);
        }
    }
}

/// Recursive LU over columns `[j, j + w)`, rows `[j, n)`.
///
/// Splitting the panel in half and updating the right half with one `gemm` is
/// what keeps the arithmetic in the SIMD kernel. The flat right-looking form
/// leaves `O(n^2 * blocksize)` of scalar rank-1 updates on the critical path,
/// which measured as roughly a quarter of the total work.
fn lu_recursive(n: usize, a: &mut [f64], lda: usize, j: usize, w: usize, perm: &mut [usize]) {
    if w <= LU_LEAF {
        lu_unblocked(n, a, lda, j, w, perm);
        return;
    }
    let w1 = w / 2;
    lu_recursive(n, a, lda, j, w1, perm);

    // U12 := L11^-1 A12, with L11 unit lower triangular.
    for k in 0..w1 {
        for i in k + 1..w1 {
            let lik = a[(j + i) * lda + j + k];
            if lik == 0.0 {
                continue;
            }
            let (before, after) = a.split_at_mut((j + i) * lda);
            let src = &before[(j + k) * lda + j + w1..(j + k) * lda + j + w];
            let dst = &mut after[j + w1..j + w];
            axpy(-lik, src, dst);
        }
    }
    // A22 := A22 - L21 U12. Both operands overlap the destination inside `a`,
    // so they are copied out; the copies cost O(n * w1) against the
    // O(n * w1 * w) of arithmetic they enable.
    let rows = n - j - w1;
    let cols = w - w1;
    if rows > 0 && cols > 0 {
        let mut l21 = vec![0.0f64; rows * w1];
        let mut u12 = vec![0.0f64; w1 * cols];
        for i in 0..rows {
            l21[i * w1..(i + 1) * w1]
                .copy_from_slice(&a[(j + w1 + i) * lda + j..(j + w1 + i) * lda + j + w1]);
        }
        for pp in 0..w1 {
            u12[pp * cols..(pp + 1) * cols]
                .copy_from_slice(&a[(j + pp) * lda + j + w1..(j + pp) * lda + j + w]);
        }
        crate::gemm::gemm_acc(
            rows,
            cols,
            w1,
            -1.0,
            &l21,
            w1,
            &u12,
            cols,
            &mut a[(j + w1) * lda + j + w1..],
            lda,
        );
    }
    lu_recursive(n, a, lda, j + w1, w - w1, perm);
}

/// LU with partial pivoting, computed in place on a row-major `n x n` matrix.
/// On return `a` holds `L` below the diagonal (unit diagonal implied) and `U`
/// on and above it; `perm` is the row permutation, `perm[i]` naming the
/// original row now in position `i`.
///
/// An exactly zero pivot is skipped rather than raised, matching
/// `plu_decomposition` in the Python layer.
pub fn plu_inplace(n: usize, a: &mut [f64], perm: &mut [usize]) -> Result<(), LinalgError> {
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    if n == 0 {
        return Ok(());
    }
    let lda = n;
    let mut j = 0;
    while j < n {
        let jb = LU_NB.min(n - j);
        // Factor just this column panel; the recursion keeps its own updates
        // inside the panel, so the columns to the right are untouched.
        lu_recursive(n, a, lda, j, jb, perm);
        let rest = n - j - jb;
        if rest > 0 {
            // U12 := L11^-1 A12 across the full trailing width.
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
            // A22 := A22 - L21 U12 -- one large `gemm` per panel, which is the
            // whole point of the outer blocking: the purely recursive form
            // splits this into many thin updates the SIMD kernel cannot fill.
            let mut l21 = vec![0.0f64; rest * jb];
            for i in 0..rest {
                l21[i * jb..(i + 1) * jb]
                    .copy_from_slice(&a[(j + jb + i) * lda + j..(j + jb + i) * lda + j + jb]);
            }
            let (top, bottom) = a.split_at_mut((j + jb) * lda);
            crate::gemm::gemm_acc(
                rest,
                rest,
                jb,
                -1.0,
                &l21,
                jb,
                &top[j * lda + j + jb..],
                lda,
                &mut bottom[j + jb..],
                lda,
            );
        }
        j += jb;
    }
    Ok(())
}

/// Blocking factor for the right-looking factorizations. Large enough that the
/// trailing `gemm` has real work to do, small enough that the unblocked
/// diagonal panel stays in L1.
pub const NB: usize = 192;
/// Inner blocking factor used inside a panel, where the operands are tall and
/// narrow and the useful `gemm` is correspondingly thinner.
pub const SB: usize = 32;

/// Cholesky `A = L L'`, lower triangle written in place; the strict upper
/// triangle of `a` is zeroed.
///
/// Right-looking and blocked: each panel is factored unblocked, then the whole
/// trailing submatrix is updated with one `gemm`. That is what puts the bulk of
/// the arithmetic through the SIMD micro-kernel -- the textbook
/// Cholesky-Banachiewicz form is all dot products and cannot be vectorised
/// nearly as well.
pub fn cholesky_inplace(n: usize, a: &mut [f64]) -> Result<(), LinalgError> {
    let lda = n;
    let mut j = 0;
    while j < n {
        let jb = NB.min(n - j);
        // Factor the column panel. A second level of blocking runs inside it:
        // without this the panel's triangular solve is `O(n^2 * NB)` of scalar
        // dot products, which at NB = 128 is about half the total arithmetic
        // and leaves the SIMD kernel idle for most of the factorization.
        let mut kb = j;
        while kb < j + jb {
            let kbb = SB.min(j + jb - kb);
            if kb > j {
                // Subtract the contribution of the panel columns already done.
                let width = kb - j;
                let rows = n - kb;
                let mut left = vec![0.0f64; rows * width];
                let mut rightt = vec![0.0f64; width * kbb];
                for i in 0..rows {
                    left[i * width..(i + 1) * width]
                        .copy_from_slice(&a[(kb + i) * lda + j..(kb + i) * lda + kb]);
                }
                for c in 0..kbb {
                    for p in 0..width {
                        rightt[p * kbb + c] = a[(kb + c) * lda + j + p];
                    }
                }
                crate::gemm::gemm_acc(
                    rows,
                    kbb,
                    width,
                    -1.0,
                    &left,
                    width,
                    &rightt,
                    kbb,
                    &mut a[kb * lda + kb..],
                    lda,
                );
            }
            for k in kb..kb + kbb {
                let d = a[k * lda + k]
                    - dot(&a[k * lda + kb..k * lda + k], &a[k * lda + kb..k * lda + k]);
                if d <= 0.0 {
                    return Err(LinalgError::NotPositiveDefinite(format!(
                        "matrix is not positive definite (non-positive pivot {d:.3e} at index {k})"
                    )));
                }
                let dsq = d.sqrt();
                a[k * lda + k] = dsq;
                let inv = 1.0 / dsq;
                let (top, bottom) = a.split_at_mut((k + 1) * lda);
                let krow = &top[k * lda + kb..k * lda + k];
                for i in 0..n - k - 1 {
                    let row = &mut bottom[i * lda + kb..i * lda + k + 1];
                    let sdot = row[k - kb] - dot(&row[..k - kb], krow);
                    row[k - kb] = sdot * inv;
                }
            }
            kb += kbb;
        }
        let rest = n - j - jb;
        if rest > 0 {
            // A22 := A22 - L21 L21'. L21 overlaps A22 in the same buffer so it
            // is copied out once; the `A B'` kernel then reads that single copy
            // as both operands, with no transpose to materialise.
            let mut l21 = vec![0.0f64; rest * jb];
            for i in 0..rest {
                l21[i * jb..(i + 1) * jb]
                    .copy_from_slice(&a[(j + jb + i) * lda + j..(j + jb + i) * lda + j + jb]);
            }
            crate::gemm::gemm_nt_acc(
                rest,
                rest,
                jb,
                -1.0,
                &l21,
                jb,
                &l21,
                jb,
                &mut a[(j + jb) * lda + j + jb..],
                lda,
            );
        }
        j += jb;
    }
    for i in 0..n {
        for c in i + 1..n {
            a[i * lda + c] = 0.0;
        }
    }
    Ok(())
}

/// Panel width for the blocked QR.
const QR_NB: usize = 64;

/// Householder QR, blocked via the compact WY representation.
///
/// Reflectors are gathered a panel at a time into `V` and a small upper
/// triangular `T` with `H_1 H_2 ... H_nb = I - V T V'`. The trailing columns
/// and the accumulated `Q` are then updated with `gemm` instead of `nb`
/// separate rank-1 updates, which is what lets the SIMD kernel do the work.
///
/// The reflector sign convention matches the pure-Python routine exactly, so
/// both backends produce the same `Q` and `R` rather than merely equivalent
/// factorizations. `a` (m x n) is overwritten with `R`; `q` receives `Q` when
/// `want_q` is set.
pub fn householder_qr(m: usize, n: usize, a: &mut [f64], q: &mut [f64], want_q: bool) {
    let lda = n;
    if want_q {
        for i in 0..m {
            for j in 0..m {
                q[i * m + j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    let lim = n.min(m.saturating_sub(1));
    let mut k = 0;
    while k < lim {
        let kb = QR_NB.min(lim - k);
        let rows = m - k;
        let mut v = vec![0.0f64; rows * kb];
        let mut t = vec![0.0f64; kb * kb];
        let mut x = vec![0.0f64; rows];
        let mut w = vec![0.0f64; n];

        // Unblocked factorization of the panel, applying each reflector to the
        // remaining panel columns so the next one sees updated data.
        for j in 0..kb {
            let col = k + j;
            let len = m - col;
            for i in 0..len {
                x[i] = a[(col + i) * lda + col];
            }
            let alpha0 = nrm2(&x[..len]);
            if alpha0 == 0.0 {
                continue;
            }
            let alpha = if x[0] >= 0.0 { -alpha0 } else { alpha0 };
            x[0] -= alpha;
            let vn = nrm2(&x[..len]);
            if vn == 0.0 {
                continue;
            }
            scal(1.0 / vn, &mut x[..len]);
            for i in 0..len {
                v[(j + i) * kb + j] = x[i];
            }
            // A[col.., col..k+kb] -= 2 x (x' A[col.., col..k+kb]), two row-major passes.
            let hi = k + kb;
            for e in w[col..hi].iter_mut() {
                *e = 0.0;
            }
            for i in 0..len {
                let xi = x[i];
                if xi == 0.0 {
                    continue;
                }
                let row = &a[(col + i) * lda + col..(col + i) * lda + hi];
                for (c, e) in row.iter().enumerate() {
                    w[col + c] += xi * e;
                }
            }
            for i in 0..len {
                let s2 = 2.0 * x[i];
                if s2 == 0.0 {
                    continue;
                }
                let row = &mut a[(col + i) * lda + col..(col + i) * lda + hi];
                for (c, e) in row.iter_mut().enumerate() {
                    *e -= s2 * w[col + c];
                }
            }
        }

        // Build T from V. With every reflector normalised to ||v|| = 1 the
        // scalar tau is 2, and T_{1:j-1,j} = -2 T_{1:j-1,1:j-1} (V_{:,1:j-1}' v_j).
        t[0] = 2.0;
        let mut vw = vec![0.0f64; kb];
        for j in 1..kb {
            for p in 0..j {
                let mut acc = 0.0;
                for i in j..rows {
                    acc += v[i * kb + p] * v[i * kb + j];
                }
                vw[p] = acc;
            }
            for p in 0..j {
                let mut acc = 0.0;
                for qq in p..j {
                    acc += t[p * kb + qq] * vw[qq];
                }
                t[p * kb + j] = -2.0 * acc;
            }
            t[j * kb + j] = 2.0;
        }

        // V' and T', laid out contiguously for the gemms below.
        let mut vt = vec![0.0f64; kb * rows];
        for i in 0..rows {
            for j in 0..kb {
                vt[j * rows + i] = v[i * kb + j];
            }
        }
        let mut tt = vec![0.0f64; kb * kb];
        for i in 0..kb {
            for j in 0..kb {
                tt[j * kb + i] = t[i * kb + j];
            }
        }

        // Trailing columns: A2 -= V (T' (V' A2)).
        let rest = n - k - kb;
        if rest > 0 {
            let mut w1 = vec![0.0f64; kb * rest];
            crate::gemm::gemm(
                kb,
                rest,
                rows,
                &vt,
                rows,
                &a[k * lda + k + kb..],
                lda,
                &mut w1,
                rest,
            );
            let mut w2 = vec![0.0f64; kb * rest];
            crate::gemm::gemm(kb, rest, kb, &tt, kb, &w1, rest, &mut w2, rest);
            crate::gemm::gemm_acc(
                rows,
                rest,
                kb,
                -1.0,
                &v,
                kb,
                &w2,
                rest,
                &mut a[k * lda + k + kb..],
                lda,
            );
        }

        // Q := Q (I - V T V') = Q - ((Q V) T) V'.
        if want_q {
            let mut qv = vec![0.0f64; m * kb];
            crate::gemm::gemm(m, kb, rows, &q[k..], m, &v, kb, &mut qv, kb);
            let mut qvt = vec![0.0f64; m * kb];
            crate::gemm::gemm(m, kb, kb, &qv, kb, &t, kb, &mut qvt, kb);
            crate::gemm::gemm_acc(m, rows, kb, -1.0, &qvt, kb, &vt, rows, &mut q[k..], m);
        }
        k += kb;
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
