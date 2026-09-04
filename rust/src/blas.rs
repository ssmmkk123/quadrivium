//! Cache-blocked dense primitives on row-major `f64` slices.
//!
//! These are the innermost operations every factorization in this crate is
//! built from. They deliberately work on plain slices rather than `ndarray`
//! views: the hot loops then compile to straight-line code the autovectorizer
//! can turn into SIMD without bounds checks in the middle.

/// Block size for the register-level kernel. Chosen so one block of A, B and C
/// stays inside L1 on typical x86-64 and aarch64 cores.
pub const MC: usize = 96;
pub const KC: usize = 128;
pub const NC: usize = 256;

/// Dot product with a 4-way split accumulator so the compiler can pipeline the
/// FMAs instead of serialising on a single dependency chain.
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
    let chunks = n / 4;
    for c in 0..chunks {
        let i = c * 4;
        s0 += x[i] * y[i];
        s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2];
        s3 += x[i + 3] * y[i + 3];
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for i in chunks * 4..n {
        s += x[i] * y[i];
    }
    s
}

/// `y := y + alpha * x`
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    for i in 0..n {
        y[i] += alpha * x[i];
    }
}

#[inline]
pub fn scal(alpha: f64, x: &mut [f64]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

#[inline]
pub fn nrm2(x: &[f64]) -> f64 {
    // Two-pass scaling guards against overflow for large-magnitude vectors
    // without the cost of the fully incremental Blue algorithm.
    let mut scale = 0.0f64;
    for &v in x {
        let a = v.abs();
        if a > scale {
            scale = a;
        }
    }
    if scale == 0.0 || !scale.is_finite() {
        return if scale.is_finite() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let inv = 1.0 / scale;
    let mut s = 0.0;
    for &v in x {
        let t = v * inv;
        s += t * t;
    }
    scale * s.sqrt()
}

/// `C := C + alpha * A * B`, blocked. This is the workhorse behind the
/// right-looking blocked factorizations.
///
/// The argument list is the usual BLAS one -- three dimensions, a scalar, and a
/// pointer/stride pair per operand -- and splitting it into a struct would only
/// move the same information one level down.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nn_acc(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    let mut jc = 0;
    while jc < n {
        let nb = NC.min(n - jc);
        let mut pc = 0;
        while pc < k {
            let kb = KC.min(k - pc);
            let mut ic = 0;
            while ic < m {
                let mb = MC.min(m - ic);
                // Register-blocked micro kernel over one (mb x kb) x (kb x nb) tile.
                for i in 0..mb {
                    let arow = &a[(ic + i) * lda + pc..(ic + i) * lda + pc + kb];
                    let crow = &mut c[(ic + i) * ldc + jc..(ic + i) * ldc + jc + nb];
                    for p in 0..kb {
                        let av = alpha * arow[p];
                        if av == 0.0 {
                            continue;
                        }
                        let brow = &b[(pc + p) * ldb + jc..(pc + p) * ldb + jc + nb];
                        for j in 0..nb {
                            crow[j] += av * brow[j];
                        }
                    }
                }
                ic += mb;
            }
            pc += kb;
        }
        jc += nb;
    }
}

/// Swap two rows of a row-major matrix in place.
#[inline]
pub fn swap_rows(a: &mut [f64], lda: usize, ncols: usize, r1: usize, r2: usize) {
    if r1 == r2 {
        return;
    }
    let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
    let (top, bot) = a.split_at_mut(hi * lda);
    let x = &mut top[lo * lda..lo * lda + ncols];
    let y = &mut bot[..ncols];
    x.swap_with_slice(y);
}
