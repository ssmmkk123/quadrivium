//! Cache-blocked dense primitives on row-major `f64` slices.
//!
//! These are the innermost operations every factorization in this crate is
//! built from. They deliberately work on plain slices rather than `ndarray`
//! views: the hot loops then compile to straight-line code the autovectorizer
//! can turn into SIMD without bounds checks in the middle.

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
