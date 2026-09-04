//! Fast Fourier transforms: iterative radix-2, mixed-radix Cooley-Tukey, and
//! Bluestein's chirp-z for lengths with a large prime factor.

use num_complex::Complex64;
use std::f64::consts::PI;

#[inline]
fn is_pow2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn next_pow2(n: usize) -> usize {
    let mut m = 1usize;
    while m < n {
        m <<= 1;
    }
    m
}

/// In-place bit-reversal permutation.
fn bit_reverse(a: &mut [Complex64]) {
    let n = a.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
}

/// Iterative radix-2 Cooley-Tukey, decimation in time. `n` must be a power of
/// two. Does not apply the `1/n` scaling; callers do that once at the top.
fn radix2(a: &mut [Complex64], inverse: bool) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    bit_reverse(a);
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        // One twiddle table per stage, built by repeated rotation from a seed
        // computed with sin/cos so the error stays at O(sqrt(log n)) ulp.
        let theta = sign * 2.0 * PI / size as f64;
        let wstep = Complex64::new(theta.cos(), theta.sin());
        let mut start = 0usize;
        while start < n {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..half {
                let u = a[start + k];
                let t = a[start + k + half] * w;
                a[start + k] = u + t;
                a[start + k + half] = u - t;
                // Re-seed periodically to stop the running product from drifting.
                w = if k % 64 == 63 {
                    let ang = theta * (k + 1) as f64;
                    Complex64::new(ang.cos(), ang.sin())
                } else {
                    w * wstep
                };
            }
            start += size;
        }
        size <<= 1;
    }
}

/// Bluestein's algorithm: rewrite the DFT as a convolution so any length costs
/// `O(n log n)`.
fn bluestein(a: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = a.len();
    if n <= 1 {
        return a.to_vec();
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let m = next_pow2(2 * n - 1);
    let mut chirp = Vec::with_capacity(n);
    for k in 0..n {
        // k*k mod 2n keeps the angle small, which matters for large n.
        let kk = (k as u128 * k as u128 % (2 * n as u128)) as f64;
        let ang = sign * PI * kk / n as f64;
        chirp.push(Complex64::new(ang.cos(), ang.sin()));
    }
    let mut fa = vec![Complex64::new(0.0, 0.0); m];
    let mut fb = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..n {
        fa[k] = a[k] * chirp[k];
        fb[k] = chirp[k].conj();
        if k > 0 {
            fb[m - k] = chirp[k].conj();
        }
    }
    radix2(&mut fa, false);
    radix2(&mut fb, false);
    for i in 0..m {
        fa[i] *= fb[i];
    }
    radix2(&mut fa, true);
    let inv_m = 1.0 / m as f64;
    (0..n).map(|k| fa[k] * inv_m * chirp[k]).collect()
}

/// Radix-`p` butterfly on `c`, writing `p` outputs into `out`.
///
/// The small radices get closed forms rather than a `p x p` matrix product:
/// radix 5 costs a handful of real multiplies where the generic form costs 25
/// complex ones, and these are the radices that actually turn up (10000 is
/// 2^4 * 5^4). Any other radix falls back to a direct DFT that indexes the
/// shared twiddle table -- building a private matrix per call cost more than
/// the whole transform for lengths like 30030 = 2*3*5*7*11*13.
#[inline]
fn butterfly(
    p: usize,
    sign: f64,
    c: &[Complex64],
    tw: &[Complex64],
    pstep: usize,
    out: &mut [Complex64],
) {
    match p {
        2 => {
            out[0] = c[0] + c[1];
            out[1] = c[0] - c[1];
        }
        3 => {
            const S3: f64 = 0.866_025_403_784_438_6; // sqrt(3)/2
            let t1 = c[1] + c[2];
            let t2 = c[0] - t1 * 0.5;
            let d = c[1] - c[2];
            let r = Complex64::new(-sign * S3 * d.im, sign * S3 * d.re);
            out[0] = c[0] + t1;
            out[1] = t2 + r;
            out[2] = t2 - r;
        }
        4 => {
            let t0 = c[0] + c[2];
            let t1 = c[0] - c[2];
            let t2 = c[1] + c[3];
            let t3 = c[1] - c[3];
            let r = Complex64::new(-sign * t3.im, sign * t3.re);
            out[0] = t0 + t2;
            out[1] = t1 + r;
            out[2] = t0 - t2;
            out[3] = t1 - r;
        }
        5 => {
            const C1: f64 = 0.309_016_994_374_947_45; // cos(2pi/5)
            const S1: f64 = 0.951_056_516_295_153_5; // sin(2pi/5)
            const C2: f64 = -0.809_016_994_374_947_5; // cos(4pi/5)
            const S2: f64 = 0.587_785_252_292_473_1; // sin(4pi/5)
            let t1 = c[1] + c[4];
            let t2 = c[2] + c[3];
            let t3 = c[1] - c[4];
            let t4 = c[2] - c[3];
            out[0] = c[0] + t1 + t2;
            let m1 = c[0] + t1 * C1 + t2 * C2;
            let m2 = c[0] + t1 * C2 + t2 * C1;
            let s1 = t3 * S1 + t4 * S2;
            let s2 = t3 * S2 - t4 * S1;
            let r1 = Complex64::new(-sign * s1.im, sign * s1.re);
            let r2 = Complex64::new(-sign * s2.im, sign * s2.re);
            out[1] = m1 + r1;
            out[4] = m1 - r1;
            out[2] = m2 + r2;
            out[3] = m2 - r2;
        }
        _ => {
            let big = tw.len();
            for (q, o) in out.iter_mut().enumerate().take(p) {
                let mut acc = Complex64::new(0.0, 0.0);
                for r in 0..p {
                    acc += c[r] * tw[(q * r * pstep) % big];
                }
                *o = acc;
            }
        }
    }
}

fn smallest_factor(n: usize) -> Option<usize> {
    // Radix 4 first: it halves the number of passes over the data compared
    // with two radix-2 passes, for the same arithmetic.
    if n % 4 == 0 {
        return Some(4);
    }
    let mut q = 2usize;
    while q * q <= n {
        if n % q == 0 {
            return Some(q);
        }
        q += 1;
    }
    None
}

/// Recursive Cooley-Tukey working out-of-place between two caller-owned
/// buffers, with no allocation of its own.
///
/// `x` is read with `stride`, so the decimation never has to gather into a
/// temporary. `out` and `scratch` are disjoint length-`n` regions: the `p`
/// sub-transforms write into slices of `scratch` while using the matching
/// slices of `out` as their own scratch, then the combine step reads `scratch`
/// and writes `out`.
///
/// `tw` holds `w_N^t` for the top-level `N`; a sub-problem of length `n` uses
/// the same table with its exponents scaled by `step = N / n`.
#[allow(clippy::too_many_arguments)]
fn rec(
    x: &[Complex64],
    stride: usize,
    n: usize,
    out: &mut [Complex64],
    scratch: &mut [Complex64],
    tw: &[Complex64],
    step: usize,
    sign: f64,
) {
    if n == 1 {
        out[0] = x[0];
        return;
    }
    let big = tw.len();
    let p = match smallest_factor(n) {
        Some(p) if p <= 32 => p,
        // A short prime length is its own base case: one direct DFT is far
        // cheaper than setting up a chirp-z transform.
        None if n <= 64 => n,
        // A long prime factor: chirp-z is the only O(n log n) route, and it is
        // rare enough to afford its own allocation.
        _ => {
            let gathered: Vec<Complex64> = (0..n).map(|j| x[j * stride]).collect();
            out[..n].copy_from_slice(&bluestein(&gathered, sign > 0.0));
            return;
        }
    };
    // `w_p^t == tw[t * (N / p)]`, so the generic radix needs no table of its own.
    let pstep = big / p;
    let mut col = [Complex64::new(0.0, 0.0); 64];
    let mut bfly = [Complex64::new(0.0, 0.0); 64];
    if n == p {
        // Base case: a single butterfly over the strided input. Terminating
        // here rather than recursing to length 1 removes one call per input
        // element -- about 200k calls at n = 100000, which dominated.
        for r in 0..p {
            col[r] = x[r * stride];
        }
        butterfly(p, sign, &col[..p], tw, pstep, &mut bfly[..p]);
        out[..p].copy_from_slice(&bfly[..p]);
        return;
    }
    let m = n / p;
    for r in 0..p {
        let (o, sc) = (
            &mut scratch[r * m..(r + 1) * m],
            &mut out[r * m..(r + 1) * m],
        );
        rec(&x[r * stride..], stride * p, m, o, sc, tw, step * p, sign);
    }
    for k in 0..m {
        for r in 0..p {
            let idx = (r * k * step) % big;
            col[r] = scratch[r * m + k] * tw[idx];
        }
        butterfly(p, sign, &col[..p], tw, pstep, &mut bfly[..p]);
        for q in 0..p {
            out[q * m + k] = bfly[q];
        }
    }
}

/// Unscaled forward/inverse transform for any length.
fn transform(a: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = a.len();
    if n <= 1 {
        return a.to_vec();
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    if is_pow2(n) {
        let mut buf = a.to_vec();
        radix2(&mut buf, inverse);
        return buf;
    }
    if smallest_factor(n).is_none() {
        return bluestein(a, inverse);
    }
    // One twiddle table serves every level of the recursion.
    let mut tw = Vec::with_capacity(n);
    for t in 0..n {
        let ang = sign * 2.0 * PI * t as f64 / n as f64;
        tw.push(Complex64::new(ang.cos(), ang.sin()));
    }
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    let mut scratch = vec![Complex64::new(0.0, 0.0); n];
    rec(a, 1, n, &mut out, &mut scratch, &tw, 1, sign);
    out
}

/// Forward DFT.
pub fn fft(a: &[Complex64]) -> Vec<Complex64> {
    transform(a, false)
}

/// Inverse DFT, including the `1/n` normalisation.
pub fn ifft(a: &[Complex64]) -> Vec<Complex64> {
    let n = a.len();
    let mut out = transform(a, true);
    if n > 1 {
        let inv = 1.0 / n as f64;
        for v in out.iter_mut() {
            *v *= inv;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[Complex64], inverse: bool) -> Vec<Complex64> {
        let n = a.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                let mut s = Complex64::new(0.0, 0.0);
                for (j, &v) in a.iter().enumerate() {
                    let ang = sign * 2.0 * PI * j as f64 * k as f64 / n as f64;
                    s += v * Complex64::new(ang.cos(), ang.sin());
                }
                s
            })
            .collect()
    }

    fn sample(n: usize, seed: u64) -> Vec<Complex64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let re = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let im = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
                Complex64::new(re, im)
            })
            .collect()
    }

    #[test]
    fn matches_naive_dft_for_many_lengths() {
        // Powers of two, small composites, primes, and a prime > 32 to force
        // the Bluestein branch inside the mixed-radix recursion.
        for &n in &[
            1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 30, 31, 32, 36, 60, 64, 97, 100,
            128, 121, 210,
        ] {
            let a = sample(n, 5 + n as u64);
            let got = fft(&a);
            let want = naive(&a, false);
            let err = (0..n)
                .map(|i| (got[i] - want[i]).norm())
                .fold(0.0f64, f64::max);
            let scale = (n as f64).sqrt();
            assert!(err < 1e-9 * scale, "n={n} fft error {err:e}");
        }
    }

    #[test]
    fn ifft_inverts_fft() {
        for &n in &[1usize, 5, 8, 13, 60, 97, 256, 1000] {
            let a = sample(n, 77 + n as u64);
            let back = ifft(&fft(&a));
            let err = (0..n)
                .map(|i| (back[i] - a[i]).norm())
                .fold(0.0f64, f64::max);
            assert!(err < 1e-10, "n={n} round-trip error {err:e}");
        }
    }
}
