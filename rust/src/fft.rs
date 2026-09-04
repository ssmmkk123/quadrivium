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

fn smallest_factor(n: usize) -> Option<usize> {
    let mut q = 2usize;
    while q * q <= n {
        if n % q == 0 {
            return Some(q);
        }
        q += 1;
    }
    None
}

/// Unscaled forward/inverse transform for any length.
fn transform(a: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let n = a.len();
    if n <= 1 {
        return a.to_vec();
    }
    if is_pow2(n) {
        let mut buf = a.to_vec();
        radix2(&mut buf, inverse);
        return buf;
    }
    match smallest_factor(n) {
        // Prime length: chirp-z is the only O(n log n) route.
        None => bluestein(a, inverse),
        Some(p) if p > 32 => bluestein(a, inverse),
        Some(p) => {
            // Cooley-Tukey: p sub-transforms of length n/p, recombined with
            // twiddles and a length-p DFT across the sub-results.
            let m = n / p;
            let sign = if inverse { 1.0 } else { -1.0 };
            let mut subs: Vec<Vec<Complex64>> = Vec::with_capacity(p);
            for r in 0..p {
                let part: Vec<Complex64> = (0..m).map(|j| a[j * p + r]).collect();
                subs.push(transform(&part, inverse));
            }
            // The p-point DFT matrix is the same for all m columns, so it is
            // built once here rather than re-deriving p^2 sines and cosines
            // inside the loop below.
            let mut dft_p = vec![Complex64::new(0.0, 0.0); p * p];
            for q in 0..p {
                for r in 0..p {
                    let ang = sign * 2.0 * PI * ((q * r) % p) as f64 / p as f64;
                    dft_p[q * p + r] = Complex64::new(ang.cos(), ang.sin());
                }
            }
            // Twiddles advance by one rotation per column; re-seeding from
            // sin/cos periodically stops the running product from drifting.
            let mut w = vec![Complex64::new(1.0, 0.0); p];
            let mut wstep = vec![Complex64::new(1.0, 0.0); p];
            for (r, ws) in wstep.iter_mut().enumerate() {
                let ang = sign * 2.0 * PI * r as f64 / n as f64;
                *ws = Complex64::new(ang.cos(), ang.sin());
            }
            let mut out = vec![Complex64::new(0.0, 0.0); n];
            let mut col = vec![Complex64::new(0.0, 0.0); p];
            for k in 0..m {
                for r in 0..p {
                    col[r] = subs[r][k] * w[r];
                    w[r] = if k % 64 == 63 {
                        let ang = sign * 2.0 * PI * ((r * (k + 1)) % n) as f64 / n as f64;
                        Complex64::new(ang.cos(), ang.sin())
                    } else {
                        w[r] * wstep[r]
                    };
                }
                for q in 0..p {
                    let row = &dft_p[q * p..q * p + p];
                    let mut acc = Complex64::new(0.0, 0.0);
                    for r in 0..p {
                        acc += col[r] * row[r];
                    }
                    out[q * m + k] = acc;
                }
            }
            out
        }
    }
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
