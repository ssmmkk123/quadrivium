//! Packed, cache-blocked `gemm` with an AVX2/FMA micro-kernel.
//!
//! This is the operation every dense factorization in the crate reduces to, so
//! it is the one place where hand-written SIMD is worth the cost. The structure
//! is the standard Goto/BLIS one:
//!
//! * the `k` and `n` loops cut `B` into `KC x NC` blocks that stay in L3;
//! * the `m` loop cuts `A` into `MC x KC` blocks that stay in L2;
//! * both blocks are *packed* into contiguous, micro-kernel-ordered scratch, so
//!   the innermost loop walks memory linearly with no stride arithmetic;
//! * the micro-kernel holds a `MR x NR` tile of `C` entirely in registers and
//!   streams `A` and `B` past it.
//!
//! Everything is guarded by a runtime AVX2 check with a scalar fallback, so the
//! crate still builds and runs correctly on machines without it.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::prelude::*;

/// Rows of `C` held in registers by the micro-kernel.
pub const MR: usize = 6;
/// Columns of `C` held in registers (two 256-bit lanes of four `f64`).
pub const NR: usize = 8;

/// Cache block sizes. `KC * NR` and `MC * KC` are sized to sit in L1/L2.
const KC: usize = 256;
const MC: usize = 192;
const NC: usize = 4080;

/// Pack a `mc x kc` block of row-major `A` so the micro-kernel reads it as
/// consecutive `MR`-tall columns.
fn pack_a(a: &[f64], lda: usize, mc: usize, kc: usize, out: &mut [f64]) {
    let mut pos = 0;
    let mut i = 0;
    while i < mc {
        let rows = MR.min(mc - i);
        for p in 0..kc {
            for r in 0..rows {
                out[pos + r] = a[(i + r) * lda + p];
            }
            for r in rows..MR {
                out[pos + r] = 0.0;
            }
            pos += MR;
        }
        i += MR;
    }
}

/// Pack a `kc x nc` block of row-major `B` into consecutive `NR`-wide rows.
fn pack_b(b: &[f64], ldb: usize, kc: usize, nc: usize, out: &mut [f64]) {
    let mut pos = 0;
    let mut j = 0;
    while j < nc {
        let cols = NR.min(nc - j);
        for p in 0..kc {
            for c in 0..cols {
                out[pos + c] = b[p * ldb + j + c];
            }
            for c in cols..NR {
                out[pos + c] = 0.0;
            }
            pos += NR;
        }
        j += NR;
    }
}

/// `C[0..MR][0..NR] += alpha * A_pack * B_pack` -- the register-resident kernel.
///
/// # Safety
/// Requires AVX2 and FMA. `a` must hold `kc * MR` values and `b` `kc * NR`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn micro_kernel_avx2(
    kc: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
) {
    unsafe {
        // Twelve accumulators: six rows, two 4-wide lanes each.
        let mut c0 = [_mm256_setzero_pd(); MR];
        let mut c1 = [_mm256_setzero_pd(); MR];
        let mut ap = a.as_ptr();
        let mut bp = b.as_ptr();
        for _ in 0..kc {
            let b0 = _mm256_loadu_pd(bp);
            let b1 = _mm256_loadu_pd(bp.add(4));
            // Unrolled over MR so the broadcasts and FMAs interleave.
            let a0 = _mm256_broadcast_sd(&*ap);
            c0[0] = _mm256_fmadd_pd(a0, b0, c0[0]);
            c1[0] = _mm256_fmadd_pd(a0, b1, c1[0]);
            let a1 = _mm256_broadcast_sd(&*ap.add(1));
            c0[1] = _mm256_fmadd_pd(a1, b0, c0[1]);
            c1[1] = _mm256_fmadd_pd(a1, b1, c1[1]);
            let a2 = _mm256_broadcast_sd(&*ap.add(2));
            c0[2] = _mm256_fmadd_pd(a2, b0, c0[2]);
            c1[2] = _mm256_fmadd_pd(a2, b1, c1[2]);
            let a3 = _mm256_broadcast_sd(&*ap.add(3));
            c0[3] = _mm256_fmadd_pd(a3, b0, c0[3]);
            c1[3] = _mm256_fmadd_pd(a3, b1, c1[3]);
            let a4 = _mm256_broadcast_sd(&*ap.add(4));
            c0[4] = _mm256_fmadd_pd(a4, b0, c0[4]);
            c1[4] = _mm256_fmadd_pd(a4, b1, c1[4]);
            let a5 = _mm256_broadcast_sd(&*ap.add(5));
            c0[5] = _mm256_fmadd_pd(a5, b0, c0[5]);
            c1[5] = _mm256_fmadd_pd(a5, b1, c1[5]);
            ap = ap.add(MR);
            bp = bp.add(NR);
        }
        // Scatter the tile back, scaling by alpha. Edge tiles write only the
        // rows and columns that exist.
        let mut buf = [0.0f64; NR];
        for i in 0..mr {
            _mm256_storeu_pd(buf.as_mut_ptr(), c0[i]);
            _mm256_storeu_pd(buf.as_mut_ptr().add(4), c1[i]);
            let row = &mut c[i * ldc..i * ldc + nr];
            for (j, r) in row.iter_mut().enumerate() {
                *r += alpha * buf[j];
            }
        }
    }
}

/// Portable fallback with the same semantics as the AVX2 kernel.
#[allow(clippy::too_many_arguments)]
fn micro_kernel_scalar(
    kc: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
) {
    let mut acc = [[0.0f64; NR]; MR];
    for p in 0..kc {
        let ablk = &a[p * MR..p * MR + MR];
        let bblk = &b[p * NR..p * NR + NR];
        for i in 0..MR {
            let av = ablk[i];
            for j in 0..NR {
                acc[i][j] += av * bblk[j];
            }
        }
    }
    for i in 0..mr {
        let row = &mut c[i * ldc..i * ldc + nr];
        for (j, r) in row.iter_mut().enumerate() {
            *r += alpha * acc[i][j];
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn micro_kernel(
    kc: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr: usize,
    nr: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the features were just confirmed present, and the packed
            // buffers are sized `kc * MR` and `kc * NR` by their packers.
            unsafe { micro_kernel_avx2(kc, alpha, a, b, c, ldc, mr, nr) };
            return;
        }
    }
    micro_kernel_scalar(kc, alpha, a, b, c, ldc, mr, nr);
}

/// Below this many multiply-adds the rayon fan-out costs more than it saves.
const PAR_MIN_FLOPS: usize = 1 << 21;

/// `C := C + alpha * A * B` for row-major operands.
///
/// `a` is `m x k` with stride `lda`, `b` is `k x n` with stride `ldb`, and `c`
/// is `m x n` with stride `ldc`.
///
/// The `m` loop is the one parallelised: each thread owns a disjoint band of
/// rows of `C`, so no synchronisation is needed inside the kernel and the
/// packed `B` block is shared read-only across all of them.
#[allow(clippy::too_many_arguments)]
pub fn gemm_acc(
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
    if m == 0 || n == 0 || k == 0 || alpha == 0.0 {
        return;
    }
    let threads = rayon::current_num_threads();
    let parallel = m * n * k >= PAR_MIN_FLOPS && threads > 1;
    // Split `m` finely enough that every thread gets a band. With the stock
    // 192-row blocks a 512-row problem would hand work to three threads and
    // leave nine idle.
    let mc_blk = if parallel {
        m.div_ceil(threads).max(MR).next_multiple_of(MR).min(MC)
    } else {
        MC
    };
    // Size the packing scratch to the problem, not to the maximum block: the
    // full-size `B` buffer is 8 MB and allocating it dwarfs a small `gemm`.
    let kc_max = KC.min(k);
    let pa_len = mc_blk.div_ceil(MR) * MR * kc_max;
    let mut pb = vec![0.0f64; NC.min(n).div_ceil(NR) * NR * kc_max];
    let mut pa_serial = vec![0.0f64; pa_len];

    let mut jc = 0;
    while jc < n {
        let nc = NC.min(n - jc);
        let mut pc = 0;
        while pc < k {
            let kc = KC.min(k - pc);
            pack_b(&b[pc * ldb + jc..], ldb, kc, nc, &mut pb);
            let pb_ref: &[f64] = &pb;
            if parallel {
                c.par_chunks_mut(mc_blk * ldc).enumerate().for_each_init(
                    || vec![0.0f64; pa_len],
                    |pa, (blk, cband)| {
                        let ic = blk * mc_blk;
                        if ic >= m {
                            return;
                        }
                        let mc = mc_blk.min(m - ic);
                        pack_a(&a[ic * lda + pc..], lda, mc, kc, pa);
                        band_kernel(mc, nc, kc, alpha, pa, pb_ref, cband, ldc, jc);
                    },
                );
            } else {
                let mut ic = 0;
                while ic < m {
                    let mc = mc_blk.min(m - ic);
                    pack_a(&a[ic * lda + pc..], lda, mc, kc, &mut pa_serial);
                    band_kernel(
                        mc,
                        nc,
                        kc,
                        alpha,
                        &pa_serial,
                        pb_ref,
                        &mut c[ic * ldc..],
                        ldc,
                        jc,
                    );
                    ic += mc_blk;
                }
            }
            pc += kc;
        }
        jc += nc;
    }
}

/// Run the micro-kernel across one `mc x nc` band of `C`.
#[allow(clippy::too_many_arguments)]
fn band_kernel(
    mc: usize,
    nc: usize,
    kc: usize,
    alpha: f64,
    pa: &[f64],
    pb: &[f64],
    cband: &mut [f64],
    ldc: usize,
    jc: usize,
) {
    let mut jr = 0;
    while jr < nc {
        let nr = NR.min(nc - jr);
        let bpanel = &pb[(jr / NR) * NR * kc..];
        let mut ir = 0;
        while ir < mc {
            let mr = MR.min(mc - ir);
            let apanel = &pa[(ir / MR) * MR * kc..];
            let off = ir * ldc + jc + jr;
            micro_kernel(kc, alpha, apanel, bpanel, &mut cband[off..], ldc, mr, nr);
            ir += MR;
        }
        jr += NR;
    }
}

/// Pack a `kc x nc` block of `B` given *transposed* (`nc x kc`, row-major with
/// stride `ldb`), into the same layout `pack_b` produces.
fn pack_b_trans(b: &[f64], ldb: usize, kc: usize, nc: usize, out: &mut [f64]) {
    let mut pos = 0;
    let mut j = 0;
    while j < nc {
        let cols = NR.min(nc - j);
        for p in 0..kc {
            for c in 0..cols {
                out[pos + c] = b[(j + c) * ldb + p];
            }
            for c in cols..NR {
                out[pos + c] = 0.0;
            }
            pos += NR;
        }
        j += NR;
    }
}

/// `C := C + alpha * A * B'` for row-major `A` (m x k) and `B` (n x k).
///
/// Saves materialising `B'`, which for the Cholesky trailing update
/// (`A22 -= L21 L21'`) is an extra copy of the panel on every block.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_acc(
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
    if m == 0 || n == 0 || k == 0 || alpha == 0.0 {
        return;
    }
    let threads = rayon::current_num_threads();
    let parallel = m * n * k >= PAR_MIN_FLOPS && threads > 1;
    let mc_blk = if parallel {
        m.div_ceil(threads).max(MR).next_multiple_of(MR).min(MC)
    } else {
        MC
    };
    let kc_max = KC.min(k);
    let pa_len = mc_blk.div_ceil(MR) * MR * kc_max;
    let mut pb = vec![0.0f64; NC.min(n).div_ceil(NR) * NR * kc_max];
    let mut pa_serial = vec![0.0f64; pa_len];

    let mut jc = 0;
    while jc < n {
        let nc = NC.min(n - jc);
        let mut pc = 0;
        while pc < k {
            let kc = KC.min(k - pc);
            pack_b_trans(&b[jc * ldb + pc..], ldb, kc, nc, &mut pb);
            let pb_ref: &[f64] = &pb;
            if parallel {
                c.par_chunks_mut(mc_blk * ldc).enumerate().for_each_init(
                    || vec![0.0f64; pa_len],
                    |pa, (blk, cband)| {
                        let ic = blk * mc_blk;
                        if ic >= m {
                            return;
                        }
                        let mc = mc_blk.min(m - ic);
                        pack_a(&a[ic * lda + pc..], lda, mc, kc, pa);
                        band_kernel(mc, nc, kc, alpha, pa, pb_ref, cband, ldc, jc);
                    },
                );
            } else {
                let mut ic = 0;
                while ic < m {
                    let mc = mc_blk.min(m - ic);
                    pack_a(&a[ic * lda + pc..], lda, mc, kc, &mut pa_serial);
                    band_kernel(
                        mc,
                        nc,
                        kc,
                        alpha,
                        &pa_serial,
                        pb_ref,
                        &mut c[ic * ldc..],
                        ldc,
                        jc,
                    );
                    ic += mc_blk;
                }
            }
            pc += kc;
        }
        jc += nc;
    }
}

/// `C := A * B`, overwriting `C`.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    for i in 0..m {
        for v in c[i * ldc..i * ldc + n].iter_mut() {
            *v = 0.0;
        }
    }
    gemm_acc(m, n, k, 1.0, a, lda, b, ldb, c, ldc);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn naive(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn matches_naive_across_edge_shapes() {
        // Shapes that exercise partial micro-tiles in both dimensions and
        // more than one cache block along k.
        for &(m, n, k) in &[
            (1, 1, 1),
            (6, 8, 4),
            (5, 7, 3),
            (7, 9, 300),
            (13, 17, 19),
            (64, 64, 64),
            (100, 100, 100),
            (193, 201, 257),
        ] {
            let a = fill(m * k, 1 + m as u64);
            let b = fill(k * n, 2 + n as u64);
            let want = naive(m, n, k, &a, &b);
            let mut got = vec![0.0f64; m * n];
            gemm(m, n, k, &a, k, &b, n, &mut got, n);
            let err = (0..m * n)
                .map(|i| (got[i] - want[i]).abs())
                .fold(0.0f64, f64::max);
            assert!(err < 1e-10, "shape {m}x{n}x{k}: error {err:e}");
        }
    }

    #[test]
    fn accumulates_into_existing_c() {
        let (m, n, k) = (20, 24, 30);
        let a = fill(m * k, 7);
        let b = fill(k * n, 8);
        let base = fill(m * n, 9);
        let want = naive(m, n, k, &a, &b);
        let mut got = base.clone();
        gemm_acc(m, n, k, -2.0, &a, k, &b, n, &mut got, n);
        let err = (0..m * n)
            .map(|i| (got[i] - (base[i] - 2.0 * want[i])).abs())
            .fold(0.0f64, f64::max);
        assert!(err < 1e-10, "accumulate error {err:e}");
    }
}
