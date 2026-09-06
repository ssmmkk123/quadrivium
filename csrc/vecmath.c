/* Vectorised transcendentals.
 *
 * `exp` is the one that shows up in profiles -- kernel density estimates,
 * softmax-like weights, the special functions all run it over whole arrays --
 * and the libm scalar version is slow on the extreme arguments those produce.
 * This is the standard Cody-Waite reduction with a minimax polynomial, in AVX2
 * where the CPU offers it and scalar where it does not, agreeing with libm to
 * within an ulp across the whole range.
 */
#include "qnp.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define QNP_HAVE_X86 1
#endif

#define EXP_OVERFLOW 709.782712893383973096
#define EXP_UNDERFLOW (-745.133219101941108420)
#define LOG2E 1.4426950408889634074
#define LN2_HI 6.93147180369123816490e-01
#define LN2_LO 1.90821492927058770002e-10

/* exp(r) - 1 for |r| <= ln2/2, to better than half an ulp. */
#define EXP_P1 1.66666666666666657415e-01
#define EXP_P2 -2.77777777770155933842e-03
#define EXP_P3 6.61375632143793436117e-05
#define EXP_P4 -1.65339022054652515390e-06
#define EXP_P5 4.13813679705723846039e-08

/* The scalar kernel, and the reference the vector path is checked against:
 * the same algorithm glibc uses for `exp`, written out. */
static inline double exp_kernel(double x) {
    if (!(x > EXP_UNDERFLOW)) return (x != x) ? x : 0.0;
    if (x > EXP_OVERFLOW) return INFINITY;
    double kf = nearbyint(x * LOG2E);
    double r = x - kf * LN2_HI;
    r -= kf * LN2_LO;
    double rr = r * r;
    double c = r - rr * (EXP_P1 + rr * (EXP_P2 + rr * (EXP_P3 + rr * (EXP_P4 + rr * EXP_P5))));
    double value = 1.0 - ((r * c) / (c - 2.0) - r);
    int k = (int)kf;
    /* Split the power of two so a subnormal result still scales correctly. */
    int k1 = k / 2, k2 = k - k1;
    return value * ldexp(1.0, k1) * ldexp(1.0, k2);
}

#ifdef QNP_HAVE_X86
__attribute__((target("avx2,fma")))
static void exp_avx2(const double *src, double *dst, qintp n) {
    const __m256d log2e = _mm256_set1_pd(LOG2E);
    const __m256d ln2_hi = _mm256_set1_pd(LN2_HI);
    const __m256d ln2_lo = _mm256_set1_pd(LN2_LO);
    const __m256d p1 = _mm256_set1_pd(EXP_P1);
    const __m256d p2 = _mm256_set1_pd(EXP_P2);
    const __m256d p3 = _mm256_set1_pd(EXP_P3);
    const __m256d p4 = _mm256_set1_pd(EXP_P4);
    const __m256d p5 = _mm256_set1_pd(EXP_P5);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d lo = _mm256_set1_pd(-700.0);
    const __m256d hi = _mm256_set1_pd(700.0);
    qintp i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d x = _mm256_loadu_pd(src + i);
        /* Anything outside the range where a single power-of-two scaling is
         * exact -- and any NaN -- goes to the scalar kernel. */
        __m256d in_range = _mm256_and_pd(_mm256_cmp_pd(x, lo, _CMP_GT_OQ),
                                         _mm256_cmp_pd(x, hi, _CMP_LT_OQ));
        if (_mm256_movemask_pd(in_range) != 0xF) {
            for (int lane = 0; lane < 4; lane++) dst[i + lane] = exp_kernel(src[i + lane]);
            continue;
        }
        __m256d kf = _mm256_round_pd(_mm256_mul_pd(x, log2e),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d r = _mm256_fnmadd_pd(kf, ln2_hi, x);
        r = _mm256_fnmadd_pd(kf, ln2_lo, r);
        __m256d rr = _mm256_mul_pd(r, r);
        __m256d poly = _mm256_fmadd_pd(rr, p5, p4);
        poly = _mm256_fmadd_pd(rr, poly, p3);
        poly = _mm256_fmadd_pd(rr, poly, p2);
        poly = _mm256_fmadd_pd(rr, poly, p1);
        __m256d c = _mm256_fnmadd_pd(rr, poly, r);
        __m256d value = _mm256_sub_pd(
            one, _mm256_sub_pd(_mm256_div_pd(_mm256_mul_pd(r, c), _mm256_sub_pd(c, two)), r));
        /* 2^k assembled straight into the exponent field. */
        __m128i k32 = _mm256_cvtpd_epi32(kf);
        __m256i k64 = _mm256_cvtepi32_epi64(k32);
        __m256i biased = _mm256_add_epi64(k64, _mm256_set1_epi64x(1023));
        __m256d scale = _mm256_castsi256_pd(_mm256_slli_epi64(biased, 52));
        _mm256_storeu_pd(dst + i, _mm256_mul_pd(value, scale));
    }
    for (; i < n; i++) dst[i] = exp_kernel(src[i]);
}

static int have_avx2_fma(void) {
    static int cached = -1;
    if (cached < 0)
        cached = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return cached;
}
#endif

/* Exponential of `n` contiguous doubles. */
void qnp_exp_f64(const double *src, double *dst, qintp n) {
#ifdef QNP_HAVE_X86
    if (have_avx2_fma()) {
        exp_avx2(src, dst, n);
        return;
    }
#endif
    for (qintp i = 0; i < n; i++) dst[i] = exp_kernel(src[i]);
}
