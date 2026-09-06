/* Element-wise operations: type resolution, loop selection, broadcasting.
 *
 * Every loop comes in two forms from one macro: a unit-stride form the
 * compiler can vectorise, and a general strided form.  The iterator hands the
 * loops the longest contiguous run it can find, so views and broadcasts pay
 * only for the dimensions that actually need walking.
 */
#include "qnp.h"

typedef void (*qbinloop)(char *o, qintp os, const char *a, qintp as,
                         const char *b, qintp bs, qintp n);
typedef void (*qunloop)(char *o, qintp os, const char *a, qintp as, qintp n);

#define BIN_LOOP(NAME, ITYPE, OTYPE, EXPR)                                     \
static void NAME(char *o, qintp os, const char *a, qintp as,                   \
                 const char *b, qintp bs, qintp n) {                           \
    qintp i;                                                                   \
    if (os == (qintp)sizeof(OTYPE) && as == (qintp)sizeof(ITYPE) &&            \
        bs == (qintp)sizeof(ITYPE)) {                                          \
        OTYPE *op = (OTYPE *)o;                                                \
        const ITYPE *ap = (const ITYPE *)a, *bp = (const ITYPE *)b;            \
        for (i = 0; i < n; i++) { ITYPE x = ap[i], y = bp[i]; op[i] = (EXPR); }\
        return;                                                                \
    }                                                                          \
    if (as == 0 && os == (qintp)sizeof(OTYPE) && bs == (qintp)sizeof(ITYPE)) { \
        OTYPE *op = (OTYPE *)o;                                                \
        const ITYPE x = *(const ITYPE *)a;                                     \
        const ITYPE *bp = (const ITYPE *)b;                                    \
        for (i = 0; i < n; i++) { ITYPE y = bp[i]; op[i] = (EXPR); }           \
        return;                                                                \
    }                                                                          \
    if (bs == 0 && os == (qintp)sizeof(OTYPE) && as == (qintp)sizeof(ITYPE)) { \
        OTYPE *op = (OTYPE *)o;                                                \
        const ITYPE y = *(const ITYPE *)b;                                     \
        const ITYPE *ap = (const ITYPE *)a;                                    \
        for (i = 0; i < n; i++) { ITYPE x = ap[i]; op[i] = (EXPR); }           \
        return;                                                                \
    }                                                                          \
    for (i = 0; i < n; i++, o += os, a += as, b += bs) {                       \
        ITYPE x = *(const ITYPE *)a, y = *(const ITYPE *)b;                     \
        *(OTYPE *)o = (EXPR);                                                   \
    }                                                                           \
}

#define UN_LOOP(NAME, ITYPE, OTYPE, EXPR)                                      \
static void NAME(char *o, qintp os, const char *a, qintp as, qintp n) {        \
    qintp i;                                                                   \
    if (os == (qintp)sizeof(OTYPE) && as == (qintp)sizeof(ITYPE)) {            \
        OTYPE *op = (OTYPE *)o;                                                \
        const ITYPE *ap = (const ITYPE *)a;                                    \
        for (i = 0; i < n; i++) { ITYPE x = ap[i]; op[i] = (EXPR); }           \
        return;                                                                \
    }                                                                          \
    for (i = 0; i < n; i++, o += os, a += as) {                                \
        ITYPE x = *(const ITYPE *)a;                                           \
        *(OTYPE *)o = (EXPR);                                                  \
    }                                                                          \
}

typedef unsigned char qbool;

/* ---- complex transcendentals ----------------------------------------- */

qcomplex qc_sqrt(qcomplex a) {
    if (a.re == 0.0 && a.im == 0.0) return qc(0.0, a.im);
    double m = hypot(a.re, a.im);
    double re = sqrt(0.5 * (m + a.re));
    double im = sqrt(0.5 * (m - a.re));
    if (a.im < 0.0) im = -im;
    return qc(re, im);
}
qcomplex qc_exp(qcomplex a) {
    double e = exp(a.re);
    return qc(e * cos(a.im), e * sin(a.im));
}
qcomplex qc_log(qcomplex a) { return qc(log(hypot(a.re, a.im)), atan2(a.im, a.re)); }
qcomplex qc_pow(qcomplex a, qcomplex b) {
    if (a.re == 0.0 && a.im == 0.0) {
        if (b.re == 0.0 && b.im == 0.0) return qc(1.0, 0.0);
        return qc(0.0, 0.0);
    }
    /* Small integer exponents are exact and common; repeated squaring avoids
     * the log/exp round trip and its rounding. */
    if (b.im == 0.0 && b.re == floor(b.re) && fabs(b.re) <= 64.0) {
        long p = (long)b.re;
        int inv = p < 0;
        unsigned long e = (unsigned long)(inv ? -p : p);
        qcomplex r = qc(1.0, 0.0), base = a;
        while (e) {
            if (e & 1UL) r = qc_mul(r, base);
            base = qc_mul(base, base);
            e >>= 1;
        }
        return inv ? qc_div(qc(1.0, 0.0), r) : r;
    }
    return qc_exp(qc_mul(b, qc_log(a)));
}
qcomplex qc_sin(qcomplex a) { return qc(sin(a.re) * cosh(a.im), cos(a.re) * sinh(a.im)); }
qcomplex qc_cos(qcomplex a) { return qc(cos(a.re) * cosh(a.im), -sin(a.re) * sinh(a.im)); }
qcomplex qc_tan(qcomplex a) { return qc_div(qc_sin(a), qc_cos(a)); }
qcomplex qc_sinh(qcomplex a) { return qc(sinh(a.re) * cos(a.im), cosh(a.re) * sin(a.im)); }
qcomplex qc_cosh(qcomplex a) { return qc(cosh(a.re) * cos(a.im), sinh(a.re) * sin(a.im)); }
qcomplex qc_tanh(qcomplex a) { return qc_div(qc_sinh(a), qc_cosh(a)); }

/* The inverse functions follow the principal-branch identities. */
static qcomplex qc_asin(qcomplex a) {
    qcomplex ia = qc(-a.im, a.re);                       /* i*a */
    qcomplex root = qc_sqrt(qc_sub(qc(1.0, 0.0), qc_mul(a, a)));
    qcomplex l = qc_log(qc_add(ia, root));
    return qc(l.im, -l.re);                              /* -i*log(...) */
}
static qcomplex qc_acos(qcomplex a) {
    qcomplex s = qc_asin(a);
    return qc(M_PI / 2.0 - s.re, -s.im);
}
static qcomplex qc_atan(qcomplex a) {
    qcomplex ia = qc(-a.im, a.re);
    qcomplex num = qc_sub(qc(1.0, 0.0), ia);
    qcomplex den = qc_add(qc(1.0, 0.0), ia);
    qcomplex l = qc_log(qc_div(num, den));
    return qc(-0.5 * l.im, 0.5 * l.re);                  /* (i/2)*log(...) */
}
static qcomplex qc_asinh(qcomplex a) {
    return qc_log(qc_add(a, qc_sqrt(qc_add(qc_mul(a, a), qc(1.0, 0.0)))));
}
static qcomplex qc_acosh(qcomplex a) {
    qcomplex p = qc_sqrt(qc_add(a, qc(1.0, 0.0)));
    qcomplex m = qc_sqrt(qc_sub(a, qc(1.0, 0.0)));
    return qc_log(qc_add(a, qc_mul(p, m)));
}
static qcomplex qc_atanh(qcomplex a) {
    qcomplex num = qc_add(qc(1.0, 0.0), a);
    qcomplex den = qc_sub(qc(1.0, 0.0), a);
    qcomplex l = qc_log(qc_div(num, den));
    return qc(0.5 * l.re, 0.5 * l.im);
}
static qcomplex qc_log1p(qcomplex a) { return qc_log(qc_add(qc(1.0, 0.0), a)); }
static qcomplex qc_expm1(qcomplex a) { return qc_sub(qc_exp(a), qc(1.0, 0.0)); }

/* ---- integer helpers with Python's floor/modulo semantics ------------- */

static inline int64_t ifloordiv(int64_t x, int64_t y) {
    if (y == 0) return 0;
    int64_t q = x / y;
    if ((x % y != 0) && ((x < 0) != (y < 0))) q--;
    return q;
}
static inline int64_t imod(int64_t x, int64_t y) {
    if (y == 0) return 0;
    int64_t r = x % y;
    if (r != 0 && ((r < 0) != (y < 0))) r += y;
    return r;
}
static inline double dfloordiv(double x, double y) { return floor(x / y); }
static inline double dmod(double x, double y) {
    double r = fmod(x, y);
    if (r != 0.0 && ((r < 0.0) != (y < 0.0))) r += y;
    return r;
}
static inline int64_t ipow(int64_t x, int64_t y) {
    int64_t r = 1;
    while (y > 0) {
        if (y & 1) r *= x;
        x *= x;
        y >>= 1;
    }
    return r;
}
static inline double dmax(double x, double y) {
    /* NaN propagates, matching np.maximum rather than fmax. */
    if (isnan(x)) return x;
    if (isnan(y)) return y;
    return x > y ? x : y;
}
static inline double dmin(double x, double y) {
    if (isnan(x)) return x;
    if (isnan(y)) return y;
    return x < y ? x : y;
}
static inline int cgt(qcomplex x, qcomplex y) {
    return x.re > y.re || (x.re == y.re && x.im > y.im);
}
static inline double dsign(double x) {
    if (isnan(x)) return x;
    return (x > 0.0) - (x < 0.0);
}

/* ---- arithmetic ------------------------------------------------------- */

BIN_LOOP(add_b, qbool, qbool, x | y)
BIN_LOOP(add_i, int64_t, int64_t, x + y)
BIN_LOOP(add_d, double, double, x + y)
BIN_LOOP(add_c, qcomplex, qcomplex, qc_add(x, y))
BIN_LOOP(sub_i, int64_t, int64_t, x - y)
BIN_LOOP(sub_d, double, double, x - y)
BIN_LOOP(sub_c, qcomplex, qcomplex, qc_sub(x, y))
BIN_LOOP(mul_b, qbool, qbool, x & y)
BIN_LOOP(mul_i, int64_t, int64_t, x * y)
BIN_LOOP(mul_d, double, double, x * y)
BIN_LOOP(mul_c, qcomplex, qcomplex, qc_mul(x, y))
BIN_LOOP(div_d, double, double, x / y)
BIN_LOOP(div_c, qcomplex, qcomplex, qc_div(x, y))
BIN_LOOP(fdiv_i, int64_t, int64_t, ifloordiv(x, y))
BIN_LOOP(fdiv_d, double, double, dfloordiv(x, y))
BIN_LOOP(mod_i, int64_t, int64_t, imod(x, y))
BIN_LOOP(mod_d, double, double, dmod(x, y))
BIN_LOOP(pow_i, int64_t, int64_t, ipow(x, y))
BIN_LOOP(pow_d, double, double, pow(x, y))
BIN_LOOP(pow_c, qcomplex, qcomplex, qc_pow(x, y))
BIN_LOOP(max_b, qbool, qbool, x | y)
BIN_LOOP(max_i, int64_t, int64_t, x > y ? x : y)
BIN_LOOP(max_d, double, double, dmax(x, y))
BIN_LOOP(max_c, qcomplex, qcomplex, cgt(x, y) ? x : y)
BIN_LOOP(min_b, qbool, qbool, x & y)
BIN_LOOP(min_i, int64_t, int64_t, x < y ? x : y)
BIN_LOOP(min_d, double, double, dmin(x, y))
BIN_LOOP(min_c, qcomplex, qcomplex, cgt(x, y) ? y : x)
BIN_LOOP(hypot_d, double, double, hypot(x, y))
BIN_LOOP(atan2_d, double, double, atan2(x, y))
BIN_LOOP(copysign_d, double, double, copysign(x, y))

/* ---- comparisons ------------------------------------------------------ */

#define CMP_SET(OP, SYM)                                                     \
    BIN_LOOP(OP##_b, qbool, qbool, x SYM y)                                  \
    BIN_LOOP(OP##_i, int64_t, qbool, x SYM y)                                \
    BIN_LOOP(OP##_d, double, qbool, x SYM y)
CMP_SET(lt, <)
CMP_SET(le, <=)
CMP_SET(gt, >)
CMP_SET(ge, >=)
#undef CMP_SET
BIN_LOOP(eq_b, qbool, qbool, x == y)
BIN_LOOP(eq_i, int64_t, qbool, x == y)
BIN_LOOP(eq_d, double, qbool, x == y)
BIN_LOOP(eq_c, qcomplex, qbool, x.re == y.re && x.im == y.im)
BIN_LOOP(ne_b, qbool, qbool, x != y)
BIN_LOOP(ne_i, int64_t, qbool, x != y)
BIN_LOOP(ne_d, double, qbool, x != y)
BIN_LOOP(ne_c, qcomplex, qbool, x.re != y.re || x.im != y.im)
BIN_LOOP(lt_c, qcomplex, qbool, x.re < y.re || (x.re == y.re && x.im < y.im))
BIN_LOOP(le_c, qcomplex, qbool, x.re < y.re || (x.re == y.re && x.im <= y.im))
BIN_LOOP(gt_c, qcomplex, qbool, x.re > y.re || (x.re == y.re && x.im > y.im))
BIN_LOOP(ge_c, qcomplex, qbool, x.re > y.re || (x.re == y.re && x.im >= y.im))

BIN_LOOP(and_b, qbool, qbool, x & y)
BIN_LOOP(and_i, int64_t, int64_t, x & y)
BIN_LOOP(or_b, qbool, qbool, x | y)
BIN_LOOP(or_i, int64_t, int64_t, x | y)
BIN_LOOP(xor_b, qbool, qbool, x ^ y)
BIN_LOOP(xor_i, int64_t, int64_t, x ^ y)
BIN_LOOP(lshift_i, int64_t, int64_t, x << y)
BIN_LOOP(rshift_i, int64_t, int64_t, x >> y)
BIN_LOOP(land_b, qbool, qbool, x && y)
BIN_LOOP(land_i, int64_t, qbool, x && y)
BIN_LOOP(land_d, double, qbool, x != 0.0 && y != 0.0)
BIN_LOOP(lor_b, qbool, qbool, x || y)
BIN_LOOP(lor_i, int64_t, qbool, x || y)
BIN_LOOP(lor_d, double, qbool, x != 0.0 || y != 0.0)

/* Loop table: [op][dtype], NULL where the combination is invalid. */
static qbinloop bin_table[QOP_NBINARY][QNP_NTYPES] = {
    [QOP_ADD]      = {add_b, add_i, add_d, add_c},
    [QOP_SUB]      = {NULL, sub_i, sub_d, sub_c},
    [QOP_MUL]      = {mul_b, mul_i, mul_d, mul_c},
    [QOP_TRUEDIV]  = {NULL, NULL, div_d, div_c},
    [QOP_FLOORDIV] = {NULL, fdiv_i, fdiv_d, NULL},
    [QOP_MOD]      = {NULL, mod_i, mod_d, NULL},
    [QOP_POW]      = {NULL, pow_i, pow_d, pow_c},
    [QOP_MAXIMUM]  = {max_b, max_i, max_d, max_c},
    [QOP_MINIMUM]  = {min_b, min_i, min_d, min_c},
    [QOP_HYPOT]    = {NULL, NULL, hypot_d, NULL},
    [QOP_ARCTAN2]  = {NULL, NULL, atan2_d, NULL},
    [QOP_COPYSIGN] = {NULL, NULL, copysign_d, NULL},
    [QOP_EQ]       = {eq_b, eq_i, eq_d, eq_c},
    [QOP_NE]       = {ne_b, ne_i, ne_d, ne_c},
    [QOP_LT]       = {lt_b, lt_i, lt_d, lt_c},
    [QOP_LE]       = {le_b, le_i, le_d, le_c},
    [QOP_GT]       = {gt_b, gt_i, gt_d, gt_c},
    [QOP_GE]       = {ge_b, ge_i, ge_d, ge_c},
    [QOP_AND]      = {and_b, and_i, NULL, NULL},
    [QOP_OR]       = {or_b, or_i, NULL, NULL},
    [QOP_XOR]      = {xor_b, xor_i, NULL, NULL},
    [QOP_LOGICAL_AND] = {land_b, land_i, land_d, NULL},
    [QOP_LOGICAL_OR]  = {lor_b, lor_i, lor_d, NULL},
    [QOP_LSHIFT]      = {NULL, lshift_i, NULL, NULL},
    [QOP_RSHIFT]      = {NULL, rshift_i, NULL, NULL},
};

static const char *bin_names[QOP_NBINARY] = {
    "add", "subtract", "multiply", "true_divide", "floor_divide", "remainder",
    "power", "maximum", "minimum", "hypot", "arctan2", "copysign",
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "bitwise_and", "bitwise_or", "bitwise_xor", "logical_and", "logical_or",
    "left_shift", "right_shift",
};

/* ---- unary loops ------------------------------------------------------ */

UN_LOOP(neg_i, int64_t, int64_t, -x)
UN_LOOP(neg_d, double, double, -x)
UN_LOOP(neg_c, qcomplex, qcomplex, qc_neg(x))
UN_LOOP(abs_b, qbool, qbool, x)
UN_LOOP(abs_i, int64_t, int64_t, x < 0 ? -x : x)
UN_LOOP(abs_d, double, double, fabs(x))
UN_LOOP(abs_c, qcomplex, double, qc_abs(x))
UN_LOOP(sq_i, int64_t, int64_t, x * x)
UN_LOOP(sq_d, double, double, x * x)
UN_LOOP(sq_c, qcomplex, qcomplex, qc_mul(x, x))
UN_LOOP(recip_d, double, double, 1.0 / x)
UN_LOOP(recip_c, qcomplex, qcomplex, qc_div(qc(1.0, 0.0), x))
UN_LOOP(sign_i, int64_t, int64_t, (x > 0) - (x < 0))
UN_LOOP(sign_d, double, double, dsign(x))
UN_LOOP(sign_c, qcomplex, qcomplex,
        (x.re == 0.0 && x.im == 0.0) ? qc(0.0, 0.0)
                                     : qc_div(x, qc(qc_abs(x), 0.0)))
UN_LOOP(conj_c, qcomplex, qcomplex, qc_conj(x))
UN_LOOP(real_c, qcomplex, double, x.re)
UN_LOOP(imag_c, qcomplex, double, x.im)
UN_LOOP(angle_c, qcomplex, double, atan2(x.im, x.re))
UN_LOOP(floor_d, double, double, floor(x))
UN_LOOP(ceil_d, double, double, ceil(x))
UN_LOOP(trunc_d, double, double, trunc(x))
UN_LOOP(rint_d, double, double, nearbyint(x))
UN_LOOP(isfin_i, int64_t, qbool, ((void)x, 1))
UN_LOOP(isfin_d, double, qbool, isfinite(x))
UN_LOOP(isfin_c, qcomplex, qbool, isfinite(x.re) && isfinite(x.im))
UN_LOOP(isnan_i, int64_t, qbool, ((void)x, 0))
UN_LOOP(isnan_d, double, qbool, isnan(x))
UN_LOOP(isnan_c, qcomplex, qbool, isnan(x.re) || isnan(x.im))
UN_LOOP(isinf_i, int64_t, qbool, ((void)x, 0))
UN_LOOP(isinf_d, double, qbool, isinf(x))
UN_LOOP(isinf_c, qcomplex, qbool, isinf(x.re) || isinf(x.im))
UN_LOOP(not_b, qbool, qbool, !x)
UN_LOOP(not_i, int64_t, qbool, !x)
UN_LOOP(not_d, double, qbool, x == 0.0)
UN_LOOP(not_c, qcomplex, qbool, x.re == 0.0 && x.im == 0.0)
UN_LOOP(inv_b, qbool, qbool, !x)
UN_LOOP(inv_i, int64_t, int64_t, ~x)
UN_LOOP(signbit_d, double, qbool, signbit(x))

#define MATH_UNARY(NAME, FN, CFN)                                            \
    UN_LOOP(NAME##_d, double, double, FN(x))                                 \
    UN_LOOP(NAME##_c, qcomplex, qcomplex, CFN(x))
MATH_UNARY(sqrt, sqrt, qc_sqrt)
MATH_UNARY(exp_strided, exp, qc_exp)
/* The unit-stride case goes through the vectorised kernel; other layouts fall
 * back to the strided loop the macro generated. */
static void exp_d(char *o, qintp os, const char *a, qintp as, qintp n) {
    if (os == (qintp)sizeof(double) && as == (qintp)sizeof(double)) {
        qnp_exp_f64((const double *)a, (double *)o, n);
        return;
    }
    exp_strided_d(o, os, a, as, n);
}
MATH_UNARY(log, log, qc_log)
MATH_UNARY(sin, sin, qc_sin)
MATH_UNARY(cos, cos, qc_cos)
MATH_UNARY(tan, tan, qc_tan)
MATH_UNARY(sinh, sinh, qc_sinh)
MATH_UNARY(cosh, cosh, qc_cosh)
MATH_UNARY(tanh, tanh, qc_tanh)
#undef MATH_UNARY
UN_LOOP(log2_d, double, double, log2(x))
UN_LOOP(log10_d, double, double, log10(x))
UN_LOOP(log1p_d, double, double, log1p(x))
UN_LOOP(expm1_d, double, double, expm1(x))
UN_LOOP(asin_d, double, double, asin(x))
UN_LOOP(acos_d, double, double, acos(x))
UN_LOOP(atan_d, double, double, atan(x))
UN_LOOP(asinh_d, double, double, asinh(x))
UN_LOOP(acosh_d, double, double, acosh(x))
UN_LOOP(atanh_d, double, double, atanh(x))
UN_LOOP(log2_c, qcomplex, qcomplex, qc_div(qc_log(x), qc(M_LN2, 0.0)))
UN_LOOP(asin_c, qcomplex, qcomplex, qc_asin(x))
UN_LOOP(acos_c, qcomplex, qcomplex, qc_acos(x))
UN_LOOP(atan_c, qcomplex, qcomplex, qc_atan(x))
UN_LOOP(asinh_c, qcomplex, qcomplex, qc_asinh(x))
UN_LOOP(acosh_c, qcomplex, qcomplex, qc_acosh(x))
UN_LOOP(atanh_c, qcomplex, qcomplex, qc_atanh(x))
UN_LOOP(log1p_c, qcomplex, qcomplex, qc_log1p(x))
UN_LOOP(expm1_c, qcomplex, qcomplex, qc_expm1(x))
UN_LOOP(log10_c, qcomplex, qcomplex, qc_div(qc_log(x), qc(M_LN10, 0.0)))

static qunloop un_table[QOP_NUNARY][QNP_NTYPES] = {
    [QOP_NEG]   = {NULL, neg_i, neg_d, neg_c},
    [QOP_ABS]   = {abs_b, abs_i, abs_d, abs_c},
    [QOP_SQRT]  = {NULL, NULL, sqrt_d, sqrt_c},
    [QOP_EXP]   = {NULL, NULL, exp_d, exp_strided_c},
    [QOP_LOG]   = {NULL, NULL, log_d, log_c},
    [QOP_LOG2]  = {NULL, NULL, log2_d, log2_c},
    [QOP_LOG10] = {NULL, NULL, log10_d, log10_c},
    [QOP_LOG1P] = {NULL, NULL, log1p_d, log1p_c},
    [QOP_EXPM1] = {NULL, NULL, expm1_d, expm1_c},
    [QOP_SIN]   = {NULL, NULL, sin_d, sin_c},
    [QOP_COS]   = {NULL, NULL, cos_d, cos_c},
    [QOP_TAN]   = {NULL, NULL, tan_d, tan_c},
    [QOP_ARCSIN] = {NULL, NULL, asin_d, asin_c},
    [QOP_ARCCOS] = {NULL, NULL, acos_d, acos_c},
    [QOP_ARCTAN] = {NULL, NULL, atan_d, atan_c},
    [QOP_SINH]  = {NULL, NULL, sinh_d, sinh_c},
    [QOP_COSH]  = {NULL, NULL, cosh_d, cosh_c},
    [QOP_TANH]  = {NULL, NULL, tanh_d, tanh_c},
    [QOP_ARCSINH] = {NULL, NULL, asinh_d, asinh_c},
    [QOP_ARCCOSH] = {NULL, NULL, acosh_d, acosh_c},
    [QOP_ARCTANH] = {NULL, NULL, atanh_d, atanh_c},
    [QOP_SIGN]  = {NULL, sign_i, sign_d, sign_c},
    [QOP_FLOOR] = {NULL, NULL, floor_d, NULL},
    [QOP_CEIL]  = {NULL, NULL, ceil_d, NULL},
    [QOP_TRUNC] = {NULL, NULL, trunc_d, NULL},
    [QOP_RINT]  = {NULL, NULL, rint_d, NULL},
    [QOP_SQUARE] = {NULL, sq_i, sq_d, sq_c},
    [QOP_RECIPROCAL] = {NULL, NULL, recip_d, recip_c},
    [QOP_CONJ]  = {NULL, NULL, NULL, conj_c},
    [QOP_REAL]  = {NULL, NULL, NULL, real_c},
    [QOP_IMAG]  = {NULL, NULL, NULL, imag_c},
    [QOP_ANGLE] = {NULL, NULL, NULL, angle_c},
    [QOP_ISFINITE] = {NULL, isfin_i, isfin_d, isfin_c},
    [QOP_ISNAN] = {NULL, isnan_i, isnan_d, isnan_c},
    [QOP_ISINF] = {NULL, isinf_i, isinf_d, isinf_c},
    [QOP_NOT]   = {not_b, not_i, not_d, not_c},
    [QOP_INVERT] = {inv_b, inv_i, NULL, NULL},
    [QOP_SIGNBIT] = {NULL, NULL, signbit_d, NULL},
    [QOP_POS]   = {NULL, NULL, NULL, NULL},
};

static const char *un_names[QOP_NUNARY] = {
    "negative", "positive", "absolute", "sqrt", "exp", "log", "log2", "log10",
    "log1p", "expm1", "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "sign", "floor",
    "ceil", "trunc", "rint", "square", "reciprocal", "conjugate", "real",
    "imag", "angle", "isfinite", "isnan", "isinf", "logical_not", "invert",
    "signbit",
};

/* ---- type resolution -------------------------------------------------- */

static int is_comparison(int op) { return op >= QOP_EQ && op <= QOP_GE; }

/* Chooses the dtype the loop runs in and the dtype it produces. */
static int resolve_binary(int op, int dta, int dtb, int *in_dt, int *out_dt) {
    int common = qnp_promote(dta, dtb);
    switch (op) {
        case QOP_TRUEDIV:
            *in_dt = (common <= QNP_FLOAT64) ? QNP_FLOAT64 : QNP_COMPLEX128;
            *out_dt = *in_dt;
            return 0;
        case QOP_SUB: case QOP_FLOORDIV: case QOP_MOD: case QOP_POW:
            *in_dt = (common == QNP_BOOL) ? QNP_INT64 : common;
            *out_dt = *in_dt;
            return 0;
        case QOP_HYPOT: case QOP_ARCTAN2: case QOP_COPYSIGN:
            if (common == QNP_COMPLEX128) { *in_dt = -1; return -1; }
            *in_dt = QNP_FLOAT64;
            *out_dt = QNP_FLOAT64;
            return 0;
        case QOP_AND: case QOP_OR: case QOP_XOR:
            if (common > QNP_INT64) { *in_dt = -1; return -1; }
            *in_dt = common;
            *out_dt = common;
            return 0;
        case QOP_LSHIFT: case QOP_RSHIFT:
            if (common > QNP_INT64) { *in_dt = -1; return -1; }
            *in_dt = QNP_INT64;
            *out_dt = QNP_INT64;
            return 0;
        case QOP_LOGICAL_AND: case QOP_LOGICAL_OR:
            *in_dt = common;
            *out_dt = QNP_BOOL;
            return 0;
        default:
            *in_dt = common;
            *out_dt = is_comparison(op) ? QNP_BOOL : common;
            return 0;
    }
}

/* ---- driver ----------------------------------------------------------- */

static QArray *as_operand(PyObject *obj, int *weak) {
    *weak = 0;
    if (QArray_Check(obj)) { Py_INCREF(obj); return (QArray *)obj; }
    int w;
    if (qnp_scalar_dtype(obj, &w) >= 0) *weak = 1;
    return qnp_from_any(obj, -1, 0);
}

static int check_out(QArray *out, int natural, const char *name) {
    if (out->dtype < natural) {
        PyErr_Format(PyExc_TypeError,
                     "Cannot cast ufunc '%s' output from dtype('%s') to dtype('%s') "
                     "with casting rule 'same_kind'",
                     name,
                     natural == QNP_COMPLEX128 ? "complex128" :
                     natural == QNP_FLOAT64 ? "float64" :
                     natural == QNP_INT64 ? "int64" : "bool",
                     out->dtype == QNP_COMPLEX128 ? "complex128" :
                     out->dtype == QNP_FLOAT64 ? "float64" :
                     out->dtype == QNP_INT64 ? "int64" : "bool");
        return -1;
    }
    if (!(out->flags & QNP_WRITEABLE)) {
        PyErr_SetString(PyExc_ValueError, "output array is read-only");
        return -1;
    }
    return 0;
}

/* Copies `src` into `dst` only where `mask` is true. */
static int copy_where(QArray *dst, QArray *src, QArray *mask) {
    QArray *ops[3] = {dst, src, mask};
    qintp shape[QNP_MAXDIMS];
    int nd;
    if (qnp_broadcast_shapes(3, ops, shape, &nd) < 0) return -1;
    if (nd != dst->nd || memcmp(shape, dst->shape, (size_t)nd * sizeof(qintp))) {
        PyErr_SetString(PyExc_ValueError, "where mask does not fit the output shape");
        return -1;
    }
    QIter it;
    if (qnp_iter_init(&it, 3, ops, shape, nd) < 0) return -1;
    int isz = QNP_ITEMSIZE(dst->dtype);
    while (qnp_iter_next(&it)) {
        char *d = it.ptr[0];
        const char *s = it.ptr[1];
        const char *m = it.ptr[2];
        for (qintp i = 0; i < it.inner_len; i++) {
            if (*(const qbool *)m) memcpy(d, s, (size_t)isz);
            d += it.inner_stride[0];
            s += it.inner_stride[1];
            m += it.inner_stride[2];
        }
    }
    return 0;
}

static PyObject *finish(QArray *result, PyObject *out) {
    if (out != NULL && out != Py_None) {
        Py_INCREF(out);
        Py_DECREF(result);
        return out;
    }
    return qnp_wrap_scalar_or_array(result);
}

/* Recognises `x ** k` for the handful of exponents worth special-casing. */
static int power_shortcut(QArray *b, double *k) {
    if (qnp_size(b) != 1 || b->dtype == QNP_COMPLEX128) return 0;
    double v;
    switch (b->dtype) {
        case QNP_BOOL: v = *(qbool *)b->data; break;
        case QNP_INT64: v = (double)*(int64_t *)b->data; break;
        default: v = *(double *)b->data; break;
    }
    *k = v;
    return v == 2.0 || v == 0.5 || v == 1.0 || v == -1.0 || v == 3.0;
}

PyObject *qnp_binary_op(int op, PyObject *ao, PyObject *bo, PyObject *outo,
                        PyObject *whereo) {
    int weak_a, weak_b;
    QArray *a = as_operand(ao, &weak_a);
    if (a == NULL) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
        return NULL;
    }
    QArray *b = as_operand(bo, &weak_b);
    if (b == NULL) {
        Py_DECREF(a);
        if (PyErr_ExceptionMatches(PyExc_TypeError)) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
        return NULL;
    }
    if (op == QOP_POW) {
        double k;
        if (power_shortcut(b, &k) && a->dtype != QNP_BOOL) {
            int sub = (k == 2.0) ? QOP_SQUARE : (k == 0.5) ? QOP_SQRT
                    : (k == -1.0) ? QOP_RECIPROCAL : -1;
            if (k == 0.5 && a->dtype == QNP_INT64) sub = QOP_SQRT;
            if (k == -1.0 && a->dtype == QNP_INT64) sub = -1;
            if (sub >= 0) {
                PyObject *r = qnp_unary_op(sub, (PyObject *)a, outo);
                Py_DECREF(a); Py_DECREF(b);
                return r;
            }
        }
    }
    int in_dt, out_dt;
    if (resolve_binary(op, a->dtype, b->dtype, &in_dt, &out_dt) < 0 ||
        bin_table[op][in_dt] == NULL) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc '%s' not supported for the input types", bin_names[op]);
        Py_DECREF(a); Py_DECREF(b);
        return NULL;
    }
    if (op == QOP_POW && in_dt == QNP_INT64) {
        /* NumPy refuses integer powers of negative exponent rather than
         * silently truncating the reciprocal to zero. */
        qintp n = qnp_size(b);
        const char *p = b->data;
        QArray *cb = qnp_ascontiguous(b);
        if (cb == NULL) { Py_DECREF(a); Py_DECREF(b); return NULL; }
        p = cb->data;
        for (qintp i = 0; i < n; i++) {
            if (((const int64_t *)p)[i] < 0) {
                Py_DECREF(cb); Py_DECREF(a); Py_DECREF(b);
                PyErr_SetString(PyExc_ValueError,
                                "Integers to negative integer powers are not allowed.");
                return NULL;
            }
        }
        Py_DECREF(cb);
    }
    QArray *ca = qnp_astype(a, in_dt, 0);
    QArray *cb = qnp_astype(b, in_dt, 0);
    Py_DECREF(a); Py_DECREF(b);
    if (ca == NULL || cb == NULL) { Py_XDECREF(ca); Py_XDECREF(cb); return NULL; }

    qintp shape[QNP_MAXDIMS];
    int nd;
    QArray *pair[2] = {ca, cb};
    if (qnp_broadcast_shapes(2, pair, shape, &nd) < 0) { Py_DECREF(ca); Py_DECREF(cb); return NULL; }

    QArray *out = NULL;
    int use_out_directly = 0;
    QArray *where = NULL;
    if (whereo != NULL && whereo != Py_None && whereo != Py_True) {
        where = qnp_from_any(whereo, QNP_BOOL, 1);
        if (where == NULL) { Py_DECREF(ca); Py_DECREF(cb); return NULL; }
    }
    if (outo != NULL && outo != Py_None) {
        if (!QArray_Check(outo)) {
            PyErr_SetString(PyExc_TypeError, "output must be an array");
            Py_DECREF(ca); Py_DECREF(cb); Py_XDECREF(where);
            return NULL;
        }
        QArray *dest = (QArray *)outo;
        if (check_out(dest, out_dt, bin_names[op]) < 0) {
            Py_DECREF(ca); Py_DECREF(cb); Py_XDECREF(where);
            return NULL;
        }
        if (dest->nd == nd && !memcmp(dest->shape, shape, (size_t)nd * sizeof(qintp)) &&
            dest->dtype == out_dt && where == NULL) {
            out = dest;
            Py_INCREF(out);
            use_out_directly = 1;
        }
    }
    if (out == NULL) {
        out = qnp_new(nd, shape, out_dt);
        if (out == NULL) { Py_DECREF(ca); Py_DECREF(cb); Py_XDECREF(where); return NULL; }
    }

    QArray *ops[3] = {out, ca, cb};
    QIter it;
    if (qnp_iter_init(&it, 3, ops, shape, nd) < 0) {
        Py_DECREF(ca); Py_DECREF(cb); Py_DECREF(out); Py_XDECREF(where);
        return NULL;
    }
    qbinloop loop = bin_table[op][in_dt];
    while (qnp_iter_next(&it))
        loop(it.ptr[0], it.inner_stride[0], it.ptr[1], it.inner_stride[1],
             it.ptr[2], it.inner_stride[2], it.inner_len);
    Py_DECREF(ca);
    Py_DECREF(cb);

    if (!use_out_directly && outo != NULL && outo != Py_None) {
        QArray *dest = (QArray *)outo;
        int rc = where != NULL ? copy_where(dest, out, where) : qnp_copy_into(dest, out);
        Py_DECREF(out);
        Py_XDECREF(where);
        if (rc < 0) return NULL;
        Py_INCREF(dest);
        return (PyObject *)dest;
    }
    Py_XDECREF(where);
    return finish(out, outo);
}

/* NumPy applies an object-dtype ufunc by calling the method of the same name
 * on the element -- which is how duck-typed numbers such as this package's
 * dual numbers ride through `np.sin`.  There is no object dtype here, so the
 * dispatch happens up front, and only for operands that are not numeric. */
static PyObject *object_unary_dispatch(int op, PyObject *obj) {
    if (QArray_Check(obj) || PyFloat_Check(obj) || PyLong_Check(obj) ||
        PyComplex_Check(obj) || PyBool_Check(obj) || PyList_Check(obj) ||
        PyTuple_Check(obj) || PyObject_CheckBuffer(obj))
        return NULL;
    PyObject *method = PyObject_GetAttrString(obj, un_names[op]);
    if (method == NULL) { PyErr_Clear(); return NULL; }
    PyObject *result = PyObject_CallNoArgs(method);
    Py_DECREF(method);
    return result;
}

PyObject *qnp_unary_op(int op, PyObject *ao, PyObject *outo) {
    if (!QArray_Check(ao)) {
        PyObject *delegated = object_unary_dispatch(op, ao);
        if (delegated != NULL) return delegated;
        if (PyErr_Occurred()) return NULL;
    }
    QArray *a = qnp_from_any(ao, -1, 0);
    if (a == NULL) return NULL;
    int in_dt = a->dtype;
    int out_dt;
    switch (op) {
        case QOP_POS: {
            QArray *r = qnp_astype(a, a->dtype, 1);
            Py_DECREF(a);
            if (r == NULL) return NULL;
            return finish(r, outo);
        }
        case QOP_CONJ: case QOP_REAL:
            if (in_dt != QNP_COMPLEX128) {
                QArray *r = qnp_astype(a, in_dt, 1);
                Py_DECREF(a);
                if (r == NULL) return NULL;
                return finish(r, outo);
            }
            out_dt = (op == QOP_CONJ) ? QNP_COMPLEX128 : QNP_FLOAT64;
            break;
        case QOP_IMAG: case QOP_ANGLE:
            if (in_dt != QNP_COMPLEX128) {
                if (op == QOP_IMAG) {
                    QArray *r = qnp_new(a->nd, a->shape, in_dt);
                    Py_DECREF(a);
                    if (r == NULL) return NULL;
                    memset(r->data, 0, (size_t)qnp_size(r) * (size_t)QNP_ITEMSIZE(in_dt));
                    return finish(r, outo);
                }
                in_dt = QNP_FLOAT64;   /* angle of a real number is 0 or pi */
                QArray *f = qnp_astype(a, QNP_FLOAT64, 0);
                Py_DECREF(a);
                if (f == NULL) return NULL;
                a = f;
                qintp n = qnp_size(a);
                QArray *ca = qnp_ascontiguous(a);
                Py_DECREF(a);
                if (ca == NULL) return NULL;
                QArray *r = qnp_new(ca->nd, ca->shape, QNP_FLOAT64);
                if (r == NULL) { Py_DECREF(ca); return NULL; }
                const double *src = (const double *)ca->data;
                double *dst = (double *)r->data;
                for (qintp i = 0; i < n; i++) dst[i] = src[i] < 0 ? M_PI : 0.0;
                Py_DECREF(ca);
                return finish(r, outo);
            }
            out_dt = QNP_FLOAT64;
            break;
        case QOP_ABS:
            out_dt = (in_dt == QNP_COMPLEX128) ? QNP_FLOAT64 : in_dt;
            break;
        case QOP_ISFINITE: case QOP_ISNAN: case QOP_ISINF: case QOP_NOT:
        case QOP_SIGNBIT:
            if (op == QOP_SIGNBIT && in_dt < QNP_FLOAT64) in_dt = QNP_FLOAT64;
            if (op != QOP_NOT && in_dt == QNP_BOOL) in_dt = QNP_INT64;
            out_dt = QNP_BOOL;
            break;
        case QOP_NEG: case QOP_INVERT:
            if (op == QOP_NEG && in_dt == QNP_BOOL) in_dt = QNP_INT64;
            out_dt = in_dt;
            break;
        case QOP_FLOOR: case QOP_CEIL: case QOP_TRUNC: case QOP_RINT:
            if (in_dt <= QNP_INT64) {
                QArray *r = qnp_astype(a, in_dt, 1);
                Py_DECREF(a);
                if (r == NULL) return NULL;
                return finish(r, outo);
            }
            out_dt = in_dt;
            break;
        case QOP_SIGN: case QOP_SQUARE:
            if (in_dt == QNP_BOOL) in_dt = QNP_INT64;
            out_dt = in_dt;
            break;
        default:
            if (in_dt < QNP_FLOAT64) in_dt = QNP_FLOAT64;
            out_dt = in_dt;
            break;
    }
    if (un_table[op][in_dt] == NULL) {
        PyErr_Format(PyExc_TypeError,
                     "ufunc '%s' not supported for the input type", un_names[op]);
        Py_DECREF(a);
        return NULL;
    }
    QArray *ca = qnp_astype(a, in_dt, 0);
    Py_DECREF(a);
    if (ca == NULL) return NULL;

    QArray *out = NULL;
    int use_out_directly = 0;
    if (outo != NULL && outo != Py_None) {
        if (!QArray_Check(outo)) {
            PyErr_SetString(PyExc_TypeError, "output must be an array");
            Py_DECREF(ca);
            return NULL;
        }
        QArray *dest = (QArray *)outo;
        if (check_out(dest, out_dt, un_names[op]) < 0) { Py_DECREF(ca); return NULL; }
        if (dest->nd == ca->nd && dest->dtype == out_dt &&
            !memcmp(dest->shape, ca->shape, (size_t)ca->nd * sizeof(qintp))) {
            out = dest;
            Py_INCREF(out);
            use_out_directly = 1;
        }
    }
    if (out == NULL) {
        out = qnp_new(ca->nd, ca->shape, out_dt);
        if (out == NULL) { Py_DECREF(ca); return NULL; }
    }
    QArray *ops[2] = {out, ca};
    QIter it;
    if (qnp_iter_init(&it, 2, ops, ca->shape, ca->nd) < 0) {
        Py_DECREF(ca); Py_DECREF(out);
        return NULL;
    }
    qunloop loop = un_table[op][in_dt];
    while (qnp_iter_next(&it))
        loop(it.ptr[0], it.inner_stride[0], it.ptr[1], it.inner_stride[1], it.inner_len);
    Py_DECREF(ca);
    if (!use_out_directly && outo != NULL && outo != Py_None) {
        QArray *dest = (QArray *)outo;
        int rc = qnp_copy_into(dest, out);
        Py_DECREF(out);
        if (rc < 0) return NULL;
        Py_INCREF(dest);
        return (PyObject *)dest;
    }
    return finish(out, outo);
}

/* ---- np.where --------------------------------------------------------- */

PyObject *qnp_where3(PyObject *condo, PyObject *xo, PyObject *yo) {
    QArray *cond = qnp_from_any(condo, QNP_BOOL, 1);
    if (cond == NULL) return NULL;
    QArray *x = qnp_from_any(xo, -1, 0);
    QArray *y = qnp_from_any(yo, -1, 0);
    if (x == NULL || y == NULL) { Py_DECREF(cond); Py_XDECREF(x); Py_XDECREF(y); return NULL; }
    int dt = qnp_promote(x->dtype, y->dtype);
    QArray *cx = qnp_astype(x, dt, 0);
    QArray *cy = qnp_astype(y, dt, 0);
    Py_DECREF(x); Py_DECREF(y);
    if (cx == NULL || cy == NULL) {
        Py_DECREF(cond); Py_XDECREF(cx); Py_XDECREF(cy);
        return NULL;
    }
    QArray *ops[4] = {NULL, cond, cx, cy};
    QArray *inputs[3] = {cond, cx, cy};
    qintp shape[QNP_MAXDIMS];
    int nd;
    if (qnp_broadcast_shapes(3, inputs, shape, &nd) < 0) {
        Py_DECREF(cond); Py_DECREF(cx); Py_DECREF(cy);
        return NULL;
    }
    QArray *out = qnp_new(nd, shape, dt);
    if (out == NULL) { Py_DECREF(cond); Py_DECREF(cx); Py_DECREF(cy); return NULL; }
    ops[0] = out;
    QIter it;
    if (qnp_iter_init(&it, 4, ops, shape, nd) < 0) {
        Py_DECREF(cond); Py_DECREF(cx); Py_DECREF(cy); Py_DECREF(out);
        return NULL;
    }
    int isz = QNP_ITEMSIZE(dt);
    while (qnp_iter_next(&it)) {
        char *o = it.ptr[0];
        const char *c = it.ptr[1], *p = it.ptr[2], *q = it.ptr[3];
        for (qintp i = 0; i < it.inner_len; i++) {
            memcpy(o, *(const qbool *)c ? p : q, (size_t)isz);
            o += it.inner_stride[0];
            c += it.inner_stride[1];
            p += it.inner_stride[2];
            q += it.inner_stride[3];
        }
    }
    Py_DECREF(cond); Py_DECREF(cx); Py_DECREF(cy);
    return qnp_wrap_scalar_or_array(out);
}
