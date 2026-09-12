/* Pseudo-random generation.
 *
 * PCG64 with NumPy's SeedSequence, its ziggurat samplers and its bounded
 * integer scheme, all reproduced exactly: a given seed yields the same stream
 * of numbers this library produced when it was built on NumPy, so seeded
 * examples, tests and published results carry over unchanged.
 */
#include "qnp.h"
#include "ziggurat_tables.h"
#include <float.h>

#if defined(__SIZEOF_INT128__) && !defined(QNP_FORCE_PORTABLE_UINT128)
typedef unsigned __int128 qu128;
static inline qu128 u128_make(uint64_t hi, uint64_t lo) {
    return ((qu128)hi << 64) | lo;
}
static inline uint64_t u128_hi(qu128 v) { return (uint64_t)(v >> 64); }
static inline uint64_t u128_lo(qu128 v) { return (uint64_t)v; }
static inline qu128 u128_add(qu128 a, qu128 b) { return a + b; }
static inline qu128 u128_mul(qu128 a, qu128 b) { return a * b; }
static inline qu128 u128_mul64(uint64_t a, uint64_t b) { return (qu128)a * b; }
#else
/* MSVC has no native unsigned __int128.  Unsigned limb arithmetic preserves
 * the exact PCG64 stream and serialized state, including modulo-2^128 wrap. */
typedef struct { uint64_t hi, lo; } qu128;
static inline qu128 u128_make(uint64_t hi, uint64_t lo) {
    qu128 v = {hi, lo};
    return v;
}
static inline uint64_t u128_hi(qu128 v) { return v.hi; }
static inline uint64_t u128_lo(qu128 v) { return v.lo; }
static inline qu128 u128_add(qu128 a, qu128 b) {
    uint64_t lo = a.lo + b.lo;
    return u128_make(a.hi + b.hi + (lo < a.lo), lo);
}
static inline qu128 u128_mul64(uint64_t a, uint64_t b) {
    const uint64_t mask = UINT32_MAX;
    uint64_t a0 = a & mask, a1 = a >> 32;
    uint64_t b0 = b & mask, b1 = b >> 32;
    uint64_t w0 = a0 * b0;
    uint64_t t = a1 * b0 + (w0 >> 32);
    uint64_t w1 = (t & mask) + a0 * b1;
    uint64_t hi = a1 * b1 + (t >> 32) + (w1 >> 32);
    return u128_make(hi, (w1 << 32) | (w0 & mask));
}
static inline qu128 u128_mul(qu128 a, qu128 b) {
    qu128 product = u128_mul64(a.lo, b.lo);
    product.hi += a.hi * b.lo + a.lo * b.hi;
    return product;
}
#endif

typedef unsigned char qbool;

/* ---- SeedSequence ------------------------------------------------------ */

#define INIT_A 0x43b0d7e5u
#define MULT_A 0x931e8875u
#define INIT_B 0x8b51f9ddu
#define MULT_B 0x58f38dedu
#define MIX_MULT_L 0xca01f9ddu
#define MIX_MULT_R 0x4973f715u
#define XSHIFT 16

static uint32_t hashmix(uint32_t value, uint32_t *hash_const) {
    value ^= *hash_const;
    *hash_const *= MULT_A;
    value *= *hash_const;
    value ^= value >> XSHIFT;
    return value;
}

static uint32_t mix_words(uint32_t x, uint32_t y) {
    uint32_t result = MIX_MULT_L * x - MIX_MULT_R * y;
    result ^= result >> XSHIFT;
    return result;
}

static void mix_entropy(uint32_t *pool, const uint32_t *entropy, size_t n) {
    uint32_t hash_const = INIT_A;
    for (int i = 0; i < 4; i++)
        pool[i] = hashmix(i < (int)n ? entropy[i] : 0u, &hash_const);
    for (int src = 0; src < 4; src++)
        for (int dst = 0; dst < 4; dst++)
            if (src != dst) pool[dst] = mix_words(pool[dst], hashmix(pool[src], &hash_const));
    for (size_t src = 4; src < n; src++)
        for (int dst = 0; dst < 4; dst++)
            pool[dst] = mix_words(pool[dst], hashmix(entropy[src], &hash_const));
}

static void generate_state(const uint32_t *pool, uint32_t *out, int nwords) {
    uint32_t hash_const = INIT_B;
    for (int i = 0; i < nwords; i++) {
        uint32_t value = pool[i % 4];
        value ^= hash_const;
        hash_const *= MULT_B;
        value *= hash_const;
        value ^= value >> XSHIFT;
        out[i] = value;
    }
}

/* ---- PCG64 ------------------------------------------------------------- */

#define PCG_MULT_HI 0x2360ED051FC65DA4ULL
#define PCG_MULT_LO 0x4385DF649FCCF645ULL

typedef struct {
    qu128 state;
    qu128 inc;
    int has_uint32;
    uint32_t uinteger;
} PCG64;

static inline void pcg_step(PCG64 *rng) {
    qu128 mult = u128_make(PCG_MULT_HI, PCG_MULT_LO);
    rng->state = u128_add(u128_mul(rng->state, mult), rng->inc);
}

static inline uint64_t rotr64(uint64_t v, unsigned r) {
    return (v >> r) | (v << ((-r) & 63));
}

static inline uint64_t pcg_next64(PCG64 *rng) {
    pcg_step(rng);
    qu128 s = rng->state;
    uint64_t value = u128_hi(s) ^ u128_lo(s);
    unsigned rot = (unsigned)(u128_hi(s) >> 58);
    return rotr64(value, rot);
}

static inline uint32_t pcg_next32(PCG64 *rng) {
    if (rng->has_uint32) {
        rng->has_uint32 = 0;
        return rng->uinteger;
    }
    uint64_t next = pcg_next64(rng);
    rng->has_uint32 = 1;
    rng->uinteger = (uint32_t)(next >> 32);
    return (uint32_t)next;
}

static inline double pcg_next_double(PCG64 *rng) {
    return (double)(pcg_next64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static void pcg_seed(PCG64 *rng, const uint32_t *pool) {
    uint32_t words[8];
    generate_state(pool, words, 8);
    uint64_t u64[4];
    for (int i = 0; i < 4; i++)
        u64[i] = (uint64_t)words[2 * i] | ((uint64_t)words[2 * i + 1] << 32);
    qu128 initstate = u128_make(u64[0], u64[1]);
    rng->state = u128_make(0, 0);
    rng->inc = u128_make((u64[2] << 1) | (u64[3] >> 63), (u64[3] << 1) | 1u);
    pcg_step(rng);
    rng->state = u128_add(rng->state, initstate);
    pcg_step(rng);
    rng->has_uint32 = 0;
    rng->uinteger = 0;
}

/* ---- distributions ----------------------------------------------------- */

static double standard_normal_one(PCG64 *rng) {
    while (1) {
        uint64_t r = pcg_next64(rng);
        int idx = (int)(r & 0xff);
        r >>= 8;
        uint64_t sign = r & 0x1;
        uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
        double x = (double)rabs * wi_double[idx];
        if (sign) x = -x;
        if (rabs < ki_double[idx]) return x;
        if (idx == 0) {
            for (;;) {
                double xx = -ZIGGURAT_NOR_INV_R * log1p(-pcg_next_double(rng));
                double yy = -log1p(-pcg_next_double(rng));
                if (yy + yy > xx * xx)
                    return ((rabs >> 8) & 0x1) ? -(ZIGGURAT_NOR_R + xx)
                                               : ZIGGURAT_NOR_R + xx;
            }
        }
        if ((fi_double[idx - 1] - fi_double[idx]) * pcg_next_double(rng) + fi_double[idx]
            < exp(-0.5 * x * x))
            return x;
    }
}

static double standard_exponential_one(PCG64 *rng) {
    while (1) {
        uint64_t ri = pcg_next64(rng) >> 3;
        int idx = (int)(ri & 0xff);
        ri >>= 8;
        double x = (double)ri * we_double[idx];
        if (ri < ke_double[idx]) return x;
        if (idx == 0) return ZIGGURAT_EXP_R - log1p(-pcg_next_double(rng));
        if ((fe_double[idx - 1] - fe_double[idx]) * pcg_next_double(rng) + fe_double[idx]
            < exp(-x))
            return x;
    }
}

/* Lemire's method, which is what NumPy's `Generator` uses for bounded
 * integers (the masked rejection below it belongs to the legacy RandomState).
 * Reproducing the choice matters: it decides how many words each draw
 * consumes, and so the whole downstream stream. */
static uint32_t lemire_uint32(PCG64 *rng, uint32_t range) {
    const uint32_t range_excl = range + 1;
    uint64_t m = (uint64_t)pcg_next32(rng) * range_excl;
    uint32_t leftover = (uint32_t)m;
    if (leftover < range_excl) {
        const uint32_t threshold = (uint32_t)((0xFFFFFFFFu - range) % range_excl);
        while (leftover < threshold) {
            m = (uint64_t)pcg_next32(rng) * range_excl;
            leftover = (uint32_t)m;
        }
    }
    return (uint32_t)(m >> 32);
}

static uint64_t lemire_uint64(PCG64 *rng, uint64_t range) {
    const uint64_t range_excl = range + 1;
    qu128 m = u128_mul64(pcg_next64(rng), range_excl);
    uint64_t leftover = u128_lo(m);
    if (leftover < range_excl) {
        const uint64_t threshold = (0xFFFFFFFFFFFFFFFFULL - range) % range_excl;
        while (leftover < threshold) {
            m = u128_mul64(pcg_next64(rng), range_excl);
            leftover = u128_lo(m);
        }
    }
    return u128_hi(m);
}

/* Masked rejection.  NumPy keeps this one for `shuffle`, so the two draw
 * paths have to stay distinct even though both are "a bounded integer". */
static uint64_t random_interval(PCG64 *rng, uint64_t max) {
    if (max == 0) return 0;
    uint64_t mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    uint64_t value;
    if (max <= 0xFFFFFFFFULL) {
        while ((value = (uint64_t)(pcg_next32(rng) & (uint32_t)mask)) > max) {}
    } else {
        while ((value = pcg_next64(rng) & mask) > max) {}
    }
    return value;
}

static uint64_t bounded_uint64(PCG64 *rng, uint64_t range) {
    if (range == 0) return 0;
    if (range <= 0xFFFFFFFFULL) {
        if (range == 0xFFFFFFFFULL) return pcg_next32(rng);
        return lemire_uint32(rng, (uint32_t)range);
    }
    if (range == 0xFFFFFFFFFFFFFFFFULL) return pcg_next64(rng);
    return lemire_uint64(rng, range);
}

static double poisson_mult(PCG64 *rng, double lam) {
    double enlam = exp(-lam), prod = 1.0;
    int64_t x = 0;
    while (1) {
        prod *= pcg_next_double(rng);
        if (prod > enlam) x += 1;
        else return (double)x;
    }
}

/* Transformed rejection (Hoermann's PTRS), as used above lambda = 10. */
static double poisson_ptrs(PCG64 *rng, double lam) {
    double slam = sqrt(lam);
    double loglam = log(lam);
    double b = 0.931 + 2.53 * slam;
    double a = -0.059 + 0.02483 * b;
    double invalpha = 1.1239 + 1.1328 / (b - 3.4);
    double vr = 0.9277 - 3.6224 / (b - 2);
    while (1) {
        double U = pcg_next_double(rng) - 0.5;
        double V = pcg_next_double(rng);
        double us = 0.5 - fabs(U);
        double k = floor((2 * a / us + b) * U + lam + 0.43);
        if ((us >= 0.07) && (V <= vr)) return k;
        if ((k < 0) || ((us < 0.013) && (V > us))) continue;
        if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
            (-lam + k * loglam - lgamma(k + 1)))
            return k;
    }
}

/* ---- the Generator object ---------------------------------------------- */

typedef struct {
    PyObject_HEAD
    PCG64 rng;
    PyObject *seed_obj;
} QGenerator;

static PyTypeObject QGenerator_Type;

static void generator_dealloc(QGenerator *self) {
    Py_XDECREF(self->seed_obj);
    PyObject_Del(self);
}

/* Shape from a `size=` argument: None means a single scalar. */
static int size_to_shape(PyObject *size, qintp *shape, int *nd, int *scalar) {
    if (size == NULL || size == Py_None) { *nd = 0; *scalar = 1; return 0; }
    *scalar = 0;
    return qnp_shape_from_object(size, shape, nd);
}

typedef double (*Sampler)(PCG64 *rng, void *ctx);

/* Draws one variate from up to two per-element parameters, so that
 * `uniform(low_array, high_array, size=...)` broadcasts the way NumPy's
 * generators do. */
typedef double (*ParamSampler)(PCG64 *rng, const double *params);
typedef int (*ParamValidator)(QArray **params);

static PyObject *fill_samples(QGenerator *self, PyObject *size, Sampler fn, void *ctx) {
    qintp shape[QNP_MAXDIMS];
    int nd, scalar;
    if (size_to_shape(size, shape, &nd, &scalar) < 0) return NULL;
    QArray *out = qnp_new(nd, shape, QNP_FLOAT64);
    if (out == NULL) return NULL;
    qintp n = qnp_size(out);
    double *p = (double *)out->data;
    for (qintp i = 0; i < n; i++) p[i] = fn(&self->rng, ctx);
    if (scalar) {
        double v = *p;
        Py_DECREF(out);
        return PyFloat_FromDouble(v);
    }
    return (PyObject *)out;
}

/* The parameterised path: every argument becomes a float64 array, the result
 * shape is `size` when given and the broadcast of the parameters otherwise,
 * and the output is filled in C order -- one draw per element, as NumPy does. */
/* A distribution parameter confined to [0, inf) is checked before any variate
 * is drawn, so an invalid scale cannot pass silently as a stream of samples. */
static int reject_negative(QArray *p, const char *message) {
    QArray *ops[1] = {p};
    QIter it;
    if (qnp_iter_init(&it, 1, ops, p->shape, p->nd) < 0) return -1;
    while (qnp_iter_next(&it)) {
        const char *q = it.ptr[0];
        for (qintp i = 0; i < it.inner_len; i++, q += it.inner_stride[0]) {
            if (*(const double *)q < 0.0) {
                PyErr_SetString(PyExc_ValueError, message);
                return -1;
            }
        }
    }
    return 0;
}

static int validate_poisson(QArray **params) {
    QArray *p = params[0];
    if (qnp_size(p) == 0) return 0;
    QIter it;
    if (qnp_iter_init(&it, 1, params, p->shape, p->nd) < 0) return -1;
    /* Leave ten standard deviations before int64 overflow, as in NumPy. */
    const double largest = (double)INT64_MAX - 10.0 * sqrt((double)INT64_MAX);
    while (qnp_iter_next(&it)) {
        const char *q = it.ptr[0];
        for (qintp i = 0; i < it.inner_len; i++, q += it.inner_stride[0]) {
            double lam = *(const double *)q;
            if (!isfinite(lam) || lam < 0.0 || lam > largest) {
                PyErr_SetString(PyExc_ValueError,
                                "lam must be finite, nonnegative and within the int64 sampling range");
                return -1;
            }
        }
    }
    return 0;
}

static int validate_uniform(QArray **params) {
    /* Check raw endpoints as well as their broadcast differences: even an
     * empty requested output must not make a nonfinite scalar acceptable. */
    for (int k = 0; k < 2; k++) {
        QArray *p = params[k];
        if (qnp_size(p) == 0) continue;
        QArray *ops[1] = {p};
        QIter it;
        if (qnp_iter_init(&it, 1, ops, p->shape, p->nd) < 0) return -1;
        while (qnp_iter_next(&it)) {
            const char *q = it.ptr[0];
            for (qintp i = 0; i < it.inner_len; i++, q += it.inner_stride[0]) {
                if (!isfinite(*(const double *)q)) {
                    PyErr_SetString(PyExc_ValueError, "uniform bounds must be finite");
                    return -1;
                }
            }
        }
    }
    qintp shape[QNP_MAXDIMS];
    int nd;
    if (qnp_broadcast_shapes(2, params, shape, &nd) < 0) return -1;
    for (int d = 0; d < nd; d++) if (shape[d] == 0) return 0;
    qintp count = 1;
    for (int d = 0; d < nd; d++) {
        if (shape[d] > PY_SSIZE_T_MAX / (qintp)sizeof(double) / count) {
            PyErr_SetString(PyExc_ValueError, "broadcast parameter array is too big");
            return -1;
        }
        count *= shape[d];
    }
    QIter it;
    if (qnp_iter_init(&it, 2, params, shape, nd) < 0) return -1;
    while (qnp_iter_next(&it)) {
        const char *lo = it.ptr[0], *hi = it.ptr[1];
        for (qintp i = 0; i < it.inner_len; i++) {
            double width = *(const double *)hi - *(const double *)lo;
            if (!isfinite(width) || width < 0.0) {
                PyErr_SetString(PyExc_ValueError,
                                "high - low must be finite and nonnegative");
                return -1;
            }
            lo += it.inner_stride[0];
            hi += it.inner_stride[1];
        }
    }
    return 0;
}

static PyObject *fill_with_params(QGenerator *self, PyObject *size, int nparams,
                                  PyObject **param_objs, ParamSampler fn,
                                  int integral, int check_index,
                                  const char *check_msg, ParamValidator validate) {
    QArray *params[2] = {NULL, NULL};
    for (int i = 0; i < nparams; i++) {
        params[i] = qnp_from_any(param_objs[i], QNP_FLOAT64, 1);
        if (params[i] == NULL) {
            for (int k = 0; k < i; k++) Py_CLEAR(params[k]);
            return NULL;
        }
    }
    if (check_index >= 0 && reject_negative(params[check_index], check_msg) < 0)
        goto fail;
    qintp shape[QNP_MAXDIMS];
    int nd = 0, scalar = 0;
    if (size == NULL || size == Py_None) {
        if (qnp_broadcast_shapes(nparams, params, shape, &nd) < 0) goto fail;
        scalar = (nd == 0);
    } else {
        if (qnp_shape_from_object(size, shape, &nd) < 0) goto fail;
    }
    QArray *out = qnp_new(nd, shape, QNP_FLOAT64);
    if (out == NULL) goto fail;
    QArray *ops[3];
    ops[0] = out;
    QArray *broadcast[2] = {NULL, NULL};
    for (int i = 0; i < nparams; i++) {
        PyObject *b = qnp_broadcast_to((PyObject *)params[i], shape, nd);
        if (b == NULL) {
            for (int k = 0; k < i; k++) Py_CLEAR(broadcast[k]);
            Py_DECREF(out);
            goto fail;
        }
        broadcast[i] = (QArray *)b;
        ops[i + 1] = broadcast[i];
    }
    /* Reject incompatible or unallocatable outputs before walking a possibly
     * large parameter broadcast. Validation still precedes every RNG draw,
     * and checks raw parameters even when the requested output is empty. */
    if (validate != NULL && validate(params) < 0) {
        for (int i = 0; i < nparams; i++) Py_CLEAR(broadcast[i]);
        Py_DECREF(out);
        goto fail;
    }
    QIter it;
    if (qnp_iter_init(&it, nparams + 1, ops, shape, nd) < 0) {
        for (int i = 0; i < nparams; i++) Py_CLEAR(broadcast[i]);
        Py_DECREF(out);
        goto fail;
    }
    while (qnp_iter_next(&it)) {
        char *dst = it.ptr[0];
        const char *a = nparams > 0 ? it.ptr[1] : NULL;
        const char *b = nparams > 1 ? it.ptr[2] : NULL;
        for (qintp i = 0; i < it.inner_len; i++) {
            double values[2] = {0.0, 0.0};
            if (a != NULL) values[0] = *(const double *)a;
            if (b != NULL) values[1] = *(const double *)b;
            *(double *)dst = fn(&self->rng, values);
            dst += it.inner_stride[0];
            if (a != NULL) a += it.inner_stride[1];
            if (b != NULL) b += it.inner_stride[2];
        }
    }
    for (int i = 0; i < nparams; i++) Py_CLEAR(broadcast[i]);
    for (int i = 0; i < nparams; i++) Py_CLEAR(params[i]);
    if (integral) {
        QArray *ints = qnp_astype(out, QNP_INT64, 1);
        Py_DECREF(out);
        if (ints == NULL) return NULL;
        if (scalar) {
            int64_t v = *(int64_t *)ints->data;
            Py_DECREF(ints);
            return PyLong_FromLongLong((long long)v);
        }
        return (PyObject *)ints;
    }
    if (scalar) {
        double v = *(double *)out->data;
        Py_DECREF(out);
        return PyFloat_FromDouble(v);
    }
    return (PyObject *)out;
fail:
    for (int i = 0; i < nparams; i++) Py_XDECREF(params[i]);
    return NULL;
}

static double sample_uniform01(PCG64 *rng, void *ctx) { (void)ctx; return pcg_next_double(rng); }
static double sample_normal01(PCG64 *rng, void *ctx) { (void)ctx; return standard_normal_one(rng); }
static double sample_expo1(PCG64 *rng, void *ctx) { (void)ctx; return standard_exponential_one(rng); }

static double param_normal(PCG64 *rng, const double *p) {
    return p[0] + p[1] * standard_normal_one(rng);
}
static double param_uniform(PCG64 *rng, const double *p) {
    return p[0] + (p[1] - p[0]) * pcg_next_double(rng);
}
static double param_expo(PCG64 *rng, const double *p) {
    return p[0] * standard_exponential_one(rng);
}
static double param_poisson(PCG64 *rng, const double *p) {
    double lam = p[0];
    if (lam >= 10.0) return poisson_ptrs(rng, lam);
    if (lam == 0.0) return 0.0;
    return poisson_mult(rng, lam);
}

static PyObject *gen_random(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *size = Py_None;
    static char *kwlist[] = {"size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:random", kwlist, &size)) return NULL;
    return fill_samples(self, size, sample_uniform01, NULL);
}

static PyObject *gen_standard_normal(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *size = Py_None;
    static char *kwlist[] = {"size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:standard_normal", kwlist, &size))
        return NULL;
    return fill_samples(self, size, sample_normal01, NULL);
}

static PyObject *gen_standard_exponential(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *size = Py_None;
    static char *kwlist[] = {"size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:standard_exponential", kwlist, &size))
        return NULL;
    return fill_samples(self, size, sample_expo1, NULL);
}

static PyObject *gen_normal(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *loc = NULL, *scale = NULL, *size = Py_None;
    static char *kwlist[] = {"loc", "scale", "size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO:normal", kwlist,
                                     &loc, &scale, &size)) return NULL;
    PyObject *zero = PyFloat_FromDouble(0.0), *one = PyFloat_FromDouble(1.0);
    if (zero == NULL || one == NULL) { Py_XDECREF(zero); Py_XDECREF(one); return NULL; }
    PyObject *params[2] = {loc ? loc : zero, scale ? scale : one};
    PyObject *result = fill_with_params(self, size, 2, params, param_normal, 0, 1, "scale < 0", NULL);
    Py_DECREF(zero);
    Py_DECREF(one);
    return result;
}

static PyObject *gen_uniform(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *low = NULL, *high = NULL, *size = Py_None;
    static char *kwlist[] = {"low", "high", "size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO:uniform", kwlist,
                                     &low, &high, &size)) return NULL;
    PyObject *zero = PyFloat_FromDouble(0.0), *one = PyFloat_FromDouble(1.0);
    if (zero == NULL || one == NULL) { Py_XDECREF(zero); Py_XDECREF(one); return NULL; }
    PyObject *params[2] = {low ? low : zero, high ? high : one};
    PyObject *result = fill_with_params(self, size, 2, params, param_uniform, 0, -1, NULL,
                                       validate_uniform);
    Py_DECREF(zero);
    Py_DECREF(one);
    return result;
}

static PyObject *gen_exponential(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *scale = NULL, *size = Py_None;
    static char *kwlist[] = {"scale", "size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO:exponential", kwlist, &scale, &size))
        return NULL;
    PyObject *one = PyFloat_FromDouble(1.0);
    if (one == NULL) return NULL;
    PyObject *params[1] = {scale ? scale : one};
    PyObject *result = fill_with_params(self, size, 1, params, param_expo, 0, 0, "scale < 0", NULL);
    Py_DECREF(one);
    return result;
}

static PyObject *gen_poisson(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *lam = NULL, *size = Py_None;
    static char *kwlist[] = {"lam", "size", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO:poisson", kwlist, &lam, &size))
        return NULL;
    PyObject *one = PyFloat_FromDouble(1.0);
    if (one == NULL) return NULL;
    PyObject *params[1] = {lam ? lam : one};
    PyObject *result = fill_with_params(self, size, 1, params, param_poisson, 1, -1, NULL,
                                       validate_poisson);
    Py_DECREF(one);
    return result;
}

static PyObject *gen_integers(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *low_o, *high_o = Py_None, *size = Py_None, *dtype = NULL;
    int endpoint = 0;
    static char *kwlist[] = {"low", "high", "size", "dtype", "endpoint", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOOp:integers", kwlist,
                                     &low_o, &high_o, &size, &dtype, &endpoint))
        return NULL;
    long long low, high;
    if (high_o == Py_None) {
        low = 0;
        high = PyLong_AsLongLong(low_o);
    } else {
        low = PyLong_AsLongLong(low_o);
        high = PyLong_AsLongLong(high_o);
    }
    if (PyErr_Occurred()) return NULL;
    if (endpoint) high += 1;
    if (high <= low) {
        PyErr_SetString(PyExc_ValueError, "low >= high");
        return NULL;
    }
    uint64_t range = (uint64_t)(high - low) - 1;
    qintp shape[QNP_MAXDIMS];
    int nd, scalar;
    if (size_to_shape(size, shape, &nd, &scalar) < 0) return NULL;
    QArray *out = qnp_new(nd, shape, QNP_INT64);
    if (out == NULL) return NULL;
    qintp n = qnp_size(out);
    int64_t *p = (int64_t *)out->data;
    for (qintp i = 0; i < n; i++) p[i] = low + (int64_t)bounded_uint64(&self->rng, range);
    if (scalar) {
        int64_t v = *p;
        Py_DECREF(out);
        return PyLong_FromLongLong((long long)v);
    }
    return (PyObject *)out;
}

/* Fisher-Yates over the trailing index, matching NumPy's shuffle order.
 *
 * NumPy draws the swap partner two different ways depending on the caller:
 * `shuffle` and `permutation` use masked rejection, while the shuffle inside
 * `choice` uses Lemire.  Both are reproduced, because which one runs decides
 * the whole subsequent stream. */
static void shuffle_indices_masked(PCG64 *rng, int64_t *idx, qintp n) {
    for (qintp i = n - 1; i > 0; i--) {
        uint64_t j = random_interval(rng, (uint64_t)i);
        int64_t t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}

static void shuffle_indices_lemire(PCG64 *rng, int64_t *idx, qintp n) {
    for (qintp i = n - 1; i > 0; i--) {
        uint64_t j = bounded_uint64(rng, (uint64_t)i);
        int64_t t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
}

static PyObject *gen_permutation(QGenerator *self, PyObject *arg) {
    qintp n;
    QArray *source = NULL;
    if (!QArray_Check(arg) && PyIndex_Check(arg)) {
        Py_ssize_t v = PyNumber_AsSsize_t(arg, PyExc_OverflowError);
        if (v == -1 && PyErr_Occurred()) return NULL;
        n = v;
    } else {
        source = qnp_from_any(arg, -1, 0);
        if (source == NULL) return NULL;
        if (source->nd == 0) {
            Py_DECREF(source);
            PyErr_SetString(PyExc_ValueError, "permutation of a 0-d array");
            return NULL;
        }
        n = source->shape[0];
    }
    QArray *idx = qnp_new(1, &n, QNP_INT64);
    if (idx == NULL) { Py_XDECREF(source); return NULL; }
    int64_t *ip = (int64_t *)idx->data;
    for (qintp i = 0; i < n; i++) ip[i] = i;
    shuffle_indices_masked(&self->rng, ip, n);
    if (source == NULL) return (PyObject *)idx;
    PyObject *result = qnp_take_axis(source, idx, 0);
    Py_DECREF(source);
    Py_DECREF(idx);
    return result;
}

static PyObject *gen_shuffle(QGenerator *self, PyObject *arg) {
    if (!QArray_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "shuffle expects an array");
        return NULL;
    }
    QArray *a = (QArray *)arg;
    if (a->nd == 0) {
        PyErr_SetString(PyExc_ValueError, "shuffle of a 0-d array");
        return NULL;
    }
    qintp n = a->shape[0];
    QArray *idx = qnp_new(1, &n, QNP_INT64);
    if (idx == NULL) return NULL;
    int64_t *ip = (int64_t *)idx->data;
    for (qintp i = 0; i < n; i++) ip[i] = i;
    shuffle_indices_masked(&self->rng, ip, n);
    PyObject *shuffled = qnp_take_axis(a, idx, 0);
    Py_DECREF(idx);
    if (shuffled == NULL) return NULL;
    int rc = qnp_copy_into(a, (QArray *)shuffled);
    Py_DECREF(shuffled);
    if (rc < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *gen_choice(QGenerator *self, PyObject *args, PyObject *kwds) {
    PyObject *a_obj, *size = Py_None, *p_obj = Py_None;
    int replace = 1;
    static char *kwlist[] = {"a", "size", "replace", "p", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OpO:choice", kwlist,
                                     &a_obj, &size, &replace, &p_obj)) return NULL;
    QArray *population = NULL;
    qintp pop_size;
    if (!QArray_Check(a_obj) && PyIndex_Check(a_obj)) {
        Py_ssize_t v = PyNumber_AsSsize_t(a_obj, PyExc_OverflowError);
        if (v == -1 && PyErr_Occurred()) return NULL;
        pop_size = v;
    } else {
        population = qnp_from_any(a_obj, -1, 0);
        if (population == NULL) return NULL;
        if (population->nd == 0) {
            Py_DECREF(population);
            PyErr_SetString(PyExc_ValueError, "a must be at least one dimensional");
            return NULL;
        }
        pop_size = population->shape[0];
    }
    if (pop_size <= 0) {
        Py_XDECREF(population);
        PyErr_SetString(PyExc_ValueError, "a must be a positive size or a non-empty array");
        return NULL;
    }
    qintp shape[QNP_MAXDIMS];
    int nd, scalar;
    if (size_to_shape(size, shape, &nd, &scalar) < 0) { Py_XDECREF(population); return NULL; }
    QArray *idx = qnp_new(nd, shape, QNP_INT64);
    if (idx == NULL) { Py_XDECREF(population); return NULL; }
    qintp count = qnp_size(idx);
    int64_t *ip = (int64_t *)idx->data;
    if (!replace) {
        if (count > pop_size) {
            Py_DECREF(idx); Py_XDECREF(population);
            PyErr_SetString(PyExc_ValueError,
                            "cannot take a larger sample than population when replace is False");
            return NULL;
        }
        if (p_obj != Py_None) {
            Py_DECREF(idx); Py_XDECREF(population);
            PyErr_SetString(PyExc_NotImplementedError,
                            "weighted sampling without replacement");
            return NULL;
        }
        /* Floyd's algorithm over an open-addressed set, then a shuffle: what
         * NumPy does for every sample size, not just small ones. */
        uint64_t set_size = (uint64_t)(1.2 * (double)count);
        uint64_t mask = set_size;
        mask |= mask >> 1; mask |= mask >> 2; mask |= mask >> 4;
        mask |= mask >> 8; mask |= mask >> 16; mask |= mask >> 32;
        set_size = 1 + mask;
        uint64_t *table = (uint64_t *)PyMem_Malloc((size_t)set_size * sizeof(uint64_t));
        if (table == NULL) {
            Py_DECREF(idx); Py_XDECREF(population);
            return PyErr_NoMemory();
        }
        const uint64_t empty = 0xFFFFFFFFFFFFFFFFULL;
        for (uint64_t i = 0; i < set_size; i++) table[i] = empty;
        for (qintp j = pop_size - count; j < pop_size; j++) {
            uint64_t value = bounded_uint64(&self->rng, (uint64_t)j);
            uint64_t loc = value & mask;
            while (table[loc] != empty && table[loc] != value) loc = (loc + 1) & mask;
            if (table[loc] == empty) {
                table[loc] = value;
                ip[j - pop_size + count] = (int64_t)value;
            } else {
                loc = (uint64_t)j & mask;
                while (table[loc] != empty) loc = (loc + 1) & mask;
                table[loc] = (uint64_t)j;
                ip[j - pop_size + count] = j;
            }
        }
        PyMem_Free(table);
        shuffle_indices_lemire(&self->rng, ip, count);
    } else if (p_obj == Py_None) {
        for (qintp i = 0; i < count; i++)
            ip[i] = (int64_t)bounded_uint64(&self->rng, (uint64_t)(pop_size - 1));
    } else {
        QArray *p0 = qnp_from_any(p_obj, QNP_FLOAT64, 1);
        if (p0 == NULL) { Py_DECREF(idx); Py_XDECREF(population); return NULL; }
        QArray *pw = qnp_ascontiguous(p0);
        Py_DECREF(p0);
        if (pw == NULL) { Py_DECREF(idx); Py_XDECREF(population); return NULL; }
        if (pw->nd != 1 || qnp_size(pw) != pop_size) {
            Py_DECREF(pw); Py_DECREF(idx); Py_XDECREF(population);
            PyErr_SetString(PyExc_ValueError, "p must be one-dimensional with the same size as a");
            return NULL;
        }
        double tolerance = sqrt(DBL_EPSILON);
        if (!QArray_Check(p_obj) && PyObject_CheckBuffer(p_obj)) {
            Py_buffer view;
            if (PyObject_GetBuffer(p_obj, &view, PyBUF_FULL_RO) == 0) {
                const char *format = view.format;
                if (format != NULL) {
                    if (*format == '@' || *format == '=' || *format == '<'
                            || *format == '>' || *format == '|') format++;
                    /* Widening a float32 probability array must not make
                     * its original normalization rounding unacceptable. */
                    if (!strcmp(format, "f") && view.itemsize == sizeof(float))
                        tolerance = sqrt(FLT_EPSILON);
                }
                PyBuffer_Release(&view);
            } else {
                PyErr_Clear();
            }
        }
        /* cumsum, normalise, then a right-side search: NumPy's exact recipe. */
        double *cdf = (double *)PyMem_Malloc((size_t)pop_size * sizeof(double));
        if (cdf == NULL) {
            Py_DECREF(pw); Py_DECREF(idx); Py_XDECREF(population);
            return PyErr_NoMemory();
        }
        const double *probs = (const double *)pw->data;
        double acc = 0.0, sum = 0.0, correction = 0.0;
        for (qintp i = 0; i < pop_size; i++) {
            if (!isfinite(probs[i]) || probs[i] < 0.0) {
                PyMem_Free(cdf);
                Py_DECREF(pw); Py_DECREF(idx); Py_XDECREF(population);
                PyErr_SetString(PyExc_ValueError, "probabilities must be finite and nonnegative");
                return NULL;
            }
            /* Validate with compensated summation, but retain the original
             * cumulative rounding so valid seeded choices stay identical. */
            double adjusted = probs[i] - correction;
            double next = sum + adjusted;
            correction = (next - sum) - adjusted;
            sum = next;
            acc += probs[i];
            cdf[i] = acc;
        }
        if (!isfinite(sum) || fabs(sum - 1.0) > tolerance) {
            PyMem_Free(cdf);
            Py_DECREF(pw); Py_DECREF(idx); Py_XDECREF(population);
            PyErr_SetString(PyExc_ValueError, "probabilities must sum to one");
            return NULL;
        }
        double total = cdf[pop_size - 1];
        for (qintp i = 0; i < pop_size; i++) cdf[i] /= total;
        for (qintp i = 0; i < count; i++) {
            double u = pcg_next_double(&self->rng);
            qintp lo = 0, hi = pop_size;
            while (lo < hi) {
                qintp mid = lo + (hi - lo) / 2;
                if (cdf[mid] <= u) lo = mid + 1; else hi = mid;
            }
            ip[i] = lo;
        }
        PyMem_Free(cdf);
        Py_DECREF(pw);
    }
    if (population == NULL) {
        if (scalar) {
            int64_t v = *ip;
            Py_DECREF(idx);
            return PyLong_FromLongLong((long long)v);
        }
        return (PyObject *)idx;
    }
    PyObject *picked = qnp_take_axis(population, idx, 0);
    Py_DECREF(population);
    Py_DECREF(idx);
    if (picked == NULL) return NULL;
    if (scalar && QArray_Check(picked) && ((QArray *)picked)->nd >= 1) {
        PyObject *key = PyLong_FromLong(0);
        PyObject *item = qnp_getitem((QArray *)picked, key);
        Py_DECREF(key);
        Py_DECREF(picked);
        return item;
    }
    return picked;
}

static PyObject *gen_repr(QGenerator *self) {
    (void)self;
    return PyUnicode_FromString("Generator(PCG64)");
}

static PyObject *gen_bit_generator(QGenerator *self, void *closure) {
    (void)closure;
    return PyUnicode_FromString("PCG64");
}

/* JSON-serializable state, with all cached bits included. Set atomically. */
static PyObject *gen_get_state(QGenerator *self, void *closure) {
    (void)closure;
    return Py_BuildValue("{s:s,s:i,s:(KKKK),s:i,s:I}","bit_generator","PCG64","version",1,
        "words",(unsigned long long)u128_hi(self->rng.state),(unsigned long long)u128_lo(self->rng.state),
        (unsigned long long)u128_hi(self->rng.inc),(unsigned long long)u128_lo(self->rng.inc),
        "has_uint32",self->rng.has_uint32,"uinteger",self->rng.uinteger);
}
static int gen_set_state(QGenerator *self,PyObject *value,void *closure) {
    (void)closure;
    if(!value||!PyDict_Check(value)){PyErr_SetString(PyExc_TypeError,"state must be a PCG64 state dictionary");return -1;}
    PyObject *kind=PyDict_GetItemString(value,"bit_generator"),*words=PyDict_GetItemString(value,"words"),*version=PyDict_GetItemString(value,"version"),*has=PyDict_GetItemString(value,"has_uint32"),*cached=PyDict_GetItemString(value,"uinteger");
    if(!kind||!PyUnicode_Check(kind)||PyUnicode_CompareWithASCIIString(kind,"PCG64")||!words||!version||!has||!cached){PyErr_SetString(PyExc_ValueError,"invalid PCG64 state");return -1;}
    long ver=PyLong_AsLong(version),h=PyLong_AsLong(has);unsigned long c=PyLong_AsUnsignedLong(cached);
    if(PyErr_Occurred())return -1;
    if(ver!=1||(h!=0&&h!=1)||c>UINT32_MAX){PyErr_SetString(PyExc_ValueError,"invalid PCG64 state version or cached bits");return -1;}
    PyObject *seq=PySequence_Fast(words,"state words must be a sequence");if(!seq)return -1;
    if(PySequence_Fast_GET_SIZE(seq)!=4){Py_DECREF(seq);PyErr_SetString(PyExc_ValueError,"state requires four uint64 words");return -1;}
    uint64_t w[4];for(int i=0;i<4;i++){w[i]=PyLong_AsUnsignedLongLong(PySequence_Fast_GET_ITEM(seq,i));if(PyErr_Occurred()){Py_DECREF(seq);return -1;}}
    Py_DECREF(seq);
    if(!(w[3]&1)){PyErr_SetString(PyExc_ValueError,"PCG64 increment must be odd");return -1;}
    self->rng.state=u128_make(w[0],w[1]);self->rng.inc=u128_make(w[2],w[3]);self->rng.has_uint32=(int)h;self->rng.uinteger=(uint32_t)c;return 0;
}

static PyGetSetDef generator_getset[] = {
    {"state", (getter)gen_get_state, (setter)gen_set_state, "Serializable complete PCG64 state.", NULL},
    {"bit_generator", (getter)gen_bit_generator, NULL, NULL, NULL},
    {NULL}
};

static PyMethodDef generator_methods[] = {
    {"random", (PyCFunction)gen_random, METH_VARARGS | METH_KEYWORDS, "Uniform samples in [0, 1)."},
    {"standard_normal", (PyCFunction)gen_standard_normal, METH_VARARGS | METH_KEYWORDS, "Standard normal samples."},
    {"standard_exponential", (PyCFunction)gen_standard_exponential, METH_VARARGS | METH_KEYWORDS, "Standard exponential samples."},
    {"normal", (PyCFunction)gen_normal, METH_VARARGS | METH_KEYWORDS, "Normal samples."},
    {"uniform", (PyCFunction)gen_uniform, METH_VARARGS | METH_KEYWORDS, "Uniform samples over an interval."},
    {"exponential", (PyCFunction)gen_exponential, METH_VARARGS | METH_KEYWORDS, "Exponential samples."},
    {"poisson", (PyCFunction)gen_poisson, METH_VARARGS | METH_KEYWORDS, "Poisson samples."},
    {"integers", (PyCFunction)gen_integers, METH_VARARGS | METH_KEYWORDS, "Random integers in a half-open range."},
    {"permutation", (PyCFunction)gen_permutation, METH_O, "Randomly permuted copy."},
    {"shuffle", (PyCFunction)gen_shuffle, METH_O, "Shuffle an array in place."},
    {"choice", (PyCFunction)gen_choice, METH_VARARGS | METH_KEYWORDS, "Random sample from a population."},
    {NULL}
};

static PyTypeObject QGenerator_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "quadrivium._qnp.Generator",
    .tp_basicsize = sizeof(QGenerator),
    .tp_dealloc = (destructor)generator_dealloc,
    .tp_repr = (reprfunc)gen_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = generator_methods,
    .tp_getset = generator_getset,
};

/* ---- seeding ----------------------------------------------------------- */

static int coerce_entropy(PyObject *seed, uint32_t **words, size_t *count) {
    if (seed == NULL || seed == Py_None) {
        /* Fresh entropy: the OS pool, hashed the same way a seed would be. */
        uint32_t *buf = (uint32_t *)PyMem_Malloc(4 * sizeof(uint32_t));
        if (buf == NULL) { PyErr_NoMemory(); return -1; }
        PyObject *osmod = PyImport_ImportModule("os");
        if (osmod == NULL) { PyMem_Free(buf); return -1; }
        PyObject *raw = PyObject_CallMethod(osmod, "urandom", "i", 16);
        Py_DECREF(osmod);
        if (raw == NULL) { PyMem_Free(buf); return -1; }
        memcpy(buf, PyBytes_AS_STRING(raw), 16);
        Py_DECREF(raw);
        *words = buf;
        *count = 4;
        return 0;
    }
    PyObject *as_int = PyNumber_Long(seed);
    if (as_int == NULL) {
        PyErr_SetString(PyExc_TypeError, "seed must be an integer or None");
        return -1;
    }
    PyObject *zero = PyLong_FromLong(0);
    if (zero == NULL) { Py_DECREF(as_int); return -1; }
    int negative = PyObject_RichCompareBool(as_int, zero, Py_LT);
    Py_DECREF(zero);
    if (negative < 0) { Py_DECREF(as_int); return -1; }
    if (negative) {
        Py_DECREF(as_int);
        PyErr_SetString(PyExc_ValueError, "expected non-negative integer seed");
        return -1;
    }
    PyObject *bits_obj = PyObject_CallMethod(as_int, "bit_length", NULL);
    if (bits_obj == NULL) { Py_DECREF(as_int); return -1; }
    long nbits = PyLong_AsLong(bits_obj);
    Py_DECREF(bits_obj);
    if (nbits < 0 && PyErr_Occurred()) { Py_DECREF(as_int); return -1; }
    size_t nwords = nbits > 0 ? ((size_t)nbits + 31) / 32 : 1;
    uint32_t *buf = (uint32_t *)PyMem_Malloc(nwords * sizeof(uint32_t));
    if (buf == NULL) { Py_DECREF(as_int); PyErr_NoMemory(); return -1; }
    PyObject *mask = PyLong_FromUnsignedLong(0xFFFFFFFFu);
    PyObject *shift = PyLong_FromLong(32);
    PyObject *cur = Py_NewRef(as_int);
    int failed = (mask == NULL || shift == NULL);
    for (size_t i = 0; i < nwords && !failed; i++) {
        PyObject *low = PyNumber_And(cur, mask);
        if (low == NULL) { failed = 1; break; }
        buf[i] = (uint32_t)PyLong_AsUnsignedLong(low);
        Py_DECREF(low);
        if (PyErr_Occurred()) { failed = 1; break; }
        PyObject *next = PyNumber_Rshift(cur, shift);
        Py_DECREF(cur);
        cur = next;
        if (cur == NULL) { failed = 1; break; }
    }
    Py_XDECREF(cur);
    Py_XDECREF(mask);
    Py_XDECREF(shift);
    Py_DECREF(as_int);
    if (failed) { PyMem_Free(buf); return -1; }
    *words = buf;
    *count = nwords;
    return 0;
}

static PyObject *py_default_rng(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seed = Py_None;
    if (!PyArg_ParseTuple(args, "|O:default_rng", &seed)) return NULL;
    if (Py_TYPE(seed) == &QGenerator_Type) {
        Py_INCREF(seed);
        return seed;
    }
    if (seed != Py_None && !(!QArray_Check(seed) && PyIndex_Check(seed))) {
        /* Anything that already behaves like a generator is passed through,
         * which is how a caller's own NumPy Generator keeps working. */
        PyObject *probe = PyObject_GetAttrString(seed, "standard_normal");
        if (probe != NULL) {
            Py_DECREF(probe);
            Py_INCREF(seed);
            return seed;
        }
        PyErr_Clear();
    }
    uint32_t *entropy = NULL;
    size_t count = 0;
    if (coerce_entropy(seed, &entropy, &count) < 0) return NULL;
    uint32_t pool[4];
    mix_entropy(pool, entropy, count);
    PyMem_Free(entropy);
    QGenerator *gen = PyObject_New(QGenerator, &QGenerator_Type);
    if (gen == NULL) return NULL;
    gen->seed_obj = Py_NewRef(seed);
    pcg_seed(&gen->rng, pool);
    return (PyObject *)gen;
}

PyMethodDef qnp_random_methods[] = {
    {"default_rng", py_default_rng, METH_VARARGS,
     "Construct a PCG64-backed generator from a seed."},
    {NULL}
};

int qnp_add_random(PyObject *module) {
    if (PyType_Ready(&QGenerator_Type) < 0) return -1;
    Py_INCREF(&QGenerator_Type);
    if (PyModule_AddObject(module, "Generator", (PyObject *)&QGenerator_Type) < 0) return -1;
    return 0;
}
