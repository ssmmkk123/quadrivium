/* Core declarations for the C array backend behind quadrivium.
 *
 * The package used to lean on NumPy for its array type and for the handful of
 * numerical primitives NumPy exposes.  This extension provides the same
 * semantics for the subset the library actually needs: strided N-dimensional
 * arrays over six dtypes, broadcasting element-wise operations, the indexing
 * grammar, reductions, dense linear algebra, transforms and a PCG64 generator.
 *
 * Boolean/integer and single/double real/complex storage are supported.
 * Enum values remain stable; qnp_promote implements precision-aware promotion.
 */
#ifndef QNP_H
#define QNP_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef Py_ssize_t qintp;

/* Native complex128.  A plain struct rather than C99 `double complex`: the
 * naive multiply below is what NumPy does and what vectorizes, while the C99
 * operator emits a libgcc call for infinity/NaN bookkeeping we do not want. */
typedef struct { double re, im; } qcomplex;

enum {
    QNP_BOOL = 0,
    QNP_INT64 = 1,
    QNP_FLOAT64 = 2,
    QNP_COMPLEX128 = 3,
    QNP_FLOAT32 = 4,
    QNP_COMPLEX64 = 5,
    QNP_NTYPES = 6
};

/* Flags mirror the NumPy names because the semantics are the same. */
#define QNP_C_CONTIGUOUS 0x0001
#define QNP_F_CONTIGUOUS 0x0002
#define QNP_OWNDATA      0x0004
#define QNP_WRITEABLE    0x0008

#define QNP_MAXDIMS 16

typedef struct {
    PyObject_HEAD
    char *data;
    PyObject *base;        /* owner keeping `data` alive for views, else NULL */
    qintp *shape;          /* nd entries, allocated with `strides` */
    qintp *strides;        /* nd entries, in bytes */
    int nd;
    int dtype;
    int flags;
    Py_ssize_t exports;    /* live PEP 3118 buffer exports */
    PyObject *weakreflist;
} QArray;

extern PyTypeObject QArray_Type;
extern PyTypeObject QDtype_Type;
extern PyTypeObject QFlatiter_Type;
extern PyObject *QNP_LinAlgError;

#define QArray_Check(op) PyObject_TypeCheck(op, &QArray_Type)
#define QArray_CheckExact(op) (Py_TYPE(op) == &QArray_Type)
#define QNP_DATA(a) (((QArray *)(a))->data)
#define QNP_NDIM(a) (((QArray *)(a))->nd)
#define QNP_SHAPE(a) (((QArray *)(a))->shape)
#define QNP_STRIDES(a) (((QArray *)(a))->strides)
#define QNP_TYPE(a) (((QArray *)(a))->dtype)

static const int qnp_itemsize_table[QNP_NTYPES] = {1, 8, 8, 16, 4, 8};
#define QNP_ITEMSIZE(t) (qnp_itemsize_table[(t)])

/* ---- complex helpers ------------------------------------------------- */
static inline qcomplex qc(double re, double im) { qcomplex z; z.re = re; z.im = im; return z; }
static inline qcomplex qc_add(qcomplex a, qcomplex b) { return qc(a.re + b.re, a.im + b.im); }
static inline qcomplex qc_sub(qcomplex a, qcomplex b) { return qc(a.re - b.re, a.im - b.im); }
static inline qcomplex qc_mul(qcomplex a, qcomplex b) {
    return qc(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
static inline qcomplex qc_neg(qcomplex a) { return qc(-a.re, -a.im); }
static inline qcomplex qc_conj(qcomplex a) { return qc(a.re, -a.im); }
static inline double qc_abs(qcomplex a) { return hypot(a.re, a.im); }
/* Smith's algorithm, as NumPy uses: scaling by the larger part keeps the
 * intermediate products from overflowing for well-scaled operands. */
static inline qcomplex qc_div(qcomplex a, qcomplex b) {
    double in;
    if (fabs(b.re) >= fabs(b.im)) {
        if (b.re == 0.0 && b.im == 0.0) return qc(a.re / b.re, a.im / b.re);
        in = b.im / b.re;
        double d = b.re + b.im * in;
        return qc((a.re + a.im * in) / d, (a.im - a.re * in) / d);
    }
    in = b.re / b.im;
    double d = b.re * in + b.im;
    return qc((a.re * in + a.im) / d, (a.im * in - a.re) / d);
}
qcomplex qc_sqrt(qcomplex a);
qcomplex qc_exp(qcomplex a);
qcomplex qc_log(qcomplex a);
qcomplex qc_pow(qcomplex a, qcomplex b);
qcomplex qc_sin(qcomplex a);
qcomplex qc_cos(qcomplex a);
qcomplex qc_tan(qcomplex a);
qcomplex qc_sinh(qcomplex a);
qcomplex qc_cosh(qcomplex a);
qcomplex qc_tanh(qcomplex a);

/* Compact storage uses double-width scalar registers, without temporary arrays. */
static inline int qnp_is_complex(int t) { return t == QNP_COMPLEX128 || t == QNP_COMPLEX64; }
static inline int qnp_wide_type(int t) { return t == QNP_FLOAT32 ? QNP_FLOAT64 : t == QNP_COMPLEX64 ? QNP_COMPLEX128 : t; }
static inline qcomplex qnp_read_number(const char *p, int t) {
    switch (t) {
        case QNP_BOOL: return qc(*(const unsigned char *)p, 0);
        case QNP_INT64: return qc((double)*(const int64_t *)p, 0);
        case QNP_FLOAT32: return qc(*(const float *)p, 0);
        case QNP_FLOAT64: return qc(*(const double *)p, 0);
        case QNP_COMPLEX64: return qc(((const float *)p)[0], ((const float *)p)[1]);
        default: return *(const qcomplex *)p;
    }
}
static inline void qnp_write_number(char *p, int t, qcomplex z) {
    switch (t) {
        case QNP_BOOL: *(unsigned char *)p = z.re != 0 || z.im != 0; break;
        case QNP_INT64: *(int64_t *)p = (int64_t)z.re; break;
        case QNP_FLOAT32: *(float *)p = (float)z.re; break;
        case QNP_FLOAT64: *(double *)p = z.re; break;
        case QNP_COMPLEX64: ((float *)p)[0] = (float)z.re; ((float *)p)[1] = (float)z.im; break;
        default: *(qcomplex *)p = z; break;
    }
}

/* ---- array creation and conversion ----------------------------------- */
QArray *qnp_new(int nd, const qintp *shape, int dtype);
QArray *qnp_new_like(QArray *proto, int dtype);
QArray *qnp_new_view(QArray *base, char *data, int nd, const qintp *shape,
                     const qintp *strides, int dtype);
QArray *qnp_new_view_as(PyTypeObject *type, QArray *base, char *data, int nd,
                        const qintp *shape, const qintp *strides, int dtype);
QArray *qnp_from_any(PyObject *obj, int dtype_hint, int force_dtype);
QArray *qnp_ascontiguous(QArray *a);
QArray *qnp_astype(QArray *a, int dtype, int copy);
PyObject *qnp_from_scalar(int dtype, const void *value);
int qnp_scalar_dtype(PyObject *obj, int *weak);
int qnp_pack_scalar(PyObject *obj, int dtype, void *out);
PyObject *qnp_getitem_ptr(int dtype, const char *ptr);
int qnp_setitem_ptr(int dtype, char *ptr, PyObject *value);
void qnp_update_flags(QArray *a);
qintp qnp_size(const QArray *a);
int qnp_is_c_contiguous(const QArray *a);

/* ---- promotion ------------------------------------------------------- */
int qnp_promote(int a, int b);
int qnp_dtype_from_object(PyObject *obj, int *ok);
PyObject *qnp_dtype_object(int dtype);

/* ---- element-wise machinery ------------------------------------------ */
PyObject *qnp_binary_op(int op, PyObject *a, PyObject *b, PyObject *out, PyObject *where);
PyObject *qnp_unary_op(int op, PyObject *a, PyObject *out);
PyObject *qnp_where3(PyObject *cond, PyObject *x, PyObject *y);

/* Binary op codes. */
enum {
    QOP_ADD, QOP_SUB, QOP_MUL, QOP_TRUEDIV, QOP_FLOORDIV, QOP_MOD, QOP_POW,
    QOP_MAXIMUM, QOP_MINIMUM, QOP_HYPOT, QOP_ARCTAN2, QOP_COPYSIGN,
    QOP_EQ, QOP_NE, QOP_LT, QOP_LE, QOP_GT, QOP_GE,
    QOP_AND, QOP_OR, QOP_XOR, QOP_LOGICAL_AND, QOP_LOGICAL_OR,
    QOP_LSHIFT, QOP_RSHIFT,
    QOP_NBINARY
};

/* Unary op codes. */
enum {
    QOP_NEG, QOP_POS, QOP_ABS, QOP_SQRT, QOP_EXP, QOP_LOG, QOP_LOG2, QOP_LOG10,
    QOP_LOG1P, QOP_EXPM1, QOP_SIN, QOP_COS, QOP_TAN, QOP_ARCSIN, QOP_ARCCOS,
    QOP_ARCTAN, QOP_SINH, QOP_COSH, QOP_TANH, QOP_ARCSINH, QOP_ARCCOSH,
    QOP_ARCTANH, QOP_SIGN, QOP_FLOOR, QOP_CEIL, QOP_TRUNC, QOP_RINT,
    QOP_SQUARE, QOP_RECIPROCAL, QOP_CONJ, QOP_REAL, QOP_IMAG, QOP_ANGLE,
    QOP_ISFINITE, QOP_ISNAN, QOP_ISINF, QOP_NOT, QOP_INVERT, QOP_SIGNBIT,
    QOP_NUNARY
};

/* ---- reductions ------------------------------------------------------ */
enum { QRED_SUM, QRED_PROD, QRED_MAX, QRED_MIN, QRED_ANY, QRED_ALL,
       QRED_ARGMAX, QRED_ARGMIN, QRED_MEAN, QRED_COUNT_NONZERO };
PyObject *qnp_reduce(int kind, PyObject *a, PyObject *axis, PyObject *out, int keepdims);
PyObject *qnp_moment(PyObject *a, PyObject *axis, double ddof, int want_std, int keepdims);
PyObject *qnp_accumulate(int kind, PyObject *a, PyObject *axis, PyObject *out);
double qnp_pairwise_sum_f64(const double *x, qintp n, qintp stride);
void qnp_exp_f64(const double *src, double *dst, qintp n);

/* ---- indexing -------------------------------------------------------- */
PyObject *qnp_getitem(QArray *self, PyObject *key);
int qnp_setitem(QArray *self, PyObject *key, PyObject *value);
/* memcmp is undefined for a null pointer even with a zero byte count, and a
 * zero-dimensional array carries a null shape, so the count is checked first. */
static inline int qnp_same_shape(int nd, const qintp *a, const qintp *b) {
    return nd == 0 || !memcmp(a, b, (size_t)nd * sizeof(qintp));
}

int qnp_may_share_memory(QArray *x, QArray *y);
int qnp_overlap_needs_copy(QArray *out, QArray *in);
int qnp_copy_into(QArray *dst, QArray *src);
PyObject *qnp_take_axis(QArray *a, QArray *idx, int axis);

/* ---- shape ----------------------------------------------------------- */
PyObject *qnp_reshape(QArray *a, PyObject *shape);
PyObject *qnp_transpose(QArray *a, PyObject *axes);
PyObject *qnp_ravel(QArray *a);
PyObject *qnp_broadcast_to(PyObject *a, const qintp *shape, int nd);
int qnp_broadcast_shapes(int n, QArray **arrays, qintp *out_shape, int *out_nd);

/* ---- iteration ------------------------------------------------------- */
/* A flattened, dimension-coalesced walk over up to QNP_MAXOPS operands that
 * share a broadcast shape.  `qnp_iter_next` advances the per-operand pointers
 * and reports how many contiguous inner elements may be processed at once. */
#define QNP_MAXOPS 4
typedef struct {
    int nop;
    int nd;
    qintp shape[QNP_MAXDIMS];
    qintp strides[QNP_MAXOPS][QNP_MAXDIMS];
    char *ptr[QNP_MAXOPS];
    char *base[QNP_MAXOPS];
    qintp index[QNP_MAXDIMS];
    qintp inner_len;
    qintp inner_stride[QNP_MAXOPS];
    qintp total;
    qintp done;
} QIter;

int qnp_iter_init(QIter *it, int nop, QArray **ops, const qintp *shape, int nd);
int qnp_iter_next(QIter *it);

/* ---- module-level helpers used across translation units --------------- */
PyObject *qnp_matmul(PyObject *a, PyObject *b);
void qnp_gemm_f64(const double *A, const double *B, double *C,
                  qintp m, qintp n, qintp k);
int qnp_parse_axis(PyObject *obj, int nd, int *axis);
PyObject *qnp_wrap_scalar_or_array(QArray *a);
PyObject *qnp_shape_tuple(const QArray *a);
int qnp_shape_from_object(PyObject *obj, qintp *shape, int *nd);

/* Registration hooks for the pieces that live in their own files. */
int qnp_add_linalg(PyObject *module, PyObject *dict);
int qnp_add_fft(PyObject *module, PyObject *dict);
int qnp_add_random(PyObject *module);
int qnp_add_sort(PyObject *module);
int qnp_add_shape(PyObject *module);
int qnp_add_reduce(PyObject *module);
int qnp_add_ufunc(PyObject *module);
int qnp_add_index(PyObject *module);

extern PyMethodDef qnp_ufunc_methods[];
extern PyMethodDef qnp_reduce_methods[];
extern PyMethodDef qnp_shape_methods[];
extern PyMethodDef qnp_sort_methods[];
extern PyMethodDef qnp_linalg_methods[];
extern PyMethodDef qnp_fft_methods[];
extern PyMethodDef qnp_random_methods[];
extern PyMethodDef qnp_array_methods[];

/* Error helper: mirrors NumPy's message shapes closely enough that callers
 * matching on text keep working. */
PyObject *qnp_err_broadcast(const qintp *a, int and_, const qintp *b, int bnd);

#ifdef __cplusplus
}
#endif
#endif /* QNP_H */
