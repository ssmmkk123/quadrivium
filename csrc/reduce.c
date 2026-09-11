/* Reductions and accumulations.
 *
 * Floating-point sums use the same pairwise scheme and block size as NumPy, so
 * a sum here and a sum there agree bit for bit rather than merely to within a
 * tolerance -- which is what makes the numerical results of the library
 * reproducible across the change of backend.
 */
#include "qnp.h"

typedef unsigned char qbool;

#define PW_BLOCKSIZE 128

double qnp_pairwise_sum_f64(const double *a, qintp n, qintp stride) {
    qintp i;
    if (n < 8) {
        double res = 0.0;
        for (i = 0; i < n; i++) res += a[i * stride];
        return res;
    }
    if (n <= PW_BLOCKSIZE) {
        double r[8];
        for (i = 0; i < 8; i++) r[i] = a[i * stride];
        for (i = 8; i < n - (n % 8); i += 8) {
            r[0] += a[(i + 0) * stride];
            r[1] += a[(i + 1) * stride];
            r[2] += a[(i + 2) * stride];
            r[3] += a[(i + 3) * stride];
            r[4] += a[(i + 4) * stride];
            r[5] += a[(i + 5) * stride];
            r[6] += a[(i + 6) * stride];
            r[7] += a[(i + 7) * stride];
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i * stride];
        return res;
    }
    qintp n2 = n / 2;
    n2 -= n2 % 8;
    return qnp_pairwise_sum_f64(a, n2, stride) +
           qnp_pairwise_sum_f64(a + n2 * stride, n - n2, stride);
}

static qcomplex pairwise_sum_c128(const qcomplex *a, qintp n, qintp stride) {
    qintp i;
    if (n < 8) {
        qcomplex res = qc(0.0, 0.0);
        for (i = 0; i < n; i++) res = qc_add(res, a[i * stride]);
        return res;
    }
    if (n <= PW_BLOCKSIZE) {
        qcomplex r[8];
        for (i = 0; i < 8; i++) r[i] = a[i * stride];
        for (i = 8; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] = qc_add(r[k], a[(i + k) * stride]);
        qcomplex res = qc_add(qc_add(qc_add(r[0], r[1]), qc_add(r[2], r[3])),
                              qc_add(qc_add(r[4], r[5]), qc_add(r[6], r[7])));
        for (; i < n; i++) res = qc_add(res, a[i * stride]);
        return res;
    }
    qintp n2 = n / 2;
    n2 -= n2 % 8;
    return qc_add(pairwise_sum_c128(a, n2, stride),
                  pairwise_sum_c128(a + n2 * stride, n - n2, stride));
}

/* ---- shaping the reduction ------------------------------------------- */

typedef struct {
    QArray *source;      /* owned; reduced axes are trailing and fused */
    int nkeep;
    qintp keep_shape[QNP_MAXDIMS];
    qintp keep_stride[QNP_MAXDIMS];
    qintp run;           /* elements per reduction */
    qintp run_stride;    /* byte stride between them */
    qintp out_shape[QNP_MAXDIMS];
    int out_nd;
} RedPlan;

static void plan_free(RedPlan *p) { Py_CLEAR(p->source); }

static int plan_build(RedPlan *p, QArray *a, PyObject *axis_obj, int keepdims) {
    int red[QNP_MAXDIMS] = {0};
    int nred = 0;
    if (axis_obj == NULL || axis_obj == Py_None) {
        for (int i = 0; i < a->nd; i++) red[i] = 1;
        nred = a->nd;
    } else if (PyTuple_Check(axis_obj) || PyList_Check(axis_obj)) {
        PyObject *seq = PySequence_Fast(axis_obj, "axis must be an int or a tuple of ints");
        if (seq == NULL) return -1;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        for (Py_ssize_t i = 0; i < n; i++) {
            int ax;
            if (qnp_parse_axis(PySequence_Fast_GET_ITEM(seq, i), a->nd, &ax) < 0) {
                Py_DECREF(seq);
                return -1;
            }
            if (red[ax]) {
                Py_DECREF(seq);
                PyErr_SetString(PyExc_ValueError, "duplicate value in 'axis'");
                return -1;
            }
            red[ax] = 1;
            nred++;
        }
        Py_DECREF(seq);
    } else {
        int ax;
        if (qnp_parse_axis(axis_obj, a->nd, &ax) < 0) return -1;
        red[ax] = 1;
        nred = 1;
    }
    int perm[QNP_MAXDIMS];
    int w = 0;
    p->nkeep = 0;
    p->out_nd = 0;
    for (int i = 0; i < a->nd; i++) {
        if (red[i]) continue;
        perm[w++] = i;
        p->keep_shape[p->nkeep] = a->shape[i];
        p->keep_stride[p->nkeep] = a->strides[i];
        p->nkeep++;
    }
    for (int i = 0; i < a->nd; i++) if (red[i]) perm[w++] = i;
    /* Output shape: reduced axes drop out, or become length 1 with keepdims. */
    for (int i = 0; i < a->nd; i++) {
        if (red[i]) { if (keepdims) p->out_shape[p->out_nd++] = 1; }
        else p->out_shape[p->out_nd++] = a->shape[i];
    }
    qintp rshape[QNP_MAXDIMS], rstride[QNP_MAXDIMS];
    for (int i = 0; i < nred; i++) {
        rshape[i] = a->shape[perm[p->nkeep + i]];
        rstride[i] = a->strides[perm[p->nkeep + i]];
    }
    /* Fuse the reduced axes into one run where memory allows. */
    int m = nred;
    for (int i = m - 1; i > 0; i--) {
        if (rstride[i - 1] != rstride[i] * rshape[i]) continue;
        rshape[i - 1] *= rshape[i];
        rstride[i - 1] = rstride[i];
        for (int j = i; j < m - 1; j++) { rshape[j] = rshape[j + 1]; rstride[j] = rstride[j + 1]; }
        m--;
    }
    if (m <= 1) {
        Py_INCREF(a);
        p->source = a;
        p->run = m == 0 ? 1 : rshape[0];
        p->run_stride = m == 0 ? 0 : rstride[0];
        return 0;
    }
    /* Non-contiguous reduced block: one repack buys a flat inner loop. */
    PyObject *tv = qnp_transpose(a, NULL);
    if (tv == NULL) return -1;
    qintp pshape[QNP_MAXDIMS], pstride[QNP_MAXDIMS];
    for (int i = 0; i < a->nd; i++) {
        pshape[i] = a->shape[perm[i]];
        pstride[i] = a->strides[perm[i]];
    }
    Py_DECREF(tv);
    QArray *view = qnp_new_view(a, a->data, a->nd, pshape, pstride, a->dtype);
    if (view == NULL) return -1;
    QArray *packed = qnp_astype(view, a->dtype, 1);
    Py_DECREF(view);
    if (packed == NULL) return -1;
    p->source = packed;
    for (int i = 0; i < p->nkeep; i++) {
        p->keep_shape[i] = packed->shape[i];
        p->keep_stride[i] = packed->strides[i];
    }
    qintp run = 1;
    for (int i = p->nkeep; i < packed->nd; i++) run *= packed->shape[i];
    p->run = run;
    p->run_stride = QNP_ITEMSIZE(a->dtype);
    return 0;
}

/* ---- the kernels ------------------------------------------------------ */

static int reduce_run(int kind, int dtype, const char *p, qintp stride, qintp n,
                      char *out, int out_dtype) {
    if (dtype >= QNP_FLOAT32) {
        if (!n && (kind==QRED_MAX || kind==QRED_MIN || kind==QRED_ARGMAX || kind==QRED_ARGMIN)) {
            PyErr_SetString(PyExc_ValueError,"zero-size array to reduction operation with no identity"); return -1;
        }
        qcomplex acc=qc(kind==QRED_PROD ? 1 : 0,0), correction=qc(0,0), best=qc(0,0);
        qintp at=0; int64_t hits=0;
        for(qintp i=0;i<n;i++) {
            qcomplex z=qnp_read_number(p+i*stride,dtype);
            hits += z.re!=0 || z.im!=0;
            if(kind==QRED_SUM || kind==QRED_MEAN) {
                if(isfinite(acc.re)&&isfinite(acc.im)&&isfinite(z.re)&&isfinite(z.im)) {
                    qcomplex y=qc_sub(z,correction), t=qc_add(acc,y);
                    correction=qc_sub(qc_sub(t,acc),y); acc=t;
                } else {acc=qc_add(acc,z);correction=qc(0,0);}
            } else if(kind==QRED_PROD) acc=dtype==QNP_FLOAT32?qc(acc.re*z.re,0):qc_mul(acc,z);
            else {
                int greater=z.re>best.re || (z.re==best.re && z.im>best.im);
                int smaller=z.re<best.re || (z.re==best.re && z.im<best.im);
                if(i==0 || (!(isnan(best.re)||isnan(best.im)) && (isnan(z.re)||isnan(z.im)|| ((kind==QRED_MAX||kind==QRED_ARGMAX)?greater:smaller)))) {best=z;at=i;}
            }
        }
        if(kind==QRED_MEAN) acc=n?qc(acc.re/n,acc.im/n):qc(NAN,NAN);
        if(kind==QRED_SUM || kind==QRED_MEAN || kind==QRED_PROD) qnp_write_number(out,out_dtype,acc);
        else if(kind==QRED_ANY) *(qbool *)out=hits!=0;
        else if(kind==QRED_ALL) *(qbool *)out=hits==n;
        else if(kind==QRED_COUNT_NONZERO) *(int64_t *)out=hits;
        else if(kind==QRED_ARGMAX || kind==QRED_ARGMIN) *(int64_t *)out=at;
        else qnp_write_number(out,out_dtype,best);
        return 0;
    }
    qintp es = stride / QNP_ITEMSIZE(dtype);
    switch (kind) {
        case QRED_SUM: case QRED_MEAN: {
            /* The mean of nothing is undefined, not zero: an empty count has
             * to surface as NaN so missing data cannot pass as a real value. */
            if (dtype == QNP_FLOAT64) {
                double s = qnp_pairwise_sum_f64((const double *)p, n, es);
                if (kind != QRED_MEAN) *(double *)out = s;
                else *(double *)out = n ? s / (double)n : (double)NAN;
            } else if (dtype == QNP_COMPLEX128) {
                qcomplex s = pairwise_sum_c128((const qcomplex *)p, n, es);
                if (kind == QRED_MEAN) {
                    if (n) { s.re /= (double)n; s.im /= (double)n; }
                    else { s.re = (double)NAN; s.im = (double)NAN; }
                }
                *(qcomplex *)out = s;
            } else if (dtype == QNP_INT64) {
                int64_t s = 0;
                const int64_t *v = (const int64_t *)p;
                for (qintp i = 0; i < n; i++) s += v[i * es];
                if (kind == QRED_MEAN) *(double *)out = n ? (double)s / (double)n : (double)NAN;
                else *(int64_t *)out = s;
            } else {
                int64_t s = 0;
                const qbool *v = (const qbool *)p;
                for (qintp i = 0; i < n; i++) s += v[i * es];
                if (kind == QRED_MEAN) *(double *)out = n ? (double)s / (double)n : (double)NAN;
                else *(int64_t *)out = s;
            }
            return 0;
        }
        case QRED_PROD: {
            if (dtype == QNP_FLOAT64) {
                double s = 1.0;
                const double *v = (const double *)p;
                for (qintp i = 0; i < n; i++) s *= v[i * es];
                *(double *)out = s;
            } else if (dtype == QNP_COMPLEX128) {
                qcomplex s = qc(1.0, 0.0);
                const qcomplex *v = (const qcomplex *)p;
                for (qintp i = 0; i < n; i++) s = qc_mul(s, v[i * es]);
                *(qcomplex *)out = s;
            } else {
                int64_t s = 1;
                for (qintp i = 0; i < n; i++)
                    s *= (dtype == QNP_INT64) ? ((const int64_t *)p)[i * es]
                                              : (int64_t)((const qbool *)p)[i * es];
                *(int64_t *)out = s;
            }
            return 0;
        }
        case QRED_MAX: case QRED_MIN: case QRED_ARGMAX: case QRED_ARGMIN: {
            if (n == 0) {
                PyErr_SetString(PyExc_ValueError,
                                "zero-size array to reduction operation with no identity");
                return -1;
            }
            int want_max = (kind == QRED_MAX || kind == QRED_ARGMAX);
            int want_arg = (kind == QRED_ARGMAX || kind == QRED_ARGMIN);
            qintp best = 0;
            if (dtype == QNP_FLOAT64) {
                const double *v = (const double *)p;
                double b = v[0];
                for (qintp i = 1; i < n; i++) {
                    double x = v[i * es];
                    if (isnan(b)) break;         /* NaN wins, as in NumPy */
                    if (isnan(x) || (want_max ? x > b : x < b)) { b = x; best = i; }
                }
                if (!want_arg) *(double *)out = b;
            } else if (dtype == QNP_INT64) {
                const int64_t *v = (const int64_t *)p;
                int64_t b = v[0];
                for (qintp i = 1; i < n; i++) {
                    int64_t x = v[i * es];
                    if (want_max ? x > b : x < b) { b = x; best = i; }
                }
                if (!want_arg) *(int64_t *)out = b;
            } else if (dtype == QNP_BOOL) {
                const qbool *v = (const qbool *)p;
                qbool b = v[0];
                for (qintp i = 1; i < n; i++) {
                    qbool x = v[i * es];
                    if (want_max ? x > b : x < b) { b = x; best = i; }
                }
                if (!want_arg) *(qbool *)out = b;
            } else {
                const qcomplex *v = (const qcomplex *)p;
                qcomplex b = v[0];
                for (qintp i = 1; i < n; i++) {
                    qcomplex x = v[i * es];
                    int gt = x.re > b.re || (x.re == b.re && x.im > b.im);
                    if (want_max ? gt : !gt && (x.re != b.re || x.im != b.im)) { b = x; best = i; }
                }
                if (!want_arg) *(qcomplex *)out = b;
            }
            if (want_arg) *(int64_t *)out = best;
            (void)out_dtype;
            return 0;
        }
        case QRED_ANY: case QRED_ALL: case QRED_COUNT_NONZERO: {
            int64_t hits = 0;
            for (qintp i = 0; i < n; i++) {
                int nz;
                switch (dtype) {
                    case QNP_BOOL: nz = ((const qbool *)p)[i * es] != 0; break;
                    case QNP_INT64: nz = ((const int64_t *)p)[i * es] != 0; break;
                    case QNP_FLOAT64: nz = ((const double *)p)[i * es] != 0.0; break;
                    default: {
                        qcomplex z = ((const qcomplex *)p)[i * es];
                        nz = (z.re != 0.0 || z.im != 0.0);
                        break;
                    }
                }
                if (kind == QRED_ANY && nz) { *(qbool *)out = 1; return 0; }
                if (kind == QRED_ALL && !nz) { *(qbool *)out = 0; return 0; }
                hits += nz;
            }
            if (kind == QRED_ANY) *(qbool *)out = 0;
            else if (kind == QRED_ALL) *(qbool *)out = 1;
            else *(int64_t *)out = hits;
            return 0;
        }
    }
    PyErr_SetString(PyExc_RuntimeError, "unknown reduction");
    return -1;
}

static int result_dtype(int kind, int dtype) {
    switch (kind) {
        case QRED_SUM: case QRED_PROD:
            return dtype <= QNP_INT64 ? QNP_INT64 : dtype;
        case QRED_MEAN:
            return dtype <= QNP_INT64 ? QNP_FLOAT64 : dtype;
        case QRED_ANY: case QRED_ALL: return QNP_BOOL;
        case QRED_ARGMAX: case QRED_ARGMIN: case QRED_COUNT_NONZERO: return QNP_INT64;
        default: return dtype;
    }
}

PyObject *qnp_reduce(int kind, PyObject *ao, PyObject *axis_obj, PyObject *out_obj,
                     int keepdims) {
    QArray *a = qnp_from_any(ao, -1, 0);
    if (a == NULL) return NULL;
    RedPlan plan;
    memset(&plan, 0, sizeof(plan));
    if (plan_build(&plan, a, axis_obj, keepdims) < 0) { Py_DECREF(a); return NULL; }
    Py_DECREF(a);
    QArray *src = plan.source;
    int odt = result_dtype(kind, src->dtype);
    QArray *out = qnp_new(plan.out_nd, plan.out_shape, odt);
    if (out == NULL) { plan_free(&plan); return NULL; }
    qintp outer = 1;
    for (int i = 0; i < plan.nkeep; i++) outer *= plan.keep_shape[i];
    int osz = QNP_ITEMSIZE(odt);
    qintp idx[QNP_MAXDIMS] = {0};
    const char *base = src->data;
    char *dst = out->data;
    int rc = 0;
    for (qintp k = 0; k < outer; k++) {
        const char *p = base;
        for (int d = 0; d < plan.nkeep; d++) p += idx[d] * plan.keep_stride[d];
        if (reduce_run(kind, src->dtype, p, plan.run_stride, plan.run, dst, odt) < 0) {
            rc = -1;
            break;
        }
        dst += osz;
        for (int d = plan.nkeep - 1; d >= 0; d--) {
            if (++idx[d] < plan.keep_shape[d]) break;
            idx[d] = 0;
        }
    }
    plan_free(&plan);
    if (rc < 0) { Py_DECREF(out); return NULL; }
    if (out_obj != NULL && out_obj != Py_None) {
        if (!QArray_Check(out_obj)) {
            Py_DECREF(out);
            PyErr_SetString(PyExc_TypeError, "output must be an array");
            return NULL;
        }
        rc = qnp_copy_into((QArray *)out_obj, out);
        Py_DECREF(out);
        if (rc < 0) return NULL;
        Py_INCREF(out_obj);
        return out_obj;
    }
    return qnp_wrap_scalar_or_array(out);
}

/* ---- variance and standard deviation ---------------------------------- */

PyObject *qnp_moment(PyObject *ao, PyObject *axis_obj, double ddof, int want_std,
                     int keepdims) {
    QArray *a = qnp_from_any(ao, -1, 0);
    if (a == NULL) return NULL;
    int complex_input = qnp_is_complex(a->dtype);
    PyObject *mean = qnp_reduce(QRED_MEAN, (PyObject *)a, axis_obj, NULL, 1);
    if (mean == NULL) { Py_DECREF(a); return NULL; }
    PyObject *dev = qnp_binary_op(QOP_SUB, (PyObject *)a, mean, NULL, NULL);
    Py_DECREF(mean);
    if (dev == NULL) { Py_DECREF(a); return NULL; }
    PyObject *sq;
    if (complex_input) {
        PyObject *conj = qnp_unary_op(QOP_CONJ, dev, NULL);
        if (conj == NULL) { Py_DECREF(dev); Py_DECREF(a); return NULL; }
        PyObject *prod = qnp_binary_op(QOP_MUL, dev, conj, NULL, NULL);
        Py_DECREF(conj);
        if (prod == NULL) { Py_DECREF(dev); Py_DECREF(a); return NULL; }
        sq = qnp_unary_op(QOP_REAL, prod, NULL);
        Py_DECREF(prod);
    } else {
        sq = qnp_unary_op(QOP_SQUARE, dev, NULL);
    }
    Py_DECREF(dev);
    if (sq == NULL) { Py_DECREF(a); return NULL; }
    PyObject *total = qnp_reduce(QRED_SUM, sq, axis_obj, NULL, keepdims);
    Py_DECREF(sq);
    if (total == NULL) { Py_DECREF(a); return NULL; }
    /* Count of elements that went into each reduction. */
    qintp count = 1;
    if (axis_obj == NULL || axis_obj == Py_None) count = qnp_size(a);
    else if (PyTuple_Check(axis_obj) || PyList_Check(axis_obj)) {
        PyObject *seq = PySequence_Fast(axis_obj, "axis");
        if (seq == NULL) { Py_DECREF(a); Py_DECREF(total); return NULL; }
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq); i++) {
            int ax;
            if (qnp_parse_axis(PySequence_Fast_GET_ITEM(seq, i), a->nd, &ax) < 0) {
                Py_DECREF(seq); Py_DECREF(a); Py_DECREF(total);
                return NULL;
            }
            count *= a->shape[ax];
        }
        Py_DECREF(seq);
    } else {
        int ax;
        if (qnp_parse_axis(axis_obj, a->nd, &ax) < 0) { Py_DECREF(a); Py_DECREF(total); return NULL; }
        count = a->shape[ax];
    }
    Py_DECREF(a);
    double denom = (double)count - ddof;
    PyObject *divisor = PyFloat_FromDouble(denom);
    if (divisor == NULL) { Py_DECREF(total); return NULL; }
    PyObject *var = qnp_binary_op(QOP_TRUEDIV, total, divisor, NULL, NULL);
    Py_DECREF(total);
    Py_DECREF(divisor);
    if (var == NULL || !want_std) return var;
    PyObject *std = qnp_unary_op(QOP_SQRT, var, NULL);
    Py_DECREF(var);
    return std;
}

/* ---- cumulative operations -------------------------------------------- */

PyObject *qnp_accumulate(int kind, PyObject *ao, PyObject *axis_obj, PyObject *out_obj) {
    QArray *a = qnp_from_any(ao, -1, 0);
    if (a == NULL) return NULL;
    QArray *src;
    if (axis_obj == NULL || axis_obj == Py_None) {
        PyObject *flat = qnp_ravel(a);
        Py_DECREF(a);
        if (flat == NULL) return NULL;
        src = (QArray *)flat;
    } else {
        src = a;
    }
    int axis = src->nd - 1;
    if (axis_obj != NULL && axis_obj != Py_None &&
        qnp_parse_axis(axis_obj, src->nd, &axis) < 0) { Py_DECREF(src); return NULL; }
    if (src->nd == 0) {
        qintp one = 1;
        QArray *r = qnp_new(1, &one, result_dtype(QRED_SUM, src->dtype));
        if (r == NULL) { Py_DECREF(src); return NULL; }
        int rc = qnp_copy_into(r, src);
        Py_DECREF(src);
        if (rc < 0) { Py_DECREF(r); return NULL; }
        return (PyObject *)r;
    }
    int odt = result_dtype(QRED_SUM, src->dtype);
    QArray *out = qnp_new(src->nd, src->shape, odt);
    if (out == NULL) { Py_DECREF(src); return NULL; }
    QArray *csrc = qnp_astype(src, odt, 0);
    Py_DECREF(src);
    if (csrc == NULL) { Py_DECREF(out); return NULL; }
    qintp len = csrc->shape[axis];
    qintp sstride = csrc->strides[axis];
    qintp dstride = out->strides[axis];
    qintp outer = qnp_size(csrc) / (len ? len : 1);
    qintp idx[QNP_MAXDIMS] = {0};
    for (qintp k = 0; k < outer; k++) {
        const char *p = csrc->data;
        char *q = out->data;
        for (int d = 0, w = 0; d < csrc->nd; d++) {
            if (d == axis) continue;
            p += idx[w] * csrc->strides[d];
            q += idx[w] * out->strides[d];
            w++;
        }
        if (odt >= QNP_FLOAT32) {
            if(len) {
                qcomplex acc=qnp_read_number(p,odt);
                qnp_write_number(q,odt,acc);
                for(qintp i=1;i<len;i++) {
                    qcomplex v=qnp_read_number(p+i*sstride,odt);
                    acc=kind==QRED_PROD?(odt==QNP_FLOAT32?qc(acc.re*v.re,0):qc_mul(acc,v)):qc_add(acc,v);
                    qnp_write_number(q+i*dstride,odt,acc);
                }
            }
        } else if (odt == QNP_FLOAT64) {
            /* The first cumulative value is the first input itself. Starting
             * from an identity loses signed zero and contaminates complex
             * infinities with the otherwise unnecessary identity multiply. */
            double acc = *(const double *)p;
            *(double *)q = acc;
            for (qintp i = 1; i < len; i++) {
                double v = *(const double *)(p + i * sstride);
                acc = kind == QRED_PROD ? acc * v : acc + v;
                *(double *)(q + i * dstride) = acc;
            }
        } else if (odt == QNP_COMPLEX128) {
            qcomplex acc = *(const qcomplex *)p;
            *(qcomplex *)q = acc;
            for (qintp i = 1; i < len; i++) {
                qcomplex v = *(const qcomplex *)(p + i * sstride);
                acc = kind == QRED_PROD ? qc_mul(acc, v) : qc_add(acc, v);
                *(qcomplex *)(q + i * dstride) = acc;
            }
        } else {
            /* Integer accumulation wraps modulo 2**64. Unsigned arithmetic
             * makes that behavior defined even when a prefix overflows. */
            uint64_t acc = *(const uint64_t *)p;
            *(uint64_t *)q = acc;
            for (qintp i = 1; i < len; i++) {
                uint64_t v = *(const uint64_t *)(p + i * sstride);
                acc = kind == QRED_PROD ? acc * v : acc + v;
                *(uint64_t *)(q + i * dstride) = acc;
            }
        }
        for (int d = csrc->nd - 2; d >= 0; d--) {
            qintp dim = 0, w = 0;
            for (int e = 0; e < csrc->nd; e++) {
                if (e == axis) continue;
                if (w == d) { dim = csrc->shape[e]; break; }
                w++;
            }
            if (++idx[d] < dim) break;
            idx[d] = 0;
        }
    }
    Py_DECREF(csrc);
    if (out_obj != NULL && out_obj != Py_None) {
        if (!QArray_Check(out_obj)) {
            Py_DECREF(out);
            PyErr_SetString(PyExc_TypeError, "output must be an array");
            return NULL;
        }
        int rc = qnp_copy_into((QArray *)out_obj, out);
        Py_DECREF(out);
        if (rc < 0) return NULL;
        Py_INCREF(out_obj);
        return out_obj;
    }
    return (PyObject *)out;
}

/* ---- reduceat ---------------------------------------------------------- */

/* NumPy's `ufunc.reduceat`: segment i covers `indices[i]` up to the next index
 * (or the end), and a segment that does not advance yields the single element
 * at its start.  Sparse matrix-vector products lean on this, so it is a C loop
 * rather than a Python one over slices. */
static PyObject *qnp_reduceat(int kind, PyObject *ao, PyObject *idxo, int axis) {
    QArray *a0 = qnp_from_any(ao, -1, 0);
    if (a0 == NULL) return NULL;
    if (a0->nd == 0) {
        Py_DECREF(a0);
        PyErr_SetString(PyExc_ValueError, "reduceat needs at least one dimension");
        return NULL;
    }
    if (axis < 0) axis += a0->nd;
    if (axis < 0 || axis >= a0->nd) {
        Py_DECREF(a0);
        PyErr_SetString(PyExc_ValueError, "axis out of bounds for reduceat");
        return NULL;
    }
    QArray *idx0 = qnp_from_any(idxo, QNP_INT64, 1);
    if (idx0 == NULL) { Py_DECREF(a0); return NULL; }
    QArray *idx = qnp_ascontiguous(idx0);
    Py_DECREF(idx0);
    if (idx == NULL) { Py_DECREF(a0); return NULL; }
    int odt = result_dtype(kind, a0->dtype);
    QArray *a = qnp_astype(a0, odt, 0);
    Py_DECREF(a0);
    if (a == NULL) { Py_DECREF(idx); return NULL; }
    qintp segments = qnp_size(idx);
    qintp len = a->shape[axis];
    qintp shape[QNP_MAXDIMS];
    for (int i = 0; i < a->nd; i++) shape[i] = a->shape[i];
    shape[axis] = segments;
    QArray *out = qnp_new(a->nd, shape, odt);
    if (out == NULL) { Py_DECREF(a); Py_DECREF(idx); return NULL; }
    /* Each segment is a strided run along the reduced axis.  Iterate the
     * other dimensions directly: constructing a window, reduction result and
     * destination view for every segment made short segments allocation-bound.
     * Reusing reduce_run retains the exact dtype rules and pairwise sum order. */
    /* The output allocator already checked its shape product. Dividing it
     * avoids overflowing a product of huge dimensions in an empty array. */
    qintp outer = qnp_size(out) / (segments ? segments : 1);
    qintp keep_shape[QNP_MAXDIMS], src_stride[QNP_MAXDIMS], dst_stride[QNP_MAXDIMS];
    int nkeep = 0;
    for (int d = 0; d < a->nd; d++) {
        if (d == axis) continue;
        keep_shape[nkeep] = a->shape[d];
        src_stride[nkeep] = a->strides[d];
        dst_stride[nkeep] = out->strides[d];
        nkeep++;
    }
    qintp coord[QNP_MAXDIMS] = {0};
    const int64_t *ip = (const int64_t *)idx->data;
    for (qintp s = 0; s < segments; s++) {
        int64_t start = ip[s];
        if (start < 0) start += len;
        if (start < 0 || start >= len) {
            PyErr_Format(PyExc_IndexError,
                         "index %lld out of range for reduceat on axis %d of size %zd",
                         (long long)ip[s], axis, len);
            Py_DECREF(a); Py_DECREF(idx); Py_DECREF(out);
            return NULL;
        }
        int64_t next = (s + 1 < segments) ? ip[s + 1] : len;
        if (next < 0) next += len;
        if (next > len) next = len;
        qintp count = (next > start) ? (qintp)(next - start) : 1;
        const char *src = a->data + start * a->strides[axis];
        char *dst = out->data + s * out->strides[axis];
        for (qintp k = 0; k < outer; k++) {
            if (reduce_run(kind, a->dtype, src, a->strides[axis], count,
                           dst, odt) < 0) {
                Py_DECREF(a); Py_DECREF(idx); Py_DECREF(out);
                return NULL;
            }
            /* Advance within the unreduced dimensions without recomputing
             * multidimensional offsets or allocating iterator objects. */
            for (int d = nkeep - 1; d >= 0; d--) {
                if (++coord[d] < keep_shape[d]) {
                    src += src_stride[d];
                    dst += dst_stride[d];
                    break;
                }
                src -= (coord[d] - 1) * src_stride[d];
                dst -= (coord[d] - 1) * dst_stride[d];
                coord[d] = 0;
            }
        }
    }
    Py_DECREF(a);
    Py_DECREF(idx);
    return (PyObject *)out;
}

/* One-dimensional float sums skip dtype dispatch and outer iteration;
 * gapped, reversed and broadcast views use the same pairwise kernel. */
static PyObject *reduceat_fast_sum(QArray *a, QArray *idx) {
    qintp segments = qnp_size(idx);
    qintp len = a->shape[0];
    QArray *out = qnp_new(1, &segments, QNP_FLOAT64);
    if (out == NULL) return NULL;
    const int64_t *ip = (const int64_t *)idx->data;
    const double *values = (const double *)a->data;
    qintp stride = a->strides[0] / (qintp)sizeof(double);
    double *dst = (double *)out->data;
    for (qintp s = 0; s < segments; s++) {
        int64_t start = ip[s];
        if (start < 0) start += len;
        if (start < 0 || start >= len) {
            PyErr_Format(PyExc_IndexError,
                         "index %lld out of range for reduceat on axis 0 of size %zd",
                         (long long)ip[s], len);
            Py_DECREF(out);
            return NULL;
        }
        int64_t next = (s + 1 < segments) ? ip[s + 1] : len;
        if (next < 0) next += len;
        if (next > len) next = len;
        if (next <= start) dst[s] = values[start * stride];
        else dst[s] = qnp_pairwise_sum_f64(values + start * stride,
                                         (qintp)(next - start), stride);
    }
    return (PyObject *)out;
}

static PyObject *py_reduceat(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *idxo;
    int kind = QRED_SUM;
    int axis = 0;
    static char *kwlist[] = {"a", "indices", "axis", "kind", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|ii:reduceat", kwlist,
                                     &ao, &idxo, &axis, &kind)) return NULL;
    if (kind == QRED_SUM && (axis == 0 || axis == -1) && QArray_Check(ao) &&
        ((QArray *)ao)->nd == 1 && ((QArray *)ao)->dtype == QNP_FLOAT64 &&
        qnp_size((QArray *)ao) > 0) {
        QArray *idx0 = qnp_from_any(idxo, QNP_INT64, 1);
        if (idx0 == NULL) return NULL;
        QArray *idx = qnp_ascontiguous(idx0);
        Py_DECREF(idx0);
        if (idx == NULL) return NULL;
        PyObject *result = reduceat_fast_sum((QArray *)ao, idx);
        Py_DECREF(idx);
        return result;
    }
    return qnp_reduceat(kind, ao, idxo, axis);
}

/* ---- Python entry points ---------------------------------------------- */

static PyObject *reduce_entry(int kind, PyObject *args, PyObject *kwds) {
    PyObject *a, *axis = Py_None, *out = Py_None;
    int keepdims = 0;
    static char *kwlist[] = {"a", "axis", "out", "keepdims", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp", kwlist, &a, &axis, &out, &keepdims))
        return NULL;
    return qnp_reduce(kind, a, axis, out, keepdims);
}

#define REDUCE_FN(NAME, KIND)                                                 \
static PyObject *py_##NAME(PyObject *self, PyObject *args, PyObject *kwds) {  \
    (void)self; return reduce_entry(KIND, args, kwds);                        \
}
REDUCE_FN(sum, QRED_SUM)
REDUCE_FN(prod, QRED_PROD)
REDUCE_FN(amax, QRED_MAX)
REDUCE_FN(amin, QRED_MIN)
REDUCE_FN(any, QRED_ANY)
REDUCE_FN(all, QRED_ALL)
REDUCE_FN(argmax, QRED_ARGMAX)
REDUCE_FN(argmin, QRED_ARGMIN)
REDUCE_FN(mean, QRED_MEAN)
REDUCE_FN(count_nonzero, QRED_COUNT_NONZERO)
#undef REDUCE_FN

static PyObject *py_var(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *a, *axis = Py_None;
    double ddof = 0.0;
    int keepdims = 0;
    static char *kwlist[] = {"a", "axis", "ddof", "keepdims", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Odp", kwlist, &a, &axis, &ddof, &keepdims))
        return NULL;
    return qnp_moment(a, axis, ddof, 0, keepdims);
}

static PyObject *py_std(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *a, *axis = Py_None;
    double ddof = 0.0;
    int keepdims = 0;
    static char *kwlist[] = {"a", "axis", "ddof", "keepdims", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Odp", kwlist, &a, &axis, &ddof, &keepdims))
        return NULL;
    return qnp_moment(a, axis, ddof, 1, keepdims);
}

static PyObject *py_cumsum(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *a, *axis = Py_None, *out = Py_None;
    static char *kwlist[] = {"a", "axis", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist, &a, &axis, &out))
        return NULL;
    return qnp_accumulate(QRED_SUM, a, axis, out);
}

static PyObject *py_cumprod(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *a, *axis = Py_None, *out = Py_None;
    static char *kwlist[] = {"a", "axis", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist, &a, &axis, &out))
        return NULL;
    return qnp_accumulate(QRED_PROD, a, axis, out);
}

PyMethodDef qnp_reduce_methods[] = {
    {"sum", (PyCFunction)py_sum, METH_VARARGS | METH_KEYWORDS, "Sum of array elements."},
    {"prod", (PyCFunction)py_prod, METH_VARARGS | METH_KEYWORDS, "Product of array elements."},
    {"amax", (PyCFunction)py_amax, METH_VARARGS | METH_KEYWORDS, "Maximum along an axis."},
    {"amin", (PyCFunction)py_amin, METH_VARARGS | METH_KEYWORDS, "Minimum along an axis."},
    {"any", (PyCFunction)py_any, METH_VARARGS | METH_KEYWORDS, "True if any element is true."},
    {"all", (PyCFunction)py_all, METH_VARARGS | METH_KEYWORDS, "True if all elements are true."},
    {"argmax", (PyCFunction)py_argmax, METH_VARARGS | METH_KEYWORDS, "Index of the maximum."},
    {"argmin", (PyCFunction)py_argmin, METH_VARARGS | METH_KEYWORDS, "Index of the minimum."},
    {"mean", (PyCFunction)py_mean, METH_VARARGS | METH_KEYWORDS, "Arithmetic mean."},
    {"count_nonzero", (PyCFunction)py_count_nonzero, METH_VARARGS | METH_KEYWORDS, "Number of nonzero elements."},
    {"var", (PyCFunction)py_var, METH_VARARGS | METH_KEYWORDS, "Variance."},
    {"std", (PyCFunction)py_std, METH_VARARGS | METH_KEYWORDS, "Standard deviation."},
    {"cumsum", (PyCFunction)py_cumsum, METH_VARARGS | METH_KEYWORDS, "Cumulative sum."},
    {"cumprod", (PyCFunction)py_cumprod, METH_VARARGS | METH_KEYWORDS, "Cumulative product."},
    {"reduceat", (PyCFunction)py_reduceat, METH_VARARGS | METH_KEYWORDS,
     "Reduce over the segments named by an index array."},
    {NULL}
};
