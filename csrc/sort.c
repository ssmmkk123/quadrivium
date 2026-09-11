/* Sorting, searching and the order statistics built on them. */
#include "qnp.h"

typedef unsigned char qbool;

/* Comparison used everywhere: NaN sorts last, matching NumPy. */
static inline int less_d(double a, double b) {
    if (isnan(b)) return !isnan(a);
    if (isnan(a)) return 0;
    return a < b;
}

static void insertion_d(double *v, qintp n) {
    for (qintp i = 1; i < n; i++) {
        double x = v[i];
        qintp j = i;
        while (j > 0 && less_d(x, v[j - 1])) { v[j] = v[j - 1]; j--; }
        v[j] = x;
    }
}

static void introsort_d(double *v, qintp n, int depth) {
    while (n > 16) {
        if (depth-- == 0) {   /* pathological input: fall back to a safe sort */
            insertion_d(v, n);
            return;
        }
        /* Median of three, moved to the front as the pivot. */
        qintp mid = n / 2;
        if (less_d(v[mid], v[0])) { double t = v[mid]; v[mid] = v[0]; v[0] = t; }
        if (less_d(v[n - 1], v[0])) { double t = v[n - 1]; v[n - 1] = v[0]; v[0] = t; }
        if (less_d(v[n - 1], v[mid])) { double t = v[n - 1]; v[n - 1] = v[mid]; v[mid] = t; }
        double pivot = v[mid];
        qintp i = 0, j = n - 1;
        while (i <= j) {
            while (less_d(v[i], pivot)) i++;
            while (less_d(pivot, v[j])) j--;
            if (i <= j) {
                double t = v[i]; v[i] = v[j]; v[j] = t;
                i++;
                if (j == 0) break;
                j--;
            }
        }
        if (j + 1 < n - i) { introsort_d(v, j + 1, depth); v += i; n -= i; }
        else { introsort_d(v + i, n - i, depth); n = j + 1; }
    }
    insertion_d(v, n);
}

static void sort_run_d(double *v, qintp n) {
    int depth = 2;
    for (qintp t = n; t > 1; t >>= 1) depth += 2;
    introsort_d(v, n, depth);
}

static void sort_run_i(int64_t *v, qintp n) {
    for (qintp i = 1; i < n; i++) {
        int64_t x = v[i];
        qintp j = i;
        while (j > 0 && x < v[j - 1]) { v[j] = v[j - 1]; j--; }
        v[j] = x;
    }
}

/* Stable merge sort over indices; used for argsort and lexsort. */
typedef struct { const char *base; qintp stride; int dtype; } KeyRef;

static int key_less(const KeyRef *k, qintp a, qintp b) {
    const char *pa = k->base + a * k->stride, *pb = k->base + b * k->stride;
    switch (k->dtype) {
        case QNP_FLOAT32: return less_d(*(const float *)pa, *(const float *)pb);
        case QNP_COMPLEX64: {
            qcomplex za=qnp_read_number(pa,k->dtype),zb=qnp_read_number(pb,k->dtype);
            return za.re!=zb.re ? less_d(za.re,zb.re) : less_d(za.im,zb.im);
        }
        case QNP_FLOAT64: return less_d(*(const double *)pa, *(const double *)pb);
        case QNP_INT64: return *(const int64_t *)pa < *(const int64_t *)pb;
        case QNP_BOOL: return *(const qbool *)pa < *(const qbool *)pb;
        default: {
            const qcomplex *za = (const qcomplex *)pa, *zb = (const qcomplex *)pb;
            if (za->re != zb->re) return za->re < zb->re;
            return za->im < zb->im;
        }
    }
}

static void merge_indices(const KeyRef *k, int64_t *idx, int64_t *scratch,
                          qintp lo, qintp hi) {
    if (hi - lo < 2) return;
    qintp mid = lo + (hi - lo) / 2;
    merge_indices(k, idx, scratch, lo, mid);
    merge_indices(k, idx, scratch, mid, hi);
    qintp i = lo, j = mid, w = lo;
    while (i < mid && j < hi)
        scratch[w++] = key_less(k, idx[j], idx[i]) ? idx[j++] : idx[i++];
    while (i < mid) scratch[w++] = idx[i++];
    while (j < hi) scratch[w++] = idx[j++];
    memcpy(idx + lo, scratch + lo, (size_t)(hi - lo) * sizeof(int64_t));
}

/* Applies `fn` to every 1-D run along `axis`. */
typedef int (*RunFn)(void *ctx, const char *src, qintp sstride, char *dst,
                     qintp dstride, qintp n);

static int walk_runs(QArray *a, QArray *out, int axis, RunFn fn, void *ctx) {
    qintp len = a->shape[axis];
    qintp outer = qnp_size(a) / (len ? len : 1);
    qintp idx[QNP_MAXDIMS] = {0};
    for (qintp k = 0; k < outer; k++) {
        const char *src = a->data;
        char *dst = out->data;
        for (int d = 0, w = 0; d < a->nd; d++) {
            if (d == axis) continue;
            src += idx[w] * a->strides[d];
            dst += idx[w] * out->strides[d];
            w++;
        }
        if (fn(ctx, src, a->strides[axis], dst, out->strides[axis], len) < 0) return -1;
        for (int d = a->nd - 2; d >= 0; d--) {
            qintp dim = 0;
            for (int e = 0, w = 0; e < a->nd; e++) {
                if (e == axis) continue;
                if (w == d) { dim = a->shape[e]; break; }
                w++;
            }
            if (++idx[d] < dim) break;
            idx[d] = 0;
        }
    }
    return 0;
}

typedef struct { int dtype; } SortCtx;

static int sort_run(void *ctxv, const char *src, qintp sstride, char *dst,
                    qintp dstride, qintp n) {
    SortCtx *ctx = (SortCtx *)ctxv;
    int isz = QNP_ITEMSIZE(ctx->dtype);
    for (qintp i = 0; i < n; i++) memcpy(dst + i * dstride, src + i * sstride, (size_t)isz);
    if (dstride != isz) {
        /* Sorting needs a packed run; the destination is contiguous in every
         * call the library makes, so this is a guard rather than a hot path. */
        char *tmp = (char *)PyMem_Malloc((size_t)n * (size_t)isz);
        if (tmp == NULL) { PyErr_NoMemory(); return -1; }
        for (qintp i = 0; i < n; i++) memcpy(tmp + i * isz, dst + i * dstride, (size_t)isz);
        if (sort_run(ctxv,tmp,isz,tmp,isz,n)<0) { PyMem_Free(tmp); return -1; }
        for (qintp i = 0; i < n; i++) memcpy(dst + i * dstride, tmp + i * isz, (size_t)isz);
        PyMem_Free(tmp);
        return 0;
    }
    if (ctx->dtype == QNP_FLOAT64) sort_run_d((double *)dst, n);
    else if (ctx->dtype == QNP_INT64) sort_run_i((int64_t *)dst, n);
    else if (ctx->dtype == QNP_BOOL) {
        qintp ones = 0;
        for (qintp i = 0; i < n; i++) ones += ((qbool *)dst)[i] != 0;
        memset(dst, 0, (size_t)(n - ones));
        memset(dst + (n - ones), 1, (size_t)ones);
    } else {
        /* Complex: lexicographic by (real, imag), via the index sort. */
        KeyRef key = {dst, isz, ctx->dtype};
        int64_t *idx = (int64_t *)PyMem_Malloc((size_t)n * sizeof(int64_t));
        int64_t *scratch = (int64_t *)PyMem_Malloc((size_t)n * sizeof(int64_t));
        char *copy = (char *)PyMem_Malloc((size_t)n * (size_t)isz);
        if (idx == NULL || scratch == NULL || copy == NULL) {
            PyMem_Free(idx); PyMem_Free(scratch); PyMem_Free(copy);
            PyErr_NoMemory();
            return -1;
        }
        memcpy(copy, dst, (size_t)n * (size_t)isz);
        key.base = copy;
        for (qintp i = 0; i < n; i++) idx[i] = i;
        merge_indices(&key, idx, scratch, 0, n);
        for (qintp i = 0; i < n; i++)
            memcpy(dst + i * isz, copy + idx[i] * isz, (size_t)isz);
        PyMem_Free(idx); PyMem_Free(scratch); PyMem_Free(copy);
    }
    return 0;
}

static int argsort_run(void *ctxv, const char *src, qintp sstride, char *dst,
                       qintp dstride, qintp n) {
    SortCtx *ctx = (SortCtx *)ctxv;
    int64_t *idx = (int64_t *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(int64_t));
    int64_t *scratch = (int64_t *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(int64_t));
    if (idx == NULL || scratch == NULL) {
        PyMem_Free(idx); PyMem_Free(scratch);
        PyErr_NoMemory();
        return -1;
    }
    for (qintp i = 0; i < n; i++) idx[i] = i;
    KeyRef key = {src, sstride, ctx->dtype};
    merge_indices(&key, idx, scratch, 0, n);
    for (qintp i = 0; i < n; i++) *(int64_t *)(dst + i * dstride) = idx[i];
    PyMem_Free(idx);
    PyMem_Free(scratch);
    return 0;
}

static PyObject *sort_entry(PyObject *args, PyObject *kwds, int want_indices) {
    PyObject *ao, *axis_o = NULL, *kind = NULL;
    static char *kwlist[] = {"a", "axis", "kind", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist, &ao, &axis_o, &kind))
        return NULL;
    QArray *a0 = qnp_from_any(ao, -1, 0);
    if (a0 == NULL) return NULL;
    QArray *a = a0;
    PyObject *flat = NULL;
    if (axis_o == Py_None) {
        flat = qnp_ravel(a0);
        Py_DECREF(a0);
        if (flat == NULL) return NULL;
        a = (QArray *)flat;
    }
    if (a->nd == 0) {
        PyErr_SetString(PyExc_ValueError, "sort: input must be at least one dimensional");
        Py_DECREF(a);
        return NULL;
    }
    int axis = a->nd - 1;
    if (axis_o != NULL && axis_o != Py_None && qnp_parse_axis(axis_o, a->nd, &axis) < 0) {
        Py_DECREF(a);
        return NULL;
    }
    QArray *out = qnp_new(a->nd, a->shape, want_indices ? QNP_INT64 : a->dtype);
    if (out == NULL) { Py_DECREF(a); return NULL; }
    SortCtx ctx = {a->dtype};
    int rc = walk_runs(a, out, axis, want_indices ? argsort_run : sort_run, &ctx);
    Py_DECREF(a);
    if (rc < 0) { Py_DECREF(out); return NULL; }
    return (PyObject *)out;
}

static PyObject *py_sort(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self; return sort_entry(args, kwds, 0);
}
static PyObject *py_argsort(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self; return sort_entry(args, kwds, 1);
}

static PyObject *py_lexsort(PyObject *self, PyObject *arg) {
    (void)self;
    PyObject *fast = PySequence_Fast(arg, "lexsort expects a sequence of keys");
    if (fast == NULL) return NULL;
    Py_ssize_t nkeys = PySequence_Fast_GET_SIZE(fast);
    if (nkeys == 0) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "need at least one key to sort by");
        return NULL;
    }
    QArray **keys = (QArray **)PyMem_Calloc((size_t)nkeys, sizeof(QArray *));
    if (keys == NULL) { Py_DECREF(fast); return PyErr_NoMemory(); }
    qintp n = -1;
    int rc = 0;
    for (Py_ssize_t i = 0; i < nkeys; i++) {
        QArray *k = qnp_from_any(PySequence_Fast_GET_ITEM(fast, i), -1, 0);
        if (k == NULL) { rc = -1; break; }
        keys[i] = qnp_ascontiguous(k);
        Py_DECREF(k);
        if (keys[i] == NULL) { rc = -1; break; }
        if (n < 0) n = qnp_size(keys[i]);
        else if (qnp_size(keys[i]) != n) {
            PyErr_SetString(PyExc_ValueError, "all keys need to be the same shape");
            rc = -1;
            break;
        }
    }
    Py_DECREF(fast);
    QArray *out = NULL;
    if (rc == 0) {
        out = qnp_new(1, &n, QNP_INT64);
        if (out == NULL) rc = -1;
    }
    if (rc == 0) {
        int64_t *idx = (int64_t *)out->data;
        int64_t *scratch = (int64_t *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(int64_t));
        if (scratch == NULL) { rc = -1; PyErr_NoMemory(); }
        else {
            for (qintp i = 0; i < n; i++) idx[i] = i;
            /* The last key is the primary one, so the least significant key is
             * sorted first and the primary last; the merge sort's stability
             * carries each earlier ordering through the next pass. */
            for (Py_ssize_t k = 0; k < nkeys; k++) {
                KeyRef key = {keys[k]->data, QNP_ITEMSIZE(keys[k]->dtype), keys[k]->dtype};
                merge_indices(&key, idx, scratch, 0, n);
            }
            PyMem_Free(scratch);
        }
    }
    for (Py_ssize_t i = 0; i < nkeys; i++) Py_XDECREF(keys[i]);
    PyMem_Free(keys);
    if (rc < 0) { Py_XDECREF(out); return NULL; }
    return (PyObject *)out;
}

static PyObject *py_searchsorted(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *vo;
    const char *side = "left";
    static char *kwlist[] = {"a", "v", "side", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|s:searchsorted", kwlist, &ao, &vo, &side))
        return NULL;
    QArray *a0 = qnp_from_any(ao, QNP_FLOAT64, 0);
    QArray *v0 = qnp_from_any(vo, QNP_FLOAT64, 0);
    if (a0 == NULL || v0 == NULL) { Py_XDECREF(a0); Py_XDECREF(v0); return NULL; }
    int dt = qnp_promote(a0->dtype, v0->dtype);
    dt = qnp_promote(dt, QNP_FLOAT64);
    QArray *af = qnp_astype(a0, dt, 0), *vf = qnp_astype(v0, dt, 0);
    Py_DECREF(a0); Py_DECREF(v0);
    if (af == NULL || vf == NULL) { Py_XDECREF(af); Py_XDECREF(vf); return NULL; }
    QArray *a = qnp_ascontiguous(af), *v = qnp_ascontiguous(vf);
    Py_DECREF(af); Py_DECREF(vf);
    if (a == NULL || v == NULL) { Py_XDECREF(a); Py_XDECREF(v); return NULL; }
    int right = (side[0] == 'r');
    qintp n = qnp_size(a), m = qnp_size(v);
    QArray *out = qnp_new(v->nd, v->shape, QNP_INT64);
    if (out == NULL) { Py_DECREF(a); Py_DECREF(v); return NULL; }
    const double *ap = (const double *)a->data;
    const double *vp = (const double *)v->data;
    int64_t *op = (int64_t *)out->data;
    for (qintp i = 0; i < m; i++) {
        double x = vp[i];
        qintp lo = 0, hi = n;
        while (lo < hi) {
            qintp mid = lo + (hi - lo) / 2;
            int go_right = right ? !(x < ap[mid]) : (ap[mid] < x);
            if (go_right) lo = mid + 1; else hi = mid;
        }
        op[i] = lo;
    }
    Py_DECREF(a);
    Py_DECREF(v);
    return qnp_wrap_scalar_or_array(out);
}

static PyObject *py_bincount(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *xo, *weights = Py_None;
    Py_ssize_t minlength = 0;
    static char *kwlist[] = {"x", "weights", "minlength", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|On:bincount", kwlist,
                                     &xo, &weights, &minlength)) return NULL;
    QArray *x0 = qnp_from_any(xo, QNP_INT64, 1);
    if (x0 == NULL) return NULL;
    QArray *x = qnp_ascontiguous(x0);
    Py_DECREF(x0);
    if (x == NULL) return NULL;
    qintp n = qnp_size(x);
    const int64_t *xp = (const int64_t *)x->data;
    int64_t hi = 0;
    for (qintp i = 0; i < n; i++) {
        if (xp[i] < 0) {
            Py_DECREF(x);
            PyErr_SetString(PyExc_ValueError, "'list' argument must have no negative elements");
            return NULL;
        }
        if (xp[i] > hi) hi = xp[i];
    }
    qintp length = n ? hi + 1 : 0;
    if (length < minlength) length = minlength;
    QArray *out;
    if (weights == Py_None) {
        out = qnp_new(1, &length, QNP_INT64);
        if (out == NULL) { Py_DECREF(x); return NULL; }
        memset(out->data, 0, (size_t)length * sizeof(int64_t));
        int64_t *op = (int64_t *)out->data;
        for (qintp i = 0; i < n; i++) op[xp[i]]++;
    } else {
        QArray *w0 = qnp_from_any(weights, QNP_FLOAT64, 1);
        if (w0 == NULL) { Py_DECREF(x); return NULL; }
        QArray *w = qnp_ascontiguous(w0);
        Py_DECREF(w0);
        if (w == NULL) { Py_DECREF(x); return NULL; }
        if (qnp_size(w) != n) {
            Py_DECREF(x); Py_DECREF(w);
            PyErr_SetString(PyExc_ValueError, "The weights and list don't have the same length.");
            return NULL;
        }
        out = qnp_new(1, &length, QNP_FLOAT64);
        if (out == NULL) { Py_DECREF(x); Py_DECREF(w); return NULL; }
        memset(out->data, 0, (size_t)length * sizeof(double));
        double *op = (double *)out->data;
        const double *wp = (const double *)w->data;
        for (qintp i = 0; i < n; i++) op[xp[i]] += wp[i];
        Py_DECREF(w);
    }
    Py_DECREF(x);
    return (PyObject *)out;
}

static PyObject *py_clip(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *lo = Py_None, *hi = Py_None, *out = Py_None;
    static char *kwlist[] = {"a", "a_min", "a_max", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOO:clip", kwlist, &ao, &lo, &hi, &out))
        return NULL;
    PyObject *cur = ao;
    Py_INCREF(cur);
    if (lo != Py_None) {
        PyObject *r = qnp_binary_op(QOP_MAXIMUM, cur, lo, NULL, NULL);
        Py_DECREF(cur);
        if (r == NULL) return NULL;
        cur = r;
    }
    if (hi != Py_None) {
        PyObject *r = qnp_binary_op(QOP_MINIMUM, cur, hi, out == Py_None ? NULL : out, NULL);
        Py_DECREF(cur);
        return r;
    }
    if (out != Py_None) {
        PyObject *r = qnp_binary_op(QOP_ADD, cur, PyLong_FromLong(0), out, NULL);
        Py_DECREF(cur);
        return r;
    }
    if (cur == ao) {   /* neither bound given: NumPy still returns a copy */
        QArray *a = qnp_from_any(ao, -1, 0);
        Py_DECREF(cur);
        if (a == NULL) return NULL;
        QArray *c = qnp_astype(a, a->dtype, 1);
        Py_DECREF(a);
        return (PyObject *)c;
    }
    return cur;
}

/* Linear-interpolation quantiles, NumPy's default method. */
static PyObject *py_quantile(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *qo;
    const char *method = "linear";
    static char *kwlist[] = {"a", "q", "method", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|s:quantile", kwlist, &ao, &qo, &method))
        return NULL;
    QArray *a0 = qnp_from_any(ao, QNP_FLOAT64, 1);
    if (a0 == NULL) return NULL;
    PyObject *flat = qnp_ravel(a0);
    Py_DECREF(a0);
    if (flat == NULL) return NULL;
    QArray *sorted = qnp_astype((QArray *)flat, QNP_FLOAT64, 1);
    Py_DECREF(flat);
    if (sorted == NULL) return NULL;
    qintp n = qnp_size(sorted);
    if (n == 0) {
        Py_DECREF(sorted);
        PyErr_SetString(PyExc_ValueError, "quantile of an empty array");
        return NULL;
    }
    sort_run_d((double *)sorted->data, n);
    const double *sp = (const double *)sorted->data;
    QArray *q = qnp_from_any(qo, QNP_FLOAT64, 1);
    if (q == NULL) { Py_DECREF(sorted); return NULL; }
    QArray *cq = qnp_ascontiguous(q);
    Py_DECREF(q);
    if (cq == NULL) { Py_DECREF(sorted); return NULL; }
    QArray *out = qnp_new(cq->nd, cq->shape, QNP_FLOAT64);
    if (out == NULL) { Py_DECREF(sorted); Py_DECREF(cq); return NULL; }
    qintp m = qnp_size(cq);
    const double *qp = (const double *)cq->data;
    double *op = (double *)out->data;
    int lower = !strcmp(method, "lower");
    int higher = !strcmp(method, "higher");
    int nearest = !strcmp(method, "nearest");
    int midpoint = !strcmp(method, "midpoint");
    for (qintp i = 0; i < m; i++) {
        double pos = qp[i] * (double)(n - 1);
        qintp lo = (qintp)floor(pos);
        qintp hi = (qintp)ceil(pos);
        if (lo < 0) lo = 0;
        if (hi > n - 1) hi = n - 1;
        double frac = pos - (double)lo;
        if (lower) op[i] = sp[lo];
        else if (higher) op[i] = sp[hi];
        else if (nearest) op[i] = sp[frac < 0.5 ? lo : hi];
        else if (midpoint) op[i] = 0.5 * (sp[lo] + sp[hi]);
        else op[i] = sp[lo] + frac * (sp[hi] - sp[lo]);
    }
    Py_DECREF(sorted);
    Py_DECREF(cq);
    return qnp_wrap_scalar_or_array(out);
}

PyMethodDef qnp_sort_methods[] = {
    {"sort", (PyCFunction)py_sort, METH_VARARGS | METH_KEYWORDS, "Sorted copy of an array."},
    {"argsort", (PyCFunction)py_argsort, METH_VARARGS | METH_KEYWORDS, "Indices that would sort an array."},
    {"lexsort", py_lexsort, METH_O, "Indirect sort on several keys."},
    {"searchsorted", (PyCFunction)py_searchsorted, METH_VARARGS | METH_KEYWORDS, "Insertion points in a sorted array."},
    {"bincount", (PyCFunction)py_bincount, METH_VARARGS | METH_KEYWORDS, "Count occurrences of each value."},
    {"clip", (PyCFunction)py_clip, METH_VARARGS | METH_KEYWORDS, "Limit values to an interval."},
    {"quantile", (PyCFunction)py_quantile, METH_VARARGS | METH_KEYWORDS, "Quantiles of the data."},
    {NULL}
};
