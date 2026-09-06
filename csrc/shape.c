/* Array creation and shape manipulation. */
#include "qnp.h"

typedef unsigned char qbool;

int qnp_parse_axis(PyObject *obj, int nd, int *axis) {
    Py_ssize_t v = PyNumber_AsSsize_t(obj, PyExc_IndexError);
    if (v == -1 && PyErr_Occurred()) return -1;
    if (v < 0) v += nd;
    if (v < 0 || v >= (nd ? nd : 1)) {
        PyErr_Format(PyExc_ValueError,
                     "axis %zd is out of bounds for array of dimension %d",
                     (Py_ssize_t)v, nd);
        return -1;
    }
    *axis = (int)v;
    return 0;
}

static int parse_dtype_arg(PyObject *obj, int fallback) {
    if (obj == NULL || obj == Py_None) return fallback;
    int ok = 1;
    int dt = qnp_dtype_from_object(obj, &ok);
    return ok ? dt : -1;
}

/* ---- creation --------------------------------------------------------- */

static PyObject *make_filled(PyObject *shape_obj, PyObject *dtype_obj, int mode,
                             PyObject *fill) {
    qintp shape[QNP_MAXDIMS];
    int nd;
    if (qnp_shape_from_object(shape_obj, shape, &nd) < 0) return NULL;
    int dtype = parse_dtype_arg(dtype_obj, QNP_FLOAT64);
    if (dtype < 0) return NULL;
    if (mode == 2 && fill != NULL) {
        /* `full` takes its dtype from the fill value when none is given. */
        if (dtype_obj == NULL || dtype_obj == Py_None) {
            int weak;
            int sdt = qnp_scalar_dtype(fill, &weak);
            dtype = (sdt >= 0) ? sdt : QNP_FLOAT64;
            if (dtype == QNP_BOOL && !PyBool_Check(fill)) dtype = QNP_FLOAT64;
        }
    }
    QArray *out = qnp_new(nd, shape, dtype);
    if (out == NULL) return NULL;
    qintp n = qnp_size(out);
    int isz = QNP_ITEMSIZE(dtype);
    if (mode == 0) {
        memset(out->data, 0, (size_t)n * (size_t)isz);
    } else if (mode == 1 || mode == 2) {
        char scratch[16];
        PyObject *one = NULL;
        PyObject *value = fill;
        if (mode == 1) {
            one = PyLong_FromLong(1);
            value = one;
        }
        if (qnp_setitem_ptr(dtype, scratch, value) < 0) {
            Py_XDECREF(one);
            Py_DECREF(out);
            return NULL;
        }
        Py_XDECREF(one);
        if (dtype == QNP_FLOAT64) {
            double v = *(double *)scratch;
            double *p = (double *)out->data;
            for (qintp i = 0; i < n; i++) p[i] = v;
        } else if (dtype == QNP_INT64) {
            int64_t v = *(int64_t *)scratch;
            int64_t *p = (int64_t *)out->data;
            for (qintp i = 0; i < n; i++) p[i] = v;
        } else if (dtype == QNP_BOOL) {
            memset(out->data, *(qbool *)scratch, (size_t)n);
        } else {
            qcomplex v = *(qcomplex *)scratch;
            qcomplex *p = (qcomplex *)out->data;
            for (qintp i = 0; i < n; i++) p[i] = v;
        }
    }
    return (PyObject *)out;
}

#define CREATION_FN(NAME, MODE)                                               \
static PyObject *py_##NAME(PyObject *self, PyObject *args, PyObject *kwds) {  \
    (void)self;                                                               \
    PyObject *shape_obj, *dtype_obj = NULL;                                   \
    static char *kwlist[] = {"shape", "dtype", NULL};                         \
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:" #NAME, kwlist,        \
                                     &shape_obj, &dtype_obj)) return NULL;    \
    return make_filled(shape_obj, dtype_obj, MODE, NULL);                     \
}
CREATION_FN(zeros, 0)
CREATION_FN(ones, 1)
CREATION_FN(empty, 3)
#undef CREATION_FN

static PyObject *py_full(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *shape_obj, *fill, *dtype_obj = NULL;
    static char *kwlist[] = {"shape", "fill_value", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:full", kwlist,
                                     &shape_obj, &fill, &dtype_obj)) return NULL;
    return make_filled(shape_obj, dtype_obj, 2, fill);
}

static PyObject *like_helper(PyObject *args, PyObject *kwds, int mode, const char *name) {
    PyObject *proto, *dtype_obj = NULL, *fill = NULL;
    if (mode == 2) {
        static char *kwlist[] = {"a", "fill_value", "dtype", NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", kwlist,
                                         &proto, &fill, &dtype_obj)) return NULL;
    } else {
        static char *kwlist[] = {"a", "dtype", NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                         &proto, &dtype_obj)) return NULL;
    }
    (void)name;
    QArray *a = qnp_from_any(proto, -1, 0);
    if (a == NULL) return NULL;
    PyObject *shape = qnp_shape_tuple(a);
    if (shape == NULL) { Py_DECREF(a); return NULL; }
    PyObject *dt = dtype_obj;
    PyObject *owned = NULL;
    if (dt == NULL || dt == Py_None) {
        owned = qnp_dtype_object(a->dtype);
        dt = owned;
    }
    PyObject *out = make_filled(shape, dt, mode == 2 ? 2 : mode, fill);
    Py_XDECREF(owned);
    Py_DECREF(shape);
    Py_DECREF(a);
    return out;
}

static PyObject *py_zeros_like(PyObject *s, PyObject *a, PyObject *k) { (void)s; return like_helper(a, k, 0, "zeros_like"); }
static PyObject *py_ones_like(PyObject *s, PyObject *a, PyObject *k) { (void)s; return like_helper(a, k, 1, "ones_like"); }
static PyObject *py_empty_like(PyObject *s, PyObject *a, PyObject *k) { (void)s; return like_helper(a, k, 3, "empty_like"); }
static PyObject *py_full_like(PyObject *s, PyObject *a, PyObject *k) { (void)s; return like_helper(a, k, 2, "full_like"); }

static PyObject *py_array(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *obj, *dtype_obj = NULL, *order = NULL;
    int copy = 1;
    static char *kwlist[] = {"object", "dtype", "copy", "order", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OpO:array", kwlist,
                                     &obj, &dtype_obj, &copy, &order)) return NULL;
    int dtype = parse_dtype_arg(dtype_obj, -1);
    if (dtype == -1 && dtype_obj != NULL && dtype_obj != Py_None) return NULL;
    QArray *a = qnp_from_any(obj, dtype, dtype >= 0);
    if (a == NULL) return NULL;
    if (copy && QArray_Check(obj) && (QArray *)obj == a) {
        QArray *dup = qnp_astype(a, a->dtype, 1);
        Py_DECREF(a);
        return (PyObject *)dup;
    }
    return (PyObject *)a;
}

static PyObject *py_asarray(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *obj, *dtype_obj = NULL;
    static char *kwlist[] = {"a", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:asarray", kwlist,
                                     &obj, &dtype_obj)) return NULL;
    int dtype = parse_dtype_arg(dtype_obj, -1);
    if (dtype == -1 && dtype_obj != NULL && dtype_obj != Py_None) return NULL;
    return (PyObject *)qnp_from_any(obj, dtype, dtype >= 0);
}

static PyObject *py_ascontiguousarray(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *obj, *dtype_obj = NULL;
    static char *kwlist[] = {"a", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:ascontiguousarray", kwlist,
                                     &obj, &dtype_obj)) return NULL;
    int dtype = parse_dtype_arg(dtype_obj, -1);
    if (dtype == -1 && dtype_obj != NULL && dtype_obj != Py_None) return NULL;
    QArray *a = qnp_from_any(obj, dtype, dtype >= 0);
    if (a == NULL) return NULL;
    if (a->nd == 0) {
        qintp one = 1;
        QArray *r = qnp_new(1, &one, a->dtype);
        if (r == NULL) { Py_DECREF(a); return NULL; }
        memcpy(r->data, a->data, (size_t)QNP_ITEMSIZE(a->dtype));
        Py_DECREF(a);
        return (PyObject *)r;
    }
    QArray *c = qnp_ascontiguous(a);
    Py_DECREF(a);
    return (PyObject *)c;
}

static PyObject *py_arange(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *a1, *a2 = NULL, *a3 = NULL, *dtype_obj = NULL;
    static char *kwlist[] = {"start", "stop", "step", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOO:arange", kwlist,
                                     &a1, &a2, &a3, &dtype_obj)) return NULL;
    PyObject *start_o = a2 ? a1 : NULL;
    PyObject *stop_o = a2 ? a2 : a1;
    PyObject *step_o = a3;
    int all_int = 1;
    PyObject *probes[3] = {start_o, stop_o, step_o};
    for (int i = 0; i < 3; i++)
        if (probes[i] != NULL && probes[i] != Py_None && !PyLong_Check(probes[i]))
            all_int = 0;
    double start = 0.0, stop, step = 1.0;
    if (start_o != NULL && (start = PyFloat_AsDouble(start_o)) == -1.0 && PyErr_Occurred())
        return NULL;
    if ((stop = PyFloat_AsDouble(stop_o)) == -1.0 && PyErr_Occurred()) return NULL;
    if (step_o != NULL && step_o != Py_None &&
        (step = PyFloat_AsDouble(step_o)) == -1.0 && PyErr_Occurred()) return NULL;
    if (step == 0.0) {
        PyErr_SetString(PyExc_ValueError, "arange: step cannot be zero");
        return NULL;
    }
    int dtype = parse_dtype_arg(dtype_obj, all_int ? QNP_INT64 : QNP_FLOAT64);
    if (dtype < 0) return NULL;
    double span = (stop - start) / step;
    qintp n = (qintp)ceil(span - 1e-12);
    if (n < 0) n = 0;
    if (all_int) {
        long long istart = (long long)start, istep = (long long)step, istop = (long long)stop;
        long long diff = istop - istart;
        n = (qintp)((diff + (istep > 0 ? istep - 1 : istep + 1)) / istep);
        if (n < 0) n = 0;
    }
    QArray *out = qnp_new(1, &n, dtype);
    if (out == NULL) return NULL;
    for (qintp i = 0; i < n; i++) {
        double v = start + (double)i * step;
        switch (dtype) {
            case QNP_BOOL: ((qbool *)out->data)[i] = v != 0.0; break;
            case QNP_INT64: ((int64_t *)out->data)[i] = (int64_t)((long long)start + (long long)i * (long long)step); break;
            case QNP_FLOAT64: ((double *)out->data)[i] = v; break;
            default: ((qcomplex *)out->data)[i] = qc(v, 0.0); break;
        }
    }
    return (PyObject *)out;
}

static PyObject *py_linspace(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    double start, stop;
    Py_ssize_t num = 50;
    int endpoint = 1;
    PyObject *dtype_obj = NULL;
    static char *kwlist[] = {"start", "stop", "num", "endpoint", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dd|npO:linspace", kwlist,
                                     &start, &stop, &num, &endpoint, &dtype_obj))
        return NULL;
    if (num < 0) {
        PyErr_SetString(PyExc_ValueError, "number of samples must be non-negative");
        return NULL;
    }
    int dtype = parse_dtype_arg(dtype_obj, QNP_FLOAT64);
    if (dtype < 0) return NULL;
    qintp n = num;
    QArray *out = qnp_new(1, &n, dtype);
    if (out == NULL) return NULL;
    double div = endpoint ? (double)(num - 1) : (double)num;
    double step = (num > 1) ? (stop - start) / div : 0.0;
    for (qintp i = 0; i < n; i++) {
        double v = start + (double)i * step;
        if (endpoint && num > 1 && i == n - 1) v = stop;   /* exact endpoint */
        switch (dtype) {
            case QNP_BOOL: ((qbool *)out->data)[i] = v != 0.0; break;
            case QNP_INT64: ((int64_t *)out->data)[i] = (int64_t)v; break;
            case QNP_FLOAT64: ((double *)out->data)[i] = v; break;
            default: ((qcomplex *)out->data)[i] = qc(v, 0.0); break;
        }
    }
    return (PyObject *)out;
}

static PyObject *py_eye(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    Py_ssize_t n, m = -1, k = 0;
    PyObject *dtype_obj = NULL;
    static char *kwlist[] = {"N", "M", "k", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n|nnO:eye", kwlist,
                                     &n, &m, &k, &dtype_obj)) return NULL;
    if (m < 0) m = n;
    int dtype = parse_dtype_arg(dtype_obj, QNP_FLOAT64);
    if (dtype < 0) return NULL;
    qintp shape[2] = {n, m};
    QArray *out = qnp_new(2, shape, dtype);
    if (out == NULL) return NULL;
    memset(out->data, 0, (size_t)(n * m) * (size_t)QNP_ITEMSIZE(dtype));
    for (qintp i = 0; i < n; i++) {
        qintp j = i + k;
        if (j < 0 || j >= m) continue;
        char *p = out->data + (i * m + j) * QNP_ITEMSIZE(dtype);
        switch (dtype) {
            case QNP_BOOL: *(qbool *)p = 1; break;
            case QNP_INT64: *(int64_t *)p = 1; break;
            case QNP_FLOAT64: *(double *)p = 1.0; break;
            default: *(qcomplex *)p = qc(1.0, 0.0); break;
        }
    }
    return (PyObject *)out;
}

/* ---- shape manipulation ----------------------------------------------- */

PyObject *qnp_reshape(QArray *a, PyObject *shape_obj) {
    qintp shape[QNP_MAXDIMS];
    int nd;
    if (qnp_shape_from_object(shape_obj, shape, &nd) < 0) return NULL;
    qintp size = qnp_size(a);
    int unknown = -1;
    qintp known = 1;
    for (int i = 0; i < nd; i++) {
        if (shape[i] == -1) {
            if (unknown >= 0) {
                PyErr_SetString(PyExc_ValueError,
                                "can only specify one unknown dimension");
                return NULL;
            }
            unknown = i;
        } else known *= shape[i];
    }
    if (unknown >= 0) {
        if (known == 0 || size % known != 0) {
            PyObject *sh = qnp_shape_tuple(a);
            PyErr_Format(PyExc_ValueError, "cannot reshape array of size %zd from shape %R",
                         size, sh);
            Py_XDECREF(sh);
            return NULL;
        }
        shape[unknown] = size / known;
        known *= shape[unknown];
    }
    if (known != size) {
        PyObject *sh = qnp_shape_tuple(a);
        PyErr_Format(PyExc_ValueError,
                     "cannot reshape array of size %zd from shape %R into the requested shape",
                     size, sh);
        Py_XDECREF(sh);
        return NULL;
    }
    if (a->flags & QNP_C_CONTIGUOUS) {
        qintp strides[QNP_MAXDIMS];
        qintp s = QNP_ITEMSIZE(a->dtype);
        for (int i = nd - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
        return (PyObject *)qnp_new_view(a, a->data, nd, shape, strides, a->dtype);
    }
    QArray *c = qnp_ascontiguous(a);
    if (c == NULL) return NULL;
    qintp strides[QNP_MAXDIMS];
    qintp s = QNP_ITEMSIZE(a->dtype);
    for (int i = nd - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
    QArray *view = qnp_new_view(c, c->data, nd, shape, strides, a->dtype);
    Py_DECREF(c);
    return (PyObject *)view;
}

PyObject *qnp_ravel(QArray *a) {
    qintp n = qnp_size(a);
    PyObject *shape = PyTuple_New(1);
    if (shape == NULL) return NULL;
    PyTuple_SET_ITEM(shape, 0, PyLong_FromSsize_t(n));
    PyObject *r = qnp_reshape(a, shape);
    Py_DECREF(shape);
    return r;
}

PyObject *qnp_transpose(QArray *a, PyObject *axes_obj) {
    int perm[QNP_MAXDIMS];
    int nd = a->nd;
    if (axes_obj == NULL || axes_obj == Py_None) {
        for (int i = 0; i < nd; i++) perm[i] = nd - 1 - i;
    } else {
        qintp tmp[QNP_MAXDIMS];
        int n;
        if (qnp_shape_from_object(axes_obj, tmp, &n) < 0) return NULL;
        if (n != nd) {
            PyErr_SetString(PyExc_ValueError, "axes don't match array");
            return NULL;
        }
        int seen[QNP_MAXDIMS] = {0};
        for (int i = 0; i < nd; i++) {
            qintp v = tmp[i];
            if (v < 0) v += nd;
            if (v < 0 || v >= nd || seen[v]) {
                PyErr_SetString(PyExc_ValueError, "repeated or out-of-range axis in transpose");
                return NULL;
            }
            seen[v] = 1;
            perm[i] = (int)v;
        }
    }
    qintp shape[QNP_MAXDIMS], strides[QNP_MAXDIMS];
    for (int i = 0; i < nd; i++) {
        shape[i] = a->shape[perm[i]];
        strides[i] = a->strides[perm[i]];
    }
    return (PyObject *)qnp_new_view(a, a->data, nd, shape, strides, a->dtype);
}

PyObject *qnp_broadcast_to(PyObject *obj, const qintp *shape, int nd) {
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    if (a->nd > nd) {
        qnp_err_broadcast(shape, nd, a->shape, a->nd);
        Py_DECREF(a);
        return NULL;
    }
    qintp strides[QNP_MAXDIMS];
    int offset = nd - a->nd;
    for (int i = 0; i < nd; i++) {
        if (i < offset) { strides[i] = 0; continue; }
        qintp dim = a->shape[i - offset];
        if (dim == shape[i]) strides[i] = a->strides[i - offset];
        else if (dim == 1) strides[i] = 0;
        else {
            qnp_err_broadcast(shape, nd, a->shape, a->nd);
            Py_DECREF(a);
            return NULL;
        }
    }
    QArray *view = qnp_new_view(a, a->data, nd, shape, strides, a->dtype);
    Py_DECREF(a);
    if (view != NULL) view->flags &= ~QNP_WRITEABLE;
    return (PyObject *)view;
}

static PyObject *py_broadcast_to(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj, *shape_obj;
    if (!PyArg_ParseTuple(args, "OO:broadcast_to", &obj, &shape_obj)) return NULL;
    qintp shape[QNP_MAXDIMS];
    int nd;
    if (qnp_shape_from_object(shape_obj, shape, &nd) < 0) return NULL;
    return qnp_broadcast_to(obj, shape, nd);
}

static PyObject *py_reshape(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj, *shape_obj;
    if (!PyArg_ParseTuple(args, "OO:reshape", &obj, &shape_obj)) return NULL;
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    PyObject *r = qnp_reshape(a, shape_obj);
    Py_DECREF(a);
    return r;
}

static PyObject *py_transpose(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *obj, *axes = NULL;
    if (!PyArg_ParseTuple(args, "O|O:transpose", &obj, &axes)) return NULL;
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    PyObject *r = qnp_transpose(a, axes);
    Py_DECREF(a);
    return r;
}

static PyObject *py_concatenate(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *seq;
    PyObject *axis_o = NULL;
    static char *kwlist[] = {"arrays", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:concatenate", kwlist, &seq, &axis_o))
        return NULL;
    PyObject *fast = PySequence_Fast(seq, "concatenate expects a sequence of arrays");
    if (fast == NULL) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n == 0) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "need at least one array to concatenate");
        return NULL;
    }
    QArray **parts = (QArray **)PyMem_Calloc((size_t)n, sizeof(QArray *));
    if (parts == NULL) { Py_DECREF(fast); return PyErr_NoMemory(); }
    int dtype = QNP_BOOL;
    int nd = -1;
    int rc = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        parts[i] = qnp_from_any(PySequence_Fast_GET_ITEM(fast, i), -1, 0);
        if (parts[i] == NULL) { rc = -1; break; }
        dtype = qnp_promote(dtype, parts[i]->dtype);
        if (nd < 0) nd = parts[i]->nd;
        else if (parts[i]->nd != nd) {
            PyErr_SetString(PyExc_ValueError,
                            "all the input arrays must have the same number of dimensions");
            rc = -1;
            break;
        }
    }
    Py_DECREF(fast);
    PyObject *result = NULL;
    if (rc == 0) {
        if (axis_o == Py_None) {
            /* Flatten everything, then join along the only axis there is. */
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject *flat = qnp_ravel(parts[i]);
                Py_DECREF(parts[i]);
                parts[i] = (QArray *)flat;
                if (flat == NULL) { rc = -1; break; }
            }
            nd = 1;
        }
        int axis = 0;
        if (rc == 0 && axis_o != NULL && axis_o != Py_None &&
            qnp_parse_axis(axis_o, nd, &axis) < 0) rc = -1;
        if (rc == 0) {
            qintp shape[QNP_MAXDIMS];
            for (int d = 0; d < nd; d++) shape[d] = parts[0]->shape[d];
            shape[axis] = 0;
            for (Py_ssize_t i = 0; i < n; i++) {
                for (int d = 0; d < nd; d++) {
                    if (d == axis) continue;
                    if (parts[i]->shape[d] != shape[d]) {
                        PyErr_Format(PyExc_ValueError,
                                     "all the input array dimensions except for the "
                                     "concatenation axis must match exactly, but along "
                                     "dimension %d, the array at index 0 has size %zd and "
                                     "the array at index %zd has size %zd",
                                     d, shape[d], i, parts[i]->shape[d]);
                        rc = -1;
                        break;
                    }
                }
                if (rc < 0) break;
                shape[axis] += parts[i]->shape[axis];
            }
            if (rc == 0) {
                QArray *out = qnp_new(nd, shape, dtype);
                if (out == NULL) rc = -1;
                else {
                    qintp offset = 0;
                    for (Py_ssize_t i = 0; i < n && rc == 0; i++) {
                        qintp sub_shape[QNP_MAXDIMS];
                        for (int d = 0; d < nd; d++) sub_shape[d] = parts[i]->shape[d];
                        QArray *dest = qnp_new_view(out, out->data + offset * out->strides[axis],
                                                    nd, sub_shape, out->strides, out->dtype);
                        if (dest == NULL) rc = -1;
                        else {
                            rc = qnp_copy_into(dest, parts[i]);
                            Py_DECREF(dest);
                        }
                        offset += parts[i]->shape[axis];
                    }
                    if (rc == 0) result = (PyObject *)out;
                    else Py_DECREF(out);
                }
            }
        }
    }
    for (Py_ssize_t i = 0; i < n; i++) Py_XDECREF(parts[i]);
    PyMem_Free(parts);
    return result;
}

static PyObject *py_roll(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *obj, *shift_o, *axis_o = NULL;
    static char *kwlist[] = {"a", "shift", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:roll", kwlist,
                                     &obj, &shift_o, &axis_o)) return NULL;
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    QArray *src = a;
    PyObject *flat = NULL;
    int axis = 0;
    qintp shifts[QNP_MAXDIMS] = {0};
    int naxes = 0;
    if (axis_o == NULL || axis_o == Py_None) {
        flat = qnp_ravel(a);
        if (flat == NULL) { Py_DECREF(a); return NULL; }
        src = (QArray *)flat;
        axis = 0;
        Py_ssize_t s = PyNumber_AsSsize_t(shift_o, PyExc_OverflowError);
        if (s == -1 && PyErr_Occurred()) { Py_DECREF(a); Py_DECREF(flat); return NULL; }
        shifts[0] = s;
        naxes = 1;
    } else if (PyTuple_Check(axis_o)) {
        PyErr_SetString(PyExc_NotImplementedError, "roll over several axes at once");
        Py_DECREF(a);
        return NULL;
    } else {
        if (qnp_parse_axis(axis_o, a->nd, &axis) < 0) { Py_DECREF(a); return NULL; }
        Py_ssize_t s = PyNumber_AsSsize_t(shift_o, PyExc_OverflowError);
        if (s == -1 && PyErr_Occurred()) { Py_DECREF(a); return NULL; }
        shifts[0] = s;
        naxes = 1;
    }
    (void)naxes;
    QArray *out = qnp_new(src->nd, src->shape, src->dtype);
    if (out == NULL) { Py_DECREF(a); Py_XDECREF(flat); return NULL; }
    qintp len = src->nd ? src->shape[axis] : 1;
    int rc = 0;
    if (len == 0) {
        rc = qnp_copy_into(out, src);
    } else {
        qintp shift = shifts[0] % len;
        if (shift < 0) shift += len;
        /* Two contiguous block copies along the rolled axis. */
        qintp pieces[2][3] = {{0, len - shift, shift}, {len - shift, shift, 0}};
        for (int k = 0; k < 2 && rc == 0; k++) {
            qintp start = pieces[k][0], count = pieces[k][1], dest = pieces[k][2];
            if (count <= 0) continue;
            qintp shape[QNP_MAXDIMS];
            for (int d = 0; d < src->nd; d++) shape[d] = src->shape[d];
            shape[axis] = count;
            QArray *sv = qnp_new_view(src, src->data + start * src->strides[axis],
                                      src->nd, shape, src->strides, src->dtype);
            QArray *dv = qnp_new_view(out, out->data + dest * out->strides[axis],
                                      out->nd, shape, out->strides, out->dtype);
            if (sv == NULL || dv == NULL) rc = -1;
            else rc = qnp_copy_into(dv, sv);
            Py_XDECREF(sv);
            Py_XDECREF(dv);
        }
    }
    if (rc == 0 && flat != NULL && a->nd != 1) {
        /* Rolling without an axis flattens, but the result keeps the shape. */
        PyObject *shape_tuple = qnp_shape_tuple(a);
        if (shape_tuple == NULL) rc = -1;
        else {
            PyObject *reshaped = qnp_reshape(out, shape_tuple);
            Py_DECREF(shape_tuple);
            Py_DECREF(out);
            out = (QArray *)reshaped;
            if (out == NULL) rc = -1;
        }
    }
    Py_DECREF(a);
    Py_XDECREF(flat);
    if (rc < 0) { Py_XDECREF(out); return NULL; }
    return (PyObject *)out;
}

static PyObject *py_repeat(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *obj, *reps_o, *axis_o = NULL;
    static char *kwlist[] = {"a", "repeats", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:repeat", kwlist,
                                     &obj, &reps_o, &axis_o)) return NULL;
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    QArray *src = a;
    PyObject *flat = NULL;
    int axis = 0;
    if (axis_o == NULL || axis_o == Py_None) {
        flat = qnp_ravel(a);
        if (flat == NULL) { Py_DECREF(a); return NULL; }
        src = (QArray *)flat;
    } else if (qnp_parse_axis(axis_o, a->nd, &axis) < 0) {
        Py_DECREF(a);
        return NULL;
    }
    qintp dimlen = src->nd ? src->shape[axis] : 1;
    QArray *reps = qnp_from_any(reps_o, QNP_INT64, 1);
    if (reps == NULL) { Py_DECREF(a); Py_XDECREF(flat); return NULL; }
    QArray *creps = qnp_ascontiguous(reps);
    Py_DECREF(reps);
    if (creps == NULL) { Py_DECREF(a); Py_XDECREF(flat); return NULL; }
    qintp nreps = qnp_size(creps);
    if (nreps != 1 && nreps != dimlen) {
        PyErr_SetString(PyExc_ValueError,
                        "operands could not be broadcast together with the repeat counts");
        Py_DECREF(a); Py_XDECREF(flat); Py_DECREF(creps);
        return NULL;
    }
    const int64_t *rp = (const int64_t *)creps->data;
    qintp total = 0;
    for (qintp i = 0; i < dimlen; i++) total += rp[nreps == 1 ? 0 : i];
    qintp shape[QNP_MAXDIMS];
    for (int d = 0; d < src->nd; d++) shape[d] = src->shape[d];
    shape[axis] = total;
    QArray *out = qnp_new(src->nd, shape, src->dtype);
    int rc = 0;
    if (out == NULL) rc = -1;
    qintp w = 0;
    for (qintp i = 0; i < dimlen && rc == 0; i++) {
        qintp count = rp[nreps == 1 ? 0 : i];
        qintp sshape[QNP_MAXDIMS];
        for (int d = 0; d < src->nd; d++) sshape[d] = src->shape[d];
        sshape[axis] = 1;
        QArray *sv = qnp_new_view(src, src->data + i * src->strides[axis],
                                  src->nd, sshape, src->strides, src->dtype);
        if (sv == NULL) { rc = -1; break; }
        for (qintp k = 0; k < count && rc == 0; k++, w++) {
            QArray *dv = qnp_new_view(out, out->data + w * out->strides[axis],
                                      out->nd, sshape, out->strides, out->dtype);
            if (dv == NULL) rc = -1;
            else { rc = qnp_copy_into(dv, sv); Py_DECREF(dv); }
        }
        Py_DECREF(sv);
    }
    Py_DECREF(a);
    Py_XDECREF(flat);
    Py_DECREF(creps);
    if (rc < 0) { Py_XDECREF(out); return NULL; }
    return (PyObject *)out;
}

static PyObject *py_diff(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *obj;
    int n = 1;
    PyObject *axis_o = NULL;
    static char *kwlist[] = {"a", "n", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO:diff", kwlist, &obj, &n, &axis_o))
        return NULL;
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    int axis = a->nd - 1;
    if (axis_o != NULL && axis_o != Py_None && qnp_parse_axis(axis_o, a->nd, &axis) < 0) {
        Py_DECREF(a);
        return NULL;
    }
    if (a->nd == 0) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "diff requires input that is at least one dimensional");
        return NULL;
    }
    PyObject *cur = (PyObject *)a;
    for (int pass = 0; pass < n; pass++) {
        QArray *c = (QArray *)cur;
        qintp len = c->shape[axis];
        qintp shape[QNP_MAXDIMS];
        for (int d = 0; d < c->nd; d++) shape[d] = c->shape[d];
        shape[axis] = len > 0 ? len - 1 : 0;
        QArray *hi = qnp_new_view(c, c->data + c->strides[axis], c->nd, shape, c->strides, c->dtype);
        QArray *lo = qnp_new_view(c, c->data, c->nd, shape, c->strides, c->dtype);
        if (hi == NULL || lo == NULL) {
            Py_XDECREF(hi); Py_XDECREF(lo); Py_DECREF(cur);
            return NULL;
        }
        PyObject *next = qnp_binary_op(c->dtype == QNP_BOOL ? QOP_XOR : QOP_SUB,
                                       (PyObject *)hi, (PyObject *)lo, NULL, NULL);
        Py_DECREF(hi); Py_DECREF(lo); Py_DECREF(cur);
        if (next == NULL) return NULL;
        cur = next;
    }
    return cur;
}

static PyObject *py_outer(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *bo, *out_o = NULL;
    static char *kwlist[] = {"a", "b", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:outer", kwlist, &ao, &bo, &out_o))
        return NULL;
    QArray *a = qnp_from_any(ao, -1, 0);
    QArray *b = qnp_from_any(bo, -1, 0);
    if (a == NULL || b == NULL) { Py_XDECREF(a); Py_XDECREF(b); return NULL; }
    int dt = qnp_promote(a->dtype, b->dtype);
    QArray *fa = qnp_astype(a, dt, 0), *fb = qnp_astype(b, dt, 0);
    Py_DECREF(a); Py_DECREF(b);
    if (fa == NULL || fb == NULL) { Py_XDECREF(fa); Py_XDECREF(fb); return NULL; }
    QArray *ca = qnp_ascontiguous(fa), *cb = qnp_ascontiguous(fb);
    Py_DECREF(fa); Py_DECREF(fb);
    if (ca == NULL || cb == NULL) { Py_XDECREF(ca); Py_XDECREF(cb); return NULL; }
    qintp m = qnp_size(ca), n = qnp_size(cb);
    qintp shape[2] = {m, n};
    QArray *out;
    if (out_o != NULL && out_o != Py_None && QArray_Check(out_o) &&
        ((QArray *)out_o)->dtype == dt && ((QArray *)out_o)->nd == 2 &&
        ((QArray *)out_o)->shape[0] == m && ((QArray *)out_o)->shape[1] == n &&
        (((QArray *)out_o)->flags & QNP_C_CONTIGUOUS)) {
        out = (QArray *)out_o;
        Py_INCREF(out);
    } else {
        out = qnp_new(2, shape, dt);
    }
    if (out == NULL) { Py_DECREF(ca); Py_DECREF(cb); return NULL; }
    if (dt == QNP_FLOAT64) {
        const double *x = (const double *)ca->data, *y = (const double *)cb->data;
        double *o = (double *)out->data;
        for (qintp i = 0; i < m; i++) {
            double xi = x[i];
            double *row = o + i * n;
            for (qintp j = 0; j < n; j++) row[j] = xi * y[j];
        }
    } else if (dt == QNP_COMPLEX128) {
        const qcomplex *x = (const qcomplex *)ca->data, *y = (const qcomplex *)cb->data;
        qcomplex *o = (qcomplex *)out->data;
        for (qintp i = 0; i < m; i++)
            for (qintp j = 0; j < n; j++) o[i * n + j] = qc_mul(x[i], y[j]);
    } else if (dt == QNP_INT64) {
        const int64_t *x = (const int64_t *)ca->data, *y = (const int64_t *)cb->data;
        int64_t *o = (int64_t *)out->data;
        for (qintp i = 0; i < m; i++)
            for (qintp j = 0; j < n; j++) o[i * n + j] = x[i] * y[j];
    } else {
        const qbool *x = (const qbool *)ca->data, *y = (const qbool *)cb->data;
        qbool *o = (qbool *)out->data;
        for (qintp i = 0; i < m; i++)
            for (qintp j = 0; j < n; j++) o[i * n + j] = x[i] & y[j];
    }
    Py_DECREF(ca);
    Py_DECREF(cb);
    if (out_o != NULL && out_o != Py_None && (PyObject *)out != out_o) {
        int rc = qnp_copy_into((QArray *)out_o, out);
        Py_DECREF(out);
        if (rc < 0) return NULL;
        Py_INCREF(out_o);
        return out_o;
    }
    return (PyObject *)out;
}

static PyObject *py_where(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cond, *x = NULL, *y = NULL;
    if (!PyArg_ParseTuple(args, "O|OO:where", &cond, &x, &y)) return NULL;
    if (x == NULL) {
        PyObject *nz = PyObject_CallMethod(PyImport_AddModule("quadrivium._qnp"),
                                           "nonzero", "O", cond);
        return nz;
    }
    if (y == NULL) {
        PyErr_SetString(PyExc_ValueError, "either both or neither of x and y should be given");
        return NULL;
    }
    return qnp_where3(cond, x, y);
}

static PyObject *py_copy(PyObject *self, PyObject *arg) {
    (void)self;
    QArray *a = qnp_from_any(arg, -1, 0);
    if (a == NULL) return NULL;
    QArray *c = qnp_astype(a, a->dtype, 1);
    Py_DECREF(a);
    return (PyObject *)c;
}

static PyObject *py_astype(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *obj, *dtype_obj;
    int copy = 1;
    static char *kwlist[] = {"a", "dtype", "copy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|p:astype", kwlist,
                                     &obj, &dtype_obj, &copy)) return NULL;
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    int ok = 1;
    int dt = qnp_dtype_from_object(dtype_obj, &ok);
    if (!ok) { Py_DECREF(a); return NULL; }
    QArray *r = qnp_astype(a, dt, copy);
    Py_DECREF(a);
    return (PyObject *)r;
}

/* np.interp: piecewise linear interpolation on sorted sample points. */
static PyObject *py_interp(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *xo, *xpo, *fpo;
    if (!PyArg_ParseTuple(args, "OOO:interp", &xo, &xpo, &fpo)) return NULL;
    QArray *x = qnp_from_any(xo, QNP_FLOAT64, 1);
    QArray *xp = qnp_from_any(xpo, QNP_FLOAT64, 1);
    QArray *fp = qnp_from_any(fpo, QNP_FLOAT64, 1);
    if (x == NULL || xp == NULL || fp == NULL) {
        Py_XDECREF(x); Py_XDECREF(xp); Py_XDECREF(fp);
        return NULL;
    }
    QArray *cx = qnp_ascontiguous(x), *cxp = qnp_ascontiguous(xp), *cfp = qnp_ascontiguous(fp);
    Py_DECREF(x); Py_DECREF(xp); Py_DECREF(fp);
    if (cx == NULL || cxp == NULL || cfp == NULL) {
        Py_XDECREF(cx); Py_XDECREF(cxp); Py_XDECREF(cfp);
        return NULL;
    }
    qintp n = qnp_size(cxp);
    if (n != qnp_size(cfp) || n == 0) {
        Py_DECREF(cx); Py_DECREF(cxp); Py_DECREF(cfp);
        PyErr_SetString(PyExc_ValueError, "interp: xp and fp must have the same nonzero length");
        return NULL;
    }
    QArray *out = qnp_new(cx->nd, cx->shape, QNP_FLOAT64);
    if (out == NULL) {
        Py_DECREF(cx); Py_DECREF(cxp); Py_DECREF(cfp);
        return NULL;
    }
    const double *xv = (const double *)cx->data;
    const double *xpv = (const double *)cxp->data;
    const double *fpv = (const double *)cfp->data;
    double *o = (double *)out->data;
    qintp m = qnp_size(cx);
    for (qintp i = 0; i < m; i++) {
        double v = xv[i];
        if (v <= xpv[0]) { o[i] = fpv[0]; continue; }
        if (v >= xpv[n - 1]) { o[i] = fpv[n - 1]; continue; }
        qintp lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            qintp mid = (lo + hi) / 2;
            if (xpv[mid] <= v) lo = mid; else hi = mid;
        }
        double t = (v - xpv[lo]) / (xpv[hi] - xpv[lo]);
        o[i] = fpv[lo] + t * (fpv[hi] - fpv[lo]);
    }
    Py_DECREF(cx); Py_DECREF(cxp); Py_DECREF(cfp);
    return qnp_wrap_scalar_or_array(out);
}

/* Byte range an array can touch, for the memory-overlap query below. */
static void memory_extent(QArray *a, char **low, char **high) {
    char *lo = a->data, *hi = a->data + QNP_ITEMSIZE(a->dtype);
    for (int i = 0; i < a->nd; i++) {
        if (a->shape[i] == 0) { hi = lo; break; }
        qintp span = a->strides[i] * (a->shape[i] - 1);
        if (span > 0) hi += span;
        else lo += span;
    }
    *low = lo;
    *high = hi;
}

static PyObject *py_shares_memory(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *ao, *bo;
    if (!PyArg_ParseTuple(args, "OO:shares_memory", &ao, &bo)) return NULL;
    if (!QArray_Check(ao) || !QArray_Check(bo)) Py_RETURN_FALSE;
    char *alo, *ahi, *blo, *bhi;
    memory_extent((QArray *)ao, &alo, &ahi);
    memory_extent((QArray *)bo, &blo, &bhi);
    /* Bounding-box overlap: exact for the contiguous and sliced views the
     * library builds, and conservative in the same direction as NumPy's
     * `may_share_memory` otherwise. */
    return PyBool_FromLong(alo < bhi && blo < ahi);
}

PyMethodDef qnp_shape_methods[] = {
    {"shares_memory", py_shares_memory, METH_VARARGS,
     "True when two arrays can address the same bytes."},
    {"zeros", (PyCFunction)py_zeros, METH_VARARGS | METH_KEYWORDS, "Array of zeros."},
    {"ones", (PyCFunction)py_ones, METH_VARARGS | METH_KEYWORDS, "Array of ones."},
    {"empty", (PyCFunction)py_empty, METH_VARARGS | METH_KEYWORDS, "Uninitialised array."},
    {"full", (PyCFunction)py_full, METH_VARARGS | METH_KEYWORDS, "Array filled with a value."},
    {"zeros_like", (PyCFunction)py_zeros_like, METH_VARARGS | METH_KEYWORDS, "Zeros shaped like the input."},
    {"ones_like", (PyCFunction)py_ones_like, METH_VARARGS | METH_KEYWORDS, "Ones shaped like the input."},
    {"empty_like", (PyCFunction)py_empty_like, METH_VARARGS | METH_KEYWORDS, "Uninitialised array shaped like the input."},
    {"full_like", (PyCFunction)py_full_like, METH_VARARGS | METH_KEYWORDS, "Filled array shaped like the input."},
    {"array", (PyCFunction)py_array, METH_VARARGS | METH_KEYWORDS, "Build an array from data."},
    {"asarray", (PyCFunction)py_asarray, METH_VARARGS | METH_KEYWORDS, "Convert to an array without copying when possible."},
    {"ascontiguousarray", (PyCFunction)py_ascontiguousarray, METH_VARARGS | METH_KEYWORDS, "Contiguous copy."},
    {"arange", (PyCFunction)py_arange, METH_VARARGS | METH_KEYWORDS, "Evenly spaced values in a half-open interval."},
    {"linspace", (PyCFunction)py_linspace, METH_VARARGS | METH_KEYWORDS, "Evenly spaced samples over an interval."},
    {"eye", (PyCFunction)py_eye, METH_VARARGS | METH_KEYWORDS, "Matrix with ones on a diagonal."},
    {"reshape", py_reshape, METH_VARARGS, "Array with a new shape."},
    {"transpose", py_transpose, METH_VARARGS, "Permuted view of the array."},
    {"broadcast_to", py_broadcast_to, METH_VARARGS, "Read-only broadcast view."},
    {"concatenate", (PyCFunction)py_concatenate, METH_VARARGS | METH_KEYWORDS, "Join arrays along an axis."},
    {"roll", (PyCFunction)py_roll, METH_VARARGS | METH_KEYWORDS, "Cyclically shift elements."},
    {"repeat", (PyCFunction)py_repeat, METH_VARARGS | METH_KEYWORDS, "Repeat elements along an axis."},
    {"diff", (PyCFunction)py_diff, METH_VARARGS | METH_KEYWORDS, "Discrete differences along an axis."},
    {"outer", (PyCFunction)py_outer, METH_VARARGS | METH_KEYWORDS, "Outer product of two vectors."},
    {"where", py_where, METH_VARARGS, "Choose between two arrays element-wise."},
    {"copy", py_copy, METH_O, "Copy of the array."},
    {"astype", (PyCFunction)py_astype, METH_VARARGS | METH_KEYWORDS, "Cast to another dtype."},
    {"interp", py_interp, METH_VARARGS, "Piecewise linear interpolation."},
    {NULL}
};
