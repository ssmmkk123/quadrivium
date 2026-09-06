/* The array object itself: layout, construction from Python data, casting,
 * the buffer protocol, attributes and the operator hooks. */
#include "qnp.h"
#include "structmember.h"

PyObject *QNP_LinAlgError = NULL;
static PyObject *qnp_printer = NULL;   /* set by the Python layer */

/* ---------------------------------------------------------------- dtype */

typedef struct {
    PyObject_HEAD
    int num;
} QDtype;

static const char *dtype_names[QNP_NTYPES] = {"bool", "int64", "float64", "complex128"};
static const char dtype_kinds[QNP_NTYPES] = {'b', 'i', 'f', 'c'};
static const char *dtype_formats[QNP_NTYPES] = {"?", "q", "d", "Zd"};
static QDtype *dtype_singletons[QNP_NTYPES];

static PyObject *dtype_repr(QDtype *self) {
    return PyUnicode_FromFormat("dtype('%s')", dtype_names[self->num]);
}

static PyObject *dtype_str(QDtype *self) {
    return PyUnicode_FromString(dtype_names[self->num]);
}

static PyObject *dtype_get_name(QDtype *self, void *c) {
    (void)c; return PyUnicode_FromString(dtype_names[self->num]);
}
static PyObject *dtype_get_kind(QDtype *self, void *c) {
    (void)c; return PyUnicode_FromStringAndSize(&dtype_kinds[self->num], 1);
}
static PyObject *dtype_get_itemsize(QDtype *self, void *c) {
    (void)c; return PyLong_FromLong(QNP_ITEMSIZE(self->num));
}
static PyObject *dtype_get_char(QDtype *self, void *c) {
    (void)c;
    static const char chars[QNP_NTYPES] = {'?', 'l', 'd', 'D'};
    return PyUnicode_FromStringAndSize(&chars[self->num], 1);
}
static PyObject *dtype_get_type(QDtype *self, void *c) {
    (void)c;
    PyObject *t;
    switch (self->num) {
        case QNP_BOOL: t = (PyObject *)&PyBool_Type; break;
        case QNP_INT64: t = (PyObject *)&PyLong_Type; break;
        case QNP_FLOAT64: t = (PyObject *)&PyFloat_Type; break;
        default: t = (PyObject *)&PyComplex_Type; break;
    }
    Py_INCREF(t);
    return t;
}

static PyGetSetDef dtype_getset[] = {
    {"name", (getter)dtype_get_name, NULL, NULL, NULL},
    {"kind", (getter)dtype_get_kind, NULL, NULL, NULL},
    {"itemsize", (getter)dtype_get_itemsize, NULL, NULL, NULL},
    {"char", (getter)dtype_get_char, NULL, NULL, NULL},
    {"type", (getter)dtype_get_type, NULL, NULL, NULL},
    {NULL}
};

static PyObject *dtype_richcompare(PyObject *self, PyObject *other, int op) {
    if (op != Py_EQ && op != Py_NE) Py_RETURN_NOTIMPLEMENTED;
    int ok = 1;
    int a = ((QDtype *)self)->num;
    int b = qnp_dtype_from_object(other, &ok);
    if (!ok) {
        PyErr_Clear();
        if (op == Py_EQ) Py_RETURN_FALSE;
        Py_RETURN_TRUE;
    }
    int eq = (a == b);
    if (op == Py_NE) eq = !eq;
    return PyBool_FromLong(eq);
}

static Py_hash_t dtype_hash(QDtype *self) { return (Py_hash_t)(self->num + 1); }

static PyObject *dtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    (void)type;
    PyObject *obj;
    static char *kwlist[] = {"obj", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:dtype", kwlist, &obj)) return NULL;
    int ok = 1;
    int num = qnp_dtype_from_object(obj, &ok);
    if (!ok) return NULL;
    return qnp_dtype_object(num);
}

PyTypeObject QDtype_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "quadrivium._qnp.dtype",
    .tp_basicsize = sizeof(QDtype),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_repr = (reprfunc)dtype_repr,
    .tp_str = (reprfunc)dtype_str,
    .tp_hash = (hashfunc)dtype_hash,
    .tp_richcompare = dtype_richcompare,
    .tp_getset = dtype_getset,
    .tp_new = dtype_new,
};

PyObject *qnp_dtype_object(int dtype) {
    if (dtype < 0 || dtype >= QNP_NTYPES) {
        PyErr_SetString(PyExc_TypeError, "unknown dtype");
        return NULL;
    }
    Py_INCREF(dtype_singletons[dtype]);
    return (PyObject *)dtype_singletons[dtype];
}

/* Accepts what the library actually writes: `float`, `complex`, `bool`, `int`,
 * a dtype instance, a NumPy-style string or character code, or None. */
int qnp_dtype_from_object(PyObject *obj, int *ok) {
    *ok = 1;
    if (obj == NULL || obj == Py_None) { *ok = 0; return -1; }
    if (Py_TYPE(obj) == &QDtype_Type) return ((QDtype *)obj)->num;
    if (obj == (PyObject *)&PyFloat_Type) return QNP_FLOAT64;
    if (obj == (PyObject *)&PyComplex_Type) return QNP_COMPLEX128;
    if (obj == (PyObject *)&PyBool_Type) return QNP_BOOL;
    if (obj == (PyObject *)&PyLong_Type) return QNP_INT64;
    if (PyUnicode_Check(obj)) {
        const char *s = PyUnicode_AsUTF8(obj);
        if (!s) { *ok = 0; return -1; }
        if (!strcmp(s, "float64") || !strcmp(s, "float") || !strcmp(s, "d") ||
            !strcmp(s, "f8") || !strcmp(s, "double")) return QNP_FLOAT64;
        if (!strcmp(s, "complex128") || !strcmp(s, "complex") || !strcmp(s, "D") ||
            !strcmp(s, "c16")) return QNP_COMPLEX128;
        if (!strcmp(s, "bool") || !strcmp(s, "?") || !strcmp(s, "b1")) return QNP_BOOL;
        if (!strcmp(s, "int64") || !strcmp(s, "int") || !strcmp(s, "i8") ||
            !strcmp(s, "l") || !strcmp(s, "q") || !strcmp(s, "intp") ||
            !strcmp(s, "long")) return QNP_INT64;
        PyErr_Format(PyExc_TypeError, "data type '%s' not understood", s);
        *ok = 0;
        return -1;
    }
    /* A NumPy dtype or scalar type passed in from user code. */
    PyObject *name = PyObject_GetAttrString(obj, "name");
    if (name != NULL) {
        int num = qnp_dtype_from_object(name, ok);
        Py_DECREF(name);
        if (*ok) return num;
        PyErr_Clear();
        *ok = 1;
    } else {
        PyErr_Clear();
    }
    PyErr_Format(PyExc_TypeError, "data type %R not understood", obj);
    *ok = 0;
    return -1;
}

int qnp_promote(int a, int b) { return a > b ? a : b; }

/* ------------------------------------------------------- array plumbing */

qintp qnp_size(const QArray *a) {
    qintp n = 1;
    for (int i = 0; i < a->nd; i++) n *= a->shape[i];
    return n;
}

int qnp_is_c_contiguous(const QArray *a) {
    qintp expected = QNP_ITEMSIZE(a->dtype);
    for (int i = a->nd - 1; i >= 0; i--) {
        if (a->shape[i] == 1) continue;
        if (a->strides[i] != expected) return 0;
        expected *= a->shape[i];
    }
    return 1;
}

static int is_f_contiguous(const QArray *a) {
    qintp expected = QNP_ITEMSIZE(a->dtype);
    for (int i = 0; i < a->nd; i++) {
        if (a->shape[i] == 1) continue;
        if (a->strides[i] != expected) return 0;
        expected *= a->shape[i];
    }
    return 1;
}

void qnp_update_flags(QArray *a) {
    a->flags &= ~(QNP_C_CONTIGUOUS | QNP_F_CONTIGUOUS);
    if (qnp_is_c_contiguous(a)) a->flags |= QNP_C_CONTIGUOUS;
    if (is_f_contiguous(a)) a->flags |= QNP_F_CONTIGUOUS;
}

static QArray *array_alloc_header(int nd) {
    if (nd > QNP_MAXDIMS) {
        PyErr_Format(PyExc_ValueError, "maximum supported dimension is %d", QNP_MAXDIMS);
        return NULL;
    }
    QArray *self = PyObject_GC_New(QArray, &QArray_Type);
    if (self == NULL) return NULL;
    self->data = NULL;
    self->base = NULL;
    self->nd = nd;
    self->dtype = QNP_FLOAT64;
    self->flags = QNP_WRITEABLE;
    self->exports = 0;
    self->weakreflist = NULL;
    self->shape = NULL;
    self->strides = NULL;
    if (nd > 0) {
        self->shape = (qintp *)PyMem_Malloc(2 * (size_t)nd * sizeof(qintp));
        if (self->shape == NULL) { PyObject_GC_Del(self); return (QArray *)PyErr_NoMemory(); }
        self->strides = self->shape + nd;
    }
    return self;
}

QArray *qnp_new(int nd, const qintp *shape, int dtype) {
    QArray *self = array_alloc_header(nd);
    if (self == NULL) return NULL;
    self->dtype = dtype;
    qintp total = 1;
    int itemsize = QNP_ITEMSIZE(dtype);
    for (int i = 0; i < nd; i++) {
        if (shape[i] < 0) {
            PyErr_SetString(PyExc_ValueError, "negative dimensions are not allowed");
            PyMem_Free(self->shape);
            PyObject_GC_Del(self);
            return NULL;
        }
        self->shape[i] = shape[i];
        if (shape[i] != 0 && total > PY_SSIZE_T_MAX / shape[i]) {
            PyErr_SetString(PyExc_ValueError, "array is too big");
            PyMem_Free(self->shape);
            PyObject_GC_Del(self);
            return NULL;
        }
        total *= shape[i];
    }
    qintp stride = itemsize;
    for (int i = nd - 1; i >= 0; i--) {
        self->strides[i] = stride;
        stride *= self->shape[i];
    }
    /* One spare item keeps zero-sized arrays with a valid, unique pointer. */
    size_t nbytes = (size_t)(total ? total : 1) * (size_t)itemsize;
    self->data = (char *)PyMem_Malloc(nbytes);
    if (self->data == NULL) {
        PyMem_Free(self->shape);
        PyObject_GC_Del(self);
        return (QArray *)PyErr_NoMemory();
    }
    self->flags |= QNP_OWNDATA;
    qnp_update_flags(self);
    PyObject_GC_Track(self);
    return self;
}

QArray *qnp_new_like(QArray *proto, int dtype) {
    return qnp_new(proto->nd, proto->shape, dtype < 0 ? proto->dtype : dtype);
}

QArray *qnp_new_view_as(PyTypeObject *type, QArray *base, char *data, int nd,
                        const qintp *shape, const qintp *strides, int dtype) {
    QArray *self;
    if (type == &QArray_Type) {
        self = array_alloc_header(nd);
    } else {
        if (nd > QNP_MAXDIMS) {
            PyErr_Format(PyExc_ValueError, "maximum supported dimension is %d", QNP_MAXDIMS);
            return NULL;
        }
        self = (QArray *)type->tp_alloc(type, 0);
        if (self != NULL) {
            self->data = NULL;
            self->base = NULL;
            self->nd = nd;
            self->dtype = QNP_FLOAT64;
            self->flags = QNP_WRITEABLE;
            self->exports = 0;
            self->weakreflist = NULL;
            self->shape = NULL;
            self->strides = NULL;
            if (nd > 0) {
                self->shape = (qintp *)PyMem_Malloc(2 * (size_t)nd * sizeof(qintp));
                if (self->shape == NULL) {
                    Py_DECREF(self);
                    return (QArray *)PyErr_NoMemory();
                }
                self->strides = self->shape + nd;
            }
        }
    }
    if (self == NULL) return NULL;
    self->dtype = dtype;
    for (int i = 0; i < nd; i++) {
        self->shape[i] = shape[i];
        self->strides[i] = strides[i];
    }
    self->data = data;
    PyObject *owner = (PyObject *)base;
    while (QArray_Check(owner) && ((QArray *)owner)->base != NULL)
        owner = ((QArray *)owner)->base;
    Py_INCREF(owner);
    self->base = owner;
    if (!(base->flags & QNP_WRITEABLE)) self->flags &= ~QNP_WRITEABLE;
    qnp_update_flags(self);
    if (!PyObject_GC_IsTracked((PyObject *)self)) PyObject_GC_Track(self);
    return self;
}

QArray *qnp_new_view(QArray *base, char *data, int nd, const qintp *shape,
                     const qintp *strides, int dtype) {
    QArray *self = array_alloc_header(nd);
    if (self == NULL) return NULL;
    self->dtype = dtype;
    for (int i = 0; i < nd; i++) {
        self->shape[i] = shape[i];
        self->strides[i] = strides[i];
    }
    self->data = data;
    /* Views chain to the ultimate owner so the base list stays one deep. */
    PyObject *owner = (PyObject *)base;
    while (QArray_Check(owner) && ((QArray *)owner)->base != NULL)
        owner = ((QArray *)owner)->base;
    Py_INCREF(owner);
    self->base = owner;
    if (!(base->flags & QNP_WRITEABLE)) self->flags &= ~QNP_WRITEABLE;
    qnp_update_flags(self);
    PyObject_GC_Track(self);
    return self;
}

static int array_traverse(QArray *self, visitproc visit, void *arg) {
    Py_VISIT(self->base);
    return 0;
}

static int array_clear(QArray *self) {
    Py_CLEAR(self->base);
    return 0;
}

static void array_dealloc(QArray *self) {
    /* Reached directly for base instances and through `subtype_dealloc` for
     * Python subclasses, so the free has to go through the concrete type. */
    PyTypeObject *type = Py_TYPE(self);
    PyObject_GC_UnTrack(self);
    if (self->weakreflist != NULL) PyObject_ClearWeakRefs((PyObject *)self);
    if (self->flags & QNP_OWNDATA) PyMem_Free(self->data);
    Py_XDECREF(self->base);
    PyMem_Free(self->shape);
    type->tp_free((PyObject *)self);
}

/* --------------------------------------------------- scalar conversion */

PyObject *qnp_getitem_ptr(int dtype, const char *ptr) {
    switch (dtype) {
        case QNP_BOOL: return PyBool_FromLong(*(const unsigned char *)ptr);
        case QNP_INT64: return PyLong_FromLongLong(*(const int64_t *)ptr);
        case QNP_FLOAT64: return PyFloat_FromDouble(*(const double *)ptr);
        default: {
            const qcomplex *z = (const qcomplex *)ptr;
            return PyComplex_FromDoubles(z->re, z->im);
        }
    }
}

int qnp_setitem_ptr(int dtype, char *ptr, PyObject *value) {
    switch (dtype) {
        case QNP_BOOL: {
            int truth = PyObject_IsTrue(value);
            if (truth < 0) return -1;
            *(unsigned char *)ptr = (unsigned char)truth;
            return 0;
        }
        case QNP_INT64: {
            long long v;
            if (PyFloat_Check(value)) {
                double d = PyFloat_AS_DOUBLE(value);
                v = (long long)d;
            } else {
                PyObject *idx = PyNumber_Long(value);
                if (idx == NULL) return -1;
                v = PyLong_AsLongLong(idx);
                Py_DECREF(idx);
                if (v == -1 && PyErr_Occurred()) return -1;
            }
            *(int64_t *)ptr = (int64_t)v;
            return 0;
        }
        case QNP_FLOAT64: {
            double d = PyFloat_AsDouble(value);
            if (d == -1.0 && PyErr_Occurred()) return -1;
            *(double *)ptr = d;
            return 0;
        }
        default: {
            Py_complex c = PyComplex_AsCComplex(value);
            if (c.real == -1.0 && PyErr_Occurred()) return -1;
            ((qcomplex *)ptr)->re = c.real;
            ((qcomplex *)ptr)->im = c.imag;
            return 0;
        }
    }
}

PyObject *qnp_from_scalar(int dtype, const void *value) {
    return qnp_getitem_ptr(dtype, (const char *)value);
}

/* Dtype of a Python scalar, and whether it is "weak" in the NEP 50 sense:
 * a plain Python number takes on the dtype of the array it meets. */
int qnp_scalar_dtype(PyObject *obj, int *weak) {
    if (weak) *weak = 1;
    if (PyBool_Check(obj)) return QNP_BOOL;
    if (PyLong_Check(obj)) return QNP_INT64;
    if (PyFloat_Check(obj)) return QNP_FLOAT64;
    if (PyComplex_Check(obj)) return QNP_COMPLEX128;
    if (weak) *weak = 0;
    return -1;
}

/* -------------------------------------------------- casting and copying */

/* Copy `n` elements from a strided source of dtype `sdt` to a strided
 * destination of dtype `ddt`, converting as it goes. */
void qnp_cast_strided(char *dst, qintp dstride, int ddt,
                      const char *src, qintp sstride, int sdt, qintp n) {
    qintp i;
    if (ddt == sdt) {
        int isz = QNP_ITEMSIZE(ddt);
        if (dstride == isz && sstride == isz) {
            memcpy(dst, src, (size_t)n * (size_t)isz);
            return;
        }
        for (i = 0; i < n; i++, dst += dstride, src += sstride)
            memcpy(dst, src, (size_t)isz);
        return;
    }
#define CAST_LOOP(DTYPE, STYPE, EXPR)                                        \
    for (i = 0; i < n; i++, dst += dstride, src += sstride) {                \
        STYPE v = *(const STYPE *)src; (void)v;                              \
        *(DTYPE *)dst = (EXPR);                                              \
    }
    switch (ddt) {
        case QNP_BOOL:
            switch (sdt) {
                case QNP_INT64: CAST_LOOP(unsigned char, int64_t, v != 0) break;
                case QNP_FLOAT64: CAST_LOOP(unsigned char, double, v != 0.0) break;
                default: CAST_LOOP(unsigned char, qcomplex, (v.re != 0.0 || v.im != 0.0)) break;
            }
            break;
        case QNP_INT64:
            switch (sdt) {
                case QNP_BOOL: CAST_LOOP(int64_t, unsigned char, (int64_t)v) break;
                case QNP_FLOAT64: CAST_LOOP(int64_t, double, (int64_t)v) break;
                default: CAST_LOOP(int64_t, qcomplex, (int64_t)v.re) break;
            }
            break;
        case QNP_FLOAT64:
            switch (sdt) {
                case QNP_BOOL: CAST_LOOP(double, unsigned char, (double)v) break;
                case QNP_INT64: CAST_LOOP(double, int64_t, (double)v) break;
                default: CAST_LOOP(double, qcomplex, v.re) break;
            }
            break;
        default:
            switch (sdt) {
                case QNP_BOOL: CAST_LOOP(qcomplex, unsigned char, qc((double)v, 0.0)) break;
                case QNP_INT64: CAST_LOOP(qcomplex, int64_t, qc((double)v, 0.0)) break;
                default: CAST_LOOP(qcomplex, double, qc(v, 0.0)) break;
            }
            break;
    }
#undef CAST_LOOP
}

/* Element-by-element copy from `src` into `dst`, broadcasting `src` up to the
 * destination shape and casting on the way. */
int qnp_copy_into(QArray *dst, QArray *src) {
    QArray *bsrc = NULL;
    if (!(dst->nd == src->nd &&
          !memcmp(dst->shape, src->shape, (size_t)dst->nd * sizeof(qintp)))) {
        PyObject *b = qnp_broadcast_to((PyObject *)src, dst->shape, dst->nd);
        if (b == NULL) {
            PyErr_Clear();
            PyObject *sh1 = qnp_shape_tuple(src), *sh2 = qnp_shape_tuple(dst);
            PyErr_Format(PyExc_ValueError,
                         "could not broadcast input array from shape %R into shape %R",
                         sh1, sh2);
            Py_XDECREF(sh1); Py_XDECREF(sh2);
            return -1;
        }
        bsrc = (QArray *)b;
        src = bsrc;
    }
    QArray *ops[2] = {dst, src};
    QIter it;
    if (qnp_iter_init(&it, 2, ops, dst->shape, dst->nd) < 0) { Py_XDECREF(bsrc); return -1; }
    while (qnp_iter_next(&it)) {
        qnp_cast_strided(it.ptr[0], it.inner_stride[0], dst->dtype,
                         it.ptr[1], it.inner_stride[1], src->dtype, it.inner_len);
    }
    Py_XDECREF(bsrc);
    return 0;
}

QArray *qnp_astype(QArray *a, int dtype, int copy) {
    if (!copy && a->dtype == dtype) { Py_INCREF(a); return a; }
    QArray *out = qnp_new(a->nd, a->shape, dtype);
    if (out == NULL) return NULL;
    if (qnp_copy_into(out, a) < 0) { Py_DECREF(out); return NULL; }
    return out;
}

QArray *qnp_ascontiguous(QArray *a) {
    if (a->flags & QNP_C_CONTIGUOUS) { Py_INCREF(a); return a; }
    return qnp_astype(a, a->dtype, 1);
}

/* ------------------------------------------------ building from Python */

/* Walk a nested sequence to discover its shape.  Returns -1 on a ragged
 * structure, matching NumPy's refusal to build object arrays here. */
static int discover_shape(PyObject *obj, qintp *shape, int *nd, int depth) {
    if (depth > QNP_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "nesting deeper than the supported dimension limit");
        return -1;
    }
    if (QArray_Check(obj)) {
        QArray *a = (QArray *)obj;
        if (depth + a->nd > QNP_MAXDIMS) {
            PyErr_SetString(PyExc_ValueError, "nesting deeper than the supported dimension limit");
            return -1;
        }
        for (int i = 0; i < a->nd; i++) shape[depth + i] = a->shape[i];
        if (depth + a->nd > *nd) *nd = depth + a->nd;
        return 0;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        if (depth > *nd) *nd = depth;
        return 0;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(obj);
    shape[depth] = n;
    if (depth + 1 > *nd) *nd = depth + 1;
    if (n == 0) return 0;
    PyObject **items = PySequence_Fast_ITEMS(obj);
    if (discover_shape(items[0], shape, nd, depth + 1) < 0) return -1;
    return 0;
}

static int discover_dtype(PyObject *obj, int *dtype, int depth) {
    if (QArray_Check(obj)) {
        *dtype = qnp_promote(*dtype, ((QArray *)obj)->dtype);
        return 0;
    }
    if (PyList_Check(obj) || PyTuple_Check(obj)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(obj);
        PyObject **items = PySequence_Fast_ITEMS(obj);
        for (Py_ssize_t i = 0; i < n; i++) {
            if (discover_dtype(items[i], dtype, depth + 1) < 0) return -1;
            if (*dtype == QNP_COMPLEX128) return 0;   /* cannot go higher */
        }
        return 0;
    }
    int weak;
    int dt = qnp_scalar_dtype(obj, &weak);
    if (dt < 0) {
        /* Anything else that converts to a float (a Fraction, a Decimal, a
         * foreign scalar) is treated as one. */
        if (PyIndex_Check(obj)) dt = QNP_INT64;
        else dt = QNP_FLOAT64;
    }
    *dtype = qnp_promote(*dtype, dt);
    return 0;
}

/* Depth-first fill of a freshly allocated, C-contiguous destination. */
static int fill_from_nested(QArray *out, char **cursor, PyObject *obj, int depth) {
    int itemsize = QNP_ITEMSIZE(out->dtype);
    if (depth == out->nd) {
        if (qnp_setitem_ptr(out->dtype, *cursor, obj) < 0) return -1;
        *cursor += itemsize;
        return 0;
    }
    qintp expected = out->shape[depth];
    if (QArray_Check(obj)) {
        QArray *a = (QArray *)obj;
        if (a->nd != out->nd - depth) {
            PyErr_SetString(PyExc_ValueError,
                            "setting an array element with a sequence: the requested "
                            "array has an inhomogeneous shape");
            return -1;
        }
        for (int i = 0; i < a->nd; i++) {
            if (a->shape[i] != out->shape[depth + i]) {
                PyErr_SetString(PyExc_ValueError,
                                "setting an array element with a sequence: the requested "
                                "array has an inhomogeneous shape");
                return -1;
            }
        }
        qintp n = qnp_size(a);
        qintp sub[QNP_MAXDIMS];
        qintp stride = itemsize;
        for (int i = a->nd - 1; i >= 0; i--) { sub[i] = stride; stride *= a->shape[i]; }
        QArray *dstview = qnp_new_view(out, *cursor, a->nd, a->shape, sub, out->dtype);
        if (dstview == NULL) return -1;
        int rc = qnp_copy_into(dstview, a);
        Py_DECREF(dstview);
        if (rc < 0) return -1;
        *cursor += n * itemsize;
        return 0;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        PyErr_SetString(PyExc_ValueError,
                        "setting an array element with a sequence: the requested "
                        "array has an inhomogeneous shape");
        return -1;
    }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(obj);
    if (n != expected) {
        PyErr_SetString(PyExc_ValueError,
                        "setting an array element with a sequence: the requested "
                        "array has an inhomogeneous shape");
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(obj);
    /* The overwhelmingly common shape here is a flat list of floats. */
    if (depth == out->nd - 1 && out->dtype == QNP_FLOAT64) {
        double *dst = (double *)*cursor;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *v = items[i];
            if (PyFloat_CheckExact(v)) dst[i] = PyFloat_AS_DOUBLE(v);
            else {
                double d = PyFloat_AsDouble(v);
                if (d == -1.0 && PyErr_Occurred()) {
                    if (PyList_Check(v) || PyTuple_Check(v) || QArray_Check(v)) {
                        PyErr_Clear();
                        PyErr_SetString(PyExc_ValueError,
                                        "setting an array element with a sequence: the "
                                        "requested array has an inhomogeneous shape");
                    }
                    return -1;
                }
                dst[i] = d;
            }
        }
        *cursor += n * itemsize;
        return 0;
    }
    for (Py_ssize_t i = 0; i < n; i++)
        if (fill_from_nested(out, cursor, items[i], depth + 1) < 0) return -1;
    return 0;
}

/* Zero-copy import of any object exporting a PEP 3118 buffer we understand.
 * This is what lets a real NumPy array still be handed to the library. */
static QArray *from_buffer_object(PyObject *obj, int *unsupported) {
    *unsupported = 1;
    if (!PyObject_CheckBuffer(obj)) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_FULL_RO) < 0) { PyErr_Clear(); return NULL; }
    int dtype = -1;
    if (view.format != NULL) {
        const char *f = view.format;
        if (f[0] == '@' || f[0] == '=' || f[0] == '<' || f[0] == '|') f++;
        if (!strcmp(f, "d")) dtype = QNP_FLOAT64;
        else if (!strcmp(f, "Zd")) dtype = QNP_COMPLEX128;
        else if (!strcmp(f, "?")) dtype = QNP_BOOL;
        else if (!strcmp(f, "q") || !strcmp(f, "l") || !strcmp(f, "n")) dtype = QNP_INT64;
        else if (!strcmp(f, "i")) dtype = -2;      /* 32-bit int: convert below */
        else if (!strcmp(f, "f")) dtype = -3;      /* float32: convert below */
    }
    if (dtype == -1 || view.ndim > QNP_MAXDIMS || view.suboffsets != NULL) {
        PyBuffer_Release(&view);
        return NULL;
    }
    *unsupported = 0;
    qintp shape[QNP_MAXDIMS], strides[QNP_MAXDIMS];
    for (int i = 0; i < view.ndim; i++) {
        shape[i] = view.shape ? view.shape[i] : 0;
        strides[i] = view.strides ? view.strides[i] : 0;
    }
    if (view.strides == NULL) {          /* C-contiguous by contract */
        qintp s = view.itemsize;
        for (int i = view.ndim - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
    }
    QArray *out;
    if (dtype >= 0) {
        out = qnp_new(view.ndim, shape, dtype);
        if (out != NULL) {
            QArray tmp;
            memset(&tmp, 0, sizeof(tmp));
            /* A stack-allocated stand-in: only the fields the copy reads are
             * set, and it never escapes this scope or gets refcounted. */
            tmp.data = (char *)view.buf;
            tmp.nd = view.ndim;
            tmp.shape = shape;
            tmp.strides = strides;
            tmp.dtype = dtype;
            if (qnp_copy_into(out, &tmp) < 0) Py_CLEAR(out);
        }
    } else {
        out = qnp_new(view.ndim, shape, dtype == -2 ? QNP_INT64 : QNP_FLOAT64);
        if (out != NULL) {
            /* Narrow types are widened one element at a time; they only turn
             * up when foreign data is handed in. */
            qintp n = qnp_size(out);
            char *src = (char *)view.buf;
            char *dst = out->data;
            qintp idx[QNP_MAXDIMS] = {0};
            for (qintp k = 0; k < n; k++) {
                char *p = src;
                for (int i = 0; i < view.ndim; i++) p += idx[i] * strides[i];
                if (dtype == -2) *(int64_t *)dst = *(const int32_t *)p;
                else *(double *)dst = *(const float *)p;
                dst += QNP_ITEMSIZE(out->dtype);
                for (int i = view.ndim - 1; i >= 0; i--) {
                    if (++idx[i] < shape[i]) break;
                    idx[i] = 0;
                }
            }
        }
    }
    PyBuffer_Release(&view);
    return out;
}

QArray *qnp_from_any(PyObject *obj, int dtype_hint, int force_dtype) {
    if (QArray_Check(obj)) {
        QArray *a = (QArray *)obj;
        if (force_dtype && dtype_hint >= 0 && a->dtype != dtype_hint)
            return qnp_astype(a, dtype_hint, 1);
        Py_INCREF(a);
        return a;
    }
    int weak;
    int sdt = qnp_scalar_dtype(obj, &weak);
    if (sdt >= 0) {
        int dt = (force_dtype && dtype_hint >= 0) ? dtype_hint : sdt;
        QArray *out = qnp_new(0, NULL, dt);
        if (out == NULL) return NULL;
        if (qnp_setitem_ptr(dt, out->data, obj) < 0) { Py_DECREF(out); return NULL; }
        return out;
    }
    if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
        int unsupported = 0;
        QArray *fb = from_buffer_object(obj, &unsupported);
        if (fb != NULL) {
            if (force_dtype && dtype_hint >= 0 && fb->dtype != dtype_hint) {
                QArray *cast = qnp_astype(fb, dtype_hint, 0);
                Py_DECREF(fb);
                return cast;
            }
            return fb;
        }
        if (PyErr_Occurred()) return NULL;
        if (unsupported) {
            /* Generic sequences (range, a custom container) go through the
             * list path; everything else is a scalar-like object. */
            if (PySequence_Check(obj) && !PyUnicode_Check(obj)) {
                PyObject *as_list = PySequence_List(obj);
                if (as_list == NULL) return NULL;
                QArray *out = qnp_from_any(as_list, dtype_hint, force_dtype);
                Py_DECREF(as_list);
                return out;
            }
            int dt = (force_dtype && dtype_hint >= 0) ? dtype_hint : QNP_FLOAT64;
            if (!force_dtype || dtype_hint < 0) {
                int probe = -1;
                if (discover_dtype(obj, &probe, 0) < 0) return NULL;
                dt = probe < 0 ? QNP_FLOAT64 : probe;
            }
            QArray *out = qnp_new(0, NULL, dt);
            if (out == NULL) return NULL;
            if (qnp_setitem_ptr(dt, out->data, obj) < 0) { Py_DECREF(out); return NULL; }
            return out;
        }
    }
    qintp shape[QNP_MAXDIMS];
    int nd = 0;
    if (discover_shape(obj, shape, &nd, 0) < 0) return NULL;
    int dtype;
    if (force_dtype && dtype_hint >= 0) {
        dtype = dtype_hint;
    } else {
        /* -1 means "nothing seen yet"; an empty sequence lands on float64. */
        dtype = -1;
        if (discover_dtype(obj, &dtype, 0) < 0) return NULL;
        if (dtype < 0) dtype = QNP_FLOAT64;
    }
    QArray *out = qnp_new(nd, shape, dtype);
    if (out == NULL) return NULL;
    char *cursor = out->data;
    if (fill_from_nested(out, &cursor, obj, 0) < 0) { Py_DECREF(out); return NULL; }
    return out;
}

/* --------------------------------------------------- buffer protocol */

static int array_getbuffer(QArray *self, Py_buffer *view, int flags) {
    if (view == NULL) {
        PyErr_SetString(PyExc_BufferError, "NULL view in getbuffer");
        return -1;
    }
    if ((flags & PyBUF_WRITABLE) && !(self->flags & QNP_WRITEABLE)) {
        PyErr_SetString(PyExc_BufferError, "array is not writeable");
        return -1;
    }
    if (!(flags & PyBUF_STRIDES) && !(self->flags & QNP_C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_BufferError, "array is not C-contiguous");
        return -1;
    }
    view->buf = self->data;
    view->obj = (PyObject *)self;
    Py_INCREF(self);
    view->len = qnp_size(self) * QNP_ITEMSIZE(self->dtype);
    view->readonly = (self->flags & QNP_WRITEABLE) ? 0 : 1;
    view->itemsize = QNP_ITEMSIZE(self->dtype);
    view->format = (flags & PyBUF_FORMAT) ? (char *)dtype_formats[self->dtype] : NULL;
    view->ndim = self->nd;
    view->shape = (flags & PyBUF_ND) ? self->shape : NULL;
    view->strides = (flags & PyBUF_STRIDES) ? self->strides : NULL;
    view->suboffsets = NULL;
    view->internal = NULL;
    self->exports++;
    return 0;
}

static void array_releasebuffer(QArray *self, Py_buffer *view) {
    (void)view;
    self->exports--;
}

static PyBufferProcs array_as_buffer = {
    (getbufferproc)array_getbuffer,
    (releasebufferproc)array_releasebuffer,
};

/* ------------------------------------------------------- attributes */

PyObject *qnp_shape_tuple(const QArray *a) {
    PyObject *t = PyTuple_New(a->nd);
    if (t == NULL) return NULL;
    for (int i = 0; i < a->nd; i++) {
        PyObject *v = PyLong_FromSsize_t(a->shape[i]);
        if (v == NULL) { Py_DECREF(t); return NULL; }
        PyTuple_SET_ITEM(t, i, v);
    }
    return t;
}

static PyObject *array_get_shape(QArray *self, void *c) { (void)c; return qnp_shape_tuple(self); }

static int array_set_shape(QArray *self, PyObject *value, void *c) {
    (void)c;
    if (value == NULL) { PyErr_SetString(PyExc_AttributeError, "cannot delete shape"); return -1; }
    PyObject *reshaped = qnp_reshape(self, value);
    if (reshaped == NULL) return -1;
    QArray *r = (QArray *)reshaped;
    if (r->data != self->data) {
        Py_DECREF(reshaped);
        PyErr_SetString(PyExc_AttributeError,
                        "Incompatible shape for in-place modification. Use `.reshape()`");
        return -1;
    }
    PyMem_Free(self->shape);
    self->shape = (qintp *)PyMem_Malloc(2 * (size_t)(r->nd ? r->nd : 1) * sizeof(qintp));
    if (self->shape == NULL) { Py_DECREF(reshaped); PyErr_NoMemory(); return -1; }
    self->strides = self->shape + (r->nd ? r->nd : 1);
    self->nd = r->nd;
    memcpy(self->shape, r->shape, (size_t)r->nd * sizeof(qintp));
    memcpy(self->strides, r->strides, (size_t)r->nd * sizeof(qintp));
    Py_DECREF(reshaped);
    qnp_update_flags(self);
    return 0;
}

static PyObject *array_get_ndim(QArray *self, void *c) { (void)c; return PyLong_FromLong(self->nd); }
static PyObject *array_get_size(QArray *self, void *c) { (void)c; return PyLong_FromSsize_t(qnp_size(self)); }
static PyObject *array_get_itemsize(QArray *self, void *c) { (void)c; return PyLong_FromLong(QNP_ITEMSIZE(self->dtype)); }
static PyObject *array_get_nbytes(QArray *self, void *c) {
    (void)c; return PyLong_FromSsize_t(qnp_size(self) * QNP_ITEMSIZE(self->dtype));
}
static PyObject *array_get_dtype(QArray *self, void *c) { (void)c; return qnp_dtype_object(self->dtype); }

static PyObject *array_get_strides(QArray *self, void *c) {
    (void)c;
    PyObject *t = PyTuple_New(self->nd);
    if (t == NULL) return NULL;
    for (int i = 0; i < self->nd; i++) {
        PyObject *v = PyLong_FromSsize_t(self->strides[i]);
        if (v == NULL) { Py_DECREF(t); return NULL; }
        PyTuple_SET_ITEM(t, i, v);
    }
    return t;
}

static PyObject *array_get_T(QArray *self, void *c) { (void)c; return qnp_transpose(self, NULL); }

static PyObject *array_get_base(QArray *self, void *c) {
    (void)c;
    PyObject *b = self->base ? self->base : Py_None;
    Py_INCREF(b);
    return b;
}

/* `real` and `imag` are strided views into the interleaved complex buffer,
 * exactly as in NumPy, so `a.real += 1` writes through. */
static PyObject *array_get_real(QArray *self, void *c) {
    (void)c;
    if (self->dtype != QNP_COMPLEX128) { Py_INCREF(self); return (PyObject *)self; }
    return (PyObject *)qnp_new_view(self, self->data, self->nd, self->shape,
                                    self->strides, QNP_FLOAT64);
}

static PyObject *array_get_imag(QArray *self, void *c) {
    (void)c;
    if (self->dtype != QNP_COMPLEX128) {
        QArray *out = qnp_new(self->nd, self->shape, self->dtype);
        if (out == NULL) return NULL;
        memset(out->data, 0, (size_t)qnp_size(out) * (size_t)QNP_ITEMSIZE(out->dtype));
        return (PyObject *)out;
    }
    return (PyObject *)qnp_new_view(self, self->data + sizeof(double), self->nd,
                                    self->shape, self->strides, QNP_FLOAT64);
}

static int array_set_real(QArray *self, PyObject *value, void *c) {
    (void)c;
    PyObject *target = array_get_real(self, NULL);
    if (target == NULL) return -1;
    QArray *src = qnp_from_any(value, QNP_FLOAT64, 0);
    if (src == NULL) { Py_DECREF(target); return -1; }
    int rc = qnp_copy_into((QArray *)target, src);
    Py_DECREF(target); Py_DECREF(src);
    return rc;
}

static int array_set_imag(QArray *self, PyObject *value, void *c) {
    (void)c;
    if (self->dtype != QNP_COMPLEX128) {
        PyErr_SetString(PyExc_TypeError, "array does not have imaginary part to set");
        return -1;
    }
    PyObject *target = array_get_imag(self, NULL);
    if (target == NULL) return -1;
    QArray *src = qnp_from_any(value, QNP_FLOAT64, 0);
    if (src == NULL) { Py_DECREF(target); return -1; }
    int rc = qnp_copy_into((QArray *)target, src);
    Py_DECREF(target); Py_DECREF(src);
    return rc;
}

/* NumPy's flags object answers to both `flags["C_CONTIGUOUS"]` and
 * `flags.c_contiguous`, and code in the wild uses each. */
typedef struct {
    PyObject_HEAD
    QArray *array;      /* kept so `flags.writeable = False` writes through */
} QFlags;

static void flags_dealloc(QFlags *self) {
    Py_XDECREF(self->array);
    PyObject_Del(self);
}

static const struct { const char *upper; const char *lower; int bit; } flag_names[] = {
    {"C_CONTIGUOUS", "c_contiguous", QNP_C_CONTIGUOUS},
    {"F_CONTIGUOUS", "f_contiguous", QNP_F_CONTIGUOUS},
    {"OWNDATA", "owndata", QNP_OWNDATA},
    {"WRITEABLE", "writeable", QNP_WRITEABLE},
    {"ALIGNED", "aligned", -1},
    {"C", "c", QNP_C_CONTIGUOUS},
    {"F", "f", QNP_F_CONTIGUOUS},
};

static int flags_lookup(QFlags *self, const char *name, int *value) {
    for (size_t i = 0; i < sizeof(flag_names) / sizeof(flag_names[0]); i++) {
        if (!strcmp(name, flag_names[i].upper) || !strcmp(name, flag_names[i].lower)) {
            *value = flag_names[i].bit < 0
                ? 1 : ((self->array->flags & flag_names[i].bit) != 0);
            return 1;
        }
    }
    return 0;
}

static int flags_setattro(PyObject *self, PyObject *name, PyObject *value) {
    const char *text = PyUnicode_AsUTF8(name);
    if (text != NULL && (!strcmp(text, "writeable") || !strcmp(text, "WRITEABLE"))) {
        if (value == NULL) {
            PyErr_SetString(PyExc_AttributeError, "cannot delete the writeable flag");
            return -1;
        }
        int truth = PyObject_IsTrue(value);
        if (truth < 0) return -1;
        QArray *array = ((QFlags *)self)->array;
        if (truth) {
            if (array->base != NULL && !(((QArray *)array->base)->flags & QNP_WRITEABLE)) {
                PyErr_SetString(PyExc_ValueError,
                                "cannot set WRITEABLE flag on a view of a read-only array");
                return -1;
            }
            array->flags |= QNP_WRITEABLE;
        } else {
            array->flags &= ~QNP_WRITEABLE;
        }
        return 0;
    }
    return PyObject_GenericSetAttr(self, name, value);
}

static PyObject *flags_getattro(PyObject *self, PyObject *name) {
    const char *text = PyUnicode_AsUTF8(name);
    int value;
    if (text != NULL && flags_lookup((QFlags *)self, text, &value))
        return PyBool_FromLong(value);
    return PyObject_GenericGetAttr(self, name);
}

static PyObject *flags_subscript(PyObject *self, PyObject *key) {
    const char *text = PyUnicode_Check(key) ? PyUnicode_AsUTF8(key) : NULL;
    int value;
    if (text != NULL && flags_lookup((QFlags *)self, text, &value))
        return PyBool_FromLong(value);
    PyErr_Format(PyExc_KeyError, "%R", key);
    return NULL;
}

static PyObject *flags_repr(QFlags *self) {
    int bits = self->array->flags;
    return PyUnicode_FromFormat(
        "  C_CONTIGUOUS : %s\n  F_CONTIGUOUS : %s\n  OWNDATA : %s\n  WRITEABLE : %s",
        (bits & QNP_C_CONTIGUOUS) ? "True" : "False",
        (bits & QNP_F_CONTIGUOUS) ? "True" : "False",
        (bits & QNP_OWNDATA) ? "True" : "False",
        (bits & QNP_WRITEABLE) ? "True" : "False");
}

static PyMappingMethods flags_as_mapping = {NULL, flags_subscript, NULL};

static PyTypeObject QFlags_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "quadrivium._qnp.flagsobj",
    .tp_basicsize = sizeof(QFlags),
    .tp_dealloc = (destructor)flags_dealloc,
    .tp_repr = (reprfunc)flags_repr,
    .tp_as_mapping = &flags_as_mapping,
    .tp_getattro = flags_getattro,
    .tp_setattro = flags_setattro,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

static PyObject *array_get_flat(QArray *self, void *c) {
    (void)c;
    PyObject *flat = qnp_ravel(self);
    if (flat == NULL) return NULL;
    /* `ravel` returns a view when the layout permits one and a copy when it
     * does not; a copy is marked read-only so a write that could not reach the
     * original raises instead of vanishing. */
    if (((QArray *)flat)->base != (PyObject *)self &&
        !(self->flags & QNP_C_CONTIGUOUS))
        ((QArray *)flat)->flags &= ~QNP_WRITEABLE;
    return flat;
}

static PyObject *array_get_flags(QArray *self, void *c) {
    (void)c;
    QFlags *flags = PyObject_New(QFlags, &QFlags_Type);
    if (flags == NULL) return NULL;
    Py_INCREF(self);
    flags->array = self;
    return (PyObject *)flags;
}

static PyGetSetDef array_getset[] = {
    {"shape", (getter)array_get_shape, (setter)array_set_shape, NULL, NULL},
    {"ndim", (getter)array_get_ndim, NULL, NULL, NULL},
    {"size", (getter)array_get_size, NULL, NULL, NULL},
    {"itemsize", (getter)array_get_itemsize, NULL, NULL, NULL},
    {"nbytes", (getter)array_get_nbytes, NULL, NULL, NULL},
    {"dtype", (getter)array_get_dtype, NULL, NULL, NULL},
    {"strides", (getter)array_get_strides, NULL, NULL, NULL},
    {"T", (getter)array_get_T, NULL, NULL, NULL},
    {"base", (getter)array_get_base, NULL, NULL, NULL},
    {"real", (getter)array_get_real, (setter)array_set_real, NULL, NULL},
    {"imag", (getter)array_get_imag, (setter)array_set_imag, NULL, NULL},
    {"flags", (getter)array_get_flags, NULL, NULL, NULL},
    {"flat", (getter)array_get_flat, NULL, NULL, NULL},
    {NULL}
};

/* ------------------------------------------------------ number protocol */

static PyObject *nb_add(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_ADD, a, b, NULL, NULL); }
static PyObject *nb_sub(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_SUB, a, b, NULL, NULL); }
static PyObject *nb_mul(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_MUL, a, b, NULL, NULL); }
static PyObject *nb_truediv(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_TRUEDIV, a, b, NULL, NULL); }
static PyObject *nb_floordiv(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_FLOORDIV, a, b, NULL, NULL); }
static PyObject *nb_mod(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_MOD, a, b, NULL, NULL); }
static PyObject *nb_and(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_AND, a, b, NULL, NULL); }
static PyObject *nb_or(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_OR, a, b, NULL, NULL); }
static PyObject *nb_xor(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_XOR, a, b, NULL, NULL); }
static PyObject *nb_matmul(PyObject *a, PyObject *b) { return qnp_matmul(a, b); }
static PyObject *nb_lshift(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_LSHIFT, a, b, NULL, NULL); }
static PyObject *nb_rshift(PyObject *a, PyObject *b) { return qnp_binary_op(QOP_RSHIFT, a, b, NULL, NULL); }

static PyObject *nb_power(PyObject *a, PyObject *b, PyObject *m) {
    if (m != Py_None) Py_RETURN_NOTIMPLEMENTED;
    return qnp_binary_op(QOP_POW, a, b, NULL, NULL);
}

static PyObject *nb_negative(PyObject *a) { return qnp_unary_op(QOP_NEG, a, NULL); }
static PyObject *nb_positive(PyObject *a) { return qnp_unary_op(QOP_POS, a, NULL); }
static PyObject *nb_absolute(PyObject *a) { return qnp_unary_op(QOP_ABS, a, NULL); }
static PyObject *nb_invert(PyObject *a) { return qnp_unary_op(QOP_INVERT, a, NULL); }

/* In-place forms write into the left operand when it is an array we own; the
 * saved allocation is the whole point of `a += b` in the numerical code. */
static PyObject *inplace(int op, PyObject *a, PyObject *b) {
    if (!QArray_Check(a)) return qnp_binary_op(op, a, b, NULL, NULL);
    PyObject *r = qnp_binary_op(op, a, b, a, NULL);
    return r;
}
static PyObject *nb_iadd(PyObject *a, PyObject *b) { return inplace(QOP_ADD, a, b); }
static PyObject *nb_isub(PyObject *a, PyObject *b) { return inplace(QOP_SUB, a, b); }
static PyObject *nb_imul(PyObject *a, PyObject *b) { return inplace(QOP_MUL, a, b); }
static PyObject *nb_itruediv(PyObject *a, PyObject *b) { return inplace(QOP_TRUEDIV, a, b); }
static PyObject *nb_ifloordiv(PyObject *a, PyObject *b) { return inplace(QOP_FLOORDIV, a, b); }
static PyObject *nb_imod(PyObject *a, PyObject *b) { return inplace(QOP_MOD, a, b); }
static PyObject *nb_iand(PyObject *a, PyObject *b) { return inplace(QOP_AND, a, b); }
static PyObject *nb_ior(PyObject *a, PyObject *b) { return inplace(QOP_OR, a, b); }
static PyObject *nb_ixor(PyObject *a, PyObject *b) { return inplace(QOP_XOR, a, b); }
static PyObject *nb_ilshift(PyObject *a, PyObject *b) { return inplace(QOP_LSHIFT, a, b); }
static PyObject *nb_irshift(PyObject *a, PyObject *b) { return inplace(QOP_RSHIFT, a, b); }
static PyObject *nb_ipow(PyObject *a, PyObject *b, PyObject *m) {
    if (m != Py_None) Py_RETURN_NOTIMPLEMENTED;
    return inplace(QOP_POW, a, b);
}
static PyObject *nb_imatmul(PyObject *a, PyObject *b) { return qnp_matmul(a, b); }

static int array_bool(QArray *self) {
    qintp n = qnp_size(self);
    if (n == 1) {
        PyObject *v = qnp_getitem_ptr(self->dtype, self->data);
        if (v == NULL) return -1;
        int r = PyObject_IsTrue(v);
        Py_DECREF(v);
        return r;
    }
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The truth value of an empty array is ambiguous.");
        return -1;
    }
    PyErr_SetString(PyExc_ValueError,
                    "The truth value of an array with more than one element is "
                    "ambiguous. Use a.any() or a.all()");
    return -1;
}

static PyObject *array_as_pyscalar(QArray *self, const char *what) {
    if (qnp_size(self) != 1) {
        PyErr_Format(PyExc_TypeError,
                     "only length-1 arrays can be converted to Python %s", what);
        return NULL;
    }
    return qnp_getitem_ptr(self->dtype, self->data);
}

static PyObject *nb_float(QArray *self) {
    PyObject *v = array_as_pyscalar(self, "scalars");
    if (v == NULL) return NULL;
    PyObject *f = PyNumber_Float(v);
    Py_DECREF(v);
    return f;
}

static PyObject *nb_int(QArray *self) {
    PyObject *v = array_as_pyscalar(self, "integers");
    if (v == NULL) return NULL;
    PyObject *f = PyNumber_Long(v);
    Py_DECREF(v);
    return f;
}

static PyObject *nb_index(QArray *self) {
    if (self->dtype != QNP_INT64 && self->dtype != QNP_BOOL) {
        PyErr_SetString(PyExc_TypeError,
                        "only integer scalar arrays can be converted to a scalar index");
        return NULL;
    }
    return nb_int(self);
}

static PyNumberMethods array_as_number = {
    .nb_add = nb_add, .nb_subtract = nb_sub, .nb_multiply = nb_mul,
    .nb_remainder = nb_mod, .nb_power = nb_power, .nb_negative = nb_negative,
    .nb_positive = nb_positive, .nb_absolute = nb_absolute, .nb_bool = (inquiry)array_bool,
    .nb_invert = nb_invert, .nb_and = nb_and, .nb_xor = nb_xor, .nb_or = nb_or,
    .nb_int = (unaryfunc)nb_int, .nb_float = (unaryfunc)nb_float,
    .nb_inplace_add = nb_iadd, .nb_inplace_subtract = nb_isub,
    .nb_inplace_multiply = nb_imul, .nb_inplace_remainder = nb_imod,
    .nb_inplace_power = nb_ipow, .nb_inplace_and = nb_iand,
    .nb_inplace_xor = nb_ixor, .nb_inplace_or = nb_ior,
    .nb_floor_divide = nb_floordiv, .nb_true_divide = nb_truediv,
    .nb_inplace_floor_divide = nb_ifloordiv, .nb_inplace_true_divide = nb_itruediv,
    .nb_index = (unaryfunc)nb_index,
    .nb_lshift = nb_lshift, .nb_rshift = nb_rshift,
    .nb_inplace_lshift = nb_ilshift, .nb_inplace_rshift = nb_irshift,
    .nb_matrix_multiply = nb_matmul, .nb_inplace_matrix_multiply = nb_imatmul,
};

static PyObject *array_richcompare(PyObject *a, PyObject *b, int op) {
    static const int map[6] = {QOP_LT, QOP_LE, QOP_EQ, QOP_NE, QOP_GT, QOP_GE};
    return qnp_binary_op(map[op], a, b, NULL, NULL);
}

/* ------------------------------------------------ mapping and sequence */

static Py_ssize_t array_length(QArray *self) {
    if (self->nd == 0) {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object");
        return -1;
    }
    return self->shape[0];
}

static PyObject *array_subscript(QArray *self, PyObject *key) { return qnp_getitem(self, key); }
static int array_ass_subscript(QArray *self, PyObject *key, PyObject *value) {
    if (value == NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot delete array elements");
        return -1;
    }
    return qnp_setitem(self, key, value);
}

static PyMappingMethods array_as_mapping = {
    (lenfunc)array_length,
    (binaryfunc)array_subscript,
    (objobjargproc)array_ass_subscript,
};

/* `x in arr` is (arr == x).any(), as in NumPy. */
static int array_contains(QArray *self, PyObject *value) {
    PyObject *eq = qnp_binary_op(QOP_EQ, (PyObject *)self, value, NULL, NULL);
    if (eq == NULL) return -1;
    PyObject *any = qnp_reduce(QRED_ANY, eq, Py_None, NULL, 0);
    Py_DECREF(eq);
    if (any == NULL) return -1;
    int r = PyObject_IsTrue(any);
    Py_DECREF(any);
    return r;
}

/* Indexing an array by a single integer also goes through the sequence
 * protocol, which is what makes `PySequence_Check` true and lets C callers --
 * the compiled kernels among them -- treat an array as a sequence of rows. */
static PyObject *array_item(QArray *self, Py_ssize_t index) {
    if (self->nd == 0) {
        PyErr_SetString(PyExc_IndexError, "too many indices for array");
        return NULL;
    }
    PyObject *key = PyLong_FromSsize_t(index);
    if (key == NULL) return NULL;
    PyObject *item = qnp_getitem(self, key);
    Py_DECREF(key);
    return item;
}

static int array_ass_item(QArray *self, Py_ssize_t index, PyObject *value) {
    if (value == NULL) {
        PyErr_SetString(PyExc_ValueError, "cannot delete array elements");
        return -1;
    }
    PyObject *key = PyLong_FromSsize_t(index);
    if (key == NULL) return -1;
    int rc = qnp_setitem(self, key, value);
    Py_DECREF(key);
    return rc;
}

static PySequenceMethods array_as_sequence = {
    .sq_length = (lenfunc)array_length,
    .sq_item = (ssizeargfunc)array_item,
    .sq_ass_item = (ssizeobjargproc)array_ass_item,
    .sq_contains = (objobjproc)array_contains,
};

/* ------------------------------------------------------------- display */

static PyObject *array_repr_impl(QArray *self, int is_repr) {
    if (qnp_printer != NULL) {
        PyObject *r = PyObject_CallFunction(qnp_printer, "Oi", (PyObject *)self, is_repr);
        if (r != NULL) return r;
        PyErr_Clear();
    }
    PyObject *sh = qnp_shape_tuple(self);
    PyObject *r = PyUnicode_FromFormat("<array shape=%R dtype=%s>", sh,
                                       dtype_names[self->dtype]);
    Py_XDECREF(sh);
    return r;
}

static PyObject *array_repr(QArray *self) { return array_repr_impl(self, 1); }
static PyObject *array_str(QArray *self) { return array_repr_impl(self, 0); }

/* --------------------------------------------------------- constructor */

static PyObject *array_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    (void)type;
    PyObject *shape_obj, *dtype_obj = NULL;
    static char *kwlist[] = {"shape", "dtype", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O:ndarray", kwlist,
                                     &shape_obj, &dtype_obj)) return NULL;
    qintp shape[QNP_MAXDIMS];
    int nd;
    if (qnp_shape_from_object(shape_obj, shape, &nd) < 0) return NULL;
    int dtype = QNP_FLOAT64;
    if (dtype_obj != NULL && dtype_obj != Py_None) {
        int ok = 1;
        dtype = qnp_dtype_from_object(dtype_obj, &ok);
        if (!ok) return NULL;
    }
    return (PyObject *)qnp_new(nd, shape, dtype);
}

/* Iterating an array yields rows, which for 1-D means Python scalars. */
typedef struct {
    PyObject_HEAD
    QArray *arr;
    qintp index;
} QArrayIter;

static void arrayiter_dealloc(QArrayIter *it) {
    Py_XDECREF(it->arr);
    PyObject_Del(it);
}

static PyObject *arrayiter_next(QArrayIter *it) {
    if (it->arr == NULL) return NULL;
    if (it->index >= it->arr->shape[0]) {
        Py_CLEAR(it->arr);
        return NULL;
    }
    PyObject *key = PyLong_FromSsize_t(it->index++);
    if (key == NULL) return NULL;
    PyObject *item = qnp_getitem(it->arr, key);
    Py_DECREF(key);
    return item;
}

static PyTypeObject QArrayIter_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "quadrivium._qnp.arrayiterator",
    .tp_basicsize = sizeof(QArrayIter),
    .tp_dealloc = (destructor)arrayiter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)arrayiter_next,
};

static PyObject *array_iter(QArray *self) {
    if (self->nd == 0) {
        PyErr_SetString(PyExc_TypeError, "iteration over a 0-d array");
        return NULL;
    }
    QArrayIter *it = PyObject_New(QArrayIter, &QArrayIter_Type);
    if (it == NULL) return NULL;
    Py_INCREF(self);
    it->arr = self;
    it->index = 0;
    return (PyObject *)it;
}

PyTypeObject QArray_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "quadrivium._qnp.ndarray",
    .tp_basicsize = sizeof(QArray),
    .tp_dealloc = (destructor)array_dealloc,
    .tp_repr = (reprfunc)array_repr,
    .tp_str = (reprfunc)array_str,
    .tp_as_number = &array_as_number,
    .tp_as_sequence = &array_as_sequence,
    .tp_as_mapping = &array_as_mapping,
    .tp_as_buffer = &array_as_buffer,
    .tp_hash = PyObject_HashNotImplemented,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE,
    .tp_traverse = (traverseproc)array_traverse,
    .tp_clear = (inquiry)array_clear,
    .tp_richcompare = array_richcompare,
    .tp_weaklistoffset = offsetof(QArray, weakreflist),
    .tp_iter = (getiterfunc)array_iter,
    .tp_methods = qnp_array_methods,
    .tp_getset = array_getset,
    .tp_new = array_new,
};

/* ----------------------------------------------------- module wiring */

static PyObject *set_printer(PyObject *self, PyObject *fn) {
    (void)self;
    Py_XDECREF(qnp_printer);
    qnp_printer = (fn == Py_None) ? NULL : Py_NewRef(fn);
    Py_RETURN_NONE;
}

PyObject *qnp_wrap_scalar_or_array(QArray *a) {
    if (a != NULL && a->nd == 0) {
        PyObject *v = qnp_getitem_ptr(a->dtype, a->data);
        Py_DECREF(a);
        return v;
    }
    return (PyObject *)a;
}

int qnp_shape_from_object(PyObject *obj, qintp *shape, int *nd) {
    if (QArray_Check(obj)) {
        /* A shape given as an array: a 0-d one is a single dimension, and
         * anything else is read as the sequence of dimensions it is. */
        QArray *a = (QArray *)obj;
        if (a->nd == 0) {
            PyObject *value = qnp_getitem_ptr(a->dtype, a->data);
            if (value == NULL) return -1;
            Py_ssize_t n = PyNumber_AsSsize_t(value, PyExc_OverflowError);
            Py_DECREF(value);
            if (n == -1 && PyErr_Occurred()) return -1;
            shape[0] = n;
            *nd = 1;
            return 0;
        }
        PyObject *as_list = PyObject_CallMethod(obj, "tolist", NULL);
        if (as_list == NULL) return -1;
        int rc = qnp_shape_from_object(as_list, shape, nd);
        Py_DECREF(as_list);
        return rc;
    }
    if (PyIndex_Check(obj)) {
        Py_ssize_t n = PyNumber_AsSsize_t(obj, PyExc_OverflowError);
        if (n == -1 && PyErr_Occurred()) return -1;
        shape[0] = n;
        *nd = 1;
        return 0;
    }
    PyObject *seq = PySequence_Fast(obj, "shape must be an int or a sequence of ints");
    if (seq == NULL) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n > QNP_MAXDIMS) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "maximum supported dimension is %d", QNP_MAXDIMS);
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; i++) {
        Py_ssize_t v = PyNumber_AsSsize_t(items[i], PyExc_OverflowError);
        if (v == -1 && PyErr_Occurred()) { Py_DECREF(seq); return -1; }
        shape[i] = v;
    }
    *nd = (int)n;
    Py_DECREF(seq);
    return 0;
}

PyMethodDef qnp_array_core_methods[] = {
    {"set_printer", set_printer, METH_O,
     "Install the Python callable used to render arrays."},
    {NULL}
};

int qnp_init_types(PyObject *module) {
    if (PyType_Ready(&QDtype_Type) < 0) return -1;
    if (PyType_Ready(&QArray_Type) < 0) return -1;
    if (PyType_Ready(&QArrayIter_Type) < 0) return -1;
    if (PyType_Ready(&QFlags_Type) < 0) return -1;
    for (int i = 0; i < QNP_NTYPES; i++) {
        QDtype *d = PyObject_New(QDtype, &QDtype_Type);
        if (d == NULL) return -1;
        d->num = i;
        dtype_singletons[i] = d;
    }
    QNP_LinAlgError = PyErr_NewExceptionWithDoc(
        "quadrivium._qnp.LinAlgError",
        "Raised when a linear algebra routine fails to produce a result.",
        PyExc_ValueError, NULL);
    if (QNP_LinAlgError == NULL) return -1;
    Py_INCREF(&QArray_Type);
    if (PyModule_AddObject(module, "ndarray", (PyObject *)&QArray_Type) < 0) return -1;
    Py_INCREF(&QDtype_Type);
    if (PyModule_AddObject(module, "dtype", (PyObject *)&QDtype_Type) < 0) return -1;
    Py_INCREF(QNP_LinAlgError);
    if (PyModule_AddObject(module, "LinAlgError", QNP_LinAlgError) < 0) return -1;
    for (int i = 0; i < QNP_NTYPES; i++) {
        PyObject *d = qnp_dtype_object(i);
        if (d == NULL || PyModule_AddObject(module, dtype_names[i], d) < 0) return -1;
    }
    return 0;
}
