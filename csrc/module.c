/* Module assembly: the ufunc wrappers, the ndarray method table and the
 * module object itself. */
#include "qnp.h"

extern PyMethodDef qnp_index_methods[];
extern PyMethodDef qnp_array_core_methods[];
int qnp_init_types(PyObject *module);

/* ---- ufunc wrappers ---------------------------------------------------- */

static PyObject *binary_wrapper(int op, PyObject *args, PyObject *kwds) {
    PyObject *a, *b, *out = Py_None, *where = Py_None;
    static char *kwlist[] = {"x1", "x2", "out", "where", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OO", kwlist, &a, &b, &out, &where))
        return NULL;
    return qnp_binary_op(op, a, b, out, where);
}

static PyObject *unary_wrapper(int op, PyObject *args, PyObject *kwds) {
    PyObject *a, *out = Py_None;
    static char *kwlist[] = {"x", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &a, &out)) return NULL;
    return qnp_unary_op(op, a, out);
}

#define BINARY_FN(NAME, OP)                                                    \
static PyObject *py_##NAME(PyObject *self, PyObject *args, PyObject *kwds) {   \
    (void)self; return binary_wrapper(OP, args, kwds);                         \
}
#define UNARY_FN(NAME, OP)                                                     \
static PyObject *py_##NAME(PyObject *self, PyObject *args, PyObject *kwds) {   \
    (void)self; return unary_wrapper(OP, args, kwds);                          \
}

BINARY_FN(add, QOP_ADD)
BINARY_FN(subtract, QOP_SUB)
BINARY_FN(multiply, QOP_MUL)
BINARY_FN(divide, QOP_TRUEDIV)
BINARY_FN(floor_divide, QOP_FLOORDIV)
BINARY_FN(remainder, QOP_MOD)
BINARY_FN(power, QOP_POW)
BINARY_FN(maximum, QOP_MAXIMUM)
BINARY_FN(minimum, QOP_MINIMUM)
BINARY_FN(hypot, QOP_HYPOT)
BINARY_FN(arctan2, QOP_ARCTAN2)
BINARY_FN(copysign, QOP_COPYSIGN)
BINARY_FN(equal, QOP_EQ)
BINARY_FN(not_equal, QOP_NE)
BINARY_FN(less, QOP_LT)
BINARY_FN(less_equal, QOP_LE)
BINARY_FN(greater, QOP_GT)
BINARY_FN(greater_equal, QOP_GE)
BINARY_FN(logical_and, QOP_LOGICAL_AND)
BINARY_FN(logical_or, QOP_LOGICAL_OR)
BINARY_FN(bitwise_and, QOP_AND)
BINARY_FN(bitwise_or, QOP_OR)
BINARY_FN(bitwise_xor, QOP_XOR)
BINARY_FN(left_shift, QOP_LSHIFT)
BINARY_FN(right_shift, QOP_RSHIFT)

UNARY_FN(negative, QOP_NEG)
UNARY_FN(absolute, QOP_ABS)
UNARY_FN(sqrt, QOP_SQRT)
UNARY_FN(exp, QOP_EXP)
UNARY_FN(log, QOP_LOG)
UNARY_FN(log2, QOP_LOG2)
UNARY_FN(log10, QOP_LOG10)
UNARY_FN(log1p, QOP_LOG1P)
UNARY_FN(expm1, QOP_EXPM1)
UNARY_FN(sin, QOP_SIN)
UNARY_FN(cos, QOP_COS)
UNARY_FN(tan, QOP_TAN)
UNARY_FN(arcsin, QOP_ARCSIN)
UNARY_FN(arccos, QOP_ARCCOS)
UNARY_FN(arctan, QOP_ARCTAN)
UNARY_FN(sinh, QOP_SINH)
UNARY_FN(cosh, QOP_COSH)
UNARY_FN(tanh, QOP_TANH)
UNARY_FN(arcsinh, QOP_ARCSINH)
UNARY_FN(arccosh, QOP_ARCCOSH)
UNARY_FN(arctanh, QOP_ARCTANH)
UNARY_FN(sign, QOP_SIGN)
UNARY_FN(floor, QOP_FLOOR)
UNARY_FN(ceil, QOP_CEIL)
UNARY_FN(trunc, QOP_TRUNC)
UNARY_FN(rint, QOP_RINT)
UNARY_FN(square, QOP_SQUARE)
UNARY_FN(reciprocal, QOP_RECIPROCAL)
UNARY_FN(conjugate, QOP_CONJ)
UNARY_FN(real, QOP_REAL)
UNARY_FN(imag, QOP_IMAG)
UNARY_FN(angle, QOP_ANGLE)
UNARY_FN(isfinite, QOP_ISFINITE)
UNARY_FN(isnan, QOP_ISNAN)
UNARY_FN(isinf, QOP_ISINF)
UNARY_FN(logical_not, QOP_NOT)
UNARY_FN(invert, QOP_INVERT)
UNARY_FN(signbit, QOP_SIGNBIT)
#undef BINARY_FN
#undef UNARY_FN

static PyObject *py_matmul(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *a, *b;
    if (!PyArg_ParseTuple(args, "OO:matmul", &a, &b)) return NULL;
    return qnp_matmul(a, b);
}

/* Rounding to a number of decimals, the way NumPy does it: scale, round to
 * even, scale back. */
static PyObject *round_impl(PyObject *obj, int decimals, PyObject *out) {
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    if (a->dtype == QNP_BOOL || a->dtype == QNP_INT64) {
        if (decimals >= 0) {
            QArray *c = qnp_astype(a, a->dtype, 1);
            Py_DECREF(a);
            return (PyObject *)c;
        }
    }
    if (decimals == 0) {
        PyObject *r = qnp_unary_op(QOP_RINT, (PyObject *)a, out);
        Py_DECREF(a);
        return r;
    }
    double scale = pow(10.0, (double)decimals);
    PyObject *sc = PyFloat_FromDouble(scale);
    if (sc == NULL) { Py_DECREF(a); return NULL; }
    PyObject *scaled = qnp_binary_op(QOP_MUL, (PyObject *)a, sc, NULL, NULL);
    Py_DECREF(a);
    if (scaled == NULL) { Py_DECREF(sc); return NULL; }
    PyObject *rounded = qnp_unary_op(QOP_RINT, scaled, NULL);
    Py_DECREF(scaled);
    if (rounded == NULL) { Py_DECREF(sc); return NULL; }
    PyObject *result = qnp_binary_op(QOP_TRUEDIV, rounded, sc, out, NULL);
    Py_DECREF(rounded);
    Py_DECREF(sc);
    return result;
}

static PyObject *py_round(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *a, *out = Py_None;
    int decimals = 0;
    static char *kwlist[] = {"a", "decimals", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO:round", kwlist, &a, &decimals, &out))
        return NULL;
    return round_impl(a, decimals, out);
}

/* ---- ndarray methods --------------------------------------------------- */

static PyObject *arr_copy(QArray *self, PyObject *args) {
    (void)args;
    return (PyObject *)qnp_astype(self, self->dtype, 1);
}

static PyObject *arr_astype(QArray *self, PyObject *args, PyObject *kwds) {
    PyObject *dtype_obj;
    int copy = 1;
    static char *kwlist[] = {"dtype", "copy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p:astype", kwlist, &dtype_obj, &copy))
        return NULL;
    int ok = 1;
    int dt = qnp_dtype_from_object(dtype_obj, &ok);
    if (!ok) return NULL;
    return (PyObject *)qnp_astype(self, dt, copy);
}

static PyObject *arr_reshape(QArray *self, PyObject *args) {
    PyObject *shape = args;
    if (PyTuple_GET_SIZE(args) == 1) shape = PyTuple_GET_ITEM(args, 0);
    return qnp_reshape(self, shape);
}

static PyObject *arr_ravel(QArray *self, PyObject *args) { (void)args; return qnp_ravel(self); }

static PyObject *arr_flatten(QArray *self, PyObject *args) {
    (void)args;
    QArray *c = qnp_ascontiguous(self);
    if (c == NULL) return NULL;
    QArray *copy = qnp_astype(c, c->dtype, 1);
    Py_DECREF(c);
    if (copy == NULL) return NULL;
    PyObject *flat = qnp_ravel(copy);
    Py_DECREF(copy);
    return flat;
}

static PyObject *arr_transpose(QArray *self, PyObject *args) {
    PyObject *axes = NULL;
    if (PyTuple_GET_SIZE(args) == 1) axes = PyTuple_GET_ITEM(args, 0);
    else if (PyTuple_GET_SIZE(args) > 1) axes = args;
    return qnp_transpose(self, axes);
}

static PyObject *arr_conj(QArray *self, PyObject *args) {
    (void)args;
    return qnp_unary_op(QOP_CONJ, (PyObject *)self, NULL);
}

static PyObject *arr_fill(QArray *self, PyObject *value) {
    QArray *src = qnp_from_any(value, self->dtype, 1);
    if (src == NULL) return NULL;
    int rc = qnp_copy_into(self, src);
    Py_DECREF(src);
    if (rc < 0) return NULL;
    Py_RETURN_NONE;
}

static PyObject *arr_item(QArray *self, PyObject *args) {
    if (PyTuple_GET_SIZE(args) == 0) {
        if (qnp_size(self) != 1) {
            PyErr_SetString(PyExc_ValueError,
                            "can only convert an array of size 1 to a Python scalar");
            return NULL;
        }
        return qnp_getitem_ptr(self->dtype, self->data);
    }
    PyObject *key = PyTuple_GET_SIZE(args) == 1 ? PyTuple_GET_ITEM(args, 0) : args;
    return qnp_getitem(self, key);
}

static PyObject *tolist_recursive(QArray *a, const char *data, int depth) {
    if (depth == a->nd) return qnp_getitem_ptr(a->dtype, data);
    PyObject *list = PyList_New(a->shape[depth]);
    if (list == NULL) return NULL;
    for (qintp i = 0; i < a->shape[depth]; i++) {
        PyObject *item = tolist_recursive(a, data + i * a->strides[depth], depth + 1);
        if (item == NULL) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}

static PyObject *arr_tolist(QArray *self, PyObject *args) {
    (void)args;
    return tolist_recursive(self, self->data, 0);
}

static PyObject *arr_squeeze(QArray *self, PyObject *args, PyObject *kwds) {
    PyObject *axis_o = Py_None;
    static char *kwlist[] = {"axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:squeeze", kwlist, &axis_o)) return NULL;
    qintp shape[QNP_MAXDIMS], strides[QNP_MAXDIMS];
    int nd = 0;
    int only = -1;
    if (axis_o != Py_None && qnp_parse_axis(axis_o, self->nd, &only) < 0) return NULL;
    for (int i = 0; i < self->nd; i++) {
        if (self->shape[i] == 1 && (only < 0 || only == i)) continue;
        shape[nd] = self->shape[i];
        strides[nd] = self->strides[i];
        nd++;
    }
    return (PyObject *)qnp_new_view(self, self->data, nd, shape, strides, self->dtype);
}

static PyObject *arr_dot(QArray *self, PyObject *other) {
    return qnp_matmul((PyObject *)self, other);
}

static PyObject *arr_round(QArray *self, PyObject *args, PyObject *kwds) {
    int decimals = 0;
    PyObject *out = Py_None;
    static char *kwlist[] = {"decimals", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO:round", kwlist, &decimals, &out))
        return NULL;
    return round_impl((PyObject *)self, decimals, out);
}

static PyObject *arr_clip(QArray *self, PyObject *args, PyObject *kwds) {
    PyObject *lo = Py_None, *hi = Py_None, *out = Py_None;
    static char *kwlist[] = {"a_min", "a_max", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO:clip", kwlist, &lo, &hi, &out))
        return NULL;
    PyObject *cur = Py_NewRef((PyObject *)self);
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
    if (cur == (PyObject *)self) {
        Py_DECREF(cur);
        return (PyObject *)qnp_astype(self, self->dtype, 1);
    }
    return cur;
}

static PyObject *arr_view(QArray *self, PyObject *args) {
    PyObject *what = NULL;
    if (!PyArg_ParseTuple(args, "|O:view", &what)) return NULL;
    PyTypeObject *type = &QArray_Type;
    int dt = self->dtype;
    if (what != NULL && what != Py_None) {
        if (PyType_Check(what) && PyType_IsSubtype((PyTypeObject *)what, &QArray_Type)) {
            type = (PyTypeObject *)what;
        } else {
            int ok = 1;
            dt = qnp_dtype_from_object(what, &ok);
            if (!ok) return NULL;
            if (QNP_ITEMSIZE(dt) != QNP_ITEMSIZE(self->dtype)) {
                PyErr_SetString(PyExc_ValueError,
                                "view is limited to dtypes of the same item size");
                return NULL;
            }
        }
    }
    QArray *view = qnp_new_view_as(type, self, self->data, self->nd, self->shape,
                                   self->strides, dt);
    if (view == NULL) return NULL;
    if (type != &QArray_Type) {
        /* Subclasses get the hook NumPy calls when a view adopts their type. */
        PyObject *hook = PyObject_GetAttrString((PyObject *)view, "__array_finalize__");
        if (hook == NULL) {
            PyErr_Clear();
        } else {
            PyObject *result = PyObject_CallFunctionObjArgs(hook, (PyObject *)self, NULL);
            Py_DECREF(hook);
            if (result == NULL) { Py_DECREF(view); return NULL; }
            Py_DECREF(result);
        }
    }
    return (PyObject *)view;
}

#define ARR_REDUCE(NAME, KIND)                                                 \
static PyObject *arr_##NAME(QArray *self, PyObject *args, PyObject *kwds) {    \
    PyObject *axis = Py_None, *out = Py_None;                                  \
    int keepdims = 0;                                                          \
    static char *kwlist[] = {"axis", "out", "keepdims", NULL};                 \
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOp:" #NAME, kwlist,        \
                                     &axis, &out, &keepdims)) return NULL;     \
    return qnp_reduce(KIND, (PyObject *)self, axis, out, keepdims);            \
}
ARR_REDUCE(sum, QRED_SUM)
ARR_REDUCE(prod, QRED_PROD)
ARR_REDUCE(max, QRED_MAX)
ARR_REDUCE(min, QRED_MIN)
ARR_REDUCE(any, QRED_ANY)
ARR_REDUCE(all, QRED_ALL)
ARR_REDUCE(argmax, QRED_ARGMAX)
ARR_REDUCE(argmin, QRED_ARGMIN)
ARR_REDUCE(mean, QRED_MEAN)
#undef ARR_REDUCE

static PyObject *arr_moment(QArray *self, PyObject *args, PyObject *kwds, int want_std) {
    PyObject *axis = Py_None;
    double ddof = 0.0;
    int keepdims = 0;
    static char *kwlist[] = {"axis", "ddof", "keepdims", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Odp", kwlist, &axis, &ddof, &keepdims))
        return NULL;
    return qnp_moment((PyObject *)self, axis, ddof, want_std, keepdims);
}
static PyObject *arr_std(QArray *s, PyObject *a, PyObject *k) { return arr_moment(s, a, k, 1); }
static PyObject *arr_var(QArray *s, PyObject *a, PyObject *k) { return arr_moment(s, a, k, 0); }

static PyObject *arr_cumsum(QArray *self, PyObject *args, PyObject *kwds) {
    PyObject *axis = Py_None, *out = Py_None;
    static char *kwlist[] = {"axis", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO:cumsum", kwlist, &axis, &out))
        return NULL;
    return qnp_accumulate(QRED_SUM, (PyObject *)self, axis, out);
}

static PyObject *arr_cumprod(QArray *self, PyObject *args, PyObject *kwds) {
    PyObject *axis = Py_None, *out = Py_None;
    static char *kwlist[] = {"axis", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO:cumprod", kwlist, &axis, &out))
        return NULL;
    return qnp_accumulate(QRED_PROD, (PyObject *)self, axis, out);
}

/* Methods that simply forward to the module-level function of the same name. */
static PyObject *call_module_fn(const char *name, PyObject *args, PyObject *kwds) {
    PyObject *module = PyImport_ImportModule("quadrivium._qnp");
    if (module == NULL) return NULL;
    PyObject *fn = PyObject_GetAttrString(module, name);
    Py_DECREF(module);
    if (fn == NULL) return NULL;
    PyObject *result = PyObject_Call(fn, args, kwds);
    Py_DECREF(fn);
    return result;
}

#define FORWARD_METHOD(NAME)                                                   \
static PyObject *arr_##NAME(QArray *self, PyObject *args, PyObject *kwds) {    \
    PyObject *full = PyTuple_New(PyTuple_GET_SIZE(args) + 1);                  \
    if (full == NULL) return NULL;                                             \
    PyTuple_SET_ITEM(full, 0, Py_NewRef((PyObject *)self));                    \
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); i++)                    \
        PyTuple_SET_ITEM(full, i + 1, Py_NewRef(PyTuple_GET_ITEM(args, i)));   \
    PyObject *r = call_module_fn(#NAME, full, kwds);                           \
    Py_DECREF(full);                                                           \
    return r;                                                                  \
}
FORWARD_METHOD(sort)
FORWARD_METHOD(argsort)
FORWARD_METHOD(repeat)
FORWARD_METHOD(take)
FORWARD_METHOD(nonzero)
FORWARD_METHOD(searchsorted)
#undef FORWARD_METHOD

PyMethodDef qnp_array_methods[] = {
    {"copy", (PyCFunction)arr_copy, METH_VARARGS, "Copy of the array."},
    {"astype", (PyCFunction)arr_astype, METH_VARARGS | METH_KEYWORDS, "Cast to another dtype."},
    {"reshape", (PyCFunction)arr_reshape, METH_VARARGS, "Array with a new shape."},
    {"ravel", (PyCFunction)arr_ravel, METH_VARARGS, "Flattened view where possible."},
    {"flatten", (PyCFunction)arr_flatten, METH_VARARGS, "Flattened copy."},
    {"transpose", (PyCFunction)arr_transpose, METH_VARARGS, "Permuted view."},
    {"conj", (PyCFunction)arr_conj, METH_VARARGS, "Complex conjugate."},
    {"conjugate", (PyCFunction)arr_conj, METH_VARARGS, "Complex conjugate."},
    {"fill", (PyCFunction)arr_fill, METH_O, "Fill with a scalar."},
    {"item", (PyCFunction)arr_item, METH_VARARGS, "Element as a Python scalar."},
    {"tolist", (PyCFunction)arr_tolist, METH_VARARGS, "Nested list of Python scalars."},
    {"squeeze", (PyCFunction)arr_squeeze, METH_VARARGS | METH_KEYWORDS, "Drop length-1 dimensions."},
    {"dot", (PyCFunction)arr_dot, METH_O, "Matrix product."},
    {"round", (PyCFunction)arr_round, METH_VARARGS | METH_KEYWORDS, "Round to a number of decimals."},
    {"clip", (PyCFunction)arr_clip, METH_VARARGS | METH_KEYWORDS, "Limit values to an interval."},
    {"view", (PyCFunction)arr_view, METH_VARARGS, "View with another dtype of the same size."},
    {"sum", (PyCFunction)arr_sum, METH_VARARGS | METH_KEYWORDS, "Sum of elements."},
    {"prod", (PyCFunction)arr_prod, METH_VARARGS | METH_KEYWORDS, "Product of elements."},
    {"max", (PyCFunction)arr_max, METH_VARARGS | METH_KEYWORDS, "Maximum."},
    {"min", (PyCFunction)arr_min, METH_VARARGS | METH_KEYWORDS, "Minimum."},
    {"any", (PyCFunction)arr_any, METH_VARARGS | METH_KEYWORDS, "True if any element is true."},
    {"all", (PyCFunction)arr_all, METH_VARARGS | METH_KEYWORDS, "True if all elements are true."},
    {"argmax", (PyCFunction)arr_argmax, METH_VARARGS | METH_KEYWORDS, "Index of the maximum."},
    {"argmin", (PyCFunction)arr_argmin, METH_VARARGS | METH_KEYWORDS, "Index of the minimum."},
    {"mean", (PyCFunction)arr_mean, METH_VARARGS | METH_KEYWORDS, "Arithmetic mean."},
    {"std", (PyCFunction)arr_std, METH_VARARGS | METH_KEYWORDS, "Standard deviation."},
    {"var", (PyCFunction)arr_var, METH_VARARGS | METH_KEYWORDS, "Variance."},
    {"cumsum", (PyCFunction)arr_cumsum, METH_VARARGS | METH_KEYWORDS, "Cumulative sum."},
    {"cumprod", (PyCFunction)arr_cumprod, METH_VARARGS | METH_KEYWORDS, "Cumulative product."},
    {"sort", (PyCFunction)arr_sort, METH_VARARGS | METH_KEYWORDS, "Sorted copy."},
    {"argsort", (PyCFunction)arr_argsort, METH_VARARGS | METH_KEYWORDS, "Indices that would sort."},
    {"repeat", (PyCFunction)arr_repeat, METH_VARARGS | METH_KEYWORDS, "Repeat elements."},
    {"take", (PyCFunction)arr_take, METH_VARARGS | METH_KEYWORDS, "Elements at the given indices."},
    {"nonzero", (PyCFunction)arr_nonzero, METH_VARARGS | METH_KEYWORDS, "Indices of nonzero elements."},
    {"searchsorted", (PyCFunction)arr_searchsorted, METH_VARARGS | METH_KEYWORDS, "Insertion points."},
    {NULL}
};

PyMethodDef qnp_ufunc_methods[] = {
#define BIN(NAME) {#NAME, (PyCFunction)py_##NAME, METH_VARARGS | METH_KEYWORDS, #NAME " element-wise."}
    BIN(add), BIN(subtract), BIN(multiply), BIN(divide), BIN(floor_divide),
    BIN(remainder), BIN(power), BIN(maximum), BIN(minimum), BIN(hypot),
    BIN(arctan2), BIN(copysign), BIN(equal), BIN(not_equal), BIN(less),
    BIN(less_equal), BIN(greater), BIN(greater_equal), BIN(logical_and),
    BIN(logical_or), BIN(bitwise_and), BIN(bitwise_or), BIN(bitwise_xor),
    BIN(left_shift), BIN(right_shift),
    BIN(negative), BIN(absolute), BIN(sqrt), BIN(exp), BIN(log), BIN(log2),
    BIN(log10), BIN(log1p), BIN(expm1), BIN(sin), BIN(cos), BIN(tan),
    BIN(arcsin), BIN(arccos), BIN(arctan), BIN(sinh), BIN(cosh), BIN(tanh),
    BIN(arcsinh), BIN(arccosh), BIN(arctanh), BIN(sign), BIN(floor), BIN(ceil),
    BIN(trunc), BIN(rint), BIN(square), BIN(reciprocal), BIN(conjugate), BIN(real),
    BIN(imag), BIN(angle), BIN(isfinite), BIN(isnan), BIN(isinf),
    BIN(logical_not), BIN(invert), BIN(signbit), BIN(round),
#undef BIN
    {"matmul", py_matmul, METH_VARARGS, "Matrix product."},
    {NULL}
};

/* ---- module ------------------------------------------------------------ */

static int add_table(PyObject *module, PyMethodDef *table) {
    for (PyMethodDef *def = table; def->ml_name != NULL; def++) {
        PyObject *fn = PyCFunction_New(def, NULL);
        if (fn == NULL) return -1;
        if (PyModule_AddObject(module, def->ml_name, fn) < 0) {
            Py_DECREF(fn);
            return -1;
        }
    }
    return 0;
}

static PyObject *make_namespace(const char *name, PyMethodDef *table) {
    PyObject *module = PyImport_AddModule(name);
    if (module == NULL) return NULL;
    Py_INCREF(module);
    if (add_table(module, table) < 0) { Py_DECREF(module); return NULL; }
    return module;
}

static struct PyModuleDef qnp_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "quadrivium._qnp",
    .m_doc = "Compiled array core: strided N-dimensional arrays over four dtypes, with\n"
             "broadcasting element-wise operations, indexing, reductions, dense linear\n"
             "algebra, transforms and a PCG64 generator.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__qnp(void) {
    PyObject *module = PyModule_Create(&qnp_module);
    if (module == NULL) return NULL;
    if (qnp_init_types(module) < 0) goto fail;
    if (add_table(module, qnp_ufunc_methods) < 0) goto fail;
    if (add_table(module, qnp_shape_methods) < 0) goto fail;
    if (add_table(module, qnp_reduce_methods) < 0) goto fail;
    if (add_table(module, qnp_sort_methods) < 0) goto fail;
    if (add_table(module, qnp_index_methods) < 0) goto fail;
    if (add_table(module, qnp_array_core_methods) < 0) goto fail;
    if (add_table(module, qnp_random_methods) < 0) goto fail;
    if (qnp_add_random(module) < 0) goto fail;
    PyObject *linalg = make_namespace("quadrivium._qnp.linalg", qnp_linalg_methods);
    if (linalg == NULL) goto fail;
    if (PyObject_SetAttrString(linalg, "LinAlgError", QNP_LinAlgError) < 0) goto fail;
    if (PyModule_AddObject(module, "linalg", linalg) < 0) { Py_DECREF(linalg); goto fail; }
    PyObject *fft = make_namespace("quadrivium._qnp.fft", qnp_fft_methods);
    if (fft == NULL) goto fail;
    if (PyModule_AddObject(module, "fft", fft) < 0) { Py_DECREF(fft); goto fail; }
    if (PyModule_AddStringConstant(module, "__version__", "1.2.0") < 0) goto fail;
    return module;
fail:
    Py_DECREF(module);
    return NULL;
}
