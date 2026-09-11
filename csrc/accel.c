#include "qaccel.h"

void *qaccel_alloc(qintp count, size_t size) {
    if (count < 0 || size == 0 || (size_t)count > (size_t)PY_SSIZE_T_MAX / size) {
        PyErr_NoMemory();
        return NULL;
    }
    void *p = PyMem_Malloc(count ? (size_t)count * size : 1);
    if (p == NULL) PyErr_NoMemory();
    return p;
}

static QArray *as_array(PyObject *obj, int nd) {
    QArray *a = qnp_from_any(obj, QNP_FLOAT64, 1);
    if (a == NULL) return NULL;
    if (a->nd != nd) {
        PyErr_Format(PyExc_ValueError, "expected a %d-dimensional array, got %d", nd, a->nd);
        Py_DECREF(a);
        return NULL;
    }
    return a;
}

QArray *qaccel_input(PyObject *obj, int nd) {
    QArray *a = as_array(obj, nd);
    if (a == NULL) return NULL;
    QArray *out = qnp_ascontiguous(a);
    Py_DECREF(a);
    return out;
}

QArray *qaccel_copy(PyObject *obj, int nd) {
    QArray *a = as_array(obj, nd);
    if (a == NULL) return NULL;
    /* Conversion from a sequence or another dtype may already have created
     * private storage. Reuse it rather than copying the same input twice. */
    if (Py_REFCNT(a) == 1 && a->base == NULL &&
        (a->flags & (QNP_OWNDATA | QNP_C_CONTIGUOUS | QNP_WRITEABLE)) ==
            (QNP_OWNDATA | QNP_C_CONTIGUOUS | QNP_WRITEABLE)) return a;
    QArray *out = qnp_astype(a, QNP_FLOAT64, 1);
    Py_DECREF(a);
    return out;
}

QArray *qaccel_mutable(PyObject *obj, int nd) {
    if (!QArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "in-place kernel requires a quadrivium array");
        return NULL;
    }
    QArray *a = (QArray *)obj;
    if (a->nd != nd || a->dtype != QNP_FLOAT64) {
        PyErr_Format(PyExc_ValueError, "in-place kernel requires a %d-dimensional float64 array", nd);
        return NULL;
    }
    if (!qnp_is_c_contiguous(a)) {
        PyErr_SetString(PyExc_ValueError, "in-place array must be C-contiguous");
        return NULL;
    }
    if (!(a->flags & QNP_WRITEABLE)) {
        PyErr_SetString(PyExc_ValueError, "in-place array must be writable");
        return NULL;
    }
    Py_INCREF(a);
    return a;
}

int qaccel_square(QArray *a) {
    if (a->nd != 2 || a->shape[0] != a->shape[1]) {
        PyErr_SetString(PyExc_ValueError, "matrix must be square");
        return -1;
    }
    return 0;
}

int qaccel_add(PyObject *module) {
    PyObject *accel = PyModule_New("quadrivium._qnp._accel");
    if (accel == NULL) return -1;
    if (PyModule_AddFunctions(accel, qaccel_direct_methods) < 0 ||
        PyModule_AddFunctions(accel, qaccel_grid_methods) < 0 ||
        PyModule_AddFunctions(accel, qaccel_ode_methods) < 0 ||
        PyModule_AddFunctions(accel, qaccel_transform_methods) < 0 ||
        PyModule_AddStringConstant(accel, "__version__", "1.2.0") < 0 ||
        PyModule_AddObject(module, "_accel", accel) < 0) {
        Py_DECREF(accel);
        return -1;
    }
    return 0;
}
