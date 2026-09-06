/* Broadcasting iteration.
 *
 * Operands are padded to a common rank, dimensions that are laid out
 * consecutively for *every* operand are fused, and the innermost dimension is
 * handed to the caller as a run of `inner_len` elements.  For the common case
 * -- same shape, all C-contiguous -- fusion collapses the whole array into a
 * single run, so an element-wise kernel sees one flat loop.
 */
#include "qnp.h"

PyObject *qnp_err_broadcast(const qintp *a, int and_, const qintp *b, int bnd) {
    PyObject *sa = PyUnicode_FromString("(");
    PyObject *sb = PyUnicode_FromString("(");
    for (int i = 0; i < and_; i++) {
        PyObject *piece = PyUnicode_FromFormat(i ? ",%zd" : "%zd", a[i]);
        PyObject *joined = PyUnicode_Concat(sa, piece);
        Py_XDECREF(sa); Py_XDECREF(piece);
        sa = joined;
    }
    for (int i = 0; i < bnd; i++) {
        PyObject *piece = PyUnicode_FromFormat(i ? ",%zd" : "%zd", b[i]);
        PyObject *joined = PyUnicode_Concat(sb, piece);
        Py_XDECREF(sb); Py_XDECREF(piece);
        sb = joined;
    }
    PyErr_Format(PyExc_ValueError,
                 "operands could not be broadcast together with shapes %U) %U) ",
                 sa, sb);
    Py_XDECREF(sa); Py_XDECREF(sb);
    return NULL;
}

int qnp_broadcast_shapes(int n, QArray **arrays, qintp *out_shape, int *out_nd) {
    int nd = 0;
    for (int i = 0; i < n; i++)
        if (arrays[i]->nd > nd) nd = arrays[i]->nd;
    for (int i = 0; i < nd; i++) out_shape[i] = 1;
    for (int k = 0; k < n; k++) {
        QArray *a = arrays[k];
        int offset = nd - a->nd;
        for (int i = 0; i < a->nd; i++) {
            qintp dim = a->shape[i];
            qintp cur = out_shape[offset + i];
            if (dim == cur || dim == 1) continue;
            if (cur == 1) { out_shape[offset + i] = dim; continue; }
            for (int j = 0; j < n; j++) {
                if (j == k) continue;
                qnp_err_broadcast(arrays[j]->shape, arrays[j]->nd, a->shape, a->nd);
                return -1;
            }
            PyErr_SetString(PyExc_ValueError, "operands could not be broadcast together");
            return -1;
        }
    }
    *out_nd = nd;
    return 0;
}

int qnp_iter_init(QIter *it, int nop, QArray **ops, const qintp *shape, int nd) {
    if (nop > QNP_MAXOPS) {
        PyErr_SetString(PyExc_RuntimeError, "too many operands for the iterator");
        return -1;
    }
    it->nop = nop;
    it->nd = nd;
    for (int i = 0; i < nd; i++) it->shape[i] = shape[i];
    for (int k = 0; k < nop; k++) {
        QArray *a = ops[k];
        int offset = nd - a->nd;
        it->base[k] = a->data;
        for (int i = 0; i < nd; i++) {
            if (i < offset) { it->strides[k][i] = 0; continue; }
            qintp dim = a->shape[i - offset];
            if (dim == 1 && shape[i] != 1) it->strides[k][i] = 0;
            else if (dim != shape[i] && shape[i] != 0) {
                qnp_err_broadcast(shape, nd, a->shape, a->nd);
                return -1;
            } else it->strides[k][i] = a->strides[i - offset];
        }
    }
    /* Drop unit dimensions: they contribute nothing but loop overhead. */
    int w = 0;
    for (int i = 0; i < nd; i++) {
        if (it->shape[i] == 1) continue;
        it->shape[w] = it->shape[i];
        for (int k = 0; k < nop; k++) it->strides[k][w] = it->strides[k][i];
        w++;
    }
    nd = w;
    /* Fuse neighbours that are contiguous with one another for all operands. */
    for (int i = nd - 1; i > 0; i--) {
        int fusable = 1;
        for (int k = 0; k < nop; k++) {
            if (it->strides[k][i - 1] != it->strides[k][i] * it->shape[i]) { fusable = 0; break; }
        }
        if (!fusable) continue;
        it->shape[i - 1] *= it->shape[i];
        for (int k = 0; k < nop; k++) it->strides[k][i - 1] = it->strides[k][i];
        for (int j = i; j < nd - 1; j++) {
            it->shape[j] = it->shape[j + 1];
            for (int k = 0; k < nop; k++) it->strides[k][j] = it->strides[k][j + 1];
        }
        nd--;
    }
    it->nd = nd;
    if (nd == 0) {
        it->inner_len = 1;
        for (int k = 0; k < nop; k++) it->inner_stride[k] = 0;
        it->total = 1;
    } else {
        it->inner_len = it->shape[nd - 1];
        for (int k = 0; k < nop; k++) it->inner_stride[k] = it->strides[k][nd - 1];
        it->total = 1;
        for (int i = 0; i < nd - 1; i++) it->total *= it->shape[i];
        if (it->inner_len == 0) it->total = 0;
    }
    for (int i = 0; i < QNP_MAXDIMS; i++) it->index[i] = 0;
    it->done = 0;
    for (int k = 0; k < nop; k++) it->ptr[k] = it->base[k];
    return 0;
}

int qnp_iter_next(QIter *it) {
    if (it->done >= it->total) return 0;
    if (it->done > 0) {
        /* Odometer over the outer dimensions, adjusting pointers in place. */
        int i = it->nd - 2;
        for (; i >= 0; i--) {
            it->index[i]++;
            for (int k = 0; k < it->nop; k++) it->ptr[k] += it->strides[k][i];
            if (it->index[i] < it->shape[i]) break;
            for (int k = 0; k < it->nop; k++)
                it->ptr[k] -= it->strides[k][i] * it->shape[i];
            it->index[i] = 0;
        }
    }
    it->done++;
    return 1;
}
