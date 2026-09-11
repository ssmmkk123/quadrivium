/* Indexing: basic slicing (which returns views), advanced indexing with
 * integer and boolean arrays, and the assignment forms of both.
 *
 * The grouping rule follows NumPy: a plain integer joins the advanced group
 * when advanced indices are present, the group's dimensions land where they
 * sat in the source when the group is contiguous, and at the front otherwise.
 */
#include "qnp.h"

typedef unsigned char qbool;

enum { IDX_INT, IDX_SLICE, IDX_NEWAXIS, IDX_ARRAY };

typedef struct {
    int kind;
    qintp start, step, len;   /* IDX_SLICE */
    qintp value;              /* IDX_INT */
    QArray *array;            /* IDX_ARRAY, always int64; owned */
} IdxEntry;

typedef struct {
    IdxEntry entries[2 * QNP_MAXDIMS];
    int n;
    int consumed;             /* source dimensions used up */
    int has_advanced;
} IdxSpec;

static void spec_clear(IdxSpec *spec) {
    for (int i = 0; i < spec->n; i++)
        if (spec->entries[i].kind == IDX_ARRAY) Py_CLEAR(spec->entries[i].array);
    spec->n = 0;
}

/* ---- nonzero ---------------------------------------------------------- */

/* Indices of the true/nonzero elements, one int64 array per dimension. */
static int nonzero_arrays(QArray *a, QArray **out, qintp *count) {
    QArray *c = qnp_ascontiguous(a);
    if (c == NULL) return -1;
    qintp n = qnp_size(c);
    int isz = QNP_ITEMSIZE(c->dtype);
    qintp hits = 0;
    const char *p = c->data;
    for (qintp i = 0; i < n; i++, p += isz) {
        int nz;
        switch (c->dtype) {
            case QNP_BOOL: nz = *(const qbool *)p != 0; break;
            case QNP_INT64: nz = *(const int64_t *)p != 0; break;
            case QNP_FLOAT32: case QNP_COMPLEX64: { qcomplex z=qnp_read_number(p,a->dtype); nz=z.re != 0 || z.im != 0; break; }
            case QNP_FLOAT64: nz = *(const double *)p != 0.0; break;
            default: {
                const qcomplex *z = (const qcomplex *)p;
                nz = (z->re != 0.0 || z->im != 0.0);
                break;
            }
        }
        hits += nz;
    }
    int nd = c->nd ? c->nd : 1;
    for (int d = 0; d < nd; d++) {
        out[d] = qnp_new(1, &hits, QNP_INT64);
        if (out[d] == NULL) {
            for (int k = 0; k < d; k++) Py_CLEAR(out[k]);
            Py_DECREF(c);
            return -1;
        }
    }
    qintp idx[QNP_MAXDIMS] = {0};
    qintp w = 0;
    p = c->data;
    for (qintp i = 0; i < n; i++, p += isz) {
        int nz;
        switch (c->dtype) {
            case QNP_BOOL: nz = *(const qbool *)p != 0; break;
            case QNP_INT64: nz = *(const int64_t *)p != 0; break;
            case QNP_FLOAT32: case QNP_COMPLEX64: { qcomplex z=qnp_read_number(p,a->dtype); nz=z.re != 0 || z.im != 0; break; }
            case QNP_FLOAT64: nz = *(const double *)p != 0.0; break;
            default: {
                const qcomplex *z = (const qcomplex *)p;
                nz = (z->re != 0.0 || z->im != 0.0);
                break;
            }
        }
        if (nz) {
            for (int d = 0; d < c->nd; d++) ((int64_t *)out[d]->data)[w] = idx[d];
            if (c->nd == 0) ((int64_t *)out[0]->data)[w] = 0;
            w++;
        }
        for (int d = c->nd - 1; d >= 0; d--) {
            if (++idx[d] < c->shape[d]) break;
            idx[d] = 0;
        }
    }
    *count = nd;
    Py_DECREF(c);
    return 0;
}

/* ---- key parsing ------------------------------------------------------ */

static int is_index_array(PyObject *obj) {
    if (QArray_Check(obj)) {
        int dt = ((QArray *)obj)->dtype;
        return dt == QNP_INT64 || dt == QNP_BOOL;
    }
    if (PyList_Check(obj) || PyTuple_Check(obj)) return 1;
    return 0;
}

static int append_entry(IdxSpec *spec, IdxEntry e) {
    if (spec->n >= 2 * QNP_MAXDIMS) {
        PyErr_SetString(PyExc_IndexError, "too many indices for array");
        return -1;
    }
    spec->entries[spec->n++] = e;
    return 0;
}

/* Turns one key element into entries, consuming source dimensions. */
static int parse_one(IdxSpec *spec, QArray *a, PyObject *obj, int *dim) {
    IdxEntry e;
    memset(&e, 0, sizeof(e));
    if (obj == Py_None) {
        e.kind = IDX_NEWAXIS;
        return append_entry(spec, e);
    }
    if (PySlice_Check(obj)) {
        if (*dim >= a->nd) {
            PyErr_SetString(PyExc_IndexError, "too many indices for array");
            return -1;
        }
        Py_ssize_t start, stop, step, len;
        if (PySlice_GetIndicesEx(obj, a->shape[*dim], &start, &stop, &step, &len) < 0)
            return -1;
        e.kind = IDX_SLICE;
        e.start = start;
        e.step = step;
        e.len = len;
        (*dim)++;
        return append_entry(spec, e);
    }
    if (PyBool_Check(obj)) {
        /* A bare True/False acts as a length-1 or length-0 new axis. */
        qintp len = (obj == Py_True) ? 1 : 0;
        QArray *idx = qnp_new(1, &len, QNP_INT64);
        if (idx == NULL) return -1;
        if (len) ((int64_t *)idx->data)[0] = 0;
        e.kind = IDX_ARRAY;
        e.array = idx;
        spec->has_advanced = 1;
        e.len = -1;                 /* consumes no source dimension */
        return append_entry(spec, e);
    }
    if (PyIndex_Check(obj) && !QArray_Check(obj)) {
        Py_ssize_t v = PyNumber_AsSsize_t(obj, PyExc_IndexError);
        if (v == -1 && PyErr_Occurred()) return -1;
        if (*dim >= a->nd) {
            PyErr_SetString(PyExc_IndexError, "too many indices for array");
            return -1;
        }
        qintp dimlen = a->shape[*dim];
        if (v < 0) v += dimlen;
        if (v < 0 || v >= dimlen) {
            PyErr_Format(PyExc_IndexError,
                         "index %zd is out of bounds for axis %d with size %zd",
                         (Py_ssize_t)(v < 0 ? v - dimlen : v), *dim, dimlen);
            return -1;
        }
        e.kind = IDX_INT;
        e.value = v;
        (*dim)++;
        return append_entry(spec, e);
    }
    if (is_index_array(obj) || QArray_Check(obj)) {
        QArray *idx = qnp_from_any(obj, -1, 0);
        if (idx == NULL) return -1;
        if (idx->dtype == QNP_BOOL) {
            /* A boolean mask covers as many dimensions as it has, and turns
             * into that many integer arrays. */
            int nd = idx->nd;
            if (*dim + nd > a->nd) {
                Py_DECREF(idx);
                PyErr_SetString(PyExc_IndexError,
                                "boolean index did not match indexed array");
                return -1;
            }
            for (int i = 0; i < nd; i++) {
                if (idx->shape[i] != a->shape[*dim + i]) {
                    PyErr_Format(PyExc_IndexError,
                                 "boolean index did not match indexed array along "
                                 "axis %d; size of axis is %zd but size of "
                                 "corresponding boolean axis is %zd",
                                 *dim + i, a->shape[*dim + i], idx->shape[i]);
                    Py_DECREF(idx);
                    return -1;
                }
            }
            QArray *parts[QNP_MAXDIMS];
            qintp nparts = 0;
            if (nonzero_arrays(idx, parts, &nparts) < 0) { Py_DECREF(idx); return -1; }
            Py_DECREF(idx);
            for (qintp i = 0; i < nparts; i++) {
                e.kind = IDX_ARRAY;
                e.array = parts[i];
                spec->has_advanced = 1;
                if (append_entry(spec, e) < 0) {
                    for (qintp k = i; k < nparts; k++) Py_CLEAR(parts[k]);
                    return -1;
                }
                (*dim)++;
            }
            return 0;
        }
        if (idx->dtype != QNP_INT64) {
            /* Casting here would silently truncate 0.5 to 0.  An empty list
             * carries no values to type, so it stays acceptable. */
            if (!((PyList_Check(obj) || PyTuple_Check(obj)) && qnp_size(idx) == 0)) {
                Py_DECREF(idx);
                PyErr_SetString(PyExc_IndexError,
                                "arrays used as indices must be of integer "
                                "(or boolean) type");
                return -1;
            }
            QArray *cast = qnp_astype(idx, QNP_INT64, 0);
            Py_DECREF(idx);
            if (cast == NULL) return -1;
            idx = cast;
        }
        if (*dim >= a->nd) {
            Py_DECREF(idx);
            PyErr_SetString(PyExc_IndexError, "too many indices for array");
            return -1;
        }
        e.kind = IDX_ARRAY;
        e.array = idx;
        spec->has_advanced = 1;
        (*dim)++;
        return append_entry(spec, e);
    }
    PyErr_Format(PyExc_IndexError,
                 "only integers, slices, ellipsis, None and integer or boolean "
                 "arrays are valid indices (got %s)", Py_TYPE(obj)->tp_name);
    return -1;
}

static int parse_key(IdxSpec *spec, QArray *a, PyObject *key) {
    spec->n = 0;
    spec->has_advanced = 0;
    PyObject *tuple = NULL;
    PyObject **items;
    Py_ssize_t nitems;
    if (PyTuple_Check(key)) {
        items = &PyTuple_GET_ITEM(key, 0);
        nitems = PyTuple_GET_SIZE(key);
    } else {
        tuple = PyTuple_Pack(1, key);
        if (tuple == NULL) return -1;
        items = &PyTuple_GET_ITEM(tuple, 0);
        nitems = 1;
    }
    /* Work out how many dimensions an ellipsis stands for. */
    int explicit_dims = 0, ellipsis_at = -1;
    for (Py_ssize_t i = 0; i < nitems; i++) {
        PyObject *o = items[i];
        if (o == Py_Ellipsis) {
            if (ellipsis_at >= 0) {
                Py_XDECREF(tuple);
                PyErr_SetString(PyExc_IndexError,
                                "an index can only have a single ellipsis ('...')");
                return -1;
            }
            ellipsis_at = (int)i;
        } else if (o == Py_None || PyBool_Check(o)) {
            /* consumes nothing */
        } else if (QArray_Check(o) && ((QArray *)o)->dtype == QNP_BOOL) {
            explicit_dims += ((QArray *)o)->nd;
        } else {
            explicit_dims += 1;
        }
    }
    int dim = 0;
    int rc = 0;
    for (Py_ssize_t i = 0; i < nitems && rc == 0; i++) {
        PyObject *o = items[i];
        if (o == Py_Ellipsis) {
            int fill = a->nd - explicit_dims;
            for (int k = 0; k < fill; k++) {
                IdxEntry e;
                memset(&e, 0, sizeof(e));
                e.kind = IDX_SLICE;
                e.start = 0;
                e.step = 1;
                e.len = a->shape[dim];
                dim++;
                if ((rc = append_entry(spec, e)) < 0) break;
            }
            continue;
        }
        rc = parse_one(spec, a, o, &dim);
    }
    Py_XDECREF(tuple);
    if (rc < 0) { spec_clear(spec); return -1; }
    /* Trailing dimensions the key never mentioned stay whole. */
    while (dim < a->nd) {
        IdxEntry e;
        memset(&e, 0, sizeof(e));
        e.kind = IDX_SLICE;
        e.start = 0;
        e.step = 1;
        e.len = a->shape[dim];
        dim++;
        if (append_entry(spec, e) < 0) { spec_clear(spec); return -1; }
    }
    spec->consumed = dim;
    return 0;
}

/* ---- basic indexing: build a view ------------------------------------- */

/* The key may name more result dimensions than an array header can hold; the
 * limit must be enforced before anything is written to a QNP_MAXDIMS buffer. */
static int too_many_dims(int nd) {
    if (nd >= QNP_MAXDIMS) {
        PyErr_Format(PyExc_IndexError,
                     "number of dimensions produced by the index exceeds the "
                     "maximum supported dimension of %d", QNP_MAXDIMS);
        return 1;
    }
    return 0;
}

static PyObject *basic_view(QArray *a, IdxSpec *spec) {
    qintp shape[QNP_MAXDIMS] = {0}, strides[QNP_MAXDIMS] = {0};
    int nd = 0, dim = 0;
    qintp offset = 0;
    for (int i = 0; i < spec->n; i++) {
        IdxEntry *e = &spec->entries[i];
        if (e->kind == IDX_NEWAXIS) {
            if (too_many_dims(nd)) return NULL;
            shape[nd] = 1;
            strides[nd] = 0;
            nd++;
            continue;
        }
        if (e->kind == IDX_INT) {
            offset += e->value * a->strides[dim];
            dim++;
            continue;
        }
        if (too_many_dims(nd)) return NULL;
        offset += e->start * a->strides[dim];
        shape[nd] = e->len;
        strides[nd] = e->step * a->strides[dim];
        nd++;
        dim++;
    }
    QArray *view = qnp_new_view(a, a->data + offset, nd, shape, strides, a->dtype);
    if (view == NULL) return NULL;
    if (nd == 0) {
        PyObject *v = qnp_getitem_ptr(view->dtype, view->data);
        Py_DECREF(view);
        return v;
    }
    return (PyObject *)view;
}

/* ---- advanced indexing ------------------------------------------------ */

typedef struct {
    /* Description of the gather/scatter derived from a parsed key. */
    qintp bshape[QNP_MAXDIMS];
    int bnd;
    int nadv;
    QArray *adv[QNP_MAXDIMS];     /* borrowed, broadcast to bshape */
    qintp adv_stride[QNP_MAXDIMS];/* source stride of the indexed dimension */
    qintp adv_dimlen[QNP_MAXDIMS];
    int adv_axis[QNP_MAXDIMS];
    qintp basic_shape[QNP_MAXDIMS];
    qintp basic_src_stride[QNP_MAXDIMS];
    int basic_nd;
    qintp base_offset;
    qintp out_shape[QNP_MAXDIMS];
    int out_nd;
    int adv_out_pos;              /* where the advanced block sits in the output */
} AdvPlan;

static void plan_release(AdvPlan *p);

static int build_plan(QArray *a, IdxSpec *spec, AdvPlan *p) {
    p->nadv = 0;
    p->basic_nd = 0;
    p->base_offset = 0;
    int dim = 0;
    int first_adv = -1, last_adv = -1, seen_basic_between = 0;
    int adv_result_pos = 0;
    int basic_before = 0;
    QArray *adv_raw[QNP_MAXDIMS];
    for (int i = 0; i < spec->n; i++) {
        IdxEntry *e = &spec->entries[i];
        if (e->kind == IDX_ARRAY) {
            if (first_adv < 0) { first_adv = i; adv_result_pos = p->basic_nd; }
            else if (last_adv >= 0 && i > last_adv + 1) seen_basic_between = 1;
            last_adv = i;
            adv_raw[p->nadv] = e->array;
            if (e->len == -1) {           /* bare True/False */
                p->adv_stride[p->nadv] = 0;
                p->adv_dimlen[p->nadv] = 1;
                p->adv_axis[p->nadv] = -1;
            } else {
                p->adv_stride[p->nadv] = a->strides[dim];
                p->adv_dimlen[p->nadv] = a->shape[dim];
                p->adv_axis[p->nadv] = dim;
                dim++;
            }
            p->nadv++;
        } else if (e->kind == IDX_INT) {
            /* Mixed with advanced indices an integer is a 0-d index array. */
            if (first_adv < 0) { first_adv = i; adv_result_pos = p->basic_nd; }
            else if (last_adv >= 0 && i > last_adv + 1) seen_basic_between = 1;
            last_adv = i;
            QArray *scalar = qnp_new(0, NULL, QNP_INT64);
            if (scalar == NULL) return -1;
            *(int64_t *)scalar->data = e->value;
            e->array = scalar;        /* the spec owns it and frees it later */
            e->kind = IDX_ARRAY;
            adv_raw[p->nadv] = scalar;
            p->adv_stride[p->nadv] = a->strides[dim];
            p->adv_dimlen[p->nadv] = a->shape[dim];
            p->adv_axis[p->nadv] = dim;
            dim++;
            p->nadv++;
        } else if (e->kind == IDX_NEWAXIS) {
            if (first_adv >= 0 && last_adv >= 0 && i > last_adv) { /* after group */ }
            if (too_many_dims(p->basic_nd)) return -1;
            p->basic_shape[p->basic_nd] = 1;
            p->basic_src_stride[p->basic_nd] = 0;
            p->basic_nd++;
            if (first_adv < 0) basic_before++;
        } else {
            if (too_many_dims(p->basic_nd)) return -1;
            p->base_offset += e->start * a->strides[dim];
            p->basic_shape[p->basic_nd] = e->len;
            p->basic_src_stride[p->basic_nd] = e->step * a->strides[dim];
            p->basic_nd++;
            dim++;
            if (first_adv < 0) basic_before++;
        }
    }
    /* Detect a basic index sitting between two advanced ones. */
    if (first_adv >= 0) {
        for (int i = first_adv; i <= last_adv; i++)
            if (spec->entries[i].kind != IDX_ARRAY) seen_basic_between = 1;
    }
    if (qnp_broadcast_shapes(p->nadv, adv_raw, p->bshape, &p->bnd) < 0) return -1;
    for (int k = 0; k < p->nadv; k++) {
        PyObject *b = qnp_broadcast_to((PyObject *)adv_raw[k], p->bshape, p->bnd);
        if (b == NULL) return -1;
        p->adv[k] = (QArray *)b;
    }
    p->adv_out_pos = seen_basic_between ? 0 : adv_result_pos;
    p->out_nd = p->basic_nd + p->bnd;
    if (p->out_nd > QNP_MAXDIMS) {
        PyErr_Format(PyExc_IndexError,
                     "number of dimensions produced by the index exceeds the "
                     "maximum supported dimension of %d", QNP_MAXDIMS);
        plan_release(p);
        return -1;
    }
    int w = 0;
    for (int i = 0; i < p->adv_out_pos; i++) p->out_shape[w++] = p->basic_shape[i];
    for (int i = 0; i < p->bnd; i++) p->out_shape[w++] = p->bshape[i];
    for (int i = p->adv_out_pos; i < p->basic_nd; i++) p->out_shape[w++] = p->basic_shape[i];
    return 0;
}

static void plan_release(AdvPlan *p) {
    for (int k = 0; k < p->nadv; k++) Py_CLEAR(p->adv[k]);
}

/* Byte offsets into the source for every position of the advanced block. */
static qintp *gather_offsets(AdvPlan *p, qintp *count) {
    qintp b = 1;
    for (int i = 0; i < p->bnd; i++) b *= p->bshape[i];
    *count = b;
    qintp *offsets = (qintp *)PyMem_Malloc((size_t)(b ? b : 1) * sizeof(qintp));
    if (offsets == NULL) { PyErr_NoMemory(); return NULL; }
    for (qintp i = 0; i < b; i++) offsets[i] = p->base_offset;
    for (int k = 0; k < p->nadv; k++) {
        QArray *idx = p->adv[k];
        QArray *ops[1] = {idx};
        QIter it;
        if (qnp_iter_init(&it, 1, ops, p->bshape, p->bnd) < 0) {
            PyMem_Free(offsets);
            return NULL;
        }
        qintp w = 0;
        qintp dimlen = p->adv_dimlen[k];
        qintp stride = p->adv_stride[k];
        while (qnp_iter_next(&it)) {
            const char *ip = it.ptr[0];
            if (it.inner_stride[0] == (qintp)sizeof(int64_t) && k == 0 &&
                p->base_offset == 0) {
                /* The common case: one contiguous index array over a whole
                 * axis, where the offset is a single multiply per element. */
                const int64_t *src = (const int64_t *)ip;
                qintp n = it.inner_len;
                qintp bad = -1;
                for (qintp i = 0; i < n; i++) {
                    int64_t v = src[i];
                    if (v < 0) v += dimlen;
                    if ((qintp)v >= dimlen || v < 0) { bad = i; break; }
                    offsets[w + i] = v * stride;
                }
                if (bad >= 0) {
                    PyErr_Format(PyExc_IndexError,
                                 "index %lld is out of bounds for axis %d with size %zd",
                                 (long long)src[bad], p->adv_axis[k], dimlen);
                    PyMem_Free(offsets);
                    return NULL;
                }
                w += n;
                continue;
            }
            for (qintp i = 0; i < it.inner_len; i++, ip += it.inner_stride[0]) {
                int64_t v = *(const int64_t *)ip;
                if (v < 0) v += dimlen;
                if (v < 0 || v >= dimlen) {
                    PyErr_Format(PyExc_IndexError,
                                 "index %lld is out of bounds for axis %d with size %zd",
                                 (long long)(*(const int64_t *)ip), p->adv_axis[k], dimlen);
                    PyMem_Free(offsets);
                    return NULL;
                }
                offsets[w++] += v * stride;
            }
        }
    }
    return offsets;
}

/* A stack-resident array header describing a block; never escapes. */
static void make_block(QArray *tmp, qintp *shape, qintp *strides, int nd,
                       char *data, int dtype) {
    memset(tmp, 0, sizeof(*tmp));
    tmp->data = data;
    tmp->nd = nd;
    tmp->shape = shape;
    tmp->strides = strides;
    tmp->dtype = dtype;
    tmp->flags = QNP_WRITEABLE;
}

static PyObject *advanced_getitem(QArray *a, IdxSpec *spec) {
    AdvPlan p;
    if (build_plan(a, spec, &p) < 0) return NULL;
    QArray *out = qnp_new(p.out_nd, p.out_shape, a->dtype);
    if (out == NULL) { plan_release(&p); return NULL; }
    qintp nblocks;
    qintp *offsets = gather_offsets(&p, &nblocks);
    if (offsets == NULL) { plan_release(&p); Py_DECREF(out); return NULL; }

    int isz = QNP_ITEMSIZE(a->dtype);
    /* Strides in the output for the advanced block and for the basic dims. */
    qintp adv_out_stride[QNP_MAXDIMS], basic_out_stride[QNP_MAXDIMS];
    int w = 0;
    for (int i = 0; i < p.adv_out_pos; i++) basic_out_stride[i] = out->strides[w++];
    for (int i = 0; i < p.bnd; i++) adv_out_stride[i] = out->strides[w++];
    for (int i = p.adv_out_pos; i < p.basic_nd; i++) basic_out_stride[i] = out->strides[w++];
    /* Flattened stride through the advanced block, in output bytes. */
    qintp block_stride[QNP_MAXDIMS];
    qintp acc = 1;
    for (int i = p.bnd - 1; i >= 0; i--) { block_stride[i] = acc; acc *= p.bshape[i]; }

    if (p.basic_nd == 0) {
        /* With no basic dimensions the result is exactly the advanced block,
         * laid out contiguously, so the destination just walks forward. This
         * is the plain gather -- `v[idx]` -- and it is worth the special
         * cases: it carries sparse matrix-vector products. */
        char *dst = out->data;
        if (isz == 8) {
            double *o = (double *)dst;
            const char *base = a->data;
            for (qintp i = 0; i < nblocks; i++)
                o[i] = *(const double *)(base + offsets[i]);
        } else {
            const char *base = a->data;
            for (qintp i = 0; i < nblocks; i++, dst += isz)
                memcpy(dst, base + offsets[i], (size_t)isz);
        }
    } else {
        QArray src_block, dst_block;
        for (qintp i = 0; i < nblocks; i++) {
            qintp dst = 0, rem = i;
            for (int d = 0; d < p.bnd; d++) {
                qintp c = rem / block_stride[d];
                rem -= c * block_stride[d];
                dst += c * adv_out_stride[d];
            }
            make_block(&src_block, p.basic_shape, p.basic_src_stride, p.basic_nd,
                       a->data + offsets[i], a->dtype);
            make_block(&dst_block, p.basic_shape, basic_out_stride, p.basic_nd,
                       out->data + dst, out->dtype);
            if (qnp_copy_into(&dst_block, &src_block) < 0) {
                PyMem_Free(offsets);
                plan_release(&p);
                Py_DECREF(out);
                return NULL;
            }
        }
    }
    PyMem_Free(offsets);
    plan_release(&p);
    return qnp_wrap_scalar_or_array(out);
}

static int advanced_setitem(QArray *a, IdxSpec *spec, PyObject *value) {
    AdvPlan p;
    if (build_plan(a, spec, &p) < 0) return -1;
    QArray *src = qnp_from_any(value, a->dtype, 1);
    if (src == NULL) { plan_release(&p); return -1; }
    PyObject *bsrc = qnp_broadcast_to((PyObject *)src, p.out_shape, p.out_nd);
    Py_DECREF(src);
    if (bsrc == NULL) {
        plan_release(&p);
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError,
                        "shape mismatch: value array could not be broadcast to the "
                        "indexing result shape");
        return -1;
    }
    QArray *val = (QArray *)bsrc;
    qintp nblocks;
    qintp *offsets = gather_offsets(&p, &nblocks);
    if (offsets == NULL) { plan_release(&p); Py_DECREF(val); return -1; }

    int isz = QNP_ITEMSIZE(a->dtype);
    qintp adv_val_stride[QNP_MAXDIMS], basic_val_stride[QNP_MAXDIMS];
    int w = 0;
    for (int i = 0; i < p.adv_out_pos; i++) basic_val_stride[i] = val->strides[w++];
    for (int i = 0; i < p.bnd; i++) adv_val_stride[i] = val->strides[w++];
    for (int i = p.adv_out_pos; i < p.basic_nd; i++) basic_val_stride[i] = val->strides[w++];
    qintp block_stride[QNP_MAXDIMS];
    qintp acc = 1;
    for (int i = p.bnd - 1; i >= 0; i--) { block_stride[i] = acc; acc *= p.bshape[i]; }

    int rc = 0;
    if (p.basic_nd == 0) {
        for (qintp i = 0; i < nblocks; i++) {
            qintp off = 0, rem = i;
            for (int d = 0; d < p.bnd; d++) {
                qintp c = rem / block_stride[d];
                rem -= c * block_stride[d];
                off += c * adv_val_stride[d];
            }
            memcpy(a->data + offsets[i], val->data + off, (size_t)isz);
        }
    } else {
        QArray src_block, dst_block;
        for (qintp i = 0; i < nblocks && rc == 0; i++) {
            qintp off = 0, rem = i;
            for (int d = 0; d < p.bnd; d++) {
                qintp c = rem / block_stride[d];
                rem -= c * block_stride[d];
                off += c * adv_val_stride[d];
            }
            make_block(&dst_block, p.basic_shape, p.basic_src_stride, p.basic_nd,
                       a->data + offsets[i], a->dtype);
            make_block(&src_block, p.basic_shape, basic_val_stride, p.basic_nd,
                       val->data + off, val->dtype);
            rc = qnp_copy_into(&dst_block, &src_block);
        }
    }
    PyMem_Free(offsets);
    plan_release(&p);
    Py_DECREF(val);
    return rc;
}

/* ---- fast paths for the two idioms that dominate ----------------------- */

/* `a[mask]` and `a[idx]`, and their assignment forms, are the hot indexing
 * operations in this package -- a sparse matrix-vector product runs both on
 * every call. The general machinery below builds an offset array and a
 * broadcast plan first; these walk the data once instead. */

static int mask_matches(QArray *a, PyObject *key) {
    if (!QArray_Check(key)) return 0;
    QArray *m = (QArray *)key;
    if (m->dtype != QNP_BOOL || m->nd != a->nd) return 0;
    for (int i = 0; i < a->nd; i++)
        if (m->shape[i] != a->shape[i]) return 0;
    return 1;
}

static int int_index_matches(QArray *a, PyObject *key) {
    if (!QArray_Check(key) || a->nd == 0) return 0;
    QArray *i = (QArray *)key;
    return i->dtype == QNP_INT64 && i->nd == 1 && (i->flags & QNP_C_CONTIGUOUS);
}

static PyObject *masked_take(QArray *a, QArray *mask) {
    QArray *ops[2] = {a, mask};
    QIter it;
    qintp count = 0;
    if (qnp_iter_init(&it, 2, ops, a->shape, a->nd) < 0) return NULL;
    while (qnp_iter_next(&it)) {
        const char *m = it.ptr[1];
        for (qintp i = 0; i < it.inner_len; i++, m += it.inner_stride[1])
            count += (*(const qbool *)m != 0);
    }
    QArray *out = qnp_new(1, &count, a->dtype);
    if (out == NULL) return NULL;
    if (qnp_iter_init(&it, 2, ops, a->shape, a->nd) < 0) { Py_DECREF(out); return NULL; }
    int isz = QNP_ITEMSIZE(a->dtype);
    char *dst = out->data;
    while (qnp_iter_next(&it)) {
        const char *src = it.ptr[0];
        const char *m = it.ptr[1];
        /* Eight-byte elements read and written directly; `memcpy` per element
         * costs more than the selection itself at these sizes. */
        if (isz == 8 && it.inner_stride[0] == 8 && it.inner_stride[1] == 1) {
            const int64_t *values = (const int64_t *)src;
            const qbool *flags = (const qbool *)m;
            int64_t *target = (int64_t *)dst;
            qintp taken = 0;
            for (qintp i = 0; i < it.inner_len; i++)
                if (flags[i]) target[taken++] = values[i];
            dst = (char *)(target + taken);
            continue;
        }
        for (qintp i = 0; i < it.inner_len; i++) {
            if (*(const qbool *)m) {
                memcpy(dst, src, (size_t)isz);
                dst += isz;
            }
            src += it.inner_stride[0];
            m += it.inner_stride[1];
        }
    }
    return (PyObject *)out;
}

static int masked_assign(QArray *a, QArray *mask, PyObject *value) {
    QArray *src = qnp_from_any(value, a->dtype, 1);
    if (src == NULL) return -1;
    qintp supplied = qnp_size(src);
    if (src->nd > 1 || (supplied != 1 && src->nd == 0)) {
        Py_DECREF(src);
        return 1;                       /* not a shape this path handles */
    }
    QArray *csrc = qnp_ascontiguous(src);
    Py_DECREF(src);
    if (csrc == NULL) return -1;
    QArray *ops[2] = {a, mask};
    QIter it;
    if (qnp_iter_init(&it, 2, ops, a->shape, a->nd) < 0) { Py_DECREF(csrc); return -1; }
    int isz = QNP_ITEMSIZE(a->dtype);
    const char *values = csrc->data;
    qintp taken = 0;
    int scalar = (supplied == 1 && csrc->nd == 0) || supplied == 1;
    while (qnp_iter_next(&it)) {
        char *dst = it.ptr[0];
        const char *m = it.ptr[1];
        if (isz == 8 && it.inner_stride[0] == 8 && it.inner_stride[1] == 1 &&
            (scalar || taken + it.inner_len <= supplied)) {
            int64_t *target = (int64_t *)dst;
            const qbool *flags = (const qbool *)m;
            const int64_t *source = (const int64_t *)values;
            if (scalar) {
                int64_t only = source[0];
                for (qintp i = 0; i < it.inner_len; i++) if (flags[i]) target[i] = only;
            } else {
                for (qintp i = 0; i < it.inner_len; i++)
                    if (flags[i]) target[i] = source[taken++];
            }
            continue;
        }
        for (qintp i = 0; i < it.inner_len; i++) {
            if (*(const qbool *)m) {
                if (!scalar && taken >= supplied) {
                    Py_DECREF(csrc);
                    PyErr_SetString(PyExc_ValueError,
                                    "NumPy boolean array indexing assignment cannot assign "
                                    "fewer values than the mask selects");
                    return -1;
                }
                memcpy(dst, values + (scalar ? 0 : taken * isz), (size_t)isz);
                taken++;
            }
            dst += it.inner_stride[0];
            m += it.inner_stride[1];
        }
    }
    Py_DECREF(csrc);
    if (!scalar && taken != supplied) {
        PyErr_SetString(PyExc_ValueError,
                        "NumPy boolean array indexing assignment needs one value per "
                        "selected element");
        return -1;
    }
    return 0;
}

static PyObject *integer_take(QArray *a, QArray *idx) {
    qintp n = qnp_size(idx);
    qintp shape[QNP_MAXDIMS];
    shape[0] = n;
    for (int i = 1; i < a->nd; i++) shape[i] = a->shape[i];
    QArray *out = qnp_new(a->nd, shape, a->dtype);
    if (out == NULL) return NULL;
    qintp dimlen = a->shape[0];
    const int64_t *ip = (const int64_t *)idx->data;
    if (a->nd == 1 && a->dtype == QNP_FLOAT64) {
        const double *src = (const double *)a->data;
        qintp step = a->strides[0] / (qintp)sizeof(double);
        double *dst = (double *)out->data;
        for (qintp i = 0; i < n; i++) {
            int64_t v = ip[i];
            if (v < 0) v += dimlen;
            if (v < 0 || v >= dimlen) {
                PyErr_Format(PyExc_IndexError,
                             "index %lld is out of bounds for axis 0 with size %zd",
                             (long long)ip[i], dimlen);
                Py_DECREF(out);
                return NULL;
            }
            dst[i] = src[v * step];
        }
        return (PyObject *)out;
    }
    qintp row = QNP_ITEMSIZE(a->dtype);
    for (int i = 1; i < a->nd; i++) row *= a->shape[i];
    QArray *ca = qnp_ascontiguous(a);
    if (ca == NULL) { Py_DECREF(out); return NULL; }
    char *dst = out->data;
    for (qintp i = 0; i < n; i++, dst += row) {
        int64_t v = ip[i];
        if (v < 0) v += dimlen;
        if (v < 0 || v >= dimlen) {
            PyErr_Format(PyExc_IndexError,
                         "index %lld is out of bounds for axis 0 with size %zd",
                         (long long)ip[i], dimlen);
            Py_DECREF(out); Py_DECREF(ca);
            return NULL;
        }
        memcpy(dst, ca->data + v * row, (size_t)row);
    }
    Py_DECREF(ca);
    return (PyObject *)out;
}

/* ---- entry points ----------------------------------------------------- */

PyObject *qnp_getitem(QArray *self, PyObject *key) {
    if (mask_matches(self, key)) return masked_take(self, (QArray *)key);
    if (int_index_matches(self, key)) return integer_take(self, (QArray *)key);
    IdxSpec spec;
    if (parse_key(&spec, self, key) < 0) return NULL;
    PyObject *result;
    if (spec.has_advanced) result = advanced_getitem(self, &spec);
    else result = basic_view(self, &spec);
    spec_clear(&spec);
    return result;
}

int qnp_setitem(QArray *self, PyObject *key, PyObject *value) {
    if (!(self->flags & QNP_WRITEABLE)) {
        PyErr_SetString(PyExc_ValueError, "assignment destination is read-only");
        return -1;
    }
    if (mask_matches(self, key)) {
        int rc = masked_assign(self, (QArray *)key, value);
        if (rc <= 0) return rc;         /* 1 means "fall through to the general path" */
    }
    IdxSpec spec;
    if (parse_key(&spec, self, key) < 0) return -1;
    int rc;
    if (spec.has_advanced) {
        rc = advanced_setitem(self, &spec, value);
    } else {
        PyObject *view = basic_view(self, &spec);
        if (view == NULL) { spec_clear(&spec); return -1; }
        if (!QArray_Check(view)) {
            /* A fully-indexed element: write it straight through. */
            qintp offset = 0;
            int dim = 0;
            for (int i = 0; i < spec.n; i++) {
                IdxEntry *e = &spec.entries[i];
                if (e->kind == IDX_INT) { offset += e->value * self->strides[dim]; dim++; }
                else if (e->kind == IDX_SLICE) { offset += e->start * self->strides[dim]; dim++; }
            }
            rc = qnp_setitem_ptr(self->dtype, self->data + offset, value);
            if (rc < 0 && PyErr_ExceptionMatches(PyExc_TypeError)) {
                /* A sequence assigned to a scalar slot is a shape error. */
                PyErr_Clear();
                PyErr_SetString(PyExc_ValueError,
                                "setting an array element with a sequence");
            }
        } else {
            QArray *src = qnp_from_any(value, ((QArray *)view)->dtype, 1);
            if (src == NULL) rc = -1;
            else {
                rc = qnp_copy_into((QArray *)view, src);
                Py_DECREF(src);
            }
        }
        Py_DECREF(view);
    }
    spec_clear(&spec);
    return rc;
}

/* ---- helpers exported to the rest of the module ----------------------- */

/* `np.take` along one axis, which is also how fancy row/column selection is
 * implemented by the Python layer. */
PyObject *qnp_take_axis(QArray *a, QArray *idx, int axis) {
    if (axis < 0) axis += a->nd;
    if (axis < 0 || axis >= a->nd) {
        PyErr_SetString(PyExc_ValueError, "axis out of bounds");
        return NULL;
    }
    qintp shape[QNP_MAXDIMS];
    int nd = 0;
    for (int i = 0; i < axis; i++) shape[nd++] = a->shape[i];
    for (int i = 0; i < idx->nd; i++) shape[nd++] = idx->shape[i];
    for (int i = axis + 1; i < a->nd; i++) shape[nd++] = a->shape[i];
    QArray *out = qnp_new(nd, shape, a->dtype);
    if (out == NULL) return NULL;
    QArray *cidx = qnp_ascontiguous(idx);
    if (cidx == NULL) { Py_DECREF(out); return NULL; }
    qintp nidx = qnp_size(cidx);
    qintp outer = 1;
    for (int i = 0; i < axis; i++) outer *= a->shape[i];
    qintp inner = 1;
    for (int i = axis + 1; i < a->nd; i++) inner *= a->shape[i];
    QArray *ca = qnp_ascontiguous(a);
    if (ca == NULL) { Py_DECREF(out); Py_DECREF(cidx); return NULL; }
    int isz = QNP_ITEMSIZE(a->dtype);
    qintp dimlen = a->shape[axis];
    const int64_t *ip = (const int64_t *)cidx->data;
    char *dst = out->data;
    for (qintp o = 0; o < outer; o++) {
        const char *plane = ca->data + o * dimlen * inner * isz;
        for (qintp k = 0; k < nidx; k++) {
            int64_t v = ip[k];
            if (v < 0) v += dimlen;
            if (v < 0 || v >= dimlen) {
                PyErr_Format(PyExc_IndexError,
                             "index %lld is out of bounds for axis %d with size %zd",
                             (long long)ip[k], axis, dimlen);
                Py_DECREF(out); Py_DECREF(cidx); Py_DECREF(ca);
                return NULL;
            }
            memcpy(dst, plane + v * inner * isz, (size_t)(inner * isz));
            dst += inner * isz;
        }
    }
    Py_DECREF(cidx);
    Py_DECREF(ca);
    return (PyObject *)out;
}

static PyObject *py_nonzero(PyObject *self, PyObject *arg) {
    (void)self;
    QArray *a = qnp_from_any(arg, -1, 0);
    if (a == NULL) return NULL;
    QArray *parts[QNP_MAXDIMS];
    qintp n = 0;
    if (nonzero_arrays(a, parts, &n) < 0) { Py_DECREF(a); return NULL; }
    Py_DECREF(a);
    PyObject *t = PyTuple_New(n);
    if (t == NULL) {
        for (qintp i = 0; i < n; i++) Py_CLEAR(parts[i]);
        return NULL;
    }
    for (qintp i = 0; i < n; i++) PyTuple_SET_ITEM(t, i, (PyObject *)parts[i]);
    return t;
}

static PyObject *py_flatnonzero(PyObject *self, PyObject *arg) {
    (void)self;
    QArray *a = qnp_from_any(arg, -1, 0);
    if (a == NULL) return NULL;
    PyObject *flat = qnp_ravel(a);
    Py_DECREF(a);
    if (flat == NULL) return NULL;
    QArray *parts[QNP_MAXDIMS];
    qintp n = 0;
    if (nonzero_arrays((QArray *)flat, parts, &n) < 0) { Py_DECREF(flat); return NULL; }
    Py_DECREF(flat);
    for (qintp i = 1; i < n; i++) Py_CLEAR(parts[i]);
    return (PyObject *)parts[0];
}

static PyObject *py_take(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *io, *axis_o = Py_None;
    static char *kwlist[] = {"a", "indices", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:take", kwlist, &ao, &io, &axis_o))
        return NULL;
    QArray *a = qnp_from_any(ao, -1, 0);
    if (a == NULL) return NULL;
    QArray *idx = qnp_from_any(io, QNP_INT64, 1);
    if (idx == NULL) { Py_DECREF(a); return NULL; }
    PyObject *result;
    if (axis_o == Py_None) {
        PyObject *flat = qnp_ravel(a);
        if (flat == NULL) { Py_DECREF(a); Py_DECREF(idx); return NULL; }
        result = qnp_take_axis((QArray *)flat, idx, 0);
        Py_DECREF(flat);
    } else {
        int axis;
        if (qnp_parse_axis(axis_o, a->nd, &axis) < 0) {
            Py_DECREF(a); Py_DECREF(idx);
            return NULL;
        }
        result = qnp_take_axis(a, idx, axis);
    }
    Py_DECREF(a);
    Py_DECREF(idx);
    return result;
}

/* np.add.at: accumulate into repeated destinations rather than overwriting. */
static PyObject *py_add_at(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *ao, *key, *value;
    if (!PyArg_ParseTuple(args, "OOO:add_at", &ao, &key, &value)) return NULL;
    if (!QArray_Check(ao)) {
        PyErr_SetString(PyExc_TypeError, "add.at expects an array destination");
        return NULL;
    }
    QArray *a = (QArray *)ao;
    if(!(a->flags & QNP_WRITEABLE)){PyErr_SetString(PyExc_ValueError,"add.at destination is read-only");return NULL;}
    IdxSpec spec;
    if (parse_key(&spec, a, key) < 0) return NULL;
    if (!spec.has_advanced) {
        /* Without repeated indices this is exactly `a[key] += value`. */
        PyObject *cur = qnp_getitem(a, key);
        spec_clear(&spec);
        if (cur == NULL) return NULL;
        PyObject *sum = qnp_binary_op(QOP_ADD, cur, value, NULL, NULL);
        Py_DECREF(cur);
        if (sum == NULL) return NULL;
        int rc = qnp_setitem(a, key, sum);
        Py_DECREF(sum);
        if (rc < 0) return NULL;
        Py_RETURN_NONE;
    }
    AdvPlan p;
    if (build_plan(a, &spec, &p) < 0) { spec_clear(&spec); return NULL; }
    qintp nblocks;
    qintp *offsets = gather_offsets(&p, &nblocks);
    if (offsets == NULL) { plan_release(&p); spec_clear(&spec); return NULL; }
    QArray *src = qnp_from_any(value, a->dtype, 1);
    if (src == NULL) {
        PyMem_Free(offsets); plan_release(&p); spec_clear(&spec);
        return NULL;
    }
    PyObject *bsrc = qnp_broadcast_to((PyObject *)src, p.out_shape, p.out_nd);
    Py_DECREF(src);
    if (bsrc == NULL) {
        PyMem_Free(offsets); plan_release(&p); spec_clear(&spec);
        return NULL;
    }
    QArray *val = (QArray *)bsrc;
    QArray *cval = qnp_ascontiguous(val);
    Py_DECREF(val);
    if (cval == NULL) {
        PyMem_Free(offsets); plan_release(&p); spec_clear(&spec);
        return NULL;
    }
    qintp blocksize = 1;
    for (int i = 0; i < p.basic_nd; i++) blocksize *= p.basic_shape[i];
    int rc = 0;
    if (p.basic_nd == 0) {
        for (qintp i = 0; i < nblocks; i++) {
            char *dst = a->data + offsets[i];
            const char *s = cval->data + i * QNP_ITEMSIZE(a->dtype);
            switch (a->dtype) {
                case QNP_BOOL: *(qbool *)dst |= *(const qbool *)s; break;
                case QNP_INT64: *(int64_t *)dst += *(const int64_t *)s; break;
                case QNP_FLOAT32: case QNP_COMPLEX64: qnp_write_number(dst,a->dtype,qc_add(qnp_read_number(dst,a->dtype),qnp_read_number(s,a->dtype))); break;
                case QNP_FLOAT64: *(double *)dst += *(const double *)s; break;
                default: {
                    qcomplex *d = (qcomplex *)dst;
                    const qcomplex *v = (const qcomplex *)s;
                    d->re += v->re; d->im += v->im;
                    break;
                }
            }
        }
    } else {
        PyErr_SetString(PyExc_NotImplementedError,
                        "add.at with sliced destinations is not supported");
        rc = -1;
    }
    (void)blocksize;
    Py_DECREF(cval);
    PyMem_Free(offsets);
    plan_release(&p);
    spec_clear(&spec);
    if (rc < 0) return NULL;
    Py_RETURN_NONE;
}

PyMethodDef qnp_index_methods[] = {
    {"nonzero", py_nonzero, METH_O, "Indices of the nonzero elements."},
    {"flatnonzero", py_flatnonzero, METH_O, "Indices of nonzero elements of the flattened array."},
    {"take", (PyCFunction)py_take, METH_VARARGS | METH_KEYWORDS, "Elements at the given indices."},
    {"add_at", py_add_at, METH_VARARGS, "Unbuffered in-place addition at the given indices."},
    {NULL}
};
