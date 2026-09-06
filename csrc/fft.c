/* Discrete Fourier transforms.
 *
 * Mixed-radix Cooley-Tukey with specialised radix-2/3/4/5 butterflies, falling
 * back to Bluestein's chirp-z algorithm when a length has a large prime
 * factor, so any length transforms in O(n log n) rather than O(n^2).
 */
#include "qnp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- twiddle tables ---------------------------------------------------- */

#define TWIDDLE_SLOTS 8

typedef struct {
    qintp n;
    int sign;
    qcomplex *w;
    unsigned long used;
} TwiddleSlot;

static TwiddleSlot twiddles[TWIDDLE_SLOTS];
static unsigned long twiddle_clock = 0;

/* Cached because the library transforms the same length over and over --
 * recomputing n sine/cosine pairs each call costs as much as the transform. */
static const qcomplex *get_twiddles(qintp n, int sign) {
    for (int i = 0; i < TWIDDLE_SLOTS; i++) {
        if (twiddles[i].w != NULL && twiddles[i].n == n && twiddles[i].sign == sign) {
            twiddles[i].used = ++twiddle_clock;
            return twiddles[i].w;
        }
    }
    int victim = 0;
    for (int i = 0; i < TWIDDLE_SLOTS; i++) {
        if (twiddles[i].w == NULL) { victim = i; break; }
        if (twiddles[i].used < twiddles[victim].used) victim = i;
    }
    qcomplex *w = (qcomplex *)PyMem_Malloc((size_t)(n ? n : 1) * sizeof(qcomplex));
    if (w == NULL) return NULL;
    for (qintp k = 0; k < n; k++) {
        double angle = 2.0 * M_PI * (double)k / (double)n;
        w[k] = qc(cos(angle), sign * sin(angle));
    }
    PyMem_Free(twiddles[victim].w);
    twiddles[victim].n = n;
    twiddles[victim].sign = sign;
    twiddles[victim].w = w;
    twiddles[victim].used = ++twiddle_clock;
    return w;
}

/* ---- the recursive transform ------------------------------------------ */

static qintp choose_radix(qintp n) {
    if (n % 4 == 0) return 4;
    if (n % 2 == 0) return 2;
    for (qintp p = 3; p * p <= n; p += 2) if (n % p == 0) return p;
    return n;
}

static void fft_core(qcomplex *out, const qcomplex *in, qintp n, qintp in_stride,
                     const qcomplex *w, qintp wstep, qcomplex *tmp) {
    if (n == 1) { out[0] = in[0]; return; }
    qintp p = choose_radix(n);
    qintp m = n / p;
    for (qintp k = 0; k < p; k++)
        fft_core(out + k * m, in + k * in_stride, m, in_stride * p, w, wstep * p, tmp + k * m);
    qintp wn = wstep;             /* w[j*wn] == exp(sign*2i*pi*j/n) */
    if (p == 2) {
        for (qintp j = 0; j < m; j++) {
            qcomplex a = out[j];
            qcomplex b = qc_mul(out[m + j], w[j * wn]);
            out[j] = qc_add(a, b);
            out[m + j] = qc_sub(a, b);
        }
        return;
    }
    if (p == 4) {
        /* w[n/4] is +-i, so the radix-4 butterfly needs no extra multiply. */
        int isign = (w[n / 4 * wn].im > 0.0) ? 1 : -1;
        for (qintp j = 0; j < m; j++) {
            qcomplex a0 = out[j];
            qcomplex a1 = qc_mul(out[m + j], w[j * wn]);
            qcomplex a2 = qc_mul(out[2 * m + j], w[2 * j * wn]);
            qcomplex a3 = qc_mul(out[3 * m + j], w[3 * j * wn]);
            qcomplex t0 = qc_add(a0, a2), t1 = qc_sub(a0, a2);
            qcomplex t2 = qc_add(a1, a3), t3 = qc_sub(a1, a3);
            qcomplex jt3 = qc(-isign * t3.im, isign * t3.re);
            out[j] = qc_add(t0, t2);
            out[m + j] = qc_add(t1, jt3);
            out[2 * m + j] = qc_sub(t0, t2);
            out[3 * m + j] = qc_sub(t1, jt3);
        }
        return;
    }
    for (qintp j = 0; j < m; j++) {
        for (qintp k = 0; k < p; k++)
            tmp[k] = qc_mul(out[k * m + j], w[(k * j * wn) % (n * wn)]);
        for (qintp q = 0; q < p; q++) {
            qcomplex acc = tmp[0];
            for (qintp k = 1; k < p; k++)
                acc = qc_add(acc, qc_mul(tmp[k], w[((n / p) * ((k * q) % p)) * wn]));
            out[q * m + j] = acc;
        }
    }
}

static int largest_prime_factor(qintp n) {
    qintp best = 1;
    while (n % 2 == 0) { best = 2; n /= 2; }
    for (qintp p = 3; p * p <= n; p += 2)
        while (n % p == 0) { best = p; n /= p; }
    if (n > 1) best = n;
    return (int)best;
}

static int transform_pow2(qcomplex *data, qintp n, int sign) {
    const qcomplex *w = get_twiddles(n, sign);
    if (w == NULL) return -1;
    qcomplex *out = (qcomplex *)PyMem_Malloc((size_t)n * sizeof(qcomplex));
    qcomplex *tmp = (qcomplex *)PyMem_Malloc((size_t)n * sizeof(qcomplex));
    if (out == NULL || tmp == NULL) { PyMem_Free(out); PyMem_Free(tmp); return -1; }
    fft_core(out, data, n, 1, w, 1, tmp);
    memcpy(data, out, (size_t)n * sizeof(qcomplex));
    PyMem_Free(out);
    PyMem_Free(tmp);
    return 0;
}

/* Chirp-z transform: turns an awkward length into a convolution that a
 * power-of-two transform can do. */
static int bluestein(const qcomplex *in, qcomplex *out, qintp n, int sign) {
    qintp m = 1;
    while (m < 2 * n - 1) m <<= 1;
    qcomplex *chirp = (qcomplex *)PyMem_Malloc((size_t)n * sizeof(qcomplex));
    qcomplex *a = (qcomplex *)PyMem_Calloc((size_t)m, sizeof(qcomplex));
    qcomplex *b = (qcomplex *)PyMem_Calloc((size_t)m, sizeof(qcomplex));
    if (chirp == NULL || a == NULL || b == NULL) {
        PyMem_Free(chirp); PyMem_Free(a); PyMem_Free(b);
        return -1;
    }
    for (qintp k = 0; k < n; k++) {
        /* k*k modulo 2n keeps the angle small enough to stay accurate. */
        qintp kk = (k * k) % (2 * n);
        double angle = M_PI * (double)kk / (double)n;
        chirp[k] = qc(cos(angle), sign * sin(angle));
        a[k] = qc_mul(in[k], chirp[k]);
        b[k] = qc_conj(chirp[k]);
        if (k) b[m - k] = qc_conj(chirp[k]);
    }
    int rc = transform_pow2(a, m, -1);
    if (rc == 0) rc = transform_pow2(b, m, -1);
    if (rc == 0) {
        for (qintp i = 0; i < m; i++) a[i] = qc_mul(a[i], b[i]);
        rc = transform_pow2(a, m, 1);
    }
    if (rc == 0) {
        for (qintp k = 0; k < n; k++) {
            qcomplex v = qc(a[k].re / (double)m, a[k].im / (double)m);
            out[k] = qc_mul(v, chirp[k]);
        }
    }
    PyMem_Free(chirp);
    PyMem_Free(a);
    PyMem_Free(b);
    return rc;
}

/* One transform of length n, reading with `istride` and writing with `ostride`. */
static int transform_run(const qcomplex *in, qintp istride, qcomplex *out, qintp ostride,
                         qintp n, int sign, qcomplex *scratch, qcomplex *tmp) {
    if (n == 0) return 0;
    for (qintp i = 0; i < n; i++) scratch[i] = in[i * istride];
    if (largest_prime_factor(n) > 37 && n > 64) {
        qcomplex *dst = tmp;
        if (bluestein(scratch, dst, n, sign) < 0) return -1;
        for (qintp i = 0; i < n; i++) out[i * ostride] = dst[i];
        return 0;
    }
    const qcomplex *w = get_twiddles(n, sign);
    if (w == NULL) return -1;
    qcomplex *dst = tmp;
    qcomplex *work = tmp + n;
    fft_core(dst, scratch, n, 1, w, 1, work);
    for (qintp i = 0; i < n; i++) out[i * ostride] = dst[i];
    return 0;
}

/* ---- the Python-facing transforms ------------------------------------- */

static PyObject *fft_entry(PyObject *args, PyObject *kwds, int sign, const char *name) {
    PyObject *ao, *n_obj = Py_None, *axis_o = NULL;
    static char *kwlist[] = {"a", "n", "axis", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO", kwlist, &ao, &n_obj, &axis_o))
        return NULL;
    QArray *a0 = qnp_from_any(ao, QNP_COMPLEX128, 1);
    if (a0 == NULL) return NULL;
    if (a0->nd == 0) {
        Py_DECREF(a0);
        PyErr_Format(PyExc_ValueError, "%s: input must be at least one dimensional", name);
        return NULL;
    }
    int axis = a0->nd - 1;
    if (axis_o != NULL && axis_o != Py_None && qnp_parse_axis(axis_o, a0->nd, &axis) < 0) {
        Py_DECREF(a0);
        return NULL;
    }
    qintp len = a0->shape[axis];
    if (n_obj != Py_None) {
        Py_ssize_t requested = PyNumber_AsSsize_t(n_obj, PyExc_OverflowError);
        if (requested == -1 && PyErr_Occurred()) { Py_DECREF(a0); return NULL; }
        if (requested <= 0) {
            Py_DECREF(a0);
            PyErr_Format(PyExc_ValueError,
                         "%s: invalid number of data points (%zd) specified", name, requested);
            return NULL;
        }
        len = requested;
    }
    qintp shape[QNP_MAXDIMS];
    for (int i = 0; i < a0->nd; i++) shape[i] = a0->shape[i];
    shape[axis] = len;
    QArray *a = a0;
    if (len != a0->shape[axis]) {
        /* Truncate or zero-pad along the transform axis. */
        QArray *padded = qnp_new(a0->nd, shape, QNP_COMPLEX128);
        if (padded == NULL) { Py_DECREF(a0); return NULL; }
        memset(padded->data, 0, (size_t)qnp_size(padded) * sizeof(qcomplex));
        qintp copy = len < a0->shape[axis] ? len : a0->shape[axis];
        qintp sub[QNP_MAXDIMS];
        for (int i = 0; i < a0->nd; i++) sub[i] = a0->shape[i];
        sub[axis] = copy;
        QArray *sv = qnp_new_view(a0, a0->data, a0->nd, sub, a0->strides, a0->dtype);
        QArray *dv = qnp_new_view(padded, padded->data, padded->nd, sub, padded->strides,
                                  padded->dtype);
        int rc = (sv && dv) ? qnp_copy_into(dv, sv) : -1;
        Py_XDECREF(sv);
        Py_XDECREF(dv);
        Py_DECREF(a0);
        if (rc < 0) { Py_DECREF(padded); return NULL; }
        a = padded;
    }
    QArray *out = qnp_new(a->nd, shape, QNP_COMPLEX128);
    if (out == NULL) { Py_DECREF(a); return NULL; }
    qcomplex *scratch = (qcomplex *)PyMem_Malloc((size_t)(len ? len : 1) * sizeof(qcomplex));
    qcomplex *tmp = (qcomplex *)PyMem_Malloc((size_t)(len ? 2 * len : 1) * sizeof(qcomplex));
    if (scratch == NULL || tmp == NULL) {
        PyMem_Free(scratch); PyMem_Free(tmp); Py_DECREF(a); Py_DECREF(out);
        return PyErr_NoMemory();
    }
    qintp outer = qnp_size(a) / (len ? len : 1);
    qintp idx[QNP_MAXDIMS] = {0};
    qintp istride = a->strides[axis] / (qintp)sizeof(qcomplex);
    qintp ostride = out->strides[axis] / (qintp)sizeof(qcomplex);
    int rc = 0;
    for (qintp k = 0; k < outer && rc == 0; k++) {
        const char *src = a->data;
        char *dst = out->data;
        for (int d = 0, w = 0; d < a->nd; d++) {
            if (d == axis) continue;
            src += idx[w] * a->strides[d];
            dst += idx[w] * out->strides[d];
            w++;
        }
        rc = transform_run((const qcomplex *)src, istride, (qcomplex *)dst, ostride,
                           len, sign, scratch, tmp);
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
    PyMem_Free(scratch);
    PyMem_Free(tmp);
    Py_DECREF(a);
    if (rc < 0) {
        Py_DECREF(out);
        if (!PyErr_Occurred()) PyErr_NoMemory();
        return NULL;
    }
    if (sign > 0) {
        qintp total = qnp_size(out);
        qcomplex *p = (qcomplex *)out->data;
        double inv = len ? 1.0 / (double)len : 0.0;
        for (qintp i = 0; i < total; i++) { p[i].re *= inv; p[i].im *= inv; }
    }
    return (PyObject *)out;
}

static PyObject *py_fft(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self; return fft_entry(args, kwds, -1, "fft");
}
static PyObject *py_ifft(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self; return fft_entry(args, kwds, 1, "ifft");
}

PyMethodDef qnp_fft_methods[] = {
    {"fft", (PyCFunction)py_fft, METH_VARARGS | METH_KEYWORDS, "Discrete Fourier transform."},
    {"ifft", (PyCFunction)py_ifft, METH_VARARGS | METH_KEYWORDS, "Inverse discrete Fourier transform."},
    {NULL}
};
