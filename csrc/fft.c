/* Discrete Fourier transforms.
 *
 * Mixed-radix Cooley-Tukey with specialised radix-2/3/4/5 butterflies, falling
 * back to Bluestein's chirp-z algorithm when a length has a large prime
 * factor, so any length transforms in O(n log n) rather than O(n^2).
 */
#include "qnp.h"
#include "qaccel.h"

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
    unsigned int users;
} TwiddleSlot;

static TwiddleSlot twiddles[TWIDDLE_SLOTS];
static unsigned long twiddle_clock = 0;

/* Cached because the library transforms the same length over and over --
 * recomputing n sine/cosine pairs each call costs as much as the transform. */
static const qcomplex *get_twiddles(qintp n, int sign) {
    for (int i = 0; i < TWIDDLE_SLOTS; i++) {
        if (twiddles[i].w != NULL && twiddles[i].n == n && twiddles[i].sign == sign) {
            twiddles[i].used = ++twiddle_clock;
            twiddles[i].users++;
            return twiddles[i].w;
        }
    }
    int victim = -1;
    for (int i = 0; i < TWIDDLE_SLOTS; i++) {
        if (twiddles[i].users) continue;
        if (twiddles[i].w == NULL) { victim = i; break; }
        if (victim < 0 || twiddles[i].used < twiddles[victim].used) victim = i;
    }
    qcomplex *w = qaccel_alloc(n, sizeof(qcomplex));
    if (w == NULL) return NULL;
    for (qintp k = 0; k < n; k++) {
        double angle = 2.0 * M_PI * (double)k / (double)n;
        w[k] = qc(cos(angle), sign * sin(angle));
    }
    /* At most 1 MiB per slot (8 MiB total), independent of lengths visited.
     * Active entries are pinned while another thread computes without GIL. */
    if (victim < 0 || n > 65536) return w;
    PyMem_Free(twiddles[victim].w);
    twiddles[victim].n = n;
    twiddles[victim].sign = sign;
    twiddles[victim].w = w;
    twiddles[victim].used = ++twiddle_clock;
    twiddles[victim].users = 1;
    return w;
}

static void release_twiddles(const qcomplex *w) {
    for (int i = 0; i < TWIDDLE_SLOTS; i++) {
        if (twiddles[i].w == w) { twiddles[i].users--; return; }
    }
    PyMem_Free((void *)w);
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
    if (p == 3) {
        const double sine = 0.8660254037844386; /* sqrt(3) / 2 */
        double sign = w[m * wn].im > 0.0 ? 1.0 : -1.0;
        for (qintp j = 0; j < m; j++) {
            qcomplex a0 = out[j];
            qcomplex a1 = qc_mul(out[m + j], w[j * wn]);
            qcomplex a2 = qc_mul(out[2 * m + j], w[2 * j * wn]);
            qcomplex sum = qc_add(a1, a2), difference = qc_sub(a1, a2);
            qcomplex center = qc(a0.re - 0.5 * sum.re, a0.im - 0.5 * sum.im);
            qcomplex rotation = qc(-sign * sine * difference.im,
                                    sign * sine * difference.re);
            out[j] = qc_add(a0, sum);
            out[m + j] = qc_add(center, rotation);
            out[2 * m + j] = qc_sub(center, rotation);
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
    if (p == 5) {
        const double c1 = 0.30901699437494745, s1 = 0.9510565162951535;
        const double c2 = -0.8090169943749475, s2 = 0.5877852522924731;
        double sign = w[m * wn].im > 0.0 ? 1.0 : -1.0;
        for (qintp j = 0; j < m; j++) {
            qcomplex a0 = out[j];
            qcomplex a1 = qc_mul(out[m + j], w[j * wn]);
            qcomplex a2 = qc_mul(out[2 * m + j], w[2 * j * wn]);
            qcomplex a3 = qc_mul(out[3 * m + j], w[3 * j * wn]);
            qcomplex a4 = qc_mul(out[4 * m + j], w[4 * j * wn]);
            qcomplex t1 = qc_add(a1, a4), t2 = qc_add(a2, a3);
            qcomplex t3 = qc_sub(a1, a4), t4 = qc_sub(a2, a3);
            qcomplex center1 = qc(a0.re + c1 * t1.re + c2 * t2.re,
                                  a0.im + c1 * t1.im + c2 * t2.im);
            qcomplex center2 = qc(a0.re + c2 * t1.re + c1 * t2.re,
                                  a0.im + c2 * t1.im + c1 * t2.im);
            qcomplex sine1 = qc(s1 * t3.re + s2 * t4.re,
                                s1 * t3.im + s2 * t4.im);
            qcomplex sine2 = qc(s2 * t3.re - s1 * t4.re,
                                s2 * t3.im - s1 * t4.im);
            qcomplex rotation1 = qc(-sign * sine1.im, sign * sine1.re);
            qcomplex rotation2 = qc(-sign * sine2.im, sign * sine2.re);
            out[j] = qc_add(qc_add(a0, t1), t2);
            out[m + j] = qc_add(center1, rotation1);
            out[4 * m + j] = qc_sub(center1, rotation1);
            out[2 * m + j] = qc_add(center2, rotation2);
            out[3 * m + j] = qc_sub(center2, rotation2);
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

static qintp largest_prime_factor(qintp n) {
    qintp best = 1;
    while (n % 2 == 0) { best = 2; n /= 2; }
    for (qintp p = 3; p * p <= n; p += 2)
        while (n % p == 0) { best = p; n /= p; }
    if (n > 1) best = n;
    return best;
}

static int transform_pow2(qcomplex *data, qintp n, int sign) {
    const qcomplex *w = get_twiddles(n, sign);
    if (w == NULL) return -1;
    qcomplex *out = (qcomplex *)PyMem_Malloc((size_t)n * sizeof(qcomplex));
    if (out == NULL) { release_twiddles(w); return -1; }
    Py_BEGIN_ALLOW_THREADS
    /* Radix 2/4 butterflies need no temporary; output is a valid unused
     * work pointer for the recursion. */
    fft_core(out, data, n, 1, w, 1, out);
    memcpy(data, out, (size_t)n * sizeof(qcomplex));
    Py_END_ALLOW_THREADS
    release_twiddles(w);
    PyMem_Free(out);
    return 0;
}

/* Chirp-z transform: turns an awkward length into a convolution that a
 * power-of-two transform can do. */
static int bluestein(const qcomplex *in, qcomplex *out, qintp n, int sign) {
    qintp m = 1;
    if (n > PY_SSIZE_T_MAX/2) return -1;
    while (m < 2 * n - 1) {
        if (m > PY_SSIZE_T_MAX / 2 / (qintp)sizeof(qcomplex)) return -1;
        m <<= 1;
    }
    qcomplex *chirp = (qcomplex *)PyMem_Malloc((size_t)n * sizeof(qcomplex));
    qcomplex *a = (qcomplex *)PyMem_Calloc((size_t)m, sizeof(qcomplex));
    qcomplex *b = (qcomplex *)PyMem_Calloc((size_t)m, sizeof(qcomplex));
    if (chirp == NULL || a == NULL || b == NULL) {
        PyMem_Free(chirp); PyMem_Free(a); PyMem_Free(b);
        return -1;
    }
    /* Successive squares differ by 2*k+1. Maintain k*k modulo 2*n without
     * ever forming the square: its product would overflow above k=2^32.
     * Both addends stay below 2*n, and the array-size bound makes 4*n safe. */
    uint64_t square = 0, period = 2 * (uint64_t)n;
    for (qintp k = 0; k < n; k++) {
        double angle = M_PI * (double)square / (double)n;
        chirp[k] = qc(cos(angle), sign * sin(angle));
        a[k] = qc_mul(in[k], chirp[k]);
        b[k] = qc_conj(chirp[k]);
        if (k) b[m - k] = qc_conj(chirp[k]);
        square += 2 * (uint64_t)k + 1;
        if (square >= period) square -= period;
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
    if (largest_prime_factor(n) > 37 && n > 64) {
        for (qintp i = 0; i < n; i++) scratch[i] = in[i * istride];
        qcomplex *dst = tmp;
        if (bluestein(scratch, dst, n, sign) < 0) return -1;
        for (qintp i = 0; i < n; i++) out[i * ostride] = dst[i];
        return 0;
    }
    const qcomplex *w = get_twiddles(n, sign);
    if (w == NULL) return -1;
    qcomplex *dst = ostride == 1 ? out : tmp;
    Py_BEGIN_ALLOW_THREADS
    fft_core(dst, in, n, istride, w, 1, scratch);
    if (ostride != 1)
        for (qintp i = 0; i < n; i++) out[i * ostride] = dst[i];
    Py_END_ALLOW_THREADS
    release_twiddles(w);
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
    /* An empty transform axis is rejected the same way an explicit n <= 0 is;
     * there is no spectrum of nothing, and numpy.fft refuses it too. */
    if (len < 1) {
        Py_DECREF(a0);
        PyErr_Format(PyExc_ValueError,
                     "%s: invalid number of data points (%zd) specified",
                     name, (Py_ssize_t)len);
        return NULL;
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
    /* Transform runs release the GIL. A caller may assign a.shape between
     * rows, which replaces its shape/stride allocation while retaining data.
     * Keep the original traversal metadata in private stack storage. */
    int input_nd = a->nd;
    qintp input_shape[QNP_MAXDIMS], input_strides[QNP_MAXDIMS];
    for (int d = 0; d < input_nd; ++d) {
        input_shape[d] = a->shape[d];
        input_strides[d] = a->strides[d];
    }
    if (a->nd == 1 && (len & (len - 1)) == 0) {
        const qcomplex *w = get_twiddles(len, sign);
        if (w == NULL) { Py_DECREF(a); Py_DECREF(out); return NULL; }
        qcomplex *dst = (qcomplex *)out->data;
        qintp stride = a->strides[0] / (qintp)sizeof(qcomplex);
        Py_BEGIN_ALLOW_THREADS
        fft_core(dst, (const qcomplex *)a->data, len, stride, w, 1, dst);
        if (sign > 0) {
            double inv = 1. / (double)len;
            for (qintp i=0;i<len;i++) { dst[i].re *= inv; dst[i].im *= inv; }
        }
        Py_END_ALLOW_THREADS
        release_twiddles(w);
        Py_DECREF(a);
        return (PyObject *)out;
    }
    qcomplex *scratch = qaccel_alloc(len, sizeof(qcomplex));
    qcomplex *tmp = qaccel_alloc(len, sizeof(qcomplex));
    if (scratch == NULL || tmp == NULL) {
        PyMem_Free(scratch); PyMem_Free(tmp); Py_DECREF(a); Py_DECREF(out);
        return PyErr_NoMemory();
    }
    qintp outer = qnp_size(out) / len;
    qintp idx[QNP_MAXDIMS] = {0};
    qintp istride = input_strides[axis] / (qintp)sizeof(qcomplex);
    qintp ostride = out->strides[axis] / (qintp)sizeof(qcomplex);
    int rc = 0;
    for (qintp k = 0; k < outer && rc == 0; k++) {
        const char *src = a->data;
        char *dst = out->data;
        for (int d = 0, w = 0; d < input_nd; d++) {
            if (d == axis) continue;
            src += idx[w] * input_strides[d];
            dst += idx[w] * out->strides[d];
            w++;
        }
        rc = transform_run((const qcomplex *)src, istride, (qcomplex *)dst, ostride,
                           len, sign, scratch, tmp);
        for (int d = input_nd - 2; d >= 0; d--) {
            qintp dim = 0;
            for (int e = 0, w = 0; e < input_nd; e++) {
                if (e == axis) continue;
                if (w == d) { dim = input_shape[e]; break; }
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

static PyObject *accel_fft_entry(PyObject *args, PyObject *kwds, int sign) {
    PyObject *obj;
    static char *names[] = {"a",NULL};
    if (!PyArg_ParseTupleAndKeywords(args,kwds,"O",names,&obj)) return NULL;
    QArray *a = qnp_from_any(obj,QNP_COMPLEX128,1);
    if (a == NULL) return NULL;
    if (a->nd != 1) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError,"expected a 1-dimensional array");
        return NULL;
    }
    if (a->shape[0] == 0) {
        QArray *out = qnp_new(1,a->shape,QNP_COMPLEX128);
        Py_DECREF(a);
        return (PyObject *)out;
    }
    Py_DECREF(a);
    return fft_entry(args,kwds,sign,sign < 0 ? "fft" : "ifft");
}

PyObject *qaccel_fft(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self; return accel_fft_entry(args,kwds,-1);
}

PyObject *qaccel_ifft(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self; return accel_fft_entry(args,kwds,1);
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
