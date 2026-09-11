/* Direct factorisations for the compiled acceleration backend.
 *
 * Inputs are copied only when a factorisation overwrites them.  Block updates
 * operate on disjoint views of the original work array, so neither LU nor
 * Cholesky needs the panel-sized operand copies required by the old bridge.
 */
#include "qaccel.h"
#include <float.h>

#define DMIN(a, b) ((a) < (b) ? (a) : (b))
#define DMAX(a, b) ((a) > (b) ? (a) : (b))
#define LU_LEAF 16
#define LU_NB 128
#define CHOL_NB 192
#define CHOL_SB 32
#define QR_NB 64

#if defined(_MSC_VER)
#define DIRECT_RESTRICT __restrict
#else
#define DIRECT_RESTRICT restrict
#endif

/* A rotation's two rows are disjoint, so vectorisation needs no alias check. */
static void rotate_rows(qintp n, double c, double s,
                        double *DIRECT_RESTRICT x, double *DIRECT_RESTRICT y) {
    for (qintp i = 0; i < n; i++) {
        double u = x[i], w = y[i];
        x[i] = c * u - s * w; y[i] = s * u + c * w;
    }
}

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define DIRECT_AVX2 1
__attribute__((target("avx2")))
static void rotate_rows_avx2(qintp n, double c, double s, double *x, double *y) {
    __m256d cv = _mm256_set1_pd(c), sv = _mm256_set1_pd(s);
    qintp i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d u = _mm256_loadu_pd(x + i), w = _mm256_loadu_pd(y + i);
        _mm256_storeu_pd(x + i, _mm256_sub_pd(_mm256_mul_pd(cv, u), _mm256_mul_pd(sv, w)));
        _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_mul_pd(sv, u), _mm256_mul_pd(cv, w)));
    }
    rotate_rows(n - i, c, s, x + i, y + i);
}
#endif

static double direct_dot(const double *x, const double *y, qintp n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    qintp i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += x[i] * y[i]; s1 += x[i + 1] * y[i + 1];
        s2 += x[i + 2] * y[i + 2]; s3 += x[i + 3] * y[i + 3];
    }
    double s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += x[i] * y[i];
    return s;
}

static void direct_axpy(qintp n, double alpha, const double *x, double *y) {
    for (qintp i = 0; i < n; i++) y[i] += alpha * x[i];
}

/* Ordinary magnitudes need one pass; only extreme exponents need scaling. */
static double direct_norm(const double *x, qintp n) {
    double ss = direct_dot(x, x, n);
    if (isnan(ss) || (isfinite(ss) && ss >= DBL_MIN * (double)n)) return sqrt(ss);
    double scale = 0.0;
    for (qintp i = 0; i < n; i++) if (fabs(x[i]) > scale) scale = fabs(x[i]);
    if (scale == 0.0 || isinf(scale)) return scale;
    double s = 0.0;
    for (qintp i = 0; i < n; i++) { double v = x[i] / scale; s += v * v; }
    return scale * sqrt(s);
}

static qintp triangular_solve(qintp n, const double *a, double *b,
                             int unit, int back, int transposed) {
    if (transposed) {
        for (qintp i = n; i-- > 0;) {
            if (!unit) {
                if (a[i * n + i] == 0.0) return i;
                b[i] /= a[i * n + i];
            }
            if (b[i] != 0.0) direct_axpy(i, -b[i], a + i * n, b);
        }
    } else {
        for (qintp row = 0; row < n; row++) {
            qintp i = back ? n - 1 - row : row;
            double s = b[i] - (back ? direct_dot(a + i * n + i + 1, b + i + 1, n - i - 1)
                                    : direct_dot(a + i * n, b, i));
            if (unit) b[i] = s;
            else {
                if (a[i * n + i] == 0.0) return i;
                b[i] = s / a[i * n + i];
            }
        }
    }
    return -1;
}

static PyObject *py_triangular(PyObject *args, PyObject *kwargs, int back) {
    PyObject *ao, *bo;
    int unit = 0, transposed = 0;
    static char *forward_keys[] = {"l", "b", "unit_diagonal", NULL};
    static char *back_keys[] = {"u", "b", "unit_diagonal", "transposed", NULL};
    if (back) {
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pp:back_substitution", back_keys,
                                         &ao, &bo, &unit, &transposed)) return NULL;
    } else if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p:forward_substitution", forward_keys,
                                            &ao, &bo, &unit)) return NULL;
    QArray *a = qaccel_input(ao, 2), *b = NULL;
    if (a == NULL) return NULL;
    if (qaccel_square(a) < 0) goto fail;
    /* Converting b can run user Python code, including changing a's shape. */
    qintp n = a->shape[0], bad;
    const double *ap = (const double *)a->data;
    b = qaccel_copy(bo, 1);
    if (b == NULL) goto fail;
    if (b->shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "rhs must match the matrix size");
        goto fail;
    }
    double *bp = (double *)b->data;
    Py_BEGIN_ALLOW_THREADS
    bad = triangular_solve(n, ap, bp, unit, back, transposed);
    Py_END_ALLOW_THREADS
    if (bad >= 0) {
        PyErr_Format(PyExc_RuntimeError, "singular:zero diagonal entry at row %zd", bad);
        goto fail;
    }
    Py_DECREF(a);
    return (PyObject *)b;
fail:
    Py_DECREF(a); Py_XDECREF(b); return NULL;
}

static PyObject *py_forward(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self; return py_triangular(args, kwargs, 0);
}
static PyObject *py_back(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self; return py_triangular(args, kwargs, 1);
}

static PyObject *py_symmetric(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *ao;
    double atol = 1e-12, rtol = 1e-5;
    static char *keys[] = {"a", "atol", "rtol", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dd:is_symmetric", keys, &ao, &atol, &rtol))
        return NULL;
    QArray *a = qaccel_input(ao, 2);
    if (a == NULL) return NULL;
    qintp n = a->shape[0];
    int result = n == a->shape[1];
    const double *p = (const double *)a->data;
    Py_BEGIN_ALLOW_THREADS
    for (qintp i = 0; result && i < n; i++) {
        if (isnan(p[i * n + i])) { result = 0; break; }
        for (qintp j = 0; j < i; j++) {
            double x = p[i * n + j], y = p[j * n + i];
            if (x == y) continue;
            double delta = fabs(x - y);
            if (!isfinite(x) || !isfinite(y) || !(delta <= atol + rtol * fabs(y)) ||
                !(delta <= atol + rtol * fabs(x))) { result = 0; break; }
        }
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(a);
    return PyBool_FromLong(result);
}

static void lu_panel(qintp n, double *a, qintp j, qintp w, int64_t *perm) {
    if (w > LU_LEAF) {
        qintp left = w / 2;
        lu_panel(n, a, j, left, perm);
        for (qintp k = j; k < j + left; k++)
            for (qintp i = k + 1; i < j + left; i++)
                if (a[i * n + k] != 0.0)
                    direct_axpy(w - left, -a[i * n + k], a + k * n + j + left,
                                a + i * n + j + left);
        qaccel_gemm(n - j - left, w - left, left, -1.0,
                    a + (j + left) * n + j, n, a + j * n + j + left, n,
                    a + (j + left) * n + j + left, n, 0);
        lu_panel(n, a, j + left, w - left, perm);
        return;
    }
    for (qintp k = j; k < j + w; k++) {
        qintp pivot = k;
        double best = fabs(a[k * n + k]);
        for (qintp i = k + 1; i < n; i++) {
            double v = fabs(a[i * n + k]);
            if (v > best) { best = v; pivot = i; }
        }
        if (pivot != k) {
            for (qintp c = 0; c < n; c++) {
                double tmp = a[k * n + c];
                a[k * n + c] = a[pivot * n + c]; a[pivot * n + c] = tmp;
            }
            int64_t tmp = perm[k]; perm[k] = perm[pivot]; perm[pivot] = tmp;
        }
        double piv = a[k * n + k];
        if (piv == 0.0) continue;
        double inv = 1.0 / piv;
        for (qintp i = k + 1; i < n; i++) a[i * n + k] *= inv;
        for (qintp i = k + 1; i < n; i++)
            if (a[i * n + k] != 0.0)
                direct_axpy(j + w - k - 1, -a[i * n + k],
                            a + k * n + k + 1, a + i * n + k + 1);
    }
}

static PyObject *py_plu(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *ao;
    static char *keys[] = {"a", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:plu", keys, &ao)) return NULL;
    QArray *a = qaccel_copy(ao, 2), *perm = NULL;
    if (a == NULL) return NULL;
    if (qaccel_square(a) < 0) goto fail;
    qintp n = a->shape[0];
    perm = qnp_new(1, &n, QNP_INT64);
    if (perm == NULL) goto fail;
    double *p = (double *)a->data;
    int64_t *pv = (int64_t *)perm->data;
    Py_BEGIN_ALLOW_THREADS
    for (qintp i = 0; i < n; i++) pv[i] = i;
    for (qintp j = 0; j < n; j += LU_NB) {
        qintp jb = DMIN(LU_NB, n - j), rest = n - j - jb;
        lu_panel(n, p, j, jb, pv);
        if (rest > 0) {
            for (qintp k = j; k < j + jb; k++)
                for (qintp i = k + 1; i < j + jb; i++)
                    if (p[i * n + k] != 0.0)
                        direct_axpy(rest, -p[i * n + k], p + k * n + j + jb,
                                    p + i * n + j + jb);
            qaccel_gemm(rest, rest, jb, -1.0, p + (j + jb) * n + j, n,
                        p + j * n + j + jb, n, p + (j + jb) * n + j + jb, n, 0);
        }
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NN", perm, a);
fail:
    Py_DECREF(a); Py_XDECREF(perm); return NULL;
}

static qintp cholesky_kernel(qintp n, double *a, double *bad_value) {
    for (qintp j = 0; j < n; j += CHOL_NB) {
        qintp jb = DMIN(CHOL_NB, n - j);
        for (qintp kb = j; kb < j + jb; kb += CHOL_SB) {
            qintp kbb = DMIN(CHOL_SB, j + jb - kb);
            if (kb > j)
                qaccel_gemm(n - kb, kbb, kb - j, -1.0, a + kb * n + j, n,
                            a + kb * n + j, n, a + kb * n + kb, n, 1);
            for (qintp k = kb; k < kb + kbb; k++) {
                const double *row = a + k * n + kb;
                double d = a[k * n + k] - direct_dot(row, row, k - kb);
                if (d <= 0.0) { *bad_value = d; return k; }
                double dsq = sqrt(d), inv = 1.0 / dsq;
                a[k * n + k] = dsq;
                for (qintp i = k + 1; i < n; i++)
                    a[i * n + k] = (a[i * n + k] - direct_dot(a + i * n + kb, row, k - kb)) * inv;
            }
        }
        qintp rest = n - j - jb;
        if (rest > 0)
            qaccel_gemm(rest, rest, jb, -1.0, a + (j + jb) * n + j, n,
                        a + (j + jb) * n + j, n, a + (j + jb) * n + j + jb, n, 1);
    }
    for (qintp i = 0; i < n; i++)
        for (qintp j = i + 1; j < n; j++) a[i * n + j] = 0.0;
    return -1;
}

static PyObject *py_cholesky(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *ao;
    static char *keys[] = {"a", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:cholesky", keys, &ao)) return NULL;
    QArray *a = qaccel_copy(ao, 2);
    if (a == NULL) return NULL;
    if (qaccel_square(a) < 0) { Py_DECREF(a); return NULL; }
    qintp n = a->shape[0], bad;
    double *data = (double *)a->data;
    double value = 0.0;
    Py_BEGIN_ALLOW_THREADS
    bad = cholesky_kernel(n, data, &value);
    Py_END_ALLOW_THREADS
    if (bad >= 0) {
        char message[192];
        PyOS_snprintf(message, sizeof(message), "notpd:matrix is not positive definite (non-positive pivot %.3e at index %zd)", value, bad);
        PyErr_SetString(PyExc_RuntimeError, message);
        Py_DECREF(a); return NULL;
    }
    return (PyObject *)a;
}

/* Build the compact WY representation H = I - V T V'. */
static void build_wy(qintp rows, qintp width, const double *v,
                     double *t, double *tt, double *vt, double *work) {
    memset(t, 0, (size_t)(width * width) * sizeof(double));
    for (qintp j = 0; j < width; j++) {
        for (qintp p = 0; p < j; p++) {
            double acc = 0.0;
            for (qintp i = j; i < rows; i++) acc += v[i * width + p] * v[i * width + j];
            work[p] = acc;
        }
        for (qintp p = 0; p < j; p++) {
            double acc = 0.0;
            for (qintp z = p; z < j; z++) acc += t[p * width + z] * work[z];
            t[p * width + j] = -2.0 * acc;
        }
        t[j * width + j] = 2.0;
    }
    for (qintp i = 0; i < rows; i++)
        for (qintp j = 0; j < width; j++) vt[j * rows + i] = v[i * width + j];
    for (qintp i = 0; i < width; i++)
        for (qintp j = 0; j < width; j++) tt[j * width + i] = t[i * width + j];
}

static void wy_apply(qintp rows, qintp cols, qintp width,
                     const double *v, const double *vt, const double *t,
                     double *dest, qintp ld, double *w1, double *w2) {
    if (!rows || !cols || !width) return;
    memset(w1, 0, (size_t)(width * cols) * sizeof(double));
    memset(w2, 0, (size_t)(width * cols) * sizeof(double));
    qaccel_gemm(width, cols, rows, 1.0, vt, rows, dest, ld, w1, cols, 0);
    qaccel_gemm(width, cols, width, 1.0, t, width, w1, cols, w2, cols, 0);
    qaccel_gemm(rows, cols, width, -1.0, v, width, w2, cols, dest, ld, 0);
}

static void householder_kernel(qintp m, qintp n, double *r, double *q,
                               qintp qcols, qintp lim, double *v, double *vt,
                               double *t, double *tt, double *x, double *small,
                               double *leading, double *w1, double *w2) {
    for (qintp k = 0; k < lim; k += QR_NB) {
        qintp width = DMIN(QR_NB, lim - k), rows = m - k;
        memset(v, 0, (size_t)(rows * width) * sizeof(double));
        for (qintp j = 0; j < width; j++) {
            qintp col = k + j, len = m - col;
            leading[col] = 0.0;
            for (qintp i = 0; i < len; i++) x[i] = r[(col + i) * n + col];
            double norm = direct_norm(x, len);
            if (norm == 0.0) continue;
            double alpha = x[0] >= 0.0 ? -norm : norm;
            /* Scaling before subtraction prevents x[0] - alpha overflowing. */
            if (isfinite(norm) && fabs(x[0]) > DBL_MAX - norm) {
                for (qintp i = 0; i < len; i++) x[i] /= norm;
                x[0] -= alpha / norm;
            } else x[0] -= alpha;
            double vn = direct_norm(x, len);
            if (vn == 0.0) continue;
            for (qintp i = 0; i < len; i++) {
                x[i] /= vn;
                v[(j + i) * width + j] = x[i];
            }
            leading[col] = x[0];
            memset(small, 0, (size_t)width * sizeof(double));
            for (qintp i = 0; i < len; i++)
                if (x[i] != 0.0)
                    direct_axpy(width - j, x[i], r + (col + i) * n + col, small + j);
            for (qintp i = 0; i < len; i++)
                if (x[i] != 0.0)
                    direct_axpy(width - j, -2.0 * x[i], small + j, r + (col + i) * n + col);
        }
        build_wy(rows, width, v, t, tt, vt, small);
        wy_apply(rows, n - k - width, width, v, vt, tt, r + k * n + k + width, n, w1, w2);
        /* Preserve each reflector below R's diagonal for the reverse Q pass. */
        for (qintp j = 0; j < width; j++)
            for (qintp i = j + 1; i < rows; i++) r[(k + i) * n + k + j] = v[i * width + j];
    }
    if (q != NULL) {
        memset(q, 0, (size_t)(m * qcols) * sizeof(double));
        for (qintp i = 0; i < qcols; i++) q[i * qcols + i] = 1.0;
        qintp k = lim ? ((lim - 1) / QR_NB) * QR_NB : 0;
        if (lim) do {
            qintp width = DMIN(QR_NB, lim - k), rows = m - k;
            memset(v, 0, (size_t)(rows * width) * sizeof(double));
            for (qintp j = 0; j < width; j++) {
                v[j * width + j] = leading[k + j];
                for (qintp i = j + 1; i < rows; i++) v[i * width + j] = r[(k + i) * n + k + j];
            }
            build_wy(rows, width, v, t, tt, vt, small);
            /* Earlier columns still contain their untouched identity columns;
             * they are identically zero in these rows and need no update. */
            wy_apply(rows, qcols - k, width, v, vt, t, q + k * qcols + k, qcols, w1, w2);
            if (k == 0) break;
            k -= QR_NB;
        } while (1);
    }
    for (qintp i = 0; i < m; i++)
        for (qintp j = 0; j < DMIN(i, n); j++) r[i * n + j] = 0.0;
}

static PyObject *py_householder(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *ao;
    int want_q = 1, reduced = 0;
    static char *keys[] = {"a", "want_q", "reduced", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|pp:householder_qr", keys, &ao, &want_q, &reduced))
        return NULL;
    QArray *r = qaccel_copy(ao, 2), *q = NULL;
    double *v = NULL, *vt = NULL, *t = NULL, *tt = NULL, *x = NULL;
    double *small = NULL, *leading = NULL, *w1 = NULL, *w2 = NULL;
    if (r == NULL) return NULL;
    qintp m = r->shape[0], n = r->shape[1];
    qintp qcols = reduced ? DMIN(m, n) : m;
    qintp qs[2] = {want_q ? m : 0, qcols};
    q = qnp_new(2, qs, QNP_FLOAT64);
    if (q == NULL) goto fail;
    qintp lim = DMIN(n, m ? m - 1 : 0), nb = DMIN(QR_NB, lim);
    qintp workcols = DMAX(n, want_q ? qcols : 0);
#define ALLOC_WORK(name, count) do { name = qaccel_alloc((count), sizeof(double)); if (name == NULL) goto fail; } while (0)
    ALLOC_WORK(v, m * nb); ALLOC_WORK(vt, m * nb);
    ALLOC_WORK(t, nb * nb); ALLOC_WORK(tt, nb * nb);
    ALLOC_WORK(x, m); ALLOC_WORK(small, nb); ALLOC_WORK(leading, lim);
    ALLOC_WORK(w1, nb * workcols); ALLOC_WORK(w2, nb * workcols);
#undef ALLOC_WORK
    double *rp = (double *)r->data, *qp = want_q ? (double *)q->data : NULL;
    Py_BEGIN_ALLOW_THREADS
    householder_kernel(m, n, rp, qp, qcols, lim, v, vt, t, tt, x, small, leading, w1, w2);
    Py_END_ALLOW_THREADS
    PyMem_Free(v); PyMem_Free(vt); PyMem_Free(t); PyMem_Free(tt); PyMem_Free(x);
    PyMem_Free(small); PyMem_Free(leading); PyMem_Free(w1); PyMem_Free(w2);
    if (reduced && m > n) {
        /* This unexported, owned work array is now just the top n rows. */
        char *data = PyMem_Realloc(r->data, (size_t)(n ? n * n : 1) * sizeof(double));
        if (data != NULL) r->data = data;
        r->shape[0] = n;
        if (n <= 1) r->flags |= QNP_F_CONTIGUOUS;
    }
    return Py_BuildValue("NN", q, r);
fail:
    PyMem_Free(v); PyMem_Free(vt); PyMem_Free(t); PyMem_Free(tt); PyMem_Free(x);
    PyMem_Free(small); PyMem_Free(leading); PyMem_Free(w1); PyMem_Free(w2);
    Py_DECREF(r); Py_XDECREF(q); return NULL;
}

static PyObject *py_least_squares(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *ao, *bo;
    static char *keys[] = {"a", "b", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:qr_least_squares", keys, &ao, &bo)) return NULL;
    QArray *a = qaccel_copy(ao, 2), *b = NULL;
    double *x = NULL, *projection = NULL;
    if (a == NULL) return NULL;
    qintp m = a->shape[0], n = a->shape[1], bad = -1;
    if (m < n) {
        PyErr_SetString(PyExc_ValueError, "least squares requires at least as many rows as columns");
        goto fail;
    }
    b = qaccel_copy(bo, 1);
    if (b == NULL) goto fail;
    if (b->shape[0] != m) {
        PyErr_SetString(PyExc_ValueError, "rhs must match the matrix row count"); goto fail;
    }
    x = qaccel_alloc(m, sizeof(double));
    if (x == NULL) goto fail;
    projection = qaccel_alloc(n, sizeof(double));
    if (projection == NULL) goto fail;
    double *ap = (double *)a->data, *bp = (double *)b->data;
    Py_BEGIN_ALLOW_THREADS
    for (qintp col = 0; col < DMIN(n, m ? m - 1 : 0); col++) {
        qintp len = m - col;
        for (qintp i = 0; i < len; i++) x[i] = ap[(col + i) * n + col];
        double norm = direct_norm(x, len);
        if (norm == 0.0) continue;
        double alpha = x[0] >= 0.0 ? -norm : norm;
        if (isfinite(norm) && fabs(x[0]) > DBL_MAX - norm) {
            for (qintp i = 0; i < len; i++) x[i] /= norm;
            x[0] -= alpha / norm;
        } else x[0] -= alpha;
        double vn = direct_norm(x, len);
        if (vn == 0.0) continue;
        for (qintp i = 0; i < len; i++) x[i] /= vn;
        memset(projection + col, 0, (size_t)(n - col) * sizeof(double));
        for (qintp i = 0; i < len; i++)
            direct_axpy(n - col, x[i], ap + (col + i) * n + col, projection + col);
        for (qintp i = 0; i < len; i++)
            direct_axpy(n - col, -2.0 * x[i], projection + col, ap + (col + i) * n + col);
        direct_axpy(len, -2.0 * direct_dot(x, bp + col, len), x, bp + col);
    }
    bad = triangular_solve(n, ap, bp, 0, 1, 0);
    Py_END_ALLOW_THREADS
    if (bad >= 0) {
        PyErr_Format(PyExc_RuntimeError, "singular:zero diagonal entry at row %zd", bad); goto fail;
    }
    if (m > n) {
        char *data = PyMem_Realloc(b->data, (size_t)(n ? n : 1) * sizeof(double));
        if (data != NULL) b->data = data;
        b->shape[0] = n;
    }
    PyMem_Free(x); PyMem_Free(projection); Py_DECREF(a);
    return (PyObject *)b;
fail:
    PyMem_Free(x); PyMem_Free(projection); Py_DECREF(a); Py_XDECREF(b); return NULL;
}

static PyObject *py_givens(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *ao;
    static char *keys[] = {"a", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:givens_qr", keys, &ao)) return NULL;
    QArray *r = qaccel_copy(ao, 2);
    if (r == NULL) return NULL;
    qintp m = r->shape[0], n = r->shape[1], shape[2] = {m, m};
    QArray *q = qnp_new(2, shape, QNP_FLOAT64);
    if (q == NULL) { Py_DECREF(r); return NULL; }
    double *rp = (double *)r->data, *qp = (double *)q->data;
    void (*rotate)(qintp, double, double, double *, double *) = rotate_rows;
#ifdef DIRECT_AVX2
    if (__builtin_cpu_supports("avx2")) rotate = rotate_rows_avx2;
#endif
    Py_BEGIN_ALLOW_THREADS
    memset(qp, 0, (size_t)(m * m) * sizeof(double));
    for (qintp i = 0; i < m; i++) qp[i * m + i] = 1.0;
    for (qintp j = 0; j < DMIN(m, n); j++) {
        for (qintp i = m - 1; i > j; i--) {
            double a = rp[(i - 1) * n + j], b = rp[i * n + j], c, s;
            if (b == 0.0) continue;
            if (fabs(b) > fabs(a)) {
                double tau = -a / b; s = 1.0 / sqrt(1.0 + tau * tau); c = s * tau;
            } else {
                double tau = -b / a; c = 1.0 / sqrt(1.0 + tau * tau); s = c * tau;
            }
            rotate(n - j, c, s, rp + (i - 1) * n + j, rp + i * n + j);
            /* Accumulate Q transposed so each rotation streams two rows. */
            double *q0 = qp + (i - 1) * m, *q1 = qp + i * m;
            rotate(m, c, s, q0, q1);
        }
    }
    for (qintp i = 0; i < m; i++)
        for (qintp j = 0; j < i; j++) {
            double tmp = qp[i * m + j];
            qp[i * m + j] = qp[j * m + i]; qp[j * m + i] = tmp;
        }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NN", q, r);
}

/* Transpose a rectangular buffer by permutation cycles. One bit per element
 * records completed cycles; square matrices need no scratch at all. */
static void transpose_inplace(double *data, qintp rows, qintp cols, unsigned char *visited) {
    if (rows == cols) {
        for (qintp i = 0; i < rows; i++)
            for (qintp j = 0; j < i; j++) {
                double tmp = data[i * cols + j];
                data[i * cols + j] = data[j * rows + i]; data[j * rows + i] = tmp;
            }
        return;
    }
    if (rows <= 1 || cols <= 1) return;
    qintp total = rows * cols;
    memset(visited, 0, (size_t)((total + 7) / 8));
    for (qintp start = 0; start < total; start++) {
        if (visited[start / 8] & (1u << (start % 8))) continue;
        qintp current = start;
        double value = data[start];
        do {
            qintp next = (current % cols) * rows + current / cols;
            double tmp = data[next]; data[next] = value; value = tmp;
            visited[current / 8] |= (unsigned char)(1u << (current % 8));
            current = next;
        } while (current != start);
    }
}

static PyObject *py_svd_jacobi(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *ao;
    double tol = 1e-13;
    Py_ssize_t max_sweeps = 60;
    static char *keys[] = {"a", "tol", "max_sweeps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dn:svd_jacobi", keys, &ao, &tol, &max_sweeps))
        return NULL;
    if (max_sweeps < 0) {
        PyErr_SetString(PyExc_OverflowError, "max_sweeps cannot be negative"); return NULL;
    }
    QArray *w = qaccel_copy(ao, 2);
    if (w == NULL) return NULL;
    qintp m = w->shape[0], n = w->shape[1], shape[2] = {n, n}, sweeps = 0;
    if (m < n) {
        Py_DECREF(w);
        PyErr_SetString(PyExc_ValueError, "one-sided Jacobi requires at least as many rows as columns");
        return NULL;
    }
    QArray *v = qnp_new(2, shape, QNP_FLOAT64);
    if (v == NULL) { Py_DECREF(w); return NULL; }
    unsigned char *visited = NULL;
    if (m != n && n > 1) {
        visited = qaccel_alloc((m * n + 7) / 8, sizeof(unsigned char));
        if (visited == NULL) { Py_DECREF(w); Py_DECREF(v); return NULL; }
    }
    double *wp = (double *)w->data, *vp = (double *)v->data;
    Py_BEGIN_ALLOW_THREADS
    /* Physical rows now hold the logical columns rotated by Jacobi. Both
     * products and rotations stream contiguous data throughout every sweep. */
    transpose_inplace(wp, m, n, visited);
    memset(vp, 0, (size_t)(n * n) * sizeof(double));
    for (qintp i = 0; i < n; i++) vp[i * n + i] = 1.0;
    for (; sweeps < max_sweeps;) {
        sweeps++;
        int rotated = 0;
        for (qintp p = 0; p + 1 < n; p++) {
            for (qintp q = p + 1; q < n; q++) {
                double app = 0.0, aqq = 0.0, apq = 0.0;
                double *wpcol = wp + p * m, *wqcol = wp + q * m;
                for (qintp i = 0; i < m; i++) {
                    double xp = wpcol[i], xq = wqcol[i];
                    app += xp * xp; aqq += xq * xq; apq += xp * xq;
                }
                if (apq == 0.0 || fabs(apq) < tol * sqrt(app * aqq + 1e-300)) continue;
                rotated = 1;
                double zeta = (aqq - app) / (2.0 * apq), t;
                if (zeta == 0.0) t = 1.0;
                else if (fabs(zeta) > 1e8) t = 1.0 / (2.0 * zeta);
                else t = copysign(1.0, zeta) / (fabs(zeta) + sqrt(zeta * zeta + 1.0));
                double cs = 1.0 / sqrt(1.0 + t * t), sn = cs * t;
                for (qintp i = 0; i < m; i++) {
                    double xp = wpcol[i], xq = wqcol[i];
                    wpcol[i] = cs * xp - sn * xq; wqcol[i] = sn * xp + cs * xq;
                }
                double *vpcol = vp + p * n, *vqcol = vp + q * n;
                for (qintp i = 0; i < n; i++) {
                    double xp = vpcol[i], xq = vqcol[i];
                    vpcol[i] = cs * xp - sn * xq; vqcol[i] = sn * xp + cs * xq;
                }
            }
        }
        if (!rotated) break;
    }
    transpose_inplace(wp, n, m, visited);
    transpose_inplace(vp, n, n, NULL);
    Py_END_ALLOW_THREADS
    PyMem_Free(visited);
    return Py_BuildValue("NNn", w, v, sweeps);
}

PyMethodDef qaccel_direct_methods[] = {
    {"cholesky", (PyCFunction)py_cholesky, METH_VARARGS | METH_KEYWORDS, "Blocked lower Cholesky factorisation."},
    {"plu", (PyCFunction)py_plu, METH_VARARGS | METH_KEYWORDS, "Blocked LU with partial pivoting."},
    {"householder_qr", (PyCFunction)py_householder, METH_VARARGS | METH_KEYWORDS, "Blocked Householder QR; optionally build only reduced Q."},
    {"givens_qr", (PyCFunction)py_givens, METH_VARARGS | METH_KEYWORDS, "QR using Givens rotations."},
    {"svd_jacobi", (PyCFunction)py_svd_jacobi, METH_VARARGS | METH_KEYWORDS, "One-sided Jacobi SVD."},
    {"qr_least_squares", (PyCFunction)py_least_squares, METH_VARARGS | METH_KEYWORDS, "Solve least squares without constructing Q."},
    {"forward_substitution", (PyCFunction)py_forward, METH_VARARGS | METH_KEYWORDS, "Lower triangular solve."},
    {"back_substitution", (PyCFunction)py_back, METH_VARARGS | METH_KEYWORDS, "Upper triangular or transposed lower triangular solve."},
    {"is_symmetric", (PyCFunction)py_symmetric, METH_VARARGS | METH_KEYWORDS, "Streaming symmetry test without matrix temporaries."},
    {NULL, NULL, 0, NULL}
};
