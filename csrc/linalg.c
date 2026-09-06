/* Dense linear algebra: matrix multiply, factorisations, eigenproblems, SVD.
 *
 * The kernels here are the ones the library leans on hardest, so `gemm` is
 * blocked and packed rather than a triple loop, and the factorisations are
 * right-looking so their inner loops run over contiguous rows.
 */
#include "qnp.h"
#include <stdlib.h>

typedef unsigned char qbool;

/* ---------------------------------------------------------------- gemm */

#define MR 4
#define NR 8
#define MC 128
#define KC 192
#define NC 512

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define QNP_HAVE_X86 1
#endif

/* Portable micro-kernel: a MR x NR tile of C accumulated over `kc` steps.
 * The packed panels are zero-padded, so edge tiles compute the full block and
 * store only the part that exists. */
static void micro_kernel_ref(const double *Ap, const double *Bp, double *C, qintp ldc,
                             qintp kc, int mr, int nr, int accumulate) {
    double c[MR][NR];
    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NR; j++) c[i][j] = 0.0;
    for (qintp p = 0; p < kc; p++) {
        const double *b = Bp + p * NR;
        const double *a = Ap + p * MR;
        for (int i = 0; i < MR; i++) {
            double av = a[i];
            for (int j = 0; j < NR; j++) c[i][j] += av * b[j];
        }
    }
    for (int i = 0; i < mr; i++) {
        double *row = C + i * ldc;
        if (accumulate) for (int j = 0; j < nr; j++) row[j] += c[i][j];
        else for (int j = 0; j < nr; j++) row[j] = c[i][j];
    }
}

#ifdef QNP_HAVE_X86
/* The same tile with FMA: eight accumulators hold 4x8 doubles across the whole
 * k loop, so each step is two loads, four broadcasts and eight fused
 * multiply-adds. This is where the throughput of the whole package's matrix
 * products comes from. */
__attribute__((target("avx2,fma")))
static void micro_kernel_avx2(const double *Ap, const double *Bp, double *C, qintp ldc,
                              qintp kc, int mr, int nr, int accumulate) {
    __m256d c0 = _mm256_setzero_pd(), c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd(), c3 = _mm256_setzero_pd();
    __m256d c4 = _mm256_setzero_pd(), c5 = _mm256_setzero_pd();
    __m256d c6 = _mm256_setzero_pd(), c7 = _mm256_setzero_pd();
    for (qintp p = 0; p < kc; p++) {
        const double *b = Bp + p * NR;
        const double *a = Ap + p * MR;
        __m256d b0 = _mm256_loadu_pd(b);
        __m256d b1 = _mm256_loadu_pd(b + 4);
        __m256d av = _mm256_set1_pd(a[0]);
        c0 = _mm256_fmadd_pd(av, b0, c0);
        c1 = _mm256_fmadd_pd(av, b1, c1);
        av = _mm256_set1_pd(a[1]);
        c2 = _mm256_fmadd_pd(av, b0, c2);
        c3 = _mm256_fmadd_pd(av, b1, c3);
        av = _mm256_set1_pd(a[2]);
        c4 = _mm256_fmadd_pd(av, b0, c4);
        c5 = _mm256_fmadd_pd(av, b1, c5);
        av = _mm256_set1_pd(a[3]);
        c6 = _mm256_fmadd_pd(av, b0, c6);
        c7 = _mm256_fmadd_pd(av, b1, c7);
    }
    if (mr == MR && nr == NR) {
        __m256d *acc[8] = {&c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7};
        for (int i = 0; i < MR; i++) {
            double *row = C + i * ldc;
            __m256d lo = *acc[2 * i], hi = *acc[2 * i + 1];
            if (accumulate) {
                lo = _mm256_add_pd(_mm256_loadu_pd(row), lo);
                hi = _mm256_add_pd(_mm256_loadu_pd(row + 4), hi);
            }
            _mm256_storeu_pd(row, lo);
            _mm256_storeu_pd(row + 4, hi);
        }
        return;
    }
    double tile[MR][NR];
    _mm256_storeu_pd(&tile[0][0], c0);
    _mm256_storeu_pd(&tile[0][4], c1);
    _mm256_storeu_pd(&tile[1][0], c2);
    _mm256_storeu_pd(&tile[1][4], c3);
    _mm256_storeu_pd(&tile[2][0], c4);
    _mm256_storeu_pd(&tile[2][4], c5);
    _mm256_storeu_pd(&tile[3][0], c6);
    _mm256_storeu_pd(&tile[3][4], c7);
    for (int i = 0; i < mr; i++) {
        double *row = C + i * ldc;
        if (accumulate) for (int j = 0; j < nr; j++) row[j] += tile[i][j];
        else for (int j = 0; j < nr; j++) row[j] = tile[i][j];
    }
}

static int cpu_has_fma(void) {
    static int cached = -1;
    if (cached < 0)
        cached = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return cached;
}
#endif

static void micro_kernel(const double *Ap, const double *Bp, double *C, qintp ldc,
                         qintp kc, int mr, int nr, int accumulate) {
#ifdef QNP_HAVE_X86
    if (cpu_has_fma()) {
        micro_kernel_avx2(Ap, Bp, C, ldc, kc, mr, nr, accumulate);
        return;
    }
#endif
    micro_kernel_ref(Ap, Bp, C, ldc, kc, mr, nr, accumulate);
}

static void pack_a(const double *A, qintp lda, double *Ap, qintp mc, qintp kc) {
    for (qintp i = 0; i < mc; i += MR) {
        int mr = (int)((mc - i < MR) ? mc - i : MR);
        double *dst = Ap + i * kc;
        for (qintp p = 0; p < kc; p++) {
            for (int r = 0; r < mr; r++) dst[p * MR + r] = A[(i + r) * lda + p];
            for (int r = mr; r < MR; r++) dst[p * MR + r] = 0.0;
        }
    }
}

static void pack_b(const double *B, qintp ldb, double *Bp, qintp kc, qintp nc) {
    for (qintp j = 0; j < nc; j += NR) {
        int nr = (int)((nc - j < NR) ? nc - j : NR);
        double *dst = Bp + j * kc;
        for (qintp p = 0; p < kc; p++) {
            const double *src = B + p * ldb + j;
            for (int r = 0; r < nr; r++) dst[p * NR + r] = src[r];
            for (int r = nr; r < NR; r++) dst[p * NR + r] = 0.0;
        }
    }
}

/* Inner product of two contiguous vectors, summed pairwise for accuracy. */
static double dot_f64(const double *a, const double *b, qintp n) {
    if (n < 8) {
        double s = 0.0;
        for (qintp i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }
    if (n <= 128) {
        double r[8];
        for (int i = 0; i < 8; i++) r[i] = a[i] * b[i];
        qintp i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; k++) r[k] += a[i + k] * b[i + k];
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }
    qintp half = n / 2;
    half -= half % 8;
    return dot_f64(a, b, half) + dot_f64(a + half, b + half, n - half);
}

/* c = a * B for a single left-hand row: one scaled row of B per step, which
 * streams B once and keeps the whole result in cache. */
static void gevm_f64(const double *a, const double *B, double *c, qintp n, qintp k) {
    memset(c, 0, (size_t)n * sizeof(double));
    for (qintp p = 0; p < k; p++) {
        double av = a[p];
        if (av == 0.0) continue;
        const double *brow = B + p * n;
        for (qintp j = 0; j < n; j++) c[j] += av * brow[j];
    }
}

/* C = A * b for a single right-hand column: one dot product per row. */
static void gemv_f64(const double *A, const double *b, double *c, qintp m, qintp k) {
    for (qintp i = 0; i < m; i++) c[i] = dot_f64(A + i * k, b, k);
}

/* Straight triple loop, in the order that streams both B and C contiguously.
 * Faster than the blocked path until the operands stop fitting in cache. */
static void gemm_small(const double *A, const double *B, double *C,
                       qintp m, qintp n, qintp k) {
    memset(C, 0, (size_t)(m * n) * sizeof(double));
    for (qintp i = 0; i < m; i++) {
        double *crow = C + i * n;
        const double *arow = A + i * k;
        for (qintp p = 0; p < k; p++) {
            double a = arow[p];
            if (a == 0.0) continue;
            const double *brow = B + p * n;
            for (qintp j = 0; j < n; j++) crow[j] += a * brow[j];
        }
    }
}

void qnp_gemm_f64(const double *A, const double *B, double *C,
                  qintp m, qintp n, qintp k) {
    if (m == 0 || n == 0) return;
    if (k == 0) { memset(C, 0, (size_t)(m * n) * sizeof(double)); return; }
    if (n == 1) { gemv_f64(A, B, C, m, k); return; }
    if (m == 1) { gevm_f64(A, B, C, n, k); return; }
    if ((double)m * (double)n * (double)k < 32768.0) {
        gemm_small(A, B, C, m, n, k);
        return;
    }
    /* The panels are sized to the problem, so a modest product does not pay
     * for a megabyte of scratch it will never fill. */
    qintp kc_max = k < KC ? k : KC;
    qintp nc_max = n < NC ? n : NC;
    qintp mc_max = m < MC ? m : MC;
    double *Ap = (double *)malloc((size_t)(mc_max + MR) * (size_t)kc_max * sizeof(double));
    double *Bp = (double *)malloc((size_t)(nc_max + NR) * (size_t)kc_max * sizeof(double));
    if (Ap == NULL || Bp == NULL) {
        free(Ap); free(Bp);
        gemm_small(A, B, C, m, n, k);
        return;
    }
    for (qintp jc = 0; jc < n; jc += NC) {
        qintp nc = (n - jc < NC) ? n - jc : NC;
        for (qintp pc = 0; pc < k; pc += KC) {
            qintp kc = (k - pc < KC) ? k - pc : KC;
            int accumulate = (pc != 0);
            pack_b(B + pc * n + jc, n, Bp, kc, nc);
            for (qintp ic = 0; ic < m; ic += MC) {
                qintp mc = (m - ic < MC) ? m - ic : MC;
                pack_a(A + ic * k + pc, k, Ap, mc, kc);
                for (qintp jr = 0; jr < nc; jr += NR) {
                    int nr = (int)((nc - jr < NR) ? nc - jr : NR);
                    for (qintp ir = 0; ir < mc; ir += MR) {
                        int mr = (int)((mc - ir < MR) ? mc - ir : MR);
                        micro_kernel(Ap + ir * kc, Bp + jr * kc,
                                     C + (ic + ir) * n + jc + jr, n, kc,
                                     mr, nr, accumulate);
                    }
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}

static void gemm_c128(const qcomplex *A, const qcomplex *B, qcomplex *C,
                      qintp m, qintp n, qintp k) {
    memset(C, 0, (size_t)(m * n) * sizeof(qcomplex));
    for (qintp i = 0; i < m; i++) {
        qcomplex *crow = C + i * n;
        const qcomplex *arow = A + i * k;
        for (qintp p = 0; p < k; p++) {
            qcomplex a = arow[p];
            if (a.re == 0.0 && a.im == 0.0) continue;
            const qcomplex *brow = B + p * n;
            for (qintp j = 0; j < n; j++) {
                crow[j].re += a.re * brow[j].re - a.im * brow[j].im;
                crow[j].im += a.re * brow[j].im + a.im * brow[j].re;
            }
        }
    }
}

static void gemm_i64(const int64_t *A, const int64_t *B, int64_t *C,
                     qintp m, qintp n, qintp k) {
    memset(C, 0, (size_t)(m * n) * sizeof(int64_t));
    for (qintp i = 0; i < m; i++)
        for (qintp p = 0; p < k; p++) {
            int64_t a = A[i * k + p];
            if (!a) continue;
            for (qintp j = 0; j < n; j++) C[i * n + j] += a * B[p * n + j];
        }
}

PyObject *qnp_matmul(PyObject *ao, PyObject *bo) {
    QArray *a = qnp_from_any(ao, -1, 0);
    if (a == NULL) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
        return NULL;
    }
    QArray *b = qnp_from_any(bo, -1, 0);
    if (b == NULL) {
        Py_DECREF(a);
        if (PyErr_ExceptionMatches(PyExc_TypeError)) { PyErr_Clear(); Py_RETURN_NOTIMPLEMENTED; }
        return NULL;
    }
    if (a->nd == 0 || b->nd == 0) {
        Py_DECREF(a); Py_DECREF(b);
        PyErr_SetString(PyExc_ValueError,
                        "matmul: input operand does not have enough dimensions");
        return NULL;
    }
    int dt = qnp_promote(a->dtype, b->dtype);
    if (dt == QNP_BOOL) dt = QNP_INT64;
    int a1 = (a->nd == 1), b1 = (b->nd == 1);
    if (a->nd > 2 || b->nd > 2) {
        Py_DECREF(a); Py_DECREF(b);
        PyErr_SetString(PyExc_NotImplementedError,
                        "matmul over stacks of matrices is not supported");
        return NULL;
    }
    qintp m = a1 ? 1 : a->shape[0];
    qintp ka = a1 ? a->shape[0] : a->shape[1];
    qintp kb = b1 ? b->shape[0] : b->shape[0];
    qintp n = b1 ? 1 : b->shape[1];
    if (ka != kb) {
        PyErr_Format(PyExc_ValueError,
                     "matmul: Input operand 1 has a mismatch in its core dimension 0 "
                     "(size %zd is different from %zd)", kb, ka);
        Py_DECREF(a); Py_DECREF(b);
        return NULL;
    }
    QArray *fa = qnp_astype(a, dt, 0), *fb = qnp_astype(b, dt, 0);
    Py_DECREF(a); Py_DECREF(b);
    if (fa == NULL || fb == NULL) { Py_XDECREF(fa); Py_XDECREF(fb); return NULL; }
    QArray *ca = qnp_ascontiguous(fa), *cb = qnp_ascontiguous(fb);
    Py_DECREF(fa); Py_DECREF(fb);
    if (ca == NULL || cb == NULL) { Py_XDECREF(ca); Py_XDECREF(cb); return NULL; }
    qintp shape[2];
    int nd = 0;
    if (!a1) shape[nd++] = m;
    if (!b1) shape[nd++] = n;
    QArray *out = qnp_new(nd, shape, dt);
    if (out == NULL) { Py_DECREF(ca); Py_DECREF(cb); return NULL; }
    Py_BEGIN_ALLOW_THREADS
    if (dt == QNP_FLOAT64)
        qnp_gemm_f64((const double *)ca->data, (const double *)cb->data,
                     (double *)out->data, m, n, ka);
    else if (dt == QNP_COMPLEX128)
        gemm_c128((const qcomplex *)ca->data, (const qcomplex *)cb->data,
                  (qcomplex *)out->data, m, n, ka);
    else
        gemm_i64((const int64_t *)ca->data, (const int64_t *)cb->data,
                 (int64_t *)out->data, m, n, ka);
    Py_END_ALLOW_THREADS
    Py_DECREF(ca);
    Py_DECREF(cb);
    return qnp_wrap_scalar_or_array(out);
}

/* ---------------------------------------------------------------- LU */

static int lu_decomp_d(double *A, int n, int *piv, int *sign) {
    *sign = 1;
    int singular = 0;
    for (int k = 0; k < n; k++) {
        int p = k;
        double best = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = fabs(A[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        piv[k] = p;
        if (p != k) {
            for (int j = 0; j < n; j++) {
                double t = A[k * n + j];
                A[k * n + j] = A[p * n + j];
                A[p * n + j] = t;
            }
            *sign = -*sign;
        }
        double pivot = A[k * n + k];
        if (pivot == 0.0) { singular = 1; continue; }
        for (int i = k + 1; i < n; i++) {
            double l = A[i * n + k] / pivot;
            A[i * n + k] = l;
            if (l == 0.0) continue;
            const double *row = A + k * n;
            double *dst = A + i * n;
            for (int j = k + 1; j < n; j++) dst[j] -= l * row[j];
        }
    }
    return singular ? -1 : 0;
}

static void lu_solve_d(const double *LU, int n, const int *piv, double *B, int nrhs) {
    /* The factorisation swapped whole rows, multipliers included, so the
     * permutation has to be applied to the right-hand side in full before any
     * elimination -- interleaving the two uses multipliers from the wrong row. */
    for (int k = 0; k < n; k++) {
        int p = piv[k];
        if (p == k) continue;
        for (int j = 0; j < nrhs; j++) {
            double t = B[k * nrhs + j];
            B[k * nrhs + j] = B[p * nrhs + j];
            B[p * nrhs + j] = t;
        }
    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            double l = LU[i * n + k];
            if (l == 0.0) continue;
            for (int j = 0; j < nrhs; j++) B[i * nrhs + j] -= l * B[k * nrhs + j];
        }
    }
    for (int k = n - 1; k >= 0; k--) {
        double d = LU[k * n + k];
        for (int j = 0; j < nrhs; j++) {
            double s = B[k * nrhs + j];
            for (int i = k + 1; i < n; i++) s -= LU[k * n + i] * B[i * nrhs + j];
            B[k * nrhs + j] = s / d;
        }
    }
}

static int lu_decomp_c(qcomplex *A, int n, int *piv, int *sign) {
    *sign = 1;
    int singular = 0;
    for (int k = 0; k < n; k++) {
        int p = k;
        double best = qc_abs(A[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = qc_abs(A[i * n + k]);
            if (v > best) { best = v; p = i; }
        }
        piv[k] = p;
        if (p != k) {
            for (int j = 0; j < n; j++) {
                qcomplex t = A[k * n + j];
                A[k * n + j] = A[p * n + j];
                A[p * n + j] = t;
            }
            *sign = -*sign;
        }
        qcomplex pivot = A[k * n + k];
        if (pivot.re == 0.0 && pivot.im == 0.0) { singular = 1; continue; }
        for (int i = k + 1; i < n; i++) {
            qcomplex l = qc_div(A[i * n + k], pivot);
            A[i * n + k] = l;
            for (int j = k + 1; j < n; j++)
                A[i * n + j] = qc_sub(A[i * n + j], qc_mul(l, A[k * n + j]));
        }
    }
    return singular ? -1 : 0;
}

static void lu_solve_c(const qcomplex *LU, int n, const int *piv, qcomplex *B, int nrhs) {
    for (int k = 0; k < n; k++) {
        int p = piv[k];
        if (p == k) continue;
        for (int j = 0; j < nrhs; j++) {
            qcomplex t = B[k * nrhs + j];
            B[k * nrhs + j] = B[p * nrhs + j];
            B[p * nrhs + j] = t;
        }
    }
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            qcomplex l = LU[i * n + k];
            for (int j = 0; j < nrhs; j++)
                B[i * nrhs + j] = qc_sub(B[i * nrhs + j], qc_mul(l, B[k * nrhs + j]));
        }
    }
    for (int k = n - 1; k >= 0; k--) {
        qcomplex d = LU[k * n + k];
        for (int j = 0; j < nrhs; j++) {
            qcomplex s = B[k * nrhs + j];
            for (int i = k + 1; i < n; i++)
                s = qc_sub(s, qc_mul(LU[k * n + i], B[i * nrhs + j]));
            B[k * nrhs + j] = qc_div(s, d);
        }
    }
}

/* ---- helpers for the Python boundary ---------------------------------- */

static QArray *require_matrix(PyObject *obj, int dtype_floor, const char *name) {
    QArray *a = qnp_from_any(obj, -1, 0);
    if (a == NULL) return NULL;
    if (a->nd != 2) {
        PyErr_Format(QNP_LinAlgError,
                     "%s: expected a 2-dimensional array, got %d dimensions", name, a->nd);
        Py_DECREF(a);
        return NULL;
    }
    int dt = a->dtype < dtype_floor ? dtype_floor : a->dtype;
    QArray *f = qnp_astype(a, dt, 0);
    Py_DECREF(a);
    if (f == NULL) return NULL;
    QArray *c = qnp_ascontiguous(f);
    Py_DECREF(f);
    return c;
}

static QArray *require_square(PyObject *obj, const char *name, int *n) {
    QArray *a = require_matrix(obj, QNP_FLOAT64, name);
    if (a == NULL) return NULL;
    if (a->shape[0] != a->shape[1]) {
        PyErr_Format(QNP_LinAlgError, "%s: last 2 dimensions of the array must be square", name);
        Py_DECREF(a);
        return NULL;
    }
    *n = (int)a->shape[0];
    return a;
}

static PyObject *py_solve(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *ao, *bo;
    if (!PyArg_ParseTuple(args, "OO:solve", &ao, &bo)) return NULL;
    QArray *a0 = qnp_from_any(ao, -1, 0);
    QArray *b0 = qnp_from_any(bo, -1, 0);
    if (a0 == NULL || b0 == NULL) { Py_XDECREF(a0); Py_XDECREF(b0); return NULL; }
    if (a0->nd != 2 || a0->shape[0] != a0->shape[1]) {
        PyErr_SetString(QNP_LinAlgError, "solve: the coefficient matrix must be square");
        Py_DECREF(a0); Py_DECREF(b0);
        return NULL;
    }
    int n = (int)a0->shape[0];
    int dt = qnp_promote(qnp_promote(a0->dtype, b0->dtype), QNP_FLOAT64);
    /* Both operands are copied: the factorisation and the solve overwrite. */
    QArray *a = qnp_astype(a0, dt, 1);
    Py_DECREF(a0);
    if (a == NULL) { Py_DECREF(b0); return NULL; }
    QArray *ac = qnp_ascontiguous(a);
    Py_DECREF(a);
    if (ac == NULL) { Py_DECREF(b0); return NULL; }
    a = ac;
    int vector_rhs = (b0->nd == 1);
    if (b0->nd != 1 && b0->nd != 2) {
        PyErr_SetString(QNP_LinAlgError, "solve: the right-hand side must be 1- or 2-dimensional");
        Py_DECREF(a); Py_DECREF(b0);
        return NULL;
    }
    if (b0->shape[0] != n) {
        PyErr_Format(QNP_LinAlgError,
                     "solve: incompatible dimensions -- matrix is %dx%d, right-hand "
                     "side has leading dimension %zd", n, n, b0->shape[0]);
        Py_DECREF(a); Py_DECREF(b0);
        return NULL;
    }
    QArray *bf = qnp_astype(b0, dt, 1);
    Py_DECREF(b0);
    if (bf == NULL) { Py_DECREF(a); return NULL; }
    QArray *b = qnp_ascontiguous(bf);
    if (b != bf) Py_DECREF(bf);
    if (b == NULL) { Py_DECREF(a); return NULL; }
    int nrhs = vector_rhs ? 1 : (int)b->shape[1];
    int *piv = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (piv == NULL) { Py_DECREF(a); Py_DECREF(b); return PyErr_NoMemory(); }
    int sign, rc;
    Py_BEGIN_ALLOW_THREADS
    if (dt == QNP_COMPLEX128) {
        rc = lu_decomp_c((qcomplex *)a->data, n, piv, &sign);
        if (rc == 0) lu_solve_c((const qcomplex *)a->data, n, piv, (qcomplex *)b->data, nrhs);
    } else {
        rc = lu_decomp_d((double *)a->data, n, piv, &sign);
        if (rc == 0) lu_solve_d((const double *)a->data, n, piv, (double *)b->data, nrhs);
    }
    Py_END_ALLOW_THREADS
    free(piv);
    Py_DECREF(a);
    if (rc < 0) {
        Py_DECREF(b);
        PyErr_SetString(QNP_LinAlgError, "Singular matrix");
        return NULL;
    }
    return (PyObject *)b;
}

static PyObject *py_inv(PyObject *self, PyObject *arg) {
    (void)self;
    int n;
    QArray *a0 = require_square(arg, "inv", &n);
    if (a0 == NULL) return NULL;
    int dt = a0->dtype;
    QArray *a = qnp_astype(a0, dt, 1);
    Py_DECREF(a0);
    if (a == NULL) return NULL;
    qintp shape[2] = {n, n};
    QArray *out = qnp_new(2, shape, dt);
    if (out == NULL) { Py_DECREF(a); return NULL; }
    memset(out->data, 0, (size_t)(n * n) * (size_t)QNP_ITEMSIZE(dt));
    for (int i = 0; i < n; i++) {
        if (dt == QNP_COMPLEX128) ((qcomplex *)out->data)[i * n + i] = qc(1.0, 0.0);
        else ((double *)out->data)[i * n + i] = 1.0;
    }
    int *piv = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (piv == NULL) { Py_DECREF(a); Py_DECREF(out); return PyErr_NoMemory(); }
    int sign, rc;
    Py_BEGIN_ALLOW_THREADS
    if (dt == QNP_COMPLEX128) {
        rc = lu_decomp_c((qcomplex *)a->data, n, piv, &sign);
        if (rc == 0) lu_solve_c((const qcomplex *)a->data, n, piv, (qcomplex *)out->data, n);
    } else {
        rc = lu_decomp_d((double *)a->data, n, piv, &sign);
        if (rc == 0) lu_solve_d((const double *)a->data, n, piv, (double *)out->data, n);
    }
    Py_END_ALLOW_THREADS
    free(piv);
    Py_DECREF(a);
    if (rc < 0) {
        Py_DECREF(out);
        PyErr_SetString(QNP_LinAlgError, "Singular matrix");
        return NULL;
    }
    return (PyObject *)out;
}

static PyObject *det_common(PyObject *arg, int want_slogdet) {
    int n;
    QArray *a0 = require_square(arg, want_slogdet ? "slogdet" : "det", &n);
    if (a0 == NULL) return NULL;
    int dt = a0->dtype;
    QArray *a = qnp_astype(a0, dt, 1);
    Py_DECREF(a0);
    if (a == NULL) return NULL;
    int *piv = (int *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (piv == NULL) { Py_DECREF(a); return PyErr_NoMemory(); }
    int sign;
    int rc;
    if (dt == QNP_COMPLEX128) rc = lu_decomp_c((qcomplex *)a->data, n, piv, &sign);
    else rc = lu_decomp_d((double *)a->data, n, piv, &sign);
    free(piv);
    PyObject *result = NULL;
    if (dt == QNP_COMPLEX128) {
        qcomplex det = qc((double)sign, 0.0);
        double logabs = 0.0;
        qcomplex phase = qc(1.0, 0.0);
        for (int i = 0; i < n; i++) {
            qcomplex d = ((qcomplex *)a->data)[i * n + i];
            det = qc_mul(det, d);
            double m = qc_abs(d);
            if (m == 0.0) { logabs = -INFINITY; phase = qc(0.0, 0.0); break; }
            logabs += log(m);
            phase = qc_mul(phase, qc_div(d, qc(m, 0.0)));
        }
        if (want_slogdet) {
            phase = qc_mul(phase, qc((double)sign, 0.0));
            result = Py_BuildValue("(NN)", PyComplex_FromDoubles(phase.re, phase.im),
                                   PyFloat_FromDouble(logabs));
        } else {
            result = PyComplex_FromDoubles(det.re, det.im);
        }
    } else {
        double det = (double)sign;
        double logabs = 0.0;
        int zero = 0;
        for (int i = 0; i < n; i++) {
            double d = ((double *)a->data)[i * n + i];
            det *= d;
            if (d == 0.0) zero = 1;
            else {
                logabs += log(fabs(d));
                if (d < 0.0) sign = -sign;
            }
        }
        if (want_slogdet) {
            result = Py_BuildValue("(dd)", zero ? 0.0 : (double)sign,
                                   zero ? -INFINITY : logabs);
        } else {
            result = PyFloat_FromDouble(det);
        }
    }
    (void)rc;
    Py_DECREF(a);
    return result;
}

static PyObject *py_det(PyObject *self, PyObject *arg) { (void)self; return det_common(arg, 0); }
static PyObject *py_slogdet(PyObject *self, PyObject *arg) { (void)self; return det_common(arg, 1); }

static PyObject *py_cholesky(PyObject *self, PyObject *arg) {
    (void)self;
    int n;
    QArray *a0 = require_square(arg, "cholesky", &n);
    if (a0 == NULL) return NULL;
    if (a0->dtype == QNP_COMPLEX128) {
        Py_DECREF(a0);
        PyErr_SetString(PyExc_NotImplementedError, "cholesky of a complex matrix");
        return NULL;
    }
    QArray *out = qnp_astype(a0, QNP_FLOAT64, 1);
    Py_DECREF(a0);
    if (out == NULL) return NULL;
    double *L = (double *)out->data;
    int failed = 0;
    Py_BEGIN_ALLOW_THREADS
    for (int i = 0; i < n && !failed; i++) {
        for (int j = 0; j <= i; j++) {
            double s = L[i * n + j];
            const double *ri = L + i * n, *rj = L + j * n;
            for (int k = 0; k < j; k++) s -= ri[k] * rj[k];
            if (i == j) {
                if (s <= 0.0) { failed = 1; break; }
                L[i * n + j] = sqrt(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
        for (int j = i + 1; j < n; j++) L[i * n + j] = 0.0;
    }
    Py_END_ALLOW_THREADS
    if (failed) {
        Py_DECREF(out);
        PyErr_SetString(QNP_LinAlgError, "Matrix is not positive definite");
        return NULL;
    }
    return (PyObject *)out;
}

/* ------------------------------------------------- symmetric eigenproblem */

/* Householder reduction to tridiagonal form, EISPACK `tred2`.  `z` holds the
 * matrix on entry and the accumulated transformation on exit. */
static void tred2(double *z, int n, double *d, double *e) {
    for (int i = n - 1; i > 0; i--) {
        int l = i - 1;
        double h = 0.0, scale = 0.0;
        if (l > 0) {
            for (int k = 0; k <= l; k++) scale += fabs(z[i * n + k]);
            if (scale == 0.0) {
                e[i] = z[i * n + l];
            } else {
                for (int k = 0; k <= l; k++) {
                    z[i * n + k] /= scale;
                    h += z[i * n + k] * z[i * n + k];
                }
                double f = z[i * n + l];
                double g = (f >= 0.0) ? -sqrt(h) : sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                z[i * n + l] = f - g;
                f = 0.0;
                for (int j = 0; j <= l; j++) {
                    z[j * n + i] = z[i * n + j] / h;
                    g = 0.0;
                    for (int k = 0; k <= j; k++) g += z[j * n + k] * z[i * n + k];
                    for (int k = j + 1; k <= l; k++) g += z[k * n + j] * z[i * n + k];
                    e[j] = g / h;
                    f += e[j] * z[i * n + j];
                }
                double hh = f / (h + h);
                for (int j = 0; j <= l; j++) {
                    f = z[i * n + j];
                    e[j] = g = e[j] - hh * f;
                    for (int k = 0; k <= j; k++)
                        z[j * n + k] -= (f * e[k] + g * z[i * n + k]);
                }
            }
        } else {
            e[i] = z[i * n + l];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for (int i = 0; i < n; i++) {
        int l = i - 1;
        if (d[i] != 0.0) {
            for (int j = 0; j <= l; j++) {
                double g = 0.0;
                for (int k = 0; k <= l; k++) g += z[i * n + k] * z[k * n + j];
                for (int k = 0; k <= l; k++) z[k * n + j] -= g * z[k * n + i];
            }
        }
        d[i] = z[i * n + i];
        z[i * n + i] = 1.0;
        for (int j = 0; j <= l; j++) { z[j * n + i] = 0.0; z[i * n + j] = 0.0; }
    }
}

static double pythag(double a, double b) {
    double absa = fabs(a), absb = fabs(b);
    if (absa > absb) return absa * sqrt(1.0 + (absb / absa) * (absb / absa));
    if (absb == 0.0) return 0.0;
    return absb * sqrt(1.0 + (absa / absb) * (absa / absb));
}

/* Implicit-shift QL on the tridiagonal, EISPACK `tql2`. */
static int tql2(double *d, double *e, double *z, int n) {
    for (int i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0.0;
    for (int l = 0; l < n; l++) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = fabs(d[m]) + fabs(d[m + 1]);
                if (fabs(e[m]) <= 1e-300 + 2.3e-16 * dd) break;
            }
            if (m != l) {
                if (iter++ == 50) return -1;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = pythag(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? fabs(r) : -fabs(r)));
                double s = 1.0, c = 1.0, p = 0.0;
                for (int i = m - 1; i >= l; i--) {
                    double f = s * e[i];
                    double b = c * e[i];
                    e[i + 1] = (r = pythag(f, g));
                    if (r == 0.0) { d[i + 1] -= p; e[m] = 0.0; break; }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i + 1] = g + (p = s * r);
                    g = c * r - b;
                    for (int k = 0; k < n; k++) {
                        f = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
    /* Ascending order, matching NumPy's eigh. */
    for (int i = 0; i < n - 1; i++) {
        int k = i;
        double p = d[i];
        for (int j = i + 1; j < n; j++) if (d[j] < p) { k = j; p = d[j]; }
        if (k != i) {
            d[k] = d[i];
            d[i] = p;
            for (int j = 0; j < n; j++) {
                double t = z[j * n + i];
                z[j * n + i] = z[j * n + k];
                z[j * n + k] = t;
            }
        }
    }
    return 0;
}

static PyObject *eigh_common(PyObject *arg, int want_vectors) {
    int n;
    QArray *a0 = require_square(arg, want_vectors ? "eigh" : "eigvalsh", &n);
    if (a0 == NULL) return NULL;
    if (a0->dtype == QNP_COMPLEX128) {
        Py_DECREF(a0);
        PyErr_SetString(PyExc_NotImplementedError, "eigh of a complex matrix");
        return NULL;
    }
    QArray *z = qnp_astype(a0, QNP_FLOAT64, 1);
    Py_DECREF(a0);
    if (z == NULL) return NULL;
    /* Only the lower triangle is referenced, as in NumPy's default UPLO. */
    double *zp = (double *)z->data;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) zp[i * n + j] = zp[j * n + i];
    qintp nn = n;
    QArray *w = qnp_new(1, &nn, QNP_FLOAT64);
    double *e = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (w == NULL || e == NULL) {
        Py_XDECREF(w); free(e); Py_DECREF(z);
        return PyErr_NoMemory();
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    tred2(zp, n, (double *)w->data, e);
    rc = tql2((double *)w->data, e, zp, n);
    Py_END_ALLOW_THREADS
    free(e);
    if (rc < 0) {
        Py_DECREF(w); Py_DECREF(z);
        PyErr_SetString(QNP_LinAlgError, "Eigenvalue computation did not converge");
        return NULL;
    }
    if (!want_vectors) {
        Py_DECREF(z);
        return (PyObject *)w;
    }
    /* An eigenvector is only defined up to sign, and LAPACK's choice varies
     * with the build.  Fixing the largest-magnitude entry of each column to be
     * positive makes the decomposition reproducible everywhere. */
    for (int j = 0; j < n; j++) {
        int pivot = 0;
        double best = -1.0;
        for (int i = 0; i < n; i++) {
            double m = fabs(zp[i * n + j]);
            if (m > best) { best = m; pivot = i; }
        }
        if (zp[pivot * n + j] < 0.0)
            for (int i = 0; i < n; i++) zp[i * n + j] = -zp[i * n + j];
    }
    return Py_BuildValue("(NN)", (PyObject *)w, (PyObject *)z);
}

static PyObject *py_eigh(PyObject *self, PyObject *arg) { (void)self; return eigh_common(arg, 1); }
static PyObject *py_eigvalsh(PyObject *self, PyObject *arg) { (void)self; return eigh_common(arg, 0); }

/* ------------------------------------------------ general eigenproblem */

/* Householder reduction to upper Hessenberg form, accumulating the
 * transformation in `q` when it is not NULL. */
static void hessenberg(double *a, int n, double *q) {
    double *v = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (v == NULL) return;
    if (q != NULL) {
        memset(q, 0, (size_t)(n * n) * sizeof(double));
        for (int i = 0; i < n; i++) q[i * n + i] = 1.0;
    }
    for (int k = 1; k < n - 1; k++) {
        double scale = 0.0;
        for (int i = k; i < n; i++) scale += fabs(a[i * n + k - 1]);
        if (scale == 0.0) continue;
        double h = 0.0;
        for (int i = k; i < n; i++) {
            v[i] = a[i * n + k - 1] / scale;
            h += v[i] * v[i];
        }
        double g = (v[k] >= 0.0) ? -sqrt(h) : sqrt(h);
        h -= v[k] * g;
        v[k] -= g;
        for (int j = k; j < n; j++) {
            double f = 0.0;
            for (int i = k; i < n; i++) f += v[i] * a[i * n + j];
            f /= h;
            for (int i = k; i < n; i++) a[i * n + j] -= f * v[i];
        }
        for (int i = 0; i < n; i++) {
            double f = 0.0;
            for (int j = k; j < n; j++) f += a[i * n + j] * v[j];
            f /= h;
            for (int j = k; j < n; j++) a[i * n + j] -= f * v[j];
        }
        if (q != NULL) {
            for (int i = 0; i < n; i++) {
                double f = 0.0;
                for (int j = k; j < n; j++) f += q[i * n + j] * v[j];
                f /= h;
                for (int j = k; j < n; j++) q[i * n + j] -= f * v[j];
            }
        }
        a[k * n + k - 1] = scale * g;
        for (int i = k + 1; i < n; i++) a[i * n + k - 1] = 0.0;
    }
    free(v);
}

/* Francis double-shift QR on an upper Hessenberg matrix; eigenvalues only. */
static int hqr(double *a, int n, double *wr, double *wi) {
    double anorm = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = (i - 1 > 0 ? i - 1 : 0); j < n; j++) anorm += fabs(a[i * n + j]);
    int nn = n - 1;
    double t = 0.0;
    while (nn >= 0) {
        int its = 0;
        int l;
        do {
            for (l = nn; l >= 1; l--) {
                double s = fabs(a[(l - 1) * n + l - 1]) + fabs(a[l * n + l]);
                if (s == 0.0) s = anorm;
                if (fabs(a[l * n + l - 1]) + s == s) { a[l * n + l - 1] = 0.0; break; }
            }
            double x = a[nn * n + nn];
            if (l == nn) {
                wr[nn] = x + t;
                wi[nn] = 0.0;
                nn--;
            } else {
                double y = a[(nn - 1) * n + nn - 1];
                double w = a[nn * n + nn - 1] * a[(nn - 1) * n + nn];
                if (l == nn - 1) {
                    double p = 0.5 * (y - x);
                    double q = p * p + w;
                    double z = sqrt(fabs(q));
                    x += t;
                    if (q >= 0.0) {
                        z = p + (p >= 0.0 ? fabs(z) : -fabs(z));
                        wr[nn - 1] = wr[nn] = x + z;
                        if (z) wr[nn] = x - w / z;
                        wi[nn - 1] = wi[nn] = 0.0;
                    } else {
                        wr[nn - 1] = wr[nn] = x + p;
                        wi[nn - 1] = -(wi[nn] = z);
                    }
                    nn -= 2;
                } else {
                    if (its == 60) return -1;
                    double p = 0, q = 0, r = 0, z = 0, s = 0, w2 = w, x2 = x, y2 = y;
                    if (its == 10 || its == 20) {
                        t += x;
                        for (int i = 0; i <= nn; i++) a[i * n + i] -= x;
                        s = fabs(a[nn * n + nn - 1]) + fabs(a[(nn - 1) * n + nn - 2]);
                        y2 = x2 = 0.75 * s;
                        w2 = -0.4375 * s * s;
                    }
                    its++;
                    int m;
                    for (m = nn - 2; m >= l; m--) {
                        z = a[m * n + m];
                        r = x2 - z;
                        s = y2 - z;
                        p = (r * s - w2) / a[(m + 1) * n + m] + a[m * n + m + 1];
                        q = a[(m + 1) * n + m + 1] - z - r - s;
                        r = a[(m + 2) * n + m + 1];
                        s = fabs(p) + fabs(q) + fabs(r);
                        p /= s; q /= s; r /= s;
                        if (m == l) break;
                        double u = fabs(a[m * n + m - 1]) * (fabs(q) + fabs(r));
                        double v = fabs(p) * (fabs(a[(m - 1) * n + m - 1]) +
                                              fabs(z) + fabs(a[(m + 1) * n + m + 1]));
                        if (u + v == v) break;
                    }
                    for (int i = m + 2; i <= nn; i++) {
                        a[i * n + i - 2] = 0.0;
                        if (i != m + 2) a[i * n + i - 3] = 0.0;
                    }
                    for (int k = m; k <= nn - 1; k++) {
                        if (k != m) {
                            p = a[k * n + k - 1];
                            q = a[(k + 1) * n + k - 1];
                            r = (k != nn - 1) ? a[(k + 2) * n + k - 1] : 0.0;
                            x2 = fabs(p) + fabs(q) + fabs(r);
                            if (x2 != 0.0) { p /= x2; q /= x2; r /= x2; }
                        }
                        double sg = sqrt(p * p + q * q + r * r);
                        s = (p >= 0.0) ? sg : -sg;
                        if (s == 0.0) continue;
                        if (k == m) {
                            if (l != m) a[k * n + k - 1] = -a[k * n + k - 1];
                        } else {
                            a[k * n + k - 1] = -s * x2;
                        }
                        p += s;
                        x2 = p / s;
                        double y3 = q / s, z3 = r / s;
                        q /= p;
                        r /= p;
                        for (int j = k; j <= nn; j++) {
                            p = a[k * n + j] + q * a[(k + 1) * n + j];
                            if (k != nn - 1) {
                                p += r * a[(k + 2) * n + j];
                                a[(k + 2) * n + j] -= p * z3;
                            }
                            a[(k + 1) * n + j] -= p * y3;
                            a[k * n + j] -= p * x2;
                        }
                        int mmin = nn < k + 3 ? nn : k + 3;
                        for (int i = l; i <= mmin; i++) {
                            p = x2 * a[i * n + k] + y3 * a[i * n + k + 1];
                            if (k != nn - 1) {
                                p += z3 * a[i * n + k + 2];
                                a[i * n + k + 2] -= p * r;
                            }
                            a[i * n + k + 1] -= p * q;
                            a[i * n + k] -= p;
                        }
                    }
                }
            }
        } while (nn >= 0 && l < nn);
    }
    return 0;
}

/* Eigenvector for one eigenvalue by inverse iteration on (A - lambda I). */
static int inverse_iteration(const double *A, int n, double lr, double li,
                             qcomplex *v) {
    qcomplex *M = (qcomplex *)malloc((size_t)(n * n) * sizeof(qcomplex));
    int *piv = (int *)malloc((size_t)n * sizeof(int));
    if (M == NULL || piv == NULL) { free(M); free(piv); return -1; }
    double scale = 0.0;
    for (int i = 0; i < n * n; i++) scale += fabs(A[i]);
    scale = (scale > 0.0) ? scale / (double)(n * n) : 1.0;
    double eps = 1e-13 * (scale + fabs(lr) + fabs(li) + 1.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i * n + j] = qc(A[i * n + j] - (i == j ? lr + eps : 0.0),
                              i == j ? -li : 0.0);
    int sign;
    lu_decomp_c(M, n, piv, &sign);
    for (int i = 0; i < n; i++) v[i] = qc(1.0 / sqrt((double)n), 0.0);
    for (int iter = 0; iter < 3; iter++) {
        lu_solve_c(M, n, piv, v, 1);
        double norm = 0.0;
        for (int i = 0; i < n; i++) norm += v[i].re * v[i].re + v[i].im * v[i].im;
        norm = sqrt(norm);
        if (!(norm > 0.0) || !isfinite(norm)) break;
        for (int i = 0; i < n; i++) { v[i].re /= norm; v[i].im /= norm; }
    }
    /* NumPy normalises so the largest-magnitude entry is real and positive-ish;
     * matching its exact phase convention is not required, but a deterministic
     * phase keeps results reproducible. */
    int best = 0;
    double bestmag = -1.0;
    for (int i = 0; i < n; i++) {
        double m = qc_abs(v[i]);
        if (m > bestmag) { bestmag = m; best = i; }
    }
    if (bestmag > 0.0) {
        qcomplex phase = qc_div(qc(qc_abs(v[best]), 0.0), v[best]);
        for (int i = 0; i < n; i++) v[i] = qc_mul(v[i], phase);
    }
    free(M);
    free(piv);
    return 0;
}

static PyObject *eig_common(PyObject *arg, int want_vectors) {
    int n;
    QArray *a0 = require_square(arg, want_vectors ? "eig" : "eigvals", &n);
    if (a0 == NULL) return NULL;
    if (a0->dtype == QNP_COMPLEX128) {
        Py_DECREF(a0);
        PyErr_SetString(PyExc_NotImplementedError, "eig of a complex matrix");
        return NULL;
    }
    QArray *original = qnp_astype(a0, QNP_FLOAT64, 1);
    QArray *work = qnp_astype(a0, QNP_FLOAT64, 1);
    Py_DECREF(a0);
    if (original == NULL || work == NULL) {
        Py_XDECREF(original); Py_XDECREF(work);
        return NULL;
    }
    double *wr = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *wi = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (wr == NULL || wi == NULL) {
        free(wr); free(wi);
        Py_DECREF(original); Py_DECREF(work);
        return PyErr_NoMemory();
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    hessenberg((double *)work->data, n, NULL);
    rc = hqr((double *)work->data, n, wr, wi);
    Py_END_ALLOW_THREADS
    Py_DECREF(work);
    if (rc < 0) {
        free(wr); free(wi); Py_DECREF(original);
        PyErr_SetString(QNP_LinAlgError, "Eigenvalue computation did not converge");
        return NULL;
    }
    int any_complex = 0;
    for (int i = 0; i < n; i++) if (wi[i] != 0.0) any_complex = 1;
    qintp nn = n;
    QArray *w = qnp_new(1, &nn, any_complex ? QNP_COMPLEX128 : QNP_FLOAT64);
    if (w == NULL) {
        free(wr); free(wi); Py_DECREF(original);
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        if (any_complex) ((qcomplex *)w->data)[i] = qc(wr[i], wi[i]);
        else ((double *)w->data)[i] = wr[i];
    }
    PyObject *result;
    if (!want_vectors) {
        result = (PyObject *)w;
    } else {
        qintp shape[2] = {n, n};
        QArray *v = qnp_new(2, shape, any_complex ? QNP_COMPLEX128 : QNP_FLOAT64);
        qcomplex *col = (qcomplex *)malloc((size_t)(n > 0 ? n : 1) * sizeof(qcomplex));
        if (v == NULL || col == NULL) {
            Py_XDECREF(v); free(col); Py_DECREF(w);
            free(wr); free(wi); Py_DECREF(original);
            return PyErr_NoMemory();
        }
        for (int k = 0; k < n; k++) {
            inverse_iteration((const double *)original->data, n, wr[k], wi[k], col);
            for (int i = 0; i < n; i++) {
                if (any_complex) ((qcomplex *)v->data)[i * n + k] = col[i];
                else ((double *)v->data)[i * n + k] = col[i].re;
            }
        }
        free(col);
        result = Py_BuildValue("(NN)", (PyObject *)w, (PyObject *)v);
    }
    free(wr);
    free(wi);
    Py_DECREF(original);
    return result;
}

static PyObject *py_eig(PyObject *self, PyObject *arg) { (void)self; return eig_common(arg, 1); }
static PyObject *py_eigvals(PyObject *self, PyObject *arg) { (void)self; return eig_common(arg, 0); }

/* ------------------------------------------------------------------ SVD */

/* Golub-Reinsch: Householder bidiagonalisation followed by implicit QR on the
 * bidiagonal.  `a` (m x n, m >= n) is overwritten with the thin U. */
static int svdcmp(double *a, int m, int n, double *w, double *v) {
    double *rv1 = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (rv1 == NULL) return -1;
    double g = 0.0, scale = 0.0, anorm = 0.0;
    int l = 0;
    for (int i = 0; i < n; i++) {
        l = i + 1;
        rv1[i] = scale * g;
        g = scale = 0.0;
        double s = 0.0;
        if (i < m) {
            for (int k = i; k < m; k++) scale += fabs(a[k * n + i]);
            if (scale != 0.0) {
                for (int k = i; k < m; k++) {
                    a[k * n + i] /= scale;
                    s += a[k * n + i] * a[k * n + i];
                }
                double f = a[i * n + i];
                g = (f >= 0.0) ? -sqrt(s) : sqrt(s);
                double h = f * g - s;
                a[i * n + i] = f - g;
                for (int j = l; j < n; j++) {
                    double sum = 0.0;
                    for (int k = i; k < m; k++) sum += a[k * n + i] * a[k * n + j];
                    double fj = sum / h;
                    for (int k = i; k < m; k++) a[k * n + j] += fj * a[k * n + i];
                }
                for (int k = i; k < m; k++) a[k * n + i] *= scale;
            }
        }
        w[i] = scale * g;
        g = scale = 0.0;
        s = 0.0;
        if (i < m && i != n - 1) {
            for (int k = l; k < n; k++) scale += fabs(a[i * n + k]);
            if (scale != 0.0) {
                for (int k = l; k < n; k++) {
                    a[i * n + k] /= scale;
                    s += a[i * n + k] * a[i * n + k];
                }
                double f = a[i * n + l];
                g = (f >= 0.0) ? -sqrt(s) : sqrt(s);
                double h = f * g - s;
                a[i * n + l] = f - g;
                for (int k = l; k < n; k++) rv1[k] = a[i * n + k] / h;
                for (int j = l; j < m; j++) {
                    double sum = 0.0;
                    for (int k = l; k < n; k++) sum += a[j * n + k] * a[i * n + k];
                    for (int k = l; k < n; k++) a[j * n + k] += sum * rv1[k];
                }
                for (int k = l; k < n; k++) a[i * n + k] *= scale;
            }
        }
        double t = fabs(w[i]) + fabs(rv1[i]);
        if (t > anorm) anorm = t;
    }
    for (int i = n - 1; i >= 0; i--) {
        if (i < n - 1) {
            if (g != 0.0) {
                for (int j = l; j < n; j++) v[j * n + i] = (a[i * n + j] / a[i * n + l]) / g;
                for (int j = l; j < n; j++) {
                    double s = 0.0;
                    for (int k = l; k < n; k++) s += a[i * n + k] * v[k * n + j];
                    for (int k = l; k < n; k++) v[k * n + j] += s * v[k * n + i];
                }
            }
            for (int j = l; j < n; j++) v[i * n + j] = v[j * n + i] = 0.0;
        }
        v[i * n + i] = 1.0;
        g = rv1[i];
        l = i;
    }
    for (int i = (m < n ? m : n) - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        for (int j = l; j < n; j++) a[i * n + j] = 0.0;
        if (g != 0.0) {
            g = 1.0 / g;
            for (int j = l; j < n; j++) {
                double s = 0.0;
                for (int k = l; k < m; k++) s += a[k * n + i] * a[k * n + j];
                double f = (s / a[i * n + i]) * g;
                for (int k = i; k < m; k++) a[k * n + j] += f * a[k * n + i];
            }
            for (int j = i; j < m; j++) a[j * n + i] *= g;
        } else {
            for (int j = i; j < m; j++) a[j * n + i] = 0.0;
        }
        a[i * n + i] += 1.0;
    }
    for (int k = n - 1; k >= 0; k--) {
        for (int its = 1; its <= 60; its++) {
            int flag = 1, nm = 0;
            int lk;
            for (lk = k; lk >= 0; lk--) {
                nm = lk - 1;
                if (fabs(rv1[lk]) + anorm == anorm) { flag = 0; break; }
                if (nm >= 0 && fabs(w[nm]) + anorm == anorm) break;
            }
            if (flag && lk >= 0) {
                double c = 0.0, s = 1.0;
                for (int i = lk; i <= k; i++) {
                    double f = s * rv1[i];
                    rv1[i] = c * rv1[i];
                    if (fabs(f) + anorm == anorm) break;
                    g = w[i];
                    double h = pythag(f, g);
                    w[i] = h;
                    h = 1.0 / h;
                    c = g * h;
                    s = -f * h;
                    for (int j = 0; j < m; j++) {
                        double y = a[j * n + nm], z = a[j * n + i];
                        a[j * n + nm] = y * c + z * s;
                        a[j * n + i] = z * c - y * s;
                    }
                }
            }
            double z = w[k];
            if (lk == k) {
                if (z < 0.0) {
                    w[k] = -z;
                    for (int j = 0; j < n; j++) v[j * n + k] = -v[j * n + k];
                }
                break;
            }
            if (its == 60) { free(rv1); return -1; }
            double x = w[lk], y = w[k - 1];
            g = rv1[k - 1];
            double h = rv1[k];
            double f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = pythag(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + (f >= 0.0 ? fabs(g) : -fabs(g)))) - h)) / x;
            double c = 1.0, s = 1.0;
            for (int j = lk; j <= k - 1; j++) {
                int i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y *= c;
                for (int jj = 0; jj < n; jj++) {
                    double xx = v[jj * n + j], zz = v[jj * n + i];
                    v[jj * n + j] = xx * c + zz * s;
                    v[jj * n + i] = zz * c - xx * s;
                }
                z = pythag(f, h);
                w[j] = z;
                if (z != 0.0) {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = c * g + s * y;
                x = c * y - s * g;
                for (int jj = 0; jj < m; jj++) {
                    double yy = a[jj * n + j], zz = a[jj * n + i];
                    a[jj * n + j] = yy * c + zz * s;
                    a[jj * n + i] = zz * c - yy * s;
                }
            }
            rv1[lk] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }
    /* Descending singular values, as every LAPACK-backed SVD reports them. */
    for (int i = 0; i < n - 1; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (w[j] > w[best]) best = j;
        if (best != i) {
            double t = w[i]; w[i] = w[best]; w[best] = t;
            for (int r = 0; r < m; r++) { t = a[r * n + i]; a[r * n + i] = a[r * n + best]; a[r * n + best] = t; }
            for (int r = 0; r < n; r++) { t = v[r * n + i]; v[r * n + i] = v[r * n + best]; v[r * n + best] = t; }
        }
    }
    free(rv1);
    return 0;
}

/* Extends the orthonormal columns of `U` (m x n) to a full m x m basis. */
static int complete_basis(const double *U, int m, int n, double *full) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) full[i * m + j] = U[i * n + j];
    int col = n;
    for (int seed = 0; seed < m && col < m; seed++) {
        double *v = full + 0;
        double *cand = (double *)malloc((size_t)m * sizeof(double));
        if (cand == NULL) return -1;
        for (int i = 0; i < m; i++) cand[i] = (i == seed) ? 1.0 : 0.0;
        for (int rep = 0; rep < 2; rep++) {
            for (int j = 0; j < col; j++) {
                double dot = 0.0;
                for (int i = 0; i < m; i++) dot += full[i * m + j] * cand[i];
                for (int i = 0; i < m; i++) cand[i] -= dot * full[i * m + j];
            }
        }
        double norm = 0.0;
        for (int i = 0; i < m; i++) norm += cand[i] * cand[i];
        norm = sqrt(norm);
        if (norm > 1e-8) {
            for (int i = 0; i < m; i++) full[i * m + col] = cand[i] / norm;
            col++;
        }
        (void)v;
        free(cand);
    }
    return col == m ? 0 : -1;
}

/* Thin SVD of a general m x n matrix into caller-provided buffers. */
static int thin_svd(const double *A, int m, int n, double *U, double *s, double *Vt) {
    int k = m < n ? m : n;
    int rc;
    if (m >= n) {
        double *a = (double *)malloc((size_t)(m * n) * sizeof(double));
        double *v = (double *)malloc((size_t)(n * n) * sizeof(double));
        if (a == NULL || v == NULL) { free(a); free(v); return -1; }
        memcpy(a, A, (size_t)(m * n) * sizeof(double));
        rc = svdcmp(a, m, n, s, v);
        if (rc == 0) {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) U[i * n + j] = a[i * n + j];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) Vt[i * n + j] = v[j * n + i];
        }
        free(a);
        free(v);
    } else {
        /* Work on the transpose, then swap the roles of the factors. */
        double *at = (double *)malloc((size_t)(m * n) * sizeof(double));
        double *v = (double *)malloc((size_t)(m * m) * sizeof(double));
        if (at == NULL || v == NULL) { free(at); free(v); return -1; }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++) at[i * m + j] = A[j * n + i];
        rc = svdcmp(at, n, m, s, v);
        if (rc == 0) {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < m; j++) U[i * m + j] = v[i * m + j];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++) Vt[i * n + j] = at[j * m + i];
        }
        free(at);
        free(v);
    }
    (void)k;
    return rc;
}

static PyObject *py_svd(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao;
    int full_matrices = 1, compute_uv = 1;
    static char *kwlist[] = {"a", "full_matrices", "compute_uv", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|pp:svd", kwlist,
                                     &ao, &full_matrices, &compute_uv)) return NULL;
    QArray *a = require_matrix(ao, QNP_FLOAT64, "svd");
    if (a == NULL) return NULL;
    if (a->dtype == QNP_COMPLEX128) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_NotImplementedError, "svd of a complex matrix");
        return NULL;
    }
    int m = (int)a->shape[0], n = (int)a->shape[1];
    int k = m < n ? m : n;
    qintp sshape = k;
    QArray *s = qnp_new(1, &sshape, QNP_FLOAT64);
    double *U = (double *)malloc((size_t)((m * k) > 0 ? (m * k) : 1) * sizeof(double));
    double *Vt = (double *)malloc((size_t)((k * n) > 0 ? (k * n) : 1) * sizeof(double));
    if (s == NULL || U == NULL || Vt == NULL) {
        Py_XDECREF(s); free(U); free(Vt); Py_DECREF(a);
        return PyErr_NoMemory();
    }
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = thin_svd((const double *)a->data, m, n, U, (double *)s->data, Vt);
    Py_END_ALLOW_THREADS
    Py_DECREF(a);
    if (rc < 0) {
        Py_DECREF(s); free(U); free(Vt);
        PyErr_SetString(QNP_LinAlgError, "SVD did not converge");
        return NULL;
    }
    if (!compute_uv) {
        free(U);
        free(Vt);
        return (PyObject *)s;
    }
    int ucols = full_matrices ? m : k;
    int vrows = full_matrices ? n : k;
    qintp ushape[2] = {m, ucols};
    qintp vshape[2] = {vrows, n};
    QArray *uarr = qnp_new(2, ushape, QNP_FLOAT64);
    QArray *varr = qnp_new(2, vshape, QNP_FLOAT64);
    if (uarr == NULL || varr == NULL) {
        Py_XDECREF(uarr); Py_XDECREF(varr); Py_DECREF(s);
        free(U); free(Vt);
        return PyErr_NoMemory();
    }
    rc = 0;
    if (ucols == k) {
        memcpy(uarr->data, U, (size_t)(m * k) * sizeof(double));
    } else {
        rc = complete_basis(U, m, k, (double *)uarr->data);
    }
    if (rc == 0) {
        if (vrows == k) {
            memcpy(varr->data, Vt, (size_t)(k * n) * sizeof(double));
        } else {
            /* Complete V's rows, then transpose back into Vh. */
            /* Zeroed so a degenerate rank still leaves the completion with
             * something well defined to orthogonalise against. */
            double *V = (double *)calloc((size_t)(n * k) + 1, sizeof(double));
            double *Vfull = (double *)calloc((size_t)(n * n) + 1, sizeof(double));
            if (V == NULL || Vfull == NULL) rc = -1;
            else {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < k; j++) V[i * k + j] = Vt[j * n + i];
                rc = complete_basis(V, n, k, Vfull);
                if (rc == 0)
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            ((double *)varr->data)[i * n + j] = Vfull[j * n + i];
            }
            free(V);
            free(Vfull);
        }
    }
    free(U);
    free(Vt);
    if (rc < 0) {
        Py_DECREF(uarr); Py_DECREF(varr); Py_DECREF(s);
        PyErr_SetString(QNP_LinAlgError, "could not complete an orthonormal basis");
        return NULL;
    }
    return Py_BuildValue("(NNN)", (PyObject *)uarr, (PyObject *)s, (PyObject *)varr);
}

static PyObject *py_lstsq(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *bo, *rcond_o = Py_None;
    static char *kwlist[] = {"a", "b", "rcond", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O:lstsq", kwlist, &ao, &bo, &rcond_o))
        return NULL;
    QArray *a = require_matrix(ao, QNP_FLOAT64, "lstsq");
    if (a == NULL) return NULL;
    QArray *b0 = qnp_from_any(bo, QNP_FLOAT64, 1);
    if (b0 == NULL) { Py_DECREF(a); return NULL; }
    int vector_rhs = (b0->nd == 1);
    QArray *b = qnp_ascontiguous(b0);
    Py_DECREF(b0);
    if (b == NULL) { Py_DECREF(a); return NULL; }
    int m = (int)a->shape[0], n = (int)a->shape[1];
    int nrhs = vector_rhs ? 1 : (int)b->shape[1];
    if (b->shape[0] != m) {
        PyErr_Format(QNP_LinAlgError,
                     "lstsq: incompatible dimensions -- matrix has %d rows, "
                     "right-hand side has %zd", m, b->shape[0]);
        Py_DECREF(a); Py_DECREF(b);
        return NULL;
    }
    int k = m < n ? m : n;
    double *U = (double *)malloc((size_t)((m * k) > 0 ? (m * k) : 1) * sizeof(double));
    double *Vt = (double *)malloc((size_t)((k * n) > 0 ? (k * n) : 1) * sizeof(double));
    qintp sshape = k;
    QArray *s = qnp_new(1, &sshape, QNP_FLOAT64);
    if (U == NULL || Vt == NULL || s == NULL) {
        free(U); free(Vt); Py_XDECREF(s);
        Py_DECREF(a); Py_DECREF(b);
        return PyErr_NoMemory();
    }
    QArray *acopy = a;             /* kept for the residual calculation */
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = thin_svd((const double *)a->data, m, n, U, (double *)s->data, Vt);
    Py_END_ALLOW_THREADS
    if (rc < 0) {
        free(U); free(Vt); Py_DECREF(s); Py_DECREF(b); Py_DECREF(acopy);
        PyErr_SetString(QNP_LinAlgError, "SVD did not converge in lstsq");
        return NULL;
    }
    double *sv = (double *)s->data;
    double rcond;
    if (rcond_o == Py_None) rcond = (double)(m > n ? m : n) * 2.220446049250313e-16;
    else {
        rcond = PyFloat_AsDouble(rcond_o);
        if (rcond == -1.0 && PyErr_Occurred()) {
            free(U); free(Vt); Py_DECREF(s); Py_DECREF(b); Py_DECREF(acopy);
            return NULL;
        }
        if (rcond < 0) rcond = (double)(m > n ? m : n) * 2.220446049250313e-16;
    }
    double cutoff = rcond * (k > 0 ? sv[0] : 0.0);
    int rank = 0;
    for (int i = 0; i < k; i++) if (sv[i] > cutoff) rank++;
    qintp xshape[2] = {n, nrhs};
    QArray *x = qnp_new(vector_rhs ? 1 : 2, xshape, QNP_FLOAT64);
    if (x == NULL) {
        free(U); free(Vt); Py_DECREF(s); Py_DECREF(b); Py_DECREF(acopy);
        return NULL;
    }
    memset(x->data, 0, (size_t)(n * nrhs) * sizeof(double));
    const double *bp = (const double *)b->data;
    double *xp = (double *)x->data;
    double *tmp = (double *)malloc((size_t)(k > 0 ? k : 1) * sizeof(double));
    if (tmp == NULL) {
        free(U); free(Vt); Py_DECREF(s); Py_DECREF(b); Py_DECREF(x);
        Py_DECREF(acopy);
        return PyErr_NoMemory();
    }
    for (int r = 0; r < nrhs; r++) {
        for (int i = 0; i < k; i++) {
            if (i >= rank) { tmp[i] = 0.0; continue; }
            double dot = 0.0;
            for (int j = 0; j < m; j++) dot += U[j * k + i] * bp[j * nrhs + r];
            tmp[i] = dot / sv[i];
        }
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int i = 0; i < k; i++) acc += Vt[i * n + j] * tmp[i];
            xp[j * nrhs + r] = acc;
        }
    }
    /* Residual sums of squares, reported only for the full-rank overdetermined
     * case -- the same rule NumPy applies. */
    PyObject *residuals;
    if (rank == n && m > n) {
        qintp rshape = nrhs;
        QArray *res = qnp_new(1, &rshape, QNP_FLOAT64);
        if (res == NULL) {
            free(tmp); free(U); free(Vt);
            Py_DECREF(s); Py_DECREF(b); Py_DECREF(x); Py_DECREF(acopy);
            return NULL;
        }
        const double *ap = (const double *)acopy->data;
        for (int r = 0; r < nrhs; r++) {
            double acc = 0.0;
            for (int i = 0; i < m; i++) {
                double pred = 0.0;
                const double *arow = ap + (qintp)i * n;
                for (int j = 0; j < n; j++) pred += arow[j] * xp[j * nrhs + r];
                double d = pred - bp[i * nrhs + r];
                acc += d * d;
            }
            ((double *)res->data)[r] = acc;
        }
        residuals = (PyObject *)res;
    } else {
        qintp zero = 0;
        residuals = (PyObject *)qnp_new(1, &zero, QNP_FLOAT64);
    }
    Py_DECREF(acopy);
    free(tmp);
    free(U);
    free(Vt);
    Py_DECREF(b);
    return Py_BuildValue("(NNiN)", (PyObject *)x, residuals, rank, (PyObject *)s);
}


/* ---------------------------------------------------------------- norms */

/* Vector p-norm of `n` elements read with `stride` doubles between them. */
static double vector_norm(const double *x, qintp n, qintp stride, double p, int is_inf,
                          int is_neginf) {
    if (is_inf) {
        double best = 0.0;
        for (qintp i = 0; i < n; i++) {
            double v = fabs(x[i * stride]);
            if (v > best) best = v;
        }
        return best;
    }
    if (is_neginf) {
        double best = INFINITY;
        for (qintp i = 0; i < n; i++) {
            double v = fabs(x[i * stride]);
            if (v < best) best = v;
        }
        return n ? best : 0.0;
    }
    if (p == 2.0) {
        /* One pass, no temporaries: this is the hot path of the whole
         * package, so it does not go through square-then-sum. */
        double s = 0.0;
        if (stride == 1) for (qintp i = 0; i < n; i++) s += x[i] * x[i];
        else for (qintp i = 0; i < n; i++) s += x[i * stride] * x[i * stride];
        return sqrt(s);
    }
    if (p == 1.0) {
        double s = 0.0;
        for (qintp i = 0; i < n; i++) s += fabs(x[i * stride]);
        return s;
    }
    if (p == 0.0) {
        double s = 0.0;
        for (qintp i = 0; i < n; i++) s += (x[i * stride] != 0.0);
        return s;
    }
    double s = 0.0;
    for (qintp i = 0; i < n; i++) s += pow(fabs(x[i * stride]), p);
    return pow(s, 1.0 / p);
}

static double vector_norm_c(const qcomplex *x, qintp n, qintp stride, double p,
                            int is_inf, int is_neginf) {
    if (is_inf || is_neginf) {
        double best = is_inf ? 0.0 : INFINITY;
        for (qintp i = 0; i < n; i++) {
            double v = qc_abs(x[i * stride]);
            if (is_inf ? v > best : v < best) best = v;
        }
        return n ? best : 0.0;
    }
    if (p == 2.0) {
        double s = 0.0;
        for (qintp i = 0; i < n; i++) {
            qcomplex z = x[i * stride];
            s += z.re * z.re + z.im * z.im;
        }
        return sqrt(s);
    }
    double s = 0.0;
    for (qintp i = 0; i < n; i++) s += pow(qc_abs(x[i * stride]), p);
    return pow(s, 1.0 / p);
}

/* Decodes the `ord` argument into a numeric exponent plus infinity flags. */
static int parse_ord(PyObject *ord, double *p, int *is_inf, int *is_neginf,
                     int *is_fro, int *is_nuc) {
    *is_inf = *is_neginf = *is_fro = *is_nuc = 0;
    *p = 2.0;
    if (ord == NULL || ord == Py_None) return 0;
    if (PyUnicode_Check(ord)) {
        const char *s = PyUnicode_AsUTF8(ord);
        if (s == NULL) return -1;
        if (!strcmp(s, "fro")) { *is_fro = 1; return 0; }
        if (!strcmp(s, "nuc")) { *is_nuc = 1; return 0; }
        PyErr_Format(PyExc_ValueError, "Invalid norm order '%s'", s);
        return -1;
    }
    double v = PyFloat_AsDouble(ord);
    if (v == -1.0 && PyErr_Occurred()) return -1;
    if (isinf(v)) { if (v > 0) *is_inf = 1; else *is_neginf = 1; return 0; }
    *p = v;
    return 0;
}

static PyObject *py_norm(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *ord = Py_None, *axis_o = Py_None;
    int keepdims = 0;
    static char *kwlist[] = {"x", "ord", "axis", "keepdims", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOp:norm", kwlist,
                                     &ao, &ord, &axis_o, &keepdims)) return NULL;
    QArray *a0 = qnp_from_any(ao, -1, 0);
    if (a0 == NULL) return NULL;
    int dt = a0->dtype < QNP_FLOAT64 ? QNP_FLOAT64 : a0->dtype;
    QArray *af = qnp_astype(a0, dt, 0);
    Py_DECREF(a0);
    if (af == NULL) return NULL;
    QArray *a = qnp_ascontiguous(af);
    Py_DECREF(af);
    if (a == NULL) return NULL;
    double p;
    int is_inf, is_neginf, is_fro, is_nuc;
    if (parse_ord(ord, &p, &is_inf, &is_neginf, &is_fro, &is_nuc) < 0) {
        Py_DECREF(a);
        return NULL;
    }
    if (axis_o == Py_None) {
        if (a->nd == 2 && !(ord == NULL || ord == Py_None) && !is_fro) {
            /* Genuine matrix norms. */
            qintp m = a->shape[0], n = a->shape[1];
            double result = 0.0;
            if (is_nuc || ((p == 2.0 || p == -2.0) && !is_inf && !is_neginf)) {
                if (dt == QNP_COMPLEX128) {
                    Py_DECREF(a);
                    PyErr_SetString(PyExc_NotImplementedError,
                                    "spectral norms of complex matrices");
                    return NULL;
                }
                int k = (int)(m < n ? m : n);
                double *U = (double *)malloc((size_t)((m * k) > 0 ? (m * k) : 1) * sizeof(double));
                double *Vt = (double *)malloc((size_t)((k * n) > 0 ? (k * n) : 1) * sizeof(double));
                double *s = (double *)malloc((size_t)(k > 0 ? k : 1) * sizeof(double));
                if (U == NULL || Vt == NULL || s == NULL) {
                    free(U); free(Vt); free(s); Py_DECREF(a);
                    return PyErr_NoMemory();
                }
                int rc = thin_svd((const double *)a->data, (int)m, (int)n, U, s, Vt);
                if (rc == 0) {
                    if (is_nuc) { result = 0.0; for (int i = 0; i < k; i++) result += s[i]; }
                    else if (p == 2.0) result = k ? s[0] : 0.0;
                    else result = k ? s[k - 1] : 0.0;
                }
                free(U); free(Vt); free(s);
                Py_DECREF(a);
                if (rc < 0) {
                    PyErr_SetString(QNP_LinAlgError, "SVD did not converge in norm");
                    return NULL;
                }
                return PyFloat_FromDouble(result);
            }
            /* Column sums for 1, row sums for infinity. */
            int by_column = (p == 1.0 || p == -1.0);
            qintp outer = by_column ? n : m;
            qintp inner = by_column ? m : n;
            double best = (p < 0 || is_neginf) ? INFINITY : 0.0;
            for (qintp i = 0; i < outer; i++) {
                double s = 0.0;
                for (qintp j = 0; j < inner; j++) {
                    qintp r = by_column ? j : i, c = by_column ? i : j;
                    if (dt == QNP_COMPLEX128)
                        s += qc_abs(((const qcomplex *)a->data)[r * n + c]);
                    else
                        s += fabs(((const double *)a->data)[r * n + c]);
                }
                if ((p < 0 || is_neginf) ? s < best : s > best) best = s;
            }
            Py_DECREF(a);
            return PyFloat_FromDouble(outer ? best : 0.0);
        }
        qintp n = qnp_size(a);
        double result = (dt == QNP_COMPLEX128)
            ? vector_norm_c((const qcomplex *)a->data, n, 1, is_fro ? 2.0 : p, is_inf, is_neginf)
            : vector_norm((const double *)a->data, n, 1, is_fro ? 2.0 : p, is_inf, is_neginf);
        Py_DECREF(a);
        return PyFloat_FromDouble(result);
    }
    int axis;
    if (qnp_parse_axis(axis_o, a->nd, &axis) < 0) { Py_DECREF(a); return NULL; }
    qintp shape[QNP_MAXDIMS];
    int nd = 0;
    for (int i = 0; i < a->nd; i++) {
        if (i == axis) { if (keepdims) shape[nd++] = 1; }
        else shape[nd++] = a->shape[i];
    }
    QArray *out = qnp_new(nd, shape, QNP_FLOAT64);
    if (out == NULL) { Py_DECREF(a); return NULL; }
    qintp len = a->shape[axis];
    qintp stride = a->strides[axis] / QNP_ITEMSIZE(dt);
    qintp outer = qnp_size(a) / (len ? len : 1);
    qintp idx[QNP_MAXDIMS] = {0};
    double *dst = (double *)out->data;
    for (qintp k = 0; k < outer; k++) {
        const char *base = a->data;
        for (int d = 0, w = 0; d < a->nd; d++) {
            if (d == axis) continue;
            base += idx[w] * a->strides[d];
            w++;
        }
        *dst++ = (dt == QNP_COMPLEX128)
            ? vector_norm_c((const qcomplex *)base, len, stride, is_fro ? 2.0 : p, is_inf, is_neginf)
            : vector_norm((const double *)base, len, stride, is_fro ? 2.0 : p, is_inf, is_neginf);
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
    Py_DECREF(a);
    return qnp_wrap_scalar_or_array(out);
}

PyMethodDef qnp_linalg_methods[] = {
    {"solve", py_solve, METH_VARARGS, "Solve a linear system."},
    {"inv", py_inv, METH_O, "Matrix inverse."},
    {"det", py_det, METH_O, "Determinant."},
    {"slogdet", py_slogdet, METH_O, "Sign and log-magnitude of the determinant."},
    {"cholesky", py_cholesky, METH_O, "Lower Cholesky factor."},
    {"eigh", py_eigh, METH_O, "Eigenvalues and eigenvectors of a symmetric matrix."},
    {"eigvalsh", py_eigvalsh, METH_O, "Eigenvalues of a symmetric matrix."},
    {"eig", py_eig, METH_O, "Eigenvalues and eigenvectors of a general matrix."},
    {"eigvals", py_eigvals, METH_O, "Eigenvalues of a general matrix."},
    {"svd", (PyCFunction)py_svd, METH_VARARGS | METH_KEYWORDS, "Singular value decomposition."},
    {"lstsq", (PyCFunction)py_lstsq, METH_VARARGS | METH_KEYWORDS, "Least-squares solution."},
    {"norm", (PyCFunction)py_norm, METH_VARARGS | METH_KEYWORDS, "Vector or matrix norm."},
    {NULL}
};
