/* Eigen iterations, stationary grid solvers and tridiagonal systems.
 *
 * These kernels operate directly on QArray storage.  Scratch is O(n) for
 * tridiagonal/QR iteration and two grids for cavity flow; no Python arrays or
 * allocation are needed inside a numerical sweep. */
#include "qaccel.h"

#ifdef _MSC_VER
#define GRID_RESTRICT __restrict
#else
#define GRID_RESTRICT restrict
#endif

static int nonnegative_limit(qintp limit, const char *name) {
    if (limit >= 0) return 0;
    PyErr_Format(PyExc_OverflowError, "%s cannot be negative", name);
    return -1;
}

static int distinct_outputs(QArray *a, QArray *b, const char *message) {
    if (!qnp_may_share_memory(a, b)) return 0;
    PyErr_SetString(PyExc_ValueError, message);
    return -1;
}

/* Read-only operands normally borrow their contiguous storage.  An operand
 * overlapping an in-place output needs the snapshot the old bridge provided. */
static int snapshot_if_shared(QArray **input, QArray *output) {
    if (!qnp_may_share_memory(*input, output)) return 0;
    QArray *copy = qaccel_copy((PyObject *)*input, (*input)->nd);
    if (!copy) return -1;
    Py_DECREF(*input);
    *input = copy;
    return 0;
}

/* Converting a later sequence may run Python (__float__, for example) and
 * reshape or make an earlier borrowed array read-only. Recheck its metadata
 * after all conversions, before indexing shape or touching numerical data. */
static int grid_revalidate(QArray *a, int nd, int writable) {
    if (a->nd != nd || a->dtype != QNP_FLOAT64 || !qnp_is_c_contiguous(a) ||
        (writable && !(a->flags & QNP_WRITEABLE))) {
        PyErr_SetString(PyExc_ValueError,
            "array shape, layout or writeability changed during argument conversion");
        return -1;
    }
    return 0;
}

static double jacobi_tangent(double theta) {
    if (theta == 0.0) return 1.0;
    if (fabs(theta) > 1e8) return 1.0 / (2.0 * theta);
    return copysign(1.0, theta) / (fabs(theta) + sqrt(theta * theta + 1.0));
}

static PyObject *accel_jacobi_eigen(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *obj;
    double tol = 1e-12;
    qintp max_sweeps = 100;
    static char *names[] = {"a", "tol", "max_sweeps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|dn:jacobi_eigen", names,
                                    &obj, &tol, &max_sweeps)) return NULL;
    if (nonnegative_limit(max_sweeps, "max_sweeps") < 0) return NULL;
    QArray *darr = qaccel_copy(obj, 2), *v = NULL, *vals = NULL;
    if (!darr || qaccel_square(darr) < 0) goto fail;
    qintp value_shape = darr->shape[0];
    v = qnp_new(2, darr->shape, QNP_FLOAT64);
    vals = qnp_new(1, &value_shape, QNP_FLOAT64);
    if (!v || !vals) goto fail;
    /* Keep the loop bound in a non-escaping local: passing its address to an
     * external constructor would prevent vectorization under our alias flags. */
    const qintp n = value_shape;
    double *GRID_RESTRICT d = (double *)darr->data;
    double *GRID_RESTRICT vv = (double *)v->data;
    double *GRID_RESTRICT eigs = (double *)vals->data;
    qintp sweep = 0;
    int converged = 0;
    Py_BEGIN_ALLOW_THREADS
    memset(vv, 0, (size_t)qnp_size(v) * sizeof(double));
    for (qintp i = 0; i < n; ++i) vv[i * n + i] = 1.0;
    double prev_off = INFINITY;
    while (sweep < max_sweeps) {
        ++sweep;
        double off = 0.0;
        for (qintp i = 1; i < n; ++i)
            for (qintp j = 0; j < i; ++j) {
                double a = d[i * n + j];
                off += a * a;
            }
        off = sqrt(2.0 * off);
        if (off < tol || off >= prev_off) {
            converged = 1;
            break;
        }
        prev_off = off;
        for (qintp p = 0; p + 1 < n; ++p) {
            for (qintp q = p + 1; q < n; ++q) {
                double dpq = d[p * n + q];
                if (fabs(dpq) < 1e-300) continue;
                double theta = (d[q * n + q] - d[p * n + p]) / (2.0 * dpq);
                double t = jacobi_tangent(theta);
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (qintp i = 0; i < n; ++i) {
                    double dip = d[i * n + p], diq = d[i * n + q];
                    d[i * n + p] = c * dip - s * diq;
                    d[i * n + q] = s * dip + c * diq;
                    double vip = vv[i * n + p], viq = vv[i * n + q];
                    vv[i * n + p] = c * vip - s * viq;
                    vv[i * n + q] = s * vip + c * viq;
                }
                for (qintp j = 0; j < n; ++j) {
                    double dpj = d[p * n + j], dqj = d[q * n + j];
                    d[p * n + j] = c * dpj - s * dqj;
                    d[q * n + j] = s * dpj + c * dqj;
                }
            }
        }
    }
    for (qintp i = 0; i < n; ++i) eigs[i] = d[i * n + i];
    Py_END_ALLOW_THREADS
    Py_DECREF(darr);
    return Py_BuildValue("NNnO", vals, v, sweep, converged ? Py_True : Py_False);
fail:
    Py_XDECREF(darr);
    Py_XDECREF(v);
    Py_XDECREF(vals);
    return NULL;
}

static PyObject *accel_hessenberg_qr_iterate(PyObject *self, PyObject *args,
                                           PyObject *kw) {
    (void)self;
    PyObject *ho, *vo = Py_None;
    double tol = 1e-12;
    qintp max_iter = 5000;
    static char *names[] = {"h", "v", "tol", "max_iter", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|Odn:hessenberg_qr_iterate", names,
                                    &ho, &vo, &tol, &max_iter)) return NULL;
    if (nonnegative_limit(max_iter, "max_iter") < 0) return NULL;
    QArray *ha = qaccel_mutable(ho, 2), *va = NULL;
    double *cs = NULL, *sn = NULL;
    if (!ha || qaccel_square(ha) < 0) goto fail;
    qintp n = ha->shape[0];
    if (vo != Py_None) {
        va = qaccel_mutable(vo, 2);
        if (!va) goto fail;
        if (!qnp_same_shape(2, va->shape, ha->shape)) {
            PyErr_SetString(PyExc_ValueError, "V must have the same shape as H");
            goto fail;
        }
        if (distinct_outputs(ha, va, "H and V must not overlap") < 0) goto fail;
    }
    if (n >= 2 && max_iter) {
        cs = qaccel_alloc(n - 1, sizeof(double));
        sn = qaccel_alloc(n - 1, sizeof(double));
        if (!cs || !sn) goto fail;
    }
    double *h = (double *)ha->data, *v = va ? (double *)va->data : NULL;
    qintp iter = 0;
    int converged = n < 2;
    Py_BEGIN_ALLOW_THREADS
    while (!converged && iter < max_iter) {
        ++iter;
        for (qintp k = 0; k + 1 < n; ++k) {
            double a = h[k * n + k], b = h[(k + 1) * n + k];
            double r = hypot(a, b);
            double c = r == 0.0 ? 1.0 : a / r;
            double s = r == 0.0 ? 0.0 : b / r;
            cs[k] = c;
            sn[k] = s;
            if (s != 0.0) {
                double *row0 = h + k * n, *row1 = row0 + n;
                for (qintp j = k; j < n; ++j) {
                    double u = row0[j], w = row1[j];
                    row0[j] = c * u + s * w;
                    row1[j] = c * w - s * u;
                }
            }
        }
        for (qintp k = 0; k + 1 < n; ++k) {
            double c = cs[k], s = sn[k];
            if (s == 0.0) continue;
            for (qintp i = 0; i < k + 2; ++i) {
                double *row = h + i * n;
                double u = row[k], w = row[k + 1];
                row[k] = c * u + s * w;
                row[k + 1] = c * w - s * u;
            }
            if (v) {
                for (qintp i = 0; i < n; ++i) {
                    double *row = v + i * n;
                    double u = row[k], w = row[k + 1];
                    row[k] = c * u + s * w;
                    row[k + 1] = c * w - s * u;
                }
            }
        }
        converged = 1;
        for (qintp k = 0; k + 1 < n; ++k) {
            double e = fabs(h[(k + 1) * n + k]);
            double d = fabs(h[k * n + k]) + fabs(h[(k + 1) * n + k + 1]);
            if (e > tol * (d + 1e-300)) {
                converged = 0;
                break;
            }
        }
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(cs);
    PyMem_Free(sn);
    Py_DECREF(ha);
    Py_XDECREF(va);
    return Py_BuildValue("nO", iter, converged ? Py_True : Py_False);
fail:
    PyMem_Free(cs);
    PyMem_Free(sn);
    Py_XDECREF(ha);
    Py_XDECREF(va);
    return NULL;
}

static PyObject *accel_sor_poisson(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *uo, *fo;
    double beta2, dx2, omega, tol = 1e-10;
    qintp max_iter = 20000;
    static char *names[] = {"u", "f", "beta2", "dx2", "omega", "tol", "max_iter", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOddd|dn:sor_poisson", names,
                                    &uo, &fo, &beta2, &dx2, &omega, &tol,
                                    &max_iter)) return NULL;
    if (nonnegative_limit(max_iter, "max_iter") < 0) return NULL;
    QArray *ua = qaccel_mutable(uo, 2), *fa = NULL, *history = NULL;
    if (!ua) goto fail;
    fa = qaccel_input(fo, 2);
    if (!fa) goto fail;
    if (grid_revalidate(ua, 2, 1) < 0 || grid_revalidate(fa, 2, 0) < 0) goto fail;
    if (!qnp_same_shape(2, ua->shape, fa->shape)) {
        PyErr_SetString(PyExc_ValueError, "f must have the same shape as u");
        goto fail;
    }
    qintp rows = ua->shape[0], cols = ua->shape[1];
    if (rows < 2 || cols < 2) {
        PyErr_SetString(PyExc_ValueError, "grid must be at least 2x2");
        goto fail;
    }
    if (snapshot_if_shared(&fa, ua) < 0) goto fail;
    /* Grow only as sweeps are used: a huge iteration cap on a converged grid
     * should neither allocate a huge history nor need a final array copy. */
    qintp capacity = max_iter < 256 ? max_iter : 256;
    history = qnp_new(1, &capacity, QNP_FLOAT64);
    if (!history) goto fail;
    double *u = (double *)ua->data, *f = (double *)fa->data;
    double denom = 2.0 * (1.0 + beta2);
    qintp it = 0;
    int converged = 0;
    while (it < max_iter && !converged) {
        if (it == capacity) {
            qintp next = capacity > max_iter / 2 ? max_iter : 2 * capacity;
            if (next > PY_SSIZE_T_MAX / (qintp)sizeof(double)) {
                PyErr_SetString(PyExc_OverflowError, "residual history is too large");
                goto fail;
            }
            char *data = PyMem_Realloc(history->data, (size_t)next * sizeof(double));
            if (!data) {
                PyErr_NoMemory();
                goto fail;
            }
            history->data = data;
            capacity = next;
        }
        /* Check signals every 256 sweeps even after the history has grown. */
        qintp end = capacity - it > 256 ? it + 256 : capacity;
        double *residuals = (double *)history->data;
        Py_BEGIN_ALLOW_THREADS
        while (it < end) {
            double change = 0.0;
            for (qintp i = 1; i + 1 < rows; ++i) {
                qintp row = i * cols, up = row + cols, down = row - cols;
                for (qintp j = 1; j + 1 < cols; ++j) {
                    double old = u[row + j];
                    double newval = (u[up + j] + u[down + j]
                                     + beta2 * (u[row + j + 1] + u[row + j - 1])
                                     - dx2 * f[row + j]) / denom;
                    double val = (1.0 - omega) * old + omega * newval;
                    u[row + j] = val;
                    double delta = fabs(val - old);
                    if (delta > change || isnan(delta)) change = delta;
                }
            }
            residuals[it++] = change;
            if (change < tol) {
                converged = 1;
                break;
            }
        }
        Py_END_ALLOW_THREADS
        if (PyErr_CheckSignals() < 0) goto fail;
    }
    if (it != capacity) {
        char *data = PyMem_Realloc(history->data,
                                  (size_t)(it ? it : 1) * sizeof(double));
        /* A failed optional shrink does not invalidate the computed result. */
        if (data) history->data = data;
    }
    history->shape[0] = it;
    qnp_update_flags(history);
    Py_DECREF(ua);
    Py_DECREF(fa);
    return Py_BuildValue("nON", it, converged ? Py_True : Py_False, history);
fail:
    Py_XDECREF(ua);
    Py_XDECREF(fa);
    Py_XDECREF(history);
    return NULL;
}

static PyObject *accel_lid_driven_cavity(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *po, *wo;
    double h, nu, dt, tol;
    qintp max_iter;
    static char *names[] = {"psi", "w", "h", "nu", "dt", "tol", "max_iter", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOddddn:lid_driven_cavity", names,
                                    &po, &wo, &h, &nu, &dt, &tol, &max_iter)) return NULL;
    if (nonnegative_limit(max_iter, "max_iter") < 0) return NULL;
    QArray *pa = qaccel_mutable(po, 2), *wa = NULL;
    double *psi_scratch = NULL, *w_scratch = NULL;
    if (!pa) goto fail;
    wa = qaccel_mutable(wo, 2);
    if (!wa) goto fail;
    qintp n = pa->shape[0];
    if (pa->shape[1] != n || !qnp_same_shape(2, pa->shape, wa->shape)) {
        PyErr_SetString(PyExc_ValueError, "psi and w must both be square and the same size");
        goto fail;
    }
    if (n < 3) {
        PyErr_SetString(PyExc_ValueError, "grid must be at least 3x3");
        goto fail;
    }
    if (distinct_outputs(pa, wa, "psi and w must not overlap") < 0) goto fail;
    qintp total = qnp_size(pa), iterations = 0;
    if (max_iter) {
        psi_scratch = qaccel_alloc(total, sizeof(double));
        w_scratch = qaccel_alloc(total, sizeof(double));
        if (!psi_scratch || !w_scratch) goto fail;
        double *psi = (double *)pa->data, *w = (double *)wa->data;
        double *psi_next = psi_scratch, *w_next = w_scratch;
        double hh = h * h, two_h = 2.0 * h;
        qintp last = n - 1;
        Py_BEGIN_ALLOW_THREADS
        /* Jacobi buffers keep identical boundaries throughout.  Swapping
         * pointers removes the full-grid copy from every inner sweep. */
        memcpy(psi_scratch, psi, (size_t)total * sizeof(double));
        while (iterations < max_iter) {
            ++iterations;
            for (int sweep = 0; sweep < 30; ++sweep) {
                double delta = 0.0;
                for (qintp i = 1; i < last; ++i) {
                    qintp r = i * n;
                    for (qintp j = 1; j < last; ++j) {
                        double val = 0.25 * (psi[r + n + j] + psi[r - n + j]
                                           + psi[r + j + 1] + psi[r + j - 1]
                                           + hh * w[r + j]);
                        psi_next[r + j] = val;
                        double d = fabs(val - psi[r + j]);
                        if (d > delta || isnan(d)) delta = d;
                    }
                }
                double *swap = psi;
                psi = psi_next;
                psi_next = swap;
                if (delta < 1e-10) break;
            }
            /* Rows precede columns so column wall formulas own the corners. */
            for (qintp j = 0; j < n; ++j) {
                w_next[j] = 2.0 * (psi[j] - psi[n + j]) / hh;
                w_next[last * n + j] =
                    2.0 * (psi[last * n + j] - psi[(last - 1) * n + j]) / hh - 2.0 / h;
            }
            for (qintp i = 0; i < n; ++i) {
                qintp r = i * n;
                w_next[r] = 2.0 * (psi[r] - psi[r + 1]) / hh;
                w_next[r + last] = 2.0 * (psi[r + last] - psi[r + last - 1]) / hh;
            }
            for (qintp i = 1; i < last; ++i) {
                qintp r = i * n;
                for (qintp j = 1; j < last; ++j) {
                    double u = (psi[r + n + j] - psi[r - n + j]) / two_h;
                    double v = -((psi[r + j + 1] - psi[r + j - 1]) / two_h);
                    double wx = (w[r + j + 1] - w[r + j - 1]) / two_h;
                    double wy = (w[r + n + j] - w[r - n + j]) / two_h;
                    double lapw = (w[r + n + j] + w[r - n + j]
                                   + w[r + j + 1] + w[r + j - 1]
                                   - 4.0 * w[r + j]) / hh;
                    w_next[r + j] = w[r + j] + dt * (-u * wx - v * wy + nu * lapw);
                }
            }
            double change = 0.0;
            for (qintp i = 0; i < total; ++i) {
                double d = fabs(w_next[i] - w[i]);
                if (d > change || isnan(d)) change = d;
            }
            double *swap = w;
            w = w_next;
            w_next = swap;
            if (change < tol) break;
        }
        if (psi != (double *)pa->data)
            memcpy(pa->data, psi, (size_t)total * sizeof(double));
        if (w != (double *)wa->data)
            memcpy(wa->data, w, (size_t)total * sizeof(double));
        Py_END_ALLOW_THREADS
    }
    PyMem_Free(psi_scratch);
    PyMem_Free(w_scratch);
    Py_DECREF(pa);
    Py_DECREF(wa);
    return PyLong_FromSsize_t(iterations);
fail:
    PyMem_Free(psi_scratch);
    PyMem_Free(w_scratch);
    Py_XDECREF(pa);
    Py_XDECREF(wa);
    return NULL;
}

static int tridiagonal_lengths(QArray *sub, QArray *diag, QArray *sup) {
    qintp n = diag->shape[0], edge = n ? n - 1 : 0;
    if (sub->shape[0] == edge && sup->shape[0] == edge) return 0;
    PyErr_SetString(PyExc_ValueError, "sub- and super-diagonals must have length n-1");
    return -1;
}

static PyObject *accel_thomas(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *so, *do_, *uo, *ro;
    static char *names[] = {"sub", "diag", "sup", "rhs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO:thomas", names,
                                    &so, &do_, &uo, &ro)) return NULL;
    QArray *sub = qaccel_input(so, 1), *diag = NULL, *sup = NULL, *rhs = NULL;
    if (!sub) goto fail;
    diag = qaccel_copy(do_, 1);
    if (!diag) goto fail;
    sup = qaccel_input(uo, 1);
    if (!sup) goto fail;
    rhs = qaccel_copy(ro, 1);
    if (!rhs) goto fail;
    if (grid_revalidate(sub, 1, 0) < 0 || grid_revalidate(diag, 1, 1) < 0 ||
        grid_revalidate(sup, 1, 0) < 0 || grid_revalidate(rhs, 1, 1) < 0) goto fail;
    qintp n = diag->shape[0];
    if (rhs->shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "rhs must match the diagonal length");
        goto fail;
    }
    if (tridiagonal_lengths(sub, diag, sup) < 0) goto fail;
    const double *a = (double *)sub->data, *c = (double *)sup->data;
    double *GRID_RESTRICT d = (double *)diag->data;
    double *GRID_RESTRICT x = (double *)rhs->data;
    int singular = 0;
    Py_BEGIN_ALLOW_THREADS
    for (qintp i = 1; i < n; ++i) {
        if (d[i - 1] == 0.0) { singular = 1; break; }
        double m = a[i - 1] / d[i - 1];
        d[i] -= m * c[i - 1];
        x[i] -= m * x[i - 1];
    }
    if (n && !singular) {
        if (d[n - 1] == 0.0) singular = 1;
        else {
            x[n - 1] /= d[n - 1];
            for (qintp i = n - 1; i-- > 0;)
                x[i] = (x[i] - c[i] * x[i + 1]) / d[i];
        }
    }
    Py_END_ALLOW_THREADS
    if (singular) {
        PyErr_SetString(PyExc_RuntimeError, "singular:zero pivot in Thomas algorithm");
        goto fail;
    }
    Py_DECREF(sub);
    Py_DECREF(diag);
    Py_DECREF(sup);
    return (PyObject *)rhs;
fail:
    Py_XDECREF(sub);
    Py_XDECREF(diag);
    Py_XDECREF(sup);
    Py_XDECREF(rhs);
    return NULL;
}

static PyObject *accel_thomas_batch(PyObject *self, PyObject *args, PyObject *kw) {
    (void)self;
    PyObject *so, *do_, *uo, *ro;
    static char *names[] = {"sub", "diag", "sup", "rhs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO:thomas_batch", names,
                                    &so, &do_, &uo, &ro)) return NULL;
    QArray *sub = qaccel_input(so, 1), *diag = NULL, *sup = NULL, *rhs = NULL;
    if (!sub) goto fail;
    diag = qaccel_copy(do_, 1);
    if (!diag) goto fail;
    sup = qaccel_input(uo, 1);
    if (!sup) goto fail;
    rhs = qaccel_mutable(ro, 2);
    if (!rhs) goto fail;
    if (grid_revalidate(sub, 1, 0) < 0 || grid_revalidate(diag, 1, 1) < 0 ||
        grid_revalidate(sup, 1, 0) < 0 || grid_revalidate(rhs, 2, 1) < 0) goto fail;
    qintp n = diag->shape[0], m = rhs->shape[1];
    if (rhs->shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "rhs must have one row per diagonal entry");
        goto fail;
    }
    if (tridiagonal_lengths(sub, diag, sup) < 0 ||
        snapshot_if_shared(&sub, rhs) < 0 || snapshot_if_shared(&sup, rhs) < 0) goto fail;
    const double *a = (double *)sub->data, *c = (double *)sup->data;
    double *GRID_RESTRICT d = (double *)diag->data;
    double *GRID_RESTRICT x = (double *)rhs->data;
    int singular = 0;
    Py_BEGIN_ALLOW_THREADS
    if (n && m) {
        /* Factor before touching RHS: a singular matrix must not leave the
         * caller with partially eliminated data. Only O(n) scratch is used. */
        for (qintp i = 1; i < n; ++i) {
            if (d[i - 1] == 0.0) { singular = 1; break; }
            double f = a[i - 1] / d[i - 1];
            d[i] -= f * c[i - 1];
        }
        if (d[n - 1] == 0.0) singular = 1;
        if (!singular) {
            for (qintp i = 1; i < n; ++i) {
                double f = a[i - 1] / d[i - 1];
                double *row = x + i * m, *prev = row - m;
                for (qintp k = 0; k < m; ++k) row[k] -= f * prev[k];
            }
            double inv = 1.0 / d[n - 1];
            for (qintp k = 0; k < m; ++k) x[(n - 1) * m + k] *= inv;
            for (qintp i = n - 1; i-- > 0;) {
                inv = 1.0 / d[i];
                double s = c[i], *row = x + i * m, *next = row + m;
                for (qintp k = 0; k < m; ++k) row[k] = (row[k] - s * next[k]) * inv;
            }
        }
    }
    Py_END_ALLOW_THREADS
    if (singular) {
        PyErr_SetString(PyExc_RuntimeError, "singular:zero pivot in Thomas algorithm");
        goto fail;
    }
    Py_DECREF(sub);
    Py_DECREF(diag);
    Py_DECREF(sup);
    Py_DECREF(rhs);
    Py_RETURN_NONE;
fail:
    Py_XDECREF(sub);
    Py_XDECREF(diag);
    Py_XDECREF(sup);
    Py_XDECREF(rhs);
    return NULL;
}

#define GRID_METHOD(name, doc) \
    {#name, (PyCFunction)(void(*)(void))accel_##name, METH_VARARGS | METH_KEYWORDS, doc}

PyMethodDef qaccel_grid_methods[] = {
    GRID_METHOD(jacobi_eigen, "Cyclic Jacobi eigenpairs and convergence diagnostics."),
    GRID_METHOD(hessenberg_qr_iterate, "Iterate a Hessenberg matrix and optional vectors in place."),
    GRID_METHOD(sor_poisson, "Solve a Poisson stencil in place with adaptive residual storage."),
    GRID_METHOD(lid_driven_cavity, "Solve cavity flow in place with two reusable scratch grids."),
    GRID_METHOD(thomas, "Solve a tridiagonal system without changing the inputs."),
    GRID_METHOD(thomas_batch, "Solve every RHS column in place using one diagonal factorization."),
    {NULL, NULL, 0, NULL}
};
