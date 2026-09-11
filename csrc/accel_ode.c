/* Adaptive embedded Runge-Kutta integration over native array buffers.
 *
 * Scratch storage is O(stages * components). Trajectory arrays grow
 * geometrically in place and become the returned arrays without a second
 * full-sized copy. Python callbacks retain the GIL and receive independent
 * state snapshots whenever they keep a reference to their arguments.
 */
#include "qaccel.h"

static int ode_finite(const double *values, qintp n) {
    for (qintp i = 0; i < n; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static int ode_underflow(double t, double min_step) {
    char message[256];
    PyOS_snprintf(message, sizeof(message),
        "stepsize:step size underflow at t=%.6f: required h < %.2e; "
        "the problem is likely stiff -- try an implicit solver", t, min_step);
    PyErr_SetString(PyExc_RuntimeError, message);
    return -1;
}

/* Only private, owning arrays reach this function, before being exposed to
 * Python. A failed realloc leaves the old allocation available for cleanup. */
static int ode_resize(QArray *array, qintp rows, qintp width) {
    if (rows > PY_SSIZE_T_MAX / (qintp)sizeof(double) / width) {
        PyErr_SetString(PyExc_OverflowError, "ODE trajectory is too large");
        return -1;
    }
    char *data = PyMem_Realloc(array->data,
                              (size_t)rows * (size_t)width * sizeof(double));
    if (data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    array->data = data;
    array->shape[0] = rows;
    return 0;
}

static void ode_finish_array(QArray *array, qintp rows, qintp width) {
    /* Shrinking is optional if the allocator cannot do so. Logical shape
     * always reflects the exact result, with no uninitialised trailing rows. */
    char *data = PyMem_Realloc(array->data,
                              (size_t)rows * (size_t)width * sizeof(double));
    if (data != NULL) array->data = data;
    array->shape[0] = rows;
    qnp_update_flags(array);
}

static int ode_eval(PyObject *f, double t, QArray **state,
                    double *out, qintp n, double min_step) {
    PyObject *time = PyFloat_FromDouble(t);
    if (time == NULL) return -1;
    PyObject *args[2] = {time, (PyObject *)*state};
    PyObject *result = PyObject_Vectorcall(f, args, 2, NULL);
    Py_DECREF(time);
    if (result == NULL) return -1;

    /* Accept scalars, vectors, nested sequences, and strided arrays, matching
     * the public as_vector coercion. Retain no reference to the RHS output:
     * a callback is allowed to reuse the same output buffer on every call. */
    QArray *converted = qnp_from_any(result, QNP_FLOAT64, 1);
    Py_DECREF(result);
    if (converted == NULL) return -1;
    QArray *values = qnp_ascontiguous(converted);
    Py_DECREF(converted);
    if (values == NULL) return -1;
    qintp count = qnp_size(values);
    if (count != n) {
        PyErr_Format(PyExc_ValueError,
                     "right-hand side returned %zd components, expected %zd", count, n);
        Py_DECREF(values);
        return -1;
    }
    const double *source = (const double *)values->data;
    int finite = ode_finite(source, n);
    if (finite) memcpy(out, source, (size_t)n * sizeof(double));
    Py_DECREF(values);
    if (!finite) return ode_underflow(t, min_step);

    /* Normal callbacks do not retain their input, so reuse its allocation.
     * Retained arrays, views, buffer exports, and weakrefs must keep their
     * original values. A callback may also change its argument's shape or
     * flags; such an array is discarded from this private cache as well. */
    QArray *argument = *state;
    if (Py_REFCNT(argument) != 1 || argument->exports != 0 ||
        argument->weakreflist != NULL || argument->base != NULL ||
        argument->nd != 1 || argument->shape[0] != n ||
        argument->dtype != QNP_FLOAT64 ||
        (argument->flags & (QNP_OWNDATA | QNP_WRITEABLE | QNP_C_CONTIGUOUS)) !=
            (QNP_OWNDATA | QNP_WRITEABLE | QNP_C_CONTIGUOUS)) {
        Py_CLEAR(*state);
    }
    return 0;
}

static PyObject *py_adaptive_rk(PyObject *self, PyObject *args, PyObject *kwargs) {
    (void)self;
    PyObject *f, *c_obj, *a_obj, *hi_obj, *lo_obj, *y_obj, *h_obj;
    double order, t0, tf, rtol, atol, max_step, min_step;
    qintp max_steps;
    static char *names[] = {"f", "c", "a_flat", "b_hi", "b_lo", "order",
        "t0", "tf", "y0", "rtol", "atol", "h0", "max_step", "min_step",
        "max_steps", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOdddOddOddn:adaptive_rk", names,
            &f, &c_obj, &a_obj, &hi_obj, &lo_obj, &order, &t0, &tf, &y_obj,
            &rtol, &atol, &h_obj, &max_step, &min_step, &max_steps)) return NULL;
    if (!PyCallable_Check(f)) {
        PyErr_SetString(PyExc_TypeError, "f must be callable");
        return NULL;
    }
    if (!isfinite(t0) || !isfinite(tf) || !isfinite(tf - t0)) {
        PyErr_SetString(PyExc_ValueError, "time interval must be finite");
        return NULL;
    }
    if (!isfinite(rtol) || !isfinite(atol) || rtol < 0.0 || atol < 0.0 ||
        (rtol == 0.0 && atol == 0.0)) {
        PyErr_SetString(PyExc_ValueError,
            "rtol and atol must be finite, nonnegative, and not both zero");
        return NULL;
    }
    if (isnan(max_step) || max_step <= 0.0 || !isfinite(min_step) ||
        min_step <= 0.0 || max_step < min_step) {
        PyErr_SetString(PyExc_ValueError,
                        "step bounds must satisfy 0 < min_step <= max_step");
        return NULL;
    }
    double h = fabs(tf - t0) / 100.0;
    if (h_obj != Py_None) {
        h = PyFloat_AsDouble(h_obj);
        if (h == -1.0 && PyErr_Occurred()) return NULL;
        if (!isfinite(h) || h == 0.0) {
            PyErr_SetString(PyExc_ValueError, "h0 must be finite and nonzero");
            return NULL;
        }
        h = fabs(h);
    }
    if (max_steps <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_steps must be positive");
        return NULL;
    }
    QArray *c = NULL, *a = NULL, *hi = NULL, *lo = NULL, *y_array = NULL;
    QArray *ts = NULL, *ys = NULL, *dys = NULL, *state = NULL;
    double *k = NULL, *scratch = NULL;
    PyObject *result = NULL;
    /* Own the tableau as well as y: Python callbacks may mutate the arrays
     * passed to this entry point without changing the active integration. */
    c = qaccel_copy(c_obj, 1);
    if (c == NULL) goto cleanup;
    a = qaccel_copy(a_obj, 1);
    if (a == NULL) goto cleanup;
    hi = qaccel_copy(hi_obj, 1);
    if (hi == NULL) goto cleanup;
    lo = qaccel_copy(lo_obj, 1);
    if (lo == NULL) goto cleanup;
    y_array = qaccel_copy(y_obj, 1);
    if (y_array == NULL) goto cleanup;
    qintp n = y_array->shape[0], stages = hi->shape[0];
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "y0 must have at least one component");
        goto cleanup;
    }
    if (!ode_finite((double *)y_array->data, n)) {
        PyErr_SetString(PyExc_ValueError, "y0 must be finite");
        goto cleanup;
    }
    if (stages == 0 || c->shape[0] != stages || lo->shape[0] != stages ||
        stages > PY_SSIZE_T_MAX / stages || a->shape[0] != stages * stages) {
        PyErr_SetString(PyExc_ValueError, "inconsistent Runge-Kutta tableau dimensions");
        goto cleanup;
    }
    if (!isfinite(order) || order <= 0.0 ||
        !ode_finite((double *)c->data, stages) ||
        !ode_finite((double *)a->data, a->shape[0]) ||
        !ode_finite((double *)hi->data, stages) ||
        !ode_finite((double *)lo->data, stages)) {
        PyErr_SetString(PyExc_ValueError,
                        "Runge-Kutta tableau must be finite with positive order");
        goto cleanup;
    }
    if (stages > PY_SSIZE_T_MAX / n) {
        PyErr_SetString(PyExc_OverflowError, "Runge-Kutta scratch space is too large");
        goto cleanup;
    }
    k = qaccel_alloc(stages * n, sizeof(double));
    scratch = qaccel_alloc(n, 3 * sizeof(double));
    if (k == NULL || scratch == NULL) goto cleanup;
    double *acc_hi = scratch, *acc_lo = scratch + n, *y_hi = scratch + 2 * n;
    double *y = (double *)y_array->data;
    const double *cv = (double *)c->data, *av = (double *)a->data;
    const double *bh = (double *)hi->data, *bl = (double *)lo->data;
    qintp max_rows = PY_SSIZE_T_MAX / (qintp)sizeof(double) / n;
    qintp capacity = 4096 / n;
    if (capacity < 2) capacity = 2;
    if (capacity > 64) capacity = 64;
    if (capacity > max_rows) capacity = max_rows;
    if (max_steps < capacity) capacity = max_steps + 1;
    if (t0 == tf) capacity = 1;
    qintp shape[2] = {capacity, n};
    ts = qnp_new(1, shape, QNP_FLOAT64);
    ys = qnp_new(2, shape, QNP_FLOAT64);
    dys = qnp_new(2, shape, QNP_FLOAT64);
    if (ts == NULL || ys == NULL || dys == NULL) goto cleanup;
    double direction = tf >= t0 ? 1.0 : -1.0, t = t0, err_prev = 1.0;
    h = fmin(fmax(h, min_step), max_step) * direction;
    ((double *)ts->data)[0] = t;
    memcpy(ys->data, y, (size_t)n * sizeof(double));
    qintp accepted = 0, rejected = 0;
    size_t calls = 0;
    const double accept_exp = -0.7 / order, previous_exp = 0.4 / order;
    const double reject_exp = -1.0 / order;
    for (qintp step = 0; step < max_steps; ++step) {
        if ((t - tf) * direction >= 0.0) break;
        if ((step & 255) == 0 && PyErr_CheckSignals() < 0) goto cleanup;
        if (fabs(h) > fabs(tf - t)) h = tf - t;
        if (t + h == t) {
            ode_underflow(t, min_step);
            goto cleanup;
        }
        for (qintp i = 0; i < stages; ++i) {
            if (state == NULL) state = qnp_new(1, &n, QNP_FLOAT64);
            if (state == NULL) goto cleanup;
            double *yi = (double *)state->data;
            memcpy(yi, y, (size_t)n * sizeof(double));
            for (qintp j = 0; j < i; ++j) {
                double coefficient = av[i * stages + j];
                if (coefficient == 0.0) continue;
                double scale = h * coefficient;
                const double *previous = k + j * n;
                for (qintp d = 0; d < n; ++d) yi[d] += scale * previous[d];
            }
            double stage_t = t + cv[i] * h;
            if (!isfinite(stage_t) || !ode_finite(yi, n)) {
                ode_underflow(t, min_step);
                goto cleanup;
            }
            ++calls;
            if (ode_eval(f, stage_t, &state, k + i * n, n, min_step) < 0) goto cleanup;
        }
        memset(acc_hi, 0, (size_t)n * sizeof(double));
        memset(acc_lo, 0, (size_t)n * sizeof(double));
        for (qintp i = 0; i < stages; ++i) {
            const double *stage = k + i * n;
            for (qintp d = 0; d < n; ++d) {
                acc_hi[d] += bh[i] * stage[d];
                acc_lo[d] += bl[i] * stage[d];
            }
        }
        double sum_sq = 0.0;
        for (qintp d = 0; d < n; ++d) {
            y_hi[d] = y[d] + h * acc_hi[d];
            double y_lo = y[d] + h * acc_lo[d];
            double scale = atol + rtol * fmax(fabs(y[d]), fabs(y_hi[d]));
            double error = (y_hi[d] - y_lo) / scale;
            sum_sq += error * error;
        }
        double error = sqrt(sum_sq / (double)n), factor;
        if (!isfinite(error)) {
            ode_underflow(t, min_step);
            goto cleanup;
        }
        if (error <= 1.0) {
            if (accepted + 1 == capacity) {
                if (capacity == max_rows) {
                    PyErr_SetString(PyExc_OverflowError, "ODE trajectory is too large");
                    goto cleanup;
                }
                qintp larger = capacity > max_rows / 2 ? max_rows : capacity * 2;
                if (ode_resize(ts, larger, 1) < 0 ||
                    ode_resize(ys, larger, n) < 0 ||
                    ode_resize(dys, larger, n) < 0) goto cleanup;
                capacity = larger;
            }
            memcpy((double *)dys->data + accepted * n, k, (size_t)n * sizeof(double));
            ++accepted;
            t += h;
            memcpy(y, y_hi, (size_t)n * sizeof(double));
            ((double *)ts->data)[accepted] = t;
            memcpy((double *)ys->data + accepted * n, y, (size_t)n * sizeof(double));
            if ((t - tf) * direction >= 0.0) break;
            factor = error > 0.0
                ? 0.9 * pow(error, accept_exp) * pow(err_prev, previous_exp) : 5.0;
            err_prev = fmax(error, 1e-4);
        } else {
            if (fabs(h) <= min_step) {
                ode_underflow(t, min_step);
                goto cleanup;
            }
            ++rejected;
            factor = 0.9 * pow(error, reject_exp);
        }
        h *= fmin(5.0, fmax(0.2, factor));
        if (fabs(h) > max_step) h = max_step * direction;
        if (fabs(h) < min_step) {
            ode_underflow(t, min_step);
            goto cleanup;
        }
    }
    if (state == NULL) state = qnp_new(1, &n, QNP_FLOAT64);
    if (state == NULL) goto cleanup;
    memcpy(state->data, y, (size_t)n * sizeof(double));
    ++calls;
    if (ode_eval(f, t, &state, (double *)dys->data + accepted * n, n, min_step) < 0)
        goto cleanup;
    ode_finish_array(ts, accepted + 1, 1);
    ode_finish_array(ys, accepted + 1, n);
    ode_finish_array(dys, accepted + 1, n);
    result = Py_BuildValue("OOOnnn", ts, ys, dys, accepted, rejected, (Py_ssize_t)calls);
cleanup:
    Py_XDECREF(c);
    Py_XDECREF(a);
    Py_XDECREF(hi);
    Py_XDECREF(lo);
    Py_XDECREF(y_array);
    Py_XDECREF(ts);
    Py_XDECREF(ys);
    Py_XDECREF(dys);
    Py_XDECREF(state);
    PyMem_Free(k);
    PyMem_Free(scratch);
    return result;
}

PyMethodDef qaccel_ode_methods[] = {
    {"adaptive_rk", (PyCFunction)(void(*)(void))py_adaptive_rk,
        METH_VARARGS | METH_KEYWORDS, "Adaptive embedded Runge-Kutta integration."},
    {NULL, NULL, 0, NULL}
};
