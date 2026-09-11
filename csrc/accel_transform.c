/* Elementwise and matrix kernels write their result in its final array. */
#include "qaccel.h"
#include "erfc.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double lanczos_sum(double z) {
    double a0 = .99999999999980993 + 676.5203681218851/(z+1.);
    double a1 = -1259.1392167224028/(z+2.);
    double a2 = 771.32342877765313/(z+3.);
    double a3 = -176.61502916214059/(z+4.);
    a0 += 12.507343278686905/(z+5.);
    a1 += -.13857109526572012/(z+6.);
    a2 += 9.9843695780195716e-6/(z+7.);
    a3 += 1.5056327351493116e-7/(z+8.);
    return (a0+a1)+(a2+a3);
}

static double gamma_scalar(double v) {
    if (v == floor(v) && v <= 0.) return INFINITY;
    if (v < .5) return M_PI/(sin(M_PI*v)*gamma_scalar(1.-v));
    double z = v-1., t = z+7.5;
    return 2.5066282746310002*lanczos_sum(z)*exp((z+.5)*log(t)-t);
}

static double log_gamma_scalar(double v) {
    if (v < .5) return log(M_PI/fabs(sin(M_PI*v)))-log_gamma_scalar(1.-v);
    double z = v-1., t = z+7.5;
    return .5*log(2.*M_PI)+(z+.5)*log(t)-t+log(lanczos_sum(z));
}

static PyObject *map_special(PyObject *args, PyObject *kwds, int kind) {
    PyObject *obj;
    static char *names[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", names, &obj)) return NULL;
    QArray *a = qnp_from_any(obj, QNP_FLOAT64, 1);
    if (a == NULL) return NULL;
    if (a->nd != 1) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "expected a 1-dimensional array");
        return NULL;
    }
    QArray *out = qnp_new(1, a->shape, QNP_FLOAT64);
    if (out == NULL) { Py_DECREF(a); return NULL; }
    qintp n = a->shape[0], stride = a->strides[0]/(qintp)sizeof(double);
    const double *src = (const double *)a->data;
    double *dst = (double *)out->data;
    Py_BEGIN_ALLOW_THREADS
    /* Dispatch outside the loop so the compiler can inline each kernel. */
    switch (kind) {
        case 0: for (qintp i=0;i<n;i++) dst[i]=gamma_scalar(src[i*stride]); break;
        case 1: for (qintp i=0;i<n;i++) dst[i]=log_gamma_scalar(src[i*stride]); break;
        case 2: for (qintp i=0;i<n;i++) dst[i]=erf(src[i*stride]); break;
        case 3:
            if (n >= 64) erfc_array(src,stride,dst,n);
            else for (qintp i=0;i<n;i++) dst[i]=erfc(src[i*stride]);
            break;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(a);
    return (PyObject *)out;
}

#define SPECIAL(name,kind) \
static PyObject *py_##name(PyObject *self, PyObject *args, PyObject *kwds) { \
    (void)self; return map_special(args,kwds,kind); \
}
SPECIAL(gamma,0)
SPECIAL(log_gamma,1)
SPECIAL(erf,2)
SPECIAL(erfc,3)
#undef SPECIAL

static PyObject *py_matmul(PyObject *self, PyObject *args, PyObject *kwds) {
    (void)self;
    PyObject *ao, *bo;
    static char *names[] = {"a", "b", NULL};
    if (!PyArg_ParseTupleAndKeywords(args,kwds,"OO",names,&ao,&bo)) return NULL;
    QArray *a = qaccel_input(ao,2), *b = NULL, *out = NULL;
    if (a == NULL) return NULL;
    /* Converting b can execute Python and reshape a. Retain the validated
     * matrix description; the owned reference keeps its data alive. */
    qintp rows = a->shape[0], inner = a->shape[1];
    const double *ap = (const double *)a->data;
    b = qaccel_input(bo,2);
    if (b == NULL) goto done;
    if (inner != b->shape[0]) {
        PyErr_SetString(PyExc_ValueError, "matrix shapes are not aligned");
        goto done;
    }
    qintp shape[2] = {rows, b->shape[1]};
    out = qnp_new(2,shape,QNP_FLOAT64);
    if (out != NULL) {
        Py_BEGIN_ALLOW_THREADS
        qnp_gemm_f64(ap,(double *)b->data,(double *)out->data,
                     shape[0],shape[1],inner);
        Py_END_ALLOW_THREADS
    }
done:
    Py_DECREF(a); Py_XDECREF(b);
    return (PyObject *)out;
}

/* FFT bindings live next to the core so both public transform APIs share it. */
PyObject *qaccel_fft(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *qaccel_ifft(PyObject *self, PyObject *args, PyObject *kwds);
#define METHOD(name) {#name,(PyCFunction)py_##name,METH_VARARGS|METH_KEYWORDS,NULL}
PyMethodDef qaccel_transform_methods[] = {
    METHOD(gamma), METHOD(log_gamma), METHOD(erf), METHOD(erfc), METHOD(matmul),
    {"fft",(PyCFunction)qaccel_fft,METH_VARARGS|METH_KEYWORDS,NULL},
    {"ifft",(PyCFunction)qaccel_ifft,METH_VARARGS|METH_KEYWORDS,NULL},
    {NULL}
};
