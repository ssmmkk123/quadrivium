/* Numerical kernels share the array core's storage; no foreign array bridge. */
#ifndef QACCEL_H
#define QACCEL_H
#include "qnp.h"
#include <float.h>

QArray *qaccel_input(PyObject *obj, int nd);
QArray *qaccel_copy(PyObject *obj, int nd);
QArray *qaccel_mutable(PyObject *obj, int nd);
int qaccel_square(QArray *a);
void *qaccel_alloc(qintp count, size_t size);
/* C += alpha A B (or A B^T). Leading dimensions are in doubles. */
void qaccel_gemm(qintp m, qintp n, qintp k, double alpha,
                  const double *a, qintp lda, const double *b, qintp ldb,
                  double *c, qintp ldc, int trans_b);
int qaccel_add(PyObject *module);
extern PyMethodDef qaccel_direct_methods[];
extern PyMethodDef qaccel_grid_methods[];
extern PyMethodDef qaccel_ode_methods[];
extern PyMethodDef qaccel_transform_methods[];
#endif
