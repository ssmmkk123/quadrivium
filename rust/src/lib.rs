//! Compiled kernels backing `quadrivium`.
//!
//! Every function exported here has a pure-Python twin in the package. The
//! Python side picks whichever is available, so this extension is always
//! optional: it changes how fast the library runs, never what it computes.

use numpy::prelude::*;
use numpy::{Complex64, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// PyO3 return types spell out the lifetime and the concrete array class, so the
// tuples below get names rather than being repeated at each signature.
type Arr1<'py> = Bound<'py, PyArray1<f64>>;
type Arr2<'py> = Bound<'py, PyArray2<f64>>;
type IdxArr<'py> = Bound<'py, PyArray1<i64>>;
type PluOut<'py> = (IdxArr<'py>, Arr2<'py>);
type QrOut<'py> = (Arr2<'py>, Arr2<'py>);
type EigenOut<'py> = (Arr1<'py>, Arr2<'py>, usize, bool);

mod blas;
mod cavity;
mod direct;
mod eigen;
#[path = "fft.rs"]
mod fft_mod;
mod gemm;
mod ode;
mod relax;
mod special;

use direct::LinalgError;

fn map_err(e: LinalgError) -> PyErr {
    // The Python layer re-raises these as the package's own exception types;
    // the message text is preserved so it matches the pure-Python wording.
    match e {
        LinalgError::Singular(m) => PyRuntimeError::new_err(format!("singular:{m}")),
        LinalgError::NotPositiveDefinite(m) => PyRuntimeError::new_err(format!("notpd:{m}")),
    }
}

/// Borrow a C-contiguous array's buffer directly, avoiding a copy. Returns
/// `None` for any other layout, where the caller must pack it instead.
fn as_rows<'a>(a: &'a PyReadonlyArray2<'_, f64>) -> Option<&'a [f64]> {
    if a.is_c_contiguous() {
        a.as_slice().ok()
    } else {
        None
    }
}

/// Row-major copy of a possibly non-contiguous input, as a flat `Vec`.
///
/// The common case is a C-contiguous array, where this is a single `memcpy`.
/// Walking the array with 2-D indexing instead costs an indexed load per
/// element and showed up as milliseconds on the input path of every
/// factorization.
fn to_vec2(a: &PyReadonlyArray2<f64>) -> PyResult<Vec<f64>> {
    if let Some(s) = as_rows(a) {
        return Ok(s.to_vec());
    }
    // ndarray views require aligned elements and strides. NumPy also permits
    // byte-offset buffers; let NumPy align those before constructing a view.
    if !a.is_aligned() || a.strides().iter().any(|stride| stride % 8 != 0) {
        let copy = a.call_method1("copy", ("C",))?;
        return to_vec2(&copy.extract::<PyReadonlyArray2<f64>>()?);
    }
    let v = a.as_array();
    let (m, n) = (v.nrows(), v.ncols());
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            out.push(v[[i, j]]);
        }
    }
    Ok(out)
}

fn to_vec1<T: numpy::Element + Copy>(a: &PyReadonlyArray1<'_, T>) -> PyResult<Vec<T>> {
    if !a.is_aligned() || a.strides()[0] % std::mem::size_of::<T>() as isize != 0 {
        let copy = a.call_method0("copy")?;
        return to_vec1(&copy.extract::<PyReadonlyArray1<T>>()?);
    }
    Ok(a.as_array().to_vec())
}

/// Writable kernels require row-major, aligned storage before releasing the GIL.
fn require_c_layout(a: &PyReadwriteArray2<'_, f64>, name: &str) -> PyResult<()> {
    if !a.is_c_contiguous() || !a.is_aligned() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous and aligned"
        )));
    }
    Ok(())
}

fn square_size(a: &PyReadonlyArray2<'_, f64>) -> PyResult<usize> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    Ok(shape[0])
}

#[pyfunction]
fn cholesky<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m != n {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    let mut buf = to_vec2(&a)?;
    py.detach(|| direct::cholesky_inplace(n, &mut buf))
        .map_err(map_err)?;
    buf.into_pyarray(py).reshape([n, n])
}

#[pyfunction]
fn plu<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>) -> PyResult<PluOut<'py>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m != n {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    let mut buf = to_vec2(&a)?;
    let mut perm = vec![0usize; n];
    py.detach(|| direct::plu_inplace(n, &mut buf, &mut perm))
        .map_err(map_err)?;
    let permi: Vec<i64> = perm.into_iter().map(|p| p as i64).collect();
    Ok((
        permi.into_pyarray(py),
        buf.into_pyarray(py).reshape([n, n])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, want_q=true))]
fn householder_qr<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    want_q: bool,
) -> PyResult<QrOut<'py>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut r = to_vec2(&a)?;
    let mut q = vec![0.0f64; if want_q { m * m } else { 0 }];
    py.detach(|| direct::householder_qr(m, n, &mut r, &mut q, want_q));
    Ok((
        q.into_pyarray(py)
            .reshape([if want_q { m } else { 0 }, m])?,
        r.into_pyarray(py).reshape([m, n])?,
    ))
}

/// QR by Givens rotations; returns `(Q, R)` with `Q` an `m x m` accumulator.
#[pyfunction]
fn givens_qr<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>) -> PyResult<QrOut<'py>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut r = to_vec2(&a)?;
    let mut q = vec![0.0f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    py.detach(|| direct::givens_qr(m, n, &mut r, &mut q));
    Ok((
        q.into_pyarray(py).reshape([m, m])?,
        r.into_pyarray(py).reshape([m, n])?,
    ))
}

type SvdOut<'py> = (Arr2<'py>, Arr2<'py>, usize);

/// One-sided Jacobi SVD; returns `(W, V, sweeps)` with `W = U * S`. The caller
/// reads the singular values off the column norms of `W`, exactly as the
/// pure-Python routine does.
#[pyfunction]
#[pyo3(signature = (a, tol=1e-13, max_sweeps=60))]
fn svd_jacobi<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_sweeps: usize,
) -> PyResult<SvdOut<'py>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m < n {
        return Err(PyValueError::new_err(
            "one-sided Jacobi requires at least as many rows as columns",
        ));
    }
    let mut w = to_vec2(&a)?;
    let mut v = vec![0.0f64; n * n];
    let sweeps = py.detach(|| direct::svd_jacobi(m, n, &mut w, &mut v, tol, max_sweeps));
    Ok((
        w.into_pyarray(py).reshape([m, n])?,
        v.into_pyarray(py).reshape([n, n])?,
        sweeps,
    ))
}

#[pyfunction]
fn qr_least_squares<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
) -> PyResult<Arr1<'py>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m < n {
        return Err(PyValueError::new_err(
            "least squares requires at least as many rows as columns",
        ));
    }
    if b.len() != m {
        return Err(PyValueError::new_err("rhs must match the matrix row count"));
    }
    if n == 0 {
        return Ok(Vec::<f64>::new().into_pyarray(py));
    }
    let mut a = to_vec2(&a)?;
    let mut rhs = to_vec1(&b)?;
    py.detach(|| direct::qr_least_squares(m, n, &mut a, &mut rhs))
        .map_err(map_err)?;
    rhs.truncate(n);
    rhs.shrink_to_fit();
    Ok(rhs.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (l, b, unit_diagonal=false))]
fn forward_substitution<'py>(
    py: Python<'py>,
    l: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    unit_diagonal: bool,
) -> PyResult<Arr1<'py>> {
    let n = square_size(&l)?;
    if b.len() != n {
        return Err(PyValueError::new_err("rhs must match the matrix size"));
    }
    let mut x = to_vec1(&b)?;
    let owned;
    let lb: &[f64] = match as_rows(&l) {
        Some(v) => v,
        None => {
            owned = to_vec2(&l)?;
            &owned
        }
    };
    py.detach(|| direct::forward_substitution(n, lb, n, &mut x, unit_diagonal))
        .map_err(map_err)?;
    Ok(x.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (u, b, unit_diagonal=false, transposed=false))]
fn back_substitution<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    unit_diagonal: bool,
    transposed: bool,
) -> PyResult<Arr1<'py>> {
    let n = square_size(&u)?;
    if b.len() != n {
        return Err(PyValueError::new_err("rhs must match the matrix size"));
    }
    let mut x = to_vec1(&b)?;
    let owned;
    let ub: &[f64] = match as_rows(&u) {
        Some(v) => v,
        None => {
            owned = to_vec2(&u)?;
            &owned
        }
    };
    py.detach(|| {
        if transposed {
            // `u` is the lower-triangular L; solve L' x = b in place.
            direct::back_substitution_trans(n, ub, n, &mut x, unit_diagonal)
        } else {
            direct::back_substitution(n, ub, n, &mut x, unit_diagonal)
        }
    })
    .map_err(map_err)?;
    Ok(x.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (a, atol=1e-12, rtol=1e-5))]
fn is_symmetric(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f64>,
    atol: f64,
    rtol: f64,
) -> PyResult<bool> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    if m != n {
        return Ok(false);
    }
    Ok(match as_rows(&a) {
        Some(v) => py.detach(|| direct::is_symmetric(n, v, atol, rtol)),
        None => {
            let owned = to_vec2(&a)?;
            py.detach(|| direct::is_symmetric(n, &owned, atol, rtol))
        }
    })
}

#[pyfunction]
#[pyo3(signature = (a, tol=1e-12, max_sweeps=100))]
fn jacobi_eigen<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_sweeps: usize,
) -> PyResult<EigenOut<'py>> {
    let n = square_size(&a)?;
    let mut d = to_vec2(&a)?;
    let mut v = vec![0.0f64; n * n];
    let (sweeps, conv) = py.detach(|| eigen::jacobi_eigen(n, &mut d, &mut v, tol, max_sweeps));
    let vals: Vec<f64> = (0..n).map(|i| d[i * n + i]).collect();
    Ok((
        vals.into_pyarray(py),
        v.into_pyarray(py).reshape([n, n])?,
        sweeps,
        conv,
    ))
}

/// In-place unshifted QR iteration on an upper Hessenberg matrix.
///
/// `h` and the optional accumulator `v` are mutated in place -- the caller
/// already owns freshly reduced arrays, so writing through them avoids
/// round-tripping two `n x n` matrices per call. Both must be C-contiguous.
#[pyfunction]
#[pyo3(signature = (h, v=None, tol=1e-12, max_iter=5000))]
fn hessenberg_qr_iterate(
    py: Python<'_>,
    h: PyReadwriteArray2<'_, f64>,
    v: Option<PyReadwriteArray2<'_, f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(usize, bool)> {
    let mut h = h;
    let n = h.shape()[0];
    if h.shape()[1] != n {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    require_c_layout(&h, "H")?;
    let hs = h
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("H must be C-contiguous"))?;
    match v {
        Some(mut v) => {
            if v.shape() != [n, n] {
                return Err(PyValueError::new_err("V must have the same shape as H"));
            }
            require_c_layout(&v, "V")?;
            let vs = v
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("V must be C-contiguous"))?;
            Ok(py.detach(|| eigen::hessenberg_qr_iterate(n, hs, Some(vs), tol, max_iter)))
        }
        None => Ok(py.detach(|| eigen::hessenberg_qr_iterate(n, hs, None, tol, max_iter))),
    }
}

/// SOR / Gauss-Seidel relaxation for the 5-point Poisson stencil.
///
/// `u` is updated in place. Returns `(iterations, converged, residuals)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (u, f, beta2, dx2, omega, tol=1e-10, max_iter=20000))]
fn sor_poisson<'py>(
    py: Python<'py>,
    u: PyReadwriteArray2<'py, f64>,
    f: PyReadonlyArray2<'py, f64>,
    beta2: f64,
    dx2: f64,
    omega: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<(usize, bool, Arr1<'py>)> {
    let mut u = u;
    let (rows, cols) = (u.shape()[0], u.shape()[1]);
    if f.shape() != [rows, cols] {
        return Err(PyValueError::new_err("f must have the same shape as u"));
    }
    if rows < 2 || cols < 2 {
        return Err(PyValueError::new_err("grid must be at least 2x2"));
    }
    let fv = to_vec2(&f)?;
    require_c_layout(&u, "u")?;
    let us = u
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("u must be C-contiguous"))?;
    let (it, conv, res) = py.detach(|| {
        relax::sor_poisson(
            rows - 1,
            cols - 1,
            us,
            &fv,
            beta2,
            dx2,
            omega,
            tol,
            max_iter,
        )
    });
    Ok((it, conv, res.into_pyarray(py)))
}

/// Steady lid-driven cavity flow; `psi` and `w` are filled in place.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (psi, w, h, nu, dt, tol, max_iter))]
fn lid_driven_cavity<'py>(
    py: Python<'py>,
    psi: PyReadwriteArray2<'py, f64>,
    w: PyReadwriteArray2<'py, f64>,
    h: f64,
    nu: f64,
    dt: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<usize> {
    let (mut psi, mut w) = (psi, w);
    let n = psi.shape()[0];
    if psi.shape()[1] != n || w.shape() != [n, n] {
        return Err(PyValueError::new_err(
            "psi and w must both be square and the same size",
        ));
    }
    if n < 3 {
        return Err(PyValueError::new_err("grid must be at least 3x3"));
    }
    require_c_layout(&psi, "psi")?;
    require_c_layout(&w, "w")?;
    let ps = psi
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("psi must be C-contiguous"))?;
    let ws = w
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("w must be C-contiguous"))?;
    Ok(py.detach(|| cavity::lid_driven_cavity(n, h, nu, dt, tol, max_iter, ps, ws)))
}

/// Tridiagonal solve. Inputs are read-only; the scratch copies stay here.
#[pyfunction]
fn thomas<'py>(
    py: Python<'py>,
    sub: PyReadonlyArray1<'py, f64>,
    diag: PyReadonlyArray1<'py, f64>,
    sup: PyReadonlyArray1<'py, f64>,
    rhs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Arr1<'py>> {
    let n = diag.len();
    if rhs.len() != n {
        return Err(PyValueError::new_err("rhs must match the diagonal length"));
    }
    if sub.len() != n.saturating_sub(1) || sup.len() != n.saturating_sub(1) {
        return Err(PyValueError::new_err(
            "sub- and super-diagonals must have length n-1",
        ));
    }
    let sub_v = to_vec1(&sub)?;
    let sup_v = to_vec1(&sup)?;
    let mut diag_v = to_vec1(&diag)?;
    let mut rhs_v = to_vec1(&rhs)?;
    let out = py.detach(|| relax::thomas(&sub_v, &mut diag_v, &sup_v, &mut rhs_v));
    match out {
        Some(x) => Ok(x.into_pyarray(py)),
        None => Err(map_err(LinalgError::Singular(
            "zero pivot in Thomas algorithm".to_string(),
        ))),
    }
}

/// One tridiagonal matrix against many right-hand sides; `rhs` is `n x m`
/// with one system per column and is overwritten with the solutions.
#[pyfunction]
fn thomas_batch(
    py: Python<'_>,
    sub: PyReadonlyArray1<'_, f64>,
    diag: PyReadonlyArray1<'_, f64>,
    sup: PyReadonlyArray1<'_, f64>,
    rhs: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let mut rhs = rhs;
    let n = diag.len();
    let (rows, cols) = (rhs.shape()[0], rhs.shape()[1]);
    if rows != n {
        return Err(PyValueError::new_err(
            "rhs must have one row per diagonal entry",
        ));
    }
    if sub.len() != n.saturating_sub(1) || sup.len() != n.saturating_sub(1) {
        return Err(PyValueError::new_err(
            "sub- and super-diagonals must have length n-1",
        ));
    }
    require_c_layout(&rhs, "rhs")?;
    let sub_v = to_vec1(&sub)?;
    let sup_v = to_vec1(&sup)?;
    let diag_v = to_vec1(&diag)?;
    let rs = rhs
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("rhs must be C-contiguous"))?;
    let ok = py.detach(|| relax::thomas_batch(&sub_v, &diag_v, &sup_v, rs, cols));
    if !ok {
        return Err(map_err(LinalgError::Singular(
            "zero pivot in Thomas algorithm".to_string(),
        )));
    }
    Ok(())
}

#[pyfunction]
fn fft<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let v = to_vec1(&a)?;
    let out = py.detach(|| fft_mod::fft(&v));
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn ifft<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let v = to_vec1(&a)?;
    let out = py.detach(|| fft_mod::ifft(&v));
    Ok(out.into_pyarray(py))
}

/// Convert whatever the user's right-hand side returned into a flat `Vec<f64>`,
/// accepting the same shapes `quadrivium.core.utils.as_vector` does.
fn rhs_to_vec(py: Python<'_>, obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        let v = to_vec1(&arr)?;
        if v.len() == n {
            return Ok(v);
        }
    }
    if let Ok(v) = obj.extract::<Vec<f64>>() {
        if v.len() == n {
            return Ok(v);
        }
    }
    if n == 1 {
        if let Ok(x) = obj.extract::<f64>() {
            return Ok(vec![x]);
        }
    }
    // Anything else (a 0-d array, a nested sequence) goes through NumPy, which
    // is the same coercion the pure-Python path applies.
    let np = py.import("numpy")?;
    let arr = np.call_method1("ravel", (np.call_method1("atleast_1d", (obj,))?,))?;
    let vals: Vec<f64> = arr.call_method1("astype", ("float64",))?.extract()?;
    if vals.len() != n {
        return Err(PyValueError::new_err(format!(
            "right-hand side returned {} components, expected {}",
            vals.len(),
            n
        )));
    }
    Ok(vals)
}

type OdeOut<'py> = (Arr1<'py>, Arr2<'py>, Arr2<'py>, usize, usize, usize);

#[pyfunction]
#[pyo3(signature = (f, c, a_flat, b_hi, b_lo, order, t0, tf, y0,
                    rtol, atol, h0, max_step, min_step, max_steps))]
#[allow(clippy::too_many_arguments)]
fn adaptive_rk<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    c: Vec<f64>,
    a_flat: Vec<f64>,
    b_hi: Vec<f64>,
    b_lo: Vec<f64>,
    order: f64,
    t0: f64,
    tf: f64,
    y0: PyReadonlyArray1<'py, f64>,
    rtol: f64,
    atol: f64,
    h0: Option<f64>,
    max_step: f64,
    min_step: f64,
    max_steps: usize,
) -> PyResult<OdeOut<'py>> {
    let y0v = to_vec1(&y0)?;
    let n = y0v.len();
    if n == 0 {
        return Err(PyValueError::new_err("y0 must have at least one component"));
    }
    if !t0.is_finite() || !tf.is_finite() || !(tf - t0).is_finite() {
        return Err(PyValueError::new_err("time interval must be finite"));
    }
    if y0v.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("y0 must be finite"));
    }
    if !rtol.is_finite()
        || !atol.is_finite()
        || rtol < 0.0
        || atol < 0.0
        || (rtol == 0.0 && atol == 0.0)
    {
        return Err(PyValueError::new_err(
            "rtol and atol must be finite, nonnegative, and not both zero",
        ));
    }
    if max_step.is_nan()
        || max_step <= 0.0
        || !min_step.is_finite()
        || min_step <= 0.0
        || max_step < min_step
    {
        return Err(PyValueError::new_err(
            "step bounds must satisfy 0 < min_step <= max_step",
        ));
    }
    if h0.is_some_and(|h| !h.is_finite() || h == 0.0) {
        return Err(PyValueError::new_err("h0 must be finite and nonzero"));
    }
    if max_steps == 0 {
        return Err(PyValueError::new_err("max_steps must be positive"));
    }
    let stages = b_hi.len();
    if stages == 0
        || c.len() != stages
        || b_lo.len() != stages
        || stages.checked_mul(stages) != Some(a_flat.len())
    {
        return Err(PyValueError::new_err(
            "inconsistent Runge-Kutta tableau dimensions",
        ));
    }
    if !order.is_finite()
        || order <= 0.0
        || c.iter()
            .chain(&a_flat)
            .chain(&b_hi)
            .chain(&b_lo)
            .any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err(
            "Runge-Kutta tableau must be finite with positive order",
        ));
    }
    let tab = ode::Tableau {
        c: &c,
        a_flat: &a_flat,
        b_hi: &b_hi,
        b_lo: &b_lo,
        order,
    };
    let ctl = ode::Controls {
        rtol,
        atol,
        h0,
        max_step,
        min_step,
        max_steps,
    };

    let mut calls = 0usize;
    let outcome = ode::adaptive_rk(&tab, &ctl, t0, tf, &y0v, |t, y| {
        calls += 1;
        let arr = PyArray1::from_slice(py, y);
        let res = f.call1((t, arr))?;
        rhs_to_vec(py, &res, n)
    })?;

    let sol = match outcome {
        Ok(sol) => sol,
        Err(u) => {
            // Tagged so the Python layer can re-raise it as StepSizeError with
            // the wording the pure-Python routine uses.
            return Err(PyRuntimeError::new_err(format!(
                "stepsize:step size underflow at t={:.6}: required h < {:.2e}; \
the problem is likely stiff -- try an implicit solver",
                u.t, u.min_step
            )));
        }
    };
    let rows = sol.ts.len();
    let drows = sol.dys.len() / n;
    Ok((
        sol.ts.into_pyarray(py),
        sol.ys.into_pyarray(py).reshape([rows, n])?,
        sol.dys.into_pyarray(py).reshape([drows, n])?,
        sol.accepted,
        sol.rejected,
        calls,
    ))
}

#[pyfunction]
fn matmul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Arr2<'py>> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (k2, n) = (b.shape()[0], b.shape()[1]);
    if k != k2 {
        return Err(PyValueError::new_err(format!(
            "shapes ({m},{k}) and ({k2},{n}) are not aligned"
        )));
    }
    // Both operands are read-only here, so a contiguous input is used in
    // place rather than copied.
    let a_owned;
    let av: &[f64] = match as_rows(&a) {
        Some(v) => v,
        None => {
            a_owned = to_vec2(&a)?;
            &a_owned
        }
    };
    let b_owned;
    let bv: &[f64] = match as_rows(&b) {
        Some(v) => v,
        None => {
            b_owned = to_vec2(&b)?;
            &b_owned
        }
    };
    let mut c = vec![0.0f64; m * n];
    py.detach(|| gemm::gemm(m, n, k, av, k, bv, n, &mut c, n));
    c.into_pyarray(py).reshape([m, n])
}

/// Generate a `PyArray1 -> PyArray1` binding for a scalar special function.
///
/// Every one of these releases the GIL for the duration of the map, so a
/// long-running transform on one thread never blocks the interpreter.
macro_rules! elementwise {
    ($name:ident, $kernel:path) => {
        #[pyfunction]
        fn $name<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Arr1<'py>> {
            let owned;
            let src = match x.as_slice() {
                Ok(src) => src,
                Err(_) => {
                    owned = to_vec1(&x)?;
                    &owned
                }
            };
            let mut out = vec![0.0f64; src.len()];
            py.detach(|| special::map_into(src, &mut out, $kernel));
            Ok(out.into_pyarray(py))
        }
    };
}

elementwise!(gamma, special::gamma_scalar);
elementwise!(log_gamma, special::log_gamma_scalar);
elementwise!(erf, special::erf_scalar);
elementwise!(erfc, special::erfc_scalar);

#[pymodule]
fn _quadrivium_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(plu, m)?)?;
    m.add_function(wrap_pyfunction!(householder_qr, m)?)?;
    m.add_function(wrap_pyfunction!(givens_qr, m)?)?;
    m.add_function(wrap_pyfunction!(svd_jacobi, m)?)?;
    m.add_function(wrap_pyfunction!(qr_least_squares, m)?)?;
    m.add_function(wrap_pyfunction!(forward_substitution, m)?)?;
    m.add_function(wrap_pyfunction!(back_substitution, m)?)?;
    m.add_function(wrap_pyfunction!(jacobi_eigen, m)?)?;
    m.add_function(wrap_pyfunction!(hessenberg_qr_iterate, m)?)?;
    m.add_function(wrap_pyfunction!(sor_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(lid_driven_cavity, m)?)?;
    m.add_function(wrap_pyfunction!(thomas, m)?)?;
    m.add_function(wrap_pyfunction!(thomas_batch, m)?)?;
    m.add_function(wrap_pyfunction!(fft, m)?)?;
    m.add_function(wrap_pyfunction!(ifft, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(log_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(erf, m)?)?;
    m.add_function(wrap_pyfunction!(erfc, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_rk, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(is_symmetric, m)?)?;
    Ok(())
}
