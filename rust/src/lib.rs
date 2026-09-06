//! Compiled kernels backing `quadrivium`.
//!
//! Every function exported here has a pure-Python twin in the package. The
//! Python side picks whichever is available, so this extension is always
//! optional: it changes how fast the library runs, never what it computes.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod bridge;
use bridge::{Mat, MutMat, Vec1};

// Every array crossing the boundary is one of the package's own arrays, so the
// return types are plain Python objects built by the array core.
type Arr<'py> = Bound<'py, PyAny>;
type PluOut<'py> = (Arr<'py>, Arr<'py>);
type QrOut<'py> = (Arr<'py>, Arr<'py>);
type EigenOut<'py> = (Arr<'py>, Arr<'py>, usize, bool);

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

fn square_size(a: &Mat) -> PyResult<usize> {
    if !a.is_square() {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    Ok(a.rows)
}

#[pyfunction]
fn cholesky<'py>(py: Python<'py>, a: Mat) -> PyResult<Arr<'py>> {
    let n = square_size(&a)?;
    let mut buf = a.data;
    py.detach(|| direct::cholesky_inplace(n, &mut buf))
        .map_err(map_err)?;
    bridge::new_mat(py, &buf, n, n)
}

#[pyfunction]
fn plu<'py>(py: Python<'py>, a: Mat) -> PyResult<PluOut<'py>> {
    let n = square_size(&a)?;
    let mut buf = a.data;
    let mut perm = vec![0usize; n];
    py.detach(|| direct::plu_inplace(n, &mut buf, &mut perm))
        .map_err(map_err)?;
    let permi: Vec<i64> = perm.into_iter().map(|p| p as i64).collect();
    Ok((
        bridge::new_int_vec(py, &permi)?,
        bridge::new_mat(py, &buf, n, n)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (a, want_q=true))]
fn householder_qr<'py>(py: Python<'py>, a: Mat, want_q: bool) -> PyResult<QrOut<'py>> {
    let (m, n) = (a.rows, a.cols);
    let mut r = a.data;
    let mut q = vec![0.0f64; if want_q { m * m } else { 0 }];
    py.detach(|| direct::householder_qr(m, n, &mut r, &mut q, want_q));
    Ok((
        bridge::new_mat(py, &q, if want_q { m } else { 0 }, m)?,
        bridge::new_mat(py, &r, m, n)?,
    ))
}

/// QR by Givens rotations; returns `(Q, R)` with `Q` an `m x m` accumulator.
#[pyfunction]
fn givens_qr<'py>(py: Python<'py>, a: Mat) -> PyResult<QrOut<'py>> {
    let (m, n) = (a.rows, a.cols);
    let mut r = a.data;
    let mut q = vec![0.0f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    py.detach(|| direct::givens_qr(m, n, &mut r, &mut q));
    Ok((
        bridge::new_mat(py, &q, m, m)?,
        bridge::new_mat(py, &r, m, n)?,
    ))
}

type SvdOut<'py> = (Arr<'py>, Arr<'py>, usize);

/// One-sided Jacobi SVD; returns `(W, V, sweeps)` with `W = U * S`. The caller
/// reads the singular values off the column norms of `W`, exactly as the
/// pure-Python routine does.
#[pyfunction]
#[pyo3(signature = (a, tol=1e-13, max_sweeps=60))]
fn svd_jacobi<'py>(py: Python<'py>, a: Mat, tol: f64, max_sweeps: usize) -> PyResult<SvdOut<'py>> {
    let (m, n) = (a.rows, a.cols);
    if m < n {
        return Err(PyValueError::new_err(
            "one-sided Jacobi requires at least as many rows as columns",
        ));
    }
    let mut w = a.data;
    let mut v = vec![0.0f64; n * n];
    let sweeps = py.detach(|| direct::svd_jacobi(m, n, &mut w, &mut v, tol, max_sweeps));
    Ok((
        bridge::new_mat(py, &w, m, n)?,
        bridge::new_mat(py, &v, n, n)?,
        sweeps,
    ))
}

#[pyfunction]
fn qr_least_squares<'py>(py: Python<'py>, a: Mat, b: Vec1) -> PyResult<Arr<'py>> {
    let (m, n) = (a.rows, a.cols);
    if m < n {
        return Err(PyValueError::new_err(
            "least squares requires at least as many rows as columns",
        ));
    }
    if b.len() != m {
        return Err(PyValueError::new_err("rhs must match the matrix row count"));
    }
    if n == 0 {
        return bridge::new_vec(py, &[]);
    }
    let mut a = a.data;
    let mut rhs = b.0;
    py.detach(|| direct::qr_least_squares(m, n, &mut a, &mut rhs))
        .map_err(map_err)?;
    rhs.truncate(n);
    bridge::new_vec(py, &rhs)
}

#[pyfunction]
#[pyo3(signature = (l, b, unit_diagonal=false))]
fn forward_substitution<'py>(
    py: Python<'py>,
    l: Mat,
    b: Vec1,
    unit_diagonal: bool,
) -> PyResult<Arr<'py>> {
    let n = square_size(&l)?;
    if b.len() != n {
        return Err(PyValueError::new_err("rhs must match the matrix size"));
    }
    let mut x = b.0;
    py.detach(|| direct::forward_substitution(n, &l.data, n, &mut x, unit_diagonal))
        .map_err(map_err)?;
    bridge::new_vec(py, &x)
}

#[pyfunction]
#[pyo3(signature = (u, b, unit_diagonal=false, transposed=false))]
fn back_substitution<'py>(
    py: Python<'py>,
    u: Mat,
    b: Vec1,
    unit_diagonal: bool,
    transposed: bool,
) -> PyResult<Arr<'py>> {
    let n = square_size(&u)?;
    if b.len() != n {
        return Err(PyValueError::new_err("rhs must match the matrix size"));
    }
    let mut x = b.0;
    let ub: &[f64] = &u.data;
    py.detach(|| {
        if transposed {
            // `u` is the lower-triangular L; solve L' x = b in place.
            direct::back_substitution_trans(n, ub, n, &mut x, unit_diagonal)
        } else {
            direct::back_substitution(n, ub, n, &mut x, unit_diagonal)
        }
    })
    .map_err(map_err)?;
    bridge::new_vec(py, &x)
}

#[pyfunction]
#[pyo3(signature = (a, atol=1e-12, rtol=1e-5))]
fn is_symmetric(py: Python<'_>, a: Mat, atol: f64, rtol: f64) -> PyResult<bool> {
    if !a.is_square() {
        return Ok(false);
    }
    let n = a.rows;
    Ok(py.detach(|| direct::is_symmetric(n, &a.data, atol, rtol)))
}

#[pyfunction]
#[pyo3(signature = (a, tol=1e-12, max_sweeps=100))]
fn jacobi_eigen<'py>(
    py: Python<'py>,
    a: Mat,
    tol: f64,
    max_sweeps: usize,
) -> PyResult<EigenOut<'py>> {
    let n = square_size(&a)?;
    let mut d = a.data;
    let mut v = vec![0.0f64; n * n];
    let (sweeps, conv) = py.detach(|| eigen::jacobi_eigen(n, &mut d, &mut v, tol, max_sweeps));
    let vals: Vec<f64> = (0..n).map(|i| d[i * n + i]).collect();
    Ok((
        bridge::new_vec(py, &vals)?,
        bridge::new_mat(py, &v, n, n)?,
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
    h: MutMat,
    v: Option<MutMat>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(usize, bool)> {
    let mut h = h;
    let n = h.rows;
    if h.cols != n {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    let result = match v {
        Some(mut v) => {
            if v.shape() != [n, n] {
                return Err(PyValueError::new_err("V must have the same shape as H"));
            }
            let outcome = py.detach(|| {
                eigen::hessenberg_qr_iterate(n, &mut h.data, Some(&mut v.data), tol, max_iter)
            });
            v.flush(py)?;
            outcome
        }
        None => py.detach(|| eigen::hessenberg_qr_iterate(n, &mut h.data, None, tol, max_iter)),
    };
    h.flush(py)?;
    Ok(result)
}

/// SOR / Gauss-Seidel relaxation for the 5-point Poisson stencil.
///
/// `u` is updated in place. Returns `(iterations, converged, residuals)`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (u, f, beta2, dx2, omega, tol=1e-10, max_iter=20000))]
fn sor_poisson<'py>(
    py: Python<'py>,
    u: MutMat,
    f: Mat,
    beta2: f64,
    dx2: f64,
    omega: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<(usize, bool, Arr<'py>)> {
    let mut u = u;
    let (rows, cols) = (u.rows, u.cols);
    if f.shape() != [rows, cols] {
        return Err(PyValueError::new_err("f must have the same shape as u"));
    }
    if rows < 2 || cols < 2 {
        return Err(PyValueError::new_err("grid must be at least 2x2"));
    }
    let fv = f.data;
    let (it, conv, res) = py.detach(|| {
        relax::sor_poisson(
            rows - 1,
            cols - 1,
            &mut u.data,
            &fv,
            beta2,
            dx2,
            omega,
            tol,
            max_iter,
        )
    });
    u.flush(py)?;
    Ok((it, conv, bridge::new_vec(py, &res)?))
}

/// Steady lid-driven cavity flow; `psi` and `w` are filled in place.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (psi, w, h, nu, dt, tol, max_iter))]
fn lid_driven_cavity(
    py: Python<'_>,
    psi: MutMat,
    w: MutMat,
    h: f64,
    nu: f64,
    dt: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<usize> {
    let (mut psi, mut w) = (psi, w);
    let n = psi.rows;
    if psi.cols != n || w.shape() != [n, n] {
        return Err(PyValueError::new_err(
            "psi and w must both be square and the same size",
        ));
    }
    if n < 3 {
        return Err(PyValueError::new_err("grid must be at least 3x3"));
    }
    let iterations = py.detach(|| {
        cavity::lid_driven_cavity(n, h, nu, dt, tol, max_iter, &mut psi.data, &mut w.data)
    });
    psi.flush(py)?;
    w.flush(py)?;
    Ok(iterations)
}

/// Tridiagonal solve. Inputs are read-only; the scratch copies stay here.
#[pyfunction]
fn thomas<'py>(
    py: Python<'py>,
    sub: Vec1,
    diag: Vec1,
    sup: Vec1,
    rhs: Vec1,
) -> PyResult<Arr<'py>> {
    let n = diag.len();
    if rhs.len() != n {
        return Err(PyValueError::new_err("rhs must match the diagonal length"));
    }
    if sub.len() != n.saturating_sub(1) || sup.len() != n.saturating_sub(1) {
        return Err(PyValueError::new_err(
            "sub- and super-diagonals must have length n-1",
        ));
    }
    let sub_v = sub.0;
    let sup_v = sup.0;
    let mut diag_v = diag.0;
    let mut rhs_v = rhs.0;
    let out = py.detach(|| relax::thomas(&sub_v, &mut diag_v, &sup_v, &mut rhs_v));
    match out {
        Some(x) => bridge::new_vec(py, &x),
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
    sub: Vec1,
    diag: Vec1,
    sup: Vec1,
    rhs: MutMat,
) -> PyResult<()> {
    let mut rhs = rhs;
    let n = diag.len();
    let (rows, cols) = (rhs.rows, rhs.cols);
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
    let sub_v = sub.0;
    let sup_v = sup.0;
    let diag_v = diag.0;
    let ok = py.detach(|| relax::thomas_batch(&sub_v, &diag_v, &sup_v, &mut rhs.data, cols));
    if !ok {
        return Err(map_err(LinalgError::Singular(
            "zero pivot in Thomas algorithm".to_string(),
        )));
    }
    rhs.flush(py)?;
    Ok(())
}

#[pyfunction]
fn fft<'py>(py: Python<'py>, a: bridge::CVec1) -> PyResult<Arr<'py>> {
    let v = a.0;
    let out = py.detach(|| fft_mod::fft(&v));
    bridge::new_complex_vec(py, &out)
}

#[pyfunction]
fn ifft<'py>(py: Python<'py>, a: bridge::CVec1) -> PyResult<Arr<'py>> {
    let v = a.0;
    let out = py.detach(|| fft_mod::ifft(&v));
    bridge::new_complex_vec(py, &out)
}

type OdeOut<'py> = (Arr<'py>, Arr<'py>, Arr<'py>, usize, usize, usize);

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
    y0: Vec1,
    rtol: f64,
    atol: f64,
    h0: Option<f64>,
    max_step: f64,
    min_step: f64,
    max_steps: usize,
) -> PyResult<OdeOut<'py>> {
    let y0v = y0.0;
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
        let arr = bridge::new_vec(py, y)?;
        let res = f.call1((t, arr))?;
        bridge::to_flat_vec(py, &res, n)
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
        bridge::new_vec(py, &sol.ts)?,
        bridge::new_mat(py, &sol.ys, rows, n)?,
        bridge::new_mat(py, &sol.dys, drows, n)?,
        sol.accepted,
        sol.rejected,
        calls,
    ))
}

#[pyfunction]
fn matmul<'py>(py: Python<'py>, a: Mat, b: Mat) -> PyResult<Arr<'py>> {
    let (m, k) = (a.rows, a.cols);
    let (k2, n) = (b.rows, b.cols);
    if k != k2 {
        return Err(PyValueError::new_err(format!(
            "shapes ({m},{k}) and ({k2},{n}) are not aligned"
        )));
    }
    let mut c = vec![0.0f64; m * n];
    py.detach(|| gemm::gemm(m, n, k, &a.data, k, &b.data, n, &mut c, n));
    bridge::new_mat(py, &c, m, n)
}

/// Generate an array-in, array-out binding for a scalar special function.
///
/// Every one of these releases the GIL for the duration of the map, so a
/// long-running transform on one thread never blocks the interpreter.
macro_rules! elementwise {
    ($name:ident, $kernel:path) => {
        #[pyfunction]
        fn $name<'py>(py: Python<'py>, x: Vec1) -> PyResult<Arr<'py>> {
            let src = x.0;
            let mut out = vec![0.0f64; src.len()];
            py.detach(|| special::map_into(&src, &mut out, $kernel));
            bridge::new_vec(py, &out)
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
