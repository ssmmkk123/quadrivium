//! Compiled kernels backing `quadrivium`.
//!
//! Every function exported here has a pure-Python twin in the package. The
//! Python side picks whichever is available, so this extension is always
//! optional: it changes how fast the library runs, never what it computes.

use numpy::prelude::*;
use numpy::{Complex64, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
mod direct;
mod eigen;
#[path = "fft.rs"]
mod fft_mod;
mod gemm;
mod ode;
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
    a.as_slice().ok()
}

/// Row-major copy of a possibly non-contiguous input, as a flat `Vec`.
///
/// The common case is a C-contiguous array, where this is a single `memcpy`.
/// Walking the array with 2-D indexing instead costs an indexed load per
/// element and showed up as milliseconds on the input path of every
/// factorization.
fn to_vec2(a: &PyReadonlyArray2<f64>) -> Vec<f64> {
    if let Ok(s) = a.as_slice() {
        return s.to_vec();
    }
    let v = a.as_array();
    let (m, n) = (v.nrows(), v.ncols());
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            out.push(v[[i, j]]);
        }
    }
    out
}

#[pyfunction]
fn cholesky<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (m, n) = {
        let v = a.as_array();
        (v.nrows(), v.ncols())
    };
    if m != n {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    let mut buf = to_vec2(&a);
    py.detach(|| direct::cholesky_inplace(n, &mut buf))
        .map_err(map_err)?;
    buf.into_pyarray(py).reshape([n, n])
}

#[pyfunction]
fn plu<'py>(py: Python<'py>, a: PyReadonlyArray2<'py, f64>) -> PyResult<PluOut<'py>> {
    let (m, n) = {
        let v = a.as_array();
        (v.nrows(), v.ncols())
    };
    if m != n {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    let mut buf = to_vec2(&a);
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
    let (m, n) = {
        let v = a.as_array();
        (v.nrows(), v.ncols())
    };
    let mut r = to_vec2(&a);
    let mut q = vec![0.0f64; if want_q { m * m } else { 0 }];
    py.detach(|| direct::householder_qr(m, n, &mut r, &mut q, want_q));
    Ok((
        q.into_pyarray(py)
            .reshape([if want_q { m } else { 0 }, m])?,
        r.into_pyarray(py).reshape([m, n])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (l, b, unit_diagonal=false))]
fn forward_substitution<'py>(
    py: Python<'py>,
    l: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    unit_diagonal: bool,
) -> PyResult<Arr1<'py>> {
    let n = l.as_array().nrows();
    let mut x = b.as_array().to_vec();
    let owned;
    let lb: &[f64] = match as_rows(&l) {
        Some(v) => v,
        None => {
            owned = to_vec2(&l);
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
    let n = u.as_array().nrows();
    let mut x = b.as_array().to_vec();
    let owned;
    let ub: &[f64] = match as_rows(&u) {
        Some(v) => v,
        None => {
            owned = to_vec2(&u);
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
fn is_symmetric(py: Python<'_>, a: PyReadonlyArray2<'_, f64>, atol: f64, rtol: f64) -> bool {
    let arr = a.as_array();
    let (m, n) = (arr.nrows(), arr.ncols());
    if m != n {
        return false;
    }
    match as_rows(&a) {
        Some(v) => py.detach(|| direct::is_symmetric(n, v, atol, rtol)),
        None => {
            let owned = to_vec2(&a);
            py.detach(|| direct::is_symmetric(n, &owned, atol, rtol))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (a, tol=1e-12, max_sweeps=100))]
fn jacobi_eigen<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_sweeps: usize,
) -> PyResult<EigenOut<'py>> {
    let n = a.as_array().nrows();
    let mut d = to_vec2(&a);
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

#[pyfunction]
fn fft<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let v = a.as_array().to_vec();
    let out = py.detach(|| fft_mod::fft(&v));
    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn ifft<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    let v = a.as_array().to_vec();
    let out = py.detach(|| fft_mod::ifft(&v));
    Ok(out.into_pyarray(py))
}

/// Convert whatever the user's right-hand side returned into a flat `Vec<f64>`,
/// accepting the same shapes `quadrivium.core.utils.as_vector` does.
fn rhs_to_vec(py: Python<'_>, obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<f64>>() {
        let v = arr.as_array().to_vec();
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
    let y0v = y0.as_array().to_vec();
    let n = y0v.len();
    if n == 0 {
        return Err(PyValueError::new_err("y0 must have at least one component"));
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
    let (m, k) = {
        let v = a.as_array();
        (v.nrows(), v.ncols())
    };
    let (k2, n) = {
        let v = b.as_array();
        (v.nrows(), v.ncols())
    };
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
            a_owned = to_vec2(&a);
            &a_owned
        }
    };
    let b_owned;
    let bv: &[f64] = match as_rows(&b) {
        Some(v) => v,
        None => {
            b_owned = to_vec2(&b);
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
        fn $name<'py>(py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> Arr1<'py> {
            let src = x.as_array().to_vec();
            let mut out = vec![0.0f64; src.len()];
            py.detach(|| special::map_into(&src, &mut out, $kernel));
            out.into_pyarray(py)
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
    m.add_function(wrap_pyfunction!(forward_substitution, m)?)?;
    m.add_function(wrap_pyfunction!(back_substitution, m)?)?;
    m.add_function(wrap_pyfunction!(jacobi_eigen, m)?)?;
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
