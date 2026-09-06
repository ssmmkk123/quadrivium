//! Moving arrays across the boundary without NumPy.
//!
//! The package's arrays are its own, implemented in C, and they export PEP 3118
//! buffers -- so that is what the kernels read and write.  Inputs are packed
//! into row-major `Vec`s (a `memcpy` for the contiguous case that dominates),
//! and outputs are allocated by the array core and filled in place, which keeps
//! exactly one copy on each side of a call.

use num_complex::Complex64;
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;
use pyo3::Borrowed;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyModule;
use std::ffi::CStr;

/// `complex128` as the buffer protocol spells it.  `Complex64` is two adjacent
/// `f64`s, which is exactly the "Zd" layout.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct C64(pub Complex64);

// SAFETY: `C64` is `#[repr(transparent)]` over `Complex64`, itself two
// contiguous f64 fields, matching the "Zd" format this accepts.
unsafe impl Element for C64 {
    fn is_compatible_format(format: &CStr) -> bool {
        matches!(format.to_bytes(), b"Zd" | b"=Zd" | b"<Zd")
    }
}

static ARRAY_MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// The compiled array core, imported once and cached.
fn core<'py>(py: Python<'py>) -> PyResult<&'py Bound<'py, PyModule>> {
    let cached = ARRAY_MODULE.get_or_try_init(py, || {
        PyModule::import(py, "quadrivium._qnp").map(|m| m.unbind())
    })?;
    Ok(cached.bind(py))
}

/// A read-only matrix, packed row-major.
pub struct Mat {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
}

fn dims<T: Element>(buf: &PyBuffer<T>, want: usize) -> PyResult<()> {
    if buf.dimensions() != want {
        return Err(PyValueError::new_err(format!(
            "expected a {want}-dimensional array, got {}",
            buf.dimensions()
        )));
    }
    Ok(())
}

/// Packs any strided layout into a row-major `Vec`.
fn pack<T: Element + Copy + Default>(py: Python<'_>, buf: &PyBuffer<T>) -> PyResult<Vec<T>> {
    if buf.is_c_contiguous() {
        return buf.to_vec(py);
    }
    let shape = buf.shape().to_vec();
    let strides = buf.strides().to_vec();
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut index = vec![0usize; shape.len()];
    let base = buf.buf_ptr() as *const u8;
    for _ in 0..total {
        let mut offset = 0isize;
        for (d, &i) in index.iter().enumerate() {
            offset += strides[d] * i as isize;
        }
        // SAFETY: the offset is built from the buffer's own shape and strides,
        // so it addresses an element the exporter guaranteed is live.
        out.push(unsafe { *(base.offset(offset) as *const T) });
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(out)
}

impl<'a, 'py> FromPyObject<'a, 'py> for Mat {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let buf = PyBuffer::<f64>::get(obj)?;
        dims(&buf, 2)?;
        let shape = buf.shape();
        let (rows, cols) = (shape[0], shape[1]);
        Ok(Mat {
            data: pack(obj.py(), &buf)?,
            rows,
            cols,
        })
    }
}

/// A read-only vector of `f64`.
pub struct Vec1(pub Vec<f64>);

impl Vec1 {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Vec1 {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let buf = PyBuffer::<f64>::get(obj)?;
        dims(&buf, 1)?;
        Ok(Vec1(pack(obj.py(), &buf)?))
    }
}

/// A read-only vector of `complex128`.
pub struct CVec1(pub Vec<Complex64>);

impl<'a, 'py> FromPyObject<'a, 'py> for CVec1 {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let buf = PyBuffer::<C64>::get(obj)?;
        dims(&buf, 1)?;
        let packed = if buf.is_c_contiguous() {
            buf.to_vec(obj.py())?
        } else {
            pack_complex(&buf)?
        };
        Ok(CVec1(packed.into_iter().map(|c| c.0).collect()))
    }
}

fn pack_complex(buf: &PyBuffer<C64>) -> PyResult<Vec<C64>> {
    let n = buf.shape()[0];
    let stride = buf.strides()[0];
    let base = buf.buf_ptr() as *const u8;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // SAFETY: offsets come from the exporter's own shape and stride.
        out.push(unsafe { *(base.offset(stride * i as isize) as *const C64) });
    }
    Ok(out)
}

impl Default for C64 {
    fn default() -> Self {
        C64(Complex64::new(0.0, 0.0))
    }
}

/// A matrix the kernel writes back into.  The contents are packed on the way in
/// and copied back by `flush`, so the caller sees the same array object it
/// passed, updated in place.
pub struct MutMat {
    buffer: PyBuffer<f64>,
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl MutMat {
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn flush(&self, py: Python<'_>) -> PyResult<()> {
        self.buffer.copy_from_slice(py, &self.data)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for MutMat {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let obj = &*obj;
        let buffer = PyBuffer::<f64>::get(obj)?;
        dims(&buffer, 2)?;
        if buffer.readonly() {
            return Err(PyBufferError::new_err("array is not writable"));
        }
        if !buffer.is_c_contiguous() {
            return Err(PyValueError::new_err("array must be C-contiguous"));
        }
        let shape = buffer.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let data = buffer.to_vec(obj.py())?;
        Ok(MutMat {
            buffer,
            data,
            rows,
            cols,
        })
    }
}

/// Builds a new 1-D `float64` array holding `data`.
pub fn new_vec<'py>(py: Python<'py>, data: &[f64]) -> PyResult<Bound<'py, PyAny>> {
    let array = core(py)?.call_method1("empty", (data.len(),))?;
    PyBuffer::<f64>::get(&array)?.copy_from_slice(py, data)?;
    Ok(array)
}

/// Builds a new `rows x cols` `float64` array holding `data` row-major.
pub fn new_mat<'py>(
    py: Python<'py>,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let array = core(py)?.call_method1("empty", ((rows, cols),))?;
    PyBuffer::<f64>::get(&array)?.copy_from_slice(py, data)?;
    Ok(array)
}

/// Builds a new 1-D `int64` array.
pub fn new_int_vec<'py>(py: Python<'py>, data: &[i64]) -> PyResult<Bound<'py, PyAny>> {
    let module = core(py)?;
    let dtype = module.getattr("int64")?;
    let array = module.call_method1("empty", (data.len(), dtype))?;
    PyBuffer::<i64>::get(&array)?.copy_from_slice(py, data)?;
    Ok(array)
}

/// Builds a new 1-D `complex128` array.
pub fn new_complex_vec<'py>(py: Python<'py>, data: &[Complex64]) -> PyResult<Bound<'py, PyAny>> {
    let module = core(py)?;
    let dtype = module.getattr("complex128")?;
    let array = module.call_method1("empty", (data.len(), dtype))?;
    // SAFETY: `C64` is `#[repr(transparent)]` over `Complex64`, so the two
    // slices have identical layout.
    let view = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const C64, data.len()) };
    PyBuffer::<C64>::get(&array)?.copy_from_slice(py, view)?;
    Ok(array)
}

/// Coerces a value returned by user code into a flat `Vec<f64>` of length `n`,
/// accepting the same shapes `quadrivium.core.utils.as_vector` does.
pub fn to_flat_vec(py: Python<'_>, obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<f64>> {
    if let Ok(v) = obj.extract::<Vec1>() {
        if v.0.len() == n {
            return Ok(v.0);
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
    // Anything else (a 0-d array, a nested sequence) goes through the array
    // core, which is the same coercion the pure-Python path applies.
    let module = PyModule::import(py, "quadrivium.numeric")?;
    let flat = module.call_method1("ravel", (module.call_method1("atleast_1d", (obj,))?,))?;
    let values: Vec<f64> = flat
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if values.len() != n {
        return Err(PyValueError::new_err(format!(
            "right-hand side returned {} components, expected {}",
            values.len(),
            n
        )));
    }
    Ok(values)
}
