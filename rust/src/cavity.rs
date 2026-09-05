//! Steady lid-driven cavity flow in vorticity-streamfunction form.
//!
//! The Python twin is already written in whole-array NumPy, so the cost is not
//! per-element interpreter overhead but the sheer number of sweeps: a converged
//! solve at Re = 100 runs tens of thousands of outer steps, each with an inner
//! Jacobi solve for the stream function, and every one of those allocates a
//! fresh temporary the size of the grid. Running the whole solve here keeps two
//! buffers live for the duration and touches nothing else.
//!
//! Operation order matches the Python expression by expression, so both
//! backends produce bit-identical grids.

/// Solve to steady state, writing the stream function into `psi` and the
/// vorticity into `w` (both row-major `n x n`, zeroed by the caller).
/// Returns the number of outer iterations taken.
#[allow(clippy::too_many_arguments)]
pub fn lid_driven_cavity(
    n: usize,
    h: f64,
    nu: f64,
    dt: f64,
    tol: f64,
    max_iter: usize,
    psi: &mut [f64],
    w: &mut [f64],
) -> usize {
    let hh = h * h;
    let two_h = 2.0 * h;
    let mut psi_old = vec![0.0f64; n * n];
    let mut w_new = vec![0.0f64; n * n];

    for it in 1..=max_iter {
        // --- Stream function from vorticity: lap psi = -omega, Jacobi sweeps.
        for _ in 0..30 {
            psi_old.copy_from_slice(psi);
            let mut delta = 0.0f64;
            for i in 1..n - 1 {
                let r = i * n;
                for j in 1..n - 1 {
                    let v = 0.25
                        * (psi_old[r + n + j]
                            + psi_old[r - n + j]
                            + psi_old[r + j + 1]
                            + psi_old[r + j - 1]
                            + hh * w[r + j]);
                    psi[r + j] = v;
                    let d = (v - psi_old[r + j]).abs();
                    if d > delta || d.is_nan() {
                        delta = d;
                    }
                }
            }
            if delta < 1e-10 {
                break;
            }
        }

        // --- Thom's wall vorticity. Rows first, then columns, so the corners
        // take their value from the column formulas exactly as in Python.
        let last = n - 1;
        for j in 0..n {
            w_new[j] = 2.0 * (psi[j] - psi[n + j]) / hh;
            w_new[last * n + j] =
                2.0 * (psi[last * n + j] - psi[(last - 1) * n + j]) / hh - 2.0 / h;
        }
        for i in 0..n {
            let r = i * n;
            w_new[r] = 2.0 * (psi[r] - psi[r + 1]) / hh;
            w_new[r + last] = 2.0 * (psi[r + last] - psi[r + last - 1]) / hh;
        }

        // --- Vorticity transport on the interior.
        for i in 1..n - 1 {
            let r = i * n;
            for j in 1..n - 1 {
                let u = (psi[r + n + j] - psi[r - n + j]) / two_h;
                let v = -((psi[r + j + 1] - psi[r + j - 1]) / two_h);
                let wx = (w[r + j + 1] - w[r + j - 1]) / two_h;
                let wy = (w[r + n + j] - w[r - n + j]) / two_h;
                let lapw = (w[r + n + j] + w[r - n + j] + w[r + j + 1] + w[r + j - 1]
                    - 4.0 * w[r + j])
                    / hh;
                w_new[r + j] = w[r + j] + dt * (-u * wx - v * wy + nu * lapw);
            }
        }

        let mut change = 0.0f64;
        for (a, b) in w_new.iter().zip(w.iter()) {
            let d = (a - b).abs();
            if d > change || d.is_nan() {
                change = d;
            }
        }
        w.copy_from_slice(&w_new);
        if change < tol {
            return it;
        }
    }
    max_iter
}
