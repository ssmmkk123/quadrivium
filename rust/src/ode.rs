//! Embedded Runge-Kutta driver with PI step size control.
//!
//! The right-hand side stays a Python callable -- that is the whole point of
//! the library's API -- so this kernel does not release the GIL. Measured on a
//! scalar decay problem, evaluating the callback is only about 7% of a solve;
//! the other 93% is the stage assembly, the embedded error estimate and the
//! controller, which is what moves here.
//!
//! The arithmetic follows the pure-Python routine operation for operation,
//! including the order the stage sums accumulate in, so both backends take the
//! same sequence of steps rather than merely converging to the same answer.

/// One embedded Runge-Kutta tableau in flattened form.
pub struct Tableau<'a> {
    pub c: &'a [f64],
    /// Row `i` of `A` occupies `a_flat[i * stages..i * stages + i]`.
    pub a_flat: &'a [f64],
    pub b_hi: &'a [f64],
    pub b_lo: &'a [f64],
    pub order: f64,
}

pub struct Controls {
    pub rtol: f64,
    pub atol: f64,
    pub h0: Option<f64>,
    pub max_step: f64,
    pub min_step: f64,
    pub max_steps: usize,
}

pub struct Solution {
    pub ts: Vec<f64>,
    /// Row-major `(steps + 1) x n`.
    pub ys: Vec<f64>,
    pub dys: Vec<f64>,
    pub accepted: usize,
    pub rejected: usize,
}

/// Raised when the controller drives the step below `min_step`.
pub struct StepSizeUnderflow {
    pub t: f64,
    pub min_step: f64,
}

/// Integrate `y' = f(t, y)` from `t0` to `tf`.
///
/// `eval_f` is the caller's bridge to the right-hand side; it returns `Err`
/// when the callback itself raised, which is propagated untouched.
pub fn adaptive_rk<E, F>(
    tab: &Tableau<'_>,
    ctl: &Controls,
    t0: f64,
    tf: f64,
    y0: &[f64],
    mut eval_f: F,
) -> Result<Result<Solution, StepSizeUnderflow>, E>
where
    F: FnMut(f64, &[f64]) -> Result<Vec<f64>, E>,
{
    let n = y0.len();
    let s = tab.b_hi.len();
    let mut y = y0.to_vec();
    let direction = if tf >= t0 { 1.0 } else { -1.0 };
    let mut t = t0;
    let mut h = match ctl.h0 {
        Some(v) => v.abs(),
        None => (tf - t0).abs() / 100.0,
    }
    .max(ctl.min_step)
    .min(ctl.max_step)
        * direction;

    let mut ts = vec![t];
    let mut ys = y.clone();
    let mut dys: Vec<f64> = Vec::new();
    let (mut accepted, mut rejected) = (0usize, 0usize);
    let mut err_prev = 1.0f64;

    let mut k = vec![0.0f64; s * n];
    let mut yi = vec![0.0f64; n];
    let mut acc_hi = vec![0.0f64; n];
    let mut acc_lo = vec![0.0f64; n];
    let mut y_hi = vec![0.0f64; n];

    for _ in 0..ctl.max_steps {
        if (t - tf) * direction >= 0.0 {
            break;
        }
        if h.abs() > (tf - t).abs() {
            h = tf - t;
        }
        if t + h == t {
            return Ok(Err(StepSizeUnderflow {
                t,
                min_step: ctl.min_step,
            }));
        }
        // Stages. Row i of A has exactly i entries in every tableau here.
        for i in 0..s {
            yi.copy_from_slice(&y);
            let row = &tab.a_flat[i * s..i * s + i];
            for (j, &aij) in row.iter().enumerate() {
                if aij != 0.0 {
                    let scale = h * aij;
                    for d in 0..n {
                        yi[d] += scale * k[j * n + d];
                    }
                }
            }
            let stage = eval_f(t + tab.c[i] * h, &yi)?;
            k[i * n..(i + 1) * n].copy_from_slice(&stage);
        }
        for d in 0..n {
            acc_hi[d] = 0.0;
            acc_lo[d] = 0.0;
        }
        for i in 0..s {
            let (bh, bl) = (tab.b_hi[i], tab.b_lo[i]);
            for d in 0..n {
                acc_hi[d] += bh * k[i * n + d];
                acc_lo[d] += bl * k[i * n + d];
            }
        }
        let mut sum_sq = 0.0;
        for d in 0..n {
            y_hi[d] = y[d] + h * acc_hi[d];
            let y_lo = y[d] + h * acc_lo[d];
            let scale = ctl.atol + ctl.rtol * y[d].abs().max(y_hi[d].abs());
            let e = (y_hi[d] - y_lo) / scale;
            sum_sq += e * e;
        }
        let err = (sum_sq / n as f64).sqrt();

        if !err.is_finite() {
            return Ok(Err(StepSizeUnderflow {
                t,
                min_step: ctl.min_step,
            }));
        }
        let fac;
        if err <= 1.0 {
            // k[0] is the slope at the start of the accepted step, so it is
            // the derivative belonging to the point already recorded.
            dys.extend_from_slice(&k[..n]);
            t += h;
            y.copy_from_slice(&y_hi);
            ts.push(t);
            ys.extend_from_slice(&y);
            accepted += 1;
            if (t - tf) * direction >= 0.0 {
                break;
            }
            fac = if err > 0.0 {
                0.9 * err.powf(-0.7 / tab.order) * err_prev.powf(0.4 / tab.order)
            } else {
                5.0
            };
            err_prev = err.max(1e-4);
        } else {
            if h.abs() <= ctl.min_step {
                return Ok(Err(StepSizeUnderflow {
                    t,
                    min_step: ctl.min_step,
                }));
            }
            rejected += 1;
            fac = 0.9 * err.powf(-1.0 / tab.order);
        }
        h *= 5.0f64.min(0.2f64.max(fac));
        if h.abs() > ctl.max_step {
            h = ctl.max_step * direction;
        }
        if h.abs() < ctl.min_step {
            return Ok(Err(StepSizeUnderflow {
                t,
                min_step: ctl.min_step,
            }));
        }
    }
    let last = eval_f(t, &y)?;
    dys.extend_from_slice(&last);
    Ok(Ok(Solution {
        ts,
        ys,
        dys,
        accepted,
        rejected,
    }))
}
