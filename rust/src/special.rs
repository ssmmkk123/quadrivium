//! Vectorized special functions.
//!
//! Each kernel reproduces the algorithm of its pure-Python twin -- the same
//! Lanczos coefficients, the same recurrence thresholds -- so the two backends
//! agree to the last few ulp. What changes is the loop: Python evaluates these
//! one element at a time through the interpreter, while these run the whole
//! array in compiled code and fan out across cores once it is large enough.

use rayon::prelude::*;
use std::f64::consts::PI;

/// Arrays at least this long are worth splitting across threads; below it the
/// scheduling overhead dominates the arithmetic.
const PAR_THRESHOLD: usize = 8192;

const LANCZOS_G: f64 = 7.0;
// Transcribed verbatim from the table in `quadrivium.special.functions` so both
// backends evaluate the identical approximation. Clippy flags the extra digits;
// they are kept because the point is to match that table exactly.
#[allow(clippy::excessive_precision)]
const LANCZOS_COEF: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

pub fn log_gamma_scalar(v: f64) -> f64 {
    if v < 0.5 {
        // Reflection, so the series is only ever evaluated on its good side.
        (PI / (PI * v).sin().abs()).ln() - log_gamma_scalar(1.0 - v)
    } else {
        let z = v - 1.0;
        let mut a = LANCZOS_COEF[0];
        for (i, c) in LANCZOS_COEF.iter().enumerate().skip(1) {
            a += c / (z + i as f64);
        }
        let t = z + LANCZOS_G + 0.5;
        0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
    }
}

pub fn gamma_scalar(v: f64) -> f64 {
    if v == v.floor() && v <= 0.0 {
        f64::INFINITY
    } else if v < 0.5 {
        PI / ((PI * v).sin() * gamma_scalar(1.0 - v))
    } else {
        log_gamma_scalar(v).exp()
    }
}

#[inline]
pub fn erf_scalar(x: f64) -> f64 {
    libm::erf(x)
}

#[inline]
pub fn erfc_scalar(x: f64) -> f64 {
    libm::erfc(x)
}

/// Apply a scalar kernel across a slice, in parallel for large inputs.
pub fn map_into(src: &[f64], dst: &mut [f64], f: fn(f64) -> f64) {
    if src.len() >= PAR_THRESHOLD {
        dst.par_iter_mut()
            .zip(src.par_iter())
            .for_each(|(o, &v)| *o = f(v));
    } else {
        for (o, &v) in dst.iter_mut().zip(src.iter()) {
            *o = f(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_matches_known_values() {
        // Gamma(n) = (n-1)! and Gamma(1/2) = sqrt(pi).
        let cases = [
            (1.0, 1.0),
            (2.0, 1.0),
            (5.0, 24.0),
            (10.0, 362880.0),
            (0.5, PI.sqrt()),
        ];
        for (x, want) in cases {
            let got = gamma_scalar(x);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1.0),
                "gamma({x}) = {got}, want {want}"
            );
        }
        assert!(gamma_scalar(0.0).is_infinite());
        assert!(gamma_scalar(-3.0).is_infinite());
    }

    #[test]
    fn log_gamma_matches_gamma() {
        for i in 1..40 {
            let x = i as f64 * 0.37;
            let a = log_gamma_scalar(x);
            let b = gamma_scalar(x).abs().ln();
            assert!((a - b).abs() < 1e-9, "log_gamma({x}): {a} vs {b}");
        }
    }

    #[test]
    fn erf_is_odd_and_bounded() {
        for i in 0..30 {
            let x = i as f64 * 0.2;
            assert!((erf_scalar(x) + erf_scalar(-x)).abs() < 1e-15);
            assert!((erf_scalar(x) + erfc_scalar(x) - 1.0).abs() < 1e-14);
        }
        assert!((erf_scalar(0.0)).abs() < 1e-16);
        assert!((erf_scalar(6.0) - 1.0).abs() < 1e-15);
    }
}
