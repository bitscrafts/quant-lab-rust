//! Autocorrelation function.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::TimeSeriesError;

/// Compute the autocorrelation function for lags `0..=max_lag`.
///
/// # Formula
/// `rho_k = sum_{t=k+1..n} (y_t - ybar)(y_{t-k} - ybar) / sum_{t=1..n} (y_t - ybar)^2`
///
/// `acf(0)` is always `1.0` by construction.
///
/// # Arguments
/// * `data` - Time series.
/// * `max_lag` - Maximum lag to compute (inclusive). Must satisfy
///   `max_lag < data.len()` and `max_lag >= 0`. For meaningful ACF values the
///   series should have at least `max_lag + 1` observations.
///
/// # Errors
/// - [`TimeSeriesError::InsufficientData`] when `data` has fewer than 2
///   elements.
/// - [`TimeSeriesError::InvalidLag`] when `max_lag >= data.len()` (a lag
///   equal to or larger than the sample cannot be computed).
pub fn acf(data: &[f64], max_lag: usize) -> Result<Vec<f64>, TimeSeriesError> {
    let n = data.len();
    if n < 2 {
        return Err(TimeSeriesError::InsufficientData {
            required: 2,
            actual: n,
        });
    }
    if max_lag >= n {
        return Err(TimeSeriesError::InvalidLag {
            lag: max_lag,
            len: n,
        });
    }
    let ybar: f64 = data.iter().sum::<f64>() / n as f64;
    let mut denom = 0.0_f64;
    for &v in data {
        let d = v - ybar;
        denom += d * d;
    }
    if denom == 0.0 {
        // Constant series: ACF is 1 at lag 0 and 0 elsewhere (by convention).
        let mut out = vec![0.0_f64; max_lag + 1];
        out[0] = 1.0;
        return Ok(out);
    }
    let mut out = Vec::with_capacity(max_lag + 1);
    for k in 0..=max_lag {
        if k == 0 {
            out.push(1.0);
            continue;
        }
        let mut cov = 0.0_f64;
        for t in k..n {
            cov += (data[t] - ybar) * (data[t - k] - ybar);
        }
        out.push(cov / denom);
    }
    Ok(out)
}