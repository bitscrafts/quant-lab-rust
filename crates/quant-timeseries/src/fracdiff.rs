//! Fractional differentiation (López de Prado, AFML Ch.5).
//!
//! See `README.md` in this directory for the module overview.

use crate::adf::adf_test;
use crate::error::TimeSeriesError;

/// Compute fixed-width fractional differentiation weights for `(1 - B)^d`.
///
/// The weights follow the recurrence
///
/// ```text
/// w_0 = 1
/// w_k = -w_{k-1} * (d - k + 1) / k   for k >= 1
/// ```
///
/// Weights are accumulated until `|w_k| < threshold`, at which point we stop
/// (fixed-width window). The last weight retained is the first one below
/// the threshold, so the window includes it.
///
/// # Arguments
/// * `d` - Fractional differencing order. Must be in `[0, 2]`.
/// * `threshold` - Absolute weight cutoff (e.g. `1e-4`). Must be positive.
///
/// # Returns
/// Vector of weights `[w_0, w_1, ..., w_K]` with `w_0 = 1`.
///
/// # Errors
/// - [`TimeSeriesError::InvalidParam`] when `d` is outside `[0, 2]` or
///   `threshold` is non-positive.
pub fn ffd_weights(d: f64, threshold: f64) -> Result<Vec<f64>, TimeSeriesError> {
    if !(0.0..=2.0).contains(&d) {
        return Err(TimeSeriesError::InvalidParam(format!(
            "d must be in [0, 2], got {d}"
        )));
    }
    if threshold <= 0.0 {
        return Err(TimeSeriesError::InvalidParam(format!(
            "threshold must be positive, got {threshold}"
        )));
    }
    let mut weights = vec![1.0_f64];
    let mut w = 1.0_f64;
    let mut k = 1_usize;
    loop {
        w = -w * (d - (k as f64) + 1.0) / k as f64;
        // Stop before including a weight below the threshold: the window is
        // the longest prefix of weights that are all >= threshold in absolute
        // value. This gives [1.0] for d = 0 (w_1 = 0 < threshold) and
        // [1.0, -1.0] for d = 1 (w_2 = 0 < threshold).
        if w.abs() < threshold {
            break;
        }
        weights.push(w);
        // Safety valve: never grow without bound.
        if k > 10_000 {
            break;
        }
        k += 1;
    }
    Ok(weights)
}

/// Apply fixed-width fractional differentiation to a series.
///
/// `y_t^d = sum_{k=0..K} w_k * y_{t-k}`
///
/// The first `weights.len() - 1` values are dropped (insufficient history for
/// a full window). With `d = 0` the weights are `[1.0]` and the output equals
/// the input (full memory); with `d = 1` the weights are `[1.0, -1.0]` (plus a
/// small tail) and the output is the first difference.
///
/// # Arguments
/// * `data` - Input series.
/// * `d` - Fractional differencing order in `[0, 2]`.
/// * `threshold` - Weight cutoff passed to [`ffd_weights`]. Must be positive.
///
/// # Returns
/// Fractionally differenced series. Length is `data.len() - weights.len() + 1`.
///
/// # Errors
/// Propagates [`TimeSeriesError::InvalidParam`] from [`ffd_weights`].
pub fn frac_diff(data: &[f64], d: f64, threshold: f64) -> Result<Vec<f64>, TimeSeriesError> {
    let weights = ffd_weights(d, threshold)?;
    let wlen = weights.len();
    if data.len() < wlen {
        return Err(TimeSeriesError::InsufficientData {
            required: wlen,
            actual: data.len(),
        });
    }
    let out_len = data.len() - wlen + 1;
    let mut out = Vec::with_capacity(out_len);
    for t in (wlen - 1)..data.len() {
        let mut acc = 0.0_f64;
        for (k, &w) in weights.iter().enumerate() {
            acc += w * data[t - k];
        }
        out.push(acc);
    }
    Ok(out)
}

/// Find the minimum `d` that makes the series stationary (ADF rejects the
/// unit root at 5%).
///
/// Binary search over `d` in `[0, 1]`. At each candidate `d`, compute
/// `frac_diff(data, d, threshold)` and run the ADF test with `lags = 1`.
/// The smallest `d` for which the ADF statistic is below the MacKinnon
/// critical value (-2.86) is returned, to `tolerance` precision.
///
/// # Arguments
/// * `data` - Input series.
/// * `threshold` - Weight cutoff for `frac_diff`. Must be positive.
/// * `tolerance` - Precision for the binary search (e.g. `0.01`).
///
/// # Returns
/// The minimum `d` in `[0, 1]` for which the fractionally differenced series
/// is stationary. Returns `1.0` if even `d = 1` does not yield stationarity
/// (rare; full differencing almost always suffices).
///
/// # Errors
/// Propagates errors from `frac_diff` and `adf_test`.
pub fn find_min_d(data: &[f64], threshold: f64, tolerance: f64) -> Result<f64, TimeSeriesError> {
    if tolerance <= 0.0 {
        return Err(TimeSeriesError::InvalidParam(format!(
            "tolerance must be positive, got {tolerance}"
        )));
    }
    let adf_passes = |d: f64| -> Result<bool, TimeSeriesError> {
        let diff = frac_diff(data, d, threshold)?;
        if diff.len() < 4 {
            return Ok(false);
        }
        let r = adf_test(&diff, 1)?;
        Ok(r.is_stationary)
    };

    // Check the endpoints.
    if adf_passes(0.0)? {
        return Ok(0.0);
    }
    if !adf_passes(1.0)? {
        // Even d = 1 fails: return 1.0 as the conservative answer.
        return Ok(1.0);
    }

    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while hi - lo > tolerance {
        let mid = 0.5 * (lo + hi);
        if adf_passes(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}