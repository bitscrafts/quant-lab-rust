//! Volatility measures: sample standard deviation, annualised, and rolling.
//!
//! See `README.md` in this directory for the module overview.

/// Computes the sample standard deviation of a return series.
///
/// Uses the unbiased estimator (denominator `n - 1`), consistent with
/// `qf_common::compute_stats`. Returns `0.0` for fewer than two observations
/// (volatility is undefined for a single observation).
pub fn volatility(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    let ssd: f64 = returns
        .iter()
        .map(|&r| {
            let d = r - mean;
            d * d
        })
        .sum();
    (ssd / (n - 1) as f64).sqrt()
}

/// Annualises a daily (or other periodic) volatility.
///
/// # Formula
/// sigma_annual = sigma_period * sqrt(periods_per_year)
///
/// # Arguments
/// * `returns` - Slice of period returns.
/// * `periods_per_year` - Number of periods in a year (252 for daily, 12 for
///   monthly, 4 for quarterly). Must be positive; non-positive values yield
///   `0.0`.
///
/// # Returns
/// The annualised volatility, or `0.0` if the input volatility is zero or the
/// annualisation factor is non-positive.
pub fn annualized_volatility(returns: &[f64], periods_per_year: f64) -> f64 {
    if periods_per_year <= 0.0 {
        return 0.0;
    }
    volatility(returns) * periods_per_year.sqrt()
}

/// Computes the rolling sample standard deviation of a return series.
///
/// For each window of length `window` ending at index `i` (where
/// `i >= window - 1`), the output is the sample standard deviation of
/// `returns[i+1-window..=i]`.
///
/// # Arguments
/// * `returns` - Slice of returns.
/// * `window` - Window size. Must be at least 2 to produce a meaningful
///   standard deviation.
///
/// # Returns
/// Vector of length `returns.len() - window + 1` when `window` is valid and
/// fits the input. Returns an empty vector when `window == 0`,
/// `window > returns.len()`, or the input is empty.
pub fn rolling_volatility(returns: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > returns.len() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(returns.len() - window + 1);
    for i in window - 1..returns.len() {
        let slice = &returns[i + 1 - window..=i];
        out.push(volatility(slice));
    }
    out
}