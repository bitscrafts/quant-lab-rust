//! Exponentially weighted moving average (EWMA) volatility.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::VolError;

/// Compute the EWMA conditional variance series (RiskMetrics).
///
/// Recursion (one-step-ahead variance forecast):
/// ```text
/// sigma_0^2 = r_0^2
/// sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2   for t >= 1
/// ```
///
/// With `lambda = 0` the variance tracks the lagged squared return:
/// `sigma_t^2 = r_{t-1}^2`. With `lambda = 1` the variance is constant at
/// the initial value `r_0^2`. The RiskMetrics daily decay is `lambda = 0.94`.
///
/// # Arguments
/// * `returns` - Return series (simple or log).
/// * `lambda` - Decay factor in `[0, 1]`. Higher values give a slower
///   forgetting rate.
///
/// # Returns
/// Vector of conditional variances `sigma_t^2` with the same length as
/// `returns`. Take the square root for volatility (standard deviation).
///
/// # Errors
/// - [`VolError::InvalidParam`] when `lambda` is outside `[0, 1]`.
/// - [`VolError::InsufficientData`] when `returns` is empty.
pub fn ewma_vol(returns: &[f64], lambda: f64) -> Result<Vec<f64>, VolError> {
    if !(0.0..=1.0).contains(&lambda) {
        return Err(VolError::InvalidParam(format!(
            "lambda must be in [0, 1], got {lambda}"
        )));
    }
    if returns.is_empty() {
        return Err(VolError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    let mut sigma2 = vec![0.0_f64; returns.len()];
    sigma2[0] = returns[0] * returns[0];
    for t in 1..returns.len() {
        sigma2[t] = lambda * sigma2[t - 1] + (1.0 - lambda) * returns[t - 1] * returns[t - 1];
    }
    Ok(sigma2)
}
