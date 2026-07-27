//! Risk metrics: historical Value-at-Risk and Conditional VaR (Expected
//! Shortfall), plus the `RiskModel` trait.
//!
//! See `README.md` in this directory for the module overview.
//!
//! Both metrics use the empirical distribution of a return series — no
//! parametric assumption (Gaussian, Student-t, etc.) is imposed. This is the
//! simplest non-parametric estimator and is appropriate for pedagogical
//! clarity; production systems would use a fitted distribution or
//! extreme-value theory for tail extrapolation.

use crate::error::PortfolioError;

/// Trait for risk metrics that summarise the tail of a return distribution.
pub trait RiskModel {
    /// The estimated risk metric (VaR, CVaR, etc.) at the given confidence
    /// level.
    ///
    /// `confidence` is in `[0, 1]`; e.g. `0.95` for 95% VaR (the loss that is
    /// exceeded with probability 5%).
    fn risk(&self, returns: &[f64], confidence: f64) -> Result<f64, PortfolioError>;
}

/// Historical Value-at-Risk at the given confidence level.
///
/// Returns the **loss** that is exceeded with probability `1 - confidence`.
/// Concretely, with `confidence = 0.95`, this is the 5% quantile of the
/// return series, returned as a positive number (the loss magnitude).
///
/// Formula: `VaR_alpha = -quantile_{1 - alpha}(R)`, where `quantile_p` is the
/// `p`-th empirical quantile (linear interpolation between adjacent order
/// statistics).
///
/// # Errors
/// - [`PortfolioError::InsufficientData`] when `returns` is empty.
/// - [`PortfolioError::InvalidParam`] when `confidence` is outside
///   `(0, 1)`.
pub fn historical_var(returns: &[f64], confidence: f64) -> Result<f64, PortfolioError> {
    if returns.is_empty() {
        return Err(PortfolioError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(PortfolioError::InvalidParam(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }
    let p = 1.0 - confidence;
    let quantile = empirical_quantile(returns, p);
    Ok(-quantile)
}

/// Historical Conditional VaR (Expected Shortfall) at the given confidence
/// level.
///
/// Returns the **average loss** in the tail beyond the VaR level. With
/// `confidence = 0.95`, this is the mean of all returns at or below the 5%
/// quantile, returned as a positive number.
///
/// # Errors
/// - [`PortfolioError::InsufficientData`] when `returns` is empty.
/// - [`PortfolioError::InvalidParam`] when `confidence` is outside
///   `(0, 1)`.
pub fn historical_cvar(returns: &[f64], confidence: f64) -> Result<f64, PortfolioError> {
    if returns.is_empty() {
        return Err(PortfolioError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(PortfolioError::InvalidParam(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }
    let p = 1.0 - confidence;
    let quantile = empirical_quantile(returns, p);
    // Average of returns <= quantile (the left tail). CVaR is the negative of
    // this average (a positive loss magnitude).
    let tail: Vec<f64> = returns.iter().filter(|r| **r <= quantile + 1e-12).copied().collect();
    if tail.is_empty() {
        // Fallback: should not happen with a non-empty return series.
        return Ok(-quantile);
    }
    let mean_tail: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
    Ok(-mean_tail)
}

/// Empirical quantile with linear interpolation between adjacent order
/// statistics (the same convention as NumPy's default).
fn empirical_quantile(returns: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = p * (n as f64 - 1.0);
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let frac = pos - lower as f64;
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

impl RiskModel for fn(&[f64], f64) -> Result<f64, PortfolioError> {
    fn risk(&self, returns: &[f64], confidence: f64) -> Result<f64, PortfolioError> {
        self(returns, confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn historical_var_basic() {
        // Series: -0.045, -0.035, -0.025, ..., 0.045 (10 points, step 0.01).
        let returns: Vec<f64> = (0..10).map(|i| (i as f64 - 4.5) * 0.01).collect();
        // 95% VaR -> 5% quantile. pos = 0.05 * 9 = 0.45 between sorted[0]=-0.045
        // and sorted[1]=-0.035: q = -0.045*0.55 + -0.035*0.45 = -0.0405.
        let var = historical_var(&returns, 0.95).unwrap();
        assert_abs_diff_eq!(var, 0.0405, epsilon = 1e-9);
    }

    #[test]
    fn historical_cvar_basic() {
        // Same series. CVaR at 95% is the mean of the returns <= quantile
        // (-0.0405). Only -0.045 falls in the tail -> mean = -0.045 -> CVaR = 0.045.
        let returns: Vec<f64> = (0..10).map(|i| (i as f64 - 4.5) * 0.01).collect();
        let cvar = historical_cvar(&returns, 0.95).unwrap();
        assert_abs_diff_eq!(cvar, 0.045, epsilon = 1e-9);
    }

    #[test]
    fn cvar_exceeds_var() {
        // CVaR must be at least as large as VaR (tail average vs. tail cutoff).
        let returns: Vec<f64> = vec![
            -0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.06,
        ];
        let var = historical_var(&returns, 0.90).unwrap();
        let cvar = historical_cvar(&returns, 0.90).unwrap();
        assert!(
            cvar >= var - 1e-12,
            "CVaR ({cvar}) should exceed VaR ({var})"
        );
    }

    #[test]
    fn var_rejects_empty_input() {
        assert!(matches!(
            historical_var(&[], 0.95),
            Err(PortfolioError::InsufficientData { .. })
        ));
    }

    #[test]
    fn var_rejects_invalid_confidence() {
        let returns = vec![0.01, 0.02, -0.01];
        assert!(matches!(
            historical_var(&returns, 1.5),
            Err(PortfolioError::InvalidParam(_))
        ));
    }
}