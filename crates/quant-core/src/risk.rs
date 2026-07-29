//! Risk metrics and performance statistics.
//!
//! This module provides advanced risk metrics including deflated
//! Sharpe ratios, Calmar ratio, Omega ratio, and other performance
//! measures.

use crate::error::CoreError;

/// Compute the probabilistic Sharpe ratio (PSR).
///
/// The PSR adjusts the Sharpe ratio for estimation error and
/// non-normality, providing the probability that the true Sharpe
/// ratio exceeds a benchmark.
///
/// # Arguments
///
/// * `sharpe` - Estimated Sharpe ratio
/// * `n` - Number of observations
/// * `skew` - Skewness of returns
/// * `kurtosis` - Excess kurtosis of returns
/// * `sharpe_benchmark` - Benchmark Sharpe ratio (typically 0)
///
/// # Returns
///
/// The probability (0 to 1) that the true Sharpe exceeds the benchmark.
///
/// # References
///
/// Bailey & López de Prado (2012): "The Sharpe Ratio Efficient Frontier"
///
/// # Example
///
/// ```
/// use quant_core::probabilistic_sharpe_ratio;
///
/// let psr = probabilistic_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 0.0);
/// // PSR should be very high (> 0.99) for Sharpe=1.0 with 252 observations
/// assert!(psr > 0.99);
/// ```
pub fn probabilistic_sharpe_ratio(
    sharpe: f64,
    n: f64,
    skew: f64,
    kurtosis: f64,
    sharpe_benchmark: f64,
) -> f64 {
    if n <= 0.0 {
        return 0.0;
    }

    // Adjust for higher moments (non-normality)
    let adjustment = (1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe.powi(2)) / (n - 1.0);
    let variance = (1.0 + adjustment).max(0.0);

    // Standard error of Sharpe ratio
    let std_error = variance.sqrt();

    if std_error == 0.0 {
        return if sharpe > sharpe_benchmark { 1.0 } else { 0.0 };
    }

    // Z-score for hypothesis test
    let z = (sharpe - sharpe_benchmark) * n.sqrt() / std_error;

    // CDF of standard normal
    standard_normal_cdf(z)
}

/// Compute the deflated Sharpe ratio (DSR).
///
/// The DSR further adjusts the PSR for multiple testing and
/// selection bias, accounting for the fact that the best strategy
/// was selected from many trials.
///
/// # Arguments
///
/// * `sharpe` - Estimated Sharpe ratio of selected strategy
/// * `n` - Number of observations
/// * `skew` - Skewness of returns
/// * `kurtosis` - Excess kurtosis of returns
/// * `n_trials` - Number of strategies tested
/// * `var_sharpes` - Variance of Sharpe ratios across trials
///
/// # Returns
///
/// The probability that the selected strategy has true Sharpe > 0
/// after adjusting for selection bias.
///
/// # References
///
/// Bailey & López de Prado (2014): "The Deflated Sharpe Ratio"
///
/// # Example
///
/// ```
/// use quant_core::deflated_sharpe_ratio;
///
/// // Single trial: DSR ≈ PSR
/// let dsr = deflated_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 1, 0.0);
/// assert!(dsr > 0.99);
///
/// // Multiple trials: DSR < PSR (penalized)
/// let dsr_multi = deflated_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 100, 0.1);
/// assert!(dsr_multi < dsr);
/// ```
pub fn deflated_sharpe_ratio(
    sharpe: f64,
    n: f64,
    skew: f64,
    kurtosis: f64,
    n_trials: usize,
    var_sharpes: f64,
) -> f64 {
    if n_trials == 0 {
        return 0.0;
    }

    // If only one trial, no selection bias adjustment needed
    if n_trials == 1 {
        return probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis, 0.0);
    }

    // Expected maximum Sharpe under null hypothesis (all strategies have SR=0)
    // Using Euler-Mascheroni constant approximation
    let euler_mascheroni = 0.5772156649;
    let expected_max = ((2.0 * (n_trials as f64).ln()).sqrt()
        - (0.5 * (n_trials as f64).ln().ln() + euler_mascheroni)
            / (2.0 * (n_trials as f64).ln()).sqrt())
    .max(0.0);

    // Adjust benchmark for selection bias
    let sharpe_benchmark = expected_max * var_sharpes.sqrt();

    probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis, sharpe_benchmark)
}

/// Compute the Calmar ratio: annualized return / maximum drawdown.
///
/// The Calmar ratio measures risk-adjusted return using maximum
/// drawdown as the risk measure.
///
/// # Arguments
///
/// * `annualized_return` - Annualized return (e.g., 0.15 for 15%)
/// * `max_drawdown` - Maximum drawdown (e.g., 0.20 for -20%)
///
/// # Returns
///
/// The Calmar ratio. Higher is better.
///
/// # Example
///
/// ```
/// use quant_core::calmar_ratio;
///
/// let calmar = calmar_ratio(0.20, 0.10);
/// assert_eq!(calmar, 2.0); // 20% return / 10% drawdown
/// ```
pub fn calmar_ratio(annualized_return: f64, max_drawdown: f64) -> f64 {
    if max_drawdown == 0.0 {
        return f64::INFINITY;
    }
    annualized_return / max_drawdown.abs()
}

/// Compute the Omega ratio: probability-weighted gains / losses.
///
/// The Omega ratio is the ratio of cumulative gains above a threshold
/// to cumulative losses below the threshold.
///
/// # Arguments
///
/// * `returns` - Vector of returns
/// * `threshold` - Target return threshold (e.g., 0.0 for zero)
///
/// # Returns
///
/// The Omega ratio. Values > 1 indicate more gains than losses.
///
/// # Errors
///
/// Returns [`CoreError::InsufficientData`] if returns is empty.
///
/// # Example
///
/// ```
/// use quant_core::omega_ratio;
///
/// let returns = vec![0.02, -0.01, 0.03, -0.02, 0.01];
/// let omega = omega_ratio(&returns, 0.0).unwrap();
/// // More positive returns: omega > 1
/// assert!(omega > 1.0);
/// ```
pub fn omega_ratio(returns: &[f64], threshold: f64) -> Result<f64, CoreError> {
    if returns.is_empty() {
        return Err(CoreError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let gains: f64 = returns.iter().map(|&r| (r - threshold).max(0.0)).sum();
    let losses: f64 = returns.iter().map(|&r| (threshold - r).max(0.0)).sum();

    if losses == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(gains / losses)
}

/// Compute the information ratio: excess return / tracking error.
///
/// The information ratio measures risk-adjusted excess return
/// relative to a benchmark.
///
/// # Arguments
///
/// * `returns` - Portfolio returns
/// * `benchmark_returns` - Benchmark returns
///
/// # Returns
///
/// The information ratio.
///
/// # Errors
///
/// Returns [`CoreError::InsufficientData`] if returns are empty or
/// have different lengths.
///
/// # Example
///
/// ```
/// use quant_core::information_ratio;
///
/// let returns = vec![0.05, 0.03, 0.06];
/// let benchmark = vec![0.04, 0.04, 0.04];
/// let ir = information_ratio(&returns, &benchmark).unwrap();
/// // Positive excess return with low tracking error
/// assert!(ir > 0.0);
/// ```
pub fn information_ratio(returns: &[f64], benchmark_returns: &[f64]) -> Result<f64, CoreError> {
    if returns.len() != benchmark_returns.len() {
        return Err(CoreError::InsufficientData {
            required: returns.len(),
            actual: benchmark_returns.len(),
        });
    }

    if returns.is_empty() {
        return Err(CoreError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    // Compute active returns (excess over benchmark)
    let active_returns: Vec<f64> = returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(r, b)| r - b)
        .collect();

    // Mean excess return
    let mean_excess = crate::mean(&active_returns);

    // Tracking error (std dev of active returns)
    let tracking_error = crate::std_dev(&active_returns)?;

    if tracking_error == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(mean_excess / tracking_error)
}

/// Compute the Ulcer Index: measure of downside volatility.
///
/// The Ulcer Index measures the depth and duration of drawdowns,
/// providing a drawdown-based risk measure.
///
/// # Arguments
///
/// * `prices` - Price series (not returns)
///
/// # Returns
///
/// The Ulcer Index. Lower is better (less drawdown pain).
///
/// # Errors
///
/// Returns [`CoreError::InsufficientData`] if prices is empty.
///
/// # Example
///
/// ```
/// use quant_core::ulcer_index;
///
/// // Monotonically increasing prices: no drawdown
/// let prices = vec![100.0, 101.0, 102.0, 103.0];
/// let ulcer = ulcer_index(&prices).unwrap();
/// assert_eq!(ulcer, 0.0);
///
/// // Drawdown followed by recovery
/// let prices_dd = vec![100.0, 95.0, 90.0, 95.0, 100.0];
/// let ulcer_dd = ulcer_index(&prices_dd).unwrap();
/// assert!(ulcer_dd > 0.0);
/// ```
pub fn ulcer_index(prices: &[f64]) -> Result<f64, CoreError> {
    if prices.is_empty() {
        return Err(CoreError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let mut max_price = prices[0];
    let mut squared_drawdowns = 0.0;

    for &price in prices.iter() {
        max_price = max_price.max(price);
        let drawdown = if max_price > 0.0 {
            ((price - max_price) / max_price) * 100.0 // As percentage
        } else {
            0.0
        };
        squared_drawdowns += drawdown * drawdown;
    }

    let n = prices.len() as f64;
    Ok((squared_drawdowns / n).sqrt())
}

/// Cumulative distribution function of standard normal distribution.
///
/// Uses the approximation from Abramowitz & Stegun (1964).
fn standard_normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let prob =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if x >= 0.0 { 1.0 - prob } else { prob }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psr_zero_benchmark() {
        // High Sharpe with many observations should have PSR ≈ 1
        let psr = probabilistic_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 0.0);
        assert!(psr > 0.99);
    }

    #[test]
    fn test_psr_noisy_sharpe() {
        // Low Sharpe with few observations and high skew/kurtosis
        // Should still have reasonable PSR, but not as high as ideal case
        let psr = probabilistic_sharpe_ratio(0.5, 60.0, -1.0, 5.0, 0.0);
        // PSR should be positive but not overly optimistic
        assert!(psr > 0.5 && psr < 1.0);
    }

    #[test]
    fn test_dsr_single_trial() {
        // Single trial: DSR = PSR (no selection bias)
        // With Sharpe=1.0 and 252 observations, should have good confidence
        let dsr = deflated_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 1, 0.0);
        let psr = probabilistic_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 0.0);
        // With 1 trial, DSR should equal PSR
        assert!((dsr - psr).abs() < 0.001);
        assert!(dsr > 0.99);
    }

    #[test]
    fn test_dsr_multiple_trials() {
        // Multiple trials should penalize
        let dsr = deflated_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 100, 0.1);
        let psr = probabilistic_sharpe_ratio(1.0, 252.0, 0.0, 0.0, 0.0);
        assert!(dsr < psr);
    }

    #[test]
    fn test_calmar_ratio() {
        let calmar = calmar_ratio(0.20, 0.10);
        assert_eq!(calmar, 2.0);
    }

    #[test]
    fn test_omega_ratio_above_threshold() {
        // All returns above threshold: omega = infinity
        let returns = vec![0.01, 0.02, 0.03];
        let omega = omega_ratio(&returns, 0.0).unwrap();
        assert!(omega > 100.0); // Very high
    }

    #[test]
    fn test_omega_ratio_balanced() {
        // Equal gains and losses: omega ≈ 1
        let returns = vec![0.01, -0.01, 0.01, -0.01];
        let omega = omega_ratio(&returns, 0.0).unwrap();
        assert!((omega - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_information_ratio() {
        let returns = vec![0.05, 0.03, 0.06];
        let benchmark = vec![0.04, 0.04, 0.04];
        let ir = information_ratio(&returns, &benchmark).unwrap();
        assert!(ir > 0.0);
    }

    #[test]
    fn test_ulcer_index_no_drawdown() {
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let ulcer = ulcer_index(&prices).unwrap();
        assert_eq!(ulcer, 0.0);
    }

    #[test]
    fn test_ulcer_index_with_drawdown() {
        let prices = vec![100.0, 95.0, 90.0, 95.0, 100.0];
        let ulcer = ulcer_index(&prices).unwrap();
        assert!(ulcer > 0.0);
    }

    #[test]
    fn test_standard_normal_cdf() {
        // CDF(0) = 0.5
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 0.001);

        // CDF(1.96) ≈ 0.975 (95th percentile)
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 0.01);

        // CDF(-1.96) ≈ 0.025 (5th percentile)
        assert!((standard_normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }
}
