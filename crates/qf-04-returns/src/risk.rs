//! Risk-adjusted performance metrics: Sharpe and Sortino ratios.
//!
//! See `README.md` in this directory for the module overview.

use crate::volatility::volatility;

/// Sharpe ratio of a return series.
///
/// # Formula
/// S = (mean(r) - rf) / sigma(r)
///
/// # Arguments
/// * `returns` - Slice of period returns.
/// * `risk_free_rate` - Per-period risk-free rate (same periodicity as
///   `returns`).
///
/// # Returns
/// The Sharpe ratio, or `0.0` if the volatility is zero (constant returns) or
/// there are fewer than two observations.
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let vol = volatility(returns);
    if vol == 0.0 {
        return 0.0;
    }
    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    (mean - risk_free_rate) / vol
}

/// Annualised Sharpe ratio.
///
/// # Formula
/// S_annual = sharpe(r, rf) * sqrt(periods_per_year)
///
/// # Arguments
/// * `returns` - Slice of period returns.
/// * `risk_free_rate` - Per-period risk-free rate.
/// * `periods_per_year` - Periods per year (252 for daily). Must be positive.
///
/// # Returns
/// The annualised Sharpe ratio, or `0.0` if the per-period Sharpe is zero or
/// the annualisation factor is non-positive.
pub fn annualized_sharpe(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if periods_per_year <= 0.0 {
        return 0.0;
    }
    sharpe_ratio(returns, risk_free_rate) * periods_per_year.sqrt()
}

/// Sortino ratio of a return series.
///
/// Like the Sharpe ratio, but penalises only downside volatility: returns
/// below the risk-free rate (the target). Upside volatility contributes nothing
/// to the denominator.
///
/// # Formula
/// downside_dev = sqrt( (1/n) * sum_{r_t < rf} (rf - r_t)^2 )
/// Sortino = (mean(r) - rf) / downside_dev
///
/// # Arguments
/// * `returns` - Slice of period returns.
/// * `risk_free_rate` - Per-period risk-free rate used as the target.
///
/// # Returns
/// The Sortino ratio, or `0.0` if there is no downside risk (downside
/// deviation is zero) or fewer than two observations.
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let downside_sq: f64 = returns
        .iter()
        .map(|&r| {
            let short = risk_free_rate - r;
            if short > 0.0 {
                short * short
            } else {
                0.0
            }
        })
        .sum();
    if downside_sq == 0.0 {
        return 0.0;
    }
    let downside_dev = (downside_sq / n as f64).sqrt();
    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    (mean - risk_free_rate) / downside_dev
}