//! The tangency portfolio, the capital market line, and two-fund separation.
//!
//! See `README.md` in this directory for the module overview.
//!
//! With a risk-free rate `rf`, the tangency portfolio is the point on the
//! efficient frontier that maximises the Sharpe ratio
//! `(mu_p - rf) / sigma_p`. Its closed form is
//! ```text
//! w_tan = Sigma^{-1} (mu - rf * 1) / (1' Sigma^{-1} (mu - rf * 1)).
//! ```
//! The capital market line (CML) is the straight line in `(sigma, mu)` space
//! from `(0, rf)` through the tangency point; every point on it is a
//! combination of the risk-free asset and the tangency portfolio (two-fund
//! separation).

use crate::error::PortfolioError;
use crate::linalg::{inverse, matvec};
use crate::portfolio::{portfolio_return, portfolio_volatility};

/// Tangency portfolio: weights plus the headline statistics.
#[derive(Debug, Clone)]
pub struct TangencyResult {
    /// Weights on the risky assets, summing to 1 (the tangency portfolio is
    /// fully invested in risky assets).
    pub weights: Vec<f64>,
    /// Expected return `mu_tan = w' mu`.
    pub expected_return: f64,
    /// Volatility `sigma_tan = sqrt(w' Sigma w)`.
    pub volatility: f64,
    /// Sharpe ratio `(mu_tan - rf) / sigma_tan`.
    pub sharpe: f64,
}

/// Compute the tangency portfolio (maximum-Sharpe portfolio) for a universe
/// of risky assets and a risk-free rate `rf`.
///
/// # Errors
/// - [`PortfolioError::InvalidParam`] when the universe is empty or `rf`
///   makes `mu - rf * 1` the zero vector (all expected returns equal `rf`).
/// - [`PortfolioError::SingularCovariance`] when `Sigma` is singular.
pub fn tangency_portfolio(
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    rf: f64,
) -> Result<TangencyResult, PortfolioError> {
    let n = expected_returns.len();
    if n == 0 {
        return Err(PortfolioError::InvalidParam("empty universe".into()));
    }
    if covariance.len() != n {
        return Err(PortfolioError::DimensionMismatch(format!(
            "expected_returns.len() = {n} != covariance.len() = {}",
            covariance.len()
        )));
    }
    let excess: Vec<f64> = expected_returns.iter().map(|m| m - rf).collect();
    // If all excess returns are zero the tangency portfolio is undefined.
    let excess_norm: f64 = excess.iter().map(|e| e * e).sum::<f64>().sqrt();
    if excess_norm < 1e-12 {
        return Err(PortfolioError::InvalidParam(
            "all expected returns equal rf — tangency undefined".into(),
        ));
    }
    let sigma_inv = inverse(covariance)?;
    let sigma_inv_excess = matvec(&sigma_inv, &excess);
    let denom: f64 = sigma_inv_excess.iter().sum();
    if denom.abs() < 1e-15 {
        return Err(PortfolioError::SingularCovariance(
            "1' Sigma^{-1} (mu - rf 1) ~ 0".into(),
        ));
    }
    let weights: Vec<f64> = sigma_inv_excess.iter().map(|v| v / denom).collect();
    let expected_return = portfolio_return(&weights, expected_returns);
    let volatility = portfolio_volatility(&weights, covariance);
    let sharpe = if volatility > 0.0 {
        (expected_return - rf) / volatility
    } else {
        0.0
    };
    Ok(TangencyResult {
        weights,
        expected_return,
        volatility,
        sharpe,
    })
}

/// Expected return on the capital market line at a given target volatility.
///
/// The CML is `mu = rf + Sharpe_tan * sigma`. Returns `rf` when `sigma = 0`.
pub fn capital_market_line(rf: f64, tangency: &TangencyResult, sigma: f64) -> f64 {
    rf + tangency.sharpe * sigma
}

/// Two-fund separation: produce the combined portfolio weights (risk-free
/// fraction + tangency fraction) that achieve a target volatility.
///
/// Returns `(y, w_tan)` where `y` is the fraction invested in the tangency
/// portfolio and `1 - y` is the fraction in the risk-free asset. The combined
/// portfolio has volatility `y * sigma_tan`.
///
/// `y` is allowed to exceed 1 (leverage) or go below 0 (short the tangency
/// portfolio to lend at `rf`).
pub fn two_fund_separation(tangency: &TangencyResult, target_volatility: f64) -> f64 {
    if tangency.volatility == 0.0 {
        return 0.0;
    }
    target_volatility / tangency.volatility
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_universe() -> (Vec<f64>, Vec<Vec<f64>>) {
        // Two uncorrelated assets.
        let mu = vec![0.10, 0.05];
        let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        (mu, cov)
    }

    #[test]
    fn tangency_weights_sum_to_one() {
        let (mu, cov) = sample_universe();
        let tan = tangency_portfolio(&mu, &cov, 0.02).unwrap();
        let sum: f64 = tan.weights.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn tangency_sharpe_is_maximal_around_frontier() {
        // Sample a few frontier portfolios and verify none beats tangency.
        let (mu, cov) = sample_universe();
        let rf = 0.02;
        let tan = tangency_portfolio(&mu, &cov, rf).unwrap();
        // Sweep w on asset A from 0 to 1 in 0.1 increments.
        for i in 0..=10 {
            let w = i as f64 * 0.1;
            let weights = vec![w, 1.0 - w];
            let vol = portfolio_volatility(&weights, &cov);
            let ret = portfolio_return(&weights, &mu);
            let s = if vol > 0.0 { (ret - rf) / vol } else { 0.0 };
            assert!(
                tan.sharpe + 1e-9 >= s,
                "frontier point Sharpe {s} exceeds tangency Sharpe {}",
                tan.sharpe
            );
        }
    }

    #[test]
    fn capital_market_line_at_zero_vol_is_rf() {
        let (mu, cov) = sample_universe();
        let tan = tangency_portfolio(&mu, &cov, 0.02).unwrap();
        assert_abs_diff_eq!(capital_market_line(0.02, &tan, 0.0), 0.02, epsilon = 1e-12);
    }

    #[test]
    fn capital_market_line_at_tangency_vol_is_tangency_return() {
        let (mu, cov) = sample_universe();
        let tan = tangency_portfolio(&mu, &cov, 0.02).unwrap();
        let cml_ret = capital_market_line(0.02, &tan, tan.volatility);
        assert_abs_diff_eq!(cml_ret, tan.expected_return, epsilon = 1e-9);
    }

    #[test]
    fn two_fund_separation_round_trip() {
        // A target vol of 0.5 * sigma_tan should give y = 0.5.
        let (mu, cov) = sample_universe();
        let tan = tangency_portfolio(&mu, &cov, 0.02).unwrap();
        let y = two_fund_separation(&tan, 0.5 * tan.volatility);
        assert_abs_diff_eq!(y, 0.5, epsilon = 1e-9);
    }
}
