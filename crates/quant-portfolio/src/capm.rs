//! Capital Asset Pricing Model: beta, alpha, and the security market line.
//!
//! See `README.md` in this directory for the module overview.
//!
//! CAPM states that the expected excess return of an asset is proportional to
//! its beta with the market portfolio:
//! ```text
//! E[R_i] - r_f = beta_i * (E[R_m] - r_f),
//! beta_i = Cov(R_i, R_m) / Var(R_m).
//! ```
//! The security market line (SML) is the linear relationship in `(beta, mu)`
//! space. Jensen's alpha is the gap between realised mean return and the
//! SML-predicted return: `alpha_i = mean(R_i) - (r_f + beta_i * (mean(R_m) -
//! r_f))`.

use crate::error::PortfolioError;

/// Compute the CAPM beta of an asset relative to a market return series.
///
/// `beta = Cov(R_i, R_m) / Var(R_m)`.
///
/// Uses sample covariance (denominator `n - 1`) and sample variance
/// (denominator `n - 1`), so the `(n - 1)` cancels and the estimator is
/// equivalent to using population moments.
///
/// # Errors
/// - [`PortfolioError::InsufficientData`] when there are fewer than 2
///   observations.
/// - [`PortfolioError::InvalidParam`] when the market variance is zero.
pub fn beta(asset: &[f64], market: &[f64]) -> Result<f64, PortfolioError> {
    let n = asset.len();
    if n < 2 {
        return Err(PortfolioError::InsufficientData {
            required: 2,
            actual: n,
        });
    }
    if market.len() != n {
        return Err(PortfolioError::DimensionMismatch(format!(
            "asset.len() = {n} != market.len() = {}",
            market.len()
        )));
    }
    let mean_a: f64 = asset.iter().sum::<f64>() / n as f64;
    let mean_m: f64 = market.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0_f64;
    let mut var_m = 0.0_f64;
    for (a, m) in asset.iter().zip(market.iter()) {
        cov += (a - mean_a) * (m - mean_m);
        var_m += (m - mean_m) * (m - mean_m);
    }
    if var_m == 0.0 {
        return Err(PortfolioError::InvalidParam(
            "market variance is zero".into(),
        ));
    }
    Ok(cov / var_m)
}

/// Compute Jensen's alpha: the gap between the realised mean return and the
/// return predicted by CAPM.
///
/// `alpha = mean(R_i) - (r_f + beta_i * (mean(R_m) - r_f))`.
///
/// # Errors
/// - [`PortfolioError::InsufficientData`] when there are fewer than 2
///   observations.
pub fn alpha(asset: &[f64], market: &[f64], rf: f64) -> Result<f64, PortfolioError> {
    let n = asset.len();
    if n < 2 {
        return Err(PortfolioError::InsufficientData {
            required: 2,
            actual: n,
        });
    }
    if market.len() != n {
        return Err(PortfolioError::DimensionMismatch(format!(
            "asset.len() = {n} != market.len() = {}",
            market.len()
        )));
    }
    let b = beta(asset, market)?;
    let mean_a: f64 = asset.iter().sum::<f64>() / n as f64;
    let mean_m: f64 = market.iter().sum::<f64>() / n as f64;
    Ok(mean_a - (rf + b * (mean_m - rf)))
}

/// Predicted expected return from the security market line:
/// `SML(beta) = r_f + beta * (mean(R_m) - r_f)`.
pub fn sml(beta: f64, market_mean: f64, rf: f64) -> f64 {
    rf + beta * (market_mean - rf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn beta_perfectly_correlated_is_one() {
        // Asset identical to market -> beta = 1.
        let m = vec![0.01, 0.02, -0.01, 0.03, 0.015];
        let b = beta(&m, &m).unwrap();
        assert_abs_diff_eq!(b, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn beta_uncorrelated_around_zero() {
        // Construct asset with zero covariance against market.
        let market = vec![0.01, 0.02, -0.01, 0.03];
        // Pick asset so Cov(a, m) = 0. Use constant asset (var = 0, cov = 0).
        // That makes beta = 0 / Var(m) = 0.
        let asset = vec![0.05, 0.05, 0.05, 0.05];
        let b = beta(&asset, &market).unwrap();
        assert_abs_diff_eq!(b, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn beta_half_leverage() {
        // Asset = 0.5 * market (perfectly correlated, half volatility).
        let market = vec![0.04, -0.02, 0.03, 0.01, -0.01];
        let asset: Vec<f64> = market.iter().map(|m| 0.5 * m).collect();
        let b = beta(&asset, &market).unwrap();
        assert_abs_diff_eq!(b, 0.5, epsilon = 1e-9);
    }

    #[test]
    fn alpha_zero_when_asset_is_on_sml() {
        // Asset = rf + beta * (market - rf) exactly -> alpha = 0.
        let rf = 0.02;
        let market = vec![0.05, 0.06, 0.04, 0.07, 0.05];
        let b_target = 1.2_f64;
        let asset: Vec<f64> = market.iter().map(|m| rf + b_target * (m - rf)).collect();
        let a = alpha(&asset, &market, rf).unwrap();
        assert_abs_diff_eq!(a, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn sml_at_beta_zero_is_rf() {
        assert_abs_diff_eq!(sml(0.0, 0.08, 0.02), 0.02, epsilon = 1e-12);
    }

    #[test]
    fn sml_at_beta_one_is_market_mean() {
        assert_abs_diff_eq!(sml(1.0, 0.08, 0.02), 0.08, epsilon = 1e-12);
    }
}
