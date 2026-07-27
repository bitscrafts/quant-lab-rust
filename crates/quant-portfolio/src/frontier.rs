//! The Markowitz efficient frontier.
//!
//! See `README.md` in this directory for the module overview.
//!
//! Two paths:
//! - **Two-asset closed form**: given weights `w` on asset A and `1-w` on
//!   asset B, the portfolio return and variance are explicit quadratic
//!   functions of `w`. The minimum-variance weight has a closed form.
//! - **N-asset Lagrangian**: with `n` assets and an invertible covariance
//!   matrix, the global minimum-variance portfolio and the target-return
//!   efficient portfolio both have closed forms via the inverse of `Sigma`.

use crate::error::PortfolioError;
use crate::linalg::{inverse, matvec, solve};

/// A single point on the efficient frontier: expected return and volatility
/// (standard deviation, not variance) of an efficient portfolio.
#[derive(Debug, Clone, Copy)]
pub struct FrontierPoint {
    /// Expected return `mu_p = w' mu`.
    pub expected_return: f64,
    /// Volatility `sigma_p = sqrt(w' Sigma w)`.
    pub volatility: f64,
}

/// Compute `(mu_p, sigma_p)` for a two-asset portfolio with weight `w` on
/// asset A and `1 - w` on asset B.
///
/// `mu_a`, `mu_b` are expected returns; `var_a`, `var_b` are variances
/// (sigma^2); `cov_ab` is the covariance `Cov(R_a, R_b) = rho * sigma_a *
/// sigma_b`.
pub fn two_asset_frontier_point(
    w: f64,
    mu_a: f64,
    mu_b: f64,
    var_a: f64,
    var_b: f64,
    cov_ab: f64,
) -> FrontierPoint {
    let mu_p = w * mu_a + (1.0 - w) * mu_b;
    let var_p = w * w * var_a + (1.0 - w) * (1.0 - w) * var_b + 2.0 * w * (1.0 - w) * cov_ab;
    FrontierPoint {
        expected_return: mu_p,
        volatility: var_p.sqrt(),
    }
}

/// Closed-form weight on asset A that minimises the two-asset portfolio
/// variance.
///
/// `w_A* = (sigma_B^2 - Cov(A,B)) / (sigma_A^2 + sigma_B^2 - 2 Cov(A,B))`.
///
/// Returns `0.5` when both variances are zero (degenerate).
pub fn two_asset_min_variance_weight(var_a: f64, var_b: f64, cov_ab: f64) -> f64 {
    let denom = var_a + var_b - 2.0 * cov_ab;
    if denom.abs() < 1e-15 {
        return 0.5;
    }
    (var_b - cov_ab) / denom
}

/// Global minimum-variance portfolio for `n` assets (no return constraint).
///
/// Solves the Lagrangian system
/// ```text
/// [ Sigma  1 ] [ w ]   [ 0 ]
/// [ 1'     0 ] [ l ] = [ 1 ]
/// ```
/// and returns the weights `w` such that `sum(w) = 1` and `w' Sigma w` is
/// minimised. Equivalent closed form: `w* = Sigma^{-1} 1 / (1' Sigma^{-1} 1)`.
///
/// # Errors
/// - [`PortfolioError::InvalidParam`] when the universe is empty.
/// - [`PortfolioError::SingularCovariance`] when `Sigma` is singular.
pub fn min_variance_portfolio(
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
) -> Result<Vec<f64>, PortfolioError> {
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
    // w* = Sigma^{-1} 1 / (1' Sigma^{-1} 1)
    let sigma_inv = inverse(covariance)?;
    let ones = vec![1.0_f64; n];
    let sigma_inv_ones = matvec(&sigma_inv, &ones);
    let denom: f64 = sigma_inv_ones.iter().sum();
    if denom.abs() < 1e-15 {
        return Err(PortfolioError::SingularCovariance(
            "1' Sigma^{-1} 1 ~ 0".into(),
        ));
    }
    Ok(sigma_inv_ones.iter().map(|v| v / denom).collect())
}

/// Efficient frontier portfolio that achieves a target expected return
/// `mu_target` at the minimum possible variance.
///
/// Solves the Lagrangian
/// ```text
/// minimise   w' Sigma w
/// subject to 1' w = 1
///            mu' w = mu_target
/// ```
/// The closed form uses the standard scalars
/// `a = 1' Sigma^{-1} 1`, `b = 1' Sigma^{-1} mu`, `c = mu' Sigma^{-1} mu`,
/// `d = a*c - b^2`, and produces
/// `w* = (1/d) Sigma^{-1} [ (c - b*mu_target) 1 + (a*mu_target - b) mu ]`.
///
/// # Errors
/// - [`PortfolioError::InfeasibleTarget`] when `mu_target` cannot be reached
///   by any portfolio (this happens when the covariance is singular or the
///   return vector is degenerate — under a well-posed problem the whole real
///   line is reachable unconstrained, so this only fires when the Lagrangian
///   system is degenerate).
/// - [`PortfolioError::SingularCovariance`] when `Sigma` is singular.
pub fn efficient_frontier_point(
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    mu_target: f64,
) -> Result<Vec<f64>, PortfolioError> {
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
    let sigma_inv = inverse(covariance)?;
    let ones = vec![1.0_f64; n];
    let sigma_inv_ones = matvec(&sigma_inv, &ones);
    let sigma_inv_mu = matvec(&sigma_inv, expected_returns);
    let a: f64 = sigma_inv_ones.iter().sum();
    let b: f64 = sigma_inv_ones
        .iter()
        .zip(expected_returns.iter())
        .map(|(x, m)| x * m)
        .sum();
    let c: f64 = expected_returns
        .iter()
        .zip(sigma_inv_mu.iter())
        .map(|(m, x)| m * x)
        .sum();
    let d = a * c - b * b;
    if d.abs() < 1e-15 {
        return Err(PortfolioError::InfeasibleTarget {
            target: mu_target,
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        });
    }
    let coef_ones = (c - b * mu_target) / d;
    let coef_mu = (a * mu_target - b) / d;
    let w: Vec<f64> = (0..n)
        .map(|i| coef_ones * sigma_inv_ones[i] + coef_mu * sigma_inv_mu[i])
        .collect();
    Ok(w)
}

/// Solve `A x = b` using the local Gaussian elimination (re-exported for
/// module-internal callers and tests that want the same solver).
#[allow(dead_code)]
pub(crate) fn solve_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, PortfolioError> {
    solve(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn two_asset_frontier_endpoints() {
        // w=1 -> all A; w=0 -> all B.
        let p_a = two_asset_frontier_point(1.0, 0.10, 0.05, 0.04, 0.09, 0.0);
        assert_abs_diff_eq!(p_a.expected_return, 0.10, epsilon = 1e-12);
        assert_abs_diff_eq!(p_a.volatility, 0.20, epsilon = 1e-12);
        let p_b = two_asset_frontier_point(0.0, 0.10, 0.05, 0.04, 0.09, 0.0);
        assert_abs_diff_eq!(p_b.expected_return, 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(p_b.volatility, 0.30, epsilon = 1e-12);
    }

    #[test]
    fn two_asset_min_variance_weight_uncorrelated() {
        // sigma_A = 0.2, sigma_B = 0.3, rho = 0 -> w_A = 0.09 / (0.04 + 0.09)
        let w = two_asset_min_variance_weight(0.04, 0.09, 0.0);
        assert_abs_diff_eq!(w, 0.09 / 0.13, epsilon = 1e-12);
    }

    #[test]
    fn two_asset_min_variance_weight_perfect_correlation() {
        // rho = 1, sigma_A = sigma_B -> any split is min variance; denom = 0.
        let w = two_asset_min_variance_weight(0.04, 0.04, 0.04);
        assert_abs_diff_eq!(w, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn min_variance_portfolio_diagonal() {
        // Two uncorrelated assets: w_i proportional to 1/sigma_i^2.
        // sigma_A^2 = 0.04, sigma_B^2 = 0.09 -> w_A = (1/0.04)/(1/0.04+1/0.09)
        let mu = vec![0.10, 0.05];
        let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        let w = min_variance_portfolio(&mu, &cov).unwrap();
        let expected_a = (1.0 / 0.04) / (1.0 / 0.04 + 1.0 / 0.09);
        assert_abs_diff_eq!(w[0], expected_a, epsilon = 1e-9);
        assert_abs_diff_eq!(w[0] + w[1], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn efficient_frontier_target_return_round_trip() {
        // For target return equal to asset 1's expected return, the
        // efficient portfolio should place all weight on asset 1 if assets
        // are independent and the target equals mu[0].
        let mu = vec![0.10, 0.05];
        let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        let w = efficient_frontier_point(&mu, &cov, 0.10).unwrap();
        assert_abs_diff_eq!(w[0], 1.0, epsilon = 1e-7);
        assert_abs_diff_eq!(w[1], 0.0, epsilon = 1e-7);
    }

    #[test]
    fn efficient_frontier_midpoint_has_correct_return() {
        let mu = vec![0.10, 0.05];
        let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        let target = 0.075;
        let w = efficient_frontier_point(&mu, &cov, target).unwrap();
        let realized: f64 = w.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert_abs_diff_eq!(realized, target, epsilon = 1e-9);
        // Budget constraint must hold.
        let sum: f64 = w.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
    }
}
