//! Portfolio struct and the `Allocator` trait.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::PortfolioError;
use crate::linalg::{matvec, quadratic_form};

/// A portfolio: weights over a universe of assets, the assets' expected
/// returns, and the covariance matrix of their returns.
///
/// Weights are stored as a `Vec<f64>` of length `n` (the number of assets),
/// expected returns as a `Vec<f64>` of length `n`, and the covariance matrix
/// as a row-major `Vec<Vec<f64>>` of shape `n x n`. All three must agree on `n`.
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Portfolio weights, length `n`. May sum to 1 (fully invested) or to 0
    /// (dollar-neutral) — the struct does not enforce a budget constraint.
    pub weights: Vec<f64>,
    /// Expected return per asset, length `n`.
    pub expected_returns: Vec<f64>,
    /// Covariance matrix, shape `n x n` (row-major).
    pub covariance: Vec<Vec<f64>>,
}

impl Portfolio {
    /// Construct a new portfolio, validating that all dimensions agree.
    ///
    /// # Errors
    /// - [`PortfolioError::DimensionMismatch`] when the three slices do not
    ///   have compatible shapes.
    pub fn new(
        weights: Vec<f64>,
        expected_returns: Vec<f64>,
        covariance: Vec<Vec<f64>>,
    ) -> Result<Self, PortfolioError> {
        let n = weights.len();
        if expected_returns.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "weights.len() = {n} != expected_returns.len() = {}",
                expected_returns.len()
            )));
        }
        if covariance.len() != n {
            return Err(PortfolioError::DimensionMismatch(format!(
                "weights.len() = {n} != covariance.len() = {}",
                covariance.len()
            )));
        }
        for (i, row) in covariance.iter().enumerate() {
            if row.len() != n {
                return Err(PortfolioError::DimensionMismatch(format!(
                    "covariance row {i} has length {} != {n}",
                    row.len()
                )));
            }
        }
        Ok(Self {
            weights,
            expected_returns,
            covariance,
        })
    }

    /// Expected portfolio return: `w' * mu`.
    pub fn expected_return(&self) -> f64 {
        self.weights
            .iter()
            .zip(self.expected_returns.iter())
            .map(|(w, m)| w * m)
            .sum()
    }

    /// Portfolio variance: `w' * Sigma * w`.
    pub fn variance(&self) -> f64 {
        quadratic_form(&self.weights, &self.covariance)
    }

    /// Portfolio volatility: `sqrt(w' * Sigma * w)`.
    pub fn volatility(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sharpe ratio `(mu_p - rf) / sigma_p` for a given risk-free rate `rf`.
    ///
    /// Returns `0.0` when `sigma_p == 0` (degenerate all-cash portfolio).
    pub fn sharpe(&self, rf: f64) -> f64 {
        let vol = self.volatility();
        if vol == 0.0 {
            return 0.0;
        }
        (self.expected_return() - rf) / vol
    }

    /// Compute all four headline statistics in one pass.
    pub fn stats(&self, rf: f64) -> PortfolioStats {
        PortfolioStats {
            expected_return: self.expected_return(),
            variance: self.variance(),
            volatility: self.volatility(),
            sharpe: self.sharpe(rf),
        }
    }
}

/// Aggregate statistics for a portfolio at a given risk-free rate.
#[derive(Debug, Clone, Copy)]
pub struct PortfolioStats {
    pub expected_return: f64,
    pub variance: f64,
    pub volatility: f64,
    pub sharpe: f64,
}

/// Trait for portfolio-construction rules (any procedure that turns a
/// universe's expected returns and covariance into weights).
///
/// Implementations include the global minimum-variance portfolio, the
/// target-return efficient frontier point, and the tangency portfolio —
/// see [`crate::frontier`] and [`crate::tangency`].
pub trait Allocator {
    /// Produce portfolio weights for the given universe.
    ///
    /// # Errors
    /// Implementations return [`PortfolioError`] when the inputs are
    /// degenerate (singular covariance, infeasible target, etc.).
    fn allocate(
        &self,
        expected_returns: &[f64],
        covariance: &[Vec<f64>],
    ) -> Result<Vec<f64>, PortfolioError>;
}

/// Compute the expected return `w' * mu` for given weights and expected
/// returns. Convenience free function.
pub fn portfolio_return(weights: &[f64], expected_returns: &[f64]) -> f64 {
    weights
        .iter()
        .zip(expected_returns.iter())
        .map(|(w, m)| w * m)
        .sum()
}

/// Compute the portfolio variance `w' * Sigma * w` for given weights and
/// covariance matrix.
pub fn portfolio_variance(weights: &[f64], covariance: &[Vec<f64>]) -> f64 {
    quadratic_form(weights, covariance)
}

/// Compute the portfolio volatility `sqrt(w' * Sigma * w)`.
pub fn portfolio_volatility(weights: &[f64], covariance: &[Vec<f64>]) -> f64 {
    portfolio_variance(weights, covariance).sqrt()
}

/// Sharpe ratio `(w' mu - rf) / sqrt(w' Sigma w)`. Returns `0.0` when the
/// portfolio volatility is zero.
pub fn sharpe_ratio(
    weights: &[f64],
    expected_returns: &[f64],
    covariance: &[Vec<f64>],
    rf: f64,
) -> f64 {
    let vol = portfolio_volatility(weights, covariance);
    if vol == 0.0 {
        return 0.0;
    }
    (portfolio_return(weights, expected_returns) - rf) / vol
}

/// Apply weights to the covariance matrix to obtain the vector of marginal
/// contributions to risk: `Sigma * w`. Useful for risk decomposition.
pub fn marginal_risk(weights: &[f64], covariance: &[Vec<f64>]) -> Vec<f64> {
    matvec(covariance, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn portfolio_return_two_assets() {
        let w = vec![0.6, 0.4];
        let mu = vec![0.10, 0.05];
        assert_abs_diff_eq!(portfolio_return(&w, &mu), 0.08, epsilon = 1e-12);
    }

    #[test]
    fn portfolio_variance_diagonal() {
        // Diagonal covariance: w' Sigma w = sum w_i^2 * sigma_i^2.
        let w = vec![0.5, 0.5];
        let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        assert_abs_diff_eq!(portfolio_variance(&w, &cov), 0.0325, epsilon = 1e-12);
    }

    #[test]
    fn sharpe_ratio_basic() {
        // mu_p = 0.5*0.10 + 0.5*0.05 = 0.075, vol_p = sqrt(0.0325) ~ 0.1803.
        let w = vec![0.5, 0.5];
        let mu = vec![0.10, 0.05];
        let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        let s = sharpe_ratio(&w, &mu, &cov, 0.02);
        assert_abs_diff_eq!(s, (0.075 - 0.02) / 0.0325_f64.sqrt(), epsilon = 1e-12);
    }
}
