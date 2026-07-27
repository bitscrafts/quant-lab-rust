//! Risk attribution: systematic vs idiosyncratic variance decomposition.
//!
//! Given portfolio weights `w`, factor loadings `B` (N assets x K
//! factors), the factor covariance `Sigma_F` (K x K), and the per-asset
//! idiosyncratic variances `D` (diagonal of N), the total portfolio
//! variance decomposes as `sigma_p^2 = w' B Sigma_F B' w + w' D w`
//! (systematic + idiosyncratic).
//!
//! The per-factor contribution is the diagonal approximation
//! `f_p[k]^2 * Sigma_F[k][k]` where `f_p = B' w` is the portfolio factor
//! exposure; the sum of these contributions recovers the systematic
//! variance when `Sigma_F` is diagonal.

use crate::error::FactorError;

/// Decomposition of portfolio variance into factor and idiosyncratic parts.
#[derive(Debug, Clone)]
pub struct RiskAttribution {
    /// Total portfolio variance `w' Sigma w` reconstructed from the
    /// factor model.
    pub total_variance: f64,
    /// Systematic (factor-driven) variance `w' B Sigma_F B' w`.
    pub systematic_variance: f64,
    /// Idiosyncratic variance `w' D w` (asset-specific residual).
    pub idiosyncratic_variance: f64,
    /// Per-factor contribution to the systematic variance (diagonal
    /// approximation: `sum_i sum_j w_i w_j beta_ik beta_jk * Sigma_F[kk]`).
    pub factor_contributions: Vec<f64>,
}

/// Decompose portfolio variance into systematic and idiosyncratic parts.
///
/// # Arguments
/// * `weights` - Portfolio weights, length `N`.
/// * `factor_loadings` - `N x K` matrix of factor betas (`B`).
/// * `factor_covariance` - `K x K` factor covariance matrix (`Sigma_F`).
/// * `residual_variances` - Per-asset idiosyncratic variances, length `N` (`D`).
pub fn risk_attribution(
    weights: &[f64],
    factor_loadings: &[Vec<f64>],
    factor_covariance: &[Vec<f64>],
    residual_variances: &[f64],
) -> Result<RiskAttribution, FactorError> {
    let n = weights.len();
    if factor_loadings.len() != n {
        return Err(FactorError::DimensionMismatch(format!(
            "weights length {n} but factor_loadings has {} rows",
            factor_loadings.len()
        )));
    }
    if residual_variances.len() != n {
        return Err(FactorError::DimensionMismatch(format!(
            "weights length {n} but residual_variances has length {}",
            residual_variances.len()
        )));
    }
    let k = if n > 0 { factor_loadings[0].len() } else { 0 };
    if k == 0 {
        return Err(FactorError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    for (i, row) in factor_loadings.iter().enumerate() {
        if row.len() != k {
            return Err(FactorError::DimensionMismatch(format!(
                "factor_loadings row {i} has length {} but expected {k}",
                row.len()
            )));
        }
    }
    if factor_covariance.len() != k {
        return Err(FactorError::DimensionMismatch(format!(
            "factor_covariance is {}x{} but expected {k}x{k}",
            factor_covariance.len(),
            if factor_covariance.is_empty() {
                0
            } else {
                factor_covariance[0].len()
            }
        )));
    }
    for (i, row) in factor_covariance.iter().enumerate() {
        if row.len() != k {
            return Err(FactorError::DimensionMismatch(format!(
                "factor_covariance row {i} has length {} but expected {k}",
                row.len()
            )));
        }
    }

    // Portfolio factor exposures: f_p[k] = sum_i w_i * beta_ik.
    let mut f_p = vec![0.0_f64; k];
    for (i, w_i) in weights.iter().enumerate() {
        for (kk, &beta) in factor_loadings[i].iter().enumerate() {
            f_p[kk] += w_i * beta;
        }
    }

    // Systematic variance: f_p' Sigma_F f_p.
    let mut systematic_variance = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            systematic_variance += f_p[i] * factor_covariance[i][j] * f_p[j];
        }
    }

    // Idiosyncratic variance: sum_i w_i^2 * D_i.
    let mut idiosyncratic_variance = 0.0_f64;
    for (i, w_i) in weights.iter().enumerate() {
        idiosyncratic_variance += w_i * w_i * residual_variances[i];
    }

    // Per-factor contribution (diagonal of the systematic variance):
    // contribution_k = f_p[k]^2 * Sigma_F[k][k].
    let factor_contributions: Vec<f64> = (0..k)
        .map(|kk| f_p[kk] * f_p[kk] * factor_covariance[kk][kk])
        .collect();

    let total_variance = systematic_variance + idiosyncratic_variance;

    Ok(RiskAttribution {
        total_variance,
        systematic_variance,
        idiosyncratic_variance,
        factor_contributions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_systematic_plus_idio_equals_total() {
        let weights = vec![0.6, 0.4];
        let loadings = vec![vec![1.2, 0.3], vec![0.8, -0.1]];
        let factor_cov = vec![vec![0.04, 0.0], vec![0.0, 0.01]];
        let resid = vec![0.002, 0.001];
        let ra = risk_attribution(&weights, &loadings, &factor_cov, &resid).unwrap();
        assert!(
            (ra.total_variance - (ra.systematic_variance + ra.idiosyncratic_variance)).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_factor_contribution_sum_matches_systematic_diagonal() {
        // For a diagonal factor covariance, sum(contributions) == systematic.
        let weights = vec![0.5, 0.5];
        let loadings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let factor_cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
        let resid = vec![0.0, 0.0];
        let ra = risk_attribution(&weights, &loadings, &factor_cov, &resid).unwrap();
        let sum_contrib: f64 = ra.factor_contributions.iter().sum();
        // f_p = [0.5, 0.5], Sigma_F diag = [0.04, 0.09]
        // systematic = 0.5^2*0.04 + 0.5^2*0.09 = 0.01 + 0.0225 = 0.0325
        assert!(
            (sum_contrib - ra.systematic_variance).abs() < 1e-12,
            "sum_contrib={sum_contrib} systematic={}",
            ra.systematic_variance
        );
        assert!((ra.systematic_variance - 0.0325).abs() < 1e-12);
    }
}
