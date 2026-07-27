//! Fama-French 3-factor model: market, SMB, HML.
//!
//! The regression reuses the OLS machinery from `quant-timeseries`:
//! the design matrix is `[1, Mkt-Rf, SMB, HML]` per observation, and
//! the OLS fit returns `alpha` (intercept), `beta_mkt`, `beta_smb`,
//! `beta_hml`, and the R-squared. The residual variance is the
//! idiosyncratic (asset-specific) variance.

use crate::error::FactorError;
use quant_timeseries::ols;

/// Fama-French 3-factor exposures for a single asset.
#[derive(Debug, Clone)]
pub struct FF3Exposure {
    /// Jensen's alpha (intercept of the excess-return regression).
    pub alpha: f64,
    /// Market factor loading (coefficient on `Mkt-Rf`).
    pub beta_mkt: f64,
    /// Size factor loading (coefficient on SMB, Small Minus Big).
    pub beta_smb: f64,
    /// Value factor loading (coefficient on HML, High Minus Low).
    pub beta_hml: f64,
    /// Coefficient of determination in `[0, 1]`.
    pub r_squared: f64,
    /// Idiosyncratic (residual) variance, `Var(eps)` with the (T-1)
    /// denominator inherited from the OLS residual variance.
    pub residual_var: f64,
}

/// Regress `asset_excess` (excess returns over `Rf`) on the three
/// Fama-French factors.
///
/// `factors` is a `T x 3` matrix where each row is
/// `[Mkt-Rf, SMB, HML]`. The design matrix used internally is
/// `[1, Mkt-Rf, SMB, HML]`.
pub fn ff3_regression(
    asset_excess: &[f64],
    factors: &[Vec<f64>],
) -> Result<FF3Exposure, FactorError> {
    let t = asset_excess.len();
    if factors.len() != t {
        return Err(FactorError::DimensionMismatch(format!(
            "asset length {t} but factors has {} rows",
            factors.len()
        )));
    }
    if t < 5 {
        return Err(FactorError::InsufficientData {
            required: 5,
            actual: t,
        });
    }
    // Design matrix: [1, Mkt-Rf, SMB, HML].
    let x: Vec<Vec<f64>> = factors
        .iter()
        .map(|row| {
            if row.len() != 3 {
                return vec![1.0, 0.0, 0.0, 0.0];
            }
            vec![1.0, row[0], row[1], row[2]]
        })
        .collect();
    for (i, row) in factors.iter().enumerate() {
        if row.len() != 3 {
            return Err(FactorError::DimensionMismatch(format!(
                "factor row {i} has length {} but expected 3",
                row.len()
            )));
        }
    }
    let fit = ols(&x, asset_excess).map_err(|e| FactorError::Singular(e.to_string()))?;
    let n = asset_excess.len() as f64;
    let k = 4.0_f64;
    let dof = (n - k).max(1.0);
    let residual_var = fit.residuals.iter().map(|e| e * e).sum::<f64>() / dof;
    Ok(FF3Exposure {
        alpha: fit.coeffs[0],
        beta_mkt: fit.coeffs[1],
        beta_smb: fit.coeffs[2],
        beta_hml: fit.coeffs[3],
        r_squared: fit.r_squared,
        residual_var,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ff3_single_factor_reduces_to_capm() {
        // Generate the asset from ONLY the market factor (no SMB or HML
        // contribution). The FF3 regression should recover beta_mkt close
        // to the true generating coefficient, and beta_smb/beta_hml close
        // to zero. We cannot use exactly-zero factors (singular design),
        // so we use independent noise for SMB and HML — the key test is
        // that the FF3 betas on the noise factors are near zero.
        let mkt = [
            0.005, 0.015, -0.005, 0.02, 0.005, 0.025, -0.01, 0.01, 0.02, -0.002,
        ];
        let smb = [
            0.001, -0.002, 0.0015, -0.001, 0.002, -0.0015, 0.001, -0.002, 0.0015, -0.001,
        ];
        let hml = [
            0.0015, 0.001, -0.002, -0.0015, 0.001, 0.002, -0.001, 0.0015, -0.002, 0.001,
        ];
        let alpha_true = 0.001_f64;
        let beta_mkt_true = 1.3_f64;
        let asset: Vec<f64> = (0..10)
            .map(|t| alpha_true + beta_mkt_true * mkt[t])
            .collect();
        let factors: Vec<Vec<f64>> = (0..10).map(|t| vec![mkt[t], smb[t], hml[t]]).collect();
        let ff = ff3_regression(&asset, &factors).unwrap();
        // The market beta should match the true generating coefficient.
        assert!(
            (ff.beta_mkt - beta_mkt_true).abs() < 1e-6,
            "beta_mkt={} true={}",
            ff.beta_mkt,
            beta_mkt_true
        );
        assert!(
            (ff.alpha - alpha_true).abs() < 1e-6,
            "alpha={} true={}",
            ff.alpha,
            alpha_true
        );
        // The SMB and HML betas should be zero (asset has no SMB/HML component).
        assert!(
            ff.beta_smb.abs() < 1e-6,
            "beta_smb should be 0, got {}",
            ff.beta_smb
        );
        assert!(
            ff.beta_hml.abs() < 1e-6,
            "beta_hml should be 0, got {}",
            ff.beta_hml
        );
        // R^2 should be 1.0 (asset is an exact linear combination of the
        // market and the constant; the noise factors are orthogonal to
        // the market so they do not affect the fit).
        assert!((ff.r_squared - 1.0).abs() < 1e-6, "R^2 = {}", ff.r_squared);
    }

    #[test]
    fn test_ff3_alpha_zero_when_factor_generated() {
        // Generate asset = 0.001 + 1.1*Mkt + 0.5*SMB + 0.3*HML exactly,
        // so alpha should be ~0 and the betas should match the DGP.
        let mkt = [
            0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.005, 0.015, -0.01, 0.025,
        ];
        let smb = [
            0.005, 0.01, -0.005, 0.02, -0.01, 0.0, 0.015, -0.005, 0.01, -0.015,
        ];
        let hml = [
            -0.002, 0.003, 0.001, -0.001, 0.002, -0.003, 0.0, 0.002, -0.001, 0.003,
        ];
        let alpha_true = 0.001_f64;
        let b_m = 1.1_f64;
        let b_s = 0.5_f64;
        let b_h = 0.3_f64;
        let asset: Vec<f64> = (0..10)
            .map(|t| alpha_true + b_m * mkt[t] + b_s * smb[t] + b_h * hml[t])
            .collect();
        let factors: Vec<Vec<f64>> = (0..10).map(|t| vec![mkt[t], smb[t], hml[t]]).collect();
        let ff = ff3_regression(&asset, &factors).unwrap();
        assert!(
            (ff.alpha - alpha_true).abs() < 1e-6,
            "alpha={} true={}",
            ff.alpha,
            alpha_true
        );
        assert!((ff.beta_mkt - b_m).abs() < 1e-6);
        assert!((ff.beta_smb - b_s).abs() < 1e-6);
        assert!((ff.beta_hml - b_h).abs() < 1e-6);
        assert!((ff.r_squared - 1.0).abs() < 1e-6, "R^2 = {}", ff.r_squared);
    }
}
