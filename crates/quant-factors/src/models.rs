//! Factor model trait implementations.
//!
//! This module provides wrapper structs that implement the
//! `FactorModel` trait from quant-core.

use crate::error::FactorError;
use crate::fama_french::{ff3_regression, FF3Exposure};
use crate::pca::{pca, pca_transform, PcaResult};
use quant_core::FactorModel;

/// PCA factor model.
///
/// Uses Principal Component Analysis to decompose returns into
/// orthogonal factors (principal components).
///
/// # Example
///
/// ```
/// use quant_factors::Pca;
/// use quant_core::FactorModel;
///
/// let returns = vec![
///     vec![0.01, 0.02, 0.015],
///     vec![0.02, 0.03, 0.025],
///     vec![-0.01, 0.0, -0.005],
/// ];
///
/// let mut model = Pca::new(2); // Keep top 2 components
/// model.fit(&returns).unwrap();
///
/// let asset_returns = vec![0.015, 0.025, -0.005];
/// let exposures = model.exposures(&asset_returns).unwrap();
/// assert_eq!(exposures.len(), 2); // 2 factor loadings
/// ```
pub struct Pca {
    /// Number of components to retain.
    pub n_components: usize,
    /// Fitted PCA result (None until fit() is called).
    result: Option<PcaResult>,
}

impl Pca {
    /// Create a new PCA model.
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of principal components to retain
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            result: None,
        }
    }

    /// Check if the model has been fitted.
    fn ensure_fitted(&self) -> Result<&PcaResult, FactorError> {
        self.result
            .as_ref()
            .ok_or_else(|| FactorError::InvalidParam("model not fitted yet".to_string()))
    }
}

impl FactorModel for Pca {
    type Error = FactorError;

    fn fit(&mut self, returns: &[Vec<f64>]) -> Result<(), Self::Error> {
        let result = pca(returns, self.n_components)?;
        self.result = Some(result);
        Ok(())
    }

    fn exposures(&self, returns: &[f64]) -> Result<Vec<f64>, Self::Error> {
        let result = self.ensure_fitted()?;

        // Transform single asset returns to factor scores
        let returns_matrix = vec![returns.to_vec()];
        let scores = pca_transform(&returns_matrix, &result.eigenvectors, &result.mean);

        Ok(scores[0].clone())
    }

    fn decompose(&self, returns: &[f64]) -> Result<(Vec<f64>, Vec<f64>), Self::Error> {
        let result = self.ensure_fitted()?;

        // Get factor scores (exposures)
        let exposures = self.exposures(returns)?;

        // Reconstruct returns from factors
        let scores_matrix = vec![exposures.clone()];
        let reconstructed = crate::pca::pca_reconstruct(
            &scores_matrix,
            &result.eigenvectors,
            &result.mean,
        );

        let factor_returns = reconstructed[0].clone();

        // Idiosyncratic component = actual - reconstructed
        let idiosyncratic: Vec<f64> = returns
            .iter()
            .zip(factor_returns.iter())
            .map(|(r, f)| r - f)
            .collect();

        Ok((factor_returns, idiosyncratic))
    }
}

/// Fama-French 3-factor model.
///
/// Models returns using market (Mkt-Rf), size (SMB), and value (HML) factors.
///
/// # Example
///
/// ```
/// use quant_factors::FamaFrench3;
/// use quant_core::FactorModel;
///
/// // Factor returns: [Mkt-Rf, SMB, HML]
/// let factors = vec![
///     vec![0.01, 0.002, 0.001],
///     vec![-0.005, 0.001, -0.001],
///     vec![0.015, -0.001, 0.002],
/// ];
///
/// let mut model = FamaFrench3::new(factors);
///
/// // Asset excess returns
/// let asset_returns = vec![0.012, -0.004, 0.018];
/// model.fit(&vec![asset_returns.clone()]).unwrap();
///
/// let exposures = model.exposures(&asset_returns).unwrap();
/// assert_eq!(exposures.len(), 3); // beta_mkt, beta_smb, beta_hml
/// ```
pub struct FamaFrench3 {
    /// Factor returns matrix: T x 3 [Mkt-Rf, SMB, HML]
    pub factors: Vec<Vec<f64>>,
    /// Fitted exposure (None until fit() is called).
    exposure: Option<FF3Exposure>,
}

impl FamaFrench3 {
    /// Create a new Fama-French 3-factor model.
    ///
    /// # Arguments
    ///
    /// * `factors` - T x 3 matrix of [Mkt-Rf, SMB, HML] factor returns
    pub fn new(factors: Vec<Vec<f64>>) -> Self {
        Self {
            factors,
            exposure: None,
        }
    }

    /// Check if the model has been fitted.
    fn ensure_fitted(&self) -> Result<&FF3Exposure, FactorError> {
        self.exposure
            .as_ref()
            .ok_or_else(|| FactorError::InvalidParam("model not fitted yet".to_string()))
    }
}

impl FactorModel for FamaFrench3 {
    type Error = FactorError;

    fn fit(&mut self, returns: &[Vec<f64>]) -> Result<(), Self::Error> {
        // For Fama-French, we expect a single asset's returns
        if returns.is_empty() {
            return Err(FactorError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        // Use the first asset's returns for fitting
        let asset_returns = &returns[0];

        let exposure = ff3_regression(asset_returns, &self.factors)?;
        self.exposure = Some(exposure);
        Ok(())
    }

    fn exposures(&self, _returns: &[f64]) -> Result<Vec<f64>, Self::Error> {
        let exp = self.ensure_fitted()?;

        // Return factor loadings: [beta_mkt, beta_smb, beta_hml]
        Ok(vec![exp.beta_mkt, exp.beta_smb, exp.beta_hml])
    }

    fn decompose(&self, returns: &[f64]) -> Result<(Vec<f64>, Vec<f64>), Self::Error> {
        let exp = self.ensure_fitted()?;

        if returns.len() != self.factors.len() {
            return Err(FactorError::DimensionMismatch(format!(
                "returns length {} but factors has {} rows",
                returns.len(),
                self.factors.len()
            )));
        }

        // Factor component: alpha + beta_mkt*Mkt + beta_smb*SMB + beta_hml*HML
        let factor_returns: Vec<f64> = self
            .factors
            .iter()
            .map(|f| {
                exp.alpha
                    + exp.beta_mkt * f[0]
                    + exp.beta_smb * f[1]
                    + exp.beta_hml * f[2]
            })
            .collect();

        // Idiosyncratic component: actual - predicted
        let idiosyncratic: Vec<f64> = returns
            .iter()
            .zip(factor_returns.iter())
            .map(|(r, f)| r - f)
            .collect();

        Ok((factor_returns, idiosyncratic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_implements_factor_model() {
        fn _assert_trait<T: FactorModel>() {}
        _assert_trait::<Pca>();
    }

    #[test]
    fn test_pca_fit_exposures() {
        let returns = vec![
            vec![0.01, 0.02, 0.015],
            vec![0.02, 0.03, 0.025],
            vec![-0.01, 0.0, -0.005],
            vec![0.015, 0.025, 0.02],
        ];

        let mut model = Pca::new(2);
        model.fit(&returns).unwrap();

        let asset_returns = vec![0.015, 0.025, 0.02];
        let exposures = model.exposures(&asset_returns).unwrap();

        assert_eq!(exposures.len(), 2); // 2 components
    }

    #[test]
    fn test_pca_decompose() {
        let returns = vec![
            vec![0.01, 0.02, 0.015],
            vec![0.02, 0.03, 0.025],
            vec![-0.01, 0.0, -0.005],
            vec![0.015, 0.025, 0.02],
        ];

        let mut model = Pca::new(2);
        model.fit(&returns).unwrap();

        let asset_returns = vec![0.015, 0.025, 0.02];
        let (factor_comp, idio_comp) = model.decompose(&asset_returns).unwrap();

        assert_eq!(factor_comp.len(), asset_returns.len());
        assert_eq!(idio_comp.len(), asset_returns.len());

        // Factor + idiosyncratic should approximately equal original
        for i in 0..asset_returns.len() {
            let reconstructed = factor_comp[i] + idio_comp[i];
            assert!((reconstructed - asset_returns[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fama_french_implements_factor_model() {
        fn _assert_trait<T: FactorModel>() {}
        _assert_trait::<FamaFrench3>();
    }

    #[test]
    fn test_fama_french_fit_exposures() {
        let factors = vec![
            vec![0.01, 0.002, 0.001],
            vec![-0.005, 0.001, -0.001],
            vec![0.015, -0.001, 0.002],
            vec![0.005, 0.0, 0.001],
            vec![0.02, 0.003, -0.001],
        ];

        // Generate returns from factors: r = 0.001 + 1.2*Mkt + 0.5*SMB + 0.3*HML
        let asset_returns: Vec<f64> = factors
            .iter()
            .map(|f| 0.001 + 1.2 * f[0] + 0.5 * f[1] + 0.3 * f[2])
            .collect();

        let mut model = FamaFrench3::new(factors);
        model.fit(&vec![asset_returns.clone()]).unwrap();

        let exposures = model.exposures(&asset_returns).unwrap();

        assert_eq!(exposures.len(), 3);
        // Should recover beta_mkt ≈ 1.2
        assert!((exposures[0] - 1.2).abs() < 1e-6);
        // Should recover beta_smb ≈ 0.5
        assert!((exposures[1] - 0.5).abs() < 1e-6);
        // Should recover beta_hml ≈ 0.3
        assert!((exposures[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_fama_french_decompose() {
        let factors = vec![
            vec![0.01, 0.002, 0.001],
            vec![-0.005, 0.001, -0.001],
            vec![0.015, -0.001, 0.002],
            vec![0.005, 0.0, 0.001],
            vec![0.02, 0.003, -0.001],
        ];

        // Perfect factor model (no idiosyncratic component)
        let asset_returns: Vec<f64> = factors
            .iter()
            .map(|f| 0.001 + 1.2 * f[0] + 0.5 * f[1] + 0.3 * f[2])
            .collect();

        let mut model = FamaFrench3::new(factors);
        model.fit(&vec![asset_returns.clone()]).unwrap();

        let (factor_comp, idio_comp) = model.decompose(&asset_returns).unwrap();

        assert_eq!(factor_comp.len(), asset_returns.len());
        assert_eq!(idio_comp.len(), asset_returns.len());

        // With perfect factor model, idiosyncratic should be ~0
        for &resid in &idio_comp {
            assert!(resid.abs() < 1e-10);
        }

        // Factor component should match original
        for (i, &factor_ret) in factor_comp.iter().enumerate() {
            assert!((factor_ret - asset_returns[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pca_not_fitted_error() {
        let model = Pca::new(2);
        let returns = vec![0.01, 0.02, 0.03];
        let result = model.exposures(&returns);
        assert!(result.is_err());
    }

    #[test]
    fn test_fama_french_not_fitted_error() {
        let factors = vec![vec![0.01, 0.0, 0.0]];
        let model = FamaFrench3::new(factors);
        let returns = vec![0.01];
        let result = model.exposures(&returns);
        assert!(result.is_err());
    }
}
