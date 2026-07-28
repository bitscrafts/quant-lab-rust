//! Factor model trait.
//!
//! Defines the [`FactorModel`] trait for multi-factor risk models
//! (e.g., PCA, Fama-French).

use std::error::Error;

/// A factor model decomposes returns into common factors.
///
/// Factor models (PCA, Fama-French, APT) implement this trait to
/// provide a consistent interface for factor analysis and risk
/// decomposition.
pub trait FactorModel {
    /// The error type returned when fitting or decomposing fails.
    type Error: Error;

    /// Fit the factor model to a returns matrix.
    ///
    /// # Arguments
    ///
    /// * `returns` - Matrix of returns (n_observations × n_assets)
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (e.g., singular matrix,
    /// insufficient data).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut pca = Pca::new(3); // 3 principal components
    /// pca.fit(&returns_matrix)?;
    /// ```
    fn fit(&mut self, returns: &[Vec<f64>]) -> Result<(), Self::Error>;

    /// Compute factor exposures for new returns.
    ///
    /// # Arguments
    ///
    /// * `returns` - Vector of returns for a single asset or portfolio
    ///
    /// # Returns
    ///
    /// A vector of factor exposures (loadings).
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fitted, or if
    /// the returns vector has the wrong dimension.
    fn exposures(&self, returns: &[f64]) -> Result<Vec<f64>, Self::Error>;

    /// Decompose returns into factor and idiosyncratic components.
    ///
    /// # Arguments
    ///
    /// * `returns` - Vector of returns for a single asset
    ///
    /// # Returns
    ///
    /// A tuple `(factor_returns, idiosyncratic_returns)` where
    /// factor_returns is the portion explained by factors and
    /// idiosyncratic_returns is the residual.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fitted.
    fn decompose(&self, returns: &[f64]) -> Result<(Vec<f64>, Vec<f64>), Self::Error>;
}
