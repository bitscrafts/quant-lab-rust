//! Error types for the quant-vol crate.

use thiserror::Error;

/// Errors returned by volatility-model functions.
#[derive(Debug, Error)]
pub enum VolError {
    /// A parameter was out of the valid range (e.g. `lambda` outside `[0, 1]`,
    /// negative order, non-positive threshold).
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    /// The input series was too short for the requested model or order.
    #[error("insufficient data: need {required} observations, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// The MLE optimiser failed to converge within the iteration budget.
    #[error("fitting failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// The fitted model is non-stationary (persistence >= 1).
    #[error("non-stationary model: persistence {persistence:.4} >= 1")]
    NonStationary { persistence: f64 },
}
