//! Error type for the `quant-factors` crate.

use thiserror::Error;

/// Errors returned by factor-model, PCA, and risk-attribution routines.
#[derive(Debug, Error)]
pub enum FactorError {
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    #[error("non-converged power iteration after {0} steps (delta = {1:.3e})")]
    NonConverged(usize, f64),

    #[error("singular or degenerate matrix: {0}")]
    Singular(String),

    #[error("insufficient data: required {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("infeasible decomposition: {0}")]
    Infeasible(String),
}
