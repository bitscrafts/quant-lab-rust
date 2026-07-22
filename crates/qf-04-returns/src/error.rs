//! Error types for the returns and volatility crate.
//!
//! See `README.md` in this directory for the module overview.

use thiserror::Error;

/// Errors returned by `qf-04-returns` operations.
#[derive(Error, Debug)]
pub enum ReturnsError {
    /// The input did not contain enough observations to compute the requested
    /// statistic (e.g. computing volatility needs at least two returns).
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// A parameter was invalid (e.g. a non-positive window size or a
    /// non-finite annualisation factor).
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}