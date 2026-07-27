//! Error type for the quant-stochastic crate.
//!
//! See `README.md` in this directory for the module overview.

use thiserror::Error;

/// Errors raised by `quant-stochastic` when inputs are structurally invalid
/// or a simulation fails to produce a usable result.
#[derive(Error, Debug, PartialEq)]
pub enum StochError {
    /// A parameter is outside its valid domain (e.g. non-positive rate,
    /// negative time horizon, zero steps, non-positive price or strike).
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// A simulation requires more data or steps than were provided.
    #[error("Insufficient data: need at least {required} steps/paths, got {actual}")]
    InsufficientData {
        /// Minimum number of steps or paths required.
        required: usize,
        /// Number actually supplied.
        actual: usize,
    },

    /// A Monte Carlo estimator failed to converge to a finite result within
    /// the requested number of paths.
    #[error("Monte Carlo did not converge after {n_paths} paths")]
    ConvergenceFailure {
        /// Number of paths simulated before giving up.
        n_paths: usize,
    },
}
