//! Error types for the quant-portfolio crate.

use thiserror::Error;

/// Errors returned by portfolio-optimization functions.
#[derive(Debug, Error)]
pub enum PortfolioError {
    /// A parameter was out of the valid range (e.g. negative weight, `rf`
    /// outside `[-1, 1]`, empty universe).
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    /// The covariance matrix is singular (collinear assets, zero variance,
    /// or a degenerate universe). The linear-algebra backend refused to
    /// invert the system.
    #[error("singular covariance matrix: {0}")]
    SingularCovariance(String),

    /// Too few observations for the requested statistic (e.g. historical VaR
    /// requires at least one return; covariance requires at least two rows).
    #[error("insufficient data: required {required}, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// A target expected return was requested that no feasible long-only (or
    /// unconstrained) portfolio can deliver — typically outside the span of
    /// the asset expected returns.
    #[error("infeasible target return {target:.6} outside asset range [{lo:.6}, {hi:.6}]")]
    InfeasibleTarget { target: f64, lo: f64, hi: f64 },

    /// Dimension mismatch between weights, expected returns, and the
    /// covariance matrix.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
}