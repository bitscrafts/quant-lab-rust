//! Error type for the quant-timeseries crate.
//!
//! See `README.md` in this directory for the module overview.

use thiserror::Error;

/// Errors raised by `quant-timeseries` when inputs are structurally invalid.
#[derive(Error, Debug, PartialEq)]
pub enum TimeSeriesError {
    /// The design matrix is singular (collinear columns) and the normal
    /// equations have no unique solution.
    #[error("Singular matrix: cannot solve normal equations (collinear columns?)")]
    Singular,

    /// The dimensions of `x` and `y` do not match (number of rows of `x` must
    /// equal length of `y`).
    #[error("Dimension mismatch: x has {x_rows} rows, y has {y_len}")]
    DimensionMismatch {
        /// Number of rows in the design matrix `x`.
        x_rows: usize,
        /// Length of the response vector `y`.
        y_len: usize,
    },

    /// The requested lag is invalid (zero, or larger than the data allows).
    #[error("Invalid lag: {lag} is out of range for data of length {len}")]
    InvalidLag {
        /// The offending lag value.
        lag: usize,
        /// The length of the data the lag was applied to.
        len: usize,
    },

    /// A parameter is out of its valid range.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// The data is too short for the requested computation.
    #[error("Insufficient data: need at least {required} values, got {actual}")]
    InsufficientData {
        /// Minimum number of observations required.
        required: usize,
        /// Number of observations actually supplied.
        actual: usize,
    },
}