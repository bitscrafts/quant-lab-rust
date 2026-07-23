//! Error type for the quant-core crate.
//!
//! See `README.md` in this directory for the module overview.

use thiserror::Error;

/// Errors raised by `quant-core` when inputs are structurally invalid.
#[derive(Error, Debug, PartialEq)]
pub enum CoreError {
    /// A price in a `PriceSeries` is non-finite (NaN or infinity) or
    /// non-positive (<= 0). Prices must be strictly positive and finite.
    #[error("Invalid price: values must be finite and strictly positive")]
    InvalidPrice,

    /// A statistical computation requires more observations than were
    /// provided. For example, sample variance needs at least two points.
    #[error("Insufficient data: need at least {required} values, got {actual}")]
    InsufficientData {
        /// Minimum number of observations required.
        required: usize,
        /// Number of observations actually supplied.
        actual: usize,
    },

    /// A rolling window parameter is zero or larger than the input length.
    #[error("Invalid window: {window} is out of range for data of length {len}")]
    InvalidWindow {
        /// The offending window value.
        window: usize,
        /// The length of the data the window was applied to.
        len: usize,
    },
}