//! Error type for the backtest crate.
//!
//! See `README.md` in this directory for the module overview.

use thiserror::Error;

/// Errors that can occur while building or running a backtest.
#[derive(Error, Debug)]
pub enum BacktestError {
    /// Strategy parameters are invalid (e.g. short period >= long period).
    #[error("Invalid strategy parameters: {0}")]
    InvalidParams(String),

    /// Not enough bars to run the strategy.
    #[error("Insufficient data: need at least {required} bars, got {actual}")]
    InsufficientData {
        /// Minimum number of bars the strategy needs.
        required: usize,
        /// Number of bars actually supplied.
        actual: usize,
    },

    /// Backtest configuration is invalid (e.g. negative capital).
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}