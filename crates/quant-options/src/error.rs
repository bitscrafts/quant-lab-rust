//! Error types for the quant-options crate.

use thiserror::Error;

/// Errors returned by options-pricing functions.
#[derive(Debug, Error)]
pub enum OptionsError {
    /// A parameter was out of the valid range (e.g. `s0 <= 0`, `k <= 0`,
    /// `t <= 0`, `sigma < 0`).
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    /// The implied-volatility solver failed to converge within the iteration
    /// budget (Newton step diverged or bisection interval collapsed).
    #[error("implied volatility did not converge after {iterations} iterations")]
    NoConvergence { iterations: usize },

    /// The market price was inconsistent with any positive volatility
    /// (e.g. below the intrinsic value or above the upper no-arbitrage bound).
    #[error("market price {market_price:.6} outside no-arbitrage bounds [{lower:.6}, {upper:.6}]")]
    ArbitrageViolation {
        market_price: f64,
        lower: f64,
        upper: f64,
    },
}