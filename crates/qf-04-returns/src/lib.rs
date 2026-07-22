//! Returns, volatility, and risk-adjusted metrics for the quant-finance
//! curriculum (Phase 4).
//!
//! This crate builds on [`qf_03_stocks`] (`Ohlcv`, `TimeSeries`) and
//! [`qf_common`] to introduce the foundational vocabulary of quantitative
//! finance:
//!
//! - Simple and logarithmic returns (`returns`)
//! - Volatility: sample standard deviation, annualised, and rolling
//!   (`volatility`)
//! - Risk-adjusted metrics: Sharpe and Sortino ratios (`risk`)
//! - Drawdown analysis (`drawdown`)
//!
//! All math is hand-rolled (no external statistics crates), following the
//! pedagogy-first policy of the `quant-lab` workspace. Division-by-zero
//! cases return `0.0` rather than panicking.
//!
//! # Example
//!
//! ```
//! use qf_04_returns::{simple_returns, volatility, sharpe_ratio};
//!
//! let prices = vec![100.0, 102.0, 101.0, 105.0];
//! let r = simple_returns(&prices);
//! let vol = volatility(&r);
//! let sharpe = sharpe_ratio(&r, 0.0);
//! ```

pub mod drawdown;
pub mod error;
pub mod returns;
pub mod risk;
pub mod volatility;

pub use drawdown::{drawdown, drawdown_stats, max_drawdown, DrawdownStats};
pub use error::ReturnsError;
pub use returns::{cumulative_returns, log_returns, simple_returns, Returns};
pub use risk::{annualized_sharpe, sharpe_ratio, sortino_ratio};
pub use volatility::{annualized_volatility, rolling_volatility, volatility};