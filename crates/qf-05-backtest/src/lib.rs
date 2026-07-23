//! Backtesting basics for the quant-finance curriculum (Phase 5).
//!
//! This crate introduces systematic trading-strategy evaluation:
//!
//! - Signal generation from indicators (SMA crossover)
//! - Position management (long / flat — short selling deferred to later phases)
//! - P&L accounting with transaction costs
//! - Performance evaluation reusing [`qf_04_returns`] (Sharpe, Sortino,
//!   drawdown, volatility)
//! - A passive buy-and-hold benchmark for comparison
//!
//! # Design
//!
//! The [`Strategy`] trait maps a price history up to the current bar to a
//! single [`Signal`]. The engine converts signals into position transitions
//! and applies transaction costs on every entry and exit. A backtest is a
//! simulation, not a prediction — the #1 failure mode is overfitting to
//! historical data.
//!
//! # Example
//!
//! ```
//! use qf_03_stocks::Ohlcv;
//! use qf_05_backtest::{run_backtest, BacktestConfig, BuyAndHold};
//!
//! let data = vec![
//!     Ohlcv { date: "2024-01-01".into(), open: 100.0, high: 101.0, low: 99.0, close: 100.0, volume: 1000, adj_close: None },
//!     Ohlcv { date: "2024-01-02".into(), open: 100.0, high: 102.0, low: 100.0, close: 101.0, volume: 1000, adj_close: None },
//!     Ohlcv { date: "2024-01-03".into(), open: 101.0, high: 103.0, low: 101.0, close: 102.0, volume: 1000, adj_close: None },
//! ];
//! let result = run_backtest(&BuyAndHold, &data, &BacktestConfig::default()).unwrap();
//! assert_eq!(result.equity_curve.len(), 3);
//! ```

pub mod backtest;
pub mod error;
pub mod signal;
pub mod strategy;

pub use backtest::{run_backtest, BacktestConfig, BacktestResult, Trade};
pub use error::BacktestError;
pub use signal::{Position, Signal};
pub use strategy::{BuyAndHold, SmaCrossover, Strategy};