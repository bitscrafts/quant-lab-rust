//! Foundations of quantitative finance: series types, moments, rolling
//! windows, and deterministic simulation.
//!
//! `quant-core` is the advanced-track foundation crate for the quant-finance
//! curriculum (Phase 6). All math is hand-rolled — no `rand`, `nalgebra`, or
//! `statrs` — so the pedagogy stays in the implementation.
//!
//! # Modules
//!
//! - [`series`]: `PriceSeries` newtype, simple and log returns
//! - [`moments`]: mean, variance, skewness, excess kurtosis, `Moments` trait
//! - [`rolling`]: generic rolling window, `RollingWindow` trait
//! - [`sim`]: `XorShift64` RNG, `Normal` distribution via Box-Muller, `gbm_paths`
//!
//! # Example
//!
//! ```
//! use quant_core::{XorShift64, Normal, Distribution, gbm_paths};
//!
//! let mut rng = XorShift64::new(42);
//! let paths = gbm_paths(100.0, 0.05, 0.2, 1.0, 252, 1, &mut rng);
//! assert_eq!(paths.len(), 1);
//! assert_eq!(paths[0].len(), 253);
//! ```

pub mod error;
pub mod moments;
pub mod rolling;
pub mod series;
pub mod sim;

pub use error::CoreError;
pub use moments::{excess_kurtosis, mean, skewness, std_dev, variance, Moments};
pub use rolling::{rolling, rolling_mean, rolling_std_dev, RollingWindow};
pub use series::{log_returns, series_log_returns, series_simple_returns, simple_returns, PriceSeries};
pub use sim::{gbm_paths, box_muller_normal, Distribution, Normal, Rng, XorShift64};