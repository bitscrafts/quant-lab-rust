//! Unified quantitative finance library.
//!
//! `quant-lib` consolidates all Phase 6-14.5 crates into a single
//! facade. It re-exports the public APIs of:
//!
//! - `quant-core` --- moments, rolling windows, RNG, GBM, risk metrics,
//!   core traits (Phase 14.5)
//! - `quant-timeseries` --- OLS, ACF, ADF, fractional differentiation,
//!   CUSUM breaks
//! - `quant-vol` --- EWMA, ARCH, GARCH
//! - `quant-stochastic` --- Brownian motion, GBM, Poisson, Monte Carlo,
//!   `StochasticProcess` implementations (Gbm, Heston, Vasicek)
//! - `quant-options` --- Black-Scholes, Greeks, implied vol,
//!   `OptionPricer` implementation (BlackScholes)
//! - `quant-portfolio` --- Markowitz, efficient frontier, tangency,
//!   CAPM, VaR/CVaR
//! - `quant-factors` --- PCA, Fama-French, risk attribution,
//!   `FactorModel` implementation (CapmModel)
//! - `quant-microstructure` --- LOB, OFI, market impact,
//!   `ImpactModel` and `OrderBookOps` implementations
//! - `quant-backtest` --- triple-barrier, sample weights, purged CV,
//!   Kelly, walk-forward, `GenericBacktest<L, CV, BS>` composable
//!   engine (Phase 14.5)
//!
//! # Non-destructive facade
//!
//! `quant-lib` adds **zero new logic**. Every line in this file is
//! `pub mod` or `pub use`. All math, types, and traits live in the
//! source crates. The facade exists only to give downstream projects
//! a single dependency and a unified namespace.
//!
//! # Feature flags
//!
//! All modules are enabled by the `all` feature (on by default).
//! Disable default features and enable individual modules for
//! smaller compile times:
//!
//! ```toml
//! [dependencies]
//! quant-lib = { version = "0.1", default-features = false, features = ["core", "timeseries"] }
//! ```
//!
//! # Quick start
//!
//! ```rust
//! use quant_lib::prelude::*;
//!
//! let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
//! let m = quant_lib::core::mean(&data);
//! assert!(m.is_finite());
//! ```

pub mod prelude;

#[cfg(feature = "core")]
pub mod core {
    //! Foundations: moments, rolling windows, RNG, GBM, risk metrics,
    //! and core traits for pluggable components (Phase 14.5).
    pub use quant_core::*;
}

#[cfg(feature = "core")]
pub mod traits {
    //! Core trait definitions for pluggable components (Phase 14.5).
    //!
    //! These traits enable composable, swappable implementations
    //! across the entire quant-lib ecosystem: `CrossValidator`,
    //! `Labeler`, `BetSizer`, `SampleWeighter`, `StochasticProcess`,
    //! `OptionPricer`, `Greeks`, `FactorModel`, `ImpactModel`,
    //! `OrderBookOps`, `StructuralBreakDetector`.
    pub use quant_core::traits::*;
}

#[cfg(feature = "core")]
pub mod risk {
    //! Advanced risk metrics (Phase 14.5): Probabilistic Sharpe Ratio,
    //! Deflated Sharpe Ratio, Calmar, Omega, Information Ratio, Ulcer
    //! Index.
    pub use quant_core::risk::*;
}

#[cfg(feature = "timeseries")]
pub mod timeseries {
    //! Time-series econometrics: OLS, ACF, ADF, fractional
    //! differentiation, CUSUM structural-break detection.
    pub use quant_timeseries::*;
}

#[cfg(feature = "vol")]
pub mod vol {
    //! Volatility models: EWMA, ARCH, GARCH.
    pub use quant_vol::*;
}

#[cfg(feature = "stochastic")]
pub mod stochastic {
    //! Stochastic processes: Brownian motion, GBM, Poisson
    //! jump-diffusion, Monte Carlo, `StochasticProcess` implementations.
    pub use quant_stochastic::*;
}

#[cfg(feature = "options")]
pub mod options {
    //! Options pricing: Black-Scholes, Greeks (analytical and
    //! finite-difference), implied volatility, `OptionPricer`
    //! implementation.
    pub use quant_options::*;
}

#[cfg(feature = "portfolio")]
pub mod portfolio {
    //! Portfolio optimization: Markowitz frontier, tangency, CAPM,
    //! VaR/CVaR.
    pub use quant_portfolio::*;
}

#[cfg(feature = "factors")]
pub mod factors {
    //! Factor models: PCA, Fama-French, risk attribution,
    //! `FactorModel` implementation.
    pub use quant_factors::*;
}

#[cfg(feature = "microstructure")]
pub mod microstructure {
    //! Market microstructure: LOB, OFI, market impact, `ImpactModel`
    //! and `OrderBookOps` implementations.
    pub use quant_microstructure::*;
}

#[cfg(feature = "backtest")]
pub mod backtest {
    //! AFML backtesting: triple-barrier, sample weights, purged CV,
    //! Kelly, walk-forward, `GenericBacktest<L, CV, BS>` composable
    //! engine (Phase 14.5).
    pub use quant_backtest::*;
}

// ============================================================================
// Root-level convenience re-exports
// ============================================================================

#[cfg(feature = "core")]
pub use core::{
    Distribution, Normal, PriceSeries, RollingWindow, XorShift64, mean, std_dev, variance,
};

#[cfg(feature = "core")]
pub use traits::{
    BetSizer, CrossValidator, FactorModel, Greeks, ImpactModel, Labeler, OptionPricer, OptionType,
    OrderBookOps, SampleWeighter, StochasticProcess, StructuralBreak, StructuralBreakDetector,
};

#[cfg(feature = "core")]
pub use risk::{
    calmar_ratio, deflated_sharpe_ratio, information_ratio, omega_ratio,
    probabilistic_sharpe_ratio, ulcer_index,
};

#[cfg(feature = "stochastic")]
pub use stochastic::{brownian_motion, bs_call, bs_put, gbm, normal_cdf};

#[cfg(feature = "options")]
pub use options::{delta, gamma, implied_vol, theta, vega};

#[cfg(feature = "portfolio")]
pub use portfolio::{
    efficient_frontier_point, min_variance_portfolio, sharpe_ratio, tangency_portfolio,
};

#[cfg(feature = "backtest")]
pub use backtest::{
    BacktestBuilder, GenericBacktest, KellyBetSizer, WalkForward, kelly_fraction,
    purged_kfold_splits, triple_barrier_label, walk_forward_efficiency,
};
