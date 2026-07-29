//! Common imports for `quant-lib` users.
//!
//! ```rust
//! use quant_lib::prelude::*;
//! ```
//!
//! The prelude pulls in the core utilities, all Phase 14.5 traits
//! (for composability), the most common risk metrics, the basic
//! stochastic processes, the analytical Greeks, the portfolio
//! optimisers, and the AFML backtesting primitives.

#[cfg(feature = "core")]
pub use crate::core::{
    Distribution, Moments, Normal, PriceSeries, Rng, RollingWindow, XorShift64, mean, std_dev,
    variance,
};

#[cfg(feature = "core")]
pub use crate::traits::{
    BetSizer, CrossValidator, FactorModel, Greeks, ImpactModel, Labeler, OptionPricer, OptionType,
    OrderBookOps, SampleWeighter, StochasticProcess, StructuralBreak, StructuralBreakDetector,
};

#[cfg(feature = "core")]
pub use crate::risk::{
    calmar_ratio, deflated_sharpe_ratio, information_ratio, omega_ratio,
    probabilistic_sharpe_ratio, ulcer_index,
};

#[cfg(feature = "timeseries")]
pub use crate::timeseries::{OlsFit, adf_test, frac_diff, ols};

#[cfg(feature = "vol")]
pub use crate::vol::{ArchModel, GarchModel, ewma_vol};

#[cfg(feature = "stochastic")]
pub use crate::stochastic::{brownian_motion, bs_call, bs_put, gbm, mc_call, mc_put};

#[cfg(feature = "options")]
pub use crate::options::{delta, gamma, implied_vol, theta, vega};

#[cfg(feature = "portfolio")]
pub use crate::portfolio::{
    Portfolio, efficient_frontier_point, min_variance_portfolio, sharpe_ratio, tangency_portfolio,
};

#[cfg(feature = "factors")]
pub use crate::factors::{FF3Exposure, PcaResult, ff3_regression, pca, risk_attribution};

#[cfg(feature = "microstructure")]
pub use crate::microstructure::{OrderBook, Side, execution_cost, sqrt_impact, vwap};

#[cfg(feature = "backtest")]
pub use crate::backtest::{
    AfmlBacktestConfig, AfmlBacktestResult, BacktestBuilder, BetSizing, FixedHorizonLabeler,
    GenericBacktest, KellyBetSizer, LabeledEvent, PurgedKFoldConfig, TripleBarrierConfig,
    TripleBarrierLabel, WalkForward, WalkForwardConfig, afml_backtest, kelly_fraction,
    purged_kfold_splits, sample_weights, triple_barrier_label, walk_forward_efficiency,
};
