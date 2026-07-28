//! AFML (López de Prado) backtesting framework.
//!
//! `quant-backtest` is the Phase 14 crate of the quant-finance
//! curriculum. It implements:
//!
//! - Triple-barrier labeling (profit-taking, stop-loss, time barriers)
//! - Sample weights from event uniqueness
//! - Purged k-fold cross-validation with embargo
//! - Kelly criterion bet sizing
//! - An end-to-end backtest pipeline combining all of the above
//!
//! All math is hand-rolled --- no external ML or optimisation
//! libraries. The Kelly criterion, sample weights, and purging logic
//! are implemented inline.
//!
//! # Modules
//!
//! - `triple_barrier`: `TripleBarrierConfig`, `TripleBarrierLabel`,
//!   `LabeledEvent`, `triple_barrier_label`
//! - `weights`: `concurrent_events`, `average_uniqueness`,
//!   `sample_weights`
//! - `purged_kfold`: `PurgedKFoldConfig`, `PurgedSplit`,
//!   `purged_kfold_splits`, `event_overlaps`
//! - `kelly`: `kelly_fraction`, `fractional_kelly`,
//!   `kelly_from_returns`, `compute_position_size`, `PositionSize`
//! - `backtest`: `AfmlBacktestConfig`, `BetSizing`,
//!   `AfmlBacktestResult`, `afml_backtest`
//! - `error`: `BacktestError`
//!
//! # Example
//!
//! ```
//! use quant_backtest::{AfmlBacktestConfig, BetSizing, PurgedKFoldConfig,
//!     TripleBarrierConfig, afml_backtest};
//!
//! let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01)).collect();
//! let config = AfmlBacktestConfig {
//!     barrier_config: TripleBarrierConfig {
//!         upper_barrier: 0.02, lower_barrier: -0.02,
//!         time_barrier: 5, min_return: 0.0,
//!     },
//!     cv_config: PurgedKFoldConfig { n_folds: 3, embargo: 2 },
//!     use_weights: true,
//!     bet_sizing: BetSizing::Equal,
//!     entry_step: 3,
//! };
//! let result = afml_backtest(&prices, &config).unwrap();
//! assert!(!result.events.is_empty());
//! ```

pub mod backtest;
pub mod error;
pub mod generic;
pub mod kelly;
pub mod labelers;
pub mod purged_kfold;
pub mod sizers;
pub mod triple_barrier;
pub mod walk_forward;
pub mod weights;

pub use backtest::{afml_backtest, AfmlBacktestConfig, AfmlBacktestResult, BetSizing};
pub use error::{BacktestError, BacktestResult};
pub use generic::{BacktestBuilder, GenericBacktest};
pub use kelly::{compute_position_size, fractional_kelly, kelly_fraction, kelly_from_returns, PositionSize};
pub use labelers::{DynamicBarrierLabeler, FixedHorizonLabeler, TrendScanningLabeler};
pub use purged_kfold::{event_overlaps, purged_kfold_splits, PurgedKFoldConfig, PurgedSplit};
pub use sizers::{EqualBetSizer, FixedBetSizer, KellyBetSizer};
pub use triple_barrier::{
    to_binary_label_helper, triple_barrier_label, LabeledEvent, TripleBarrierConfig,
    TripleBarrierLabel,
};
pub use walk_forward::{walk_forward_efficiency, WalkForward, WalkForwardConfig, WalkForwardSplit};
pub use weights::{average_uniqueness, concurrent_events, sample_weights};