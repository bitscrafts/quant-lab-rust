//! Generic backtest engine with trait-based composition.
//!
//! This module provides a flexible backtest framework that composes:
//! - Event labeling (via `Labeler` trait)
//! - Cross-validation (via `CrossValidator` trait)
//! - Bet sizing (via `BetSizer` trait)
//!
//! # Example
//!
//! ```
//! use quant_backtest::{BacktestBuilder, FixedHorizonLabeler, WalkForward, KellyBetSizer};
//! use quant_backtest::WalkForwardConfig;
//!
//! let labeler = FixedHorizonLabeler {
//!     horizon: 10,
//!     min_return: 0.01,
//! };
//!
//! let cv = WalkForward::new(WalkForwardConfig::rolling(100, 20, 20));
//!
//! let sizer = KellyBetSizer { fraction: 0.5 };
//!
//! let backtest = BacktestBuilder::new()
//!     .labeler(labeler)
//!     .cv(cv)
//!     .sizer(sizer)
//!     .entry_step(5)
//!     .build();
//!
//! // let results = backtest.run(&prices, &returns)?;
//! ```

use quant_core::{BetSizer, CrossValidator, Labeler};

use crate::{BacktestError, BacktestResult, WalkForwardSplit};

/// Generic backtest engine that composes labeling, cross-validation, and bet sizing.
///
/// This struct uses trait bounds to allow any combination of:
/// - `L: Labeler` - Event labeling strategy
/// - `CV: CrossValidator` - Train/test split strategy
/// - `BS: BetSizer` - Position sizing strategy
pub struct GenericBacktest<L, CV, BS>
where
    L: Labeler,
    CV: CrossValidator<Split = WalkForwardSplit>,
    BS: BetSizer,
{
    labeler: L,
    cv: CV,
    sizer: BS,
    entry_step: usize,
}

impl<L, CV, BS> GenericBacktest<L, CV, BS>
where
    L: Labeler,
    CV: CrossValidator<Split = WalkForwardSplit>,
    BS: BetSizer,
{
    /// Create a new generic backtest with specified components.
    ///
    /// # Arguments
    ///
    /// * `labeler` - Event labeling strategy
    /// * `cv` - Cross-validation strategy
    /// * `sizer` - Bet sizing strategy
    /// * `entry_step` - Number of bars between potential entries
    pub fn new(labeler: L, cv: CV, sizer: BS, entry_step: usize) -> Self {
        Self {
            labeler,
            cv,
            sizer,
            entry_step,
        }
    }

    /// Run the backtest on the provided price and return data.
    ///
    /// # Arguments
    ///
    /// * `prices` - Price series (for labeling)
    /// * `returns` - Return series (for bet sizing)
    ///
    /// # Returns
    ///
    /// Vector of `BacktestResult` containing performance metrics for each fold.
    ///
    /// # Errors
    ///
    /// Returns error if labeling fails or data is insufficient.
    pub fn run(&self, prices: &[f64], returns: &[f64]) -> Result<Vec<BacktestResult>, BacktestError>
    where
        for<'a> &'a L::Event: Into<i8>,
    {
        if prices.len() != returns.len() {
            return Err(BacktestError::InvalidInput(
                "Prices and returns must have same length".to_string(),
            ));
        }

        let n_bars = prices.len();
        let splits = self.cv.splits(0, n_bars);

        let mut results = Vec::with_capacity(splits.len());

        for split in splits {
            // Generate entry indices from split train_indices
            let entry_indices: Vec<usize> = split
                .train_indices
                .iter()
                .copied()
                .step_by(self.entry_step)
                .collect();

            // Generate labels on training data
            let labels = self
                .labeler
                .label(prices, &entry_indices)
                .map_err(|_| BacktestError::InvalidInput("Labeling failed".to_string()))?;

            // Compute bet sizes from training returns
            let train_returns: Vec<f64> = split
                .train_indices
                .iter()
                .filter_map(|&i| returns.get(i).copied())
                .collect();
            let bet_size = self.sizer.size(&train_returns);

            // Simulate trades on test data
            let mut equity = vec![1.0];
            let mut n_trades = 0;

            for &test_idx in &split.test_indices {
                // Find corresponding label if this is a labeled entry
                if let Some(label_pos) = entry_indices.iter().position(|&e| e == test_idx)
                    && let Some(event) = labels.get(label_pos)
                {
                    let label_i8: i8 = event.into();
                    if label_i8 != 0 {
                        // Trade signal
                        let position = bet_size * label_i8 as f64;
                        let ret = returns.get(test_idx).copied().unwrap_or(0.0);
                        let pnl = position * ret;
                        let new_equity = equity.last().unwrap() * (1.0 + pnl);
                        equity.push(new_equity);
                        n_trades += 1;
                    }
                }
            }

            // Compute performance metrics
            let final_equity = *equity.last().unwrap();
            let total_return = final_equity - 1.0;

            let equity_returns: Vec<f64> = equity.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

            let sharpe = if !equity_returns.is_empty() {
                let mean_ret = equity_returns.iter().sum::<f64>() / equity_returns.len() as f64;
                let std_ret = {
                    let variance = equity_returns
                        .iter()
                        .map(|r| (r - mean_ret).powi(2))
                        .sum::<f64>()
                        / equity_returns.len() as f64;
                    variance.sqrt()
                };
                if std_ret > 1e-10 {
                    mean_ret / std_ret
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let max_drawdown = {
                let mut peak = 1.0;
                let mut max_dd = 0.0;
                for &e in &equity {
                    if e > peak {
                        peak = e;
                    }
                    let dd = (peak - e) / peak;
                    if dd > max_dd {
                        max_dd = dd;
                    }
                }
                max_dd
            };

            let train_start = split.train_indices.first().copied().unwrap_or(0);
            let train_end = split.train_indices.last().copied().unwrap_or(0) + 1;
            let test_start = split.test_indices.first().copied().unwrap_or(0);
            let test_end = split.test_indices.last().copied().unwrap_or(0) + 1;

            results.push(BacktestResult {
                total_return,
                sharpe,
                max_drawdown,
                n_trades,
                train_start,
                train_end,
                test_start,
                test_end,
            });
        }

        Ok(results)
    }
}

/// Builder for `GenericBacktest` with ergonomic API.
///
/// # Example
///
/// ```
/// use quant_backtest::{BacktestBuilder, FixedHorizonLabeler, WalkForward, KellyBetSizer};
/// use quant_backtest::WalkForwardConfig;
///
/// let backtest = BacktestBuilder::new()
///     .labeler(FixedHorizonLabeler {
///         horizon: 10,
///         min_return: 0.01,
///     })
///     .cv(WalkForward::new(WalkForwardConfig::rolling(100, 20, 20)))
///     .sizer(KellyBetSizer { fraction: 0.5 })
///     .entry_step(5)
///     .build();
/// ```
pub struct BacktestBuilder<L, CV, BS>
where
    L: Labeler,
    CV: CrossValidator<Split = WalkForwardSplit>,
    BS: BetSizer,
{
    labeler: Option<L>,
    cv: Option<CV>,
    sizer: Option<BS>,
    entry_step: usize,
}

impl<L, CV, BS> Default for BacktestBuilder<L, CV, BS>
where
    L: Labeler,
    CV: CrossValidator<Split = WalkForwardSplit>,
    BS: BetSizer,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<L, CV, BS> BacktestBuilder<L, CV, BS>
where
    L: Labeler,
    CV: CrossValidator<Split = WalkForwardSplit>,
    BS: BetSizer,
{
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            labeler: None,
            cv: None,
            sizer: None,
            entry_step: 1,
        }
    }

    /// Set the event labeling strategy.
    pub fn labeler(mut self, labeler: L) -> Self {
        self.labeler = Some(labeler);
        self
    }

    /// Set the cross-validation strategy.
    pub fn cv(mut self, cv: CV) -> Self {
        self.cv = Some(cv);
        self
    }

    /// Set the bet sizing strategy.
    pub fn sizer(mut self, sizer: BS) -> Self {
        self.sizer = Some(sizer);
        self
    }

    /// Set the number of bars between potential entries (default: 1).
    pub fn entry_step(mut self, step: usize) -> Self {
        self.entry_step = step;
        self
    }

    /// Build the `GenericBacktest` instance.
    ///
    /// # Panics
    ///
    /// Panics if any required component (labeler, cv, sizer) is missing.
    pub fn build(self) -> GenericBacktest<L, CV, BS> {
        GenericBacktest::new(
            self.labeler.expect("Labeler is required"),
            self.cv.expect("CrossValidator is required"),
            self.sizer.expect("BetSizer is required"),
            self.entry_step,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedHorizonLabeler, KellyBetSizer, WalkForward, WalkForwardConfig};

    #[test]
    fn test_generic_backtest_basic() {
        let labeler = FixedHorizonLabeler {
            horizon: 5,
            min_return: 0.005,
        };

        let cv = WalkForward::new(WalkForwardConfig::rolling(20, 10, 10));

        let sizer = KellyBetSizer { fraction: 0.5 };

        let backtest = GenericBacktest::new(labeler, cv, sizer, 1);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        let mut returns_with_init = vec![0.0];
        returns_with_init.extend(returns);

        let results = backtest.run(&prices, &returns_with_init).unwrap();

        assert!(!results.is_empty());
        for result in results {
            assert!(result.sharpe.is_finite());
            assert!(result.max_drawdown >= 0.0);
        }
    }

    #[test]
    fn test_backtest_builder() {
        let labeler = FixedHorizonLabeler {
            horizon: 5,
            min_return: 0.005,
        };

        let cv = WalkForward::new(WalkForwardConfig::rolling(20, 10, 10));

        let sizer = KellyBetSizer { fraction: 0.5 };

        let backtest = BacktestBuilder::new()
            .labeler(labeler)
            .cv(cv)
            .sizer(sizer)
            .entry_step(2)
            .build();

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]) - 1.0).collect();

        let mut returns_with_init = vec![0.0];
        returns_with_init.extend(returns);

        let results = backtest.run(&prices, &returns_with_init).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_mismatched_lengths() {
        let labeler = FixedHorizonLabeler {
            horizon: 5,
            min_return: 0.005,
        };

        let cv = WalkForward::new(WalkForwardConfig::rolling(20, 10, 10));

        let sizer = KellyBetSizer { fraction: 0.5 };

        let backtest = GenericBacktest::new(labeler, cv, sizer, 1);

        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let returns: Vec<f64> = (0..40).map(|i| (i as f64) * 0.01).collect();

        let result = backtest.run(&prices, &returns);
        assert!(result.is_err());
    }
}
