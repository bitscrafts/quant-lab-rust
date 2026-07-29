//! AFML-style backtest engine (AFML Ch. 3-10).
//!
//! Combines triple-barrier labeling, sample weights, purged k-fold
//! cross-validation, and Kelly bet sizing into a single pipeline that
//! turns a price series into a risk-aware backtest report.

use crate::error::BacktestError;
use crate::kelly::{PositionSize, compute_position_size};
use crate::purged_kfold::{PurgedKFoldConfig, PurgedSplit, purged_kfold_splits};
use crate::triple_barrier::{LabeledEvent, TripleBarrierConfig, triple_barrier_label};
use crate::weights::sample_weights;

/// Bet sizing method.
#[derive(Debug, Clone, Copy)]
pub enum BetSizing {
    /// Equal position sizes (1 unit per trade).
    Equal,
    /// Full Kelly criterion.
    KellyFull,
    /// Half Kelly (more conservative).
    KellyHalf,
    /// Fixed fraction.
    Fixed(f64),
}

/// AFML backtest configuration.
#[derive(Debug, Clone)]
pub struct AfmlBacktestConfig {
    pub barrier_config: TripleBarrierConfig,
    pub cv_config: PurgedKFoldConfig,
    pub use_weights: bool,
    pub bet_sizing: BetSizing,
    /// Entry every `entry_step` bars.
    pub entry_step: usize,
}

/// AFML backtest result.
#[derive(Debug, Clone)]
pub struct AfmlBacktestResult {
    pub events: Vec<LabeledEvent>,
    pub weights: Vec<f64>,
    pub cv_splits: Vec<PurgedSplit>,
    pub in_sample_sharpe: f64,
    pub out_of_sample_sharpe: f64,
    pub position_size: PositionSize,
    pub total_return: f64,
    pub max_drawdown: f64,
}

/// Run the AFML backtest pipeline on price data.
pub fn afml_backtest(
    prices: &[f64],
    config: &AfmlBacktestConfig,
) -> Result<AfmlBacktestResult, BacktestError> {
    config.barrier_config.validate()?;
    if prices.len() < config.barrier_config.time_barrier + 1 {
        return Err(BacktestError::InsufficientData(format!(
            "need at least {} prices, got {}",
            config.barrier_config.time_barrier + 1,
            prices.len()
        )));
    }
    if config.entry_step == 0 {
        return Err(BacktestError::InvalidConfig(
            "entry_step must be positive".to_string(),
        ));
    }

    // Generate entry indices at fixed step.
    let entry_indices: Vec<usize> = (0..prices.len())
        .step_by(config.entry_step)
        .take_while(|&i| {
            i + config.barrier_config.time_barrier < prices.len() || i < prices.len() - 1
        })
        .collect();

    let events = triple_barrier_label(prices, &entry_indices, &config.barrier_config)?;
    let weights = if config.use_weights {
        sample_weights(&events, prices.len())
    } else {
        vec![1.0; events.len()]
    };

    let cv_splits = purged_kfold_splits(&events, prices.len(), &config.cv_config);

    // Trade returns per event (signed).
    let trade_returns: Vec<f64> = events.iter().map(|e| e.return_pct).collect();

    // Per-trade sizing.
    let position_size = compute_position_size(&trade_returns);
    let size_per_trade: Vec<f64> = match config.bet_sizing {
        BetSizing::Equal => trade_returns.iter().map(|_| 1.0).collect(),
        BetSizing::KellyFull => trade_returns
            .iter()
            .map(|_| position_size.kelly_full.max(0.0))
            .collect(),
        BetSizing::KellyHalf => trade_returns
            .iter()
            .map(|_| position_size.kelly_half.max(0.0))
            .collect(),
        BetSizing::Fixed(f) => trade_returns.iter().map(|_| f.max(0.0)).collect(),
    };

    // Equity curve from compounded trade returns, weighted by size.
    let mut equity = 1.0;
    let mut equity_curve = Vec::with_capacity(events.len() + 1);
    equity_curve.push(equity);
    for (i, r) in trade_returns.iter().enumerate() {
        let s = size_per_trade.get(i).copied().unwrap_or(1.0);
        equity *= 1.0 + s * r;
        equity_curve.push(equity);
    }
    let total_return = equity - 1.0;
    let max_drawdown = max_drawdown(&equity_curve);

    // In-sample vs out-of-sample Sharpe (per-fold, OOS averaged).
    let in_sample_sharpe = sharpe(&trade_returns);
    let mut oos_sharpes = Vec::new();
    for split in &cv_splits {
        let oos_returns: Vec<f64> = split
            .test_indices
            .iter()
            .filter_map(|&i| trade_returns.get(i).copied())
            .collect();
        if !oos_returns.is_empty() {
            oos_sharpes.push(sharpe(&oos_returns));
        }
    }
    let out_of_sample_sharpe = if oos_sharpes.is_empty() {
        0.0
    } else {
        oos_sharpes.iter().sum::<f64>() / oos_sharpes.len() as f64
    };

    Ok(AfmlBacktestResult {
        events,
        weights,
        cv_splits,
        in_sample_sharpe,
        out_of_sample_sharpe,
        position_size,
        total_return,
        max_drawdown,
    })
}

/// Annualised Sharpe ratio from a series of per-trade returns
/// (simplified, risk-free rate assumed 0).
fn sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std == 0.0 {
        return 0.0;
    }
    // Multiply by sqrt(252) for daily annualisation; here we just
    // return the per-trade Sharpe (caller may annualise).
    mean / std
}

/// Maximum drawdown of an equity curve (returns the most negative
/// peak-to-trough percentage).
fn max_drawdown(equity: &[f64]) -> f64 {
    if equity.is_empty() {
        return 0.0;
    }
    let mut peak = equity[0];
    let mut max_dd = 0.0;
    for &v in equity {
        if v > peak {
            peak = v;
        }
        if peak > 0.0 {
            let dd = (v - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }
    }
    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AfmlBacktestConfig {
        AfmlBacktestConfig {
            barrier_config: TripleBarrierConfig {
                upper_barrier: 0.02,
                lower_barrier: -0.02,
                time_barrier: 5,
                min_return: 0.0,
            },
            cv_config: PurgedKFoldConfig {
                n_folds: 3,
                embargo: 2,
            },
            use_weights: true,
            bet_sizing: BetSizing::Equal,
            entry_step: 3,
        }
    }

    #[test]
    fn test_afml_backtest_smoke() {
        // 100-bar synthetic price series.
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.01) + (i as f64 % 7.0 - 3.0) * 0.05)
            .collect();
        let result = afml_backtest(&prices, &cfg()).unwrap();
        assert!(!result.events.is_empty());
        assert_eq!(result.events.len(), result.weights.len());
        assert_eq!(result.cv_splits.len(), 3);
        assert!(result.in_sample_sharpe.is_finite());
        assert!(result.out_of_sample_sharpe.is_finite());
        assert!(result.total_return.is_finite());
        assert!(result.max_drawdown <= 0.0 || result.max_drawdown.is_finite());
    }

    #[test]
    fn test_insufficient_data() {
        let prices = vec![100.0, 101.0];
        let err = afml_backtest(&prices, &cfg()).unwrap_err();
        assert!(matches!(err, BacktestError::InsufficientData(_)));
    }
}
