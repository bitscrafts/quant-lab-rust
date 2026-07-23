//! Backtest engine: configuration, trade record, result, and `run_backtest`.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::BacktestError;
use crate::signal::{Position, Signal};
use crate::strategy::Strategy;
use qf_03_stocks::Ohlcv;

/// Configuration for a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital in dollars.
    pub initial_capital: f64,
    /// Transaction cost per trade as a fraction of the traded notional
    /// (e.g. `0.001` = 0.1%).
    pub transaction_cost: f64,
    /// Allow short selling. Phase 5 is long-only: set to `false`.
    pub allow_short: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            transaction_cost: 0.001,
            allow_short: false,
        }
    }
}

impl BacktestConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - `initial_capital` must be positive.
    /// - `transaction_cost` must be non-negative and below 1.
    /// - Short selling is rejected in Phase 5 (long-only).
    pub fn validate(&self) -> Result<(), BacktestError> {
        if self.initial_capital <= 0.0 {
            return Err(BacktestError::InvalidConfig(
                "initial_capital must be positive".to_string(),
            ));
        }
        if self.transaction_cost < 0.0 || self.transaction_cost >= 1.0 {
            return Err(BacktestError::InvalidConfig(
                "transaction_cost must be in [0, 1)".to_string(),
            ));
        }
        if self.allow_short {
            return Err(BacktestError::InvalidConfig(
                "short selling is not supported in Phase 5".to_string(),
            ));
        }
        Ok(())
    }
}

/// Record of a single completed round-trip trade (entry then exit).
#[derive(Debug, Clone, PartialEq)]
pub struct Trade {
    /// Entry date (bar date string).
    pub entry_date: String,
    /// Entry price.
    pub entry_price: f64,
    /// Exit date (bar date string).
    pub exit_date: String,
    /// Exit price.
    pub exit_price: f64,
    /// Number of shares held during the trade.
    pub shares: f64,
    /// Dollar profit or loss (exit value minus entry value, before costs).
    pub pnl: f64,
    /// Return of the trade as a fraction (e.g. `0.10` for +10%).
    pub return_pct: f64,
}

/// Complete results from a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Strategy name.
    pub strategy_name: String,
    /// Starting capital.
    pub initial_capital: f64,
    /// Ending capital (after closing any open position at the last close).
    pub final_capital: f64,
    /// Total return as a fraction (e.g. `0.15` for +15%).
    pub total_return: f64,
    /// Number of round-trip trades executed.
    pub num_trades: usize,
    /// Winning trades / total trades. `0.0` when `num_trades == 0`.
    pub win_rate: f64,
    /// Total transaction costs paid in dollars.
    pub total_costs: f64,
    /// Per-bar simple returns of the equity curve (length `data.len() - 1`).
    pub daily_returns: Vec<f64>,
    /// Equity curve: capital marked-to-market at each bar (length `data.len()`).
    pub equity_curve: Vec<f64>,
    /// Round-trip trade log.
    pub trades: Vec<Trade>,
}

impl BacktestResult {
    /// Annualised Sharpe ratio (assumes daily data, 252 periods/year).
    pub fn sharpe(&self, risk_free_rate: f64) -> f64 {
        qf_04_returns::annualized_sharpe(&self.daily_returns, risk_free_rate, 252.0)
    }

    /// Per-period Sortino ratio.
    pub fn sortino(&self, risk_free_rate: f64) -> f64 {
        qf_04_returns::sortino_ratio(&self.daily_returns, risk_free_rate)
    }

    /// Maximum drawdown of the equity curve (most negative value, <= 0).
    pub fn max_drawdown(&self) -> f64 {
        qf_04_returns::max_drawdown(&self.equity_curve)
    }

    /// Annualised volatility (assumes daily data, 252 periods/year).
    pub fn volatility(&self) -> f64 {
        qf_04_returns::annualized_volatility(&self.daily_returns, 252.0)
    }
}

/// Run a complete backtest simulation.
///
/// # Algorithm
///
/// 1. Validate the config; initialise `position = Flat`, `capital = initial_capital`.
/// 2. For each bar `i` in `data`:
///    - Generate `signal = strategy.signal(data, i)`.
///    - Translate the signal into a position transition, applying transaction
///      costs on every entry and exit.
///    - Mark the equity to market using the current close and position.
///    - Append to the equity curve and record the per-bar return.
/// 3. Close any open position at the last close (with transaction cost).
/// 4. Compute aggregate statistics (`total_return`, `win_rate`, etc.).
///
/// # Long-only
/// `Sell` from a long position transitions to `Flat`. `Buy` from `Flat`
/// transitions to `Long`. `Buy` while already long is a no-op (the engine does
/// not pyramid), and `Sell` while flat is a no-op.
///
/// # Arguments
/// * `strategy` - Any type implementing [`Strategy`].
/// * `data` - OHLCV bars ordered chronologically (oldest first).
/// * `config` - Backtest configuration. Validated before running.
///
/// # Returns
/// A populated [`BacktestResult`]. An empty `data` slice yields a result with
/// zeroed statistics and empty vectors.
///
/// # Errors
/// Returns [`BacktestError::InvalidConfig`] when `config.validate()` fails.
/// Returns [`BacktestError::InsufficientData`] when `data` is empty.
pub fn run_backtest<S: Strategy>(
    strategy: &S,
    data: &[Ohlcv],
    config: &BacktestConfig,
) -> Result<BacktestResult, BacktestError> {
    config.validate()?;
    if data.is_empty() {
        return Err(BacktestError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let mut capital = config.initial_capital;
    let mut position = Position::Flat;
    let mut shares: f64 = 0.0;
    let mut entry_price: f64 = 0.0;
    let mut entry_date: String = String::new();
    let mut total_costs = 0.0_f64;
    let mut trades: Vec<Trade> = Vec::new();

    let mut equity_curve: Vec<f64> = Vec::with_capacity(data.len());

    for (i, bar) in data.iter().enumerate() {
        let signal = strategy.signal(data, i);

        // Translate signal into a position transition.
        match (position, signal) {
            (Position::Flat, Signal::Buy) => {
                // Enter long: spend all capital on shares at the close.
                let price = bar.close;
                let notional = capital;
                let cost = notional * config.transaction_cost;
                shares = (notional - cost) / price;
                total_costs += cost;
                entry_price = price;
                entry_date = bar.date.clone();
                position = Position::Long;
            }
            (Position::Long, Signal::Sell) => {
                // Exit long: sell all shares at the close.
                let price = bar.close;
                let notional = shares * price;
                let cost = notional * config.transaction_cost;
                total_costs += cost;
                let gross_pnl = (price - entry_price) * shares;
                let trade_return = if entry_price > 0.0 {
                    (price - entry_price) / entry_price
                } else {
                    0.0
                };
                capital = notional - cost;
                trades.push(Trade {
                    entry_date: entry_date.clone(),
                    entry_price,
                    exit_date: bar.date.clone(),
                    exit_price: price,
                    shares,
                    pnl: gross_pnl,
                    return_pct: trade_return,
                });
                position = Position::Flat;
                shares = 0.0;
            }
            _ => {
                // No state change: Hold in any position, Buy while long,
                // Sell while flat.
            }
        }

        // Mark-to-market equity: cash (if flat) or shares * close (if long).
        let equity = match position {
            Position::Long => shares * bar.close,
            Position::Flat => capital,
            Position::Short => capital, // unreachable in long-only mode
        };
        equity_curve.push(equity);
    }

    // Close any open position at the last close.
    if position == Position::Long {
        let last = data.last().unwrap();
        let price = last.close;
        let notional = shares * price;
        let cost = notional * config.transaction_cost;
        total_costs += cost;
        let gross_pnl = (price - entry_price) * shares;
        let trade_return = if entry_price > 0.0 {
            (price - entry_price) / entry_price
        } else {
            0.0
        };
        capital = notional - cost;
        trades.push(Trade {
            entry_date: entry_date.clone(),
            entry_price,
            exit_date: last.date.clone(),
            exit_price: price,
            shares,
            pnl: gross_pnl,
            return_pct: trade_return,
        });
        position = Position::Flat;
    }

    // Equity curve was built mark-to-market; recompute the final capital for
    // consistency with the closed trade. The last equity curve entry should
    // equal `capital` when flat, but if we never had a position, capital is
    // still `initial_capital`.
    let final_capital = if position.is_flat() {
        capital
    } else {
        *equity_curve.last().unwrap_or(&capital)
    };

    let total_return = if config.initial_capital > 0.0 {
        (final_capital - config.initial_capital) / config.initial_capital
    } else {
        0.0
    };

    let num_trades = trades.len();
    let win_rate = if num_trades == 0 {
        0.0
    } else {
        trades.iter().filter(|t| t.pnl > 0.0).count() as f64 / num_trades as f64
    };

    // Per-bar simple returns of the equity curve.
    let daily_returns = qf_04_returns::simple_returns(&equity_curve);

    Ok(BacktestResult {
        strategy_name: strategy.name().to_string(),
        initial_capital: config.initial_capital,
        final_capital,
        total_return,
        num_trades,
        win_rate,
        total_costs,
        daily_returns,
        equity_curve,
        trades,
    })
}