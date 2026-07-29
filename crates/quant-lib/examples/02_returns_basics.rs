//! Example 02: Simple and Log Returns, Sharpe, Drawdown
//!
//! Level: Simple
//!
//! Builds on Example 01 by computing simple (arithmetic) and log
//! (continuously compounded) returns from the bundled `stock_prices.csv`
//! close column, then derives the annualised Sharpe ratio and the
//! maximum drawdown. Uses `simple_returns`, `log_returns`, `mean`,
//! `std_dev` from `quant-core` re-exported through `quant_lib::core`.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 02_returns_basics
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::core::{log_returns, simple_returns};
use quant_lib::prelude::*;

/// Annualisation factor for daily returns (252 trading days).
const TRADING_DAYS: f64 = 252.0;

/// Risk-free rate (annualised, continuously compounded).
const RF: f64 = 0.02;

fn main() {
    let csv = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&csv);
    let closes = common::closes(&bars);

    println!("=== Example 02: Returns, Sharpe, Drawdown ===");
    println!("Loaded {} bars from {}", bars.len(), csv.display());

    // Simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}.
    let sr = simple_returns(&closes);
    // Log returns: r_t = ln(P_t / P_{t-1}).
    let lr = log_returns(&closes);

    println!(
        "Simple returns: n={}, first={:.6}, last={:.6}",
        sr.len(),
        sr[0],
        sr[sr.len() - 1]
    );
    println!(
        "Log returns:     n={}, first={:.6}, last={:.6}",
        lr.len(),
        lr[0],
        lr[lr.len() - 1]
    );

    // Annualised mean and volatility (daily -> annual).
    let mu_daily = mean(&sr);
    let sd_daily = std_dev(&sr).unwrap();
    let mu_annual = mu_daily * TRADING_DAYS;
    let sd_annual = sd_daily * TRADING_DAYS.sqrt();
    let sharpe = (mu_annual - RF) / sd_annual;

    println!(
        "Annualised mean    = {mu_annual:.4} ({:.2}%)",
        mu_annual * 100.0
    );
    println!(
        "Annualised vol      = {sd_annual:.4} ({:.2}%)",
        sd_annual * 100.0
    );
    println!("Annualised Sharpe   = {sharpe:.4}");

    // Maximum drawdown from the price series (peak-to-trough).
    let max_dd = max_drawdown(&closes);
    println!("Max drawdown        = {max_dd:.4} ({:.2}%)", max_dd * 100.0);

    // Cross-check: sum of log returns equals ln(P_T / P_0).
    let total_log: f64 = lr.iter().sum();
    let expected_log = (closes[closes.len() - 1] / closes[0]).ln();
    assert!(
        (total_log - expected_log).abs() < 1e-9,
        "log-return sum mismatch"
    );
    println!(
        "Cross-check: sum(log r) = {total_log:.6} vs ln(P_T/P_0) = {expected_log:.6} (matches)"
    );
}

/// Maximum drawdown of a price series: the most negative peak-to-trough
/// percentage drop. Returns a non-positive number (e.g. -0.15 for a 15%
/// drawdown).
fn max_drawdown(prices: &[f64]) -> f64 {
    let mut peak = prices[0];
    let mut max_dd = 0.0;
    for &p in prices {
        if p > peak {
            peak = p;
        }
        let dd = (p - peak) / peak;
        if dd < max_dd {
            max_dd = dd;
        }
    }
    max_dd
}
