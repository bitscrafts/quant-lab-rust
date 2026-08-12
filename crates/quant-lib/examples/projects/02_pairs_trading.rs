//! Project 2: Cointegration Pairs Trading
//!
//! Level: Intermediate
//!
//! Tests whether PETR4 and VALE3 are cointegrated by OLS-regressing
//! PETR4 on VALE3, running an ADF test on the residuals, and trading
//! the spread z-score. When the z-score exceeds +2 the spread is
//! "stretched" so we short PETR4 / long VALE3; below -2 we do the
//! opposite; we exit when the z-score reverts inside +/-0.5. The
//! example reports trade count, hit rate, total P&L, and Sharpe.
//!
//! Run: `cargo run -p quant-lib --example projects-02_pairs_trading`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::{rolling_mean, rolling_std_dev, std_dev};
use quant_lib::prelude::*;

const TRADING_DAYS: f64 = 252.0;
const ZSCORE_WINDOW: usize = 60;
const ENTRY: f64 = 2.0;
const EXIT: f64 = 0.5;

fn main() {
    println!("=== Project 2: Cointegration Pairs Trading ===\n");

    let petr = common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4"));
    let vale = common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "VALE3"));
    let n = petr.len().min(vale.len());
    let p: Vec<f64> = petr.iter().take(n).map(|b| b.close).collect();
    let v: Vec<f64> = vale.iter().take(n).map(|b| b.close).collect();
    println!("PETR4: {n} bars, VALE3: {n} bars (aligned)");

    // --- Step 1: OLS regression PETR4 = alpha + beta * VALE3 + eps. ---
    let x: Vec<Vec<f64>> = v.iter().map(|vi| vec![1.0, *vi]).collect();
    let fit = ols(&x, &p).expect("OLS fit");
    let alpha = fit.coeffs[0];
    let beta = fit.coeffs[1];
    let residuals = &fit.residuals;
    println!(
        "\nOLS: PETR4 = {alpha:.4} + {beta:.4} * VALE3  (R^2 = {:.4})",
        fit.r_squared
    );

    // --- Step 2: ADF test on residuals. ---
    let adf = adf_test(residuals, 1).expect("ADF test");
    let stat = adf.statistic;
    let crit = adf.critical_value;
    let stationary = adf.is_stationary;
    println!("ADF: statistic = {stat:.4}, 5% critical = {crit}, stationary = {stationary}");

    // --- Step 3: Spread z-score (rolling 60-day mean/std of residuals). ---
    let rmean = rolling_mean(ZSCORE_WINDOW, residuals).expect("rolling mean");
    let rstd = rolling_std_dev(ZSCORE_WINDOW, residuals).expect("rolling std");
    // z[t] = (residuals[offset + t] - rmean[t]) / rstd[t], offset = window - 1.
    let offset = ZSCORE_WINDOW - 1;
    let z: Vec<f64> = (0..rmean.len())
        .map(|t| {
            let r = residuals[offset + t];
            let s = rstd[t];
            if s > 1e-12 { (r - rmean[t]) / s } else { 0.0 }
        })
        .collect();

    // --- Step 4: Trade the z-score. ---
    // Position state: 0 = flat, +1 = long spread (long PETR4, short VALE3),
    // -1 = short spread (short PETR4, long VALE3).
    let mut position = 0_i32;
    let mut trades = 0_usize;
    let mut wins = 0_usize;
    let mut entry_pnl = 0.0_f64;
    let mut daily_pnl: Vec<f64> = Vec::with_capacity(z.len());
    for (t, &zt) in z.iter().enumerate() {
        let idx = offset + t; // index into the original aligned series
        // Spread daily return = d(PETR4) - beta * d(VALE3) (the hedge ratio).
        let dpetr = if idx + 1 < p.len() {
            p[idx + 1] - p[idx]
        } else {
            0.0
        };
        let dvale = if idx + 1 < v.len() {
            v[idx + 1] - v[idx]
        } else {
            0.0
        };
        let spread_ret = dpetr - beta * dvale;
        // Mark-to-market P&L of current position (per unit of spread).
        if t > 0 {
            let pnl = position as f64 * spread_ret;
            daily_pnl.push(pnl);
        }
        // Decide on the next position.
        let new_pos = if position == 0 {
            if zt > ENTRY {
                -1
            } else if zt < -ENTRY {
                1
            } else {
                0
            }
        } else if position.abs() == 1 {
            if zt.abs() < EXIT {
                0 // exit: book win/loss
            } else {
                position // hold
            }
        } else {
            0
        };
        if new_pos != position {
            if position != 0 {
                // Closing a trade.
                trades += 1;
                let cumulative: f64 = daily_pnl.iter().copied().sum();
                let trade_pnl = cumulative - entry_pnl;
                if trade_pnl > 0.0 {
                    wins += 1;
                }
            }
            if new_pos != 0 {
                // Opening a trade.
                entry_pnl = if t > 0 {
                    daily_pnl.iter().copied().sum()
                } else {
                    0.0
                };
            }
            position = new_pos;
        }
    }
    // Close any open trade at the end.
    if position != 0 {
        trades += 1;
        let cumulative: f64 = daily_pnl.iter().copied().sum();
        let trade_pnl = cumulative - entry_pnl;
        if trade_pnl > 0.0 {
            wins += 1;
        }
    }

    let total_pnl: f64 = daily_pnl.iter().sum();
    let hit_rate = if trades > 0 {
        wins as f64 / trades as f64
    } else {
        0.0
    };
    let sharpe = annualised_sharpe(&daily_pnl);

    println!("\nPairs trading results (z-window={ZSCORE_WINDOW}, entry={ENTRY}, exit={EXIT}):");
    println!("  Trades:                {trades}");
    let hit_pct = hit_rate * 100.0;
    println!("  Hit rate:              {hit_rate:.4} ({hit_pct:.1}%)");
    println!("  Total P&L (spread):    {total_pnl:.4}");
    println!("  Annualised Sharpe:     {sharpe:.4}");
}

fn annualised_sharpe(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let m = mean(returns);
    let sd = std_dev(returns).unwrap_or(0.0);
    if sd == 0.0 {
        return 0.0;
    }
    (m / sd) * TRADING_DAYS.sqrt()
}
