//! Project 1: Cross-Sectional Momentum Strategy
//!
//! Level: Beginner
//!
//! A long/short cross-sectional momentum strategy on 8 B3 stocks: rank
//! each stock by its past 20-day return, go long the top 2 and short the
//! bottom 2 with equal weights, and rebalance every 21 bars. The example
//! computes total return, annualised Sharpe, and maximum drawdown using
//! `quant-core` primitives plus inline equal-weight portfolio math.
//!
//! Run: `cargo run -p quant-lib --example projects-01_momentum`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::{rolling_mean, simple_returns, std_dev};
use quant_lib::prelude::*;

const TRADING_DAYS: f64 = 252.0;
const LOOKBACK: usize = 20;
const REBAL: usize = 21;
const N_LONG: usize = 2;
const N_SHORT: usize = 2;

fn main() {
    println!("=== Project 1: Cross-Sectional Momentum ===\n");

    let symbols = [
        "PETR4", "VALE3", "ITSA4", "BBDC4", "B3SA3", "ABEV3", "GGBR4", "WEGE3",
    ];
    let bars: Vec<Vec<common::OhlcvBar>> = symbols
        .iter()
        .map(|s| common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), s)))
        .collect();
    for (s, b) in symbols.iter().zip(bars.iter()) {
        println!("  {s}: {} bars", b.len());
    }

    // Align on common length.
    let n = bars.iter().map(|b| b.len()).min().unwrap();
    let closes: Vec<Vec<f64>> = bars
        .iter()
        .map(|b| b.iter().take(n).map(|x| x.close).collect())
        .collect();
    let n_assets = symbols.len();

    // Daily simple returns per asset (aligned, length n-1).
    let rets: Vec<Vec<f64>> = closes.iter().map(|c| simple_returns(c)).collect();
    let t_len = rets[0].len();

    let mut port_rets: Vec<f64> = Vec::with_capacity(t_len);

    // Walk through time; rebalance every REBAL bars starting at LOOKBACK.
    let mut t = LOOKBACK;
    while t < t_len {
        // 20-day momentum for each asset: (P_t - P_{t-20}) / P_{t-20}.
        let mut mom: Vec<(usize, f64)> = (0..n_assets)
            .map(|i| {
                let p_now = closes[i][t];
                let p_past = closes[i][t - LOOKBACK];
                (i, (p_now - p_past) / p_past)
            })
            .collect();
        // Sort descending by momentum.
        mom.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Long top N_LONG at +1/N_LONG, short bottom N_SHORT at -1/N_SHORT.
        let mut weights = vec![0.0_f64; n_assets];
        for entry in mom.iter().take(N_LONG) {
            let idx = entry.0;
            weights[idx] = 1.0 / N_LONG as f64;
        }
        for entry in mom.iter().rev().take(N_SHORT) {
            let idx = entry.0;
            weights[idx] = -1.0 / N_SHORT as f64;
        }

        // Hold the position for REBAL bars.
        let end = (t + REBAL).min(t_len);
        for dt in t..end {
            let pr: f64 = (0..n_assets).map(|i| weights[i] * rets[i][dt]).sum();
            port_rets.push(pr);
        }
        t += REBAL;
    }

    // Headline metrics.
    let total_return = port_rets.iter().fold(1.0_f64, |acc, r| acc * (1.0 + r)) - 1.0;
    let ann_sharpe = annualised_sharpe(&port_rets);
    let mdd = max_drawdown(&port_rets);

    // Rolling 63-day mean of portfolio returns (quarterly smoothed view).
    let smooth = rolling_mean(63, &port_rets).unwrap_or_default();
    let last_smooth = smooth.last().copied().unwrap_or(0.0);

    println!("\nMomentum strategy (lookback={LOOKBACK}, rebalance={REBAL}):");
    println!("  Bars traded:           {}", port_rets.len());
    let tr_pct = total_return * 100.0;
    println!("  Total return:          {total_return:.4} ({tr_pct:.2}%)");
    println!("  Annualised Sharpe:     {ann_sharpe:.4}");
    let mdd_pct = mdd * 100.0;
    println!("  Max drawdown:          {mdd:.4} ({mdd_pct:.2}%)");
    println!("  Last 63-day mean ret:  {last_smooth:.6}");
}

fn annualised_sharpe(returns: &[f64]) -> f64 {
    let m = mean(returns);
    let sd = std_dev(returns).unwrap_or(0.0);
    if sd == 0.0 {
        return 0.0;
    }
    (m / sd) * TRADING_DAYS.sqrt()
}

fn max_drawdown(returns: &[f64]) -> f64 {
    let mut wealth = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut mdd = 0.0_f64;
    for &r in returns {
        wealth *= 1.0 + r;
        peak = peak.max(wealth);
        let dd = (peak - wealth) / peak;
        mdd = mdd.max(dd);
    }
    mdd
}
