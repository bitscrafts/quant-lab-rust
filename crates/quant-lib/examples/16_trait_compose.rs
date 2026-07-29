//! Example 16: Composable Backtest - BacktestBuilder with Pluggable Components
//!
//! Level: Advanced
//!
//! Showcases the Phase 14.5 composable backtest architecture:
//! `GenericBacktest<L, CV, BS>` is generic over three pluggable
//! components (Labeler, CrossValidator, BetSizer), wired together
//! with the `BacktestBuilder` fluent API.
//!
//! This example runs the same strategy three times with different
//! bet sizers (Kelly full, Kelly half, fixed 10%) to show how the
//! builder pattern lets you swap a single component without touching
//! the rest of the pipeline.
//!
//! Uses `quant-backtest` (BacktestBuilder, GenericBacktest,
//! KellyBetSizer, FixedBetSizer, EqualBetSizer, FixedHorizonLabeler,
//! WalkForward, WalkForwardConfig) and the `Labeler`, `CrossValidator`,
//! `BetSizer` traits from `quant-core`.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 16_trait_compose
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::backtest::{EqualBetSizer, FixedBetSizer};
use quant_lib::prelude::*;

fn build_and_run(
    name: &str,
    labeler: FixedHorizonLabeler,
    cv: WalkForward,
    sizer: impl BetSizer,
    entry_step: usize,
    prices: &[f64],
    returns: &[f64],
) {
    let bt = BacktestBuilder::new()
        .labeler(labeler)
        .cv(cv)
        .sizer(sizer)
        .entry_step(entry_step)
        .build();
    let results = bt.run(prices, returns).expect("backtest");
    let n = results.len();
    let avg_sharpe: f64 = if n > 0 {
        results.iter().map(|r| r.sharpe).sum::<f64>() / n as f64
    } else {
        0.0
    };
    let avg_ret: f64 = if n > 0 {
        results.iter().map(|r| r.total_return).sum::<f64>() / n as f64
    } else {
        0.0
    };
    let total_trades: usize = results.iter().map(|r| r.n_trades).sum();
    println!(
        "\n[{name}] folds={n}, avg_sharpe={avg_sharpe:+.4}, avg_return={avg_ret:+.4}, trades={total_trades}"
    );
    for (i, r) in results.iter().enumerate() {
        println!(
            "   fold {i}: ret={:+.4}, sharpe={:+.4}, dd={:+.4}, trades={}",
            r.total_return, r.sharpe, r.max_drawdown, r.n_trades
        );
    }
}

fn main() {
    println!("=== Example 16: Composable Backtest (BacktestBuilder) ===");

    // Load PETR4.
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let mut returns = vec![0.0_f64; closes.len()];
    for i in 1..closes.len() {
        returns[i] = (closes[i] - closes[i - 1]) / closes[i - 1];
    }
    println!("PETR4: {} bars", bars.len());

    // Shared labeler and CV: fixed-horizon 5-bar, walk-forward 80/20/20.
    let labeler = || FixedHorizonLabeler::new(5, 0.005);
    let cv = || WalkForward::new(WalkForwardConfig::rolling(80, 20, 20));

    // 1. Full Kelly.
    build_and_run(
        "Full Kelly",
        labeler(),
        cv(),
        KellyBetSizer::new(1.0),
        5,
        &closes,
        &returns,
    );

    // 2. Half Kelly (more conservative).
    build_and_run(
        "Half Kelly",
        labeler(),
        cv(),
        KellyBetSizer::new(0.5),
        5,
        &closes,
        &returns,
    );

    // 3. Fixed 10% bet.
    build_and_run(
        "Fixed 10%",
        labeler(),
        cv(),
        FixedBetSizer::new(0.10),
        5,
        &closes,
        &returns,
    );

    // 4. Equal (1 position = 100%) bet.
    build_and_run(
        "Equal 1.0",
        labeler(),
        cv(),
        EqualBetSizer::new(1),
        5,
        &closes,
        &returns,
    );

    // 5. Two-fund separation: change the CV to anchored (expanding) while
    // keeping the same labeler and sizer. The builder pattern makes this
    // a one-line change.
    let cv_anchored = WalkForward::new(WalkForwardConfig::anchored(80, 20, 20));
    build_and_run(
        "Anchored CV + Half Kelly",
        labeler(),
        cv_anchored,
        KellyBetSizer::new(0.5),
        5,
        &closes,
        &returns,
    );

    println!("\nNote: GenericBacktest<L, CV, BS> is fully generic over the");
    println!("Labeler, CrossValidator, and BetSizer traits. To swap any");
    println!("component (e.g. FixedHorizon vs DynamicBarrier labeler),");
    println!("change the type parameter --- no other code needs to change.");
    println!("This is the Phase 14.5 composable architecture in action.");
}
