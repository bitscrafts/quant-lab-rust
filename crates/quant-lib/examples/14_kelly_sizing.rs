//! Example 14: Kelly Criterion Bet Sizing (AFML Ch.10)
//!
//! Level: Advanced
//!
//! The Kelly criterion maximises the long-run growth rate of a bankroll
//! when betting a fixed fraction of wealth on each trial. This example:
//!
//! - Computes the full Kelly fraction from win probability and win/loss ratio
//! - Compares full vs half (fractional) Kelly
//! - Estimates Kelly from a synthetic trade-return series
//! - Derives the position size via `compute_position_size`
//! - Demonstrates the `KellyBetSizer` trait implementing `BetSizer`
//!
//! Uses `quant-backtest` (kelly_fraction, fractional_kelly,
//! kelly_from_returns, compute_position_size, KellyBetSizer).
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 14_kelly_sizing
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::backtest::{compute_position_size, fractional_kelly, kelly_from_returns};
use quant_lib::prelude::*;

fn main() {
    println!("=== Example 14: Kelly Criterion Bet Sizing ===");

    // 1. Full Kelly: f* = p - q/b where q = 1 - p, b = win/loss ratio.
    let p = 0.55; // 55% win rate
    let b = 1.5; // win 1.5x the average loss
    let f_full = kelly_fraction(p, b);
    let f_half = fractional_kelly(p, b, 0.5);
    let f_quarter = fractional_kelly(p, b, 0.25);
    println!("Win prob p = {p}, win/loss ratio b = {b}");
    println!("  full Kelly     = {f_full:.4} ({:.2}%)", f_full * 100.0);
    println!("  half Kelly     = {f_half:.4} ({:.2}%)", f_half * 100.0);
    println!(
        "  quarter Kelly  = {f_quarter:.4} ({:.2}%)",
        f_quarter * 100.0
    );

    // 2. Kelly from a trade-return series.
    let trades = vec![
        0.02, 0.03, -0.015, 0.01, 0.025, -0.02, 0.015, -0.01, 0.02, -0.015,
    ];
    let f_from_trades = kelly_from_returns(&trades);
    let ps = compute_position_size(&trades);
    println!("\nFrom {} synthetic trades:", trades.len());
    println!(
        "  win probability = {:.4} ({:.1}%)",
        ps.win_probability,
        ps.win_probability * 100.0
    );
    println!("  win/loss ratio   = {:.4}", ps.win_loss_ratio);
    println!("  full Kelly        = {:.4}", ps.kelly_full);
    println!("  half Kelly        = {:.4}", ps.kelly_half);
    println!("  kelly_from_returns= {f_from_trades:.4} (matches full)");

    // 3. KellyBetSizer implements BetSizer: derives bet size from returns.
    let sizer_full = KellyBetSizer::new(1.0); // full Kelly
    let sizer_half = KellyBetSizer::new(0.5); // half Kelly
    let size_full = sizer_full.size(&trades);
    let size_half = sizer_half.size(&trades);
    println!("\nKellyBetSizer (trait):");
    println!("  full sizer.size = {size_full:.4}");
    println!("  half sizer.size = {size_half:.4}");
    assert!((size_full - ps.kelly_full.clamp(0.0, 1.0)).abs() < 1e-9);
    assert!((size_half - 0.5 * ps.kelly_full).abs() < 1e-9);
    println!("  matches compute_position_size output (clamped to [0, 1])");

    // 4. Breakeven: p = 0.5, b = 1.0 -> f* = 0 (no edge).
    let f_zero = kelly_fraction(0.5, 1.0);
    assert!(f_zero.abs() < 1e-9, "Kelly should be 0 at fair odds");
    println!("\nFair-odds check: p=0.5, b=1.0 -> Kelly = {f_zero:.6} (zero edge)");

    // 5. Negative edge: p = 0.4, b = 1.0 -> f* < 0 (don't bet).
    let f_neg = kelly_fraction(0.4, 1.0);
    println!("Negative edge:   p=0.4, b=1.0 -> Kelly = {f_neg:.4} (don't bet)");

    // 6. Cross-check: a longer trade series from PETR4 triple-barrier labels.
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let entries: Vec<usize> = (0..closes.len()).step_by(10).collect();
    let tb_config = TripleBarrierConfig {
        upper_barrier: 0.03,
        lower_barrier: -0.02,
        time_barrier: 10,
        min_return: 0.0,
    };
    let events = triple_barrier_label(&closes, &entries, &tb_config).expect("label");
    let trade_returns: Vec<f64> = events.iter().map(|e| e.return_pct).collect();
    let ps_real = compute_position_size(&trade_returns);
    println!("\nPETR4 triple-barrier trades ({} events):", events.len());
    println!(
        "  win prob = {:.4}, b = {:.4}, Kelly = {:.4}, half = {:.4}",
        ps_real.win_probability, ps_real.win_loss_ratio, ps_real.kelly_full, ps_real.kelly_half
    );
}
