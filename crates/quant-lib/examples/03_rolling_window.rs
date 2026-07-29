//! Example 03: Rolling Windows (SMA, Rolling Std, Bollinger Bands)
//!
//! Level: Simple
//!
//! Computes a 20-bar Simple Moving Average (SMA) and 20-bar rolling
//! standard deviation on the bundled `stock_prices.csv` close column,
//! then builds Bollinger-style bands (SMA +/- 2 * rolling std). Uses
//! the `RollingWindow` trait and the `rolling_mean` / `rolling_std_dev`
//! free functions from `quant-core`.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 03_rolling_window
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::core::{rolling_mean, rolling_std_dev};
use quant_lib::prelude::*;

const WINDOW: usize = 20;
const BOLLINGER_K: f64 = 2.0;

fn main() {
    let csv = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&csv);
    let closes = common::closes(&bars);

    println!("=== Example 03: Rolling Windows & Bollinger Bands ===");
    println!("Loaded {} bars, window = {WINDOW}", bars.len());

    // Free-function form: rolling_mean(window, data).
    let sma = rolling_mean(WINDOW, &closes).expect("window valid");
    let rstd = rolling_std_dev(WINDOW, &closes).expect("window valid");

    // Trait form: the slice's `RollingWindow` impl gives the same result.
    let sma_trait = closes.as_slice().rolling_mean(WINDOW).unwrap();
    assert_eq!(sma.len(), sma_trait.len());
    for (a, b) in sma.iter().zip(sma_trait.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    println!("SMA[0]   = {:.4}  (avg of first {WINDOW} closes)", sma[0]);
    println!("SMA[last]= {:.4}", sma[sma.len() - 1]);
    println!("RSD[0]   = {:.6}", rstd[0]);

    // Bollinger bands: SMA +/- k * rolling std. Print the last few rows.
    let n = sma.len();
    let start = n.saturating_sub(5);
    println!("\n  date         close     SMA20     lower      upper");
    for i in start..n {
        let idx = i + WINDOW - 1; // closes index aligned with sma[i]
        let lower = sma[i] - BOLLINGER_K * rstd[i];
        let upper = sma[i] + BOLLINGER_K * rstd[i];
        println!(
            "  {}  {:8.2}  {:8.2}  {:8.2}  {:8.2}",
            bars[idx].date, closes[idx], sma[i], lower, upper
        );
    }

    // Sanity: SMA at the first window equals the manual mean of the slice.
    let manual: f64 = closes[..WINDOW].iter().sum::<f64>() / WINDOW as f64;
    assert!((sma[0] - manual).abs() < 1e-9);
    println!("\nCross-check: manual SMA[0] = {manual:.4} (matches)");
}
