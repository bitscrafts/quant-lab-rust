//! Example 01: Mean, Variance, and Standard Deviation
//!
//! Level: Simple
//!
//! Loads the bundled `stock_prices.csv` close column and computes
//! the first two moments of the close-price distribution. Uses the
//! `mean`, `variance`, and `std_dev` functions from `quant-core`
//! re-exported through `quant_lib::core`.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 01_mean_variance
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::prelude::*;

fn main() {
    let csv = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&csv);
    let closes = common::closes(&bars);

    println!("=== Example 01: Mean, Variance, Standard Deviation ===");
    println!("Loaded {} OHLCV bars from {}", bars.len(), csv.display());

    let mu = mean(&closes);
    let var = variance(&closes).unwrap();
    let sd = std_dev(&closes).unwrap();

    println!(
        "Close prices: first={}, last={}",
        closes[0],
        closes[closes.len() - 1]
    );
    println!("Mean   = {mu:.4}");
    println!("Var    = {var:.4}");
    println!("StdDev = {sd:.4}");

    // Cross-check against the sample-mean formula by hand.
    let n = closes.len() as f64;
    let manual_mean: f64 = closes.iter().sum::<f64>() / n;
    assert!(
        (mu - manual_mean).abs() < 1e-9,
        "mean mismatch: {mu} vs {manual_mean}"
    );
    println!("Cross-check: manual mean = {manual_mean:.4} (matches)");
}
