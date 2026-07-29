//! Example 06: Stationarity - ADF Test and Fractional Differentiation
//!
//! Level: Intermediate
//!
//! Financial price series are typically non-stationary (unit root),
//! which breaks the assumptions of many statistical models. This
//! example demonstrates:
//!
//! - `adf_test` on raw PETR4 closes (expect non-stationary)
//! - `adf_test` on log returns (expect stationary)
//! - `frac_diff` with d=0.5 (partial differencing preserving memory)
//! - `find_min_d` to locate the smallest d that achieves stationarity
//!
//! Uses `quant-timeseries`. López de Prado (AFML Ch.5) argues that full
//! differencing (d=1) destroys all memory, while fractional differencing
//! preserves as much memory as possible while achieving stationarity.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 06_stationarity
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::prelude::*;
use quant_lib::timeseries::{CusumConfig, CusumDetector, acf, find_min_d};

fn main() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

    println!("=== Example 06: Stationarity ===");
    println!("PETR4: {} bars from {}", bars.len(), path.display());

    // 1. ADF on raw closes: unit root expected (non-stationary).
    let adf_prices = adf_test(&closes, 1).expect("ADF on prices");
    println!(
        "\nADF on closes:    statistic={:+.4}, crit=-2.86, stationary={}",
        adf_prices.statistic, adf_prices.is_stationary
    );

    // 2. ADF on log returns: stationary expected.
    let lr = log_returns(&closes);
    let adf_ret = adf_test(&lr, 1).expect("ADF on returns");
    println!(
        "ADF on log ret:   statistic={:+.4}, crit=-2.86, stationary={}",
        adf_ret.statistic, adf_ret.is_stationary
    );

    // 3. Fractional differentiation with d=0.5: partial memory preserved.
    // Use threshold=1e-3 (larger than the 1e-5 academic default) so the
    // FFD weight window stays short enough for our ~1247-bar series.
    let d = 0.5;
    let threshold = 1e-3;
    let fd = frac_diff(&closes, d, threshold).expect("frac_diff");
    let adf_fd = adf_test(&fd, 1).expect("ADF on frac-diff");
    println!(
        "ADF on frac_diff(d={d}): statistic={:+.4}, stationary={}",
        adf_fd.statistic, adf_fd.is_stationary
    );
    println!(
        "  frac-diff series length = {} (from {} closes)",
        fd.len(),
        closes.len()
    );

    // 4. Find the minimum d that makes the series stationary.
    let min_d = find_min_d(&closes, threshold, 0.01).expect("find_min_d");
    println!("\nfind_min_d: minimum d for stationarity = {min_d:.2}");
    let fd_min = frac_diff(&closes, min_d, threshold).expect("frac_diff at min_d");
    let adf_min = adf_test(&fd_min, 1).expect("ADF at min_d");
    println!(
        "  ADF at d={min_d:.2}: statistic={:+.4}, stationary={}",
        adf_min.statistic, adf_min.is_stationary
    );

    // 5. Autocorrelation function of the returns: ACF(0)=1, decay for k>0.
    let acf_vals = acf(&lr, 10).expect("ACF");
    println!("\nACF of log returns (first 11 lags):");
    for (k, rho) in acf_vals.iter().enumerate() {
        println!("  rho[{k}] = {rho:+.4}");
    }
    assert!((acf_vals[0] - 1.0).abs() < 1e-9, "ACF(0) must be 1");

    // 6. CUSUM structural break detection on the returns.
    let detector = CusumDetector::new(CusumConfig::new(5.0, 0.0));
    let breaks = detector.detect(&lr).expect("CUSUM detect");
    println!("\nCUSUM breaks (threshold=5.0): {} detected", breaks.len());
    for b in &breaks {
        println!("  break at bar {}: stat={:.4}", b.index, b.statistic);
    }
}
