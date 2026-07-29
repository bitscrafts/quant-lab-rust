//! Example 07: Volatility Models - EWMA, ARCH, GARCH
//!
//! Level: Intermediate
//!
//! Fits and compares three volatility models on PETR4 log returns:
//!
//! - `ewma_vol`: RiskMetrics exponentially weighted moving average
//! - `ArchModel::fit`: ARCH(1) via Gaussian MLE
//! - `GarchModel::fit`: GARCH(1,1) via Gaussian MLE
//!
//! Reports the conditional volatility path, long-run variance, and
//! persistence of each fit. GARCH(1,1) typically dominates ARCH(1) on
//! equity returns because volatility clustering is persistent.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 07_volatility
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::prelude::*;

fn main() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);

    println!("=== Example 07: Volatility Models ===");
    println!("PETR4: {} returns", ret.len());

    // 1. EWMA (RiskMetrics): sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2.
    let lambda = 0.94;
    let ewma = ewma_vol(&ret, lambda).expect("EWMA");
    let ewma_last = ewma[ewma.len() - 1];
    println!(
        "\nEWMA(lambda={lambda}): last sigma = {:.6} ({:.2}% daily)",
        ewma_last,
        ewma_last * 100.0
    );

    // 2. ARCH(1): sigma_t^2 = omega + alpha * r_{t-1}^2.
    let arch = ArchModel::fit(&ret, 1).expect("ARCH(1) fit");
    let arch_cv = arch.conditional_variances(&ret);
    let arch_last = arch_cv[arch_cv.len() - 1].sqrt();
    println!(
        "ARCH(1): omega={:.6}, alpha={:.4}, last sigma = {:.6}",
        arch.omega, arch.alphas[0], arch_last
    );

    // 3. GARCH(1,1): sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2.
    let garch = GarchModel::fit(&ret, 1, 1).expect("GARCH(1,1) fit");
    let g_cv = garch.conditional_variances(&ret);
    let g_last = g_cv[g_cv.len() - 1].sqrt();
    let persistence = garch.persistence();
    let long_run = garch.long_run_variance();
    println!(
        "GARCH(1,1): omega={:.6}, alpha={:.4}, beta={:.4}",
        garch.omega, garch.alphas[0], garch.betas[0]
    );
    println!("  persistence = {persistence:.4} (expect < 1 for stationary)");
    println!(
        "  long-run variance = {long_run:.6} (sigma = {:.6})",
        long_run.sqrt()
    );
    println!("  last sigma = {g_last:.6} ({:.2}% daily)", g_last * 100.0);

    // 4. Volatility forecast: 5-day ahead from GARCH.
    let horizon = 5;
    let forecast = garch.forecast(horizon);
    println!("\nGARCH {horizon}-day volatility forecast:");
    for (h, v) in forecast.iter().enumerate() {
        println!("  h+{}: sigma = {:.6}", h + 1, v.sqrt());
    }

    // 5. Cross-check: unconditional variance of returns vs GARCH long-run.
    let sample_var = variance(&ret).unwrap();
    println!("\nSample variance of returns = {sample_var:.6}");
    println!("GARCH long-run variance     = {long_run:.6}");
    // They should be roughly the same order of magnitude.
    let ratio = long_run / sample_var;
    println!("Ratio (GARCH / sample) = {ratio:.4} (expect ~1 if model fits well)");
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "GARCH long-run var wildly off from sample var"
    );
    println!("GARCH long-run variance is within an order of magnitude of the sample variance.");
}
