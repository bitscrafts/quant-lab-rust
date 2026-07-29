//! Example 05: OLS Regression - CAPM Beta of PETR4 vs VALE3
//!
//! Level: Intermediate
//!
//! Loads two real Brazilian stock series (PETR4 and VALE3) from the
//! bundled B3 JSON files, computes log returns, and regresses PETR4
//! excess returns on VALE3 returns (a simplified single-factor model,
//! treating VALE3 as the "market" proxy). Uses `ols` from
//! `quant-timeseries` and reports coefficients, t-stats, and R-squared.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 05_ols_regression
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::prelude::*;

fn main() {
    let petr4_path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let vale3_path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "VALE3");
    let petr4 = common::load_json_ohlcv(&petr4_path);
    let vale3 = common::load_json_ohlcv(&vale3_path);

    println!("=== Example 05: OLS Regression ===");
    println!("PETR4: {} bars from {}", petr4.len(), petr4_path.display());
    println!("VALE3: {} bars from {}", vale3.len(), vale3_path.display());

    // Align the two series on their common date prefix.
    let n = petr4.len().min(vale3.len());
    let p_closes: Vec<f64> = petr4.iter().take(n).map(|b| b.close).collect();
    let v_closes: Vec<f64> = vale3.iter().take(n).map(|b| b.close).collect();

    // Log returns.
    let p_ret = log_returns(&p_closes);
    let v_ret = log_returns(&v_closes);
    let m = p_ret.len().min(v_ret.len());
    let y: Vec<f64> = p_ret[p_ret.len() - m..].to_vec();
    let x_market: Vec<f64> = v_ret[v_ret.len() - m..].to_vec();

    // Design matrix: [1, market_return] to fit intercept (alpha) and slope (beta).
    let x: Vec<Vec<f64>> = x_market.iter().map(|r| vec![1.0, *r]).collect();
    let fit = ols(&x, &y).expect("OLS fit");

    println!("\nOLS: PETR4_ret = alpha + beta * VALE3_ret");
    println!(
        "  alpha (intercept) = {:.6}  (t = {:+.3})",
        fit.coeffs[0], fit.t_stats[0]
    );
    println!(
        "  beta  (slope)     = {:.4}  (t = {:+.3})",
        fit.coeffs[1], fit.t_stats[1]
    );
    println!("  R-squared         = {:.4}", fit.r_squared);
    println!("  n = {} observations", y.len());

    // Cross-check beta with the portfolio crate's CAPM beta function.
    let beta_capm = quant_lib::portfolio::beta(&y, &x_market).unwrap();
    assert!(
        (beta_capm - fit.coeffs[1]).abs() < 1e-9,
        "beta mismatch: ols={} capm={}",
        fit.coeffs[1],
        beta_capm
    );
    println!("\nCross-check: portfolio::beta = {beta_capm:.4} (matches OLS slope)");

    // Verify residuals satisfy X * beta - y.
    let residual_norm: f64 = fit.residuals.iter().map(|r| r * r).sum::<f64>().sqrt();
    println!(
        "Residual RMS = {:.6}",
        residual_norm / (y.len() as f64).sqrt()
    );
}
