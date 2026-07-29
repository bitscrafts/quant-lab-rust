//! Example 09: Portfolio - Markowitz Frontier, Min Variance, Tangency, CML
//!
//! Level: Intermediate
//!
//! Builds a 3-asset universe from three real Brazilian stocks (PETR4,
//! VALE3, ITSA4), estimates the sample covariance of their log returns,
//! and computes:
//!
//! - The global minimum-variance portfolio weights
//! - The efficient frontier point at a target return
//! - The tangency (maximum-Sharpe) portfolio for a given risk-free rate
//! - The Capital Market Line at a target volatility
//! - Two-fund separation: the fraction in the tangency portfolio
//!
//! Uses `quant-portfolio`. The math is hand-rolled Gaussian elimination
//! on the normal equations (no `nalgebra`).
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 09_portfolio_frontier
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::portfolio::{
    capital_market_line, historical_cvar, historical_var, portfolio_return, portfolio_volatility,
    two_fund_separation,
};
use quant_lib::prelude::*;

const TRADING_DAYS: f64 = 252.0;

fn main() {
    let symbols = ["PETR4", "VALE3", "ITSA4"];
    let bars: Vec<Vec<common::OhlcvBar>> = symbols
        .iter()
        .map(|s| common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), s)))
        .collect();

    println!("=== Example 09: Markowitz Frontier ===");
    for (s, b) in symbols.iter().zip(bars.iter()) {
        println!("  {s}: {} bars", b.len());
    }

    // Align on the common date prefix.
    let n = bars.iter().map(|b| b.len()).min().unwrap();
    let closes: Vec<Vec<f64>> = bars
        .iter()
        .map(|b| b.iter().take(n).map(|x| x.close).collect())
        .collect();
    let rets: Vec<Vec<f64>> = closes.iter().map(|c| log_returns(c)).collect();
    let m = rets[0].len();

    // Annualised mean returns and covariance (daily -> annual).
    let mu: Vec<f64> = rets.iter().map(|r| mean(r) * TRADING_DAYS).collect();
    let cov = sample_covariance(&rets, m);

    println!("\nAnnualised expected returns:");
    for (s, m) in symbols.iter().zip(mu.iter()) {
        println!("  {s}: {m:.4} ({:.2}%)", m * 100.0);
    }
    println!("\nAnnualised covariance matrix:");
    print_matrix(&cov, &symbols);

    // 1. Global minimum-variance portfolio.
    let w_min = min_variance_portfolio(&mu, &cov).expect("min-variance");
    let mu_min = portfolio_return(&w_min, &mu);
    let sig_min = portfolio_volatility(&w_min, &cov);
    println!("\nMinimum-variance portfolio:");
    print_weights(&symbols, &w_min);
    println!(
        "  mu = {mu_min:.4}, sigma = {sig_min:.4}, Sharpe = {:.4}",
        (mu_min - 0.02) / sig_min
    );

    // 2. Efficient frontier at a target return.
    let mu_target = mu.iter().cloned().sum::<f64>() / mu.len() as f64; // mean of asset means
    let w_eff = efficient_frontier_point(&mu, &cov, mu_target).expect("frontier point");
    let mu_eff = portfolio_return(&w_eff, &mu);
    let sig_eff = portfolio_volatility(&w_eff, &cov);
    println!("\nEfficient frontier at mu_target = {mu_target:.4}:");
    print_weights(&symbols, &w_eff);
    println!("  mu = {mu_eff:.4}, sigma = {sig_eff:.4}");

    // 3. Tangency portfolio (max Sharpe) with rf = 2%.
    let rf = 0.02;
    let tan = tangency_portfolio(&mu, &cov, rf).expect("tangency");
    println!("\nTangency portfolio (rf = {rf}):");
    print_weights(&symbols, &tan.weights);
    println!(
        "  mu = {:.4}, sigma = {:.4}, Sharpe = {:.4}",
        tan.expected_return, tan.volatility, tan.sharpe
    );

    // 4. Capital Market Line at a target volatility.
    let sig_target = 0.30;
    let cml = capital_market_line(rf, &tan, sig_target);
    println!("\nCML at sigma = {sig_target:.2}: mu = {cml:.4} (= rf + Sharpe_tan * sigma)");

    // 5. Two-fund separation: fraction in tangency to hit target volatility.
    let y = two_fund_separation(&tan, sig_target);
    println!(
        "Two-fund separation: y = {y:.4} in tangency, {:.4} in risk-free",
        1.0 - y
    );

    // Sanity: weights sum to 1.
    let sum_min: f64 = w_min.iter().sum();
    let sum_tan: f64 = tan.weights.iter().sum();
    assert!(
        (sum_min - 1.0).abs() < 1e-9,
        "min-var weights must sum to 1, got {sum_min}"
    );
    assert!(
        (sum_tan - 1.0).abs() < 1e-9,
        "tangency weights must sum to 1, got {sum_tan}"
    );
    println!("\nSanity: min-var weights sum = {sum_min:.6}, tangency weights sum = {sum_tan:.6}");

    // 6. Historical VaR and CVaR at 95% on an equal-weight portfolio.
    let w_equal = [1.0 / 3.0; 3];
    let port_ret: Vec<f64> = (0..m)
        .map(|t| (0..3).map(|i| w_equal[i] * rets[i][t]).sum::<f64>() * TRADING_DAYS)
        .collect();
    let var95 = historical_var(&port_ret, 0.95).unwrap();
    let cvar95 = historical_cvar(&port_ret, 0.95).unwrap();
    println!("\nEqual-weight portfolio risk (annualised):");
    println!("  95% VaR   = {var95:.4} ({:.2}%)", var95 * 100.0);
    println!("  95% CVaR  = {cvar95:.4} ({:.2}%)", cvar95 * 100.0);
}

/// Sample covariance matrix of a list of return series (annualised).
fn sample_covariance(rets: &[Vec<f64>], m: usize) -> Vec<Vec<f64>> {
    let n = rets.len();
    let means: Vec<f64> = rets.iter().map(|r| mean(r)).collect();
    let mut cov = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let s: f64 = (0..m)
                .map(|t| (rets[i][t] - means[i]) * (rets[j][t] - means[j]))
                .sum::<f64>()
                / (m as f64 - 1.0);
            cov[i][j] = s * TRADING_DAYS;
        }
    }
    cov
}

fn print_matrix(cov: &[Vec<f64>], labels: &[&str]) {
    print!("          ");
    for l in labels {
        print!("{l:>10}");
    }
    println!();
    for (i, row) in cov.iter().enumerate() {
        print!("{:<10}", labels[i]);
        for v in row {
            print!("{v:10.4}");
        }
        println!();
    }
}

fn print_weights(labels: &[&str], w: &[f64]) {
    for (l, wi) in labels.iter().zip(w.iter()) {
        println!("  {l}: {wi:.4} ({:.2}%)", wi * 100.0);
    }
}
