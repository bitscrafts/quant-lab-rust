//! Project 3: Risk-Parity Portfolio Allocation
//!
//! Level: Advanced
//!
//! Constructs three portfolios from 8 B3 stocks and compares their
//! risk-adjusted performance: an inverse-variance (risk-parity)
//! allocation, an equal-weight allocation, and the Markowitz global
//! minimum-variance allocation. All three are evaluated on the same
//! annualised covariance matrix, and Sharpe ratios are computed at
//! rf = 0.0.
//!
//! Run: `cargo run -p quant-lib --example projects-03_risk_parity`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::{log_returns, mean};
use quant_lib::portfolio::{portfolio_return, portfolio_volatility};
use quant_lib::prelude::*;

const TRADING_DAYS: f64 = 252.0;

fn main() {
    println!("=== Project 3: Risk-Parity Portfolio ===\n");

    let symbols = [
        "PETR4", "VALE3", "ITSA4", "BBDC4", "B3SA3", "ABEV3", "GGBR4", "WEGE3",
    ];
    let bars: Vec<Vec<common::OhlcvBar>> = symbols
        .iter()
        .map(|s| common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), s)))
        .collect();
    let n = bars.iter().map(|b| b.len()).min().unwrap();
    let closes: Vec<Vec<f64>> = bars
        .iter()
        .map(|b| b.iter().take(n).map(|x| x.close).collect())
        .collect();
    let n_assets = symbols.len();

    // Log returns per asset (aligned, length n-1).
    let rets: Vec<Vec<f64>> = closes.iter().map(|c| log_returns(c)).collect();
    let m = rets[0].len();

    // Annualised mean returns and covariance matrix.
    let mu: Vec<f64> = rets.iter().map(|r| mean(r) * TRADING_DAYS).collect();
    let cov = sample_covariance(&rets, m, n_assets);

    println!("Annualised expected returns:");
    for (s, mi) in symbols.iter().zip(mu.iter()) {
        let pct = mi * 100.0;
        println!("  {s}: {mi:.4} ({pct:.2}%)");
    }

    let rf = 0.0;

    // --- 1. Inverse-variance (risk-parity) allocation. ---
    let variances: Vec<f64> = (0..n_assets).map(|i| cov[i][i]).collect();
    let inv_var: Vec<f64> = variances.iter().map(|v| 1.0 / *v).collect();
    let sum_iv: f64 = inv_var.iter().sum();
    let w_rp: Vec<f64> = inv_var.iter().map(|x| x / sum_iv).collect();

    // --- 2. Equal-weight allocation. ---
    let w_eq: Vec<f64> = vec![1.0 / n_assets as f64; n_assets];

    // --- 3. Markowitz minimum-variance allocation. ---
    let w_mv = min_variance_portfolio(&mu, &cov).expect("min-variance portfolio");

    println!("\nPortfolio weights:");
    println!(
        "  {:<8} {:>10} {:>10} {:>10}",
        "Asset", "RiskParity", "EqualWgt", "MinVar"
    );
    for i in 0..n_assets {
        println!(
            "  {:<8} {:>10.4} {:>10.4} {:>10.4}",
            symbols[i], w_rp[i], w_eq[i], w_mv[i]
        );
    }

    // --- Performance comparison. ---
    println!("\nPortfolio performance (annualised, rf = {rf}):");
    println!(
        "  {:<12} {:>10} {:>10} {:>10} {:>10}",
        "Allocation", "Return", "Vol", "Sharpe", "SumW"
    );
    print_allocation("RiskParity", &w_rp, &mu, &cov, rf);
    print_allocation("EqualWeight", &w_eq, &mu, &cov, rf);
    print_allocation("MinVariance", &w_mv, &mu, &cov, rf);

    // Sanity: weights sum to 1.
    let sum_rp: f64 = w_rp.iter().sum();
    let sum_mv: f64 = w_mv.iter().sum();
    let sum_eq: f64 = w_eq.iter().sum();
    println!("\nWeight sums: RiskParity={sum_rp:.6}, Equal={sum_eq:.6}, MinVar={sum_mv:.6}");

    // Per-asset annualised volatility for context.
    println!("\nPer-asset annualised volatility:");
    for (s, vi) in symbols.iter().zip(variances.iter()) {
        let vol = vi.sqrt();
        let vol_pct = vol * 100.0;
        println!("  {s}: {vol:.4} ({vol_pct:.2}%)");
    }
}

fn print_allocation(name: &str, w: &[f64], mu: &[f64], cov: &[Vec<f64>], rf: f64) {
    let ret = portfolio_return(w, mu);
    let vol = portfolio_volatility(w, cov);
    let sharpe = sharpe_ratio(w, mu, cov, rf);
    let sum_w: f64 = w.iter().sum();
    println!(
        "  {:<12} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
        name, ret, vol, sharpe, sum_w
    );
}

/// Sample covariance matrix of return series (annualised).
fn sample_covariance(rets: &[Vec<f64>], m: usize, n_assets: usize) -> Vec<Vec<f64>> {
    let means: Vec<f64> = rets.iter().map(|r| mean(r)).collect();
    let mut cov = vec![vec![0.0_f64; n_assets]; n_assets];
    for i in 0..n_assets {
        for j in 0..n_assets {
            let s: f64 = (0..m)
                .map(|t| (rets[i][t] - means[i]) * (rets[j][t] - means[j]))
                .sum::<f64>()
                / (m as f64 - 1.0);
            cov[i][j] = s * TRADING_DAYS;
        }
    }
    cov
}
