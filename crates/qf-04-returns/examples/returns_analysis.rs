//! Example: Returns and volatility analysis.
//!
//! Demonstrates computing simple and log returns, daily and annualised
//! volatility, Sharpe ratio, and drawdown for a synthetic price series.
//!
//! Run from the `quant-lab` directory:
//!   cargo run -p qf-04-returns --example returns_analysis

use qf_04_returns::{
    annualized_sharpe, annualized_volatility, drawdown_stats, log_returns, max_drawdown,
    simple_returns, sortino_ratio, volatility, Returns,
};

fn main() {
    println!("Returns Analysis");
    println!("================\n");

    // A synthetic 30-day price series (deterministic, no external data file).
    let prices: Vec<f64> = (0..30)
        .map(|t| {
            // gentle upward drift with a mid-series drawdown
            let base = 100.0 + 0.3 * t as f64;
            let dip = if (10..=15).contains(&t) { -8.0 } else { 0.0 };
            (base + dip).max(1.0)
        })
        .collect();

    let simple = simple_returns(&prices);
    let logr = log_returns(&prices);

    println!(
        "Simple Returns: mean={:.4}, std={:.4}",
        mean(&simple),
        volatility(&simple)
    );
    println!(
        "Log Returns: mean={:.4}, std={:.4}",
        mean(&logr),
        volatility(&logr)
    );

    let daily_vol = volatility(&simple);
    let ann_vol = annualized_volatility(&simple, 252.0);
    println!("Volatility (daily): {:.4}", daily_vol);
    println!("Volatility (annualized): {:.4}", ann_vol);

    let sharpe_ann = annualized_sharpe(&simple, 0.0, 252.0);
    println!("Sharpe Ratio (annualized): {:.2}", sharpe_ann);

    let sortino = sortino_ratio(&simple, 0.0);
    println!("Sortino Ratio: {:.2}", sortino);

    let md = max_drawdown(&prices);
    println!("Max Drawdown: {:.2}%", md * 100.0);

    let stats = drawdown_stats(&prices);
    println!(
        "Drawdown duration: {} steps, recovery: {:?}",
        stats.max_drawdown_duration, stats.recovery_time
    );

    // The `Returns` trait also works on a `Vec<f64>` directly.
    let trait_simple = prices.simple_returns();
    println!(
        "\nTrait check (Vec<f64>): first simple return = {:.4}",
        trait_simple[0]
    );
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}