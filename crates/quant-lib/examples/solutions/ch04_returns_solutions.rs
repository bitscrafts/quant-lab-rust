//! Exercise solutions for Chapter 4: Returns and Volatility
//!
//! Run: `cargo run -p quant-lib --example solutions-ch04_returns_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch04_returns_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::{log_returns, rolling_mean, rolling_std_dev, simple_returns};
use quant_lib::prelude::*;

/// Maximum drawdown of a price/equity curve (as a negative fraction).
fn max_drawdown(prices: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut mdd = 0.0_f64;
    for &p in prices {
        peak = peak.max(p);
        let dd = (p - peak) / peak;
        mdd = mdd.min(dd);
    }
    mdd
}

/// Average drawdown (mean of negative DD_t values).
fn average_drawdown(prices: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut sum = 0.0_f64;
    let mut count = 0u64;
    for &p in prices {
        peak = peak.max(p);
        let dd = (p - peak) / peak;
        if dd < 0.0 {
            sum += dd;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Proportion of bars the equity curve is in drawdown (DD_t < 0).
fn proportion_in_drawdown(prices: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut in_dd = 0u64;
    for &p in prices {
        peak = peak.max(p);
        if p < peak {
            in_dd += 1;
        }
    }
    if prices.is_empty() {
        0.0
    } else {
        in_dd as f64 / prices.len() as f64
    }
}

fn main() {
    println!("=== Chapter 4: Returns and Volatility - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 4 exercises complete.");
}

fn exercise_1() {
    println!("1. Return Conversion (log/simple round-trip):");
    let mut rng = XorShift64::new(42);
    let normal = Normal::new(0.0, 0.02);
    let mut max_err = 0.0_f64;
    for _ in 0..1000 {
        let r_simple = normal.sample(&mut rng);
        let r_log = (1.0 + r_simple).ln();
        let r_back = r_log.exp() - 1.0;
        let err = (r_back - r_simple).abs();
        max_err = max_err.max(err);
    }
    println!("   max round-trip error over 1000 samples = {max_err:.2e}");
    assert!(max_err < 1e-15, "round-trip error must be below 1e-15");
}

fn exercise_2() {
    println!("\n2. Annualisation Sanity Check (daily/weekly/monthly vol):");
    // Simulate 3 years of daily prices (252*3 bars) via GBM.
    let mut rng = XorShift64::new(7);
    let prices = gbm(100.0, 0.05, 0.20, 3.0, 252 * 3, &mut rng);
    let daily_rets = simple_returns(&prices);
    // Weekly returns: sum of 5 consecutive daily log returns.
    let weekly_rets: Vec<f64> = log_returns(&prices)
        .chunks(5)
        .map(|c| c.iter().sum::<f64>())
        .collect();
    // Monthly returns: sum of 21 consecutive daily log returns.
    let monthly_rets: Vec<f64> = log_returns(&prices)
        .chunks(21)
        .map(|c| c.iter().sum::<f64>())
        .collect();
    let sig_d = std_dev(&daily_rets).unwrap_or(0.0) * (252.0_f64).sqrt();
    let sig_w = std_dev(&weekly_rets).unwrap_or(0.0) * (52.0_f64).sqrt();
    let sig_m = std_dev(&monthly_rets).unwrap_or(0.0) * (12.0_f64).sqrt();
    println!("   annualised vol: daily={sig_d:.3}, weekly={sig_w:.3}, monthly={sig_m:.3}");
    let spread = (sig_d - sig_m).abs().max((sig_d - sig_w).abs());
    println!("   max spread = {spread:.3} (should be small relative to sigma)");
}

fn exercise_3() {
    println!("\n3. Calmar Ratio (annualised return / |max drawdown|):");
    let mut rng = XorShift64::new(21);
    // Strategy equity curve: 12% annual return with a 15% drawdown.
    let prices = gbm(100.0, 0.12, 0.15, 3.0, 252 * 3, &mut rng);
    let mdd = max_drawdown(&prices);
    let total_ret = prices.last().unwrap_or(&100.0) / prices.first().unwrap_or(&100.0) - 1.0;
    let ann_ret = (1.0 + total_ret).powf(1.0 / 3.0) - 1.0;
    let calmar = calmar_ratio(ann_ret, mdd.abs());
    println!("   annualised return = {ann_ret:.3}, max drawdown = {mdd:.3}, Calmar = {calmar:.3}");
}

fn exercise_4() {
    println!("\n4. Rolling Sharpe (windowed mean/std * sqrt(m)):");
    let mut rng = XorShift64::new(99);
    let prices = gbm(100.0, 0.08, 0.25, 3.0, 252 * 3, &mut rng);
    let rets = simple_returns(&prices);
    let win = 63; // quarterly window
    let m: f64 = 252.0;
    let r_f = 0.0;
    let means = rolling_mean(win, &rets).unwrap_or_default();
    let stds = rolling_std_dev(win, &rets).unwrap_or_default();
    let mut min_sharpe = f64::INFINITY;
    let mut max_sharpe = f64::NEG_INFINITY;
    for (mu, sd) in means.iter().zip(stds.iter()) {
        let sharpe = if *sd > 0.0 {
            (mu - r_f) / sd * m.sqrt()
        } else {
            0.0
        };
        min_sharpe = min_sharpe.min(sharpe);
        max_sharpe = max_sharpe.max(sharpe);
    }
    println!(
        "   rolling Sharpe range: [{min_sharpe:.3}, {max_sharpe:.3}] over {} windows",
        means.len()
    );
}

fn exercise_5() {
    println!("\n5. Drawdown with Recovery (avg drawdown, proportion in drawdown):");
    let mut rng = XorShift64::new(55);
    let prices = gbm(100.0, 0.06, 0.20, 5.0, 252 * 5, &mut rng);
    let avg_dd = average_drawdown(&prices);
    let prop = proportion_in_drawdown(&prices);
    println!("   average drawdown = {avg_dd:.4}, proportion in drawdown = {prop:.3}");
}

#[test]
fn test_ex1_round_trip_error_below_1e15() {
    let mut rng = XorShift64::new(42);
    let normal = Normal::new(0.0, 0.02);
    let mut max_err = 0.0_f64;
    for _ in 0..1000 {
        let r = normal.sample(&mut rng);
        let back = (1.0 + r).ln().exp() - 1.0;
        max_err = max_err.max((back - r).abs());
    }
    assert!(max_err < 1e-15);
}

#[test]
fn test_ex2_annualisation_consistent() {
    let mut rng = XorShift64::new(7);
    let prices = gbm(100.0, 0.05, 0.20, 3.0, 252 * 3, &mut rng);
    let daily_rets = simple_returns(&prices);
    let weekly_rets: Vec<f64> = log_returns(&prices)
        .chunks(5)
        .map(|c| c.iter().sum::<f64>())
        .collect();
    let monthly_rets: Vec<f64> = log_returns(&prices)
        .chunks(21)
        .map(|c| c.iter().sum::<f64>())
        .collect();
    let sig_d = std_dev(&daily_rets).unwrap_or(0.0) * (252.0_f64).sqrt();
    let sig_w = std_dev(&weekly_rets).unwrap_or(0.0) * (52.0_f64).sqrt();
    let sig_m = std_dev(&monthly_rets).unwrap_or(0.0) * (12.0_f64).sqrt();
    assert!(sig_d.is_finite() && sig_w.is_finite() && sig_m.is_finite());
    // Should agree to within ~20% (sampling noise at different frequencies).
    assert!((sig_d - sig_w).abs() / sig_d < 0.2);
    assert!((sig_d - sig_m).abs() / sig_d < 0.2);
}

#[test]
fn test_ex3_calmar_finite() {
    let mut rng = XorShift64::new(21);
    let prices = gbm(100.0, 0.12, 0.15, 3.0, 252 * 3, &mut rng);
    let mdd = max_drawdown(&prices);
    assert!(mdd <= 0.0, "max drawdown must be non-positive");
    let calmar = calmar_ratio(0.12, mdd.abs());
    assert!(calmar.is_finite() && calmar >= 0.0);
}

#[test]
fn test_ex4_rolling_sharpe_finite() {
    let mut rng = XorShift64::new(99);
    let prices = gbm(100.0, 0.08, 0.25, 3.0, 252 * 3, &mut rng);
    let rets = simple_returns(&prices);
    let win = 63;
    let means = rolling_mean(win, &rets).unwrap();
    let stds = rolling_std_dev(win, &rets).unwrap();
    assert!(!means.is_empty() && !stds.is_empty());
    assert!(means.iter().all(|v| v.is_finite()));
    assert!(stds.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn test_ex5_drawdown_stats_sane() {
    let mut rng = XorShift64::new(55);
    let prices = gbm(100.0, 0.06, 0.20, 5.0, 252 * 5, &mut rng);
    let avg_dd = average_drawdown(&prices);
    let prop = proportion_in_drawdown(&prices);
    assert!(avg_dd <= 0.0, "average drawdown is non-positive");
    assert!((0.0..=1.0).contains(&prop), "proportion in [0,1]");
    assert!(prop > 0.0, "buy-and-hold spends time in drawdown");
}
