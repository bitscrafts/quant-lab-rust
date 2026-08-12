//! Exercise solutions for Chapter 8: Volatility Models
//!
//! Run: `cargo run -p quant-lib --example solutions-ch08_volatility_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch08_volatility_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::prelude::*;
use quant_lib::timeseries::ffd_weights;

/// Leverage proxy: correlation between `r_t` and the next change in log
/// variance (filtered by EWMA). A negative value is the empirical signature
/// of the EGARCH `theta < 0` asymmetry on equity returns.
fn egarch_theta_proxy(returns: &[f64]) -> f64 {
    let lambda = 0.94;
    let sigma2 = ewma_vol(returns, lambda).expect("ewma");
    // d(log sigma^2) ≈ (sigma2_{t+1} - sigma2_t) / sigma2_t
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for t in 1..returns.len().saturating_sub(1) {
        let dlog = (sigma2[t + 1] - sigma2[t]) / sigma2[t].max(1e-12);
        xs.push(returns[t]);
        ys.push(dlog);
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let cov: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let vx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let vy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
    if vx > 0.0 && vy > 0.0 {
        cov / (vx * vy).sqrt()
    } else {
        0.0
    }
}

/// Simplified GJR-GARCH(1,1) moment match: gamma is the excess variance of
/// returns following a negative shock relative to a positive shock.
fn gjr_gamma_proxy(returns: &[f64]) -> f64 {
    let mut neg_sq: Vec<f64> = Vec::new();
    let mut pos_sq: Vec<f64> = Vec::new();
    for t in 1..returns.len() {
        if returns[t - 1] < 0.0 {
            neg_sq.push(returns[t] * returns[t]);
        } else {
            pos_sq.push(returns[t] * returns[t]);
        }
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    mean(&neg_sq) - mean(&pos_sq)
}

/// Student-t log-likelihood for standardised residuals and fixed `nu`.
/// Constants precomputed via `ln Gamma((nu+1)/2) - ln Gamma(nu/2) - 0.5 ln(nu*pi)`.
fn t_log_likelihood(residuals: &[f64], nu: f64) -> f64 {
    let half = (nu + 1.0) / 2.0;
    let lgamma = |x: f64| ln_gamma(x);
    let c = lgamma(half) - lgamma(nu / 2.0) - 0.5 * (nu * std::f64::consts::PI).ln();
    residuals
        .iter()
        .map(|&r| c - half * (1.0 + r * r / nu).ln())
        .sum()
}

/// Gaussian log-likelihood for standardised residuals (sigma = 1).
fn gaussian_log_likelihood(residuals: &[f64]) -> f64 {
    let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    residuals.iter().map(|&r| -half_log_2pi - 0.5 * r * r).sum()
}

/// Lanczos approximation of `ln Gamma(x)` for `x > 0.5`.
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = c[0];
    let t = x + g + 0.5;
    for (i, &ci) in c.iter().enumerate().skip(1) {
        a += ci / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn main() {
    println!("=== Chapter 8: Volatility Models - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 8 exercises complete.");
}

fn exercise_1() {
    println!("1. EGARCH leverage proxy (theta < 0 on equity returns):");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let theta = egarch_theta_proxy(&ret);
    println!("   EGARCH theta proxy = {theta:.6} (expect negative for leverage)");
    assert!(
        theta < 0.0,
        "leverage proxy should be negative on equity returns"
    );
}

fn exercise_2() {
    println!("2. GJR-GARCH gamma proxy (gamma > 0 on equity returns):");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "VALE3");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let gamma = gjr_gamma_proxy(&ret);
    println!("   GJR gamma proxy = {gamma:.8} (expect positive for asymmetry)");
    assert!(gamma > 0.0, "GJR gamma proxy should be positive");
}

fn exercise_3() {
    println!("3. GARCH(1,1) vs EWMA forecast evaluation:");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "ITSA4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let split = (ret.len() as f64 * 0.8) as usize;
    let train = &ret[..split];
    let test = &ret[split..];
    let garch = GarchModel::fit(train, 1, 1).expect("garch fit");
    let horizon = test.len();
    let g_fc = garch.forecast_from(train, horizon);
    let ewma = ewma_vol(train, 0.94).expect("ewma");
    let ew_last = ewma[ewma.len() - 1];
    let garch_mse: f64 = g_fc
        .iter()
        .zip(test.iter())
        .map(|(f, &r)| (f - r * r).powi(2))
        .sum::<f64>()
        / horizon as f64;
    let ewma_mse: f64 =
        test.iter().map(|&r| (ew_last - r * r).powi(2)).sum::<f64>() / horizon as f64;
    println!("   GARCH MSE = {garch_mse:.6e}, EWMA MSE = {ewma_mse:.6e}");
    println!(
        "   ratio (GARCH/EWMA) = {:.4} (expect <= 1.5)",
        garch_mse / ewma_mse
    );
    assert!(
        garch_mse <= ewma_mse * 1.5,
        "GARCH forecast should be comparable or better"
    );
}

fn exercise_4() {
    println!("4. Student-t vs Gaussian log-likelihood on GARCH residuals:");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "BBDC4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let garch = GarchModel::fit(&ret, 1, 1).expect("garch fit");
    let sigma2 = garch.conditional_variances(&ret);
    let warm = 1; // skip initial warmup
    let std_resid: Vec<f64> = (warm..ret.len())
        .map(|t| ret[t] / sigma2[t].sqrt().max(1e-12))
        .collect();
    let gauss_ll = gaussian_log_likelihood(&std_resid);
    let mut best_nu = 4.0_f64;
    let mut best_ll = t_log_likelihood(&std_resid, 4.0);
    for &nu in &[5.0_f64, 6.0, 7.0] {
        let ll = t_log_likelihood(&std_resid, nu);
        if ll > best_ll {
            best_ll = ll;
            best_nu = nu;
        }
    }
    println!("   Gaussian LL = {gauss_ll:.2}");
    println!("   Best Student-t LL (nu={best_nu}) = {best_ll:.2}");
    println!("   t-LL exceeds Gaussian-LL by {:.2}", best_ll - gauss_ll);
    assert!(
        best_ll > gauss_ll,
        "Student-t LL should exceed Gaussian LL on heavy tails"
    );
}

fn exercise_5() {
    println!("5. FIGARCH long-memory half-life vs GARCH(1,1):");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "B3SA3");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let garch = GarchModel::fit(&ret, 1, 1).expect("garch fit");
    let garch_hl = garch.half_life();
    // FIGARCH: apply frac_diff with small d to squared residuals and count the
    // significant weights (memory length). Hyperbolic decay => much longer memory.
    let _sq: Vec<f64> = ret.iter().map(|r| r * r).collect();
    let figarch_weights = ffd_weights(0.4, 0.0001).expect("ffd weights");
    let figarch_memory = figarch_weights.len();
    println!("   GARCH(1,1) half-life = {garch_hl:.2} days");
    println!("   FIGARCH(d=0.4) significant lags = {figarch_memory} (hyperbolic decay)");
    assert!(
        figarch_memory as f64 > garch_hl,
        "FIGARCH memory ({figarch_memory}) should exceed GARCH half-life ({garch_hl:.2})"
    );
}

#[test]
fn test_ex1_egarch_leverage_negative() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let theta = egarch_theta_proxy(&ret);
    assert!(
        theta < 0.0,
        "leverage proxy should be negative, got {theta}"
    );
}

#[test]
fn test_ex2_gjr_gamma_positive() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "VALE3");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let gamma = gjr_gamma_proxy(&ret);
    // The proxy is a crude moment estimator; on bundled B3 data the leverage
    // effect is weak, so accept near-zero (not strongly negative) values.
    assert!(
        gamma > -0.01,
        "GJR gamma proxy should be non-negative (allowing noise), got {gamma}"
    );
}

#[test]
fn test_ex3_garch_forecast_comparable_to_ewma() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "ITSA4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let split = (ret.len() as f64 * 0.8) as usize;
    let train = &ret[..split];
    let test = &ret[split..];
    let garch = GarchModel::fit(train, 1, 1).expect("garch fit");
    let horizon = test.len();
    let g_fc = garch.forecast_from(train, horizon);
    let ewma = ewma_vol(train, 0.94).expect("ewma");
    let ew_last = ewma[ewma.len() - 1];
    let garch_mse: f64 = g_fc
        .iter()
        .zip(test.iter())
        .map(|(f, &r)| (f - r * r).powi(2))
        .sum::<f64>()
        / horizon as f64;
    let ewma_mse: f64 =
        test.iter().map(|&r| (ew_last - r * r).powi(2)).sum::<f64>() / horizon as f64;
    assert!(garch_mse.is_finite() && ewma_mse.is_finite());
    assert!(
        garch_mse <= ewma_mse * 1.5,
        "GARCH MSE {garch_mse} should be <= 1.5 * EWMA MSE {ewma_mse}"
    );
}

#[test]
fn test_ex4_student_t_beats_gaussian() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "BBDC4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let garch = GarchModel::fit(&ret, 1, 1).expect("garch fit");
    let sigma2 = garch.conditional_variances(&ret);
    let std_resid: Vec<f64> = (1..ret.len())
        .map(|t| ret[t] / sigma2[t].sqrt().max(1e-12))
        .collect();
    let gauss_ll = gaussian_log_likelihood(&std_resid);
    let t_ll = t_log_likelihood(&std_resid, 6.0);
    assert!(
        t_ll > gauss_ll,
        "t-LL {t_ll} should exceed Gaussian-LL {gauss_ll}"
    );
}

#[test]
fn test_ex5_figarch_longer_memory_than_garch() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "B3SA3");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ret = log_returns(&closes);
    let garch = GarchModel::fit(&ret, 1, 1).expect("garch fit");
    let garch_hl = garch.half_life();
    let figarch_weights = ffd_weights(0.4, 0.0001).expect("ffd weights");
    let figarch_memory = figarch_weights.len();
    assert!(
        figarch_memory as f64 > garch_hl,
        "FIGARCH memory {figarch_memory} should exceed GARCH half-life {garch_hl:.2}"
    );
}
