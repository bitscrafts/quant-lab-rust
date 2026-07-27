//! Tests for the quant-portfolio crate (Phase 11 TDD contract, 15 tests).
//!
//! These tests exercise the public API surface only: [`Portfolio`] and its
//! statistics, the two-asset closed-form frontier, the N-asset Lagrangian
//! minimum-variance and efficient-frontier points, the tangency portfolio,
//! the capital market line, two-fund separation, CAPM beta/alpha, and
//! historical VaR / CVaR.

use approx::assert_abs_diff_eq;
use quant_portfolio::{
    alpha, beta, capital_market_line, efficient_frontier_point, historical_cvar, historical_var,
    min_variance_portfolio, portfolio_return, portfolio_variance, portfolio_volatility,
    sharpe_ratio, sml, tangency_portfolio, two_asset_frontier_point,
    two_asset_min_variance_weight, two_fund_separation, Portfolio,
};

/// Two uncorrelated assets used throughout the suite.
fn sample_universe() -> (Vec<f64>, Vec<Vec<f64>>) {
    let mu = vec![0.10, 0.05];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
    (mu, cov)
}

// --- Portfolio basics (4 tests) ----------------------------------------

#[test]
fn test_portfolio_return_two_assets() {
    let w = vec![0.6, 0.4];
    let mu = vec![0.10, 0.05];
    assert_abs_diff_eq!(portfolio_return(&w, &mu), 0.08, epsilon = 1e-12);
}

#[test]
fn test_portfolio_variance_diagonal() {
    let w = vec![0.5, 0.5];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
    // 0.25 * 0.04 + 0.25 * 0.09 = 0.0325.
    assert_abs_diff_eq!(portfolio_variance(&w, &cov), 0.0325, epsilon = 1e-12);
}

#[test]
fn test_portfolio_volatility() {
    let w = vec![0.5, 0.5];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
    assert_abs_diff_eq!(portfolio_volatility(&w, &cov), 0.0325_f64.sqrt(), epsilon = 1e-12);
}

#[test]
fn test_portfolio_sharpe() {
    let w = vec![0.5, 0.5];
    let mu = vec![0.10, 0.05];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
    // mu_p = 0.075, vol = sqrt(0.0325), rf = 0.02.
    let expected = (0.075 - 0.02) / 0.0325_f64.sqrt();
    assert_abs_diff_eq!(sharpe_ratio(&w, &mu, &cov, 0.02), expected, epsilon = 1e-12);
    // Also exercise the Portfolio struct.
    let p = Portfolio::new(w, mu, cov).unwrap();
    assert_abs_diff_eq!(p.sharpe(0.02), expected, epsilon = 1e-12);
}

// --- Two-asset frontier (3 tests) --------------------------------------

#[test]
fn test_two_asset_frontier_endpoints() {
    let p_a = two_asset_frontier_point(1.0, 0.10, 0.05, 0.04, 0.09, 0.0);
    assert_abs_diff_eq!(p_a.expected_return, 0.10, epsilon = 1e-12);
    assert_abs_diff_eq!(p_a.volatility, 0.20, epsilon = 1e-12);
    let p_b = two_asset_frontier_point(0.0, 0.10, 0.05, 0.04, 0.09, 0.0);
    assert_abs_diff_eq!(p_b.expected_return, 0.05, epsilon = 1e-12);
    assert_abs_diff_eq!(p_b.volatility, 0.30, epsilon = 1e-12);
}

#[test]
fn test_two_asset_min_variance_weight() {
    // sigma_A^2 = 0.04, sigma_B^2 = 0.09, rho = 0 -> cov = 0.
    // w_A* = 0.09 / (0.04 + 0.09) = 9/13.
    let w = two_asset_min_variance_weight(0.04, 0.09, 0.0);
    assert_abs_diff_eq!(w, 9.0 / 13.0, epsilon = 1e-12);
}

#[test]
fn test_two_asset_frontier_curve() {
    // Sweep w in 0.1 increments and verify the frontier relationship:
    // sigma_p^2 = w^2 * var_A + (1-w)^2 * var_B + 2*w*(1-w)*cov.
    for i in 0..=10 {
        let w = i as f64 * 0.1;
        let p = two_asset_frontier_point(w, 0.10, 0.05, 0.04, 0.09, 0.0);
        let expected_var = w * w * 0.04 + (1.0 - w) * (1.0 - w) * 0.09;
        assert_abs_diff_eq!(p.volatility * p.volatility, expected_var, epsilon = 1e-12);
        let expected_ret = w * 0.10 + (1.0 - w) * 0.05;
        assert_abs_diff_eq!(p.expected_return, expected_ret, epsilon = 1e-12);
    }
}

// --- N-asset minimum variance (2 tests) ---------------------------------

#[test]
fn test_min_variance_portfolio_n_assets() {
    let (mu, cov) = sample_universe();
    let w = min_variance_portfolio(&mu, &cov).unwrap();
    // Weights sum to 1 (budget constraint).
    let sum: f64 = w.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
    // Closed form: w_i proportional to 1/sigma_i^2.
    let expected_a = (1.0 / 0.04) / (1.0 / 0.04 + 1.0 / 0.09);
    assert_abs_diff_eq!(w[0], expected_a, epsilon = 1e-9);
}

#[test]
fn test_efficient_frontier_target_return() {
    let (mu, cov) = sample_universe();
    let target = 0.075;
    let w = efficient_frontier_point(&mu, &cov, target).unwrap();
    // Budget constraint.
    let sum: f64 = w.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
    // Realised return matches target.
    let realised = portfolio_return(&w, &mu);
    assert_abs_diff_eq!(realised, target, epsilon = 1e-9);
}

// --- Tangency portfolio and CML (3 tests) -------------------------------

#[test]
fn test_tangency_portfolio_max_sharpe() {
    let (mu, cov) = sample_universe();
    let rf = 0.02;
    let tan = tangency_portfolio(&mu, &cov, rf).unwrap();
    // Weights sum to 1.
    let sum: f64 = tan.weights.iter().sum();
    assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
    // Tangency Sharpe must dominate any 0.1-step grid portfolio.
    for i in 0..=10 {
        let w = vec![i as f64 * 0.1, 1.0 - i as f64 * 0.1];
        let s = sharpe_ratio(&w, &mu, &cov, rf);
        assert!(tan.sharpe + 1e-9 >= s, "grid Sharpe {s} beats tangency {}", tan.sharpe);
    }
}

#[test]
fn test_capital_market_line() {
    let (mu, cov) = sample_universe();
    let rf = 0.02;
    let tan = tangency_portfolio(&mu, &cov, rf).unwrap();
    // At sigma = 0 the CML returns rf.
    assert_abs_diff_eq!(capital_market_line(rf, &tan, 0.0), rf, epsilon = 1e-12);
    // At sigma_tan the CML returns mu_tan.
    let cml_at_tan = capital_market_line(rf, &tan, tan.volatility);
    assert_abs_diff_eq!(cml_at_tan, tan.expected_return, epsilon = 1e-9);
}

#[test]
fn test_two_fund_separation() {
    let (mu, cov) = sample_universe();
    let tan = tangency_portfolio(&mu, &cov, 0.02).unwrap();
    // Target vol of half sigma_tan -> y = 0.5.
    let y = two_fund_separation(&tan, 0.5 * tan.volatility);
    assert_abs_diff_eq!(y, 0.5, epsilon = 1e-9);
    // Combined portfolio expected return: rf + y * (mu_tan - rf).
    let combined_ret = 0.02 + y * (tan.expected_return - 0.02);
    let cml_ret = capital_market_line(0.02, &tan, 0.5 * tan.volatility);
    assert_abs_diff_eq!(combined_ret, cml_ret, epsilon = 1e-9);
}

// --- CAPM (2 tests) -----------------------------------------------------

#[test]
fn test_capm_beta() {
    // Asset = 0.5 * market -> beta = 0.5 (perfect correlation, half vol).
    let market = vec![0.04, -0.02, 0.03, 0.01, -0.01];
    let asset: Vec<f64> = market.iter().map(|m| 0.5 * m).collect();
    let b = beta(&asset, &market).unwrap();
    assert_abs_diff_eq!(b, 0.5, epsilon = 1e-9);
}

#[test]
fn test_capm_alpha() {
    // Asset lying exactly on the SML has zero alpha.
    let rf = 0.02;
    let market = vec![0.05, 0.06, 0.04, 0.07, 0.05];
    let b_target = 1.2_f64;
    let asset: Vec<f64> = market.iter().map(|m| rf + b_target * (m - rf)).collect();
    let a = alpha(&asset, &market, rf).unwrap();
    assert_abs_diff_eq!(a, 0.0, epsilon = 1e-9);
    // SML prediction at beta = 1.2 must match asset mean.
    let mean_m: f64 = market.iter().sum::<f64>() / market.len() as f64;
    let mean_a: f64 = asset.iter().sum::<f64>() / asset.len() as f64;
    assert_abs_diff_eq!(mean_a, sml(b_target, mean_m, rf), epsilon = 1e-9);
}

// --- Historical VaR / CVaR (2 tests) ------------------------------------

#[test]
fn test_historical_var() {
    // 10-point arithmetic series from -0.045 to +0.045.
    let returns: Vec<f64> = (0..10).map(|i| (i as f64 - 4.5) * 0.01).collect();
    // 95% VaR -> 5% quantile with linear interpolation between sorted[0]
    // and sorted[1] at position 0.45: q = -0.045*0.55 + -0.035*0.45 = -0.0405.
    let var = historical_var(&returns, 0.95).unwrap();
    assert_abs_diff_eq!(var, 0.0405, epsilon = 1e-9);
}

#[test]
fn test_historical_cvar() {
    let returns: Vec<f64> = (0..10).map(|i| (i as f64 - 4.5) * 0.01).collect();
    let var = historical_var(&returns, 0.95).unwrap();
    let cvar = historical_cvar(&returns, 0.95).unwrap();
    // CVaR is the mean of the tail {r <= quantile} = {-0.045}.
    assert_abs_diff_eq!(cvar, 0.045, epsilon = 1e-9);
    // CVaR must be at least as large as VaR.
    assert!(cvar >= var - 1e-12, "CVaR {cvar} < VaR {var}");
}