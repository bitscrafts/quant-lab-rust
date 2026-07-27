//! Efficient frontier, tangency portfolio, and capital market line.
//!
//! Reproduces the canonical Markowitz picture: a two-asset universe, the
//! frontier hyperbola, the global minimum-variance portfolio, the tangency
//! portfolio, and the capital market line that touches the frontier at the
//! tangency point and extends to the right.
//!
//! Run with: `cargo run -p quant-portfolio --example frontier`.

use quant_portfolio::{
    capital_market_line, efficient_frontier_point, min_variance_portfolio, portfolio_return,
    portfolio_volatility, sharpe_ratio, tangency_portfolio, two_asset_frontier_point,
};

fn main() {
    // Two uncorrelated risky assets.
    let mu = vec![0.10, 0.05];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
    let rf = 0.02;

    println!("Markowitz efficient frontier (two uncorrelated assets)");
    println!("======================================================");
    println!("Asset A: mu = 0.10, sigma = 0.20");
    println!("Asset B: mu = 0.05, sigma = 0.30");
    println!("Covariance: 0 (uncorrelated)");
    println!("Risk-free rate: {rf}");
    println!();

    // Sweep the weight on asset A from 0.0 to 1.0 in 0.1 increments and print
    // the frontier (mu_p, sigma_p). Weights above 1 or below 0 (short sales)
    // are on the upper/lower branches of the hyperbola but are not printed
    // here for clarity.
    println!("Two-asset frontier sweep (w_A, w_B, mu_p, sigma_p, Sharpe):");
    println!("  w_A    w_B   mu_p    sigma_p   Sharpe");
    let mut best_sharpe = f64::NEG_INFINITY;
    let mut best_w = 0.0;
    for i in 0..=20 {
        let w_a = i as f64 * 0.05;
        let w_b = 1.0 - w_a;
        let p = two_asset_frontier_point(w_a, 0.10, 0.05, 0.04, 0.09, 0.0);
        let s = sharpe_ratio(&[w_a, w_b], &mu, &cov, rf);
        if s > best_sharpe {
            best_sharpe = s;
            best_w = w_a;
        }
        println!(
            "  {:>4.2}   {:>4.2}   {:>6.4}   {:>7.4}   {:>6.4}",
            w_a, w_b, p.expected_return, p.volatility, s
        );
    }
    println!();

    // Global minimum-variance portfolio (closed form, N-asset Lagrangian).
    let w_gmv = min_variance_portfolio(&mu, &cov).unwrap();
    let mu_gmv = portfolio_return(&w_gmv, &mu);
    let sigma_gmv = portfolio_volatility(&w_gmv, &cov);
    println!("Global minimum-variance portfolio (Sigma^-1 * 1 / (1' Sigma^-1 * 1)):");
    println!("  weights = [{:.4}, {:.4}]", w_gmv[0], w_gmv[1]);
    println!("  mu      = {:.6}", mu_gmv);
    println!("  sigma   = {:.6}", sigma_gmv);
    println!();

    // Tangency portfolio (closed form: Sigma^-1 (mu - rf 1) / 1' Sigma^-1 (mu - rf 1)).
    let tan = tangency_portfolio(&mu, &cov, rf).unwrap();
    println!("Tangency portfolio (maximum Sharpe):");
    println!(
        "  weights   = [{:.4}, {:.4}]",
        tan.weights[0], tan.weights[1]
    );
    println!("  mu_tan    = {:.6}", tan.expected_return);
    println!("  sigma_tan = {:.6}", tan.volatility);
    println!("  Sharpe    = {:.6}", tan.sharpe);
    println!(
        "  (best grid Sharpe at w_A = {:.2}, Sharpe = {:.6})",
        best_w, best_sharpe
    );
    println!();

    // Capital market line: mu = rf + Sharpe_tan * sigma.
    println!("Capital market line (mu = rf + Sharpe_tan * sigma):");
    println!("  sigma   mu_cml");
    for i in 0..=10 {
        let sigma = i as f64 * 0.05;
        let mu_cml = capital_market_line(rf, &tan, sigma);
        println!("  {:>5.2}   {:>6.4}", sigma, mu_cml);
    }
    println!();

    // N-asset efficient frontier via the Lagrangian target-return solver.
    println!("N-asset efficient frontier (target-return Lagrangian):");
    println!("  target   w_A     w_B     mu_p    sigma_p");
    for i in 0..=10 {
        let target = 0.05 + i as f64 * 0.01;
        let w = efficient_frontier_point(&mu, &cov, target).unwrap();
        let mu_p = portfolio_return(&w, &mu);
        let sigma_p = portfolio_volatility(&w, &cov);
        println!(
            "  {:>5.3}   {:>5.3}   {:>5.3}   {:>6.4}   {:>7.4}",
            target, w[0], w[1], mu_p, sigma_p
        );
    }
}
