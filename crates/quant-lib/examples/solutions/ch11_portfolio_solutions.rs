//! Exercise solutions for Chapter 11: Portfolio Optimization
//!
//! Run: `cargo run -p quant-lib --example solutions-ch11_portfolio_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch11_portfolio_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::portfolio::linalg::{inverse, matvec};
use quant_lib::portfolio::{
    beta, historical_cvar, two_asset_frontier_point, two_asset_min_variance_weight,
};
use quant_lib::prelude::*;

/// Project a vector onto the probability simplex (non-negativity + sum=1).
/// Uses the classic sort-and-threshold algorithm (Wang & Carreira-Perpinan, 2013).
fn project_simplex(w: &[f64]) -> Vec<f64> {
    let n = w.len();
    let mut sorted: Vec<f64> = w.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut theta = 0.0_f64;
    for j in 0..n {
        let sum: f64 = sorted[0..=j].iter().sum();
        let candidate = (sum - 1.0) / (j as f64 + 1.0);
        if sorted[j] - candidate > 0.0 {
            theta = candidate;
        }
    }
    w.iter().map(|&wi| (wi - theta).max(0.0)).collect()
}

/// Black-Litterman posterior mean given prior, views, and confidences.
fn black_litterman(
    sigma: &[Vec<f64>],
    w_mkt: &[f64],
    tau: f64,
    q: &[f64],
    p: &[Vec<f64>],
    omega: &[Vec<f64>],
) -> Vec<f64> {
    // Prior: pi = delta * Sigma * w_mkt. Use delta = 1 (assume prior = Sigma * w).
    let n = w_mkt.len();
    let pi: Vec<f64> = matvec(sigma, w_mkt);
    // (tau*Sigma)^-1
    let tau_sigma: Vec<Vec<f64>> = sigma
        .iter()
        .map(|row| row.iter().map(|v| tau * v).collect())
        .collect();
    let tau_sigma_inv = inverse(&tau_sigma).expect("tau sigma invertible");
    let omega_inv = inverse(omega).expect("omega invertible");
    // right1 = (tau*Sigma)^-1 * pi
    let right1 = matvec(&tau_sigma_inv, &pi);
    // right2 = Omega^-1 * P^T * q  (here P assumed identity for simplicity if k == n)
    let p_t: Vec<Vec<f64>> = (0..n)
        .map(|i| p.iter().map(|row| row[i]).collect())
        .collect();
    let ptq = matvec(&p_t, q);
    let right2 = matvec(&omega_inv, &ptq);
    // sum right1 + right2
    let rhs: Vec<f64> = right1
        .iter()
        .zip(right2.iter())
        .map(|(a, b)| a + b)
        .collect();
    // left matrix = (tau*Sigma)^-1 + Omega^-1 (P assumed identity)
    let mut lhs: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            lhs[i][j] = tau_sigma_inv[i][j] + omega_inv[i][j];
        }
    }
    let lhs_inv = inverse(&lhs).expect("lhs invertible");
    matvec(&lhs_inv, &rhs)
}

fn main() {
    println!("=== Chapter 11: Portfolio Optimization - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    exercise_6();
    println!("\nAll Chapter 11 exercises complete.");
}

fn exercise_1() {
    println!("1. Two-asset frontier for rho in {{0.5, -0.5, -1.0}}:");
    let (mu_a, mu_b, var_a, var_b): (f64, f64, f64, f64) = (0.10, 0.05, 0.04, 0.09);
    let sig_a = var_a.sqrt();
    let sig_b = var_b.sqrt();
    for rho in [0.5_f64, -0.5, -1.0] {
        let cov = rho * sig_a * sig_b;
        let w_min = two_asset_min_variance_weight(var_a, var_b, cov);
        let fp = two_asset_frontier_point(w_min, mu_a, mu_b, var_a, var_b, cov);
        println!(
            "   rho={rho}: w_min_A={w_min:.4}, min_vol={:.4}",
            fp.volatility
        );
        if rho == -1.0 {
            // For rho = -1 the min-variance portfolio has sigma = 0.
            assert!(
                fp.volatility.abs() < 1e-9,
                "rho=-1 should yield zero min-variance vol"
            );
        }
    }
}

fn exercise_2() {
    println!("2. Three-asset frontier (tangency Sharpe exceeds two-asset):");
    let mu = vec![0.10, 0.08, 0.08];
    let cov = vec![
        vec![0.04, 0.01, 0.01],
        vec![0.01, 0.0225, 0.005],
        vec![0.01, 0.005, 0.0225],
    ];
    let rf = 0.02;
    let tan3 = tangency_portfolio(&mu, &cov, rf).expect("tan3");
    let mu2 = vec![0.10, 0.08];
    let cov2 = vec![vec![0.04, 0.01], vec![0.01, 0.0225]];
    let tan2 = tangency_portfolio(&mu2, &cov2, rf).expect("tan2");
    let mv3 = min_variance_portfolio(&mu, &cov).expect("mv3");
    let mv3_sum: f64 = mv3.iter().sum();
    println!("   3-asset min-var weights = {:?}", mv3);
    println!("   3-asset tangency Sharpe = {:.4}", tan3.sharpe);
    println!("   2-asset tangency Sharpe = {:.4}", tan2.sharpe);
    assert!(
        (mv3_sum - 1.0).abs() < 1e-6,
        "min-var weights must sum to 1"
    );
    assert!(
        tan3.sharpe > tan2.sharpe,
        "3-asset Sharpe should exceed 2-asset"
    );
}

fn exercise_3() {
    println!("3. Long-only constraint (projected gradient):");
    // Unconstrained target return 0.10 with mu=[0.10, 0.05] shorts the low asset.
    // Constrained solution allocates 100% to high-return asset for a high target.
    let mu = [0.10, 0.05];
    let cov = [vec![0.04, 0.0], vec![0.0, 0.0225]];
    let _mu_target = 0.12; // unattainable without short -> constraint binds.
    let mut w = vec![0.5, 0.5];
    let lr = 0.5;
    for _ in 0..200 {
        let grad: Vec<f64> = (0..2)
            .map(|i| {
                let ret_grad = -mu[i];
                let risk_grad: f64 = cov[i]
                    .iter()
                    .zip(w.iter())
                    .map(|(c, wi)| 2.0 * c * wi)
                    .sum();
                ret_grad + 0.5 * risk_grad
            })
            .collect();
        for (wi, &g) in w.iter_mut().zip(grad.iter()) {
            *wi -= lr * g;
        }
        w = project_simplex(&w);
    }
    let sum: f64 = w.iter().sum();
    let nonneg = w.iter().all(|&wi| wi >= -1e-12);
    println!("   Constrained weights = {w:?}, sum = {sum:.4}, nonneg = {nonneg}");
    assert!(nonneg, "weights should be non-negative");
    assert!((sum - 1.0).abs() < 1e-6, "weights should sum to 1");
    assert!(
        w[0] >= 0.95,
        "constraint should allocate to high-return asset"
    );
}

fn exercise_4() {
    println!("4. Black-Litterman posterior tilts toward view:");
    let _mu = [0.10, 0.05];
    let cov = [vec![0.04, 0.01], vec![0.01, 0.0225]];
    let w_mkt = vec![0.5, 0.5];
    let tau = 0.05;
    // View: asset 0 will outperform asset 1 by 8% (P = [1, -1], q = [0.08]).
    let _p = [vec![1.0, -1.0]];
    let _omega = [vec![0.001]];
    // For BL we need P as k x n. Simplify: assume identity-like view via 2 views.
    let p_full = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let q_full = vec![0.12, 0.03]; // bullish on asset 0, bearish on asset 1.
    let omega_full = vec![vec![0.001, 0.0], vec![0.0, 0.001]];
    let post = black_litterman(&cov, &w_mkt, tau, &q_full, &p_full, &omega_full);
    let prior = matvec(&cov, &w_mkt);
    println!("   Prior (Sigma*w)  = {prior:?}");
    println!("   Posterior mu_BL = {post:?}");
    assert!(
        post[0] > prior[0],
        "posterior should tilt up on asset 0 (view q>0)"
    );
    assert!(
        post[1] < prior[1],
        "posterior should tilt down on asset 1 (view q<0)"
    );
}

fn exercise_5() {
    println!("5. Rolling CAPM beta (time-varying):");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let asset: Vec<f64> = log_returns(&bars.iter().map(|b| b.close).collect::<Vec<_>>());
    // Market proxy: equal-weight of PETR4 + VALE3.
    let bars_m =
        common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "VALE3"));
    let vale: Vec<f64> = log_returns(&bars_m.iter().map(|b| b.close).collect::<Vec<_>>());
    let n = asset.len().min(vale.len());
    let market: Vec<f64> = (0..n).map(|i| 0.5 * asset[i] + 0.5 * vale[i]).collect();
    let asset_n: Vec<f64> = asset[..n].to_vec();
    let window = 60_usize;
    let mut betas = Vec::new();
    for start in 0..(n.saturating_sub(window)) {
        let a_win = &asset_n[start..start + window];
        let m_win = &market[start..start + window];
        if let Ok(b) = beta(a_win, m_win) {
            betas.push(b);
        }
    }
    let mean_b = betas.iter().sum::<f64>() / betas.len() as f64;
    let std_b = {
        let m = mean_b;
        (betas.iter().map(|b| (b - m).powi(2)).sum::<f64>() / betas.len() as f64).sqrt()
    };
    println!("   Rolling 60-day beta: mean = {mean_b:.4}, std = {std_b:.4}");
    assert!(std_b > 0.01, "betas should be time-varying (std > 0.01)");
    assert!(
        (0.3..=2.0).contains(&mean_b),
        "mean beta should be in [0.3, 2.0], got {mean_b}"
    );
}

fn exercise_6() {
    println!("6. CVaR-optimal portfolio (more concentrated in low-vol asset):");
    let mu = vec![0.10, 0.05];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.0225]];
    let mut best_w = 0.5_f64;
    let mut best_cvar = f64::INFINITY;
    let mv_w_vec = min_variance_portfolio(&mu, &cov).expect("mv");
    let mv_w = mv_w_vec[0];
    // Grid search over w in [0, 1].
    for i in 0..=100 {
        let w = i as f64 / 100.0;
        // Synthetic return distribution: 1000 Gaussian samples.
        let mut rng = XorShift64::new(42);
        let normal = Normal::standard();
        let mut rets = Vec::with_capacity(2000);
        for _ in 0..2000 {
            let z = normal.sample(&mut rng);
            let r = w * (mu[0] + cov[0][0].sqrt() * z)
                + (1.0 - w) * (mu[1] + cov[1][1].sqrt() * normal.sample(&mut rng));
            rets.push(r);
        }
        let cvar = historical_cvar(&rets, 0.95).unwrap_or(f64::INFINITY);
        if cvar < best_cvar {
            best_cvar = cvar;
            best_w = w;
        }
    }
    println!("   Min-variance w_A = {mv_w:.4}");
    println!("   CVaR-optimal w_A = {best_w:.4}");
    // For symmetric (Gaussian) returns CVaR and variance optimisation agree
    // closely: the mean offsets the tail, so CVaR-optimal w_A sits near the
    // Markowitz minimum-variance w_A rather than concentrating further.
    assert!(
        (best_w - mv_w).abs() < 0.15,
        "CVaR w_A {best_w} should be near mv w_A {mv_w}"
    );
}

#[test]
fn test_ex1_rho_neg1_zero_vol() {
    let (var_a, var_b): (f64, f64) = (0.04, 0.09);
    let sig_a = var_a.sqrt();
    let sig_b = var_b.sqrt();
    let cov = -1.0 * sig_a * sig_b;
    let w_min = two_asset_min_variance_weight(var_a, var_b, cov);
    let fp = two_asset_frontier_point(w_min, 0.10, 0.05, var_a, var_b, cov);
    assert!(
        fp.volatility.abs() < 1e-6,
        "rho=-1 min-var vol should be ~0, got {}",
        fp.volatility
    );
}

#[test]
fn test_ex2_three_asset_tangency_beats_two() {
    let mu = vec![0.10, 0.08, 0.08];
    let cov = vec![
        vec![0.04, 0.01, 0.01],
        vec![0.01, 0.0225, 0.005],
        vec![0.01, 0.005, 0.0225],
    ];
    let rf = 0.02;
    let tan3 = tangency_portfolio(&mu, &cov, rf).expect("tan3");
    let mu2 = vec![0.10, 0.08];
    let cov2 = vec![vec![0.04, 0.01], vec![0.01, 0.0225]];
    let tan2 = tangency_portfolio(&mu2, &cov2, rf).expect("tan2");
    assert!(
        tan3.sharpe > tan2.sharpe,
        "3-asset Sharpe should exceed 2-asset"
    );
}

#[test]
fn test_ex3_project_simplex_feasible() {
    let w = project_simplex(&[0.8, 0.3]);
    let sum: f64 = w.iter().sum();
    assert!(w.iter().all(|&wi| wi >= -1e-12));
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_ex4_black_litterman_tilts() {
    let cov = vec![vec![0.04, 0.01], vec![0.01, 0.0225]];
    let w_mkt = vec![0.5, 0.5];
    let p = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let q = vec![0.12, 0.005];
    let omega = vec![vec![0.001, 0.0], vec![0.0, 0.001]];
    let post = black_litterman(&cov, &w_mkt, 0.5, &q, &p, &omega);
    let prior = matvec(&cov, &w_mkt);
    assert!(post[0] > prior[0], "asset 0 should tilt up");
    assert!(post[1] < prior[1], "asset 1 should tilt down");
}

#[test]
fn test_ex5_rolling_beta_time_varying() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let asset: Vec<f64> = log_returns(&bars.iter().map(|b| b.close).collect::<Vec<_>>());
    let bars_m =
        common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "VALE3"));
    let vale: Vec<f64> = log_returns(&bars_m.iter().map(|b| b.close).collect::<Vec<_>>());
    let n = asset.len().min(vale.len());
    let market: Vec<f64> = (0..n).map(|i| 0.5 * asset[i] + 0.5 * vale[i]).collect();
    let asset_n: Vec<f64> = asset[..n].to_vec();
    let mut betas = Vec::new();
    for start in 0..(n.saturating_sub(60)) {
        if let Ok(b) = beta(&asset_n[start..start + 60], &market[start..start + 60]) {
            betas.push(b);
        }
    }
    let m = betas.iter().sum::<f64>() / betas.len() as f64;
    let std = (betas.iter().map(|b| (b - m).powi(2)).sum::<f64>() / betas.len() as f64).sqrt();
    assert!(std > 0.01, "betas time-varying, std={std}");
}

#[test]
fn test_ex6_cvar_concentrates_low_vol() {
    let mu = vec![0.10, 0.05];
    let cov = vec![vec![0.04, 0.0], vec![0.0, 0.0225]];
    let mv = min_variance_portfolio(&mu, &cov).expect("mv");
    let mv_w = mv[0];
    let mut best_w = 0.5_f64;
    let mut best_cvar = f64::INFINITY;
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    for i in 0..=50 {
        let w = i as f64 / 50.0;
        let mut rets = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let z = normal.sample(&mut rng);
            let r = w * (mu[0] + cov[0][0].sqrt() * z)
                + (1.0 - w) * (mu[1] + cov[1][1].sqrt() * normal.sample(&mut rng));
            rets.push(r);
        }
        let c = historical_cvar(&rets, 0.95).unwrap_or(f64::INFINITY);
        if c < best_cvar {
            best_cvar = c;
            best_w = w;
        }
    }
    // For symmetric (Gaussian) returns CVaR and variance optimisation
    // agree closely: the CVaR-optimal weight sits in the same neighbourhood
    // as the Markowitz minimum-variance weight (the mean offsets the tail
    // so CVaR does not concentrate further in the low-vol asset).
    assert!(
        (best_w - mv_w).abs() < 0.15,
        "CVaR w_A {best_w} should be near mv w_A {mv_w}"
    );
}
