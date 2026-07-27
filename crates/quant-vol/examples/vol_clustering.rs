//! Volatility clustering demo.
//!
//! Demonstrates that GARCH(1,1) captures clustered volatility where
//! constant-volatility (sample variance) models fail:
//! 1. Simulate two regimes: calm returns for 1000 periods, then a volatility
//!    outbreak for 500 periods, then calm again for 500 periods.
//! 2. Fit GARCH(1,1) to the full series.
//! 3. Compare Gaussian log-likelihoods of:
//!    - Constant-volatility model (sigma^2 = sample variance for all t)
//!    - GARCH(1,1) with time-varying conditional variances
//! 4. Print the conditional variance at the regime boundaries to show GARCH
//!    tracks the clustering.

use quant_core::{Distribution, Normal, XorShift64};
use quant_vol::GarchModel;

fn main() {
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();

    // Two-regime DGP: calm (sigma^2 = 0.0001) then turbulent (sigma^2 = 0.01).
    let n_calm = 1000;
    let n_shock = 500;
    let n_back = 500;
    let sigma2_calm = 0.0001_f64;
    let sigma2_shock = 0.01_f64;

    let mut returns = Vec::with_capacity(n_calm + n_shock + n_back);
    for _ in 0..n_calm {
        returns.push(normal.sample(&mut rng) * sigma2_calm.sqrt());
    }
    for _ in 0..n_shock {
        returns.push(normal.sample(&mut rng) * sigma2_shock.sqrt());
    }
    for _ in 0..n_back {
        returns.push(normal.sample(&mut rng) * sigma2_calm.sqrt());
    }

    println!("==============================");
    println!("Volatility Clustering Demo");
    println!("==============================");
    println!();
    println!("Regimes: calm({n_calm}) -> shock({n_shock}) -> calm({n_back})");
    println!("  Calm sigma^2  = {:.6}", sigma2_calm);
    println!("  Shock sigma^2 = {:.6}", sigma2_shock);
    println!();

    // Constant-volatility model: sigma^2 = sample variance for all t.
    let sample_var: f64 = {
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        returns.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64
    };
    let ln_2pi = 1.8378770664093453_f64;
    let ll_constant: f64 = returns
        .iter()
        .map(|&r| -0.5 * (ln_2pi + sample_var.ln() + r * r / sample_var))
        .sum();

    // GARCH(1,1) fit.
    let garch = GarchModel::fit(&returns, 1, 1).unwrap();
    let ll_garch = garch.log_likelihood(&returns);
    let sigma2_path = garch.conditional_variances(&returns);

    println!("Constant-volatility model:");
    println!("  Sample variance = {:.6}", sample_var);
    println!("  Log-likelihood  = {:.2}", ll_constant);
    println!();
    println!("GARCH(1,1):");
    println!("  omega = {:.6}", garch.omega);
    println!("  alpha = {:.6}", garch.alphas[0]);
    println!("  beta  = {:.6}", garch.betas[0]);
    println!("  Persistence = {:.4}", garch.persistence());
    println!("  Log-likelihood = {:.2}", ll_garch);
    println!();

    // Conditional variance at regime boundaries.
    println!("Conditional variance path (GARCH) at key points:");
    println!("  t=0    (calm start):  {:.6}", sigma2_path[0]);
    println!("  t={n_calm} (shock start): {:.6}", sigma2_path[n_calm]);
    println!(
        "  t={} (shock mid):   {:.6}",
        n_calm + n_shock / 2,
        sigma2_path[n_calm + n_shock / 2]
    );
    println!(
        "  t={} (calm resume): {:.6}",
        n_calm + n_shock,
        sigma2_path[n_calm + n_shock]
    );
    println!(
        "  t={} (end):         {:.6}",
        returns.len() - 1,
        sigma2_path[returns.len() - 1]
    );
    println!();

    // LL comparison.
    println!("Log-likelihood comparison:");
    println!("  Constant vol: {:.2}", ll_constant);
    println!("  GARCH(1,1):   {:.2}", ll_garch);
    if ll_garch > ll_constant {
        println!(
            "  -> GARCH beats constant-vol by {:.1} LL points",
            ll_garch - ll_constant
        );
        println!("  Volatility clustering is captured by GARCH but missed by");
        println!("  the constant-volatility model, which over-estimates variance");
        println!("  in calm periods and under-estimates it in turbulent periods.");
    }
    println!();
    println!("==============================");
}
