//! CAPM regression: beta, alpha, and the security market line.
//!
//! Generates a synthetic asset whose returns are a noisy linear function of a
//! market index, then regresses the asset on the market to recover the CAPM
//! beta and alpha. Demonstrates the security market line and Jensen's alpha.
//!
//! Run with: `cargo run -p quant-portfolio --example capm`.

use quant_core::{XorShift64, Normal, Distribution};
use quant_portfolio::{alpha, beta, sml};

fn main() {
    // Generate synthetic returns: R_asset = rf + beta * (R_market - rf) + eps.
    let rf = 0.02;
    let true_beta = 1.2_f64;
    let n = 60;
    let mut rng = XorShift64::new(42);
    let normal = Normal::new(0.0, 1.0);
    let market_mean = 0.08;
    let market_std = 0.04;
    let noise_std = 0.02;

    let mut market = Vec::with_capacity(n);
    let mut asset = Vec::with_capacity(n);
    for _ in 0..n {
        let eps_m = normal.sample(&mut rng) * market_std;
        let r_m = market_mean + eps_m;
        let eps_a = normal.sample(&mut rng) * noise_std;
        let r_a = rf + true_beta * (r_m - rf) + eps_a;
        market.push(r_m);
        asset.push(r_a);
    }

    // Compute CAPM beta and alpha.
    let beta_hat = beta(&asset, &market).unwrap();
    let alpha_hat = alpha(&asset, &market, rf).unwrap();
    let mean_m: f64 = market.iter().sum::<f64>() / n as f64;
    let mean_a: f64 = asset.iter().sum::<f64>() / n as f64;

    println!("CAPM regression (synthetic noisy linear asset)");
    println!("=============================================");
    println!("Sample size: {n} observations");
    println!("Risk-free rate: {rf}");
    println!("True beta (data generating process): {true_beta}");
    println!();
    println!("Market index:");
    println!("  sample mean return = {:.6}", mean_m);
    println!("  sample std         = {:.6}", market_std);
    println!();
    println!("Asset:");
    println!("  sample mean return = {:.6}", mean_a);
    println!("  noise std          = {:.6}", noise_std);
    println!();
    println!("CAPM estimates:");
    println!("  beta_hat  = {:.6}  (true {true_beta})", beta_hat);
    println!("  alpha_hat = {:.6}", alpha_hat);
    println!();
    println!("Security market line: E[R_i] = rf + beta_i * (E[R_m] - rf)");
    println!("  SML-predicted return at beta_hat = {:.6}", sml(beta_hat, mean_m, rf));
    println!("  realised mean return              = {:.6}", mean_a);
    println!("  Jensen's alpha (gap)              = {:.6}", alpha_hat);
    println!();

    // Sweep beta along the SML.
    println!("Security market line (E[R_m] = {:.6}, rf = {:.6}):", mean_m, rf);
    println!("  beta    SML_return");
    for i in 0..=10 {
        let b = i as f64 * 0.25;
        println!("  {:>4.2}   {:>6.4}", b, sml(b, mean_m, rf));
    }
}