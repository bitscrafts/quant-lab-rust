//! Volatility models demo.
//!
//! Compares EWMA, ARCH(1), and GARCH(1,1) on simulated GARCH(1,1) returns:
//! 1. Simulate 2000 returns from a known GARCH(1,1) data-generating process.
//! 2. Fit EWMA (lambda = 0.94), ARCH(1), and GARCH(1,1) to the same series.
//! 3. Print fitted parameters, persistence, long-run variance, and half-life.
//! 4. Compare Gaussian log-likelihoods across the three models.

use quant_core::{Distribution, Normal, XorShift64};
use quant_vol::{ArchModel, GarchModel, ewma_vol};

fn main() {
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();

    // True GARCH(1,1) data-generating process.
    let true_omega = 0.01_f64;
    let true_alpha = 0.08_f64;
    let true_beta = 0.90_f64;
    let true_pers = true_alpha + true_beta;
    let true_lr = true_omega / (1.0 - true_pers);

    let n = 2000;
    let mut sigma2 = true_lr;
    let mut returns = Vec::with_capacity(n);
    for _ in 0..n {
        let z = normal.sample(&mut rng);
        let r = sigma2.sqrt() * z;
        returns.push(r);
        sigma2 = true_omega + true_alpha * r * r + true_beta * sigma2;
    }

    println!("==============================");
    println!("Volatility Models Demo");
    println!("==============================");
    println!();
    println!("DGP: GARCH(1,1) with omega={true_omega}, alpha={true_alpha}, beta={true_beta}");
    println!("True persistence:     {:.4}", true_pers);
    println!("True long-run var:    {:.6}", true_lr);
    println!("Series length:       {n}");
    println!();

    // EWMA (RiskMetrics lambda = 0.94).
    let ewma_sigma2 = ewma_vol(&returns, 0.94).unwrap();
    let ewma_mean: f64 = ewma_sigma2[250..].iter().sum::<f64>() / (ewma_sigma2.len() - 250) as f64;
    println!("EWMA (lambda=0.94):");
    println!("  Mean conditional var (post burn-in): {:.6}", ewma_mean);
    println!();

    // ARCH(1).
    let arch = ArchModel::fit(&returns, 1).unwrap();
    let arch_ll = arch.log_likelihood(&returns);
    let arch_lr = arch.long_run_variance();
    println!("ARCH(1):");
    println!("  omega = {:.6}", arch.omega);
    println!("  alpha = {:.6}", arch.alphas[0]);
    println!("  Long-run var = {:.6}", arch_lr);
    println!("  Log-likelihood = {:.2}", arch_ll);
    println!();

    // GARCH(1,1).
    let garch = GarchModel::fit(&returns, 1, 1).unwrap();
    let garch_ll = garch.log_likelihood(&returns);
    let garch_pers = garch.persistence();
    let garch_lr = garch.long_run_variance();
    let garch_hl = garch.half_life();
    println!("GARCH(1,1):");
    println!("  omega = {:.6}", garch.omega);
    println!("  alpha = {:.6}", garch.alphas[0]);
    println!("  beta  = {:.6}", garch.betas[0]);
    println!("  Persistence = {:.4}  (true {true_pers:.4})", garch_pers);
    println!("  Long-run var = {:.6}  (true {true_lr:.6})", garch_lr);
    println!("  Half-life    = {:.1} periods", garch_hl);
    println!("  Log-likelihood = {:.2}", garch_ll);
    println!();

    // Model comparison.
    println!("Model comparison (Gaussian log-likelihood, higher is better):");
    println!("  EWMA  (no fit):  n/a (fixed lambda)");
    println!("  ARCH(1):         {:.2}", arch_ll);
    println!("  GARCH(1,1):      {:.2}", garch_ll);
    if garch_ll > arch_ll {
        println!(
            "  -> GARCH(1,1) beats ARCH(1) by {:.1} LL points",
            garch_ll - arch_ll
        );
    }
    println!();
    println!("==============================");
}
