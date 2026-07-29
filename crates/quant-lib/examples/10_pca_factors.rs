//! Example 10: Factor Models - PCA and Fama-French 3-Factor
//!
//! Level: Intermediate
//!
//! Runs Principal Component Analysis on a matrix of 4 Brazilian stock
//! returns (PETR4, VALE3, ITSA4, BBDC4) to extract the systematic
//! components, and then fits a Fama-French 3-factor regression on a
//! synthetic asset using the first PC as a "market" proxy and two
//! additional synthetic factors (SMB, HML).
//!
//! Uses `quant-factors` (pca, pca_transform, ff3_regression) and
//! `quant-portfolio` (beta) for cross-checking.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 10_pca_factors
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::factors::{pca_reconstruct, pca_transform};
use quant_lib::prelude::*;

fn main() {
    let symbols = ["PETR4", "VALE3", "ITSA4", "BBDC4"];
    let bars: Vec<Vec<common::OhlcvBar>> = symbols
        .iter()
        .map(|s| common::load_json_ohlcv(&common::b3_json_path(env!("CARGO_MANIFEST_DIR"), s)))
        .collect();

    println!("=== Example 10: PCA and Fama-French ===");
    for (s, b) in symbols.iter().zip(bars.iter()) {
        println!("  {s}: {} bars", b.len());
    }

    // Align on the common date prefix.
    let n = bars.iter().map(|b| b.len()).min().unwrap();
    let closes: Vec<Vec<f64>> = bars
        .iter()
        .map(|b| b.iter().take(n).map(|x| x.close).collect())
        .collect();
    let rets_series: Vec<Vec<f64>> = closes.iter().map(|c| log_returns(c)).collect();
    let m = rets_series[0].len();

    // Build the T x N returns matrix (rows = time, cols = assets).
    let returns: Vec<Vec<f64>> = (0..m)
        .map(|t| rets_series.iter().map(|r| r[t]).collect())
        .collect();

    // 1. PCA: keep all 4 components.
    let res = pca(&returns, 4).expect("PCA");
    println!("\nPCA eigenvalues (descending):");
    for (i, &lam) in res.eigenvalues.iter().enumerate() {
        println!(
            "  PC{}: lambda = {:.6}, explained = {:.2}%, cumul = {:.2}%",
            i + 1,
            lam,
            res.explained_variance_ratio[i] * 100.0,
            res.cumulative_variance[i] * 100.0
        );
    }
    println!(
        "\nFirst principal component (loadings on {} assets):",
        symbols.join(", ")
    );
    for (s, w) in symbols.iter().zip(res.eigenvectors[0].iter()) {
        println!("  {s}: {w:+.4}");
    }

    // 2. Transform: project the returns onto the PCs.
    let transformed = pca_transform(&returns, &res.eigenvectors, &res.mean);
    println!(
        "\nPCA transform: {} time steps x {} components",
        transformed.len(),
        transformed[0].len()
    );

    // 3. Reconstruct and measure residual variance.
    let recon = pca_reconstruct(&transformed, &res.eigenvectors, &res.mean);
    let recon_err: f64 = (0..m)
        .map(|t| {
            (0..symbols.len())
                .map(|j| (recon[t][j] - returns[t][j]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / m as f64;
    println!("Reconstruction MSE (4 PCs, should be ~0): {recon_err:.2e}");

    // 4. Fama-French 3-factor regression on a synthetic asset.
    //    Build a synthetic asset: 0.0001 + 1.2*PC1 + 0.4*PC2 + 0.1*noise.
    let mut rng = XorShift64::new(99);
    let normal = Normal::standard();
    let asset: Vec<f64> = (0..m)
        .map(|t| {
            0.0001
                + 1.2 * transformed[t][0]
                + 0.4 * transformed[t][1]
                + 0.0005 * normal.sample(&mut rng)
        })
        .collect();

    // Factors: [PC1, PC2, PC3] as [Mkt-Rf, SMB, HML].
    let factors: Vec<Vec<f64>> = (0..m)
        .map(|t| vec![transformed[t][0], transformed[t][1], transformed[t][2]])
        .collect();

    let ff = ff3_regression(&asset, &factors).expect("FF3 regression");
    println!("\nFama-French 3-factor regression (synthetic asset):");
    println!("  alpha (intercept) = {:+.6}", ff.alpha);
    println!("  beta_mkt = {:+.4}", ff.beta_mkt);
    println!("  beta_smb = {:+.4}", ff.beta_smb);
    println!("  beta_hml = {:+.4}", ff.beta_hml);
    println!("  R-squared = {:.4}", ff.r_squared);
    println!("  residual var = {:.6}", ff.residual_var);

    // The synthetic asset was built with 1.2*PC1 + 0.4*PC2, so beta_mkt ~ 1.2,
    // beta_smb ~ 0.4, beta_hml ~ 0, alpha ~ 0.0001.
    assert!(
        (ff.beta_mkt - 1.2).abs() < 0.1,
        "beta_mkt should be ~1.2, got {}",
        ff.beta_mkt
    );
    assert!(
        (ff.beta_smb - 0.4).abs() < 0.1,
        "beta_smb should be ~0.4, got {}",
        ff.beta_smb
    );
    println!("\nRecovered betas match the synthetic construction (beta_mkt~1.2, beta_smb~0.4).");
}
