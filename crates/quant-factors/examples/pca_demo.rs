//! PCA demo: run principal component analysis on synthetic correlated
//! returns, print eigenvalues, explained variance, and the
//! reconstruction error for 1, 2, and 3 retained components.

use quant_factors::{pca, pca_reconstruct, pca_transform};

fn main() {
    // Synthetic 3-asset returns with three independent sources of
    // variance: a dominant market factor, a sector factor, and
    // idiosyncratic noise. All three eigenvalues should be positive
    // and well-separated.
    let returns: Vec<Vec<f64>> = vec![
        vec![0.010, 0.005, -0.003],
        vec![0.015, 0.008, 0.002],
        vec![-0.005, -0.003, 0.004],
        vec![0.020, 0.010, -0.001],
        vec![-0.010, -0.005, 0.003],
        vec![0.005, 0.002, -0.002],
        vec![0.025, 0.012, 0.001],
        vec![-0.015, -0.008, -0.003],
        vec![0.012, 0.006, 0.004],
        vec![0.008, 0.004, -0.001],
        vec![-0.002, -0.001, 0.005],
        vec![0.018, 0.009, 0.0],
    ];

    println!("=== PCA on 12 observations x 3 assets ===\n");
    println!("Mean: {:?}", returns[0].iter().enumerate().map(|(j, _)| {
        returns.iter().map(|r| r[j]).sum::<f64>() / returns.len() as f64
    }).collect::<Vec<_>>());

    // Full PCA (all 3 components).
    let res_full = pca(&returns, 3).unwrap();
    println!("\nEigenvalues (descending):");
    for (i, &lam) in res_full.eigenvalues.iter().enumerate() {
        println!("  PC{}: {:.8}  (EVR = {:.4}, cum = {:.4})",
            i + 1, lam, res_full.explained_variance_ratio[i], res_full.cumulative_variance[i]);
    }

    // Top eigenvector.
    println!("\nTop eigenvector (PC1): {:?}", res_full.eigenvectors[0]);

    // Reconstruction error for k = 1, 2, 3 components.
    println!("\nReconstruction error (SSE) vs number of components:");
    for k in 1..=3 {
        let res = pca(&returns, k).unwrap();
        let scores = pca_transform(&returns, &res.eigenvectors, &res.mean);
        let recon = pca_reconstruct(&scores, &res.eigenvectors, &res.mean);
        let sse: f64 = returns
            .iter()
            .zip(recon.iter())
            .flat_map(|(o, r)| o.iter().zip(r.iter()).map(|(a, b)| (a - b).powi(2)))
            .sum();
        let total_var: f64 = returns
            .iter()
            .flat_map(|r| r.iter().map(|x| (x - res.mean[r.iter().position(|_| true).unwrap_or(0)]).powi(2)))
            .sum();
        let _ = total_var;
        println!("  k={}: SSE = {:.2e}  (EV captured = {:.4})", k, sse, res.cumulative_variance.last().unwrap());
    }

    // Show the factor scores (projections) for the first 3 observations.
    println!("\nFactor scores (first 3 observations, full PCA):");
    let scores = pca_transform(&returns, &res_full.eigenvectors, &res_full.mean);
    for (t, row) in scores.iter().take(3).enumerate() {
        println!("  t={}: {:?}", t, row);
    }
}