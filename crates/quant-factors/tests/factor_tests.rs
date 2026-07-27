//! TDD contract for the quant-factors crate (Phase 12).
//!
//! 15 integration tests covering: power method, deflation, PCA,
//! Fama-French 3-factor regression, and risk attribution.

// Test data is more readable with `vec![]` than with fixed-size arrays,
// even though clippy would prefer arrays for literal lists.
#![allow(clippy::useless_vec)]

use quant_factors::{
    deflate, ff3_regression, pca, pca_reconstruct, pca_transform, power_method,
    risk_attribution, top_k_eigen,
};

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[test]
fn test_power_method_identity() {
    let n = 3;
    let a: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let (lambda, v) = power_method(&a, 0, 0.0).unwrap();
    assert!((lambda - 1.0).abs() < 1e-9);
    assert!((l2(&v) - 1.0).abs() < 1e-9);
}

#[test]
fn test_power_method_diagonal() {
    let a = vec![vec![5.0, 0.0, 0.0], vec![0.0, 3.0, 0.0], vec![0.0, 0.0, 1.0]];
    let (lambda, v) = power_method(&a, 0, 0.0).unwrap();
    assert!((lambda - 5.0).abs() < 1e-6);
    assert!((v[0].abs() - 1.0).abs() < 1e-6);
}

#[test]
fn test_eigenvalues_positive() {
    let a = vec![
        vec![3.0, 1.0, 0.0],
        vec![1.0, 3.0, 0.0],
        vec![0.0, 0.0, 2.0],
    ];
    let (eigs, _vecs) = top_k_eigen(&a, 3).unwrap();
    for &lam in &eigs {
        assert!(lam > -1e-8, "non-positive eigenvalue: {lam}");
    }
}

#[test]
fn test_eigenvectors_orthonormal() {
    // For a symmetric matrix, the eigenvectors from top_k_eigen should
    // be unit vectors and (approximately) mutually orthogonal.
    let a = vec![
        vec![4.0, 1.0, 0.0],
        vec![1.0, 3.0, 1.0],
        vec![0.0, 1.0, 2.0],
    ];
    let (_eigs, vecs) = top_k_eigen(&a, 3).unwrap();
    for v in &vecs {
        assert!((l2(v) - 1.0).abs() < 1e-6, "non-unit eigenvector: {:?}", v);
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let d = dot(&vecs[i], &vecs[j]);
            assert!(d.abs() < 1e-4, "eigenvectors {i} and {j} not orthogonal (dot={d})");
        }
    }
}

#[test]
fn test_pca_variance_explained() {
    let returns = vec![
        vec![0.01, 0.02, 0.03],
        vec![0.02, 0.04, 0.06],
        vec![0.03, 0.06, 0.09],
        vec![0.04, 0.08, 0.12],
        vec![0.05, 0.10, 0.15],
    ];
    let res = pca(&returns, 3).unwrap();
    let total: f64 = res.explained_variance_ratio.iter().sum();
    assert!((total - 1.0).abs() < 1e-9, "EVR sum = {total}");
}

#[test]
fn test_pca_reconstruction_full() {
    let returns = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 3.0, 1.0],
        vec![3.0, 5.0, 2.0],
        vec![4.0, 1.0, 4.0],
        vec![5.0, 2.0, 1.0],
    ];
    let res = pca(&returns, 3).unwrap();
    let scores = pca_transform(&returns, &res.eigenvectors, &res.mean);
    let recon = pca_reconstruct(&scores, &res.eigenvectors, &res.mean);
    for (orig, rec) in returns.iter().zip(recon.iter()) {
        for (a, b) in orig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-6, "reconstruction: {a} vs {b}");
        }
    }
}

#[test]
fn test_pca_reconstruction_partial() {
    let returns = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 3.0, 1.0],
        vec![3.0, 5.0, 2.0],
        vec![4.0, 1.0, 4.0],
        vec![5.0, 2.0, 1.0],
    ];
    let res1 = pca(&returns, 1).unwrap();
    let s1 = pca_transform(&returns, &res1.eigenvectors, &res1.mean);
    let r1 = pca_reconstruct(&s1, &res1.eigenvectors, &res1.mean);
    let e1: f64 = returns
        .iter()
        .zip(r1.iter())
        .flat_map(|(o, r)| o.iter().zip(r.iter()).map(|(a, b)| (a - b).powi(2)))
        .sum();
    let res2 = pca(&returns, 2).unwrap();
    let s2 = pca_transform(&returns, &res2.eigenvectors, &res2.mean);
    let r2 = pca_reconstruct(&s2, &res2.eigenvectors, &res2.mean);
    let e2: f64 = returns
        .iter()
        .zip(r2.iter())
        .flat_map(|(o, r)| o.iter().zip(r.iter()).map(|(a, b)| (a - b).powi(2)))
        .sum();
    assert!(e1 > e2, "1-component error {e1} should exceed 2-component {e2}");
    assert!(e2 > 1e-9, "2-component error should be nonzero");
}

#[test]
fn test_ff3_single_factor_reduces() {
    // When the asset is generated from the market alone, the FF3 betas
    // on SMB and HML should be near zero and beta_mkt should match the
    // true generating coefficient. (We use independent noise for SMB and
    // HML so the design matrix is well-conditioned.)
    let mkt = vec![0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.005, 0.015, -0.01, 0.025];
    let smb = vec![0.001, -0.002, 0.0015, -0.001, 0.002, -0.0015, 0.001, -0.002, 0.0015, -0.001];
    let hml = vec![0.0015, 0.001, -0.002, -0.0015, 0.001, 0.002, -0.001, 0.0015, -0.002, 0.001];
    let alpha_true = 0.002_f64;
    let beta_mkt_true = 1.5_f64;
    let asset: Vec<f64> = (0..10)
        .map(|t| alpha_true + beta_mkt_true * mkt[t])
        .collect();
    let factors: Vec<Vec<f64>> = (0..10)
        .map(|t| vec![mkt[t], smb[t], hml[t]])
        .collect();
    let ff = ff3_regression(&asset, &factors).unwrap();
    assert!((ff.beta_mkt - beta_mkt_true).abs() < 1e-6);
    assert!(ff.beta_smb.abs() < 1e-6);
    assert!(ff.beta_hml.abs() < 1e-6);
}

#[test]
fn test_ff3_r_squared_improvement() {
    // A 3-factor model should fit at least as well as a 1-factor model
    // (R^2 is non-decreasing in the number of regressors). We compare
    // the FF3 R^2 to the squared correlation between the asset and the
    // market (the R^2 of the simple linear regression on the market alone).
    let mkt = vec![0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.005, 0.015, -0.01, 0.025];
    let smb = vec![0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.003, -0.002];
    let hml = vec![-0.001, 0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.003];
    // Generate the asset from all three factors (with a small noise) so
    // the 3-factor model has higher R^2 than the 1-factor model.
    let alpha = 0.001;
    let b_m = 1.2;
    let b_s = 0.5;
    let b_h = 0.3;
    let noise = vec![0.0001, -0.0001, 0.0002, -0.0002, 0.0001, -0.0001, 0.0002, -0.0002, 0.0001, -0.0001];
    let asset: Vec<f64> = (0..10)
        .map(|t| alpha + b_m * mkt[t] + b_s * smb[t] + b_h * hml[t] + noise[t])
        .collect();
    // 3-factor R^2 via the FF3 regression.
    let factors3: Vec<Vec<f64>> = (0..10)
        .map(|t| vec![mkt[t], smb[t], hml[t]])
        .collect();
    let ff3 = ff3_regression(&asset, &factors3).unwrap();
    // 1-factor R^2 as the squared Pearson correlation r(asset, mkt).
    let n = asset.len() as f64;
    let mean_a = asset.iter().sum::<f64>() / n;
    let mean_m = mkt.iter().sum::<f64>() / n;
    let cov: f64 = asset.iter().zip(mkt.iter()).map(|(a, m)| (a - mean_a) * (m - mean_m)).sum::<f64>() / (n - 1.0);
    let var_a: f64 = asset.iter().map(|a| (a - mean_a).powi(2)).sum::<f64>() / (n - 1.0);
    let var_m: f64 = mkt.iter().map(|m| (m - mean_m).powi(2)).sum::<f64>() / (n - 1.0);
    let r1_squared = (cov * cov) / (var_a * var_m);
    assert!(
        ff3.r_squared >= r1_squared - 1e-9,
        "FF3 R^2 {} should be >= 1-factor R^2 {}",
        ff3.r_squared,
        r1_squared
    );
}

#[test]
fn test_ff3_alpha_zero_equilibrium() {
    // When the asset is exactly generated by the 3-factor model (no
    // noise), the estimated alpha matches the true alpha and R^2 = 1.
    let mkt = vec![0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.005, 0.015, -0.01, 0.025];
    let smb = vec![0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.003, -0.002];
    let hml = vec![-0.001, 0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.003];
    let alpha_true = 0.0015;
    let b_m = 1.1;
    let b_s = 0.4;
    let b_h = -0.2;
    let asset: Vec<f64> = (0..10)
        .map(|t| alpha_true + b_m * mkt[t] + b_s * smb[t] + b_h * hml[t])
        .collect();
    let factors: Vec<Vec<f64>> = (0..10)
        .map(|t| vec![mkt[t], smb[t], hml[t]])
        .collect();
    let ff = ff3_regression(&asset, &factors).unwrap();
    assert!((ff.alpha - alpha_true).abs() < 1e-6);
    assert!((ff.beta_mkt - b_m).abs() < 1e-6);
    assert!((ff.beta_smb - b_s).abs() < 1e-6);
    assert!((ff.beta_hml - b_h).abs() < 1e-6);
    assert!((ff.r_squared - 1.0).abs() < 1e-6);
}

#[test]
fn test_systematic_plus_idio() {
    let weights = vec![0.6, 0.4];
    let loadings = vec![vec![1.2, 0.3], vec![0.8, -0.1]];
    let factor_cov = vec![vec![0.04, 0.01], vec![0.01, 0.02]];
    let resid = vec![0.002, 0.001];
    let ra = risk_attribution(&weights, &loadings, &factor_cov, &resid).unwrap();
    assert!(
        (ra.total_variance - (ra.systematic_variance + ra.idiosyncratic_variance)).abs() < 1e-12
    );
}

#[test]
fn test_factor_contribution_sum() {
    // For a diagonal factor covariance, sum(contributions) == systematic.
    let weights = vec![0.5, 0.5];
    let loadings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let factor_cov = vec![vec![0.04, 0.0], vec![0.0, 0.09]];
    let resid = vec![0.0, 0.0];
    let ra = risk_attribution(&weights, &loadings, &factor_cov, &resid).unwrap();
    let sum_contrib: f64 = ra.factor_contributions.iter().sum();
    assert!(
        (sum_contrib - ra.systematic_variance).abs() < 1e-12,
        "sum_contrib={sum_contrib} systematic={}",
        ra.systematic_variance
    );
}

#[test]
fn test_deflation_removes_component() {
    let a = vec![vec![4.0, 0.0, 0.0], vec![0.0, 3.0, 0.0], vec![0.0, 0.0, 2.0]];
    let (lambda, v) = power_method(&a, 0, 0.0).unwrap();
    assert!((lambda - 4.0).abs() < 1e-6);
    let a_def = deflate(&a, lambda, &v);
    let (lambda2, _v2) = power_method(&a_def, 0, 0.0).unwrap();
    assert!((lambda2 - 3.0).abs() < 1e-3, "second eigenvalue = {lambda2}");
}

#[test]
fn test_pca_mean_centered() {
    let returns = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let res = pca(&returns, 2).unwrap();
    assert!((res.mean[0] - 3.0).abs() < 1e-12);
    assert!((res.mean[1] - 4.0).abs() < 1e-12);
}

#[test]
fn test_factors_smoke() {
    // Every public function produces finite output on a reasonable input.
    let returns = vec![
        vec![0.01, 0.02, 0.03],
        vec![0.02, 0.01, 0.04],
        vec![-0.01, 0.0, 0.02],
        vec![0.03, 0.04, 0.01],
        vec![0.0, 0.01, 0.03],
    ];
    let res = pca(&returns, 3).unwrap();
    for &lam in &res.eigenvalues {
        assert!(lam.is_finite());
    }
    let mkt = vec![0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.005, 0.015, -0.01, 0.025];
    let smb = vec![0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.003, -0.002];
    let hml = vec![-0.001, 0.002, -0.001, 0.003, -0.002, 0.001, -0.003, 0.002, -0.001, 0.003];
    let asset: Vec<f64> = (0..10).map(|t| 0.001 + 1.1 * mkt[t] + 0.4 * smb[t]).collect();
    let factors: Vec<Vec<f64>> = (0..10).map(|t| vec![mkt[t], smb[t], hml[t]]).collect();
    let ff = ff3_regression(&asset, &factors).unwrap();
    assert!(ff.alpha.is_finite());
    assert!(ff.beta_mkt.is_finite());
    let weights = vec![0.5, 0.5];
    let loadings = vec![vec![1.0, 0.2], vec![0.8, -0.1]];
    let factor_cov = vec![vec![0.04, 0.0], vec![0.0, 0.01]];
    let resid = vec![0.001, 0.002];
    let ra = risk_attribution(&weights, &loadings, &factor_cov, &resid).unwrap();
    assert!(ra.total_variance.is_finite());
    assert!(ra.systematic_variance.is_finite());
    assert!(ra.idiosyncratic_variance.is_finite());
}