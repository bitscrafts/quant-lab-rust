//! Fama-French 3-factor demo: simulate factor returns, generate an
//! asset from the 3-factor model (with noise), regress it on the
//! factors, and compare the R^2 to the single-factor (CAPM) R^2.

use quant_factors::ff3_regression;

fn main() {
    // Simulated 3-factor returns (100 observations).
    // Mkt-Rf, SMB, HML with mild correlations.
    let n = 100;
    let mkt: Vec<f64> = (0..n)
        .map(|t| 0.001 + 0.01 * ((t as f64) * 0.3).sin() + 0.003 * (t as f64 / 7.0).cos())
        .collect();
    let smb: Vec<f64> = (0..n)
        .map(|t| 0.0005 + 0.006 * ((t as f64) * 0.5).sin() + 0.002 * (t as f64 / 11.0).cos())
        .collect();
    let hml: Vec<f64> = (0..n)
        .map(|t| -0.0003 + 0.004 * ((t as f64) * 0.2).cos() + 0.0015 * (t as f64 / 13.0).sin())
        .collect();

    // True DGP: alpha=0.001, beta_mkt=1.2, beta_smb=0.4, beta_hml=-0.3
    // plus idiosyncratic noise.
    let alpha_true = 0.001_f64;
    let b_m = 1.2_f64;
    let b_s = 0.4_f64;
    let b_h = -0.3_f64;
    let noise: Vec<f64> = (0..n)
        .map(|t| 0.002 * ((t as f64) * 1.7).sin() * ((t as f64) * 0.05).cos())
        .collect();
    let asset: Vec<f64> = (0..n)
        .map(|t| alpha_true + b_m * mkt[t] + b_s * smb[t] + b_h * hml[t] + noise[t])
        .collect();

    let factors: Vec<Vec<f64>> = (0..n).map(|t| vec![mkt[t], smb[t], hml[t]]).collect();

    // FF3 regression.
    let ff = ff3_regression(&asset, &factors).unwrap();
    println!("=== Fama-French 3-Factor Regression ===\n");
    println!("True DGP: alpha={alpha_true}, beta_mkt={b_m}, beta_smb={b_s}, beta_hml={b_h}");
    println!("Estimated:");
    println!("  alpha    = {:.6}", ff.alpha);
    println!("  beta_mkt = {:.6}", ff.beta_mkt);
    println!("  beta_smb = {:.6}", ff.beta_smb);
    println!("  beta_hml = {:.6}", ff.beta_hml);
    println!("  R^2      = {:.6}", ff.r_squared);
    println!("  resid var= {:.6e}", ff.residual_var);

    // Single-factor (CAPM) R^2 = squared correlation between asset and market.
    let mean_a = asset.iter().sum::<f64>() / n as f64;
    let mean_m = mkt.iter().sum::<f64>() / n as f64;
    let cov: f64 = asset
        .iter()
        .zip(mkt.iter())
        .map(|(a, m)| (a - mean_a) * (m - mean_m))
        .sum::<f64>()
        / (n - 1) as f64;
    let var_a: f64 = asset.iter().map(|a| (a - mean_a).powi(2)).sum::<f64>() / (n - 1) as f64;
    let var_m: f64 = mkt.iter().map(|m| (m - mean_m).powi(2)).sum::<f64>() / (n - 1) as f64;
    let r1_squared = (cov * cov) / (var_a * var_m);

    println!("\n=== Comparison: FF3 vs CAPM ===\n");
    println!("CAPM (1-factor) R^2 = {:.6}", r1_squared);
    println!("FF3  (3-factor) R^2 = {:.6}", ff.r_squared);
    println!(
        "Improvement         = {:.6} ({}%)",
        ff.r_squared - r1_squared,
        ((ff.r_squared - r1_squared) / r1_squared * 100.0)
    );
    println!("\nThe 3-factor model captures the size and value exposures that");
    println!("the single-factor CAPM attributes to idiosyncratic noise, so");
    println!("the R^2 improvement quantifies the incremental explanatory");
    println!("power of SMB and HML beyond the market.");

    // Risk attribution: the systematic variance is beta' * Sigma_F * beta,
    // the idiosyncratic is the residual variance from the regression.
    let factor_cov: Vec<Vec<f64>> = {
        let mut cov = vec![vec![0.0_f64; 3]; 3];
        let means = [
            mkt.iter().sum::<f64>() / n as f64,
            smb.iter().sum::<f64>() / n as f64,
            hml.iter().sum::<f64>() / n as f64,
        ];
        let cols = [&mkt, &smb, &hml];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] = (0..n)
                    .map(|t| (cols[i][t] - means[i]) * (cols[j][t] - means[j]))
                    .sum::<f64>()
                    / (n - 1) as f64;
            }
        }
        cov
    };
    let loadings = vec![vec![ff.beta_mkt, ff.beta_smb, ff.beta_hml]];
    let resid = vec![ff.residual_var];
    let weights = vec![1.0];
    let ra = quant_factors::risk_attribution(&weights, &loadings, &factor_cov, &resid).unwrap();
    println!("\n=== Risk Attribution ===\n");
    println!("Total variance      = {:.6e}", ra.total_variance);
    println!(
        "Systematic         = {:.6e} ({:.1}%)",
        ra.systematic_variance,
        ra.systematic_variance / ra.total_variance * 100.0
    );
    println!(
        "Idiosyncratic       = {:.6e} ({:.1}%)",
        ra.idiosyncratic_variance,
        ra.idiosyncratic_variance / ra.total_variance * 100.0
    );
    println!(
        "Factor contributions: Mkt={:.6e}, SMB={:.6e}, HML={:.6e}",
        ra.factor_contributions[0], ra.factor_contributions[1], ra.factor_contributions[2]
    );
}
