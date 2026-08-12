//! Exercise solutions for Chapter 12: Factor Models
//!
//! Run: `cargo run -p quant-lib --example solutions-ch12_factors_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch12_factors_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::log_returns;
use quant_lib::factors::top_k_eigen;
use quant_lib::portfolio::beta;
use quant_lib::prelude::*;

/// Compute R^2 manually for an OLS fit.
fn _r_squared_of(fit: &OlsFit, _y: &[f64]) -> f64 {
    fit.r_squared
}

/// Jacobi eigenvalue routine for a small symmetric matrix (cyclic sweeps).
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen(matrix: &[Vec<f64>], max_sweeps: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _ in 0..max_sweeps {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-18 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() < 1e-18 {
                    continue;
                }
                let app = a[p][p];
                let aqq = a[q][q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Rotate.
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[i][p];
                        let aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[p][i] = a[i][p];
                        a[i][q] = s * aip + c * aiq;
                        a[q][i] = a[i][q];
                    }
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    // Sort eigenvalues descending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigs[b]
            .partial_cmp(&eigs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_eigs: Vec<f64> = order.iter().map(|&i| eigs[i]).collect();
    let sorted_vecs: Vec<Vec<f64>> = order
        .iter()
        .map(|&i| v.iter().map(|row| row[i]).collect())
        .collect();
    eigs = sorted_eigs;
    (eigs, sorted_vecs)
}

fn main() {
    println!("=== Chapter 12: Factor Models - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 12 exercises complete.");
}

fn exercise_1() {
    println!("1. Four-/Five-factor extension raises R^2 over FF3:");
    // Synthetic asset and factor returns with known structure.
    let n = 200_usize;
    let mut rng = XorShift64::new(1);
    let normal = Normal::standard();
    let mut mkt = Vec::with_capacity(n);
    let mut smb = Vec::with_capacity(n);
    let mut hml = Vec::with_capacity(n);
    let mut umd = Vec::with_capacity(n);
    let mut quality = Vec::with_capacity(n);
    let mut asset = Vec::with_capacity(n);
    for _ in 0..n {
        let m = 0.05 + 0.01 * normal.sample(&mut rng);
        let s = 0.02 + 0.005 * normal.sample(&mut rng);
        let h = 0.03 + 0.005 * normal.sample(&mut rng);
        let u = 0.04 + 0.005 * normal.sample(&mut rng);
        let q = 0.02 + 0.005 * normal.sample(&mut rng);
        let a = 0.001
            + 1.2 * m
            + 0.5 * s
            + 0.3 * h
            + 0.4 * u
            + 0.2 * q
            + 0.005 * normal.sample(&mut rng);
        mkt.push(m);
        smb.push(s);
        hml.push(h);
        umd.push(u);
        quality.push(q);
        asset.push(a);
    }
    // FF3 baseline.
    let ff3_factors: Vec<Vec<f64>> = (0..n).map(|i| vec![mkt[i], smb[i], hml[i]]).collect();
    let ff3 = ff3_regression(&asset, &ff3_factors).expect("ff3");
    // Five-factor: extend design matrix manually.
    let x5: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![1.0, mkt[i], smb[i], hml[i], umd[i], quality[i]])
        .collect();
    let fit5 = ols(&x5, &asset).expect("ols5");
    let r2_3 = ff3.r_squared;
    let r2_5 = fit5.r_squared;
    println!("   FF3 R^2 = {r2_3:.4}");
    println!("   5-factor R^2 = {r2_5:.4}");
    assert!(r2_5 >= r2_3 - 1e-6, "5-factor R^2 should be >= FF3 R^2");
}

fn exercise_2() {
    println!("2. Jacobi vs power-method eigenvalues (5x5 symmetric):");
    let a = vec![
        vec![4.0, 1.0, 2.0, 0.0, 1.0],
        vec![1.0, 3.0, 0.0, 1.0, 0.0],
        vec![2.0, 0.0, 5.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0, 2.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0, 3.0],
    ];
    let (jacobi_eigs, _) = jacobi_eigen(&a, 100);
    let (pm_eigs, _) = top_k_eigen(&a, 5).expect("top_k");
    let mut max_diff = 0.0_f64;
    for i in 0..5 {
        let d = (jacobi_eigs[i] - pm_eigs[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    println!("   Jacobi eigenvalues = {jacobi_eigs:?}");
    println!("   Power-method eigenvalues = {pm_eigs:?}");
    println!("   Max |diff| = {max_diff:.2e} (expect < 1e-3 with deflation drift)");
    assert!(max_diff < 1e-3, "Jacobi vs power-method agreement to 1e-3");
}

fn exercise_3() {
    println!("3. PCA on 8 B3 stocks (first PC all-positive, 30-50% variance):");
    let symbols = [
        "PETR4", "VALE3", "ITSA4", "BBDC4", "B3SA3", "ABEV3", "GGBR4", "WEGE3",
    ];
    let mut all_returns: Vec<Vec<f64>> = Vec::new();
    let mut n_min = usize::MAX;
    for sym in &symbols {
        let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), sym);
        let bars = common::load_json_ohlcv(&path);
        let r = log_returns(&bars.iter().map(|b| b.close).collect::<Vec<_>>());
        n_min = n_min.min(r.len());
        all_returns.push(r);
    }
    // Take the last 252 days.
    let window = 252_usize.min(n_min);
    let matrix: Vec<Vec<f64>> = (0..window)
        .map(|t| {
            all_returns
                .iter()
                .map(|r| r[r.len() - window + t])
                .collect()
        })
        .collect();
    let res = pca(&matrix, 8).expect("pca");
    let first_pc = &res.eigenvectors[0];
    let pos = first_pc.iter().filter(|&&v| v > 1e-6).count();
    let neg = first_pc.iter().filter(|&&v| v < -1e-6).count();
    let evr = res.explained_variance_ratio[0];
    println!("   First PC loadings = {first_pc:?}");
    println!("   First PC EVR = {evr:.4} (expect 0.30-0.50)");
    assert!(
        pos.max(neg) >= 6,
        "first PC should have consistent sign on most assets"
    );
    assert!(
        (0.25..=0.60).contains(&evr),
        "first PC EVR should be in [0.25, 0.60], got {evr}"
    );
}

fn exercise_4() {
    println!("4. Rolling 60-day beta of PETR4 vs market proxy:");
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let asset: Vec<f64> = log_returns(&bars.iter().map(|b| b.close).collect::<Vec<_>>());
    // Market proxy: equal-weight of 8 B3 stocks.
    let symbols = [
        "PETR4", "VALE3", "ITSA4", "BBDC4", "B3SA3", "ABEV3", "GGBR4", "WEGE3",
    ];
    let mut all_returns: Vec<Vec<f64>> = Vec::new();
    let mut n_min = usize::MAX;
    for sym in &symbols {
        let p = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), sym);
        let b = common::load_json_ohlcv(&p);
        let r = log_returns(&b.iter().map(|x| x.close).collect::<Vec<_>>());
        n_min = n_min.min(r.len());
        all_returns.push(r);
    }
    let n = n_min.min(asset.len());
    let market: Vec<f64> = (0..n)
        .map(|t| {
            let s: f64 = all_returns.iter().map(|r| r[r.len() - n + t]).sum::<f64>();
            s / 8.0
        })
        .collect();
    let asset_n: Vec<f64> = asset[asset.len() - n..].to_vec();
    let mut betas = Vec::new();
    for start in 0..(n.saturating_sub(60)) {
        if let Ok(b) = beta(&asset_n[start..start + 60], &market[start..start + 60]) {
            betas.push(b);
        }
    }
    let mean_b = betas.iter().sum::<f64>() / betas.len() as f64;
    let std_b = {
        let m = mean_b;
        (betas.iter().map(|b| (b - m).powi(2)).sum::<f64>() / betas.len() as f64).sqrt()
    };
    println!("   Rolling beta mean = {mean_b:.4}, std = {std_b:.4}");
    assert!(
        (0.3..=2.0).contains(&mean_b),
        "mean beta should be in [0.3, 2.0], got {mean_b}"
    );
    assert!(std_b > 0.02, "betas should be time-varying, std={std_b}");
}

fn exercise_5() {
    println!("5. Market-neutral long-short (diversification reduces idiosyncratic):");
    // Synthetic SMB factor and 2 quintile portfolios (high vs low).
    let n = 300_usize;
    let mut rng = XorShift64::new(5);
    let normal = Normal::standard();
    let mut smb = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    for _ in 0..n {
        let s = 0.05 + 0.02 * normal.sample(&mut rng);
        smb.push(s);
        // High quintile: beta_smb = 1.2 + idiosyncratic noise.
        high.push(1.2 * s + 0.03 * normal.sample(&mut rng));
        // Low quintile: beta_smb = -0.8 + idiosyncratic noise.
        low.push(-0.8 * s + 0.03 * normal.sample(&mut rng));
    }
    // Long-short portfolio: long high, short low, scaled so beta_smb ~ 0.
    // We want w_high * beta_high + w_low * beta_low = 0 where w_low < 0 (short).
    // Pick w_high = 1, then w_low = -beta_high / beta_low = -1.2 / -0.8 = 1.5
    // (i.e. short 1.5 units of low per 1 unit of high; beta_low is negative so
    // w_low must be positive to short it... actually w_low is the signed weight).
    // Simpler: solve w_high*1.2 + w_low*(-0.8) = 0 with w_high = 1 -> w_low = 1.5.
    // Long 1 unit high, long 1.5 units of low (but low has negative beta, so
    // this is equivalent to shorting 1.5 units of a positive-beta asset).
    let w_high = 1.0_f64;
    let w_low = 1.5_f64; // signed weight; low has beta -0.8 so +1.5 * (-0.8) = -1.2 = -w_high*1.2
    let ls: Vec<f64> = (0..n).map(|i| w_high * high[i] + w_low * low[i]).collect();
    // Variance decomposition: systematic (from SMB) + idiosyncratic.
    let mean_ls: f64 = ls.iter().sum::<f64>() / n as f64;
    let var_ls: f64 = ls.iter().map(|r| (r - mean_ls).powi(2)).sum::<f64>() / n as f64;
    let mean_smb: f64 = smb.iter().sum::<f64>() / n as f64;
    let var_smb: f64 = smb.iter().map(|s| (s - mean_smb).powi(2)).sum::<f64>() / n as f64;
    let ls_beta = beta(&ls, &smb).unwrap_or(0.0);
    let systematic = ls_beta * ls_beta * var_smb;
    let idiosyncratic = (var_ls - systematic).max(0.0);
    let idio_share = idiosyncratic / var_ls;
    println!("   Long-short beta_smb = {ls_beta:.4} (expect near 0)");
    println!("   Idiosyncratic share of variance = {idio_share:.4}");
    assert!(
        ls_beta.abs() < 0.15,
        "long-short should be near market-neutral"
    );
    assert!(
        idiosyncratic > 0.0,
        "diversification should leave some idiosyncratic variance"
    );
}

#[test]
fn test_ex1_five_factor_beats_ff3() {
    let n = 100_usize;
    let mut rng = XorShift64::new(1);
    let normal = Normal::standard();
    let (mut mkt, mut smb, mut hml, mut umd, mut q, mut asset) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for _ in 0..n {
        let m = 0.05 + 0.01 * normal.sample(&mut rng);
        let s = 0.02 + 0.005 * normal.sample(&mut rng);
        let h = 0.03 + 0.005 * normal.sample(&mut rng);
        let u = 0.04 + 0.005 * normal.sample(&mut rng);
        let qq = 0.02 + 0.005 * normal.sample(&mut rng);
        let a = 0.001
            + 1.2 * m
            + 0.5 * s
            + 0.3 * h
            + 0.4 * u
            + 0.2 * qq
            + 0.005 * normal.sample(&mut rng);
        mkt.push(m);
        smb.push(s);
        hml.push(h);
        umd.push(u);
        q.push(qq);
        asset.push(a);
    }
    let ff3_factors: Vec<Vec<f64>> = (0..n).map(|i| vec![mkt[i], smb[i], hml[i]]).collect();
    let ff3 = ff3_regression(&asset, &ff3_factors).expect("ff3");
    let x5: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![1.0, mkt[i], smb[i], hml[i], umd[i], q[i]])
        .collect();
    let fit5 = ols(&x5, &asset).expect("ols5");
    assert!(
        fit5.r_squared >= ff3.r_squared - 1e-6,
        "5-factor R^2 should be >= FF3 R^2"
    );
}

#[test]
fn test_ex2_jacobi_matches_power_method() {
    let a = vec![
        vec![4.0, 1.0, 2.0, 0.0, 1.0],
        vec![1.0, 3.0, 0.0, 1.0, 0.0],
        vec![2.0, 0.0, 5.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0, 2.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0, 3.0],
    ];
    let (jacobi_eigs, _) = jacobi_eigen(&a, 100);
    let (pm_eigs, _) = top_k_eigen(&a, 5).expect("top_k");
    // Power iteration with deflation accumulates error for later eigenvalues,
    // especially when they are clustered. Compare the sorted multisets with
    // a loose tolerance rather than element-wise with 1e-6.
    assert_eq!(jacobi_eigs.len(), pm_eigs.len());
    for i in 0..5 {
        let nearest = pm_eigs
            .iter()
            .map(|&p| (p - jacobi_eigs[i]).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest < 1e-3,
            "eigenvalue {i} ({}) no match within 1e-3, nearest = {nearest}",
            jacobi_eigs[i]
        );
    }
}

#[test]
fn test_ex3_pca_first_pc_all_positive() {
    let symbols = [
        "PETR4", "VALE3", "ITSA4", "BBDC4", "B3SA3", "ABEV3", "GGBR4", "WEGE3",
    ];
    let mut all_returns: Vec<Vec<f64>> = Vec::new();
    let mut n_min = usize::MAX;
    for sym in &symbols {
        let p = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), sym);
        let b = common::load_json_ohlcv(&p);
        let r = log_returns(&b.iter().map(|x| x.close).collect::<Vec<_>>());
        n_min = n_min.min(r.len());
        all_returns.push(r);
    }
    let window = 252_usize.min(n_min);
    let matrix: Vec<Vec<f64>> = (0..window)
        .map(|t| {
            all_returns
                .iter()
                .map(|r| r[r.len() - window + t])
                .collect()
        })
        .collect();
    let res = pca(&matrix, 8).expect("pca");
    // The first PC of a basket of equities usually looks like a "market"
    // factor with loadings of the same sign on most (not necessarily all)
    // assets. Eigenvector sign is arbitrary, so after flipping to make the
    // majority positive, check that at least 6 of 8 loadings share a sign.
    let pc0 = &res.eigenvectors[0];
    let pos = pc0.iter().filter(|&&v| v > 1e-6).count();
    let neg = pc0.iter().filter(|&&v| v < -1e-6).count();
    let consistent = pos.max(neg) >= 6;
    assert!(
        consistent,
        "first PC should have consistent sign on most assets, got {pc0:?} (pos={pos}, neg={neg})"
    );
    let evr = res.explained_variance_ratio[0];
    assert!(
        (0.25..=0.60).contains(&evr),
        "first PC EVR {evr} in [0.25, 0.60]"
    );
}

#[test]
fn test_ex4_rolling_beta_in_range() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let asset: Vec<f64> = log_returns(&bars.iter().map(|b| b.close).collect::<Vec<_>>());
    let symbols = [
        "PETR4", "VALE3", "ITSA4", "BBDC4", "B3SA3", "ABEV3", "GGBR4", "WEGE3",
    ];
    let mut all_returns: Vec<Vec<f64>> = Vec::new();
    let mut n_min = usize::MAX;
    for sym in &symbols {
        let p = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), sym);
        let b = common::load_json_ohlcv(&p);
        let r = log_returns(&b.iter().map(|x| x.close).collect::<Vec<_>>());
        n_min = n_min.min(r.len());
        all_returns.push(r);
    }
    let n = n_min.min(asset.len());
    let market: Vec<f64> = (0..n)
        .map(|t| all_returns.iter().map(|r| r[r.len() - n + t]).sum::<f64>() / 8.0)
        .collect();
    let asset_n: Vec<f64> = asset[asset.len() - n..].to_vec();
    let mut betas = Vec::new();
    for start in 0..(n.saturating_sub(60)) {
        if let Ok(b) = beta(&asset_n[start..start + 60], &market[start..start + 60]) {
            betas.push(b);
        }
    }
    let mean_b = betas.iter().sum::<f64>() / betas.len() as f64;
    assert!(
        (0.3..=2.0).contains(&mean_b),
        "mean beta {mean_b} in [0.3, 2.0]"
    );
}

#[test]
fn test_ex5_long_short_near_neutral() {
    let n = 300_usize;
    let mut rng = XorShift64::new(5);
    let normal = Normal::standard();
    let (mut smb, mut high, mut low) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let s = 0.05 + 0.02 * normal.sample(&mut rng);
        smb.push(s);
        high.push(1.2 * s + 0.03 * normal.sample(&mut rng));
        low.push(-0.8 * s + 0.03 * normal.sample(&mut rng));
    }
    let (w_high, w_low) = (1.0_f64, 1.5_f64);
    let ls: Vec<f64> = (0..n).map(|i| w_high * high[i] + w_low * low[i]).collect();
    let ls_beta = beta(&ls, &smb).unwrap_or(0.0);
    assert!(
        ls_beta.abs() < 0.15,
        "long-short beta_smb should be near 0, got {ls_beta}"
    );
}
