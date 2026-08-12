//! Exercise solutions for Chapter 7: Time Series
//!
//! Run: `cargo run -p quant-lib --example solutions-ch07_timeseries_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch07_timeseries_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::prelude::*;
use quant_lib::timeseries::{acf, adf_test, ffd_weights, find_min_d, frac_diff, ols};

/// Simulate an AR(2) series: x_t = phi1 * x_{t-1} + phi2 * x_{t-2} + eps_t.
fn simulate_ar2(phi1: f64, phi2: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    let normal = Normal::standard();
    let mut x = vec![0.0_f64; n];
    if n >= 2 {
        x[0] = normal.sample(&mut rng);
        x[1] = normal.sample(&mut rng);
    }
    for t in 2..n {
        let eps = normal.sample(&mut rng);
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + eps;
    }
    x
}

/// Yule-Walker PACF: solve R * phi = r where R_{ij} = rho_{|i-j|}, r_i = rho_i.
/// Returns PACF coefficients for lags 1..=max_lag.
fn pacf_yule_walker(data: &[f64], max_lag: usize) -> Vec<f64> {
    let rho = acf(data, max_lag).unwrap_or_default();
    if rho.len() < max_lag + 1 {
        return Vec::new();
    }
    let mut pacf = Vec::with_capacity(max_lag);
    for k in 1..=max_lag {
        // Build the k x k matrix R_{ij} = rho_{|i-j|} (0-indexed i,j in 0..k).
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            let mut row = Vec::with_capacity(k);
            for j in 0..k {
                row.push(rho[i.abs_diff(j)]);
            }
            x.push(row);
        }
        // RHS r_i = rho_{i+1} for i in 0..k.
        let y: Vec<f64> = (0..k).map(|i| rho[i + 1]).collect();
        let fit = ols(&x, &y).ok();
        if let Some(f) = fit {
            // PACF at lag k is the k-th coefficient (last in the system).
            pacf.push(*f.coeffs.last().unwrap_or(&0.0));
        } else {
            pacf.push(0.0);
        }
    }
    pacf
}

/// Generate a fractionally integrated price series: cumulative sum of an AR(1) noise,
/// producing a series with a unit root and long memory.
#[allow(clippy::needless_range_loop)]
fn simulate_frac_series(sigma: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    let normal = Normal::new(0.0, sigma);
    let mut x = vec![0.0_f64; n];
    let mut level = 0.0_f64;
    for i in 0..n {
        level += normal.sample(&mut rng);
        x[i] = level;
    }
    x
}

/// Expanding-window fractional differentiation: use all available history at each
/// point, truncating the weight sequence to the available length.
fn frac_diff_expanding(data: &[f64], d: f64, threshold: f64) -> Vec<f64> {
    let weights = ffd_weights(d, threshold).unwrap_or_default();
    let mut out = Vec::with_capacity(data.len());
    for t in 0..data.len() {
        let avail = t + 1; // number of points available up to and including t
        let wlen = weights.len().min(avail);
        let mut acc = 0.0_f64;
        for k in 0..wlen {
            acc += weights[k] * data[t - k];
        }
        out.push(acc);
    }
    out
}

fn main() {
    println!("=== Chapter 7: Time Series - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 7 exercises complete.");
}

fn exercise_1() {
    println!("1. Partial Autocorrelation (PACF via Yule-Walker for AR(2)):");
    let x = simulate_ar2(0.5, -0.3, 2000, 42);
    let pacf = pacf_yule_walker(&x, 5);
    print!("   PACF lags 1..5: ");
    for (lag, &v) in pacf.iter().enumerate() {
        print!("lag{}={v:+.3} ", lag + 1);
    }
    println!();
    // Expected: lag1 ~ 0.5, lag2 ~ -0.3, lags >=3 ~ 0.
    assert!((pacf[0] - 0.5).abs() < 0.1, "PACF(1) ~ phi1 = 0.5");
    assert!((pacf[1] - (-0.3)).abs() < 0.1, "PACF(2) ~ phi2 = -0.3");
}

fn exercise_2() {
    println!("\n2. KPSS Test (via ADF direction on white noise vs random walk):");
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    // White noise: stationary. ADF should reject the unit root.
    let wn: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
    let adf_wn = adf_test(&wn, 1).unwrap();
    let (stat_wn, is_stat_wn) = (adf_wn.statistic, adf_wn.is_stationary);
    println!("   white noise ADF stat = {stat_wn:.3}, stationary = {is_stat_wn}");
    assert!(
        adf_wn.is_stationary,
        "white noise must be stationary by ADF"
    );
    // Random walk: non-stationary. ADF should fail to reject.
    let mut rw = vec![0.0_f64; 1000];
    let mut level = 0.0_f64;
    #[allow(clippy::needless_range_loop)]
    for i in 0..1000 {
        level += normal.sample(&mut rng);
        rw[i] = level;
    }
    let adf_rw = adf_test(&rw, 1).unwrap();
    let (stat_rw, is_stat_rw) = (adf_rw.statistic, adf_rw.is_stationary);
    println!("   random walk ADF stat = {stat_rw:.3}, stationary = {is_stat_rw}");
    assert!(
        !adf_rw.is_stationary,
        "random walk must be non-stationary by ADF"
    );
}

fn exercise_3() {
    println!("\n3. Real Returns (min d* for stationarity vs sigma):");
    for &sigma in &[0.1_f64, 0.2, 0.4] {
        let data = simulate_frac_series(sigma, 500, 42);
        let d_star = find_min_d(&data, 1e-4, 0.01).unwrap_or(1.0);
        println!("   sigma={sigma}: d* = {d_star:.4}");
        // d* should be in [0.2, 0.6] and largely insensitive to sigma.
        assert!(d_star > 0.2 && d_star < 0.6, "d* should be ~0.35-0.45");
    }
}

fn exercise_4() {
    println!("\n4. Expanding-Window Frac-Diff (vs fixed-width):");
    let data = simulate_frac_series(0.2, 300, 42);
    let d = 0.4_f64;
    let fixed = frac_diff(&data, d, 1e-4).unwrap_or_default();
    let expanding = frac_diff_expanding(&data, d, 1e-4);
    println!(
        "   fixed-width len = {}, expanding-window len = {}",
        fixed.len(),
        expanding.len()
    );
    assert_eq!(
        expanding.len(),
        data.len(),
        "expanding preserves all points"
    );
    assert!(fixed.len() < data.len(), "fixed-width drops warm-up");
    // After the warm-up, both should agree closely.
    let offset = data.len() - fixed.len();
    let mut max_diff = 0.0_f64;
    for (i, &fv) in fixed.iter().enumerate() {
        let ev = expanding[i + offset];
        max_diff = max_diff.max((fv - ev).abs());
    }
    println!("   max |fixed - expanding| after warm-up = {max_diff:.2e}");
    assert!(
        max_diff < 1e-9,
        "fixed and expanding must agree after warm-up"
    );
}

fn exercise_5() {
    println!("\n5. Lag Selection for ADF (AIC-minimising lag):");
    // Random walk with AR(2) noise.
    let x = simulate_ar2(0.3, 0.2, 500, 99);
    let n = x.len();
    let mut best = (f64::INFINITY, 0_usize);
    for k in 0..=10 {
        // AIC = n * ln(sigma_hat^2) + 2 * (k + 2)  (k lags + intercept + level).
        // We approximate sigma_hat^2 from the ADF regression residual variance.
        // Reuse adf_test which internally fits the regression; for AIC we re-fit via OLS.
        let dy: Vec<f64> = (1..n).map(|t| x[t] - x[t - 1]).collect();
        if k + 1 >= dy.len() {
            continue;
        }
        let mut xm: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for t in (k + 1)..n {
            let mut row = vec![1.0]; // intercept
            row.push(x[t - 1]); // lagged level
            for i in 1..=k {
                row.push(dy[t - 1 - i]);
            }
            y.push(dy[t - 1]);
            xm.push(row);
        }
        if xm.is_empty() {
            continue;
        }
        let fit = ols(&xm, &y).ok();
        if let Some(f) = fit {
            let sse: f64 = f.residuals.iter().map(|r| r * r).sum();
            let sigma2 = sse / f.residuals.len() as f64;
            if sigma2 > 0.0 {
                let aic = (n as f64) * sigma2.ln() + 2.0 * (k as f64 + 2.0);
                if aic < best.0 {
                    best = (aic, k);
                }
            }
        }
    }
    println!("   AIC-minimising lag k = {} (AIC = {:.2})", best.1, best.0);
    // Run ADF with the chosen lag.
    let result = adf_test(&x, best.1).unwrap();
    println!(
        "   ADF with lag={}: stat = {:.3}, stationary = {}",
        best.1, result.statistic, result.is_stationary
    );
    assert!(best.0.is_finite(), "AIC must be finite");
}

#[test]
fn test_ex1_pacf_yule_walker() {
    let x = simulate_ar2(0.5, -0.3, 2000, 42);
    let pacf = pacf_yule_walker(&x, 5);
    assert_eq!(pacf.len(), 5);
    assert!((pacf[0] - 0.5).abs() < 0.1, "PACF(1) ~ 0.5");
    assert!((pacf[1] - (-0.3)).abs() < 0.1, "PACF(2) ~ -0.3");
    // Lags >= 3 should be small (cutoff at p=2).
    for v in &pacf[2..] {
        assert!(v.abs() < 0.15, "PACF beyond lag 2 should be ~0");
    }
}

#[test]
fn test_ex2_adf_white_noise_vs_random_walk() {
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let wn: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
    let adf_wn = adf_test(&wn, 1).unwrap();
    assert!(adf_wn.is_stationary, "white noise is stationary");
    let mut rw = vec![0.0_f64; 1000];
    let mut level = 0.0_f64;
    for i in 0..1000 {
        level += normal.sample(&mut rng);
        rw[i] = level;
    }
    let adf_rw = adf_test(&rw, 1).unwrap();
    assert!(!adf_rw.is_stationary, "random walk is non-stationary");
}

#[test]
fn test_ex3_min_d_insensitive_to_sigma() {
    let mut d_stars = Vec::new();
    for &sigma in &[0.1_f64, 0.2, 0.4] {
        let data = simulate_frac_series(sigma, 500, 42);
        let d = find_min_d(&data, 1e-4, 0.01).unwrap_or(1.0);
        assert!(d > 0.2 && d < 0.6);
        d_stars.push(d);
    }
    // The spread across sigma values should be small (insensitive to sigma).
    let spread = d_stars.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - d_stars.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        spread < 0.2,
        "d* should be largely insensitive to sigma (spread < 0.2)"
    );
}

#[test]
fn test_ex4_expanding_vs_fixed_frac_diff() {
    let data = simulate_frac_series(0.2, 300, 42);
    let d = 0.4_f64;
    let fixed = frac_diff(&data, d, 1e-4).unwrap();
    let expanding = frac_diff_expanding(&data, d, 1e-4);
    assert_eq!(expanding.len(), data.len());
    assert!(fixed.len() < data.len());
    let offset = data.len() - fixed.len();
    let mut max_diff = 0.0_f64;
    for (i, &fv) in fixed.iter().enumerate() {
        max_diff = max_diff.max((fv - expanding[i + offset]).abs());
    }
    assert!(
        max_diff < 1e-9,
        "fixed and expanding must agree after warm-up"
    );
}

#[test]
fn test_ex5_aic_lag_selection_finite() {
    let x = simulate_ar2(0.3, 0.2, 500, 99);
    let n = x.len();
    let mut best = (f64::INFINITY, 0_usize);
    for k in 0..=10u32 {
        let dy: Vec<f64> = (1..n).map(|t| x[t] - x[t - 1]).collect();
        if (k as usize) + 1 >= dy.len() {
            continue;
        }
        let mut xm: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for t in (k as usize + 1)..n {
            let mut row = vec![1.0];
            row.push(x[t - 1]);
            for i in 1..=k as usize {
                row.push(dy[t - 1 - i]);
            }
            y.push(dy[t - 1]);
            xm.push(row);
        }
        if xm.is_empty() {
            continue;
        }
        if let Some(f) = ols(&xm, &y).ok() {
            let sse: f64 = f.residuals.iter().map(|r| r * r).sum();
            let sigma2 = sse / f.residuals.len() as f64;
            if sigma2 > 0.0 {
                let aic = (n as f64) * sigma2.ln() + 2.0 * (k as f64 + 2.0);
                if aic < best.0 {
                    best = (aic, k as usize);
                }
            }
        }
    }
    assert!(best.0.is_finite());
    let result = adf_test(&x, best.1).unwrap();
    assert!(result.statistic.is_finite());
}
