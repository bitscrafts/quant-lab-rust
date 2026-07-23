//! Stationarity analysis demo.
//!
//! Demonstrates the stationarity-memory tradeoff:
//! 1. Generate a random walk (non-stationary, unit root).
//! 2. Run the ADF test and show it fails to reject the unit-root null.
//! 3. Apply first differencing (d = 1) and retest: stationary but no memory.
//! 4. Apply fractional differentiation (d = 0.4) and retest: stationary with
//!    preserved memory (high ACF at lag 1).

use quant_core::{Distribution, Normal, XorShift64};
use quant_timeseries::{acf, adf_test, frac_diff, MACKINNON_5PCT};

fn main() {
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let n = 500;

    // Random walk: y_t = y_{t-1} + eps_t, eps ~ N(0,1).
    let mut rw = Vec::with_capacity(n);
    let mut s = 100.0_f64;
    rw.push(s);
    for _ in 1..n {
        s += normal.sample(&mut rng);
        rw.push(s);
    }

    println!("==============================");
    println!("Stationarity Analysis");
    println!("==============================");
    println!();

    // Random walk.
    let r_rw = adf_test(&rw, 1).unwrap();
    println!("Random Walk ({} steps):", n);
    println!("  ADF statistic: {:.4}", r_rw.statistic);
    println!("  Critical (5%):  {:.2}", MACKINNON_5PCT);
    println!("  Stationary:     {}", if r_rw.is_stationary { "YES" } else { "NO" });
    println!();

    // First difference (d = 1).
    let diff1 = frac_diff(&rw, 1.0, 1e-4).unwrap();
    let r_d1 = adf_test(&diff1, 1).unwrap();
    let acf_d1 = acf(&diff1, 5).unwrap();
    println!("First Difference (d=1):");
    println!("  ADF statistic: {:.4}", r_d1.statistic);
    println!("  Stationary:     {}", if r_d1.is_stationary { "YES" } else { "NO" });
    println!("  ACF(1):         {:.4} (memory destroyed)", acf_d1[1]);
    println!();

    // Fractional differentiation (d = 0.4).
    let diff04 = frac_diff(&rw, 0.4, 1e-4).unwrap();
    let r_d04 = adf_test(&diff04, 1).unwrap();
    let acf_d04 = acf(&diff04, 5).unwrap();
    println!("Fractional Diff (d=0.4):");
    println!("  ADF statistic: {:.4}", r_d04.statistic);
    println!("  Stationary:     {}", if r_d04.is_stationary { "YES" } else { "NO" });
    println!("  ACF(1):         {:.4} (memory preserved)", acf_d04[1]);
    println!();
    println!("==============================");
}