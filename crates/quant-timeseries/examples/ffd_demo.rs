//! Fractional differentiation demo.
//!
//! Demonstrates López de Prado's fixed-width fractional differentiation:
//! 1. Generate a price series via a random walk.
//! 2. Find the minimum d that achieves stationarity (binary search).
//! 3. Compare ACF of the original series, full differencing (d=1), and the
//!    minimum-d fractional differencing.
//! 4. Display the FFD weight window and how it decays.

use quant_core::{Distribution, Normal, XorShift64};
use quant_timeseries::{acf, find_min_d, frac_diff, ffd_weights};

fn main() {
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let n = 1000;

    // Generate a long random-walk price series.
    let mut prices = Vec::with_capacity(n);
    let mut s = 100.0_f64;
    prices.push(s);
    for _ in 1..n {
        s += normal.sample(&mut rng);
        prices.push(s);
    }

    println!("==============================");
    println!("Fractional Differentiation Demo");
    println!("==============================");
    println!();
    println!("Original series: {} prices", prices.len());

    // Minimum d for stationarity.
    let d_min = find_min_d(&prices, 1e-4, 0.01).unwrap();
    println!("Minimum d for stationarity: {:.2}", d_min);
    println!();

    // Memory comparison via ACF at lag 5.
    let acf_orig = acf(&prices, 5).unwrap();
    let diff1 = frac_diff(&prices, 1.0, 1e-4).unwrap();
    let acf_d1 = acf(&diff1, 5).unwrap();
    let diff_min = frac_diff(&prices, d_min, 1e-4).unwrap();
    let acf_dmin = acf(&diff_min, 5).unwrap();

    println!("Memory comparison (ACF at lag 5):");
    println!("  Original (d=0):    {:.4} (non-stationary)", acf_orig[5]);
    println!("  Full diff (d=1):   {:.4} (no memory)", acf_d1[5]);
    println!("  Frac diff (d={:.2}): {:.4} (memory preserved)", d_min, acf_dmin[5]);
    println!();

    // FFD weights for the minimum d.
    let weights = ffd_weights(d_min, 1e-4).unwrap();
    let preview: Vec<String> = weights.iter().take(8).map(|w| format!("{:.4}", w)).collect();
    println!("FFD weights (d={:.2}, thresh=1e-4):", d_min);
    println!("  [{}{}]", preview.join(", "), if weights.len() > 8 { ", ..." } else { "" });
    println!("  Window size: {}", weights.len());
    println!();
    println!("==============================");
}