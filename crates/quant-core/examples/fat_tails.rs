//! Example: Fat tails detection via excess kurtosis.
//!
//! Demonstrates that standard-normal draws have excess kurtosis near zero,
//! while multiplying every 100th draw by 10 produces a distribution with
//! excess kurtosis > 1 — a synthetic "fat tail" that breaks Gaussian
//! assumptions.
//!
//! Run from the `quant-lab` directory:
//!   cargo run -p quant-core --example fat_tails

use quant_core::{excess_kurtosis, gbm_paths, skewness, Distribution, Normal, XorShift64};

fn report(label: &str, draws: &[f64]) {
    let sk = skewness(draws).unwrap();
    let ku = excess_kurtosis(draws).unwrap();
    println!("{label}");
    println!("  n            = {}", draws.len());
    println!("  skewness     = {sk:+.4}");
    println!("  excess kurt  = {ku:+.4}");
    println!();
}

fn main() {
    println!("Fat Tails Detection");
    println!("====================\n");

    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();

    // 1. Pure Gaussian sample: excess kurtosis near 0.
    let gaussian: Vec<f64> = (0..10_000).map(|_| normal.sample(&mut rng)).collect();
    report("1. Pure Gaussian (10k N(0,1) draws)", &gaussian);

    // 2. Fat-tailed sample: every 100th draw scaled by 10.
    let mut fat: Vec<f64> = (0..10_000).map(|_| normal.sample(&mut rng)).collect();
    for i in (0..fat.len()).step_by(100) {
        fat[i] *= 10.0;
    }
    report("2. Fat tails (every 100th draw x10)", &fat);

    // 3. GBM terminal prices over 252 steps — log-normal, also has positive
    //    excess kurtosis (heavier right tail than a Gaussian).
    let paths = gbm_paths(100.0, 0.05, 0.2, 1.0, 252, 10_000, &mut rng);
    let terminals: Vec<f64> = paths.iter().map(|p| p[p.len() - 1]).collect();
    report("3. GBM terminal prices (10k paths, 252 steps)", &terminals);

    println!("Interpretation");
    println!("--------------");
    println!("Gaussian data has excess kurtosis ~0. Scaling every 100th draw");
    println!("by 10 injects rare large moves, pushing excess kurtosis well");
    println!("above 1.0 — the hallmark of fat tails. This is why Gaussian");
    println!("risk models underestimate tail risk in real markets.");
}