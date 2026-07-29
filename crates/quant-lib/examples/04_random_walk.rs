//! Example 04: Random Walks - XorShift64, Normal, Brownian Motion, GBM
//!
//! Level: Intermediate
//!
//! Demonstrates the hand-rolled simulation primitives in `quant-core`
//! and `quant-stochastic`:
//!
//! - `XorShift64` deterministic PRNG and its `Rng` trait
//! - `Normal` distribution via Box-Muller (`Distribution` trait)
//! - `brownian_motion`: standard Brownian path W_t
//! - `gbm`: geometric Brownian motion S_t (exact solution of the SDE)
//! - `quadratic_variation`: [W]_t -> T as n -> infinity
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 04_random_walk
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::prelude::*;
use quant_lib::stochastic::quadratic_variation;

fn main() {
    println!("=== Example 04: Random Walks ===");

    // 1. Deterministic PRNG: same seed -> same stream.
    let mut rng = XorShift64::new(42);
    let u0 = rng.next_f64();
    let u1 = rng.next_f64();
    let mut rng2 = XorShift64::new(42);
    assert!((u0 - rng2.next_f64()).abs() < 1e-12);
    assert!((u1 - rng2.next_f64()).abs() < 1e-12);
    println!("XorShift64 seed=42: u0={u0:.6}, u1={u1:.6} (reproducible)");

    // 2. Normal samples via Box-Muller.
    let normal = Normal::standard();
    let samples: Vec<f64> = (0..10_000).map(|_| normal.sample(&mut rng)).collect();
    let mu = mean(&samples);
    let sd = std_dev(&samples).unwrap();
    println!(
        "Normal(0,1):  n={}, mean={mu:.4}, std={sd:.4} (expect ~0, ~1)",
        samples.len()
    );

    // 3. Standard Brownian motion over [0, 1] with 252 steps.
    let n = 252;
    let dt = 1.0 / n as f64;
    let w = brownian_motion(n, dt, &mut rng);
    assert_eq!(w.len(), n + 1);
    assert!((w[0]).abs() < 1e-12);
    let qv = quadratic_variation(&w);
    // [W]_T -> T in probability as n -> infinity. For n=252, expect ~1.
    println!(
        "Brownian motion: n={n}, W_T={:.4}, [W]_T={qv:.4} (expect ~{:.4})",
        w[n], 1.0
    );

    // 4. GBM: S_t = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t).
    let s0 = 100.0_f64;
    let mu = 0.08;
    let sigma = 0.20;
    let t = 1.0;
    let path = gbm(s0, mu, sigma, t, n, &mut rng);
    assert_eq!(path.len(), n + 1);
    assert!((path[0] - s0).abs() < 1e-12);
    println!("GBM: S0={s0}, mu={mu}, sigma={sigma}, S_T={:.4}", path[n]);

    // 5. Trait-based simulation: the `StochasticProcess` trait gives the
    // same path as the free function `gbm` when seeded identically.
    let mut rng_a = XorShift64::new(7);
    let free_path = gbm(s0, mu, sigma, t, n, &mut rng_a);
    let mut rng_b = XorShift64::new(7);
    let mut proc = quant_stochastic::Gbm::new(mu, sigma, &mut rng_b);
    let trait_path = proc.simulate(s0, t, n).unwrap();
    assert_eq!(free_path.len(), trait_path.len());
    for (a, b) in free_path.iter().zip(trait_path.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    println!("Trait form `Gbm::simulate` matches free fn `gbm` (seed-matched)");

    // 6. Terminal value: single-step exact formula.
    let mut rng_c = XorShift64::new(123);
    let mut proc2 = quant_stochastic::Gbm::new(mu, sigma, &mut rng_c);
    let s_t = proc2.terminal(s0, t).unwrap();
    println!("GBM terminal (single draw): S_T = {s_t:.4}");
    assert!(s_t > 0.0);

    // All terminal values are positive (GBM stays positive).
    let pos_count = path.iter().filter(|&&p| p > 0.0).count();
    assert_eq!(pos_count, path.len());
    println!(
        "GBM positivity invariant holds: all {} points > 0",
        path.len()
    );
}
