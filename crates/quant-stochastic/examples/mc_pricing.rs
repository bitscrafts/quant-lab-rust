//! Monte Carlo option pricing demo.
//!
//! Prices a European call option by Monte Carlo and compares to the
//! analytical Black-Scholes price. Demonstrates the 1/sqrt(N) convergence
//! law: as the number of paths increases, the MC estimate approaches BS and
//! the standard error shrinks.
//!
//! Run: cargo run -p quant-stochastic --example mc_pricing

use quant_core::XorShift64;
use quant_stochastic::{bs_call, bs_put, mc_call, mc_call_antithetic, mc_put};

fn main() {
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;

    let bs_c = bs_call(s0, k, r, sigma, t);
    let bs_p = bs_put(s0, k, r, sigma, t);

    println!("==============================");
    println!("Monte Carlo Option Pricing");
    println!("==============================");
    println!();
    println!("Parameters: S0={s0}, K={k}, r={r}, sigma={sigma}, T={t}");
    println!("Black-Scholes call:  {bs_c:.6}");
    println!("Black-Scholes put:   {bs_p:.6}");
    println!();

    // Convergence table: MC call price vs N.
    println!("Convergence (plain MC, European call):");
    println!(
        "  {:>10}  {:>12}  {:>12}  {:>12}",
        "N", "MC price", "BS price", "SE"
    );
    let mut rng = XorShift64::new(42);
    for &n in &[100usize, 1000, 10000, 100000] {
        let mc = mc_call(s0, k, r, sigma, t, n, &mut rng).unwrap();
        println!(
            "  {:>10}  {:>12.6}  {:>12.6}  {:>12.6}",
            n, mc.price, bs_c, mc.std_error
        );
    }
    println!();

    // Standard error scaling: SE ~ 1/sqrt(N).
    println!("Standard error scaling (SE ~ 1/sqrt(N)):");
    let mut rng = XorShift64::new(42);
    let se_10k = mc_call(s0, k, r, sigma, t, 10000, &mut rng)
        .unwrap()
        .std_error;
    let se_40k = mc_call(s0, k, r, sigma, t, 40000, &mut rng)
        .unwrap()
        .std_error;
    let se_160k = mc_call(s0, k, r, sigma, t, 160000, &mut rng)
        .unwrap()
        .std_error;
    println!("  N=10000   SE = {:.6}", se_10k);
    println!(
        "  N=40000   SE = {:.6}  (ratio {:.2}x)",
        se_40k,
        se_10k / se_40k
    );
    println!(
        "  N=160000  SE = {:.6}  (ratio {:.2}x)",
        se_160k,
        se_10k / se_160k
    );
    println!("  Expected: 4x paths -> 2x smaller SE; 16x paths -> 4x smaller SE");
    println!();

    // Antithetic variates: same number of normal draws, lower SE.
    println!("Antithetic variates (variance reduction):");
    let mut rng_plain = XorShift64::new(42);
    let mut rng_anti = XorShift64::new(42);
    let n_draws = 50000;
    let plain = mc_call(s0, k, r, sigma, t, n_draws, &mut rng_plain).unwrap();
    let anti = mc_call_antithetic(s0, k, r, sigma, t, n_draws, &mut rng_anti).unwrap();
    println!(
        "  Plain MC:       price={:.6}, SE={:.6}, payoffs={}",
        plain.price, plain.std_error, plain.n_paths
    );
    println!(
        "  Antithetic MC:   price={:.6}, SE={:.6}, payoffs={}",
        anti.price, anti.std_error, anti.n_paths
    );
    println!(
        "  SE reduction:    {:.1}%",
        (1.0 - anti.std_error / plain.std_error) * 100.0
    );
    println!();

    // Put-call parity check.
    println!("Put-call parity (N=100000):");
    let mut rng = XorShift64::new(42);
    let call = mc_call(s0, k, r, sigma, t, 100000, &mut rng).unwrap();
    let put = mc_put(s0, k, r, sigma, t, 100000, &mut rng).unwrap();
    let parity = s0 - k * (-r * t).exp();
    println!("  MC call - MC put = {:.6}", call.price - put.price);
    println!("  S0 - K*exp(-rT)   = {:.6}  (parity)", parity);
    println!();

    println!("==============================");
}
