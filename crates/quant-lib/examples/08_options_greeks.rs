//! Example 08: Options - Black-Scholes, Greeks, Implied Vol, Monte Carlo
//!
//! Level: Intermediate
//!
//! Prices a European ATM call and put under Black-Scholes, computes the
//! analytical Greeks, recovers implied volatility from the market price,
//! and compares the analytical price against a Monte Carlo estimate with
//! antithetic variates.
//!
//! Uses `quant-stochastic` (bs_call, mc_call, mc_call_antithetic) and
//! `quant-options` (delta, gamma, vega, theta, rho, implied_vol,
//! BlackScholes pricer implementing the `OptionPricer` and `Greeks` traits).
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 08_options_greeks
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::options::rho;
use quant_lib::prelude::*;
use quant_lib::stochastic::{ci_half_width, mc_call_antithetic};

fn main() {
    println!("=== Example 08: Black-Scholes, Greeks, Implied Vol, Monte Carlo ===");

    let s0 = 100.0_f64; // spot
    let k = 100.0; // strike (ATM)
    let r = 0.05; // risk-free rate (5%)
    let sigma = 0.20; // volatility (20%)
    let t = 1.0; // time to maturity (1 year)

    // 1. Closed-form BS call and put.
    let call = bs_call(s0, k, r, sigma, t);
    let put = bs_put(s0, k, r, sigma, t);
    println!("BS call = {call:.4}, BS put = {put:.4}");

    // Put-call parity: C - P = S0 - K * exp(-rT).
    let pcp = call - put;
    let intrinsic = s0 - k * (-r * t).exp();
    assert!((pcp - intrinsic).abs() < 1e-9, "put-call parity violated");
    println!("Put-call parity: C - P = {pcp:.4} vs S0 - K*exp(-rT) = {intrinsic:.4} (holds)");

    // 2. Analytical Greeks (free-function form).
    let d = delta(s0, k, r, sigma, t, true);
    let g = gamma(s0, k, r, sigma, t);
    let v = vega(s0, k, r, sigma, t);
    let th = theta(s0, k, r, sigma, t, true);
    let rh = rho(s0, k, r, sigma, t, true);
    println!("\nGreeks (call):");
    println!("  delta = {d:.4}  (N(d1) ~ 0.6 for ATM call)");
    println!("  gamma = {g:.6}");
    println!("  vega  = {v:.4}   (per 1 vol point)");
    println!("  theta = {th:.4}  (per year)");
    println!("  rho   = {rh:.4}  (per 1 rate point)");

    // 3. Trait-based pricer: BlackScholes implements OptionPricer + Greeks.
    let pricer = quant_options::BlackScholes::new(r, sigma);
    let call_trait = pricer.price(s0, k, t, OptionType::Call).unwrap();
    let d_trait = pricer.delta(s0, k, t, OptionType::Call).unwrap();
    assert!((call_trait - call).abs() < 1e-12);
    assert!((d_trait - d).abs() < 1e-12);
    println!("\nTrait form: BlackScholes.price = {call_trait:.4}, .delta = {d_trait:.4} (matches)");

    // 4. Implied volatility: invert BS to recover the sigma we started from.
    let iv = implied_vol(call, s0, k, r, t, true).expect("implied_vol");
    assert!(
        (iv - sigma).abs() < 1e-6,
        "iv recovery failed: got {iv} expected {sigma}"
    );
    println!("\nImplied vol from market call={call:.4}: iv = {iv:.6} (matches input {sigma})");

    // 5. Monte Carlo with antithetic variates: variance reduction.
    let mut rng = XorShift64::new(42);
    let mc = mc_call(s0, k, r, sigma, t, 50_000, &mut rng).unwrap();
    let mut rng2 = XorShift64::new(42);
    let mc_anti = mc_call_antithetic(s0, k, r, sigma, t, 50_000, &mut rng2).unwrap();
    let ci = ci_half_width(&mc, 1.96);
    let ci_anti = ci_half_width(&mc_anti, 1.96);
    println!("\nMonte Carlo (50k paths):");
    println!(
        "  plain:       price = {mc_price:.4}, +/- {ci:.4} (95% CI)",
        mc_price = mc.price
    );
    println!(
        "  antithetic:   price = {:.4}, +/- {ci_anti:.4} (95% CI)",
        mc_anti.price
    );
    println!("  analytical:   price = {call:.4}");
    assert!((mc.price - call).abs() < ci, "MC price outside 95% CI");
    assert!(
        (mc_anti.price - call).abs() < ci_anti,
        "antithetic MC outside 95% CI"
    );
    println!("Both MC estimates are within their 95% CI of the analytical price.");
}
