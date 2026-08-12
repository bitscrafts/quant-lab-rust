//! Exercise solutions for Chapter 9: Stochastic Processes
//!
//! Run: `cargo run -p quant-lib --example solutions-ch09_stochastic_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch09_stochastic_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::prelude::*;
use quant_lib::stochastic::ci_half_width;

/// Monte Carlo call price with a control variate on the terminal price.
/// `C_cv = C_hat - beta*(M - E[S_T])` with `beta = Cov(payoff, S_T)/Var(S_T)`.
fn mc_call_control_variate(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_paths: usize,
    rng: &mut impl Rng,
) -> (f64, f64) {
    let normal = Normal::standard();
    let drift = (r - 0.5 * sigma * sigma) * t;
    let diffusion = sigma * t.sqrt();
    let discount = (-r * t).exp();
    let e_s_t = s0 * (r * t).exp();
    let mut payoffs = Vec::with_capacity(n_paths);
    let mut s_ts = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let z = normal.sample(rng);
        let s_t = s0 * (drift + diffusion * z).exp();
        s_ts.push(s_t);
        payoffs.push((s_t - k).max(0.0));
    }
    let n = n_paths as f64;
    let mean_p: f64 = payoffs.iter().sum::<f64>() / n;
    let mean_s: f64 = s_ts.iter().sum::<f64>() / n;
    let var_s: f64 = s_ts.iter().map(|s| (s - mean_s).powi(2)).sum::<f64>() / n;
    let cov: f64 = payoffs
        .iter()
        .zip(s_ts.iter())
        .map(|(p, s)| (p - mean_p) * (s - mean_s))
        .sum::<f64>()
        / n;
    let beta = if var_s > 0.0 { cov / var_s } else { 0.0 };
    let cv_mean = mean_p - beta * (mean_s - e_s_t);
    let cv_var: f64 = payoffs
        .iter()
        .zip(s_ts.iter())
        .map(|(p, s)| {
            let adj = p - beta * (s - e_s_t);
            (adj - cv_mean).powi(2)
        })
        .sum::<f64>()
        / n;
    let cv_se = cv_var.sqrt() / n.sqrt();
    (discount * cv_mean, discount * cv_se)
}

/// Asian option MC price: payoff = max(mean(S) - K, 0), discounted.
#[allow(clippy::too_many_arguments)]
fn mc_asian_call(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    rng: &mut impl Rng,
) -> f64 {
    let dt = t / n_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();
    let discount = (-r * t).exp();
    let normal = Normal::standard();
    let mut total = 0.0_f64;
    for _ in 0..n_paths {
        let mut s = s0;
        let mut sum = 0.0_f64;
        for _ in 0..n_steps {
            let z = normal.sample(rng);
            s *= (drift + diffusion * z).exp();
            sum += s;
        }
        let avg = sum / n_steps as f64;
        total += (avg - k).max(0.0);
    }
    discount * total / n_paths as f64
}

/// Euler-discretised GBM terminal price.
fn euler_terminal(s0: f64, mu: f64, sigma: f64, t: f64, n_steps: usize, rng: &mut impl Rng) -> f64 {
    let dt = t / n_steps as f64;
    let normal = Normal::standard();
    let mut s = s0;
    for _ in 0..n_steps {
        let z = normal.sample(rng);
        s *= 1.0 + mu * dt + sigma * dt.sqrt() * z;
    }
    s
}

/// Exact GBM terminal price.
fn exact_terminal(s0: f64, mu: f64, sigma: f64, t: f64, rng: &mut impl Rng) -> f64 {
    let normal = Normal::standard();
    let z = normal.sample(rng);
    s0 * ((mu - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp()
}

fn main() {
    println!("=== Chapter 9: Stochastic Processes - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    println!("\nAll Chapter 9 exercises complete.");
}

fn exercise_1() {
    println!("1. Control variates (variance reduction):");
    let mut rng = XorShift64::new(42);
    let (s0, k, r, sigma, t, n) = (100.0, 100.0, 0.05, 0.2, 1.0, 50_000_usize);
    let plain = mc_call(s0, k, r, sigma, t, n, &mut rng).expect("mc");
    let (cv_price, cv_se) = mc_call_control_variate(s0, k, r, sigma, t, n, &mut rng);
    let se_ratio = cv_se / plain.std_error;
    let rho = (1.0 - se_ratio * se_ratio).max(0.0).sqrt();
    println!("   Plain SE = {:.6}", plain.std_error);
    println!(
        "   Control-variate SE = {:.6} (ratio {:.4})",
        cv_se, se_ratio
    );
    println!("   Implied |rho| = {:.4} (expect 0.7-0.95)", rho);
    let bs = bs_call(s0, k, r, sigma, t);
    println!(
        "   BS = {bs:.4}, plain = {:.4}, cv = {cv_price:.4}",
        plain.price
    );
    assert!(
        cv_se < plain.std_error,
        "CV SE should be lower than plain SE"
    );
    assert!(se_ratio < 1.0, "SE ratio < 1 expected");
}

fn exercise_2() {
    println!("2. Asian option vs European (averaging reduces variance):");
    let mut rng = XorShift64::new(7);
    let (s0, k, r, sigma, t) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let asian = mc_asian_call(s0, k, r, sigma, t, 252, 20_000, &mut rng);
    let european = mc_call(s0, k, r, sigma, t, 20_000, &mut rng)
        .expect("mc")
        .price;
    let bs = bs_call(s0, k, r, sigma, t);
    println!("   Asian  = {asian:.4}");
    println!("   MC European = {european:.4}");
    println!("   BS European = {bs:.4}");
    assert!(asian < european, "Asian should be cheaper than European");
}

fn exercise_3() {
    println!("3. Euler vs exact GBM bias at dt = 1/252:");
    let mut rng_e = XorShift64::new(11);
    let mut rng_x = XorShift64::new(11);
    let (s0, mu, sigma, t) = (100.0, 0.10, 0.20, 1.0);
    let n_steps = 252_usize;
    let n_paths = 40_000_usize;
    let mut sum_e = 0.0_f64;
    let mut sum_x = 0.0_f64;
    for _ in 0..n_paths {
        sum_e += euler_terminal(s0, mu, sigma, t, n_steps, &mut rng_e);
        sum_x += exact_terminal(s0, mu, sigma, t, &mut rng_x);
    }
    let mean_e = sum_e / n_paths as f64;
    let mean_x = sum_x / n_paths as f64;
    let bias_pct = (mean_e - mean_x).abs() / mean_x * 100.0;
    println!("   Euler mean = {mean_e:.6}, exact mean = {mean_x:.6}");
    println!("   |bias| / exact = {bias_pct:.6}% (expect < 0.05%)");
    assert!(
        bias_pct < 0.05,
        "Euler bias should be < 0.05%, got {bias_pct}%"
    );
}

fn exercise_4() {
    println!("4. Confidence-interval coverage (95% CI):");
    let (s0, k, r, sigma, t) = (100.0, 100.0, 0.05, 0.2, 1.0);
    let bs = bs_call(s0, k, r, sigma, t);
    let n_paths = 5_000_usize;
    let n_trials = 200_usize;
    let z = 1.96;
    let mut hits = 0_usize;
    for seed in 0..n_trials {
        let mut rng = XorShift64::new(1 + seed as u64);
        let mc = mc_call(s0, k, r, sigma, t, n_paths, &mut rng).expect("mc");
        let hw = ci_half_width(&mc, z);
        if (bs - mc.price).abs() <= hw {
            hits += 1;
        }
    }
    let coverage = hits as f64 / n_trials as f64;
    println!("   Coverage = {coverage:.4} ({hits}/{n_trials}) (expect in [0.90, 0.99])");
    assert!(
        (0.90..=0.99).contains(&coverage),
        "coverage {coverage} should be in [0.90, 0.99]"
    );
}

#[test]
fn test_ex1_control_variate_lowers_se() {
    let mut rng = XorShift64::new(99);
    let plain = mc_call(100.0, 100.0, 0.05, 0.2, 1.0, 40_000, &mut rng).expect("mc");
    let (_, cv_se) = mc_call_control_variate(100.0, 100.0, 0.05, 0.2, 1.0, 40_000, &mut rng);
    assert!(
        cv_se < plain.std_error,
        "cv_se {cv_se} should be < plain {}",
        plain.std_error
    );
}

#[test]
fn test_ex2_asian_cheaper_than_european() {
    let mut rng = XorShift64::new(7);
    let asian = mc_asian_call(100.0, 100.0, 0.05, 0.2, 1.0, 252, 10_000, &mut rng);
    let european = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    assert!(asian < european, "asian {asian} < european {european}");
}

#[test]
fn test_ex3_euler_bias_small() {
    let mut rng_e = XorShift64::new(11);
    let (s0, mu, sigma, t) = (100.0, 0.10, 0.20, 1.0);
    let n_steps = 252_usize;
    let n_paths = 20_000_usize;
    let mut sum_e = 0.0_f64;
    for _ in 0..n_paths {
        sum_e += euler_terminal(s0, mu, sigma, t, n_steps, &mut rng_e);
    }
    let mean_e = sum_e / n_paths as f64;
    // Compare to the analytical E[S_T] = s0*exp(mu*T); this removes the
    // sampling noise of a second Monte Carlo estimate and isolates the
    // discretisation bias.
    let mean_exact = s0 * (mu * t).exp();
    let bias_pct = (mean_e - mean_exact).abs() / mean_exact * 100.0;
    assert!(bias_pct < 0.1, "Euler bias pct {bias_pct} should be < 0.1");
}

#[test]
fn test_ex4_ci_coverage_in_range() {
    let bs = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    let n_paths = 5_000_usize;
    let n_trials = 200_usize;
    let z = 1.96;
    let mut hits = 0_usize;
    for seed in 0..n_trials {
        let mut rng = XorShift64::new(1 + seed as u64);
        let mc = mc_call(100.0, 100.0, 0.05, 0.2, 1.0, n_paths, &mut rng).expect("mc");
        let hw = ci_half_width(&mc, z);
        if (bs - mc.price).abs() <= hw {
            hits += 1;
        }
    }
    let coverage = hits as f64 / n_trials as f64;
    assert!(
        (0.90..=0.99).contains(&coverage),
        "coverage {coverage} should be in [0.90, 0.99]"
    );
}
