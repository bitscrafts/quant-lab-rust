//! Exercise solutions for Chapter 6: Statistical Foundations
//!
//! Run: `cargo run -p quant-lib --example solutions-ch06_moments_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch06_moments_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::{excess_kurtosis, skewness};
use quant_lib::prelude::*;
use std::collections::HashSet;

/// Population variance (denominator n, not n-1).
fn pop_variance(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n
}

/// Manual Merton-style jump-diffusion: GBM plus N(0, jump_std^2) jumps at Poisson times.
#[allow(clippy::too_many_arguments)]
fn jump_diffusion_manual(
    s0: f64,
    mu: f64,
    sigma: f64,
    jump_rate: f64,
    jump_std: f64,
    t: f64,
    n: usize,
    n_paths: usize,
    rng: &mut XorShift64,
) -> Vec<Vec<f64>> {
    let dt = t / n as f64;
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();
    let normal = Normal::standard();
    let jump_normal = Normal::new(0.0, jump_std);
    let mut paths = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut path = Vec::with_capacity(n + 1);
        let mut s = s0;
        path.push(s);
        for _ in 0..n {
            let z = normal.sample(rng);
            s *= (drift + diffusion * z).exp();
            // Number of jumps in this step ~ Poisson(jump_rate * dt).
            let lambda_dt = jump_rate * dt;
            let mut jumps = 0u64;
            // Knuth's algorithm for small lambda.
            let mut p = (-lambda_dt).exp();
            let mut cum = p;
            let u = rng.next_f64();
            while u > cum {
                jumps += 1;
                p *= lambda_dt / jumps as f64;
                cum += p;
                if jumps > 100 {
                    break;
                }
            }
            for _ in 0..jumps {
                let j = jump_normal.sample(rng);
                s *= j.exp();
            }
            path.push(s);
        }
        paths.push(path);
    }
    paths
}

fn main() {
    println!("=== Chapter 6: Statistical Foundations - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 6 exercises complete.");
}

fn exercise_1() {
    println!("1. Sample vs Population Variance (for {{1,2,3,4,5}}):");
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let pop_var = pop_variance(&data);
    let sample_var = variance(&data).unwrap_or(0.0);
    println!("   population variance (n)   = {pop_var:.4} (expected 2.0)");
    println!("   sample variance (n-1)      = {sample_var:.4} (expected 2.5)");
    assert!((pop_var - 2.0).abs() < 1e-12);
    assert!((sample_var - 2.5).abs() < 1e-12);
}

fn exercise_2() {
    println!("\n2. Skewness Sign (negative vs positive mixtures):");
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    // Negative skew: N(0,1) with prob 0.9, -5 with prob 0.1.
    let mut neg: Vec<f64> = Vec::with_capacity(5000);
    for _ in 0..5000 {
        if rng.next_f64() < 0.9 {
            neg.push(normal.sample(&mut rng));
        } else {
            neg.push(-5.0);
        }
    }
    // Positive skew: mirror (N(0,1) with prob 0.9, +5 with prob 0.1).
    let mut pos: Vec<f64> = Vec::with_capacity(5000);
    for _ in 0..5000 {
        if rng.next_f64() < 0.9 {
            pos.push(normal.sample(&mut rng));
        } else {
            pos.push(5.0);
        }
    }
    let skew_neg = skewness(&neg).unwrap_or(0.0);
    let skew_pos = skewness(&pos).unwrap_or(0.0);
    println!("   skewness(negative mixture) = {skew_neg:.4} (expected < 0)");
    println!("   skewness(positive mixture) = {skew_pos:.4} (expected > 0)");
    assert!(
        skew_neg < 0.0,
        "negative mixture must have negative skewness"
    );
    assert!(
        skew_pos > 0.0,
        "positive mixture must have positive skewness"
    );
}

fn exercise_3() {
    println!("\n3. RNG Period (first 1000 XorShift64 outputs from seed 42 are distinct):");
    let mut rng = XorShift64::new(42);
    let mut seen = HashSet::with_capacity(1000);
    for _ in 0..1000 {
        let v = rng.next_u64();
        seen.insert(v);
    }
    println!("   distinct outputs = {} (expected 1000)", seen.len());
    assert_eq!(seen.len(), 1000, "first 1000 outputs must be distinct");
}

fn exercise_4() {
    println!("\n4. GBM Convergence (MC E[S_T] vs analytical):");
    let s0 = 100.0_f64;
    let mu = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let n_paths = 10_000;
    let mut rng = XorShift64::new(123);
    let mut terminals = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let path = gbm(s0, mu, sigma, t, 252, &mut rng);
        terminals.push(*path.last().unwrap_or(&s0));
    }
    let mc_mean = mean(&terminals);
    let analytical = s0 * (mu * t).exp();
    let err = (mc_mean - analytical).abs();
    println!("   MC E[S_T] = {mc_mean:.4}, analytical = {analytical:.4}, error = {err:.4}");
    // SE scales as sigma*sqrt(T)/sqrt(N) = 0.2/sqrt(10000) = 0.002, so 0.5 is a generous bound.
    assert!(err < 0.5, "MC estimate should be within ~0.5 of analytical");
}

fn exercise_5() {
    println!("\n5. Jump-Diffusion (excess kurtosis vs jump intensity):");
    let s0 = 100.0_f64;
    let mu = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let n_steps = 252;
    let n_paths = 5_000;
    for &lambda in &[0.0_f64, 0.5, 1.0] {
        let mut rng = XorShift64::new(42);
        let paths =
            jump_diffusion_manual(s0, mu, sigma, lambda, 0.05, t, n_steps, n_paths, &mut rng);
        let terminals: Vec<f64> = paths.iter().map(|p| *p.last().unwrap_or(&s0)).collect();
        let log_rets: Vec<f64> = terminals.iter().map(|&s| (s / s0).ln()).collect();
        let kurt = excess_kurtosis(&log_rets).unwrap_or(0.0);
        println!("   lambda={lambda:.1}: excess kurtosis = {kurt:.3}");
        assert!(kurt.is_finite());
    }
    // As lambda grows from 0 to 1, kurtosis should rise.
    let mut rng0 = XorShift64::new(42);
    let paths0 = jump_diffusion_manual(s0, mu, sigma, 0.0, 0.05, t, n_steps, n_paths, &mut rng0);
    let mut rng1 = XorShift64::new(42);
    let paths1 = jump_diffusion_manual(s0, mu, sigma, 1.0, 0.05, t, n_steps, n_paths, &mut rng1);
    let t0: Vec<f64> = paths0.iter().map(|p| *p.last().unwrap_or(&s0)).collect();
    let t1: Vec<f64> = paths1.iter().map(|p| *p.last().unwrap_or(&s0)).collect();
    let lr0: Vec<f64> = t0.iter().map(|&s| (s / s0).ln()).collect();
    let lr1: Vec<f64> = t1.iter().map(|&s| (s / s0).ln()).collect();
    let k0 = excess_kurtosis(&lr0).unwrap_or(0.0);
    let k1 = excess_kurtosis(&lr1).unwrap_or(0.0);
    println!("   kurtosis(0)={k0:.3}, kurtosis(1)={k1:.3} (should rise)");
    assert!(k1 > k0, "kurtosis should rise with jump intensity");
}

#[test]
fn test_ex1_pop_vs_sample_variance() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let pv = pop_variance(&data);
    let sv = variance(&data).unwrap();
    assert!((pv - 2.0).abs() < 1e-12, "population variance = 2.0");
    assert!((sv - 2.5).abs() < 1e-12, "sample variance = 2.5");
}

#[test]
fn test_ex2_skewness_sign() {
    let mut rng = XorShift64::new(42);
    let normal = Normal::standard();
    let mut neg: Vec<f64> = Vec::with_capacity(5000);
    for _ in 0..5000 {
        if rng.next_f64() < 0.9 {
            neg.push(normal.sample(&mut rng));
        } else {
            neg.push(-5.0);
        }
    }
    let mut pos: Vec<f64> = Vec::with_capacity(5000);
    for _ in 0..5000 {
        if rng.next_f64() < 0.9 {
            pos.push(normal.sample(&mut rng));
        } else {
            pos.push(5.0);
        }
    }
    assert!(skewness(&neg).unwrap() < 0.0);
    assert!(skewness(&pos).unwrap() > 0.0);
}

#[test]
fn test_ex3_rng_period_distinct() {
    let mut rng = XorShift64::new(42);
    let mut seen = HashSet::with_capacity(1000);
    for _ in 0..1000 {
        seen.insert(rng.next_u64());
    }
    assert_eq!(seen.len(), 1000);
}

#[test]
fn test_ex4_gbm_convergence() {
    let s0 = 100.0_f64;
    let mu = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let mut rng = XorShift64::new(123);
    let mut terminals = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let path = gbm(s0, mu, sigma, t, 252, &mut rng);
        terminals.push(*path.last().unwrap_or(&s0));
    }
    let mc_mean = mean(&terminals);
    let analytical = s0 * (mu * t).exp();
    assert!((mc_mean - analytical).abs() < 0.5);
}

#[test]
fn test_ex5_jump_diffusion_kurtosis_rises() {
    let s0 = 100.0_f64;
    let mu = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let n_steps = 252;
    let n_paths = 5_000;
    let mut rng0 = XorShift64::new(42);
    let paths0 = jump_diffusion_manual(s0, mu, sigma, 0.0, 0.05, t, n_steps, n_paths, &mut rng0);
    let mut rng1 = XorShift64::new(42);
    let paths1 = jump_diffusion_manual(s0, mu, sigma, 1.0, 0.05, t, n_steps, n_paths, &mut rng1);
    let lr0: Vec<f64> = paths0
        .iter()
        .map(|p| (*p.last().unwrap_or(&s0) / s0).ln())
        .collect();
    let lr1: Vec<f64> = paths1
        .iter()
        .map(|p| (*p.last().unwrap_or(&s0) / s0).ln())
        .collect();
    let k0 = excess_kurtosis(&lr0).unwrap_or(0.0);
    let k1 = excess_kurtosis(&lr1).unwrap_or(0.0);
    assert!(k1 > k0, "kurtosis must rise with jump intensity");
}
