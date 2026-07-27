//! Brownian motion and GBM sample paths demo.
//!
//! Generates and prints sample paths of standard Brownian motion, geometric
//! Brownian motion, and a Merton jump-diffusion. Shows the quadratic
//! variation property (sum dW^2 -> T) and the GBM terminal distribution.
//!
//! Run: cargo run -p quant-stochastic --example brownian_paths

use quant_core::XorShift64;
use quant_stochastic::{brownian_motion, gbm, jump_diffusion, poisson_count, quadratic_variation};

fn main() {
    let dt = 1.0 / 252.0;
    let n = 252;
    let t = n as f64 * dt;

    println!("==============================");
    println!("Brownian Motion and GBM Paths");
    println!("==============================");
    println!();
    println!("Parameters: n={n} steps, dt={dt:.6}, T={t:.4}");
    println!();

    // Standard Brownian motion.
    let mut rng = XorShift64::new(42);
    let w = brownian_motion(n, dt, &mut rng);
    let qv = quadratic_variation(&w);
    println!("Standard Brownian motion W_t:");
    println!("  W_0   = {:.6}", w[0]);
    println!("  W_T   = {:.6}", w[n]);
    println!(
        "  min   = {:.6}",
        w.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    println!(
        "  max   = {:.6}",
        w.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "  quadratic variation = {:.6}  (should be ~ T = {t:.4})",
        qv
    );
    println!();

    // Quadratic variation over many paths: converges to T.
    println!("Quadratic variation convergence (E[QV] -> T):");
    let mut rng = XorShift64::new(42);
    for &steps in &[100usize, 1000, 10000, 100000] {
        let dt = 1.0 / 252.0;
        let path = brownian_motion(steps, dt, &mut rng);
        let qv = quadratic_variation(&path);
        let t = steps as f64 * dt;
        println!(
            "  n={steps:>7}  QV={qv:.4}  T={t:.4}  rel err={:.3}%",
            (qv - t).abs() / t * 100.0
        );
    }
    println!();

    // Geometric Brownian motion.
    let mut rng = XorShift64::new(42);
    let s0 = 100.0_f64;
    let mu = 0.05;
    let sigma = 0.2;
    let p = gbm(s0, mu, sigma, t, n, &mut rng);
    println!("Geometric Brownian motion S_t (s0={s0}, mu={mu}, sigma={sigma}):");
    println!("  S_0 = {:.4}", p[0]);
    println!("  S_T = {:.4}", p[n]);
    println!("  log-return = {:.6}", (p[n] / s0).ln());
    println!(
        "  expected log-return = {:.6}  ((mu - 0.5*sigma^2)*T)",
        (mu - 0.5 * sigma * sigma) * t
    );
    println!();

    // Terminal distribution of GBM over many paths.
    println!("GBM terminal distribution (100000 paths):");
    let mut rng = XorShift64::new(42);
    let n_paths = 100000;
    let mut terminals = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let path = gbm(s0, mu, sigma, t, n, &mut rng);
        terminals.push(path[n]);
    }
    terminals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = terminals.iter().sum::<f64>() / n_paths as f64;
    let median = terminals[n_paths / 2];
    let p5 = terminals[(n_paths as f64 * 0.05) as usize];
    let p95 = terminals[(n_paths as f64 * 0.95) as usize];
    println!(
        "  E[S_T] = {:.4}  (analytical: S0*exp(mu*T) = {:.4})",
        mean,
        s0 * (mu * t).exp()
    );
    println!("  median = {:.4}", median);
    println!("  5th percentile  = {:.4}", p5);
    println!("  95th percentile  = {:.4}", p95);
    println!();

    // Jump-diffusion.
    let mut rng = XorShift64::new(42);
    let jd = jump_diffusion(s0, mu, sigma, 3.0, 0.1, t, n, &mut rng);
    println!(
        "Merton jump-diffusion (jump_rate=3, jump_mean=0.1, J=exp(0.1)={:.4}):",
        0.1_f64.exp()
    );
    println!("  S_0 = {:.4}", jd[0]);
    println!("  S_T = {:.4}", jd[n]);
    println!();

    // Poisson counts.
    println!("Poisson process (rate=5, T=1, expected count=5):");
    let mut rng = XorShift64::new(42);
    let counts: Vec<usize> = (0..5000)
        .map(|_| poisson_count(5.0, 1.0, &mut rng))
        .collect();
    let mean_count: f64 = counts.iter().sum::<usize>() as f64 / counts.len() as f64;
    println!("  mean count over 5000 draws = {:.3}", mean_count);
    println!();

    println!("==============================");
}
