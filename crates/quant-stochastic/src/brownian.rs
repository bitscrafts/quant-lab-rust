//! Brownian motion and geometric Brownian motion simulators.
//!
//! See `README.md` in this directory for the module overview.

use quant_core::{Distribution, Normal, Rng};

/// Simulate a standard Brownian motion path `W_0, W_1, ..., W_n`.
///
/// The path starts at `W_0 = 0` and evolves by independent Gaussian
/// increments `dW_t ~ N(0, dt)`:
/// `W_{t+dt} = W_t + sqrt(dt) * Z` where `Z ~ N(0, 1)`.
///
/// Returns a vector of length `n + 1`.
///
/// # Arguments
/// * `n` - Number of time steps (must be >= 1).
/// * `dt` - Size of each time step (must be > 0).
/// * `rng` - Source of randomness.
pub fn brownian_motion<R: Rng + ?Sized>(n: usize, dt: f64, rng: &mut R) -> Vec<f64> {
    assert!(n >= 1, "n must be >= 1");
    assert!(dt > 0.0, "dt must be > 0");
    let normal = Normal::standard();
    let sqrt_dt = dt.sqrt();
    let mut path = Vec::with_capacity(n + 1);
    path.push(0.0); // W_0 = 0
    let mut w = 0.0_f64;
    for _ in 0..n {
        let z = normal.sample(rng);
        w += sqrt_dt * z;
        path.push(w);
    }
    path
}

/// Simulate a single geometric Brownian motion path via the exact solution
/// to the SDE `dS_t = mu S_t dt + sigma S_t dW_t`:
/// `S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)`.
///
/// Returns a vector of length `n + 1` starting at `s0`.
///
/// # Arguments
/// * `s0` - Initial price (must be > 0).
/// * `mu` - Drift per unit time.
/// * `sigma` - Volatility per sqrt(unit time). `sigma = 0` gives the
///   deterministic path `S_T = s0 * exp(mu * T)`.
/// * `t` - Total time horizon (must be > 0).
/// * `n` - Number of time steps (must be >= 1).
/// * `rng` - Source of randomness.
pub fn gbm<R: Rng + ?Sized>(
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    n: usize,
    rng: &mut R,
) -> Vec<f64> {
    assert!(s0 > 0.0, "s0 must be > 0");
    assert!(t > 0.0, "t must be > 0");
    assert!(n >= 1, "n must be >= 1");
    let dt = t / n as f64;
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();
    let normal = Normal::standard();
    let mut path = Vec::with_capacity(n + 1);
    path.push(s0);
    let mut s = s0;
    for _ in 0..n {
        let z = normal.sample(rng);
        s *= (drift + diffusion * z).exp();
        path.push(s);
    }
    path
}

/// Quadratic variation of a path: `sum of (x_{t+1} - x_t)^2`.
///
/// For a Brownian motion sampled at spacing `dt`, this converges in
/// probability to `T = n * dt` as `n -> infinity` (the defining
/// property of Brownian motion).
pub fn quadratic_variation(path: &[f64]) -> f64 {
    path.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum()
}
