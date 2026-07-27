//! Monte Carlo pricing of European options under geometric Brownian motion.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::StochError;
use crate::poisson::validate_mc_inputs;
use quant_core::{Distribution, Normal, Rng};

/// Result of a Monte Carlo pricing simulation.
#[derive(Debug, Clone, Copy)]
pub struct McResult {
    /// Estimated option price (discounted expected payoff).
    pub price: f64,
    /// Standard error of the mean estimate (one standard deviation).
    pub std_error: f64,
    /// Number of simulated paths.
    pub n_paths: usize,
}

/// Monte Carlo price of a European call option under risk-neutral GBM.
///
/// Terminal price `S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)`
/// with `Z ~ N(0, 1)`. Payoff `max(S_T - K, 0)`. Price and standard error:
/// `C = exp(-rT) * mean(payoff)`, `SE = exp(-rT) * std(payoff) / sqrt(N)`.
pub fn mc_call<R: Rng + ?Sized>(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> Result<McResult, StochError> {
    validate_mc_inputs(s0, k, r, sigma, t, n_paths)?;
    let normal = Normal::standard();
    let drift = (r - 0.5 * sigma * sigma) * t;
    let diffusion = sigma * t.sqrt();
    let discount = (-r * t).exp();

    let mut payoffs = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let z = normal.sample(rng);
        let s_t = s0 * (drift + diffusion * z).exp();
        payoffs.push((s_t - k).max(0.0));
    }
    Ok(reduce(&payoffs, discount))
}

/// Monte Carlo price of a European put option under risk-neutral GBM.
pub fn mc_put<R: Rng + ?Sized>(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> Result<McResult, StochError> {
    validate_mc_inputs(s0, k, r, sigma, t, n_paths)?;
    let normal = Normal::standard();
    let drift = (r - 0.5 * sigma * sigma) * t;
    let diffusion = sigma * t.sqrt();
    let discount = (-r * t).exp();

    let mut payoffs = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let z = normal.sample(rng);
        let s_t = s0 * (drift + diffusion * z).exp();
        payoffs.push((k - s_t).max(0.0));
    }
    Ok(reduce(&payoffs, discount))
}

/// Monte Carlo call price using antithetic variates for variance reduction.
///
/// Each uniform draw produces two payoffs — one with `Z` and one with `-Z` —
/// which are perfectly negatively correlated. The variance of the average of
/// an antithetic pair is lower than the variance of two independent samples
/// whenever the payoff is monotone in `Z` (which option payoffs are).
pub fn mc_call_antithetic<R: Rng + ?Sized>(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_paths: usize,
    rng: &mut R,
) -> Result<McResult, StochError> {
    validate_mc_inputs(s0, k, r, sigma, t, n_paths)?;
    let normal = Normal::standard();
    let drift = (r - 0.5 * sigma * sigma) * t;
    let diffusion = sigma * t.sqrt();
    let discount = (-r * t).exp();

    // n_paths is the number of antithetic *pairs*; we report 2*n_paths samples
    // but only n_paths normal draws, which is the computational saving.
    let mut payoffs = Vec::with_capacity(2 * n_paths);
    for _ in 0..n_paths {
        let z = normal.sample(rng);
        let s_plus = s0 * (drift + diffusion * z).exp();
        let s_minus = s0 * (drift - diffusion * z).exp();
        payoffs.push((s_plus - k).max(0.0));
        payoffs.push((s_minus - k).max(0.0));
    }
    Ok(reduce(&payoffs, discount))
}

/// Reduce a vector of (undiscounted) payoffs to a discounted price and
/// standard error.
fn reduce(payoffs: &[f64], discount: f64) -> McResult {
    let n = payoffs.len() as f64;
    let mean: f64 = payoffs.iter().sum::<f64>() / n;
    let var: f64 = if payoffs.len() > 1 {
        let sum_sq: f64 = payoffs.iter().map(|&p| (p - mean).powi(2)).sum();
        sum_sq / (payoffs.len() as f64 - 1.0)
    } else {
        0.0
    };
    let std = var.sqrt();
    McResult {
        price: discount * mean,
        std_error: discount * std / n.sqrt(),
        n_paths: payoffs.len(),
    }
}

/// Confidence interval half-width at the given number of standard deviations
/// (e.g. 1.96 for a 95% interval). Returns `price ± z * std_error`.
pub fn ci_half_width(result: &McResult, z: f64) -> f64 {
    z * result.std_error
}