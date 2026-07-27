//! Poisson process and Merton jump-diffusion.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::StochError;
use quant_core::{Distribution, Normal, Rng};

/// Draw a single exponential variate with rate `lambda` via the inverse-CDF
/// method: `X = -ln(1 - U) / lambda` where `U ~ Uniform(0, 1)`.
///
/// Interarrival times of a Poisson process with rate `lambda` are
/// `Exp(lambda)`, with mean `1 / lambda`.
pub fn exponential_variate<R: Rng + ?Sized>(rate: f64, rng: &mut R) -> f64 {
    let u = rng.next_f64();
    // Clamp u away from 0 and 1 to avoid ln(0).
    let u = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    -(1.0 - u).ln() / rate
}

/// Simulate a homogeneous Poisson process on `[0, t]` with rate `rate`.
///
/// Returns the vector of event times (jump times) in ascending order. The
/// number of events `N(t)` is Poisson-distributed with mean `rate * t`.
///
/// # Arguments
/// * `rate` - Intensity (events per unit time), must be > 0.
/// * `t` - Time horizon, must be > 0.
/// * `rng` - Source of randomness.
pub fn poisson_process<R: Rng + ?Sized>(rate: f64, t: f64, rng: &mut R) -> Vec<f64> {
    assert!(rate > 0.0, "rate must be > 0");
    assert!(t > 0.0, "t must be > 0");
    let mut times = Vec::new();
    let mut now = 0.0_f64;
    loop {
        let gap = exponential_variate(rate, rng);
        now += gap;
        if now >= t {
            break;
        }
        times.push(now);
    }
    times
}

/// Draw a Poisson count `N(t)` with mean `rate * t` by simulating the full
/// process and returning the number of events. Cheaper when only the count
/// is needed and `rate * t` is small; for large means use the direct
/// Knuth/Gaussian approximation (not implemented here).
pub fn poisson_count<R: Rng + ?Sized>(rate: f64, t: f64, rng: &mut R) -> usize {
    poisson_process(rate, t, rng).len()
}

/// Simulate a Merton jump-diffusion path: GBM with Poisson-distributed jumps.
///
/// Between jumps the process follows GBM with drift `mu` and volatility
/// `sigma`. At each Poisson event time the price is multiplied by a jump
/// factor `J = exp(jump_mean)` (a deterministic log-normal jump; the
/// randomness is in the timing). The mean log-jump is `jump_mean`, so
/// `jump_mean = 0` gives `J = 1` (jumps do not move the price).
///
/// With `jump_rate = 0` the process reduces exactly to [`crate::brownian::gbm`].
///
/// # Arguments
/// * `s0` - Initial price (must be > 0).
/// * `mu` - GBM drift between jumps.
/// * `sigma` - GBM volatility between jumps.
/// * `jump_rate` - Poisson intensity of jumps (>= 0; 0 disables jumps).
/// * `jump_mean` - Mean of the log-jump (jump factor `J = exp(jump_mean)`).
/// * `t` - Total time horizon (must be > 0).
/// * `n` - Number of time steps (must be >= 1).
/// * `rng` - Source of randomness.
#[allow(clippy::too_many_arguments)] // Merton jump-diffusion needs all 8 params.
pub fn jump_diffusion<R: Rng + ?Sized>(
    s0: f64,
    mu: f64,
    sigma: f64,
    jump_rate: f64,
    jump_mean: f64,
    t: f64,
    n: usize,
    rng: &mut R,
) -> Vec<f64> {
    assert!(s0 > 0.0, "s0 must be > 0");
    assert!(t > 0.0, "t must be > 0");
    assert!(n >= 1, "n must be >= 1");
    assert!(jump_rate >= 0.0, "jump_rate must be >= 0");
    let dt = t / n as f64;
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();
    let normal = Normal::standard();
    let jump_factor = jump_mean.exp();

    // Pre-sample jump times on [0, t]. With jump_rate = 0 there are none.
    let jump_times = if jump_rate > 0.0 {
        poisson_process(jump_rate, t, rng)
    } else {
        Vec::new()
    };
    let mut next_jump = 0usize;

    let mut path = Vec::with_capacity(n + 1);
    path.push(s0);
    let mut s = s0;
    for step in 1..=n {
        let time = step as f64 * dt;
        let z = normal.sample(rng);
        s *= (drift + diffusion * z).exp();
        // Apply any jumps that occurred in (time - dt, time].
        while next_jump < jump_times.len() && jump_times[next_jump] <= time {
            s *= jump_factor;
            next_jump += 1;
        }
        path.push(s);
    }
    path
}

/// Validate inputs for the Monte Carlo and simulation entry points.
pub(crate) fn validate_mc_inputs(
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    n_paths: usize,
) -> Result<(), StochError> {
    if s0 <= 0.0 {
        return Err(StochError::InvalidParam("s0 must be > 0".into()));
    }
    if k <= 0.0 {
        return Err(StochError::InvalidParam("k must be > 0".into()));
    }
    if t <= 0.0 {
        return Err(StochError::InvalidParam("t must be > 0".into()));
    }
    if sigma < 0.0 {
        return Err(StochError::InvalidParam("sigma must be >= 0".into()));
    }
    if n_paths == 0 {
        return Err(StochError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }
    let _ = r; // r may be any real (risk-free rate, often >= 0 but not required)
    Ok(())
}
