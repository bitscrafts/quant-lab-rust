//! Simulation: hand-rolled RNG, normal sampler, and geometric Brownian motion.
//!
//! See `README.md` in this directory for the module overview.

use std::f64::consts::PI;

/// Trait for uniform pseudo-random number generators.
///
/// Abstracts the source of randomness so distributions and simulators can be
/// generic over the RNG. The default implementation is [`XorShift64`].
pub trait Rng {
    /// Next 64-bit unsigned integer.
    fn next_u64(&mut self) -> u64;
    /// Next `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64;
}

/// xorshift64\* (Vigna, 2014): a fast, stateless, statistically reasonable
/// non-cryptographic PRNG.
///
/// The state is a single `u64`. The algorithm:
///
/// ```text
/// x ^= x >> 12
/// x ^= x << 25
/// x ^= x >> 27
/// return x * 0x2545F4914F6CDD1D
/// ```
///
/// A seed of zero is replaced with a fixed default to avoid the degenerate
/// all-zero stream.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new RNG with the given seed. A seed of `0` is replaced with
    /// `0x853c49e6748fea9b` to avoid the degenerate zero stream.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    /// Current internal state (for testing / checkpointing).
    pub fn state(&self) -> u64 {
        self.state
    }
}

impl Rng for XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits to produce a value in [0, 1).
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Trait for probability distributions that can be sampled given an RNG.
pub trait Distribution {
    /// Draw a single sample using `rng`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64;
}

/// Normal distribution sampler using the Box-Muller transform.
///
/// One call consumes two uniform draws from the RNG and returns one normal
/// variate (the second variate from the pair is discarded).
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    /// Mean of the distribution.
    pub mean: f64,
    /// Standard deviation of the distribution (must be non-negative).
    pub std: f64,
}

impl Normal {
    /// Create a `Normal(mean, std)` distribution.
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    /// Standard normal: mean 0, std 1.
    pub fn standard() -> Self {
        Self {
            mean: 0.0,
            std: 1.0,
        }
    }
}

impl Distribution for Normal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        box_muller_normal(rng, self.mean, self.std)
    }
}

/// Draw a single normal variate via the Box-Muller transform.
///
/// Consumes two uniforms from `rng` and returns `mean + std * Z` where
/// `Z ~ N(0, 1)`. This is the free-function form of [`Normal::sample`].
pub fn box_muller_normal<R: Rng + ?Sized>(rng: &mut R, mean: f64, std: f64) -> f64 {
    let u1 = rng.next_f64();
    let u2 = rng.next_f64();
    // Clamp u1 away from 0 to avoid log(0).
    let u1 = if u1 < f64::EPSILON { f64::EPSILON } else { u1 };
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;
    let z = r * theta.cos();
    mean + std * z
}

/// Simulate geometric Brownian motion paths.
///
/// Uses the exact solution to the GBM SDE
/// `dS_t = mu * S_t dt + sigma * S_t dW_t`:
///
/// ```text
/// S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
/// ```
///
/// where `Z ~ N(0, 1)`. Each path has `n_steps + 1` values starting at `s0`.
///
/// # Arguments
/// * `s0` - Initial price.
/// * `mu` - Drift per unit time.
/// * `sigma` - Volatility per sqrt(unit time). Setting `sigma = 0` makes the
///   path deterministic: `S_T = s0 * exp(mu * T)`.
/// * `t` - Total time horizon.
/// * `n_steps` - Number of time steps per path.
/// * `n_paths` - Number of independent paths to generate.
/// * `rng` - Source of randomness.
///
/// # Returns
/// A vector of `n_paths` paths, each of length `n_steps + 1`.
pub fn gbm_paths<R: Rng + ?Sized>(
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let dt = if n_steps == 0 { 0.0 } else { t / n_steps as f64 };
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let diffusion = sigma * dt.sqrt();
    let normal = Normal::standard();

    let mut paths = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(s0);
        let mut s = s0;
        for _ in 0..n_steps {
            let z = normal.sample(rng);
            s *= (drift + diffusion * z).exp();
            path.push(s);
        }
        paths.push(path);
    }
    paths
}