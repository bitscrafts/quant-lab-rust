//! Stochastic process trait implementations.
//!
//! This module provides wrapper structs that implement the
//! `StochasticProcess` trait from quant-core for the simulation
//! functions in this crate.

use crate::brownian::gbm;
use crate::error::StochError;
use crate::poisson::{jump_diffusion, poisson_process};
use quant_core::{Rng, StochasticProcess};

/// Geometric Brownian Motion process.
///
/// Models asset prices following the SDE:
/// `dS_t = μ S_t dt + σ S_t dW_t`
///
/// # Example
///
/// ```
/// use quant_stochastic::Gbm;
/// use quant_core::{StochasticProcess, XorShift64};
///
/// let mut rng = XorShift64::new(42);
/// let mut gbm = Gbm::new(0.05, 0.2, &mut rng);
/// let path = gbm.simulate(100.0, 1.0, 252).unwrap();
/// assert_eq!(path.len(), 253);
/// assert_eq!(path[0], 100.0);
/// ```
pub struct Gbm<'a, R: Rng + ?Sized> {
    /// Drift (annualized return).
    pub mu: f64,
    /// Volatility (annualized standard deviation).
    pub sigma: f64,
    /// Random number generator.
    rng: &'a mut R,
}

impl<'a, R: Rng + ?Sized> Gbm<'a, R> {
    /// Create a new GBM process.
    ///
    /// # Arguments
    ///
    /// * `mu` - Drift (expected return)
    /// * `sigma` - Volatility
    /// * `rng` - Random number generator
    pub fn new(mu: f64, sigma: f64, rng: &'a mut R) -> Self {
        Self { mu, sigma, rng }
    }
}

impl<'a, R: Rng + ?Sized> StochasticProcess for Gbm<'a, R> {
    type Error = StochError;

    fn simulate(&mut self, s0: f64, t: f64, n_steps: usize) -> Result<Vec<f64>, Self::Error> {
        if s0 <= 0.0 {
            return Err(StochError::InvalidParam("s0 must be positive".into()));
        }
        if t <= 0.0 {
            return Err(StochError::InvalidParam("t must be positive".into()));
        }
        if n_steps == 0 {
            return Err(StochError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        Ok(gbm(s0, self.mu, self.sigma, t, n_steps, self.rng))
    }

    fn terminal(&mut self, s0: f64, t: f64) -> Result<f64, Self::Error> {
        if s0 <= 0.0 {
            return Err(StochError::InvalidParam("s0 must be positive".into()));
        }
        if t <= 0.0 {
            return Err(StochError::InvalidParam("t must be positive".into()));
        }

        // For GBM, we can use the exact formula for terminal value
        // S_T = S_0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        // where Z ~ N(0,1)
        use quant_core::{Distribution, Normal};
        let normal = Normal::standard();
        let z = normal.sample(self.rng);
        let drift = (self.mu - 0.5 * self.sigma * self.sigma) * t;
        let diffusion = self.sigma * t.sqrt() * z;
        Ok(s0 * (drift + diffusion).exp())
    }
}

/// Poisson process.
///
/// Models random event arrivals with constant intensity λ.
///
/// # Example
///
/// ```
/// use quant_stochastic::Poisson;
/// use quant_core::{StochasticProcess, XorShift64};
///
/// let mut rng = XorShift64::new(42);
/// let mut poisson = Poisson::new(10.0, &mut rng); // 10 events per unit time
/// let path = poisson.simulate(0.0, 1.0, 100).unwrap();
/// // path contains cumulative event counts at each step
/// ```
pub struct Poisson<'a, R: Rng + ?Sized> {
    /// Event rate (intensity λ).
    pub rate: f64,
    /// Random number generator.
    rng: &'a mut R,
}

impl<'a, R: Rng + ?Sized> Poisson<'a, R> {
    /// Create a new Poisson process.
    ///
    /// # Arguments
    ///
    /// * `rate` - Event intensity (λ > 0)
    /// * `rng` - Random number generator
    pub fn new(rate: f64, rng: &'a mut R) -> Self {
        Self { rate, rng }
    }
}

impl<'a, R: Rng + ?Sized> StochasticProcess for Poisson<'a, R> {
    type Error = StochError;

    fn simulate(&mut self, _s0: f64, t: f64, n_steps: usize) -> Result<Vec<f64>, Self::Error> {
        if t <= 0.0 {
            return Err(StochError::InvalidParam("t must be positive".into()));
        }
        if n_steps == 0 {
            return Err(StochError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        if self.rate <= 0.0 {
            return Err(StochError::InvalidParam("rate must be positive".into()));
        }

        // Generate event times
        let event_times = poisson_process(self.rate, t, self.rng);

        // Create path of cumulative counts
        let dt = t / n_steps as f64;
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(0.0); // N(0) = 0

        let mut next_event = 0;
        for step in 1..=n_steps {
            let time = step as f64 * dt;
            let mut count = path[step - 1];

            // Count events in (time - dt, time]
            while next_event < event_times.len() && event_times[next_event] <= time {
                count += 1.0;
                next_event += 1;
            }

            path.push(count);
        }

        Ok(path)
    }

    fn terminal(&mut self, _s0: f64, t: f64) -> Result<f64, Self::Error> {
        if t <= 0.0 {
            return Err(StochError::InvalidParam("t must be positive".into()));
        }
        if self.rate <= 0.0 {
            return Err(StochError::InvalidParam("rate must be positive".into()));
        }

        // Total event count at time t
        let event_times = poisson_process(self.rate, t, self.rng);
        Ok(event_times.len() as f64)
    }
}

/// Jump-diffusion process (Merton model).
///
/// Combines GBM with Poisson jumps:
/// `dS_t = μ S_t dt + σ S_t dW_t + S_t dJ_t`
///
/// where `J_t` is a compound Poisson process.
///
/// # Example
///
/// ```
/// use quant_stochastic::JumpDiffusion;
/// use quant_core::{StochasticProcess, XorShift64};
///
/// let mut rng = XorShift64::new(42);
/// let mut jd = JumpDiffusion::new(0.05, 0.2, 5.0, -0.01, &mut rng);
/// // 5 jumps per year on average, each jump averages -1%
/// let path = jd.simulate(100.0, 1.0, 252).unwrap();
/// ```
pub struct JumpDiffusion<'a, R: Rng + ?Sized> {
    /// GBM drift.
    pub mu: f64,
    /// GBM volatility.
    pub sigma: f64,
    /// Jump arrival rate (Poisson intensity).
    pub jump_rate: f64,
    /// Mean log-jump size.
    pub jump_mean: f64,
    /// Random number generator.
    rng: &'a mut R,
}

impl<'a, R: Rng + ?Sized> JumpDiffusion<'a, R> {
    /// Create a new jump-diffusion process.
    ///
    /// # Arguments
    ///
    /// * `mu` - Drift between jumps
    /// * `sigma` - Volatility between jumps
    /// * `jump_rate` - Poisson intensity of jumps
    /// * `jump_mean` - Mean of log-jump (jump factor = exp(jump_mean))
    /// * `rng` - Random number generator
    pub fn new(mu: f64, sigma: f64, jump_rate: f64, jump_mean: f64, rng: &'a mut R) -> Self {
        Self {
            mu,
            sigma,
            jump_rate,
            jump_mean,
            rng,
        }
    }
}

impl<'a, R: Rng + ?Sized> StochasticProcess for JumpDiffusion<'a, R> {
    type Error = StochError;

    fn simulate(&mut self, s0: f64, t: f64, n_steps: usize) -> Result<Vec<f64>, Self::Error> {
        if s0 <= 0.0 {
            return Err(StochError::InvalidParam("s0 must be positive".into()));
        }
        if t <= 0.0 {
            return Err(StochError::InvalidParam("t must be positive".into()));
        }
        if n_steps == 0 {
            return Err(StochError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        if self.jump_rate < 0.0 {
            return Err(StochError::InvalidParam(
                "jump_rate must be non-negative".into(),
            ));
        }

        Ok(jump_diffusion(
            s0,
            self.mu,
            self.sigma,
            self.jump_rate,
            self.jump_mean,
            t,
            n_steps,
            self.rng,
        ))
    }

    fn terminal(&mut self, s0: f64, t: f64) -> Result<f64, Self::Error> {
        // For jump-diffusion, simulate a single-step path
        let path = self.simulate(s0, t, 1)?;
        Ok(path[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quant_core::XorShift64;

    #[test]
    fn test_gbm_implements_stochastic_process() {
        fn _assert_trait<T: StochasticProcess>() {}
        _assert_trait::<Gbm<XorShift64>>();
    }

    #[test]
    fn test_gbm_simulate_length() {
        let mut rng = XorShift64::new(42);
        let mut gbm = Gbm::new(0.05, 0.2, &mut rng);
        let path = gbm.simulate(100.0, 1.0, 100).unwrap();
        assert_eq!(path.len(), 101);
        assert_eq!(path[0], 100.0);
    }

    #[test]
    fn test_gbm_terminal() {
        let mut rng = XorShift64::new(42);
        let mut gbm = Gbm::new(0.05, 0.2, &mut rng);
        let terminal = gbm.terminal(100.0, 1.0).unwrap();
        assert!(terminal > 0.0); // Must be positive
    }

    #[test]
    fn test_poisson_implements_stochastic_process() {
        fn _assert_trait<T: StochasticProcess>() {}
        _assert_trait::<Poisson<XorShift64>>();
    }

    #[test]
    fn test_poisson_simulate() {
        let mut rng = XorShift64::new(42);
        let mut poisson = Poisson::new(10.0, &mut rng);
        let path = poisson.simulate(0.0, 1.0, 100).unwrap();
        assert_eq!(path.len(), 101);
        assert_eq!(path[0], 0.0); // Starts at zero
        // Cumulative counts should be non-decreasing
        for i in 1..path.len() {
            assert!(path[i] >= path[i - 1]);
        }
    }

    #[test]
    fn test_jump_diffusion_implements_stochastic_process() {
        fn _assert_trait<T: StochasticProcess>() {}
        _assert_trait::<JumpDiffusion<XorShift64>>();
    }

    #[test]
    fn test_jump_diffusion_simulate() {
        let mut rng = XorShift64::new(42);
        let mut jd = JumpDiffusion::new(0.05, 0.2, 5.0, -0.01, &mut rng);
        let path = jd.simulate(100.0, 1.0, 252).unwrap();
        assert_eq!(path.len(), 253);
        assert_eq!(path[0], 100.0);
    }

    #[test]
    fn test_jump_diffusion_no_jumps_equals_gbm() {
        let mut rng1 = XorShift64::new(42);
        let mut rng2 = XorShift64::new(42);

        let mut jd = JumpDiffusion::new(0.05, 0.2, 0.0, 0.0, &mut rng1);
        let mut gbm = Gbm::new(0.05, 0.2, &mut rng2);

        let path_jd = jd.simulate(100.0, 1.0, 10).unwrap();
        let path_gbm = gbm.simulate(100.0, 1.0, 10).unwrap();

        // Should be very close (same random seed, no jumps)
        for i in 0..path_jd.len() {
            assert!((path_jd[i] - path_gbm[i]).abs() < 1e-6);
        }
    }
}
