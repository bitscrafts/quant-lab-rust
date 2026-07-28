//! Stochastic process simulation trait.
//!
//! Defines the [`StochasticProcess`] trait for simulating random paths.

use std::error::Error;

/// A stochastic process generates random sample paths.
///
/// Processes such as Geometric Brownian Motion, Poisson, and Jump
/// Diffusion implement this trait to provide a consistent simulation
/// interface.
pub trait StochasticProcess {
    /// The error type returned when simulation fails.
    type Error: Error;

    /// Simulate a complete path from time 0 to time T.
    ///
    /// # Arguments
    ///
    /// * `s0` - Initial value
    /// * `t` - Terminal time (e.g., 1.0 for one year)
    /// * `n_steps` - Number of time steps
    ///
    /// # Returns
    ///
    /// A vector of length `n_steps + 1` containing the path,
    /// starting with `s0` at index 0.
    ///
    /// # Errors
    ///
    /// Returns an error if simulation fails (e.g., invalid parameters).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let gbm = Gbm::new(0.05, 0.2, &mut rng);
    /// let path = gbm.simulate(100.0, 1.0, 252)?;
    /// assert_eq!(path.len(), 253);
    /// assert_eq!(path[0], 100.0);
    /// ```
    fn simulate(&mut self, s0: f64, t: f64, n_steps: usize) -> Result<Vec<f64>, Self::Error>;

    /// Simulate the terminal value at time T (without intermediate steps).
    ///
    /// This is more efficient than `simulate()` when only the final
    /// value is needed.
    ///
    /// # Arguments
    ///
    /// * `s0` - Initial value
    /// * `t` - Terminal time
    ///
    /// # Returns
    ///
    /// The terminal value S(T).
    ///
    /// # Errors
    ///
    /// Returns an error if simulation fails.
    fn terminal(&mut self, s0: f64, t: f64) -> Result<f64, Self::Error>;
}
