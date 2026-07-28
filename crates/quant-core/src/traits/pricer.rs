//! Option pricing and Greeks traits.
//!
//! Defines the [`OptionPricer`] and [`Greeks`] traits for option
//! valuation and sensitivity analysis.

use std::error::Error;

/// Type of option (call or put).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    /// Call option (right to buy).
    Call,
    /// Put option (right to sell).
    Put,
}

/// An option pricer computes option values.
///
/// Different pricing models (Black-Scholes, binomial tree, Monte Carlo)
/// implement this trait to provide a consistent pricing interface.
pub trait OptionPricer {
    /// The error type returned when pricing fails.
    type Error: Error;

    /// Price an option.
    ///
    /// # Arguments
    ///
    /// * `s` - Current underlying price
    /// * `k` - Strike price
    /// * `t` - Time to expiration (years)
    /// * `option_type` - Call or Put
    ///
    /// # Returns
    ///
    /// The option price.
    ///
    /// # Errors
    ///
    /// Returns an error if pricing fails (e.g., negative prices,
    /// invalid parameters).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bs = BlackScholes::new(0.05, 0.2);
    /// let call_price = bs.price(100.0, 100.0, 1.0, OptionType::Call)?;
    /// ```
    fn price(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error>;
}

/// The Greeks measure option price sensitivities.
///
/// Greeks quantify how an option's price changes with respect to
/// underlying price, time, volatility, and interest rates.
pub trait Greeks {
    /// The error type returned when computing Greeks fails.
    type Error: Error;

    /// Delta: ∂V/∂S (sensitivity to underlying price).
    ///
    /// Delta measures the rate of change of option value with respect
    /// to changes in the underlying asset's price.
    fn delta(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error>;

    /// Gamma: ∂²V/∂S² (rate of change of delta).
    ///
    /// Gamma measures the rate of change of delta with respect to
    /// changes in the underlying price.
    fn gamma(&self, s: f64, k: f64, t: f64) -> Result<f64, Self::Error>;

    /// Vega: ∂V/∂σ (sensitivity to volatility).
    ///
    /// Vega measures the rate of change of option value with respect
    /// to changes in implied volatility.
    fn vega(&self, s: f64, k: f64, t: f64) -> Result<f64, Self::Error>;

    /// Theta: ∂V/∂t (time decay).
    ///
    /// Theta measures the rate of change of option value with respect
    /// to the passage of time.
    fn theta(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error>;

    /// Rho: ∂V/∂r (sensitivity to interest rate).
    ///
    /// Rho measures the rate of change of option value with respect
    /// to changes in the risk-free interest rate.
    fn rho(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error>;
}
