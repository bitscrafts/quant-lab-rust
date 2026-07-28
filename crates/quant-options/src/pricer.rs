//! Black-Scholes pricer implementing OptionPricer and Greeks traits.

use crate::error::OptionsError;
use crate::greeks::{delta, gamma, rho, theta, vega};
use quant_core::{Greeks, OptionPricer, OptionType};
use quant_stochastic::{bs_call, bs_put};

/// Black-Scholes option pricer.
///
/// Implements both `OptionPricer` and `Greeks` traits for
/// European-style options under the Black-Scholes model.
///
/// # Example
///
/// ```
/// use quant_options::BlackScholes;
/// use quant_core::{OptionPricer, Greeks, OptionType};
///
/// let bs = BlackScholes::new(0.05, 0.2);
/// let call_price = bs.price(100.0, 100.0, 1.0, OptionType::Call).unwrap();
/// let delta_val = bs.delta(100.0, 100.0, 1.0, OptionType::Call).unwrap();
/// assert!(call_price > 0.0);
/// assert!(delta_val > 0.0 && delta_val < 1.0);
/// ```
pub struct BlackScholes {
    /// Risk-free interest rate (continuous compounding).
    pub r: f64,
    /// Volatility (annualized standard deviation).
    pub sigma: f64,
}

impl BlackScholes {
    /// Create a new Black-Scholes pricer.
    ///
    /// # Arguments
    ///
    /// * `r` - Risk-free rate (e.g., 0.05 for 5%)
    /// * `sigma` - Volatility (e.g., 0.2 for 20%)
    pub fn new(r: f64, sigma: f64) -> Self {
        Self { r, sigma }
    }

    /// Validate pricing inputs.
    fn validate_inputs(&self, s: f64, k: f64, t: f64) -> Result<(), OptionsError> {
        if s <= 0.0 {
            return Err(OptionsError::InvalidParam(
                "spot price must be positive".into(),
            ));
        }
        if k <= 0.0 {
            return Err(OptionsError::InvalidParam(
                "strike price must be positive".into(),
            ));
        }
        if t < 0.0 {
            return Err(OptionsError::InvalidParam(
                "time to expiration must be non-negative".into(),
            ));
        }
        if self.sigma < 0.0 {
            return Err(OptionsError::InvalidParam(
                "volatility must be non-negative".into(),
            ));
        }
        Ok(())
    }
}

impl OptionPricer for BlackScholes {
    type Error = OptionsError;

    fn price(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error> {
        self.validate_inputs(s, k, t)?;

        // Handle edge case: zero time to expiration
        if t == 0.0 {
            return Ok(match option_type {
                OptionType::Call => (s - k).max(0.0),
                OptionType::Put => (k - s).max(0.0),
            });
        }

        let price = match option_type {
            OptionType::Call => bs_call(s, k, self.r, self.sigma, t),
            OptionType::Put => bs_put(s, k, self.r, self.sigma, t),
        };

        Ok(price)
    }
}

impl Greeks for BlackScholes {
    type Error = OptionsError;

    fn delta(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error> {
        self.validate_inputs(s, k, t)?;

        if t == 0.0 {
            // At expiration: delta = 1 if ITM, 0 if OTM
            return Ok(match option_type {
                OptionType::Call if s > k => 1.0,
                OptionType::Put if s < k => -1.0,
                _ => 0.0,
            });
        }

        let is_call = matches!(option_type, OptionType::Call);
        Ok(delta(s, k, self.r, self.sigma, t, is_call))
    }

    fn gamma(&self, s: f64, k: f64, t: f64) -> Result<f64, Self::Error> {
        self.validate_inputs(s, k, t)?;

        if t == 0.0 {
            // At expiration: gamma = 0 (no sensitivity to spot changes)
            return Ok(0.0);
        }

        Ok(gamma(s, k, self.r, self.sigma, t))
    }

    fn vega(&self, s: f64, k: f64, t: f64) -> Result<f64, Self::Error> {
        self.validate_inputs(s, k, t)?;

        if t == 0.0 {
            // At expiration: vega = 0 (no time value remaining)
            return Ok(0.0);
        }

        Ok(vega(s, k, self.r, self.sigma, t))
    }

    fn theta(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error> {
        self.validate_inputs(s, k, t)?;

        if t == 0.0 {
            // At expiration: theta undefined (or zero)
            return Ok(0.0);
        }

        let is_call = matches!(option_type, OptionType::Call);
        Ok(theta(s, k, self.r, self.sigma, t, is_call))
    }

    fn rho(&self, s: f64, k: f64, t: f64, option_type: OptionType) -> Result<f64, Self::Error> {
        self.validate_inputs(s, k, t)?;

        if t == 0.0 {
            // At expiration: rho = 0
            return Ok(0.0);
        }

        let is_call = matches!(option_type, OptionType::Call);
        Ok(rho(s, k, self.r, self.sigma, t, is_call))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes_implements_option_pricer() {
        fn _assert_trait<T: OptionPricer>() {}
        _assert_trait::<BlackScholes>();
    }

    #[test]
    fn test_black_scholes_implements_greeks() {
        fn _assert_trait<T: Greeks>() {}
        _assert_trait::<BlackScholes>();
    }

    #[test]
    fn test_call_price() {
        let bs = BlackScholes::new(0.05, 0.2);
        let price = bs.price(100.0, 100.0, 1.0, OptionType::Call).unwrap();
        // ATM call should be positive
        assert!(price > 0.0);
        // Should be less than spot (no arbitrage)
        assert!(price < 100.0);
    }

    #[test]
    fn test_put_price() {
        let bs = BlackScholes::new(0.05, 0.2);
        let price = bs.price(100.0, 100.0, 1.0, OptionType::Put).unwrap();
        // ATM put should be positive
        assert!(price > 0.0);
        // Should be less than strike
        assert!(price < 100.0);
    }

    #[test]
    fn test_call_delta() {
        let bs = BlackScholes::new(0.05, 0.2);
        let delta_val = bs.delta(100.0, 100.0, 1.0, OptionType::Call).unwrap();
        // Call delta should be between 0 and 1
        assert!(delta_val > 0.0 && delta_val < 1.0);
    }

    #[test]
    fn test_put_delta() {
        let bs = BlackScholes::new(0.05, 0.2);
        let delta_val = bs.delta(100.0, 100.0, 1.0, OptionType::Put).unwrap();
        // Put delta should be between -1 and 0
        assert!(delta_val < 0.0 && delta_val > -1.0);
    }

    #[test]
    fn test_gamma_positive() {
        let bs = BlackScholes::new(0.05, 0.2);
        let gamma_val = bs.gamma(100.0, 100.0, 1.0).unwrap();
        // Gamma always positive
        assert!(gamma_val > 0.0);
    }

    #[test]
    fn test_vega_positive() {
        let bs = BlackScholes::new(0.05, 0.2);
        let vega_val = bs.vega(100.0, 100.0, 1.0).unwrap();
        // Vega always positive
        assert!(vega_val > 0.0);
    }

    #[test]
    fn test_greeks_trait_sum_rule() {
        // Delta(call) + Delta(put) = 1 (approximately, due to put-call parity)
        let bs = BlackScholes::new(0.05, 0.2);
        let delta_call = bs.delta(100.0, 100.0, 1.0, OptionType::Call).unwrap();
        let delta_put = bs.delta(100.0, 100.0, 1.0, OptionType::Put).unwrap();

        // delta_call - delta_put should equal 1
        assert!((delta_call - delta_put - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_expiration() {
        let bs = BlackScholes::new(0.05, 0.2);

        // ITM call at expiration
        let price = bs.price(110.0, 100.0, 0.0, OptionType::Call).unwrap();
        assert!((price - 10.0).abs() < 1e-10);

        // OTM call at expiration
        let price = bs.price(90.0, 100.0, 0.0, OptionType::Call).unwrap();
        assert!(price.abs() < 1e-10);
    }
}
