//! Price series newtype and return calculations.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::CoreError;

/// A validated series of strictly positive, finite prices.
///
/// Wraps `Vec<f64>` to enforce the invariant that every price is `> 0` and
/// finite (no `NaN`, no `+inf`). This is the foundation type for the
/// `Moments` and `RollingWindow` traits, and for return computations.
#[derive(Debug, Clone, PartialEq)]
pub struct PriceSeries(Vec<f64>);

impl PriceSeries {
    /// Create a `PriceSeries` from a vector of prices.
    ///
    /// # Errors
    /// Returns [`CoreError::InvalidPrice`] if any element is non-finite
    /// (NaN or infinity) or non-positive (<= 0).
    pub fn new(prices: Vec<f64>) -> Result<Self, CoreError> {
        for &p in &prices {
            if !p.is_finite() || p <= 0.0 {
                return Err(CoreError::InvalidPrice);
            }
        }
        Ok(Self(prices))
    }

    /// View the prices as a slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Number of prices.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the series is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Consume the wrapper and return the underlying vector.
    pub fn into_vec(self) -> Vec<f64> {
        self.0
    }
}

impl AsRef<[f64]> for PriceSeries {
    fn as_ref(&self) -> &[f64] {
        &self.0
    }
}

/// Compute simple (arithmetic) returns from a price slice.
///
/// # Formula
/// r_t = (P_t - P_{t-1}) / P_{t-1}
///
/// Returns an empty vector for inputs of length < 2. A previous price of
/// zero yields `0.0` for that step (cannot divide by zero).
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        let prev = prices[i - 1];
        if prev == 0.0 {
            out.push(0.0);
        } else {
            out.push((prices[i] - prev) / prev);
        }
    }
    out
}

/// Compute log (continuously compounded) returns from a price slice.
///
/// # Formula
/// r_t = ln(P_t / P_{t-1})
///
/// Returns an empty vector for inputs of length < 2. A non-positive price
/// yields `0.0` for that step (the log is undefined for non-positive args).
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        let prev = prices[i - 1];
        let curr = prices[i];
        if prev > 0.0 && curr > 0.0 {
            out.push((curr / prev).ln());
        } else {
            out.push(0.0);
        }
    }
    out
}

/// Simple returns on a `PriceSeries`.
pub fn series_simple_returns(series: &PriceSeries) -> Vec<f64> {
    simple_returns(series.as_slice())
}

/// Log returns on a `PriceSeries`.
pub fn series_log_returns(series: &PriceSeries) -> Vec<f64> {
    log_returns(series.as_slice())
}