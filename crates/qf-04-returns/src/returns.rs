//! Returns computation: simple, logarithmic, and cumulative.
//!
//! See `README.md` in this directory for the module overview.

use qf_03_stocks::Ohlcv;

/// Computes simple (arithmetic) returns from a price series.
///
/// # Formula
/// r_t = (P_t - P_{t-1}) / P_{t-1}
///
/// # Arguments
/// * `prices` - Slice of prices ordered chronologically (oldest first).
///
/// # Returns
/// Vector of simple returns with length `prices.len().saturating_sub(1)`.
/// Returns an empty vector for empty or single-element input. If a previous
/// price is zero, the corresponding return is `0.0` (avoids division by zero
/// panics).
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

/// Computes log (continuously compounded) returns from a price series.
///
/// # Formula
/// r_t = ln(P_t / P_{t-1})
///
/// # Arguments
/// * `prices` - Slice of prices ordered chronologically (oldest first).
///
/// # Returns
/// Vector of log returns with length `prices.len().saturating_sub(1)`.
/// Returns an empty vector for empty or single-element input. If either price
/// is non-positive, the corresponding return is `0.0` (the logarithm is
/// undefined for non-positive arguments).
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

/// Computes the running cumulative return from a series of period returns.
///
/// Compounding formula: the cumulative return at step `i` is
/// `prod_{k=0..=i} (1 + r_k) - 1`.
///
/// # Arguments
/// * `returns` - Slice of period returns (simple or log; for log returns the
///   caller should exponentiate first if a wealth interpretation is desired).
///
/// # Returns
/// Vector of cumulative returns with the same length as the input. An empty
/// input yields an empty output.
pub fn cumulative_returns(returns: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(returns.len());
    let mut wealth = 1.0_f64;
    for &r in returns {
        wealth *= 1.0 + r;
        out.push(wealth - 1.0);
    }
    out
}

/// Trait for types that can produce return series.
///
/// Implemented for slices of prices, owned `Vec<f64>`, and `Vec<Ohlcv>` (which
/// uses closing prices). This follows the trait-first architecture rule:
/// before computing on a concrete type, define the abstract capability.
pub trait Returns {
    /// Returns the simple return series derived from this object.
    fn simple_returns(&self) -> Vec<f64>;
    /// Returns the log return series derived from this object.
    fn log_returns(&self) -> Vec<f64>;
}

impl Returns for &[f64] {
    fn simple_returns(&self) -> Vec<f64> {
        simple_returns(self)
    }
    fn log_returns(&self) -> Vec<f64> {
        log_returns(self)
    }
}

impl Returns for Vec<f64> {
    fn simple_returns(&self) -> Vec<f64> {
        simple_returns(self.as_slice())
    }
    fn log_returns(&self) -> Vec<f64> {
        log_returns(self.as_slice())
    }
}

impl Returns for Vec<Ohlcv> {
    fn simple_returns(&self) -> Vec<f64> {
        let closes: Vec<f64> = self.iter().map(|o| o.close).collect();
        simple_returns(&closes)
    }
    fn log_returns(&self) -> Vec<f64> {
        let closes: Vec<f64> = self.iter().map(|o| o.close).collect();
        log_returns(&closes)
    }
}