//! Rolling-window computations and the `RollingWindow` trait.
//!
//! See `README.md` in this directory for the module overview.

use crate::error::CoreError;
use crate::moments::{mean, std_dev};

/// Apply `f` over sliding windows of length `window` on `data`.
///
/// For each window `data[i..i+window]` for `i in 0..=data.len()-window`, the
/// output is `f(window_slice)`. The output has length `data.len() - window + 1`.
///
/// # Errors
/// Returns [`CoreError::InvalidWindow`] when `window == 0` or
/// `window > data.len()`.
pub fn rolling<F, T>(window: usize, data: &[f64], f: F) -> Result<Vec<T>, CoreError>
where
    F: Fn(&[f64]) -> T,
{
    if window == 0 || window > data.len() {
        return Err(CoreError::InvalidWindow {
            window,
            len: data.len(),
        });
    }
    let mut out = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=(data.len() - window) {
        out.push(f(&data[i..i + window]));
    }
    Ok(out)
}

/// Rolling mean over `window`. Wrapper around `rolling` with `mean` as the
/// aggregation. Returns an error for invalid windows.
pub fn rolling_mean(window: usize, data: &[f64]) -> Result<Vec<f64>, CoreError> {
    rolling(window, data, mean)
}

/// Rolling sample standard deviation over `window`. Windows with fewer than
/// two elements produce `0.0` (handled by `std_dev` returning `Ok(0.0)` via
/// the underlying `variance` fallback — here we guard explicitly so the
/// output is always finite).
pub fn rolling_std_dev(window: usize, data: &[f64]) -> Result<Vec<f64>, CoreError> {
    if window < 2 {
        return Err(CoreError::InvalidWindow {
            window,
            len: data.len(),
        });
    }
    rolling(window, data, |w| std_dev(w).unwrap_or(0.0))
}

/// Trait for sliding-window aggregations.
pub trait RollingWindow {
    /// Rolling arithmetic mean.
    fn rolling_mean(&self, window: usize) -> Result<Vec<f64>, CoreError>;
    /// Rolling sample standard deviation.
    fn rolling_std_dev(&self, window: usize) -> Result<Vec<f64>, CoreError>;
}

impl RollingWindow for &[f64] {
    fn rolling_mean(&self, window: usize) -> Result<Vec<f64>, CoreError> {
        rolling_mean(window, self)
    }
    fn rolling_std_dev(&self, window: usize) -> Result<Vec<f64>, CoreError> {
        rolling_std_dev(window, self)
    }
}

impl RollingWindow for Vec<f64> {
    fn rolling_mean(&self, window: usize) -> Result<Vec<f64>, CoreError> {
        rolling_mean(window, self.as_slice())
    }
    fn rolling_std_dev(&self, window: usize) -> Result<Vec<f64>, CoreError> {
        rolling_std_dev(window, self.as_slice())
    }
}

impl RollingWindow for crate::series::PriceSeries {
    fn rolling_mean(&self, window: usize) -> Result<Vec<f64>, CoreError> {
        rolling_mean(window, self.as_slice())
    }
    fn rolling_std_dev(&self, window: usize) -> Result<Vec<f64>, CoreError> {
        rolling_std_dev(window, self.as_slice())
    }
}