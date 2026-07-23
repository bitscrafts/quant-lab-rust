//! Strategy trait and reference strategies (SMA crossover, buy-and-hold).
//!
//! See `README.md` in this directory for the module overview.

use crate::error::BacktestError;
use crate::signal::Signal;
use qf_03_stocks::Ohlcv;

/// A trading strategy maps the price history seen so far to a single signal
/// for the current bar.
///
/// # No look-ahead
///
/// `signal` is called with `data[..=index]`: the strategy must never read past
/// `index`. The engine enforces this by passing a slice and the current index;
/// implementations should compute indicators only from `data[0..=index]`.
pub trait Strategy {
    /// Generate a signal for the bar at `index`, given the full price history
    /// up to and including the current bar.
    fn signal(&self, data: &[Ohlcv], index: usize) -> Signal;

    /// Human-readable strategy name for reporting.
    fn name(&self) -> &str;
}

/// Simple Moving Average crossover strategy.
///
/// Buy when the short SMA crosses above the long SMA ("golden cross"); sell
/// when it crosses below ("death cross"); otherwise hold.
#[derive(Debug, Clone)]
pub struct SmaCrossover {
    short_period: usize,
    long_period: usize,
}

impl SmaCrossover {
    /// Create a new SMA crossover strategy.
    ///
    /// # Errors
    /// - `short_period` must be strictly less than `long_period`.
    /// - Both periods must be at least 1.
    pub fn new(short_period: usize, long_period: usize) -> Result<Self, BacktestError> {
        if short_period == 0 || long_period == 0 {
            return Err(BacktestError::InvalidParams(
                "periods must be at least 1".to_string(),
            ));
        }
        if short_period >= long_period {
            return Err(BacktestError::InvalidParams(format!(
                "short_period ({short_period}) must be strictly less than long_period ({long_period})"
            )));
        }
        Ok(Self {
            short_period,
            long_period,
        })
    }

    /// The short moving average period.
    pub fn short_period(&self) -> usize {
        self.short_period
    }

    /// The long moving average period.
    pub fn long_period(&self) -> usize {
        self.long_period
    }

    /// Compute the SMA at `index` over `period` bars, or `None` if there is
    /// not enough history.
    fn sma_at(data: &[Ohlcv], index: usize, period: usize) -> Option<f64> {
        if index + 1 < period {
            return None;
        }
        let start = index + 1 - period;
        let sum: f64 = data[start..=index].iter().map(|o| o.close).sum();
        Some(sum / period as f64)
    }
}

impl Strategy for SmaCrossover {
    fn signal(&self, data: &[Ohlcv], index: usize) -> Signal {
        // Need enough bars for the long SMA at the *previous* bar, i.e.
        // index >= long_period. When index == 0 or we don't yet have two SMA
        // observations to compare, hold.
        if index < self.long_period {
            return Signal::Hold;
        }

        let short_now = match Self::sma_at(data, index, self.short_period) {
            Some(v) => v,
            None => return Signal::Hold,
        };
        let long_now = match Self::sma_at(data, index, self.long_period) {
            Some(v) => v,
            None => return Signal::Hold,
        };
        let short_prev = match Self::sma_at(data, index.saturating_sub(1), self.short_period) {
            Some(v) => v,
            None => return Signal::Hold,
        };
        let long_prev = match Self::sma_at(data, index.saturating_sub(1), self.long_period) {
            Some(v) => v,
            None => return Signal::Hold,
        };

        let crossed_up = short_prev <= long_prev && short_now > long_now;
        let crossed_down = short_prev >= long_prev && short_now < long_now;

        if crossed_up {
            Signal::Buy
        } else if crossed_down {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn name(&self) -> &str {
        "SMA Crossover"
    }
}

/// Passive benchmark: buy on the first bar, hold forever.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuyAndHold;

impl Strategy for BuyAndHold {
    fn signal(&self, _data: &[Ohlcv], index: usize) -> Signal {
        if index == 0 {
            Signal::Buy
        } else {
            Signal::Hold
        }
    }

    fn name(&self) -> &str {
        "Buy and Hold"
    }
}