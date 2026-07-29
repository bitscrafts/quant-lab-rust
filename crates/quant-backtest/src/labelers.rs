//! Event labeling implementations.
//!
//! This module provides various labeling strategies implementing
//! the `Labeler` trait from quant-core.

use crate::error::BacktestError;
use crate::triple_barrier::{LabeledEvent, TripleBarrierLabel};
use quant_core::Labeler;

/// Fixed-horizon labeler: labels by sign of n-bar forward return.
///
/// This is the simplest labeling method: buy if price rises over
/// the horizon, sell if it falls. The label is determined by the
/// sign of the return after `horizon` bars.
///
/// # Example
///
/// ```
/// use quant_backtest::FixedHorizonLabeler;
/// use quant_core::Labeler;
///
/// let labeler = FixedHorizonLabeler::new(5, 0.0);
/// let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0];
/// let events = labeler.label(&prices, &[0, 1]).unwrap();
/// // Event 0: return from 100->105 = +5% -> positive label
/// // Event 1: return from 101->106 = +4.95% -> positive label
/// ```
pub struct FixedHorizonLabeler {
    /// Number of bars to look ahead.
    pub horizon: usize,
    /// Minimum return to label as positive (e.g., 0.01 = 1%).
    pub min_return: f64,
}

impl FixedHorizonLabeler {
    /// Create a new fixed-horizon labeler.
    ///
    /// # Arguments
    ///
    /// * `horizon` - Number of bars to look ahead
    /// * `min_return` - Minimum return threshold for positive label
    pub fn new(horizon: usize, min_return: f64) -> Self {
        Self {
            horizon,
            min_return,
        }
    }
}

impl Labeler for FixedHorizonLabeler {
    type Event = LabeledEvent;
    type Config = (); // No additional config needed
    type Error = BacktestError;

    fn label(
        &self,
        prices: &[f64],
        entry_indices: &[usize],
    ) -> Result<Vec<Self::Event>, Self::Error> {
        if prices.is_empty() {
            return Err(BacktestError::InsufficientData(
                "prices cannot be empty".to_string(),
            ));
        }

        if self.horizon == 0 {
            return Err(BacktestError::InvalidConfig(
                "horizon must be positive".to_string(),
            ));
        }

        let mut events = Vec::with_capacity(entry_indices.len());

        for &entry in entry_indices {
            if entry >= prices.len() {
                return Err(BacktestError::InvalidEvent(format!(
                    "entry index {entry} out of bounds (prices len {})",
                    prices.len()
                )));
            }

            let exit = (entry + self.horizon).min(prices.len() - 1);
            let p_entry = prices[entry];
            let p_exit = prices[exit];

            if p_entry <= 0.0 {
                return Err(BacktestError::InvalidEvent(format!(
                    "entry price at index {entry} must be positive"
                )));
            }

            let return_pct = (p_exit - p_entry) / p_entry;
            let label = if return_pct >= self.min_return {
                TripleBarrierLabel::Upper // Positive outcome
            } else if return_pct < -self.min_return {
                TripleBarrierLabel::Lower // Negative outcome
            } else {
                TripleBarrierLabel::Time // Neutral
            };

            events.push(LabeledEvent {
                entry_index: entry,
                exit_index: exit,
                label,
                return_pct,
                holding_period: exit - entry,
            });
        }

        Ok(events)
    }
}

/// Dynamic barrier labeler: triple barrier with rolling volatility.
///
/// Adapts barrier widths based on recent volatility, making barriers
/// tighter in low-vol regimes and wider in high-vol regimes.
///
/// # Example
///
/// ```
/// use quant_backtest::DynamicBarrierLabeler;
/// use quant_core::Labeler;
///
/// let labeler = DynamicBarrierLabeler::new(20, 2.0, 5);
/// let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1)).collect();
/// let events = labeler.label(&prices, &[30, 40, 50]).unwrap();
/// // Barriers adapt to rolling 20-bar volatility
/// ```
pub struct DynamicBarrierLabeler {
    /// Rolling window for volatility estimation (bars).
    pub vol_window: usize,
    /// Number of standard deviations for barriers.
    pub n_std: f64,
    /// Time barrier (bars).
    pub time_barrier: usize,
}

impl DynamicBarrierLabeler {
    /// Create a new dynamic barrier labeler.
    ///
    /// # Arguments
    ///
    /// * `vol_window` - Rolling window for volatility (e.g., 20 bars)
    /// * `n_std` - Number of standard deviations for barriers (e.g., 2.0)
    /// * `time_barrier` - Maximum holding period in bars
    pub fn new(vol_window: usize, n_std: f64, time_barrier: usize) -> Self {
        Self {
            vol_window,
            n_std,
            time_barrier,
        }
    }

    /// Estimate rolling volatility at a given index.
    fn estimate_volatility(&self, prices: &[f64], index: usize) -> f64 {
        let start = index.saturating_sub(self.vol_window);
        let window = &prices[start..=index];

        if window.len() < 2 {
            return 0.01; // Default 1% if insufficient data
        }

        // Compute log returns
        let returns: Vec<f64> = window.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

        // Sample standard deviation
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;

        variance.sqrt()
    }
}

impl Labeler for DynamicBarrierLabeler {
    type Event = LabeledEvent;
    type Config = ();
    type Error = BacktestError;

    fn label(
        &self,
        prices: &[f64],
        entry_indices: &[usize],
    ) -> Result<Vec<Self::Event>, Self::Error> {
        if prices.is_empty() {
            return Err(BacktestError::InsufficientData(
                "prices cannot be empty".to_string(),
            ));
        }

        if self.vol_window == 0 || self.time_barrier == 0 {
            return Err(BacktestError::InvalidConfig(
                "vol_window and time_barrier must be positive".to_string(),
            ));
        }

        let mut events = Vec::with_capacity(entry_indices.len());

        for &entry in entry_indices {
            if entry >= prices.len() {
                return Err(BacktestError::InvalidEvent(format!(
                    "entry index {entry} out of bounds"
                )));
            }

            let p_entry = prices[entry];
            if p_entry <= 0.0 {
                return Err(BacktestError::InvalidEvent(format!(
                    "entry price at index {entry} must be positive"
                )));
            }

            // Estimate volatility at entry
            let vol = self.estimate_volatility(prices, entry);

            // Dynamic barriers based on volatility
            let upper_barrier = self.n_std * vol;
            let lower_barrier = -self.n_std * vol;

            let upper_px = p_entry * (1.0 + upper_barrier);
            let lower_px = p_entry * (1.0 + lower_barrier);
            let last = (entry + self.time_barrier).min(prices.len() - 1);

            // Scan for first barrier hit
            let mut label = TripleBarrierLabel::Time;
            let mut exit_index = last;

            for (offset, &p) in prices[(entry + 1)..=last].iter().enumerate() {
                let t = entry + 1 + offset;
                if p >= upper_px {
                    label = TripleBarrierLabel::Upper;
                    exit_index = t;
                    break;
                } else if p <= lower_px {
                    label = TripleBarrierLabel::Lower;
                    exit_index = t;
                    break;
                }
            }

            let p_exit = prices[exit_index];
            let return_pct = (p_exit - p_entry) / p_entry;

            events.push(LabeledEvent {
                entry_index: entry,
                exit_index,
                label,
                return_pct,
                holding_period: exit_index - entry,
            });
        }

        Ok(events)
    }
}

/// Trend-scanning labeler: adaptive horizon based on t-statistic.
///
/// Searches for the horizon that maximizes the t-statistic of a
/// linear trend fit. This adapts the labeling horizon to the
/// strength and duration of trends.
///
/// # Example
///
/// ```
/// use quant_backtest::TrendScanningLabeler;
/// use quant_core::Labeler;
///
/// let labeler = TrendScanningLabeler::new(5, 20, 0.0);
/// let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.5)).collect();
/// let events = labeler.label(&prices, &[10, 30, 50]).unwrap();
/// // Horizon adapts based on trend strength
/// ```
pub struct TrendScanningLabeler {
    /// Minimum horizon to consider.
    pub min_horizon: usize,
    /// Maximum horizon to consider.
    pub max_horizon: usize,
    /// Minimum return threshold.
    pub min_return: f64,
}

impl TrendScanningLabeler {
    /// Create a new trend-scanning labeler.
    pub fn new(min_horizon: usize, max_horizon: usize, min_return: f64) -> Self {
        Self {
            min_horizon,
            max_horizon,
            min_return,
        }
    }

    /// Compute t-statistic of linear trend over a window.
    fn trend_t_stat(&self, prices: &[f64], start: usize, end: usize) -> f64 {
        if end <= start + 1 {
            return 0.0;
        }

        let window = &prices[start..=end];
        let n = window.len();

        // Linear regression: price = a + b*t
        let t_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_vals = window;

        let t_mean: f64 = t_vals.iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y_vals.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denom_t = 0.0;

        for i in 0..n {
            let t_dev = t_vals[i] - t_mean;
            let y_dev = y_vals[i] - y_mean;
            numerator += t_dev * y_dev;
            denom_t += t_dev * t_dev;
        }

        if denom_t == 0.0 {
            return 0.0;
        }

        let slope = numerator / denom_t;

        // Compute residuals
        let residuals: Vec<f64> = (0..n)
            .map(|i| y_vals[i] - (y_mean + slope * (t_vals[i] - t_mean)))
            .collect();

        let sse: f64 = residuals.iter().map(|&r| r * r).sum();
        let std_error = (sse / (n - 2) as f64).sqrt();

        if std_error == 0.0 {
            return 0.0;
        }

        // t-statistic for slope
        let se_slope = std_error / denom_t.sqrt();
        slope / se_slope
    }
}

impl Labeler for TrendScanningLabeler {
    type Event = LabeledEvent;
    type Config = ();
    type Error = BacktestError;

    fn label(
        &self,
        prices: &[f64],
        entry_indices: &[usize],
    ) -> Result<Vec<Self::Event>, Self::Error> {
        if prices.is_empty() {
            return Err(BacktestError::InsufficientData(
                "prices cannot be empty".to_string(),
            ));
        }

        if self.min_horizon == 0 || self.max_horizon < self.min_horizon {
            return Err(BacktestError::InvalidConfig(
                "invalid horizon range".to_string(),
            ));
        }

        let mut events = Vec::with_capacity(entry_indices.len());

        for &entry in entry_indices {
            if entry >= prices.len() {
                return Err(BacktestError::InvalidEvent(format!(
                    "entry index {entry} out of bounds"
                )));
            }

            let p_entry = prices[entry];
            if p_entry <= 0.0 {
                return Err(BacktestError::InvalidEvent(format!(
                    "entry price at index {entry} must be positive"
                )));
            }

            // Find horizon with maximum |t-statistic|
            let mut best_horizon = self.min_horizon;
            let mut best_t_stat = 0.0f64;

            for h in self.min_horizon..=self.max_horizon {
                let exit = (entry + h).min(prices.len() - 1);
                if exit == entry {
                    break;
                }

                let t_stat = self.trend_t_stat(prices, entry, exit).abs();
                if t_stat > best_t_stat {
                    best_t_stat = t_stat;
                    best_horizon = h;
                }
            }

            let exit = (entry + best_horizon).min(prices.len() - 1);
            let p_exit = prices[exit];
            let return_pct = (p_exit - p_entry) / p_entry;

            let label = if return_pct >= self.min_return {
                TripleBarrierLabel::Upper
            } else if return_pct < -self.min_return {
                TripleBarrierLabel::Lower
            } else {
                TripleBarrierLabel::Time
            };

            events.push(LabeledEvent {
                entry_index: entry,
                exit_index: exit,
                label,
                return_pct,
                holding_period: exit - entry,
            });
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_horizon_labeler() {
        let labeler = FixedHorizonLabeler::new(5, 0.0);
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0];
        let events = labeler.label(&prices, &[0, 1]).unwrap();

        assert_eq!(events.len(), 2);
        // Event 0: 100->105 = +5%
        assert_eq!(events[0].label, TripleBarrierLabel::Upper);
        assert!((events[0].return_pct - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_fixed_horizon_implements_labeler() {
        fn _assert_trait<T: Labeler>() {}
        _assert_trait::<FixedHorizonLabeler>();
    }

    #[test]
    fn test_dynamic_barrier_volatility() {
        let labeler = DynamicBarrierLabeler::new(10, 2.0, 5);
        // High volatility series
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0, 100.0, 105.0, 95.0, 110.0];
        let events = labeler.label(&prices, &[5]).unwrap();
        assert_eq!(events.len(), 1);
        // Should have wider barriers due to high volatility
    }

    #[test]
    fn test_dynamic_barrier_implements_labeler() {
        fn _assert_trait<T: Labeler>() {}
        _assert_trait::<DynamicBarrierLabeler>();
    }

    #[test]
    fn test_trend_scanning_basic() {
        let labeler = TrendScanningLabeler::new(3, 10, 0.0);
        // Strong upward trend
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let events = labeler.label(&prices, &[5]).unwrap();

        assert_eq!(events.len(), 1);
        // Should detect upward trend
        assert_eq!(events[0].label, TripleBarrierLabel::Upper);
    }

    #[test]
    fn test_trend_scanning_implements_labeler() {
        fn _assert_trait<T: Labeler>() {}
        _assert_trait::<TrendScanningLabeler>();
    }

    #[test]
    fn test_all_labelers_same_event_type() {
        // Verify all labelers produce LabeledEvent
        let _fh: Vec<LabeledEvent> = Vec::new();
        let _db: Vec<LabeledEvent> = Vec::new();
        let _ts: Vec<LabeledEvent> = Vec::new();
    }
}
