//! Triple-barrier labeling (AFML Ch. 3).
//!
//! For each entry point the method places three barriers around the
//! entry price:
//!
//!   - upper: profit-taking threshold (e.g. +2%)
//!   - lower: stop-loss threshold (e.g. -2%)
//!   - time: maximum holding period in bars
//!
//! The first barrier touched determines the label. If neither horizontal
//! barrier is hit within the time barrier, the time barrier labels the
//! event.
//!
//! Returns are computed as $r_t = (P_{exit} - P_{entry}) / P_{entry}$.

use crate::error::BacktestError;

/// Triple-barrier label configuration.
#[derive(Debug, Clone)]
pub struct TripleBarrierConfig {
    /// Upper barrier: profit-taking threshold (e.g., 0.02 = 2%).
    pub upper_barrier: f64,
    /// Lower barrier: stop-loss threshold (e.g., -0.02 = -2%).
    pub lower_barrier: f64,
    /// Time barrier: maximum holding period in bars.
    pub time_barrier: usize,
    /// Minimum return to label as positive at time barrier.
    pub min_return: f64,
}

impl TripleBarrierConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), BacktestError> {
        if self.upper_barrier <= 0.0 {
            return Err(BacktestError::InvalidConfig(
                "upper_barrier must be positive".to_string(),
            ));
        }
        if self.lower_barrier >= 0.0 {
            return Err(BacktestError::InvalidConfig(
                "lower_barrier must be negative".to_string(),
            ));
        }
        if self.time_barrier == 0 {
            return Err(BacktestError::InvalidConfig(
                "time_barrier must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Triple-barrier label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripleBarrierLabel {
    /// Upper barrier hit first (profit-taking).
    Upper,
    /// Lower barrier hit first (stop-loss).
    Lower,
    /// Time barrier hit (timeout).
    Time,
}

/// An event with its label and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledEvent {
    pub entry_index: usize,
    pub exit_index: usize,
    pub label: TripleBarrierLabel,
    pub return_pct: f64,
    pub holding_period: usize,
}

impl LabeledEvent {
    /// Convert the label to a binary outcome: Upper -> 1, Lower/Time -> 0.
    /// If `min_return` is positive, a Time barrier exit only yields 1 when
    /// the return exceeds `min_return`.
    pub fn to_binary(&self, min_return: f64) -> i32 {
        to_binary_label_helper(self.label, self, min_return)
    }
}

impl From<LabeledEvent> for i8 {
    /// Convert to ternary label: Upper -> 1, Time -> 0, Lower -> -1.
    fn from(event: LabeledEvent) -> Self {
        match event.label {
            TripleBarrierLabel::Upper => 1,
            TripleBarrierLabel::Time => 0,
            TripleBarrierLabel::Lower => -1,
        }
    }
}

impl From<&LabeledEvent> for i8 {
    /// Convert to ternary label: Upper -> 1, Time -> 0, Lower -> -1.
    fn from(event: &LabeledEvent) -> Self {
        match event.label {
            TripleBarrierLabel::Upper => 1,
            TripleBarrierLabel::Time => 0,
            TripleBarrierLabel::Lower => -1,
        }
    }
}

/// Convert a `TripleBarrierLabel` to a binary outcome:
/// Upper -> 1, Lower -> 0, Time -> 1 iff `event.return_pct >= min_return`.
pub fn to_binary_label_helper(
    label: TripleBarrierLabel,
    event: &LabeledEvent,
    min_return: f64,
) -> i32 {
    match label {
        TripleBarrierLabel::Upper => 1,
        TripleBarrierLabel::Lower => 0,
        TripleBarrierLabel::Time => {
            if event.return_pct >= min_return {
                1
            } else {
                0
            }
        }
    }
}

/// Apply triple-barrier labeling to a price series at the given entry
/// indices. Returns one `LabeledEvent` per entry index.
///
/// `entry_indices` must be sorted ascending and within `0..prices.len()-1`.
pub fn triple_barrier_label(
    prices: &[f64],
    entry_indices: &[usize],
    config: &TripleBarrierConfig,
) -> Result<Vec<LabeledEvent>, BacktestError> {
    config.validate()?;
    if prices.is_empty() {
        return Err(BacktestError::InsufficientData(
            "prices cannot be empty".to_string(),
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
        let event = label_one(prices, entry, config)?;
        events.push(event);
    }
    Ok(events)
}

/// Label a single event starting at `entry_index`.
fn label_one(
    prices: &[f64],
    entry_index: usize,
    config: &TripleBarrierConfig,
) -> Result<LabeledEvent, BacktestError> {
    let p_entry = prices[entry_index];
    if p_entry <= 0.0 {
        return Err(BacktestError::InvalidEvent(format!(
            "entry price at index {entry_index} must be positive"
        )));
    }
    let upper_px = p_entry * (1.0 + config.upper_barrier);
    let lower_px = p_entry * (1.0 + config.lower_barrier);
    let last = (entry_index + config.time_barrier).min(prices.len() - 1);

    for (t, &p) in prices
        .iter()
        .enumerate()
        .take(last + 1)
        .skip(entry_index + 1)
    {
        if p >= upper_px {
            return Ok(LabeledEvent {
                entry_index,
                exit_index: t,
                label: TripleBarrierLabel::Upper,
                return_pct: (p - p_entry) / p_entry,
                holding_period: t - entry_index,
            });
        }
        if p <= lower_px {
            return Ok(LabeledEvent {
                entry_index,
                exit_index: t,
                label: TripleBarrierLabel::Lower,
                return_pct: (p - p_entry) / p_entry,
                holding_period: t - entry_index,
            });
        }
    }
    // Time barrier hit.
    let p_exit = prices[last];
    Ok(LabeledEvent {
        entry_index,
        exit_index: last,
        label: TripleBarrierLabel::Time,
        return_pct: (p_exit - p_entry) / p_entry,
        holding_period: last - entry_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(upper: f64, lower: f64, t: usize) -> TripleBarrierConfig {
        TripleBarrierConfig {
            upper_barrier: upper,
            lower_barrier: lower,
            time_barrier: t,
            min_return: 0.0,
        }
    }

    #[test]
    fn test_upper_barrier_hit() {
        let prices = vec![100.0, 101.0, 103.0, 102.5];
        let events = triple_barrier_label(&prices, &[0], &cfg(0.02, -0.02, 5)).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].label, TripleBarrierLabel::Upper);
        assert!((events[0].return_pct - 0.03).abs() < 1e-9);
        assert_eq!(events[0].exit_index, 2);
        assert_eq!(events[0].holding_period, 2);
    }

    #[test]
    fn test_lower_barrier_hit() {
        let prices = vec![100.0, 99.0, 97.0, 98.0];
        let events = triple_barrier_label(&prices, &[0], &cfg(0.05, -0.02, 5)).unwrap();
        assert_eq!(events[0].label, TripleBarrierLabel::Lower);
        assert!((events[0].return_pct - (-0.03)).abs() < 1e-9);
    }

    #[test]
    fn test_time_barrier_hit() {
        let prices = vec![100.0, 100.5, 100.2, 100.1, 100.3];
        let events = triple_barrier_label(&prices, &[0], &cfg(0.02, -0.02, 5)).unwrap();
        assert_eq!(events[0].label, TripleBarrierLabel::Time);
        assert_eq!(events[0].holding_period, 4);
    }

    #[test]
    fn test_immediate_exit_on_first_bar() {
        let prices = vec![100.0, 105.0, 110.0];
        let events = triple_barrier_label(&prices, &[0], &cfg(0.02, -0.02, 5)).unwrap();
        assert_eq!(events[0].exit_index, 1);
        assert_eq!(events[0].holding_period, 1);
    }

    #[test]
    fn test_config_validation() {
        let bad = TripleBarrierConfig {
            upper_barrier: -0.01,
            lower_barrier: -0.02,
            time_barrier: 5,
            min_return: 0.0,
        };
        assert!(bad.validate().is_err());
        let bad2 = TripleBarrierConfig {
            upper_barrier: 0.01,
            lower_barrier: 0.01,
            time_barrier: 5,
            min_return: 0.0,
        };
        assert!(bad2.validate().is_err());
    }
}
