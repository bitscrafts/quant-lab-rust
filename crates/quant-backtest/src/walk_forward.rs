//! Walk-forward cross-validation (AFML Ch. 7).
//!
//! Walk-forward analysis tests a strategy on out-of-sample data
//! using a moving or expanding time window. Unlike k-fold CV,
//! walk-forward respects temporal ordering and avoids look-ahead bias.

use crate::error::BacktestError;
use quant_core::CrossValidator;

/// Walk-forward cross-validation configuration.
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// Number of in-sample bars for training.
    pub in_sample_bars: usize,
    /// Number of out-of-sample bars for testing.
    pub out_of_sample_bars: usize,
    /// Step size between windows (bars to advance).
    pub step_size: usize,
    /// If true, in-sample window expands (anchored).
    /// If false, in-sample window rolls (moving window).
    pub anchored: bool,
}

impl WalkForwardConfig {
    /// Create a rolling (non-anchored) walk-forward configuration.
    ///
    /// # Arguments
    ///
    /// * `in_sample_bars` - Training window size
    /// * `out_of_sample_bars` - Testing window size
    /// * `step_size` - Bars to advance between windows
    ///
    /// # Example
    ///
    /// ```
    /// use quant_backtest::WalkForwardConfig;
    ///
    /// // 60-bar training, 20-bar testing, advance 20 bars each step
    /// let config = WalkForwardConfig::rolling(60, 20, 20);
    /// ```
    pub fn rolling(in_sample_bars: usize, out_of_sample_bars: usize, step_size: usize) -> Self {
        Self {
            in_sample_bars,
            out_of_sample_bars,
            step_size,
            anchored: false,
        }
    }

    /// Create an anchored (expanding) walk-forward configuration.
    ///
    /// # Arguments
    ///
    /// * `initial_in_sample_bars` - Initial training window size
    /// * `out_of_sample_bars` - Testing window size
    /// * `step_size` - Bars to advance between windows
    ///
    /// # Example
    ///
    /// ```
    /// use quant_backtest::WalkForwardConfig;
    ///
    /// // Start with 60-bar training, test on 20 bars,
    /// // expand training window by 20 bars each step
    /// let config = WalkForwardConfig::anchored(60, 20, 20);
    /// ```
    pub fn anchored(
        initial_in_sample_bars: usize,
        out_of_sample_bars: usize,
        step_size: usize,
    ) -> Self {
        Self {
            in_sample_bars: initial_in_sample_bars,
            out_of_sample_bars,
            step_size,
            anchored: true,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), BacktestError> {
        if self.in_sample_bars == 0 {
            return Err(BacktestError::InvalidConfig(
                "in_sample_bars must be positive".to_string(),
            ));
        }
        if self.out_of_sample_bars == 0 {
            return Err(BacktestError::InvalidConfig(
                "out_of_sample_bars must be positive".to_string(),
            ));
        }
        if self.step_size == 0 {
            return Err(BacktestError::InvalidConfig(
                "step_size must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// A single walk-forward train/test split.
#[derive(Debug, Clone, PartialEq)]
pub struct WalkForwardSplit {
    /// Training sample indices (bar indices, not event indices).
    pub train_indices: Vec<usize>,
    /// Testing sample indices (bar indices, not event indices).
    pub test_indices: Vec<usize>,
    /// Window ID (0-indexed).
    pub window_id: usize,
}

/// Walk-forward cross-validator.
///
/// Generates walk-forward splits using either rolling or anchored windows.
///
/// # Example
///
/// ```
/// use quant_backtest::{WalkForward, WalkForwardConfig};
/// use quant_core::CrossValidator;
///
/// let config = WalkForwardConfig::rolling(60, 20, 20);
/// let wf = WalkForward::new(config);
///
/// // Generate splits for 100 bars (n_samples is not used for walk-forward)
/// let splits = wf.splits(0, 100);
/// // Should have 2 splits: [0-60,60-80], [20-80,80-100]
/// assert_eq!(splits.len(), 2);
/// ```
pub struct WalkForward {
    config: WalkForwardConfig,
}

impl WalkForward {
    /// Create a new walk-forward validator.
    pub fn new(config: WalkForwardConfig) -> Self {
        Self { config }
    }

    /// Generate walk-forward splits for a given number of bars.
    ///
    /// # Arguments
    ///
    /// * `n_bars` - Total number of time bars
    ///
    /// # Returns
    ///
    /// A vector of splits. Each split contains training and testing
    /// bar indices.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or if there
    /// are insufficient bars.
    pub fn generate_splits(&self, n_bars: usize) -> Result<Vec<WalkForwardSplit>, BacktestError> {
        self.config.validate()?;

        if n_bars < self.config.in_sample_bars + self.config.out_of_sample_bars {
            return Err(BacktestError::InsufficientData(format!(
                "Need at least {} bars, got {}",
                self.config.in_sample_bars + self.config.out_of_sample_bars,
                n_bars
            )));
        }

        let mut splits = Vec::new();
        let mut window_id = 0;
        let mut train_start = 0;

        loop {
            let train_end = if self.config.anchored {
                // Anchored: train_end advances with each window
                train_start + self.config.in_sample_bars + (window_id * self.config.step_size)
            } else {
                // Rolling: fixed window size
                train_start + self.config.in_sample_bars
            };

            let test_start = train_end;
            let test_end = test_start + self.config.out_of_sample_bars;

            // Check if we have enough data for this window
            if test_end > n_bars {
                break;
            }

            // Generate indices
            let train_indices: Vec<usize> = (train_start..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            splits.push(WalkForwardSplit {
                train_indices,
                test_indices,
                window_id,
            });

            // Advance window
            if !self.config.anchored {
                train_start += self.config.step_size;
            }
            window_id += 1;
        }

        if splits.is_empty() {
            return Err(BacktestError::InsufficientData(
                "Not enough data to create any walk-forward windows".to_string(),
            ));
        }

        Ok(splits)
    }
}

impl CrossValidator for WalkForward {
    type Split = WalkForwardSplit;

    fn splits(&self, _n_samples: usize, n_bars: usize) -> Vec<Self::Split> {
        // Note: n_samples is not used for walk-forward (we split by bars, not samples)
        self.generate_splits(n_bars).unwrap_or_default()
    }
}

/// Compute the walk-forward efficiency (WFE) ratio.
///
/// WFE = out_of_sample_sharpe / in_sample_sharpe
///
/// A ratio close to 1.0 indicates the strategy performs similarly
/// in-sample and out-of-sample. Ratios significantly below 1.0
/// suggest overfitting.
///
/// # Arguments
///
/// * `in_sample_sharpe` - Sharpe ratio on in-sample data
/// * `out_of_sample_sharpe` - Sharpe ratio on out-of-sample data
///
/// # Returns
///
/// The WFE ratio.
///
/// # Example
///
/// ```
/// use quant_backtest::walk_forward_efficiency;
///
/// let wfe = walk_forward_efficiency(1.2, 0.8);
/// assert!((wfe - 0.667).abs() < 0.001);
/// ```
pub fn walk_forward_efficiency(in_sample_sharpe: f64, out_of_sample_sharpe: f64) -> f64 {
    if in_sample_sharpe == 0.0 {
        return 0.0;
    }
    out_of_sample_sharpe / in_sample_sharpe
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_forward_rolling_splits() {
        let config = WalkForwardConfig::rolling(60, 20, 20);
        let wf = WalkForward::new(config);
        let splits = wf.generate_splits(100).unwrap();

        // Should have 2 splits: [0-60,60-80], [20-80,80-100]
        assert_eq!(splits.len(), 2);

        // First split
        assert_eq!(splits[0].train_indices, (0..60).collect::<Vec<_>>());
        assert_eq!(splits[0].test_indices, (60..80).collect::<Vec<_>>());
        assert_eq!(splits[0].window_id, 0);

        // Second split
        assert_eq!(splits[1].train_indices, (20..80).collect::<Vec<_>>());
        assert_eq!(splits[1].test_indices, (80..100).collect::<Vec<_>>());
        assert_eq!(splits[1].window_id, 1);
    }

    #[test]
    fn test_walk_forward_anchored_splits() {
        let config = WalkForwardConfig::anchored(60, 20, 20);
        let wf = WalkForward::new(config);
        let splits = wf.generate_splits(100).unwrap();

        // Should have 2 splits: [0-60,60-80], [0-80,80-100]
        assert_eq!(splits.len(), 2);

        // First split
        assert_eq!(splits[0].train_indices, (0..60).collect::<Vec<_>>());
        assert_eq!(splits[0].test_indices, (60..80).collect::<Vec<_>>());

        // Second split (training expands)
        assert_eq!(splits[1].train_indices, (0..80).collect::<Vec<_>>());
        assert_eq!(splits[1].test_indices, (80..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_walk_forward_insufficient_data() {
        let config = WalkForwardConfig::rolling(60, 20, 20);
        let wf = WalkForward::new(config);
        let result = wf.generate_splits(50);
        assert!(result.is_err());
    }

    #[test]
    fn test_walk_forward_efficiency_ratio() {
        let wfe = walk_forward_efficiency(1.2, 0.8);
        assert!((wfe - 0.666666).abs() < 0.001);
    }

    #[test]
    fn test_walk_forward_implements_cross_validator() {
        // Compile-time test
        fn _assert_trait<T: CrossValidator>() {}
        _assert_trait::<WalkForward>();
    }

    #[test]
    fn test_walk_forward_via_trait() {
        let config = WalkForwardConfig::rolling(60, 20, 20);
        let wf = WalkForward::new(config);
        let splits = wf.splits(0, 100);
        assert_eq!(splits.len(), 2);
    }
}
