//! CUSUM structural break detection.
//!
//! The CUSUM (Cumulative Sum) filter detects structural breaks in
//! time series by monitoring deviations from an expected value.

use crate::error::TimeSeriesError;
use quant_core::{StructuralBreak, StructuralBreakDetector};

/// CUSUM filter configuration.
#[derive(Debug, Clone)]
pub struct CusumConfig {
    /// Detection threshold (e.g., 5.0 for 5-sigma events).
    pub threshold: f64,
    /// Expected drift (mean shift to detect).
    pub drift: f64,
}

impl CusumConfig {
    /// Create a new CUSUM configuration.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Detection threshold (higher = fewer false positives)
    /// * `drift` - Expected mean shift to detect (typically > 0)
    ///
    /// # Example
    ///
    /// ```
    /// use quant_timeseries::CusumConfig;
    ///
    /// let config = CusumConfig::new(5.0, 0.5);
    /// ```
    pub fn new(threshold: f64, drift: f64) -> Self {
        Self { threshold, drift }
    }
}

/// CUSUM structural break detector.
///
/// Implements the CUSUM filter for detecting mean shifts in time series.
/// The filter maintains cumulative sums of positive and negative deviations
/// from the expected mean, triggering when either sum exceeds the threshold.
///
/// # Example
///
/// ```
/// use quant_timeseries::{CusumDetector, CusumConfig};
/// use quant_core::StructuralBreakDetector;
///
/// let config = CusumConfig::new(5.0, 0.5);
/// let detector = CusumDetector::new(config);
///
/// // Stationary series with mean shift at t=50
/// let mut data: Vec<f64> = (0..50).map(|_| 0.0).collect();
/// data.extend((0..50).map(|_| 1.0));
///
/// let breaks = detector.detect(&data).unwrap();
/// // Should detect break near t=50
/// ```
pub struct CusumDetector {
    config: CusumConfig,
}

impl CusumDetector {
    /// Create a new CUSUM detector with the given configuration.
    pub fn new(config: CusumConfig) -> Self {
        Self { config }
    }

    /// Detect structural breaks using the CUSUM filter.
    ///
    /// The CUSUM filter maintains two cumulative sums:
    /// - S_high: cumulative sum of positive deviations
    /// - S_low: cumulative sum of negative deviations
    ///
    /// A break is detected when either sum exceeds the threshold.
    fn detect_internal(&self, data: &[f64]) -> Result<Vec<StructuralBreak>, TimeSeriesError> {
        if data.len() < 2 {
            return Err(TimeSeriesError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }

        let mut breaks = Vec::new();
        let mut s_high = 0.0;
        let mut s_low = 0.0;

        for (i, &value) in data.iter().enumerate() {
            // Update cumulative sums
            s_high = (s_high + value - self.config.drift).max(0.0);
            s_low = (s_low - value - self.config.drift).max(0.0);

            // Check for threshold breach
            if s_high > self.config.threshold {
                breaks.push(StructuralBreak {
                    index: i,
                    statistic: s_high,
                    confidence: 0.95, // Standard 95% confidence
                });
                // Reset after detection
                s_high = 0.0;
                s_low = 0.0;
            } else if s_low > self.config.threshold {
                breaks.push(StructuralBreak {
                    index: i,
                    statistic: s_low,
                    confidence: 0.95,
                });
                // Reset after detection
                s_high = 0.0;
                s_low = 0.0;
            }
        }

        Ok(breaks)
    }
}

impl StructuralBreakDetector for CusumDetector {
    type Error = TimeSeriesError;

    fn detect(&self, data: &[f64]) -> Result<Vec<StructuralBreak>, Self::Error> {
        self.detect_internal(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cusum_no_break() {
        // Stationary series (white noise around 0)
        let data = vec![0.1, -0.05, 0.08, -0.02, 0.03, -0.04];
        let config = CusumConfig::new(5.0, 0.5);
        let detector = CusumDetector::new(config);
        let breaks = detector.detect(&data).unwrap();
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_cusum_single_break() {
        // Mean shift at t=5 from 0 to 1
        let mut data: Vec<f64> = (0..10).map(|_| 0.0).collect();
        data.extend((0..10).map(|_| 1.0));

        let config = CusumConfig::new(3.0, 0.5);
        let detector = CusumDetector::new(config);
        let breaks = detector.detect(&data).unwrap();

        // Should detect at least one break
        assert!(!breaks.is_empty());
        // Break should be detected after t=10 (when shift occurs)
        assert!(breaks[0].index >= 10);
    }

    #[test]
    fn test_cusum_threshold_sensitivity() {
        // Same data, different thresholds
        let mut data: Vec<f64> = (0..20).map(|_| 0.0).collect();
        data.extend((0..20).map(|_| 2.0));

        let config_low = CusumConfig::new(1.0, 0.5);
        let config_high = CusumConfig::new(10.0, 0.5);

        let detector_low = CusumDetector::new(config_low);
        let detector_high = CusumDetector::new(config_high);

        let breaks_low = detector_low.detect(&data).unwrap();
        let breaks_high = detector_high.detect(&data).unwrap();

        // Lower threshold should detect more (or equal) breaks
        assert!(breaks_low.len() >= breaks_high.len());
    }

    #[test]
    fn test_cusum_implements_detector() {
        // Compile-time test: CusumDetector implements StructuralBreakDetector
        fn _assert_trait<T: StructuralBreakDetector>() {}
        _assert_trait::<CusumDetector>();
    }

    #[test]
    fn test_cusum_insufficient_data() {
        let config = CusumConfig::new(5.0, 0.5);
        let detector = CusumDetector::new(config);
        let result = detector.detect(&[1.0]);
        assert!(result.is_err());
    }
}
