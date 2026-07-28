//! Structural break detection trait.
//!
//! Defines the [`StructuralBreakDetector`] trait for identifying
//! regime changes and structural breaks in time series.

use std::error::Error;

/// A structural break represents a detected regime change.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralBreak {
    /// The time index where the break occurred.
    pub index: usize,
    /// The test statistic value at the break point.
    pub statistic: f64,
    /// Confidence level (e.g., 0.95 for 95% confidence).
    pub confidence: f64,
}

/// A structural break detector identifies regime changes in time series.
///
/// Different detection methods (CUSUM, Chow test, BIC) implement this
/// trait to provide a consistent interface for break detection.
pub trait StructuralBreakDetector {
    /// The error type returned when detection fails.
    type Error: Error;

    /// Detect structural breaks in a time series.
    ///
    /// # Arguments
    ///
    /// * `data` - The time series to analyze
    ///
    /// # Returns
    ///
    /// A vector of detected breaks, sorted by time index.
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails (e.g., insufficient data,
    /// invalid configuration).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let detector = CusumDetector::new(threshold, drift);
    /// let breaks = detector.detect(&returns)?;
    /// for brk in breaks {
    ///     println!("Break at index {} (stat: {:.3})", brk.index, brk.statistic);
    /// }
    /// ```
    fn detect(&self, data: &[f64]) -> Result<Vec<StructuralBreak>, Self::Error>;
}
