//! Event labeling trait.
//!
//! Defines the [`Labeler`] trait for converting price series into
//! labeled events suitable for supervised learning.

use std::error::Error;

/// A labeler converts price data into labeled events.
///
/// Labelers implement different strategies for determining when to
/// enter trades and how to label the outcome (e.g., triple-barrier,
/// fixed-horizon, trend-scanning).
///
/// # Type Parameters
///
/// - `Event`: The type of labeled event produced (e.g., `LabeledEvent`)
/// - `Config`: Configuration type for the labeler
/// - `Error`: Error type for labeling failures
pub trait Labeler {
    /// The type of labeled event this labeler produces.
    type Event;

    /// The configuration type for this labeler.
    type Config;

    /// The error type returned when labeling fails.
    type Error: Error;

    /// Label a price series at the given entry indices.
    ///
    /// # Arguments
    ///
    /// * `prices` - The price series to label
    /// * `entry_indices` - Indices where trades are entered
    ///
    /// # Returns
    ///
    /// A vector of labeled events, one per entry index.
    ///
    /// # Errors
    ///
    /// Returns an error if labeling fails (e.g., invalid configuration,
    /// insufficient data, or invalid entry indices).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let labeler = TripleBarrierLabeler::new(config);
    /// let events = labeler.label(&prices, &entry_indices)?;
    /// for event in events {
    ///     println!("Return: {:.2}%", event.return_pct * 100.0);
    /// }
    /// ```
    fn label(&self, prices: &[f64], entry_indices: &[usize]) -> Result<Vec<Self::Event>, Self::Error>;
}
