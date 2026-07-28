//! Sample weighting trait.
//!
//! Defines the [`SampleWeighter`] trait for computing sample weights
//! based on event characteristics (e.g., uniqueness, importance).

use std::error::Error;

/// A sample weighter assigns importance weights to samples.
///
/// Weights are used during model training to adjust for overlapping
/// events, class imbalance, or other factors that affect sample
/// representativeness.
pub trait SampleWeighter {
    /// The error type returned when weighting fails.
    type Error: Error;

    /// Compute sample weights for a collection of events.
    ///
    /// # Arguments
    ///
    /// * `events` - The events to weight (typically labeled events)
    /// * `n_bars` - Total number of time bars in the dataset
    ///
    /// # Returns
    ///
    /// A vector of weights, one per event, where weights sum to
    /// the number of events (average weight = 1.0).
    ///
    /// # Errors
    ///
    /// Returns an error if weighting fails (e.g., invalid event data).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let weighter = UniquenessWeighter;
    /// let weights = weighter.weights(&events, 252)?;
    /// for (event, weight) in events.iter().zip(weights.iter()) {
    ///     println!("Event {} weight: {:.3}", event.entry_index, weight);
    /// }
    /// ```
    fn weights<E>(&self, events: &[E], n_bars: usize) -> Result<Vec<f64>, Self::Error>
    where
        E: AsRef<Self::EventRef>;

    /// The event type this weighter operates on.
    ///
    /// This is typically a reference to an event structure containing
    /// entry and exit indices.
    type EventRef: ?Sized;
}
