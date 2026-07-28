//! Cross-validation trait.
//!
//! Defines the [`CrossValidator`] trait for generating train/test splits
//! that respect temporal dependencies and prevent data leakage.

/// A cross-validation split result.
///
/// Implementers should define their own split type containing
/// training and testing sample indices.
pub trait CrossValidator {
    /// The type of a single train/test split.
    type Split;

    /// Generate cross-validation splits for `n_samples` observations
    /// spanning `n_bars` time bars.
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of samples (e.g., labeled events)
    /// * `n_bars` - Total number of time bars in the dataset
    ///
    /// # Returns
    ///
    /// A vector of splits, where each split contains training and
    /// testing indices. The exact structure is defined by `Self::Split`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cv = PurgedKFold::new(5, 10);
    /// let splits = cv.splits(100, 252);
    /// for split in splits {
    ///     // Train model on split.train_indices
    ///     // Test model on split.test_indices
    /// }
    /// ```
    fn splits(&self, n_samples: usize, n_bars: usize) -> Vec<Self::Split>;
}
