//! Bet sizing trait.
//!
//! Defines the [`BetSizer`] trait for determining position sizes
//! based on historical returns or win probabilities.

/// A bet sizer computes position sizes from historical performance.
///
/// Different sizing strategies (Kelly, fixed, equal-weighted) implement
/// this trait to provide consistent position sizing across backtesting
/// frameworks.
pub trait BetSizer {
    /// Compute the position size as a fraction of capital.
    ///
    /// # Arguments
    ///
    /// * `returns` - Historical returns for this strategy or signal
    ///
    /// # Returns
    ///
    /// A fraction in the range [0.0, 1.0] representing the portion
    /// of capital to allocate to this position. Values outside this
    /// range indicate no position (negative) or leveraged positions
    /// (> 1.0), depending on the implementation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sizer = KellyBetSizer::new(0.5); // Half-Kelly
    /// let returns = vec![0.02, -0.01, 0.03, -0.02];
    /// let position_size = sizer.size(&returns);
    /// println!("Allocate {:.1}% of capital", position_size * 100.0);
    /// ```
    fn size(&self, returns: &[f64]) -> f64;
}
