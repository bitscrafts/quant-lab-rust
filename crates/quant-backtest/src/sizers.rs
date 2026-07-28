//! Bet sizing implementations.
//!
//! This module provides various bet sizing strategies implementing
//! the `BetSizer` trait from quant-core.

use crate::kelly::kelly_from_returns;
use quant_core::BetSizer;

/// Kelly criterion bet sizer (full or fractional).
///
/// The Kelly criterion maximizes long-run growth by betting a
/// fraction of capital proportional to edge. Fractional Kelly
/// (e.g., half-Kelly) reduces variance at the cost of growth.
///
/// # Example
///
/// ```
/// use quant_backtest::KellyBetSizer;
/// use quant_core::BetSizer;
///
/// let sizer = KellyBetSizer::new(0.5); // Half-Kelly
/// let returns = vec![0.02, -0.01, 0.03, -0.02];
/// let position_size = sizer.size(&returns);
/// assert!(position_size >= 0.0 && position_size <= 1.0);
/// ```
pub struct KellyBetSizer {
    /// Kelly fraction (1.0 = full Kelly, 0.5 = half Kelly).
    pub fraction: f64,
}

impl KellyBetSizer {
    /// Create a new Kelly bet sizer.
    ///
    /// # Arguments
    ///
    /// * `fraction` - Kelly fraction (typically 0.5 for half-Kelly)
    pub fn new(fraction: f64) -> Self {
        Self { fraction }
    }

    /// Full Kelly (fraction = 1.0).
    pub fn full() -> Self {
        Self { fraction: 1.0 }
    }

    /// Half Kelly (fraction = 0.5).
    pub fn half() -> Self {
        Self { fraction: 0.5 }
    }
}

impl BetSizer for KellyBetSizer {
    fn size(&self, returns: &[f64]) -> f64 {
        let kelly_full = kelly_from_returns(returns);
        (self.fraction * kelly_full).clamp(0.0, 1.0)
    }
}

/// Fixed bet size (constant fraction of capital).
///
/// This strategy allocates a fixed fraction regardless of
/// historical performance.
///
/// # Example
///
/// ```
/// use quant_backtest::FixedBetSizer;
/// use quant_core::BetSizer;
///
/// let sizer = FixedBetSizer::new(0.1); // Always 10%
/// let returns = vec![0.02, -0.01];
/// assert_eq!(sizer.size(&returns), 0.1);
/// ```
pub struct FixedBetSizer {
    /// Fixed fraction to allocate.
    pub fraction: f64,
}

impl FixedBetSizer {
    /// Create a new fixed bet sizer.
    pub fn new(fraction: f64) -> Self {
        Self { fraction }
    }
}

impl BetSizer for FixedBetSizer {
    fn size(&self, _returns: &[f64]) -> f64 {
        self.fraction.clamp(0.0, 1.0)
    }
}

/// Equal-weighted bet size (always 100% / n_positions).
///
/// For single-position strategies, this is equivalent to 100%.
/// For multi-position portfolios, this would divide capital equally.
///
/// # Example
///
/// ```
/// use quant_backtest::EqualBetSizer;
/// use quant_core::BetSizer;
///
/// let sizer = EqualBetSizer::new(1); // Single position = 100%
/// let returns = vec![0.02, -0.01];
/// assert_eq!(sizer.size(&returns), 1.0);
/// ```
pub struct EqualBetSizer {
    /// Number of positions (for equal weighting).
    pub n_positions: usize,
}

impl EqualBetSizer {
    /// Create a new equal bet sizer.
    pub fn new(n_positions: usize) -> Self {
        Self {
            n_positions: n_positions.max(1),
        }
    }
}

impl BetSizer for EqualBetSizer {
    fn size(&self, _returns: &[f64]) -> f64 {
        1.0 / self.n_positions as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_implements_bet_sizer() {
        fn _assert_trait<T: BetSizer>() {}
        _assert_trait::<KellyBetSizer>();
    }

    #[test]
    fn test_fixed_implements_bet_sizer() {
        fn _assert_trait<T: BetSizer>() {}
        _assert_trait::<FixedBetSizer>();
    }

    #[test]
    fn test_equal_implements_bet_sizer() {
        fn _assert_trait<T: BetSizer>() {}
        _assert_trait::<EqualBetSizer>();
    }

    #[test]
    fn test_kelly_half() {
        let sizer = KellyBetSizer::half();
        let returns = vec![1.0, 1.0, 1.0, -1.0, -1.0]; // p=0.6, b=1, kelly=0.2
        let size = sizer.size(&returns);
        // Half of 0.2 = 0.1
        assert!((size - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_fixed_bet_size() {
        let sizer = FixedBetSizer::new(0.25);
        let returns = vec![0.1, -0.05, 0.2];
        assert_eq!(sizer.size(&returns), 0.25);
    }

    #[test]
    fn test_equal_bet_size() {
        let sizer = EqualBetSizer::new(4);
        let returns = vec![0.01, 0.02];
        assert_eq!(sizer.size(&returns), 0.25);
    }

    #[test]
    fn test_equal_bet_size_single_position() {
        let sizer = EqualBetSizer::new(1);
        let returns = vec![0.01];
        assert_eq!(sizer.size(&returns), 1.0);
    }
}
