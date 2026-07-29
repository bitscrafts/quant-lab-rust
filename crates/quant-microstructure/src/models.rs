//! Market microstructure trait implementations.
//!
//! This module provides wrappers implementing the traits from quant-core.

use crate::error::MicroError;
use crate::impact::{linear_impact, sqrt_impact};
use crate::orderbook::OrderBook;
use crate::types::{Fill, Order, Side};
use quant_core::{ImpactModel, OrderBookOps};

/// Square-root market impact model (Almgren-Chriss).
///
/// Impact = σ * sqrt(Q / V)
///
/// # Example
///
/// ```
/// use quant_microstructure::SqrtImpactModel;
/// use quant_core::ImpactModel;
///
/// let model = SqrtImpactModel::new(0.02); // 2% volatility
/// let impact = model.impact(1000.0, 1_000_000.0);
/// assert!(impact > 0.0);
/// ```
pub struct SqrtImpactModel {
    /// Daily volatility (e.g., 0.02 for 2%).
    pub volatility: f64,
}

impl SqrtImpactModel {
    /// Create a new square-root impact model.
    ///
    /// # Arguments
    ///
    /// * `volatility` - Daily volatility
    pub fn new(volatility: f64) -> Self {
        Self { volatility }
    }
}

impl ImpactModel for SqrtImpactModel {
    fn impact(&self, volume: f64, daily_volume: f64) -> f64 {
        sqrt_impact(self.volatility, volume, daily_volume)
    }
}

/// Linear temporary impact model.
///
/// Impact = η * (Q / V)
///
/// # Example
///
/// ```
/// use quant_microstructure::LinearImpactModel;
/// use quant_core::ImpactModel;
///
/// let model = LinearImpactModel::new(0.1); // eta = 0.1
/// let impact = model.impact(500.0, 10_000.0);
/// assert!((impact - 0.005).abs() < 1e-9);
/// ```
pub struct LinearImpactModel {
    /// Impact coefficient η.
    pub eta: f64,
}

impl LinearImpactModel {
    /// Create a new linear impact model.
    ///
    /// # Arguments
    ///
    /// * `eta` - Impact coefficient
    pub fn new(eta: f64) -> Self {
        Self { eta }
    }
}

impl ImpactModel for LinearImpactModel {
    fn impact(&self, volume: f64, daily_volume: f64) -> f64 {
        linear_impact(self.eta, volume, daily_volume)
    }
}

// OrderBook already implements the necessary methods, we just need to
// implement the OrderBookOps trait for it.

impl OrderBookOps for OrderBook {
    type Error = MicroError;
    type Order = Order;
    type Trade = Fill;

    fn add_order(&mut self, order: Self::Order) -> Result<u64, Self::Error> {
        let order_id = order.id;
        OrderBook::add_order(self, order)?;
        Ok(order_id)
    }

    fn cancel_order(&mut self, order_id: u64) -> Result<(), Self::Error> {
        OrderBook::cancel_order(self, order_id)?;
        Ok(())
    }

    fn market_order(
        &mut self,
        is_buy: bool,
        quantity: f64,
    ) -> Result<Vec<Self::Trade>, Self::Error> {
        let side = if is_buy { Side::Bid } else { Side::Ask };
        let quantity_u64 = quantity as u64;

        let fills = OrderBook::market_order(self, side, quantity_u64);
        Ok(fills)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_impact_implements_impact_model() {
        fn _assert_trait<T: ImpactModel>() {}
        _assert_trait::<SqrtImpactModel>();
    }

    #[test]
    fn test_sqrt_impact_model() {
        let model = SqrtImpactModel::new(0.02);
        let impact = model.impact(1000.0, 1_000_000.0);
        let expected = 0.02 * (0.001_f64).sqrt();
        assert!((impact - expected).abs() < 1e-9);
    }

    #[test]
    fn test_linear_impact_implements_impact_model() {
        fn _assert_trait<T: ImpactModel>() {}
        _assert_trait::<LinearImpactModel>();
    }

    #[test]
    fn test_linear_impact_model() {
        let model = LinearImpactModel::new(0.1);
        let impact = model.impact(500.0, 10_000.0);
        assert!((impact - 0.005).abs() < 1e-9);
    }

    #[test]
    fn test_order_book_implements_order_book_ops() {
        fn _assert_trait<T: OrderBookOps>() {}
        _assert_trait::<OrderBook>();
    }

    #[test]
    fn test_order_book_ops_add_order() {
        let mut book = OrderBook::new(1);
        let order = Order {
            id: 1,
            side: Side::Bid,
            price: 100,
            quantity: 10,
            timestamp: 1,
        };

        let order_id = OrderBookOps::add_order(&mut book, order).unwrap();
        assert_eq!(order_id, 1);
    }

    #[test]
    fn test_order_book_ops_cancel_order() {
        let mut book = OrderBook::new(1);
        let order = Order {
            id: 1,
            side: Side::Bid,
            price: 100,
            quantity: 10,
            timestamp: 1,
        };

        OrderBookOps::add_order(&mut book, order).unwrap();
        OrderBookOps::cancel_order(&mut book, 1).unwrap();

        // Trying to cancel again should fail
        assert!(OrderBookOps::cancel_order(&mut book, 1).is_err());
    }

    #[test]
    fn test_order_book_ops_market_order() {
        let mut book = OrderBook::new(1);

        // Add some asks
        OrderBookOps::add_order(
            &mut book,
            Order {
                id: 1,
                side: Side::Ask,
                price: 101,
                quantity: 10,
                timestamp: 1,
            },
        )
        .unwrap();

        OrderBookOps::add_order(
            &mut book,
            Order {
                id: 2,
                side: Side::Ask,
                price: 102,
                quantity: 20,
                timestamp: 2,
            },
        )
        .unwrap();

        // Market buy should fill against asks
        let fills = OrderBookOps::market_order(&mut book, true, 15.0).unwrap();

        assert!(!fills.is_empty());
        // Should fill 10 @ 101 and 5 @ 102
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].quantity, 10);
        assert_eq!(fills[1].quantity, 5);
    }

    #[test]
    fn test_sqrt_impact_scales() {
        let model = SqrtImpactModel::new(0.02);
        let small = model.impact(500.0, 1_000_000.0);
        let large = model.impact(1000.0, 1_000_000.0);
        let ratio = large / small;
        // Should scale as sqrt(2)
        assert!((ratio - 2.0_f64.sqrt()).abs() < 1e-6);
    }
}
