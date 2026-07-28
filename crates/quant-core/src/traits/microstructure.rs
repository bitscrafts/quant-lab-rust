//! Market microstructure traits.
//!
//! Defines traits for market impact models and order book operations.

use std::error::Error;

/// A market impact model estimates price impact of trades.
///
/// Impact models (square-root, linear) implement this trait to
/// provide consistent impact estimation across different market
/// models.
pub trait ImpactModel {
    /// Compute the price impact of a trade.
    ///
    /// # Arguments
    ///
    /// * `volume` - Trade size (number of shares)
    /// * `daily_volume` - Average daily trading volume
    ///
    /// # Returns
    ///
    /// The expected price impact as a fraction (e.g., 0.001 = 10 bps).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sqrt_impact = SqrtImpactModel::new(0.1);
    /// let impact = sqrt_impact.impact(10_000.0, 1_000_000.0);
    /// println!("Impact: {:.4}%", impact * 100.0);
    /// ```
    fn impact(&self, volume: f64, daily_volume: f64) -> f64;
}

/// Order book operations.
///
/// This trait defines the interface for limit order book manipulation
/// and market order execution.
pub trait OrderBookOps {
    /// The error type returned when operations fail.
    type Error: Error;

    /// The type representing an order.
    type Order;

    /// The type representing an executed trade.
    type Trade;

    /// Add a limit order to the book.
    ///
    /// # Arguments
    ///
    /// * `order` - The order to add
    ///
    /// # Returns
    ///
    /// An order ID for tracking or cancellation.
    ///
    /// # Errors
    ///
    /// Returns an error if the order is invalid (e.g., negative
    /// price or quantity).
    fn add_order(&mut self, order: Self::Order) -> Result<u64, Self::Error>;

    /// Cancel an existing order.
    ///
    /// # Arguments
    ///
    /// * `order_id` - The ID of the order to cancel
    ///
    /// # Errors
    ///
    /// Returns an error if the order does not exist.
    fn cancel_order(&mut self, order_id: u64) -> Result<(), Self::Error>;

    /// Execute a market order.
    ///
    /// # Arguments
    ///
    /// * `is_buy` - True for buy, false for sell
    /// * `quantity` - Number of shares
    ///
    /// # Returns
    ///
    /// A vector of trades executed to fill the order, possibly
    /// with partial fills at different price levels.
    ///
    /// # Errors
    ///
    /// Returns an error if the order cannot be filled (e.g.,
    /// insufficient liquidity).
    fn market_order(&mut self, is_buy: bool, quantity: f64) -> Result<Vec<Self::Trade>, Self::Error>;
}
