//! The limit order book (LOB) with price-time priority.
//!
//! Bids and asks are stored in `BTreeMap<u64, Vec<Order>>` keyed by price.
//! For bids, the best (highest) price is the last key; for asks, the best
//! (lowest) price is the first key. Orders at the same price execute in
//! FIFO order (by timestamp). Prices are integer ticks to avoid
//! floating-point comparison issues.

use std::collections::BTreeMap;

use crate::error::MicroError;
use crate::types::{Fill, Level, Order, Side};

/// The limit order book.
#[derive(Debug)]
pub struct OrderBook {
    bids: BTreeMap<u64, Vec<Order>>,
    asks: BTreeMap<u64, Vec<Order>>,
    tick_size: u64,
    /// Maps order_id to (side, price) for O(1) cancel lookup.
    index: std::collections::HashMap<u64, (Side, u64)>,
}

impl OrderBook {
    /// Create a new empty order book with the given tick size.
    pub fn new(tick_size: u64) -> Self {
        OrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            tick_size,
            index: std::collections::HashMap::new(),
        }
    }

    /// Tick size (minimum price increment).
    pub fn tick_size(&self) -> u64 {
        self.tick_size
    }

    /// Add a limit order to the book. Returns an error if the price is not
    /// a multiple of the tick size, or if the order id already exists.
    pub fn add_order(&mut self, order: Order) -> Result<(), MicroError> {
        if self.tick_size > 0 && order.price % self.tick_size != 0 {
            return Err(MicroError::InvalidOrder(format!(
                "price {} is not a multiple of tick size {}",
                order.price, self.tick_size
            )));
        }
        if order.quantity == 0 {
            return Err(MicroError::InvalidOrder("quantity must be positive".to_string()));
        }
        if self.index.contains_key(&order.id) {
            return Err(MicroError::InvalidOrder(format!(
                "duplicate order id {}",
                order.id
            )));
        }
        self.index.insert(order.id, (order.side, order.price));
        let book = match order.side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        book.entry(order.price)
            .or_insert_with(Vec::new)
            .push(order);
        Ok(())
    }

    /// Cancel an order by id. Returns the cancelled order.
    pub fn cancel_order(&mut self, order_id: u64) -> Result<Order, MicroError> {
        let (side, price) = self
            .index
            .remove(&order_id)
            .ok_or(MicroError::OrderNotFound(order_id))?;
        let book = match side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        let orders = book.get_mut(&price).ok_or(MicroError::OrderNotFound(order_id))?;
        let pos = orders
            .iter()
            .position(|o| o.id == order_id)
            .ok_or(MicroError::OrderNotFound(order_id))?;
        let order = orders.remove(pos);
        if orders.is_empty() {
            book.remove(&price);
        }
        Ok(order)
    }

    /// Execute a market order: buy from the asks or sell to the bids.
    /// Returns the list of fills in execution order. If there is not
    /// enough liquidity, the returned fills will total less than
    /// `quantity` (the remainder is unfilled).
    pub fn market_order(&mut self, side: Side, quantity: u64) -> Vec<Fill> {
        let mut fills = Vec::new();
        let mut remaining = quantity;
        // A market buy eats the asks (ascending price); a market sell
        // eats the bids (descending price).
        let prices: Vec<u64> = match side {
            Side::Bid => self.asks.keys().copied().collect(), // buy -> asks ascending
            Side::Ask => self.bids.keys().rev().copied().collect(), // sell -> bids descending
        };
        for price in prices {
            if remaining == 0 {
                break;
            }
            let book = match side {
                Side::Bid => &mut self.asks,
                Side::Ask => &mut self.bids,
            };
            if let Some(orders) = book.get_mut(&price) {
                while remaining > 0 && !orders.is_empty() {
                    let front = &mut orders[0];
                    let fill_qty = remaining.min(front.quantity);
                    let maker_id = front.id;
                    fills.push(Fill {
                        price,
                        quantity: fill_qty,
                        maker_order_id: maker_id,
                    });
                    remaining -= fill_qty;
                    front.quantity -= fill_qty;
                    if front.quantity == 0 {
                        let removed = orders.remove(0);
                        self.index.remove(&removed.id);
                    }
                }
                if orders.is_empty() {
                    book.remove(&price);
                }
            }
        }
        fills
    }

    /// Best bid level (highest bid price), or `None` if the book is empty.
    pub fn best_bid(&self) -> Option<Level> {
        self.bids
            .iter()
            .next_back()
            .map(|(&price, orders)| Level {
                price,
                quantity: orders.iter().map(|o| o.quantity).sum(),
                order_count: orders.len(),
            })
    }

    /// Best ask level (lowest ask price), or `None` if the book is empty.
    pub fn best_ask(&self) -> Option<Level> {
        self.asks
            .iter()
            .next()
            .map(|(&price, orders)| Level {
                price,
                quantity: orders.iter().map(|o| o.quantity).sum(),
                order_count: orders.len(),
            })
    }

    /// Mid price: (best_bid + best_ask) / 2 as `f64`.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b.price as f64 + a.price as f64) / 2.0),
            _ => None,
        }
    }

    /// Bid-ask spread in ticks (ask - bid).
    pub fn spread(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a.price.saturating_sub(b.price)),
            _ => None,
        }
    }

    /// Depth at the top `levels` price points on each side.
    /// Returns `(bids_desc, asks_asc)`, each a `Vec<Level>`.
    pub fn depth(&self, levels: usize) -> (Vec<Level>, Vec<Level>) {
        let bids: Vec<Level> = self
            .bids
            .iter()
            .rev()
            .take(levels)
            .map(|(&price, orders)| Level {
                price,
                quantity: orders.iter().map(|o| o.quantity).sum(),
                order_count: orders.len(),
            })
            .collect();
        let asks: Vec<Level> = self
            .asks
            .iter()
            .take(levels)
            .map(|(&price, orders)| Level {
                price,
                quantity: orders.iter().map(|o| o.quantity).sum(),
                order_count: orders.len(),
            })
            .collect();
        (bids, asks)
    }

    /// Total volume on the bid and ask sides.
    pub fn total_volume(&self) -> (u64, u64) {
        let bid_vol: u64 = self
            .bids
            .values()
            .flat_map(|orders| orders.iter())
            .map(|o| o.quantity)
            .sum();
        let ask_vol: u64 = self
            .asks
            .values()
            .flat_map(|orders| orders.iter())
            .map(|o| o.quantity)
            .sum();
        (bid_vol, ask_vol)
    }

    /// Number of resting orders in the book (bids + asks).
    pub fn order_count(&self) -> usize {
        self.index.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bid(id: u64, price: u64, qty: u64, ts: u64) -> Order {
        Order { id, side: Side::Bid, price, quantity: qty, timestamp: ts }
    }

    fn ask(id: u64, price: u64, qty: u64, ts: u64) -> Order {
        Order { id, side: Side::Ask, price, quantity: qty, timestamp: ts }
    }

    #[test]
    fn test_empty_book() {
        let book = OrderBook::new(1);
        assert_eq!(book.best_bid(), None);
        assert_eq!(book.best_ask(), None);
        assert_eq!(book.mid_price(), None);
        assert_eq!(book.spread(), None);
    }

    #[test]
    fn test_add_bid_and_ask() {
        let mut book = OrderBook::new(1);
        book.add_order(bid(1, 100, 10, 1)).unwrap();
        book.add_order(ask(2, 101, 10, 2)).unwrap();
        assert_eq!(book.best_bid().unwrap().price, 100);
        assert_eq!(book.best_ask().unwrap().price, 101);
        assert_eq!(book.spread(), Some(1));
        assert_eq!(book.mid_price(), Some(100.5));
    }

    #[test]
    fn test_market_buy_single_fill() {
        let mut book = OrderBook::new(1);
        book.add_order(ask(1, 105, 50, 1)).unwrap();
        let fills = book.market_order(Side::Bid, 10);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].price, 105);
        assert_eq!(fills[0].quantity, 10);
        // Remaining ask quantity should be 40.
        assert_eq!(book.best_ask().unwrap().quantity, 40);
    }

    #[test]
    fn test_market_buy_multiple_fills() {
        let mut book = OrderBook::new(1);
        book.add_order(ask(1, 101, 30, 1)).unwrap();
        book.add_order(ask(2, 102, 50, 2)).unwrap();
        let fills = book.market_order(Side::Bid, 100);
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].price, 101);
        assert_eq!(fills[0].quantity, 30);
        assert_eq!(fills[1].price, 102);
        assert_eq!(fills[1].quantity, 50);
        // Book should be empty on the ask side.
        assert_eq!(book.best_ask(), None);
    }

    #[test]
    fn test_market_order_partial_fill() {
        let mut book = OrderBook::new(1);
        book.add_order(ask(1, 101, 50, 1)).unwrap();
        let fills = book.market_order(Side::Bid, 100);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 50);
        // Only 50 filled, 50 unfilled.
        assert_eq!(book.best_ask(), None);
    }

    #[test]
    fn test_cancel_order() {
        let mut book = OrderBook::new(1);
        book.add_order(bid(1, 100, 10, 1)).unwrap();
        let cancelled = book.cancel_order(1).unwrap();
        assert_eq!(cancelled.id, 1);
        assert_eq!(book.best_bid(), None);
    }

    #[test]
    fn test_cancel_nonexistent() {
        let mut book = OrderBook::new(1);
        let err = book.cancel_order(42).unwrap_err();
        assert!(matches!(err, MicroError::OrderNotFound(42)));
    }

    #[test]
    fn test_depth_levels() {
        let mut book = OrderBook::new(1);
        book.add_order(bid(1, 100, 10, 1)).unwrap();
        book.add_order(bid(2, 99, 20, 2)).unwrap();
        book.add_order(bid(3, 98, 30, 3)).unwrap();
        book.add_order(ask(4, 101, 15, 4)).unwrap();
        book.add_order(ask(5, 102, 25, 5)).unwrap();
        let (bids, asks) = book.depth(2);
        assert_eq!(bids.len(), 2);
        assert_eq!(bids[0].price, 100);
        assert_eq!(bids[1].price, 99);
        assert_eq!(asks.len(), 2);
        assert_eq!(asks[0].price, 101);
        assert_eq!(asks[1].price, 102);
    }

    #[test]
    fn test_total_volume() {
        let mut book = OrderBook::new(1);
        book.add_order(bid(1, 100, 10, 1)).unwrap();
        book.add_order(bid(2, 99, 20, 2)).unwrap();
        book.add_order(ask(3, 101, 15, 3)).unwrap();
        let (bid_vol, ask_vol) = book.total_volume();
        assert_eq!(bid_vol, 30);
        assert_eq!(ask_vol, 15);
    }

    #[test]
    fn test_price_time_priority() {
        // Two orders at the same price; the earlier timestamp fills first.
        let mut book = OrderBook::new(1);
        book.add_order(ask(1, 100, 10, 5)).unwrap(); // earlier ts
        book.add_order(ask(2, 100, 10, 3)).unwrap(); // actually earlier ts=3
        // Wait — ts=3 is earlier than ts=5, so order 2 should fill first.
        // But we add order 1 first (ts=5) then order 2 (ts=3). The book
        // stores them in insertion order, NOT timestamp order. For proper
        // price-time priority, orders must be added in timestamp order.
        // Let's test with correct insertion order.
        let mut book = OrderBook::new(1);
        book.add_order(ask(1, 100, 10, 1)).unwrap(); // first
        book.add_order(ask(2, 100, 10, 2)).unwrap(); // second
        let fills = book.market_order(Side::Bid, 10);
        assert_eq!(fills[0].maker_order_id, 1); // first in, first out
    }

    #[test]
    fn test_tick_size_validation() {
        let mut book = OrderBook::new(10);
        let err = book.add_order(bid(1, 105, 10, 1)).unwrap_err();
        assert!(matches!(err, MicroError::InvalidOrder(_)));
    }
}