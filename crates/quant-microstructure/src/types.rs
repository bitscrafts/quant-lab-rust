//! Core types for the limit order book and microstructure models.
//!
//! Prices are represented as `u64` ticks to avoid floating-point comparison
//! issues. Each `Order` carries a monotonically increasing timestamp so the
//! book can enforce price-time priority (FIFO at each price level).

/// Side of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    /// The opposite side.
    pub fn opposite(self) -> Side {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Bid => write!(f, "Bid"),
            Side::Ask => write!(f, "Ask"),
        }
    }
}

/// A single limit order in the book.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Order {
    pub id: u64,
    pub side: Side,
    pub price: u64,
    pub quantity: u64,
    pub timestamp: u64,
}

/// A price level (aggregated orders at the same price).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Level {
    pub price: u64,
    pub quantity: u64,
    pub order_count: usize,
}

/// A fill produced by a market-order execution against the book.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fill {
    pub price: u64,
    pub quantity: u64,
    pub maker_order_id: u64,
}

/// A trade (for OFI / imbalance calculations).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trade {
    pub price: u64,
    pub quantity: u64,
    pub side: Side,
    pub timestamp: u64,
}

/// A snapshot of the best bid/ask for OFI calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookSnapshot {
    pub timestamp: u64,
    pub best_bid: Option<Level>,
    pub best_ask: Option<Level>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Bid.opposite(), Side::Ask);
        assert_eq!(Side::Ask.opposite(), Side::Bid);
    }

    #[test]
    fn test_side_display() {
        assert_eq!(format!("{}", Side::Bid), "Bid");
        assert_eq!(format!("{}", Side::Ask), "Ask");
    }

    #[test]
    fn test_order_equality() {
        let a = Order {
            id: 1,
            side: Side::Bid,
            price: 100,
            quantity: 10,
            timestamp: 1,
        };
        let b = Order {
            id: 1,
            side: Side::Bid,
            price: 100,
            quantity: 10,
            timestamp: 1,
        };
        assert_eq!(a, b);
    }
}
