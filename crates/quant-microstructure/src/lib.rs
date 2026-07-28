//! Market microstructure: the limit order book, order flow imbalance,
//! and market impact models.
//!
//! `quant-microstructure` is the Phase 13 crate of the quant-finance
//! curriculum. The order book uses `BTreeMap` for price levels with
//! integer (tick) prices to avoid floating-point comparison issues.
//! Orders at the same price execute in FIFO (price-time) priority.
//! All data structures are hand-rolled — no external order book
//! libraries.
//!
//! # Modules
//!
//! - [`types`]: `Side`, `Order`, `Level`, `Fill`, `Trade`, `BookSnapshot`
//! - [`orderbook`]: `OrderBook` with `add_order`, `cancel_order`,
//!   `market_order`, `best_bid`, `best_ask`, `mid_price`, `spread`,
//!   `depth`, `total_volume`
//! - [`flow`]: `order_flow_imbalance`, `vwap`, `trade_imbalance`
//! - [`impact`]: `sqrt_impact`, `linear_impact`, `execution_cost`
//! - `error`: `MicroError`
//!
//! # Example
//!
//! ```
//! use quant_microstructure::{OrderBook, Side, Order};
//!
//! let mut book = OrderBook::new(1);
//! book.add_order(Order { id: 1, side: Side::Bid, price: 100, quantity: 10, timestamp: 1 }).unwrap();
//! book.add_order(Order { id: 2, side: Side::Ask, price: 101, quantity: 20, timestamp: 2 }).unwrap();
//! assert_eq!(book.spread(), Some(1));
//! let fills = book.market_order(Side::Bid, 15);
//! assert_eq!(fills.len(), 1);
//! ```

pub mod error;
pub mod flow;
pub mod impact;
pub mod models;
pub mod orderbook;
pub mod types;

pub use error::MicroError;
pub use flow::{order_flow_imbalance, trade_imbalance, vwap};
pub use impact::{ExecutionCost, execution_cost, linear_impact, sqrt_impact};
pub use models::{LinearImpactModel, SqrtImpactModel};
pub use orderbook::OrderBook;
pub use types::{BookSnapshot, Fill, Level, Order, Side, Trade};
