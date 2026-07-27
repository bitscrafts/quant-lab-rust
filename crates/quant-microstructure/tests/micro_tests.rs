//! Integration tests for quant-microstructure (Phase 13).
//!
//! 15-test TDD contract covering: order book construction and
//! invariants, price-time priority, market-order execution, OFI,
//! VWAP, trade imbalance, and market impact models.

#![allow(clippy::useless_vec)]

use quant_microstructure::{
    execution_cost, linear_impact, order_flow_imbalance, sqrt_impact, trade_imbalance, vwap,
    BookSnapshot, Fill, Level, MicroError, Order, OrderBook, Side, Trade,
};

fn bid(id: u64, price: u64, qty: u64, ts: u64) -> Order {
    Order { id, side: Side::Bid, price, quantity: qty, timestamp: ts }
}

fn ask(id: u64, price: u64, qty: u64, ts: u64) -> Order {
    Order { id, side: Side::Ask, price, quantity: qty, timestamp: ts }
}

// =========================================================================
// R13.2: Order and Order Book Types
// =========================================================================

#[test]
fn t01_empty_book_has_no_top() {
    let book = OrderBook::new(1);
    assert_eq!(book.best_bid(), None);
    assert_eq!(book.best_ask(), None);
    assert_eq!(book.spread(), None);
    assert_eq!(book.mid_price(), None);
    assert_eq!(book.order_count(), 0);
}

#[test]
fn t02_add_orders_updates_best_levels() {
    let mut book = OrderBook::new(1);
    book.add_order(bid(1, 100, 10, 1)).unwrap();
    book.add_order(ask(2, 101, 20, 2)).unwrap();
    let bb = book.best_bid().unwrap();
    let ba = book.best_ask().unwrap();
    assert_eq!(bb.price, 100);
    assert_eq!(bb.quantity, 10);
    assert_eq!(ba.price, 101);
    assert_eq!(ba.quantity, 20);
    assert_eq!(book.spread(), Some(1));
    assert_eq!(book.order_count(), 2);
}

#[test]
fn t03_duplicate_order_id_rejected() {
    let mut book = OrderBook::new(1);
    book.add_order(bid(1, 100, 10, 1)).unwrap();
    let err = book.add_order(bid(1, 99, 5, 2)).unwrap_err();
    assert!(matches!(err, MicroError::InvalidOrder(_)));
}

#[test]
fn t04_tick_size_violation_rejected() {
    let mut book = OrderBook::new(10);
    let err = book.add_order(bid(1, 105, 10, 1)).unwrap_err();
    assert!(matches!(err, MicroError::InvalidOrder(_)));
}

// =========================================================================
// R13.3: Order Book Operations
// =========================================================================

#[test]
fn t05_market_buy_walks_asks_ascending() {
    let mut book = OrderBook::new(1);
    book.add_order(ask(1, 101, 30, 1)).unwrap();
    book.add_order(ask(2, 102, 50, 2)).unwrap();
    book.add_order(ask(3, 103, 20, 3)).unwrap();
    let fills = book.market_order(Side::Bid, 60);
    assert_eq!(fills.len(), 2);
    assert_eq!(fills[0].price, 101);
    assert_eq!(fills[0].quantity, 30);
    assert_eq!(fills[1].price, 102);
    assert_eq!(fills[1].quantity, 30);
    // Best ask is now 102 with 20 remaining.
    assert_eq!(book.best_ask().unwrap().price, 102);
    assert_eq!(book.best_ask().unwrap().quantity, 20);
}

#[test]
fn t06_market_sell_walks_bids_descending() {
    let mut book = OrderBook::new(1);
    book.add_order(bid(1, 100, 40, 1)).unwrap();
    book.add_order(bid(2, 99, 30, 2)).unwrap();
    book.add_order(bid(3, 98, 30, 3)).unwrap();
    let fills = book.market_order(Side::Ask, 50);
    assert_eq!(fills.len(), 2);
    assert_eq!(fills[0].price, 100);
    assert_eq!(fills[0].quantity, 40);
    assert_eq!(fills[1].price, 99);
    assert_eq!(fills[1].quantity, 10);
    assert_eq!(book.best_bid().unwrap().price, 99);
    assert_eq!(book.best_bid().unwrap().quantity, 20);
}

#[test]
fn t07_partial_fill_when_liquidity_insufficient() {
    let mut book = OrderBook::new(1);
    book.add_order(ask(1, 101, 25, 1)).unwrap();
    let fills = book.market_order(Side::Bid, 100);
    assert_eq!(fills.len(), 1);
    assert_eq!(fills[0].quantity, 25);
    assert_eq!(book.best_ask(), None);
}

#[test]
fn t08_cancel_order_removes_and_returns_it() {
    let mut book = OrderBook::new(1);
    book.add_order(bid(1, 100, 10, 1)).unwrap();
    book.add_order(bid(2, 99, 20, 2)).unwrap();
    let cancelled = book.cancel_order(1).unwrap();
    assert_eq!(cancelled.id, 1);
    assert_eq!(book.best_bid().unwrap().price, 99);
    assert_eq!(book.order_count(), 1);
}

#[test]
fn t09_cancel_nonexistent_returns_error() {
    let mut book = OrderBook::new(1);
    let err = book.cancel_order(42).unwrap_err();
    assert!(matches!(err, MicroError::OrderNotFound(42)));
}

#[test]
fn t10_price_time_priority_fifo_at_same_price() {
    let mut book = OrderBook::new(1);
    book.add_order(ask(1, 100, 10, 1)).unwrap();
    book.add_order(ask(2, 100, 10, 2)).unwrap();
    let fills = book.market_order(Side::Bid, 10);
    assert_eq!(fills.len(), 1);
    assert_eq!(fills[0].maker_order_id, 1); // earlier timestamp fills first
}

// =========================================================================
// R13.4: Order Flow Imbalance
// =========================================================================

#[test]
fn t11_ofi_series_from_snapshots() {
    let snaps = vec![
        BookSnapshot {
            timestamp: 1,
            best_bid: Some(Level { price: 100, quantity: 50, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 30, order_count: 1 }),
        },
        BookSnapshot {
            timestamp: 2,
            best_bid: Some(Level { price: 100, quantity: 60, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 25, order_count: 1 }),
        },
        BookSnapshot {
            timestamp: 3,
            best_bid: Some(Level { price: 100, quantity: 55, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 35, order_count: 1 }),
        },
    ];
    let ofi = order_flow_imbalance(&snaps);
    assert_eq!(ofi.len(), 2);
    // t=2: (60-50) - (25-30) = 10 - (-5) = 15
    assert!((ofi[0] - 15.0).abs() < 1e-9);
    // t=3: (55-60) - (35-25) = -5 - 10 = -15
    assert!((ofi[1] - (-15.0)).abs() < 1e-9);
}

#[test]
fn t12_vwap_weighted_average_price() {
    let fills = vec![
        Fill { price: 100, quantity: 10, maker_order_id: 1 },
        Fill { price: 102, quantity: 20, maker_order_id: 2 },
        Fill { price: 105, quantity: 30, maker_order_id: 3 },
    ];
    // VWAP = (100*10 + 102*20 + 105*30) / 60 = (1000+2040+3150)/60 = 6190/60
    let expected = 6190.0 / 60.0;
    assert!((vwap(&fills) - expected).abs() < 1e-9);
}

#[test]
fn t13_trade_imbalance_bounds() {
    let trades = vec![
        Trade { price: 100, quantity: 30, side: Side::Bid, timestamp: 1 },
        Trade { price: 101, quantity: 10, side: Side::Ask, timestamp: 2 },
    ];
    let ti = trade_imbalance(&trades);
    // (30 - 10) / 40 = 0.5
    assert!((ti - 0.5).abs() < 1e-9);
    assert!((-1.0..=1.0).contains(&ti));
}

// =========================================================================
// R13.5: Market Impact Models
// =========================================================================

#[test]
fn t14_sqrt_impact_scales_with_square_root_of_size() {
    let sigma = 0.02;
    let v = 1_000_000.0;
    let small = sqrt_impact(sigma, 500.0, v);
    let large = sqrt_impact(sigma, 1000.0, v);
    // Doubling size multiplies impact by sqrt(2).
    let ratio = large / small;
    assert!((ratio - 2.0_f64.sqrt()).abs() < 1e-6);
    // Zero volume -> zero impact.
    assert_eq!(sqrt_impact(sigma, 1000.0, 0.0), 0.0);
}

#[test]
fn t15_execution_cost_combines_spread_and_impact() {
    let ec = execution_cost(1000.0, 1_000_000.0, 0.02, 0.0002);
    // half-spread = 0.0001 = 1 bp
    assert!((ec.spread_cost - 0.0001).abs() < 1e-12);
    // impact = 0.02 * sqrt(0.001)
    let expected_impact = 0.02 * (0.001_f64).sqrt();
    assert!((ec.impact_cost - expected_impact).abs() < 1e-12);
    // total = spread + impact
    assert!((ec.total_cost - (ec.spread_cost + ec.impact_cost)).abs() < 1e-12);
    // linear impact: eta * (Q/V)
    let li = linear_impact(0.1, 500.0, 10_000.0);
    assert!((li - 0.005).abs() < 1e-9);
}