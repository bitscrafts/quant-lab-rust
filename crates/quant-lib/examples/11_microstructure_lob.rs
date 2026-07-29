//! Example 11: Microstructure - Limit Order Book, OFI, VWAP, Market Impact
//!
//! Level: Intermediate
//!
//! Simulates a limit order book (`OrderBook`) with price-time priority,
//! executes market orders against it to produce fills, computes the
//! Volume-Weighted Average Price (VWAP) and the order flow imbalance
//! (OFI) from book snapshots, and estimates the market impact of a
//! child order using the square-root (Almgren-Chriss) model.
//!
//! Uses `quant-microstructure`.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 11_microstructure_lob
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::microstructure::{
    BookSnapshot, Level, LinearImpactModel, Order, SqrtImpactModel, linear_impact,
    order_flow_imbalance,
};
use quant_lib::prelude::*;

fn main() {
    println!("=== Example 11: Microstructure - LOB, OFI, VWAP, Impact ===");

    // 1. Build an order book with tick size = 1 (prices are integer ticks).
    let mut book = OrderBook::new(1);

    // Populate the book with limit orders at several price levels.
    // Bids below 100, asks at and above 100.
    book.add_order(Order {
        id: 1,
        side: Side::Bid,
        price: 99,
        quantity: 30,
        timestamp: 1,
    })
    .unwrap();
    book.add_order(Order {
        id: 2,
        side: Side::Bid,
        price: 99,
        quantity: 20,
        timestamp: 2,
    })
    .unwrap();
    book.add_order(Order {
        id: 3,
        side: Side::Bid,
        price: 98,
        quantity: 50,
        timestamp: 3,
    })
    .unwrap();
    book.add_order(Order {
        id: 4,
        side: Side::Ask,
        price: 100,
        quantity: 15,
        timestamp: 4,
    })
    .unwrap();
    book.add_order(Order {
        id: 5,
        side: Side::Ask,
        price: 100,
        quantity: 25,
        timestamp: 5,
    })
    .unwrap();
    book.add_order(Order {
        id: 6,
        side: Side::Ask,
        price: 101,
        quantity: 40,
        timestamp: 6,
    })
    .unwrap();

    let best_bid = book.best_bid().unwrap();
    let best_ask = book.best_ask().unwrap();
    let mid = book.mid_price().unwrap();
    let spread = book.spread().unwrap();
    println!(
        "Best bid: price={}, qty={}, orders={}",
        best_bid.price, best_bid.quantity, best_bid.order_count
    );
    println!(
        "Best ask: price={}, qty={}, orders={}",
        best_ask.price, best_ask.quantity, best_ask.order_count
    );
    println!("Mid price = {mid}, spread = {spread} ticks");

    // 2. Top-of-book depth (2 levels each side).
    let (bids_d, asks_d) = book.depth(2);
    println!("\nDepth (top 2 levels):");
    for l in &bids_d {
        println!("  BID {l:?}");
    }
    for l in &asks_d {
        println!("  ASK {l:?}");
    }

    // 3. Market buy: 50 shares eats the asks (100 -> 101).
    let fills = book.market_order(Side::Bid, 50);
    println!("\nMarket buy 50: {} fills", fills.len());
    let v = vwap(&fills);
    println!("  VWAP of fills = {v:.4}");
    let mut total_qty = 0u64;
    for f in &fills {
        println!("    fill @ {} x {}", f.price, f.quantity);
        total_qty += f.quantity;
    }
    assert_eq!(total_qty, 50, "should have filled exactly 50");
    assert!((v - 100.2).abs() < 1e-9, "VWAP should be ~100.2, got {v}");
    println!("  total filled = {total_qty} (matches requested 50)");

    // 4. Order flow imbalance from snapshots.
    let snapshots = vec![
        BookSnapshot {
            timestamp: 1,
            best_bid: Some(Level {
                price: 99,
                quantity: 50,
                order_count: 2,
            }),
            best_ask: Some(Level {
                price: 100,
                quantity: 40,
                order_count: 2,
            }),
        },
        BookSnapshot {
            timestamp: 2,
            best_bid: Some(Level {
                price: 99,
                quantity: 60,
                order_count: 2,
            }),
            best_ask: Some(Level {
                price: 100,
                quantity: 35,
                order_count: 2,
            }),
        },
        BookSnapshot {
            timestamp: 3,
            best_bid: Some(Level {
                price: 99,
                quantity: 40,
                order_count: 1,
            }),
            best_ask: Some(Level {
                price: 100,
                quantity: 50,
                order_count: 3,
            }),
        },
    ];
    let ofi = order_flow_imbalance(&snapshots);
    println!("\nOFI series ({} deltas):", ofi.len());
    for (i, o) in ofi.iter().enumerate() {
        println!("  OFI[{}] = {o:+.1}", i + 1);
    }
    // OFI[0] = (60-50) - (35-40) = 10 - (-5) = 15
    assert!((ofi[0] - 15.0).abs() < 1e-9);
    // OFI[1] = (40-60) - (50-35) = -20 - 15 = -35
    assert!((ofi[1] - (-35.0)).abs() < 1e-9);
    println!("OFI calculations match manual values (+15, -35).");

    // 5. Market impact: square-root (Almgren-Chriss) model.
    let daily_vol = 1_000_000.0; // ADV in shares
    let order_size = 10_000.0; // 1% of ADV
    let vol = 0.02; // 2% daily volatility
    let impact = sqrt_impact(vol, order_size, daily_vol);
    let ec = execution_cost(order_size, daily_vol, vol, 0.0002);
    println!("\nMarket impact (Almgren-Chriss square-root):");
    println!("  order = {order_size} shares, ADV = {daily_vol}, sigma = {vol}");
    println!(
        "  participation = {:.4} ({:.2}%)",
        order_size / daily_vol,
        order_size / daily_vol * 100.0
    );
    println!(
        "  sqrt impact  = {impact:.6} ({:.2} bps)",
        impact * 10_000.0
    );
    println!(
        "  execution cost: spread = {:.6}, impact = {:.6}, total = {:.6} bps",
        ec.spread_cost * 10_000.0,
        ec.impact_cost * 10_000.0,
        ec.total_cost * 10_000.0
    );

    // 6. Trait-based impact models.
    let sqrt_model = SqrtImpactModel::new(vol);
    let lin_model = LinearImpactModel::new(0.1);
    let imp_trait = sqrt_model.impact(order_size, daily_vol);
    let lin_trait = lin_model.impact(order_size, daily_vol);
    assert!((imp_trait - impact).abs() < 1e-9);
    assert!((lin_trait - linear_impact(0.1, order_size, daily_vol)).abs() < 1e-9);
    println!("\nTrait models match free-function forms (SqrtImpactModel, LinearImpactModel).");
}
