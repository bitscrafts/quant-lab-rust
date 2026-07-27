//! Order flow imbalance (OFI) and market impact demo.
//!
//! Builds a sequence of book snapshots, computes the OFI series,
//! then estimates the execution cost of a market order under the
//! square-root impact model.

use quant_microstructure::{
    execution_cost, order_flow_imbalance, trade_imbalance, BookSnapshot, Level, Side, Trade,
};

fn main() {
    // Five snapshots; bid qty grows then shrinks while ask qty shrinks then grows.
    let snapshots = vec![
        BookSnapshot {
            timestamp: 1,
            best_bid: Some(Level { price: 100, quantity: 50, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 40, order_count: 1 }),
        },
        BookSnapshot {
            timestamp: 2,
            best_bid: Some(Level { price: 100, quantity: 60, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 35, order_count: 1 }),
        },
        BookSnapshot {
            timestamp: 3,
            best_bid: Some(Level { price: 100, quantity: 75, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 25, order_count: 1 }),
        },
        BookSnapshot {
            timestamp: 4,
            best_bid: Some(Level { price: 100, quantity: 65, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 30, order_count: 1 }),
        },
        BookSnapshot {
            timestamp: 5,
            best_bid: Some(Level { price: 100, quantity: 55, order_count: 1 }),
            best_ask: Some(Level { price: 101, quantity: 45, order_count: 1 }),
        },
    ];

    let ofi = order_flow_imbalance(&snapshots);
    println!("=== OFI series ({} transitions) ===", ofi.len());
    for (i, v) in ofi.iter().enumerate() {
        println!("  t={:>2}: OFI = {:>7.1}", i + 1, v);
    }
    let cum_ofi: f64 = ofi.iter().sum();
    println!("  cumulative OFI = {:.1}", cum_ofi);
    println!("  (positive => net buy-side pressure)");

    // Trades from a market-buy sweep.
    let trades = vec![
        Trade { price: 101, quantity: 30, side: Side::Bid, timestamp: 10 },
        Trade { price: 102, quantity: 50, side: Side::Bid, timestamp: 11 },
        Trade { price: 100, quantity: 20, side: Side::Ask, timestamp: 12 },
    ];
    let ti = trade_imbalance(&trades);
    println!("\n=== Trade imbalance ===");
    println!("  trade_imbalance = {:.4} (in [-1, 1])", ti);

    // Execution cost for a 2000-share order under sqrt impact.
    let ec = execution_cost(2000.0, 1_000_000.0, 0.02, 0.0002);
    println!("\n=== Execution cost (sqrt impact, Almgren-Chriss) ===");
    println!("  order_size     = 2000 shares");
    println!("  daily_volume   = 1,000,000 shares");
    println!("  volatility     = 2.0% daily");
    println!("  spread         = 2 bps");
    println!("  spread_cost    = {:.2} bps", ec.spread_cost * 10_000.0);
    println!("  impact_cost    = {:.2} bps", ec.impact_cost * 10_000.0);
    println!("  total_cost     = {:.2} bps", ec.total_cost * 10_000.0);
}