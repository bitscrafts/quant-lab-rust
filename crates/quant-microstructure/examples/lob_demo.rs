//! Limit order book demo: build a book, execute a market order,
//! inspect the resulting fills and post-trade state.

use quant_microstructure::{Order, OrderBook, Side};

fn main() {
    let mut book = OrderBook::new(1);

    // Resting asks (sellers) at three price levels.
    book.add_order(Order { id: 1, side: Side::Ask, price: 101, quantity: 30, timestamp: 1 }).unwrap();
    book.add_order(Order { id: 2, side: Side::Ask, price: 102, quantity: 50, timestamp: 2 }).unwrap();
    book.add_order(Order { id: 3, side: Side::Ask, price: 103, quantity: 20, timestamp: 3 }).unwrap();

    // Resting bids (buyers).
    book.add_order(Order { id: 4, side: Side::Bid, price: 99, quantity: 40, timestamp: 4 }).unwrap();
    book.add_order(Order { id: 5, side: Side::Bid, price: 98, quantity: 60, timestamp: 5 }).unwrap();

    println!("=== Initial book ===");
    println!("best_bid = {:?}", book.best_bid());
    println!("best_ask = {:?}", book.best_ask());
    println!("spread   = {:?}", book.spread());
    println!("mid      = {:?}", book.mid_price());
    let (bv, av) = book.total_volume();
    println!("volume   = (bid={bv}, ask={av})");

    let (bids, asks) = book.depth(3);
    println!("bids (top-3, desc): {:?}", bids);
    println!("asks (top-3, asc):  {:?}", asks);

    // A market buy of 75 shares walks the asks.
    let fills = book.market_order(Side::Bid, 75);
    println!("\n=== Market buy 75 shares ===");
    for f in &fills {
        println!("  fill: {} @ {} (maker #{})", f.quantity, f.price, f.maker_order_id);
    }
    let total_qty: u64 = fills.iter().map(|f| f.quantity).sum();
    let total_notional: f64 = fills.iter().map(|f| f.price as f64 * f.quantity as f64).sum();
    println!("  filled qty = {total_qty}, notional = {total_notional:.0}");
    println!("  avg fill price = {:.4}", total_notional / total_qty as f64);

    println!("\n=== Post-trade book ===");
    println!("best_ask = {:?}", book.best_ask());
    println!("order_count = {}", book.order_count());
}