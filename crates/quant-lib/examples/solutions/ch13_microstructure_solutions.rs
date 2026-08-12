//! Exercise solutions for Chapter 13: Market Microstructure
//!
//! Run: `cargo run -p quant-lib --example solutions-ch13_microstructure_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch13_microstructure_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::microstructure::{BookSnapshot, Level, Order, order_flow_imbalance};
use quant_lib::prelude::*;

fn main() {
    println!("=== Chapter 13: Market Microstructure - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 13 exercises complete.");
}

fn exercise_1() {
    println!("1. Order book builder (best bid/ask, total volume):");
    let mut book = OrderBook::new(1);
    book.add_order(Order {
        id: 1,
        side: Side::Bid,
        price: 100,
        quantity: 50,
        timestamp: 1,
    })
    .expect("add bid");
    book.add_order(Order {
        id: 2,
        side: Side::Bid,
        price: 99,
        quantity: 30,
        timestamp: 2,
    })
    .expect("add bid2");
    book.add_order(Order {
        id: 3,
        side: Side::Ask,
        price: 101,
        quantity: 40,
        timestamp: 3,
    })
    .expect("add ask");
    book.add_order(Order {
        id: 4,
        side: Side::Ask,
        price: 102,
        quantity: 20,
        timestamp: 4,
    })
    .expect("add ask2");
    let bb = book.best_bid().expect("best_bid");
    let ba = book.best_ask().expect("best_ask");
    let (bv, av) = book.total_volume();
    println!("   best_bid = (px={}, qty={})", bb.price, bb.quantity);
    println!("   best_ask = (px={}, qty={})", ba.price, ba.quantity);
    println!("   total_volume = (bid={bv}, ask={av})");
    assert_eq!(bb.price, 100);
    assert_eq!(ba.price, 101);
    assert_eq!(bv, 80);
    assert_eq!(av, 60);
}

fn exercise_2() {
    println!("2. Liquidity sweep (5 ask levels, market buy 750):");
    let mut book = OrderBook::new(1);
    // 5 ask levels totalling 1000 shares: 100@101, 200@102, 200@103, 250@104, 250@105.
    book.add_order(Order {
        id: 1,
        side: Side::Ask,
        price: 101,
        quantity: 100,
        timestamp: 1,
    })
    .expect("add");
    book.add_order(Order {
        id: 2,
        side: Side::Ask,
        price: 102,
        quantity: 200,
        timestamp: 2,
    })
    .expect("add");
    book.add_order(Order {
        id: 3,
        side: Side::Ask,
        price: 103,
        quantity: 200,
        timestamp: 3,
    })
    .expect("add");
    book.add_order(Order {
        id: 4,
        side: Side::Ask,
        price: 104,
        quantity: 250,
        timestamp: 4,
    })
    .expect("add");
    book.add_order(Order {
        id: 5,
        side: Side::Ask,
        price: 105,
        quantity: 250,
        timestamp: 5,
    })
    .expect("add");
    let fills = book.market_order(Side::Bid, 750);
    let total_filled: u64 = fills.iter().map(|f| f.quantity).sum();
    let levels_swept = fills.len();
    let vwap_fills = vwap(&fills);
    // Volume-weighted ask price of all swept levels (fill prices only).
    let vwap_expected: f64 = {
        let total_notional: f64 = fills
            .iter()
            .map(|f| f.price as f64 * f.quantity as f64)
            .sum();
        total_notional / total_filled as f64
    };
    println!("   Levels swept = {levels_swept}");
    println!("   Total filled = {total_filled} (target 750)");
    println!("   VWAP of fills = {vwap_fills:.6}");
    assert_eq!(total_filled, 750);
    assert!(
        (3..=5).contains(&levels_swept),
        "expected 3-5 levels swept, got {levels_swept}"
    );
    assert!((vwap_fills - vwap_expected).abs() < 1e-9);
}

fn exercise_3() {
    println!("3. OFI correlation with mid-price returns (positive):");
    // Build snapshots where bid qty rises with buy pressure.
    let mut snapshots: Vec<BookSnapshot> = Vec::new();
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let mut mid_prices: Vec<f64> = Vec::new();
    let mut bid_q = 100_u64;
    let mut ask_q = 100_u64;
    let mut mid = 100.0_f64;
    for t in 0..200_u64 {
        // Buy pressure: positive shock increases bid qty, raises mid.
        let shock = normal.sample(&mut rng);
        bid_q = (bid_q as f64 + 5.0 * shock + 0.5 * shock.abs() * 10.0).max(10.0) as u64;
        ask_q = (ask_q as f64 - 3.0 * shock).max(10.0) as u64;
        mid += shock;
        mid_prices.push(mid);
        snapshots.push(BookSnapshot {
            timestamp: t,
            best_bid: Some(Level {
                price: (mid * 100.0) as u64,
                quantity: bid_q,
                order_count: 1,
            }),
            best_ask: Some(Level {
                price: ((mid + 0.01) * 100.0) as u64,
                quantity: ask_q,
                order_count: 1,
            }),
        });
    }
    let ofi = order_flow_imbalance(&snapshots);
    let mid_rets: Vec<f64> = mid_prices.windows(2).map(|w| w[1] - w[0]).collect();
    let n = ofi.len().min(mid_rets.len());
    let ofi_n = &ofi[..n];
    let rets_n = &mid_rets[..n];
    let mean_o: f64 = ofi_n.iter().sum::<f64>() / n as f64;
    let mean_r: f64 = rets_n.iter().sum::<f64>() / n as f64;
    let cov: f64 = ofi_n
        .iter()
        .zip(rets_n.iter())
        .map(|(o, r)| (o - mean_o) * (r - mean_r))
        .sum();
    let var_o: f64 = ofi_n.iter().map(|o| (o - mean_o).powi(2)).sum();
    let var_r: f64 = rets_n.iter().map(|r| (r - mean_r).powi(2)).sum();
    let corr = if var_o > 0.0 && var_r > 0.0 {
        cov / (var_o * var_r).sqrt()
    } else {
        0.0
    };
    println!("   OFI vs mid-return correlation = {corr:.4} (expect positive)");
    assert!(
        corr > 0.0,
        "OFI should be positively correlated with returns"
    );
}

fn exercise_4() {
    println!("4. Square-root impact scaling (slope 1/2):");
    let sigma = 0.02_f64;
    let v_daily = 1_000_000.0_f64;
    let q1 = 1_000.0_f64;
    let q2 = 2_000.0_f64;
    let i1 = sqrt_impact(sigma, q1, v_daily);
    let i2 = sqrt_impact(sigma, q2, v_daily);
    let ratio = i2 / i1;
    let expected = 2.0_f64.sqrt();
    println!("   Impact(Q=1000) = {i1:.8}");
    println!("   Impact(Q=2000) = {i2:.8}");
    println!("   Ratio = {ratio:.8} (expect sqrt(2) = {expected:.8})");
    assert!(
        (ratio - expected).abs() < 1e-9,
        "doubling Q should multiply impact by sqrt(2)"
    );
}

fn exercise_5() {
    println!("5. Net frontier (turnover cost reduces tangency Sharpe):");
    // Two-asset universe with daily mu and cov.
    let mu = vec![0.0005, 0.0003]; // daily expected returns
    let cov = vec![vec![0.0001, 0.0], vec![0.0, 0.00005]];
    let rf = 0.0;
    let tan = tangency_portfolio(&mu, &cov, rf).expect("tan");
    let gross_sharpe = tan.sharpe;
    // Execution cost at 50% turnover on a daily volume ~ 1_000_000 shares.
    let sigma_daily = 0.02;
    let spread = 0.0002; // 2 bps
    let turnover = 0.5;
    let v_daily = 1_000_000.0;
    let q = turnover * v_daily;
    let cost = execution_cost(q, v_daily, sigma_daily, spread).total_cost;
    // Subtract daily cost from each asset's return (simplification).
    let mu_net: Vec<f64> = mu.iter().map(|&m| m - cost).collect();
    let tan_net = tangency_portfolio(&mu_net, &cov, rf).expect("tan_net");
    let net_sharpe = tan_net.sharpe;
    let drop = gross_sharpe - net_sharpe;
    println!("   Gross tangency Sharpe = {gross_sharpe:.6}");
    println!("   Net tangency Sharpe    = {net_sharpe:.6}");
    println!("   Sharpe drop = {drop:.6} (expect 0.0001-0.0010)");
    println!("   Execution cost = {:.6} bps", cost * 10_000.0);
    assert!(drop > 0.0, "costs should reduce Sharpe");
    assert!(drop < 0.01, "Sharpe drop should be modest, got {drop}");
}

#[test]
fn test_ex1_orderbook_best_bid_ask() {
    let mut book = OrderBook::new(1);
    book.add_order(Order {
        id: 1,
        side: Side::Bid,
        price: 100,
        quantity: 50,
        timestamp: 1,
    })
    .expect("add");
    book.add_order(Order {
        id: 2,
        side: Side::Bid,
        price: 99,
        quantity: 30,
        timestamp: 2,
    })
    .expect("add");
    book.add_order(Order {
        id: 3,
        side: Side::Ask,
        price: 101,
        quantity: 40,
        timestamp: 3,
    })
    .expect("add");
    book.add_order(Order {
        id: 4,
        side: Side::Ask,
        price: 102,
        quantity: 20,
        timestamp: 4,
    })
    .expect("add");
    assert_eq!(book.best_bid().unwrap().price, 100);
    assert_eq!(book.best_ask().unwrap().price, 101);
    let (bv, av) = book.total_volume();
    assert_eq!(bv, 80);
    assert_eq!(av, 60);
}

#[test]
fn test_ex2_liquidity_sweep() {
    let mut book = OrderBook::new(1);
    book.add_order(Order {
        id: 1,
        side: Side::Ask,
        price: 101,
        quantity: 100,
        timestamp: 1,
    })
    .expect("a");
    book.add_order(Order {
        id: 2,
        side: Side::Ask,
        price: 102,
        quantity: 200,
        timestamp: 2,
    })
    .expect("a");
    book.add_order(Order {
        id: 3,
        side: Side::Ask,
        price: 103,
        quantity: 200,
        timestamp: 3,
    })
    .expect("a");
    book.add_order(Order {
        id: 4,
        side: Side::Ask,
        price: 104,
        quantity: 250,
        timestamp: 4,
    })
    .expect("a");
    book.add_order(Order {
        id: 5,
        side: Side::Ask,
        price: 105,
        quantity: 250,
        timestamp: 5,
    })
    .expect("a");
    let fills = book.market_order(Side::Bid, 750);
    let total: u64 = fills.iter().map(|f| f.quantity).sum();
    assert_eq!(total, 750);
    assert!((3..=5).contains(&fills.len()));
    let vwap_fills = vwap(&fills);
    let expected: f64 = fills
        .iter()
        .map(|f| f.price as f64 * f.quantity as f64)
        .sum::<f64>()
        / 750.0;
    assert!((vwap_fills - expected).abs() < 1e-9);
}

#[test]
fn test_ex3_ofi_positive_correlation() {
    let mut snapshots: Vec<BookSnapshot> = Vec::new();
    let mut rng = XorShift64::new(7);
    let normal = Normal::standard();
    let mut mid_prices: Vec<f64> = Vec::new();
    let mut bid_q = 100_u64;
    let mut ask_q = 100_u64;
    let mut mid = 100.0_f64;
    for t in 0..200_u64 {
        let shock = normal.sample(&mut rng);
        bid_q = (bid_q as f64 + 5.0 * shock + 0.5 * shock.abs() * 10.0).max(10.0) as u64;
        ask_q = (ask_q as f64 - 3.0 * shock).max(10.0) as u64;
        mid += shock;
        mid_prices.push(mid);
        snapshots.push(BookSnapshot {
            timestamp: t,
            best_bid: Some(Level {
                price: (mid * 100.0) as u64,
                quantity: bid_q,
                order_count: 1,
            }),
            best_ask: Some(Level {
                price: ((mid + 0.01) * 100.0) as u64,
                quantity: ask_q,
                order_count: 1,
            }),
        });
    }
    let ofi = order_flow_imbalance(&snapshots);
    let mid_rets: Vec<f64> = mid_prices.windows(2).map(|w| w[1] - w[0]).collect();
    let n = ofi.len().min(mid_rets.len());
    let mean_o: f64 = ofi[..n].iter().sum::<f64>() / n as f64;
    let mean_r: f64 = mid_rets[..n].iter().sum::<f64>() / n as f64;
    let cov: f64 = ofi[..n]
        .iter()
        .zip(mid_rets[..n].iter())
        .map(|(o, r)| (o - mean_o) * (r - mean_r))
        .sum();
    let var_o: f64 = ofi[..n].iter().map(|o| (o - mean_o).powi(2)).sum();
    let var_r: f64 = mid_rets[..n].iter().map(|r| (r - mean_r).powi(2)).sum();
    let corr = if var_o > 0.0 && var_r > 0.0 {
        cov / (var_o * var_r).sqrt()
    } else {
        0.0
    };
    assert!(corr > 0.0, "OFI correlation should be positive, got {corr}");
}

#[test]
fn test_ex4_sqrt_impact_doubling() {
    let sigma = 0.02;
    let v = 1_000_000.0;
    let i1 = sqrt_impact(sigma, 1000.0, v);
    let i2 = sqrt_impact(sigma, 2000.0, v);
    let ratio = i2 / i1;
    assert!(
        (ratio - 2.0_f64.sqrt()).abs() < 1e-9,
        "ratio {ratio} should be sqrt(2)"
    );
}

#[test]
fn test_ex5_costs_reduce_sharpe() {
    let mu = vec![0.0005, 0.0003];
    let cov = vec![vec![0.0001, 0.0], vec![0.0, 0.00005]];
    let rf = 0.0;
    let tan = tangency_portfolio(&mu, &cov, rf).expect("tan");
    let cost = execution_cost(0.5 * 1_000_000.0, 1_000_000.0, 0.02, 0.0002).total_cost;
    let mu_net: Vec<f64> = mu.iter().map(|&m| m - cost).collect();
    let tan_net = tangency_portfolio(&mu_net, &cov, rf).expect("tan_net");
    assert!(tan.sharpe > tan_net.sharpe, "net Sharpe should be lower");
}
