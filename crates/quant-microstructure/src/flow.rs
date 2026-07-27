//! Order flow imbalance, VWAP, and trade imbalance.
//!
//! OFI (Cont, Kukanov & Stoikov, 2014) measures the net change in
//! available quantity at the best bid and ask between consecutive
//! snapshots. Positive OFI indicates buy-side pressure (bids growing
//! relative to asks); negative OFI indicates sell-side pressure.

use crate::types::{BookSnapshot, Fill, Trade};

/// Compute the order flow imbalance (OFI) series from a sequence of
/// book snapshots.
///
/// OFI_t = delta(bid_qty) - delta(ask_qty), where delta is the change
/// in available quantity at the best level between snapshots t-1 and t.
/// If either side is missing in a snapshot, the contribution is zero.
pub fn order_flow_imbalance(snapshots: &[BookSnapshot]) -> Vec<f64> {
    if snapshots.len() < 2 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(snapshots.len() - 1);
    for i in 1..snapshots.len() {
        let prev_bid = snapshots[i - 1].best_bid.as_ref().map(|l| l.quantity as f64).unwrap_or(0.0);
        let prev_ask = snapshots[i - 1].best_ask.as_ref().map(|l| l.quantity as f64).unwrap_or(0.0);
        let curr_bid = snapshots[i].best_bid.as_ref().map(|l| l.quantity as f64).unwrap_or(0.0);
        let curr_ask = snapshots[i].best_ask.as_ref().map(|l| l.quantity as f64).unwrap_or(0.0);
        result.push((curr_bid - prev_bid) - (curr_ask - prev_ask));
    }
    result
}

/// Volume-weighted average price from a list of fills.
///
/// Returns 0.0 if the fills list is empty or all quantities are zero.
pub fn vwap(fills: &[Fill]) -> f64 {
    let total_qty: u64 = fills.iter().map(|f| f.quantity).sum();
    if total_qty == 0 {
        return 0.0;
    }
    let total_notional: f64 = fills
        .iter()
        .map(|f| f.price as f64 * f.quantity as f64)
        .sum();
    total_notional / total_qty as f64
}

/// Trade imbalance: (buy_volume - sell_volume) / total_volume.
///
/// A value near +1 means all trades were buy-initiated; near -1 means
/// all sell-initiated; near 0 means balanced flow. Returns 0.0 if
/// there are no trades.
pub fn trade_imbalance(trades: &[Trade]) -> f64 {
    let total: u64 = trades.iter().map(|t| t.quantity).sum();
    if total == 0 {
        return 0.0;
    }
    let buy_vol: u64 = trades
        .iter()
        .filter(|t| t.side == crate::types::Side::Bid)
        .map(|t| t.quantity)
        .sum();
    let sell_vol: u64 = trades
        .iter()
        .filter(|t| t.side == crate::types::Side::Ask)
        .map(|t| t.quantity)
        .sum();
    (buy_vol as f64 - sell_vol as f64) / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Level, Side};

    #[test]
    fn test_ofi_calculation() {
        let snapshots = vec![
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
        ];
        let ofi = order_flow_imbalance(&snapshots);
        assert_eq!(ofi.len(), 1);
        // OFI = (60 - 50) - (25 - 30) = 10 - (-5) = 15
        assert!((ofi[0] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_ofi_empty_and_single() {
        assert!(order_flow_imbalance(&[]).is_empty());
        let single = vec![BookSnapshot {
            timestamp: 1,
            best_bid: None,
            best_ask: None,
        }];
        assert!(order_flow_imbalance(&single).is_empty());
    }

    #[test]
    fn test_vwap_single_fill() {
        let fills = vec![Fill { price: 100, quantity: 10, maker_order_id: 1 }];
        assert!((vwap(&fills) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_vwap_multiple_fills() {
        let fills = vec![
            Fill { price: 100, quantity: 10, maker_order_id: 1 },
            Fill { price: 102, quantity: 20, maker_order_id: 2 },
        ];
        // VWAP = (100*10 + 102*20) / 30 = (1000 + 2040) / 30 = 3040/30 = 101.333...
        let expected = (100.0 * 10.0 + 102.0 * 20.0) / 30.0;
        assert!((vwap(&fills) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_vwap_empty() {
        assert_eq!(vwap(&[]), 0.0);
    }

    #[test]
    fn test_trade_imbalance() {
        let trades = vec![
            Trade { price: 100, quantity: 30, side: Side::Bid, timestamp: 1 },
            Trade { price: 101, quantity: 10, side: Side::Ask, timestamp: 2 },
        ];
        // (30 - 10) / 40 = 0.5
        assert!((trade_imbalance(&trades) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_trade_imbalance_all_buys() {
        let trades = vec![
            Trade { price: 100, quantity: 10, side: Side::Bid, timestamp: 1 },
            Trade { price: 101, quantity: 20, side: Side::Bid, timestamp: 2 },
        ];
        assert!((trade_imbalance(&trades) - 1.0).abs() < 1e-9);
    }
}