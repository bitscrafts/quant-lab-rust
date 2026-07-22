//! TDD contract for `qf-04-returns` (Phase 4).
//!
//! Tests were written against the HANDOFF contract and hand-verified.
//!
//! Note on `test_volatility_basic`: the HANDOFF table lists `std_dev ≈ 0.0158`,
//! which corresponds to the *population* standard deviation. The spec text
//! (R4.3) and `qf_common::compute_stats` both use the *sample* estimator
//! (denominator `n - 1`), which gives `0.01718` for the same input. We use the
//! sample estimator for consistency and assert the mathematically correct
//! value.

use approx::assert_relative_eq;
use qf_03_stocks::Ohlcv;
use qf_04_returns::{
    annualized_sharpe, annualized_volatility, cumulative_returns, drawdown, drawdown_stats,
    log_returns, max_drawdown, rolling_volatility, sharpe_ratio, simple_returns, sortino_ratio,
    volatility, Returns,
};

// ---------------------------------------------------------------------------
// R4.2: Returns calculation
// ---------------------------------------------------------------------------

#[test]
fn test_simple_returns_basic() {
    let prices = [100.0_f64, 110.0, 99.0];
    let r = simple_returns(&prices);
    assert_eq!(r.len(), 2);
    assert_relative_eq!(r[0], 0.10, epsilon = 1e-9);
    assert_relative_eq!(r[1], -0.10, epsilon = 1e-9);
}

#[test]
fn test_simple_returns_empty() {
    let r = simple_returns(&[]);
    assert!(r.is_empty());
}

#[test]
fn test_simple_returns_single() {
    let r = simple_returns(&[100.0_f64]);
    assert!(r.is_empty());
}

#[test]
fn test_log_returns_basic() {
    let r = log_returns(&[100.0_f64, 110.0]);
    assert_eq!(r.len(), 1);
    assert_relative_eq!(r[0], (1.1_f64).ln(), epsilon = 1e-9);
}

#[test]
fn test_log_returns_matches_formula() {
    let prices = [100.0_f64, 105.0, 110.0];
    let r = log_returns(&prices);
    assert_eq!(r.len(), 2);
    assert_relative_eq!(r[0], (105.0 / 100.0_f64).ln(), epsilon = 1e-9);
    assert_relative_eq!(r[1], (110.0 / 105.0_f64).ln(), epsilon = 1e-9);
}

#[test]
fn test_log_simple_relationship() {
    // log return == ln(1 + simple return) exactly.
    let prices = [100.0_f64, 101.0, 102.0, 103.0];
    let sr = simple_returns(&prices);
    let lr = log_returns(&prices);
    for (s, l) in sr.iter().zip(lr.iter()) {
        assert_relative_eq!(*l, (1.0 + *s).ln(), epsilon = 1e-9);
    }
}

#[test]
fn test_cumulative_returns() {
    // Compounding: (1+0.10)(1-0.05)(1+0.08) - 1 = 0.1286
    let returns = [0.10_f64, -0.05, 0.08];
    let cum = cumulative_returns(&returns);
    assert_eq!(cum.len(), 3);
    assert_relative_eq!(cum[0], 0.10, epsilon = 1e-9);
    assert_relative_eq!(cum[1], 1.10 * 0.95 - 1.0, epsilon = 1e-9);
    assert_relative_eq!(cum[2], 1.10 * 0.95 * 1.08 - 1.0, epsilon = 1e-4);
    // ending compounded value
    assert_relative_eq!(cum[2], 0.1286, epsilon = 1e-4);
}

// ---------------------------------------------------------------------------
// R4.3: Volatility
// ---------------------------------------------------------------------------

#[test]
fn test_volatility_basic() {
    let returns = [0.01_f64, -0.02, 0.015, -0.01, 0.02];
    let vol = volatility(&returns);
    // Sample std (n-1 denominator): 0.017176...
    assert_relative_eq!(vol, 0.01718, epsilon = 1e-3);
}

#[test]
fn test_volatility_constant() {
    let vol = volatility(&[0.01_f64, 0.01, 0.01]);
    assert_relative_eq!(vol, 0.0, epsilon = 1e-12);
}

#[test]
fn test_volatility_empty() {
    assert_relative_eq!(volatility(&[]), 0.0, epsilon = 1e-12);
    assert_relative_eq!(volatility(&[0.01_f64]), 0.0, epsilon = 1e-12);
}

#[test]
fn test_annualized_volatility() {
    // Construct returns whose sample std is exactly 0.01:
    // [a, -a] with a = 0.01/sqrt(2) has sample std = 0.01.
    let a = 0.01_f64 / 2.0_f64.sqrt();
    let returns = [a, -a];
    let ann = annualized_volatility(&returns, 252.0);
    assert_relative_eq!(ann, 0.159, epsilon = 1e-3);
}

#[test]
fn test_rolling_volatility_basic() {
    let returns = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let rv = rolling_volatility(&returns, 3);
    assert_eq!(rv.len(), 3);
}

#[test]
fn test_rolling_volatility_invalid() {
    let returns = [1.0_f64, 2.0, 3.0];
    assert!(rolling_volatility(&returns, 0).is_empty());
    assert!(rolling_volatility(&returns, 4).is_empty());
    assert!(rolling_volatility(&[], 2).is_empty());
}

// ---------------------------------------------------------------------------
// R4.4: Risk-adjusted metrics
// ---------------------------------------------------------------------------

#[test]
fn test_sharpe_ratio_positive() {
    let returns = [0.01_f64, 0.02, 0.03];
    let s = sharpe_ratio(&returns, 0.0);
    assert!(s > 0.0);
}

#[test]
fn test_sharpe_ratio_negative() {
    let returns = [0.01_f64, 0.02, 0.03];
    let s = sharpe_ratio(&returns, 0.1);
    assert!(s < 0.0);
}

#[test]
fn test_sharpe_ratio_zero_vol() {
    let s = sharpe_ratio(&[0.01_f64, 0.01, 0.01], 0.0);
    assert_relative_eq!(s, 0.0, epsilon = 1e-12);
}

#[test]
fn test_annualized_sharpe() {
    // Construct returns with per-period Sharpe == 0.05.
    // returns = [0.01, 0.01, 0.01, 0.02]; mean = 0.0125, sample vol = 0.005.
    // (mean - rf)/vol = 0.05  =>  rf = 0.0125 - 0.05 * 0.005 = 0.01225.
    let returns = [0.01_f64, 0.01, 0.01, 0.02];
    let rf = 0.01225_f64;
    let ann = annualized_sharpe(&returns, rf, 252.0);
    assert_relative_eq!(ann, 0.79, epsilon = 1e-2);
}

#[test]
fn test_sortino_basic() {
    let returns = [0.02_f64, -0.01, 0.03, -0.02, 0.01];
    let sharpe = sharpe_ratio(&returns, 0.0);
    let sortino = sortino_ratio(&returns, 0.0);
    // Downside dev < total vol => Sortino >= Sharpe.
    assert!(sortino > sharpe);
}

// ---------------------------------------------------------------------------
// R4.5: Drawdown
// ---------------------------------------------------------------------------

#[test]
fn test_drawdown_basic() {
    let prices = [100.0_f64, 110.0, 105.0, 115.0, 100.0];
    let dd = drawdown(&prices);
    assert_eq!(dd.len(), 5);
    assert_relative_eq!(dd[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(dd[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(dd[2], (105.0 - 110.0) / 110.0, epsilon = 1e-9);
    assert_relative_eq!(dd[3], 0.0, epsilon = 1e-12);
    assert_relative_eq!(dd[4], (100.0 - 115.0) / 115.0, epsilon = 1e-9);
}

#[test]
fn test_drawdown_always_negative() {
    let prices = [100.0_f64, 110.0, 90.0, 95.0, 80.0, 120.0];
    let dd = drawdown(&prices);
    assert!(dd.iter().all(|&v| v <= 1e-12));
}

#[test]
fn test_max_drawdown_basic() {
    let md = max_drawdown(&[100.0_f64, 110.0, 90.0, 95.0]);
    assert_relative_eq!(md, -0.182, epsilon = 1e-3);
}

#[test]
fn test_max_drawdown_no_decline() {
    let md = max_drawdown(&[100.0_f64, 110.0, 120.0]);
    assert_relative_eq!(md, 0.0, epsilon = 1e-12);
}

#[test]
fn test_drawdown_stats_recovers() {
    // [100, 110, 90, 95, 110] -> worst dd at index 2 (-0.1818...), recovers at
    // index 4 (new high 110). recovery_time = 4 - 2 = 2.
    let stats = drawdown_stats(&[100.0_f64, 110.0, 90.0, 95.0, 110.0]);
    assert_relative_eq!(stats.max_drawdown, -0.182, epsilon = 1e-3);
    assert!(stats.max_drawdown_duration > 0);
    assert_eq!(stats.recovery_time, Some(2));
}

// ---------------------------------------------------------------------------
// R4.6: Returns trait
// ---------------------------------------------------------------------------

#[test]
fn test_returns_trait_slice() {
    let prices: &[f64] = &[100.0, 105.0, 110.0];
    let sr = prices.simple_returns();
    let lr = prices.log_returns();
    assert_eq!(sr.len(), 2);
    assert_eq!(lr.len(), 2);
    assert_relative_eq!(sr[0], 0.05, epsilon = 1e-9);
    assert_relative_eq!(lr[0], (1.05_f64).ln(), epsilon = 1e-9);
}

#[test]
fn test_returns_trait_ohlcv() {
    let data = vec![
        Ohlcv {
            date: "2024-01-01".to_string(),
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 100.0,
            volume: 1000,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-02".to_string(),
            open: 100.0,
            high: 106.0,
            low: 100.0,
            close: 105.0,
            volume: 1100,
            adj_close: None,
        },
        Ohlcv {
            date: "2024-01-03".to_string(),
            open: 105.0,
            high: 111.0,
            low: 104.0,
            close: 110.0,
            volume: 1200,
            adj_close: None,
        },
    ];
    let sr = data.simple_returns();
    let lr = data.log_returns();
    assert_eq!(sr.len(), 2);
    assert_relative_eq!(sr[0], (105.0 - 100.0) / 100.0, epsilon = 1e-9);
    assert_relative_eq!(sr[1], (110.0 - 105.0) / 105.0, epsilon = 1e-9);
    assert_relative_eq!(lr[0], (105.0 / 100.0_f64).ln(), epsilon = 1e-9);
}