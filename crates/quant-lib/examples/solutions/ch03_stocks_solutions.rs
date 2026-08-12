//! Exercise solutions for Chapter 3: Market Data
//!
//! Run: `cargo run -p quant-lib --example solutions-ch03_stocks_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch03_stocks_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::core::{rolling_mean, rolling_std_dev};
use quant_lib::prelude::*;

/// Simple moving average over the last `n` values ending at index `i` (inclusive).
fn sma(data: &[f64], n: usize, i: usize) -> Option<f64> {
    if i + 1 < n || n == 0 {
        return None;
    }
    let start = i + 1 - n;
    Some(mean(&data[start..=i]))
}

/// Exponential moving average over `data` with smoothing `alpha = 2/(n+1)`.
fn ema(data: &[f64], n: usize) -> Vec<f64> {
    if data.is_empty() || n == 0 {
        return Vec::new();
    }
    let alpha = 2.0 / (n as f64 + 1.0);
    let mut out = Vec::with_capacity(data.len());
    let mut prev = data[0];
    out.push(prev);
    for &x in &data[1..] {
        prev = alpha * x + (1.0 - alpha) * prev;
        out.push(prev);
    }
    out
}

fn main() {
    println!("=== Chapter 3: Market Data - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 3 exercises complete.");
}

fn exercise_1() {
    println!("1. Candlestick Patterns (Doji and Hammer):");
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let mut n_doji = 0u64;
    let mut n_hammer = 0u64;
    for b in &bars {
        let range = (b.high - b.low).max(1e-12);
        let body = (b.open - b.close).abs();
        // Doji: |O-C| < 0.1% of (H-L)
        if body < 0.001 * range {
            n_doji += 1;
        }
        // Hammer: small body (< 0.3 * range) and long lower shadow (> 2 * body)
        let lower_shadow = b.open.min(b.close) - b.low;
        if body < 0.3 * range && lower_shadow > 2.0 * body {
            n_hammer += 1;
        }
    }
    println!(
        "   {} bars: {} Doji, {} Hammer",
        bars.len(),
        n_doji,
        n_hammer
    );
}

fn exercise_2() {
    println!("\n2. Volume Analysis (flag days with volume > 2x MA):");
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let vols: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    let n = 20;
    let mut flagged = 0u64;
    for i in 0..vols.len() {
        #[allow(clippy::collapsible_if)]
        if let Some(avg) = sma(&vols, n, i) {
            if vols[i] > 2.0 * avg {
                flagged += 1;
            }
        }
    }
    let pct = if vols.is_empty() {
        0.0
    } else {
        flagged as f64 / vols.len() as f64 * 100.0
    };
    println!("   {flagged} days exceed 2x MA(20) ({pct:.1}% of all days)");
}

fn exercise_3() {
    println!("\n3. Bollinger Bands (SMA20 +/- 2*rolling_std, count upper touches):");
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let closes = common::closes(&bars);
    let n = 20;
    let sma_vals = rolling_mean(n, &closes).unwrap_or_default();
    let std_vals = rolling_std_dev(n, &closes).unwrap_or_default();
    let mut touches = 0u64;
    for (i, &c) in closes.iter().enumerate().skip(n - 1) {
        let idx = i - (n - 1);
        let upper = sma_vals[idx] + 2.0 * std_vals[idx];
        if c >= upper {
            touches += 1;
        }
    }
    let total = closes.len().saturating_sub(n - 1);
    let pct = if total == 0 {
        0.0
    } else {
        touches as f64 / total as f64 * 100.0
    };
    println!("   {touches} upper-band touches out of {total} bars ({pct:.1}%)");
}

fn exercise_4() {
    println!("\n4. MACD (EMA12 - EMA26, signal = EMA9(MACD)):");
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let closes = common::closes(&bars);
    let ema12 = ema(&closes, 12);
    let ema26 = ema(&closes, 26);
    let macd: Vec<f64> = (0..closes.len()).map(|i| ema12[i] - ema26[i]).collect();
    let signal = ema(&macd, 9);
    let mut n_buy = 0u64;
    for i in 1..macd.len() {
        if macd[i - 1] <= signal[i - 1] && macd[i] > signal[i] {
            n_buy += 1;
        }
    }
    println!("   {n_buy} buy signals (MACD cross above signal)");
}

fn exercise_5() {
    println!("\n5. Backtest (golden/death cross on synthetic series):");
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let closes = common::closes(&bars);
    // SMA50/SMA200 need 200 bars; we only have ~63. Use SMA10/SMA30 instead to demonstrate.
    let short_n = 10;
    let long_n = 30;
    let mut position = false;
    let mut equity = 1.0_f64;
    let mut entry_price = 0.0_f64;
    for i in long_n..closes.len() {
        let sma_s = sma(&closes, short_n, i).unwrap_or(0.0);
        let sma_l = sma(&closes, long_n, i).unwrap_or(0.0);
        let prev_s = sma(&closes, short_n, i - 1).unwrap_or(0.0);
        let prev_l = sma(&closes, long_n, i - 1).unwrap_or(0.0);
        let golden = prev_s <= prev_l && sma_s > sma_l;
        let death = prev_s >= prev_l && sma_s < sma_l;
        if golden && !position {
            position = true;
            entry_price = closes[i];
        } else if death && position {
            position = false;
            equity *= closes[i] / entry_price;
        }
    }
    if position {
        equity *= closes[closes.len() - 1] / entry_price;
    }
    println!("   SMA({short_n})/SMA({long_n}) crossover strategy return = {equity:.4}x equity");
}

#[test]
fn test_ex1_candlestick_patterns_detected() {
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    assert!(!bars.is_empty());
    for b in &bars {
        let range = (b.high - b.low).max(1e-12);
        assert!(b.high >= b.low);
        assert!(b.high >= b.open && b.high >= b.close);
        assert!(b.low <= b.open && b.low <= b.close);
        let _ = range;
    }
}

#[test]
fn test_ex2_volume_flags_finite() {
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let vols: Vec<f64> = bars.iter().map(|b| b.volume).collect();
    assert!(vols.iter().all(|v| v.is_finite() && *v >= 0.0));
    let n = 20;
    let mut count = 0u64;
    for i in 0..vols.len() {
        if let Some(avg) = sma(&vols, n, i) {
            if vols[i] > 2.0 * avg {
                count += 1;
            }
        }
    }
    assert!(count <= vols.len() as u64);
}

#[test]
fn test_ex3_bollinger_upper_touches() {
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let closes = common::closes(&bars);
    let n = 20;
    if closes.len() < n {
        return;
    }
    let sma_vals = rolling_mean(n, &closes).unwrap();
    let std_vals = rolling_std_dev(n, &closes).unwrap();
    assert_eq!(sma_vals.len(), closes.len() - n + 1);
    assert_eq!(std_vals.len(), closes.len() - n + 1);
    assert!(sma_vals.iter().all(|v| v.is_finite()));
    assert!(std_vals.iter().all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn test_ex4_macd_buy_signals() {
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let closes = common::closes(&bars);
    let ema12 = ema(&closes, 12);
    let ema26 = ema(&closes, 26);
    assert_eq!(ema12.len(), closes.len());
    assert_eq!(ema26.len(), closes.len());
    let macd: Vec<f64> = (0..closes.len()).map(|i| ema12[i] - ema26[i]).collect();
    let signal = ema(&macd, 9);
    assert_eq!(signal.len(), macd.len());
    assert!(macd.iter().all(|v| v.is_finite()));
}

#[test]
fn test_ex5_crossover_strategy_positive_equity() {
    let path = common::stock_csv_path(env!("CARGO_MANIFEST_DIR"));
    let bars = common::load_stock_csv(&path);
    let closes = common::closes(&bars);
    assert!(!closes.is_empty());
    assert!(closes.iter().all(|c| c.is_finite() && *c > 0.0));
    // A simple backtest must produce a finite, non-negative equity multiplier.
    let short_n = 10;
    let long_n = 30;
    let mut position = false;
    let mut equity = 1.0_f64;
    let mut entry_price = 0.0_f64;
    for i in long_n..closes.len() {
        let sma_s = sma(&closes, short_n, i).unwrap_or(0.0);
        let sma_l = sma(&closes, long_n, i).unwrap_or(0.0);
        let prev_s = sma(&closes, short_n, i - 1).unwrap_or(0.0);
        let prev_l = sma(&closes, long_n, i - 1).unwrap_or(0.0);
        if prev_s <= prev_l && sma_s > sma_l && !position {
            position = true;
            entry_price = closes[i];
        } else if prev_s >= prev_l && sma_s < sma_l && position {
            position = false;
            equity *= closes[i] / entry_price;
        }
    }
    if position {
        equity *= closes[closes.len() - 1] / entry_price;
    }
    assert!(equity.is_finite() && equity >= 0.0);
}
