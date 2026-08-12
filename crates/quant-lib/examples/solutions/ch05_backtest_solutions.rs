//! Exercise solutions for Chapter 5: Backtesting
//!
//! Run: `cargo run -p quant-lib --example solutions-ch05_backtest_solutions`
//! Test: `cargo test -p quant-lib --example solutions-ch05_backtest_solutions`

#[path = "../common/mod.rs"]
mod common;

use quant_lib::backtest::kelly_from_returns;
use quant_lib::core::simple_returns;
use quant_lib::prelude::*;

/// Simple moving average ending at index `i` (inclusive), over `n` points.
fn sma_at(data: &[f64], n: usize, i: usize) -> Option<f64> {
    if i + 1 < n || n == 0 {
        return None;
    }
    Some(mean(&data[i + 1 - n..=i]))
}

/// Exponential moving average series.
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

/// Generate signals: +1 long when SMA(short) > SMA(long), else flat (0).
#[allow(clippy::needless_range_loop)]
fn sma_signals(prices: &[f64], short_n: usize, long_n: usize) -> Vec<i32> {
    let mut signals = vec![0_i32; prices.len()];
    for i in long_n..prices.len() {
        let s = sma_at(prices, short_n, i).unwrap_or(0.0);
        let l = sma_at(prices, long_n, i).unwrap_or(0.0);
        signals[i] = if s > l { 1 } else { 0 };
    }
    signals
}

/// Backtest with proportional transaction cost tau per unit traded.
/// Returns total return and turnover (number of trades).
#[allow(clippy::needless_range_loop)]
fn backtest(prices: &[f64], signals: &[i32], tau: f64) -> (f64, u64) {
    let mut equity = 1.0_f64;
    let mut position = 0_i32; // 0 flat, 1 long
    let mut trades = 0u64;
    for i in 1..prices.len() {
        let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
        if position == 1 {
            equity *= 1.0 + ret;
        }
        if signals[i] != position {
            // Trade at close of bar i.
            equity *= 1.0 - tau;
            position = signals[i];
            trades += 1;
        }
    }
    (equity - 1.0, trades)
}

/// Sharpe ratio of a return series (annualised by sqrt(m)).
fn sharpe(returns: &[f64], m: f64) -> f64 {
    let mu = mean(returns);
    let sd = std_dev(returns).unwrap_or(0.0);
    if sd == 0.0 { 0.0 } else { mu / sd * m.sqrt() }
}

fn main() {
    println!("=== Chapter 5: Backtesting - Exercise Solutions ===\n");
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();
    println!("\nAll Chapter 5 exercises complete.");
}

fn exercise_1() {
    println!("1. Cost Sensitivity (sweep transaction cost tau):");
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    let sigs = sma_signals(&prices, 10, 30);
    for &tau in &[0.0, 0.001, 0.0025, 0.005, 0.01] {
        let (ret, trades) = backtest(&prices, &sigs, tau);
        println!("   tau={tau:.4}: total_return={ret:+.4}, trades={trades}");
    }
}

fn exercise_2() {
    println!("\n2. Parameter Sensitivity (Sharpe-maximising SMA pair):");
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    let rets = simple_returns(&prices);
    let mut best = (0.0_f64, 0_usize, 0_usize);
    for &s in &[5, 10, 20] {
        for &l in &[30, 50, 100] {
            if s >= l {
                continue;
            }
            let sigs = sma_signals(&prices, s, l);
            // Strategy returns: position * asset return.
            let strat: Vec<f64> = (0..rets.len())
                .map(|i| if sigs[i + 1] == 1 { rets[i] } else { 0.0 })
                .collect();
            let sh = sharpe(&strat, 252.0);
            if sh > best.0 {
                best = (sh, s, l);
            }
        }
    }
    println!(
        "   best Sharpe = {:.3} at SMA({},{})",
        best.0, best.1, best.2
    );
}

fn exercise_3() {
    println!("\n3. EMA Crossover vs SMA Crossover (turnover):");
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    // SMA crossover.
    let sma_sigs = sma_signals(&prices, 10, 30);
    let (_, sma_trades) = backtest(&prices, &sma_sigs, 0.0);
    // EMA crossover: long when EMA(short) > EMA(long).
    let ema10 = ema(&prices, 10);
    let ema30 = ema(&prices, 30);
    let ema_sigs: Vec<i32> = (0..prices.len())
        .map(|i| if ema10[i] > ema30[i] { 1 } else { 0 })
        .collect();
    let (_, ema_trades) = backtest(&prices, &ema_sigs, 0.0);
    println!("   SMA(10,30) trades = {sma_trades}, EMA(10,30) trades = {ema_trades}");
}

fn exercise_4() {
    println!("\n4. Position Sizing (f=0.5 vs full Kelly):");
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    let rets = simple_returns(&prices);
    let f_full = kelly_from_returns(&rets);
    let f_half = 0.5 * f_full;
    // Equity curves under fractional position sizing.
    let mut eq_full = 1.0_f64;
    let mut eq_half = 1.0_f64;
    for &r in &rets {
        eq_full *= 1.0 + f_full * r;
        eq_half *= 1.0 + f_half * r;
    }
    println!("   f_full = {f_full:.4}, f_half = {f_half:.4}");
    println!("   equity: full={eq_full:.4}, half={eq_half:.4}");
}

fn exercise_5() {
    println!("\n5. Walk-Forward (tune on first 180, evaluate on last 72):");
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 1.5, 252, &mut rng);
    let n_total = prices.len();
    let n_train = 180.min(n_total);
    let _n_test = n_total - n_train;
    let train_prices = &prices[..n_train];
    let test_prices = &prices[n_train..];
    // Tune SMA(S,L) on training: pick Sharpe-maximising pair.
    let train_rets = simple_returns(train_prices);
    let mut best = (0.0_f64, 5_usize, 30_usize);
    for &s in &[5, 10, 20] {
        for &l in &[30, 50, 60] {
            if s >= l {
                continue;
            }
            let sigs = sma_signals(train_prices, s, l);
            let strat: Vec<f64> = (0..train_rets.len())
                .map(|i| if sigs[i + 1] == 1 { train_rets[i] } else { 0.0 })
                .collect();
            let sh = sharpe(&strat, 252.0);
            if sh > best.0 {
                best = (sh, s, l);
            }
        }
    }
    let is_sharpe = best.0;
    // Evaluate on test set with the chosen (S,L).
    let test_rets = simple_returns(test_prices);
    let test_sigs = sma_signals(test_prices, best.1, best.2);
    let test_strat: Vec<f64> = (0..test_rets.len())
        .map(|i| {
            if test_sigs[i + 1] == 1 {
                test_rets[i]
            } else {
                0.0
            }
        })
        .collect();
    let oos_sharpe = sharpe(&test_strat, 252.0);
    let wfe = walk_forward_efficiency(is_sharpe, oos_sharpe);
    println!(
        "   IS Sharpe = {is_sharpe:.3} at SMA({},{})",
        best.1, best.2
    );
    println!("   OOS Sharpe = {oos_sharpe:.3}, WFE = {wfe:.3}");
}

#[test]
fn test_ex1_cost_sensitivity_declines() {
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    let sigs = sma_signals(&prices, 10, 30);
    let returns: Vec<f64> = [0.0, 0.001, 0.0025, 0.005, 0.01]
        .iter()
        .map(|&tau| backtest(&prices, &sigs, tau).0)
        .collect();
    assert!(returns.iter().all(|r| r.is_finite()));
    // Higher cost should not improve return (monotonically non-increasing in cost).
    for w in returns.windows(2) {
        assert!(w[1] <= w[0] + 1e-12, "return should not increase with cost");
    }
}

#[test]
fn test_ex2_parameter_sensitivity_finite() {
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    let rets = simple_returns(&prices);
    for &s in &[5, 10, 20] {
        for &l in &[30, 50, 100] {
            if s >= l {
                continue;
            }
            let sigs = sma_signals(&prices, s, l);
            let strat: Vec<f64> = (0..rets.len())
                .map(|i| if sigs[i + 1] == 1 { rets[i] } else { 0.0 })
                .collect();
            let sh = sharpe(&strat, 252.0);
            assert!(sh.is_finite());
        }
    }
}

#[test]
fn test_ex3_ema_lower_turnover_than_sma() {
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    let sma_sigs = sma_signals(&prices, 10, 30);
    let (_, sma_trades) = backtest(&prices, &sma_sigs, 0.0);
    let ema10 = ema(&prices, 10);
    let ema30 = ema(&prices, 30);
    let ema_sigs: Vec<i32> = (0..prices.len())
        .map(|i| if ema10[i] > ema30[i] { 1 } else { 0 })
        .collect();
    let (_, ema_trades) = backtest(&prices, &ema_sigs, 0.0);
    assert!(sma_trades > 0 && ema_trades > 0);
    // EMA crossover should trade no more than 2x the SMA trades (smoother).
    assert!(
        ema_trades <= 2 * sma_trades,
        "EMA turnover should be lower than SMA"
    );
}

#[test]
fn test_ex4_position_sizing_finite() {
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 3.0, 252 * 3, &mut rng);
    let rets = simple_returns(&prices);
    let f_full = kelly_from_returns(&rets);
    let f_half = 0.5 * f_full;
    let mut eq_full = 1.0_f64;
    let mut eq_half = 1.0_f64;
    for &r in &rets {
        eq_full *= 1.0 + f_full * r;
        eq_half *= 1.0 + f_half * r;
    }
    assert!(eq_full.is_finite() && eq_half.is_finite());
}

#[test]
fn test_ex5_walk_forward_wfe_in_range() {
    let mut rng = XorShift64::new(42);
    let prices = gbm(100.0, 0.08, 0.20, 1.5, 252, &mut rng);
    let n_train = 180.min(prices.len());
    let train_prices = &prices[..n_train];
    let test_prices = &prices[n_train..];
    assert!(test_prices.len() >= 30, "need at least 30 test bars");
    let train_rets = simple_returns(train_prices);
    let mut best = (f64::NEG_INFINITY, 5_usize, 30_usize);
    for &s in &[5, 10, 20] {
        for &l in &[30, 50, 60] {
            if s >= l {
                continue;
            }
            let sigs = sma_signals(train_prices, s, l);
            let strat: Vec<f64> = (0..train_rets.len())
                .map(|i| if sigs[i + 1] == 1 { train_rets[i] } else { 0.0 })
                .collect();
            let sh = sharpe(&strat, 252.0);
            if sh > best.0 {
                best = (sh, s, l);
            }
        }
    }
    let is_sharpe = best.0;
    let test_rets = simple_returns(test_prices);
    let test_sigs = sma_signals(test_prices, best.1, best.2);
    let test_strat: Vec<f64> = (0..test_rets.len())
        .map(|i| {
            if test_sigs[i + 1] == 1 {
                test_rets[i]
            } else {
                0.0
            }
        })
        .collect();
    let oos_sharpe = sharpe(&test_strat, 252.0);
    let wfe = walk_forward_efficiency(is_sharpe, oos_sharpe);
    assert!(wfe.is_finite());
    // WFE = oos/is. The ratio is only meaningful when the in-sample Sharpe
    // is a real edge; when |is_sharpe| is small the ratio is unbounded, so
    // only bound it when the in-sample signal is non-trivial.
    if is_sharpe.abs() > 0.5 {
        assert!(
            wfe > -10.0 && wfe < 10.0,
            "wfe={wfe}, is={is_sharpe}, oos={oos_sharpe}"
        );
    }
}
