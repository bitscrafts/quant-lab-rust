//! Example: Backtesting an SMA crossover against buy-and-hold.
//!
//! Demonstrates the full Phase 5 pipeline: build synthetic OHLCV data, run a
//! 10/30 SMA-crossover strategy, run a buy-and-hold benchmark, and compare
//! performance using the `qf-04-returns` metrics exposed on `BacktestResult`.
//!
//! Run from the `quant-lab` directory:
//!   cargo run -p qf-05-backtest --example backtest_demo

use qf_03_stocks::Ohlcv;
use qf_05_backtest::{run_backtest, BacktestConfig, BuyAndHold, SmaCrossover, Strategy};

/// Build a deterministic 252-bar synthetic price series with a slow upward
/// drift, a mid-year drawdown, and a recovery — enough regime change for the
/// SMA crossover to fire several signals.
fn synthetic_series() -> Vec<Ohlcv> {
    let mut out = Vec::with_capacity(252);
    let mut price = 100.0_f64;
    for t in 0..252 {
        // Slow upward drift.
        let drift = 0.0004;
        // A mid-year drawdown (bars 120..160) and a recovery.
        let regime = match t {
            120..=140 => -0.0035, // bear phase
            141..=160 => 0.0010,  // slow recovery
            _ => 0.0,
        };
        // Tiny deterministic wiggle so candles are not all equal.
        let wiggle = 0.002 * ((t as f64 * 0.7).sin());
        let r = drift + regime + wiggle;
        price = (price * (1.0 + r)).max(1.0);
        let high = price * 1.005;
        let low = price * 0.995;
        let open = price * (1.0 - 0.001 * ((t as f64 * 0.3).sin()));
        out.push(Ohlcv {
            date: format!("2024-{:03}", t + 1),
            open,
            high,
            low,
            close: price,
            volume: 1_000_000,
            adj_close: None,
        });
    }
    out
}

fn print_result(label: &str, result: &qf_05_backtest::BacktestResult) {
    println!("--- {} ---", label);
    println!("  Total Return:   {:.2}%", result.total_return * 100.0);
    println!("  Final Capital:   ${:.2}", result.final_capital);
    println!("  Trades:          {}", result.num_trades);
    println!("  Win Rate:        {:.1}%", result.win_rate * 100.0);
    println!("  Total Costs:     ${:.2}", result.total_costs);
    println!("  Sharpe (ann.):   {:.2}", result.sharpe(0.0));
    println!("  Sortino:         {:.2}", result.sortino(0.0));
    println!("  Max Drawdown:    {:.2}%", result.max_drawdown() * 100.0);
    println!("  Volatility (ann.): {:.2}%", result.volatility() * 100.0);
    println!();
}

fn main() {
    println!("==============================");
    println!("Backtest Results");
    println!("==============================\n");

    let data = synthetic_series();
    let period = format!("{} to {}", data.first().unwrap().date, data.last().unwrap().date);
    println!("Period: {}\n", period);

    let config = BacktestConfig::default();

    let strategy = SmaCrossover::new(10, 30).unwrap();
    let strat_result = run_backtest(&strategy, &data, &config).unwrap();
    let bh_result = run_backtest(&BuyAndHold, &data, &config).unwrap();

    print_result(&format!("Strategy: {} (10/30)", strategy.name()), &strat_result);
    print_result("Strategy: Buy and Hold", &bh_result);

    let outperformance = strat_result.total_return - bh_result.total_return;
    println!("Outperformance vs Buy & Hold: {:+.2}%", outperformance * 100.0);
    println!("==============================");
}