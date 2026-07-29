//! Triple-barrier labeling demo.
//!
//! Builds a synthetic random-walk price series, applies the
//! triple-barrier method (AFML Ch. 3) at evenly-spaced entry points,
//! and reports the distribution of labels, average holding period, and
//! per-event returns.

use quant_backtest::{TripleBarrierConfig, TripleBarrierLabel, triple_barrier_label};

/// Deterministic LCG so the demo is reproducible without a `rand` dep.
/// Returns a value in [0, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Use the top 32 bits, scale to [0, 1).
    let x = (*state >> 32) as u32;
    (x as f64) / (u32::MAX as f64 + 1.0)
}

fn main() {
    // Generate a synthetic price series: 250 bars, start at 100, with
    // small daily shocks drawn from the LCG.
    let n_bars = 250usize;
    let mut state: u64 = 0xC0FFEE;
    let mut prices = Vec::with_capacity(n_bars);
    let mut p = 100.0;
    for _ in 0..n_bars {
        let shock = (lcg(&mut state) - 0.5) * 0.02; // [-1%, +1%), zero mean
        p *= 1.0 + shock;
        prices.push(p);
    }

    // Enter a trade every 5 bars.
    let entries: Vec<usize> = (0..n_bars).step_by(5).collect();
    let config = TripleBarrierConfig {
        upper_barrier: 0.02,
        lower_barrier: -0.02,
        time_barrier: 10,
        min_return: 0.0,
    };

    let events = triple_barrier_label(&prices, &entries, &config).unwrap();

    let mut upper = 0usize;
    let mut lower = 0usize;
    let mut time = 0usize;
    let mut sum_hold = 0usize;
    let mut sum_ret = 0.0;
    for ev in &events {
        match ev.label {
            TripleBarrierLabel::Upper => upper += 1,
            TripleBarrierLabel::Lower => lower += 1,
            TripleBarrierLabel::Time => time += 1,
        }
        sum_hold += ev.holding_period;
        sum_ret += ev.return_pct;
    }
    let n = events.len();
    let avg_hold = sum_hold as f64 / n as f64;
    let avg_ret = sum_ret / n as f64;

    println!("=== Triple-Barrier Labeling Demo ===");
    println!("Bars: {n_bars}, entries: {n}");
    println!(
        "  Upper (profit-taking): {upper:>3} ({:.1}%)",
        100.0 * upper as f64 / n as f64
    );
    println!(
        "  Lower (stop-loss):     {lower:>3} ({:.1}%)",
        100.0 * lower as f64 / n as f64
    );
    println!(
        "  Time  (timeout):       {time:>3} ({:.1}%)",
        100.0 * time as f64 / n as f64
    );
    println!("Avg holding period: {avg_hold:.2} bars");
    println!("Avg return per trade: {avg_ret:+.4}");
    println!();
    println!("First 5 events:");
    for (i, ev) in events.iter().take(5).enumerate() {
        println!(
            "  [{i}] entry={} exit={} label={:?} ret={:+.4} hold={}",
            ev.entry_index, ev.exit_index, ev.label, ev.return_pct, ev.holding_period
        );
    }
}
