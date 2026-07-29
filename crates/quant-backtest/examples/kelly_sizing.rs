//! Kelly criterion bet-sizing demo.
//!
//! Compares full Kelly, half Kelly, and a fixed-fraction baseline on
//! three synthetic return streams (60% win rate with 1:1 payoff, 55%
//! win rate with 2:1 payoff, and 50% win rate with no edge) and prints
//! the recommended position size for each.

use quant_backtest::{
    PositionSize, compute_position_size, fractional_kelly, kelly_fraction, kelly_from_returns,
};

fn summary(name: &str, pos: &PositionSize) {
    println!(
        "{name:<20} full={:+.4} half={:+.4} p={:.3} b={:.3}",
        pos.kelly_full, pos.kelly_half, pos.win_probability, pos.win_loss_ratio
    );
}

fn main() {
    println!("=== Kelly Criterion Bet Sizing ===\n");

    // Case 1: 60% win rate, 1:1 payoff -> f* = 0.2
    let p = 0.6;
    let b = 1.0;
    let f_full = kelly_fraction(p, b);
    let f_half = fractional_kelly(p, b, 0.5);
    println!("Case 1: p={p}, b={b}");
    println!("  full Kelly = {f_full:+.4}");
    println!("  half Kelly = {f_half:+.4}\n");

    // Case 2: 55% win rate, 2:1 payoff -> f* = p - q/b = 0.55 - 0.45/2 = 0.325
    let p = 0.55;
    let b = 2.0;
    let f_full = kelly_fraction(p, b);
    let f_half = fractional_kelly(p, b, 0.5);
    println!("Case 2: p={p}, b={b}");
    println!("  full Kelly = {f_full:+.4}");
    println!("  half Kelly = {f_half:+.4}\n");

    // Case 3: 50% win rate, 1:1 payoff -> f* = 0 (no edge)
    let p = 0.5;
    let b = 1.0;
    let f_full = kelly_fraction(p, b);
    let f_half = fractional_kelly(p, b, 0.5);
    println!("Case 3: p={p}, b={b}");
    println!("  full Kelly = {f_full:+.4}");
    println!("  half Kelly = {f_half:+.4}\n");

    // Now estimate Kelly from three synthetic trade-return streams.
    // Stream A: 60% wins, 1:1 payoff.
    let stream_a: Vec<f64> = (0..100)
        .map(|i| if i % 10 < 6 { 1.0 } else { -1.0 })
        .collect();
    // Stream B: 55% wins, 2:1 payoff.
    let stream_b: Vec<f64> = (0..100)
        .map(|i| if i % 100 < 55 { 2.0 } else { -1.0 })
        .collect();
    // Stream C: 50% wins, 1:1 payoff (no edge).
    let stream_c: Vec<f64> = (0..100)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let pos_a = compute_position_size(&stream_a);
    let pos_b = compute_position_size(&stream_b);
    let pos_c = compute_position_size(&stream_c);

    println!("--- kelly_from_returns (100 trades each) ---");
    summary("Stream A (p=0.6 b=1)", &pos_a);
    summary("Stream B (p=0.55 b=2)", &pos_b);
    summary("Stream C (p=0.5 b=1)", &pos_c);

    // Sanity: stream_a kelly_from_returns should be near 0.2.
    let f_a = kelly_from_returns(&stream_a);
    let f_b = kelly_from_returns(&stream_b);
    let f_c = kelly_from_returns(&stream_c);
    println!();
    println!("Direct kelly_from_returns: A={f_a:+.4}  B={f_b:+.4}  C={f_c:+.4}");
}
