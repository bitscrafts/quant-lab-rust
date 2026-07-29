//! Example 12: Triple-Barrier Labeling (AFML Ch.3)
//!
//! Level: Advanced
//!
//! Implements the López de Prado triple-barrier method: each event is
//! labeled by which of three barriers (upper profit-taking, lower
//! stop-loss, time) is hit first. This is the labeling scheme used by
//! the AFML backtesting pipeline.
//!
//! Uses `quant-backtest` (TripleBarrierConfig, triple_barrier_label,
//! LabeledEvent) and the `Labeler` trait via `FixedHorizonLabeler`.
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 12_triple_barrier
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::prelude::*;
// TripleBarrierConfig, TripleBarrierLabel, LabeledEvent, triple_barrier_label,
// FixedHorizonLabeler, and the Labeler trait are all in the prelude.

fn main() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

    println!("=== Example 12: Triple-Barrier Labeling ===");
    println!("PETR4: {} bars from {}", bars.len(), path.display());

    // Enter a long position every 20 bars.
    let entry_indices: Vec<usize> = (0..closes.len()).step_by(20).collect();
    let config = TripleBarrierConfig {
        upper_barrier: 0.03,  // +3% profit-taking
        lower_barrier: -0.02, // -2% stop-loss
        time_barrier: 10,     // 10 bars
        min_return: 0.001,    // 0.1% min to label as positive at time barrier
    };
    println!("\nConfig: upper=+3%, lower=-2%, time=10 bars, min_return=0.1%");
    println!("Entries: {} (every 20 bars)", entry_indices.len());

    let events = triple_barrier_label(&closes, &entry_indices, &config).expect("label");
    println!("Labeled {} events", events.len());

    // Count labels by barrier type.
    let mut n_upper = 0usize;
    let mut n_lower = 0usize;
    let mut n_time = 0usize;
    for ev in &events {
        match ev.label {
            TripleBarrierLabel::Upper => n_upper += 1,
            TripleBarrierLabel::Lower => n_lower += 1,
            TripleBarrierLabel::Time => n_time += 1,
        }
    }
    println!("\nLabel distribution:");
    println!(
        "  Upper (profit-taking): {n_upper} ({:.1}%)",
        n_upper as f64 / events.len() as f64 * 100.0
    );
    println!(
        "  Lower (stop-loss):      {n_lower} ({:.1}%)",
        n_lower as f64 / events.len() as f64 * 100.0
    );
    println!(
        "  Time  (timeout):        {n_time} ({:.1}%)",
        n_time as f64 / events.len() as f64 * 100.0
    );

    // Report the first few events in detail.
    println!("\nFirst 5 events:");
    for (i, ev) in events.iter().take(5).enumerate() {
        let label_i8: i8 = ev.into();
        println!(
            "  [{i}] entry={} exit={} label={:?} ret={:+.4} hold={} ternary={label_i8}",
            ev.entry_index, ev.exit_index, ev.label, ev.return_pct, ev.holding_period
        );
    }

    // Ternary conversion: Upper -> +1, Time -> 0, Lower -> -1.
    let ternary: Vec<i8> = events.iter().map(|e| e.into()).collect();
    let sum: i32 = ternary.iter().map(|&x| x as i32).sum();
    println!("\nTernary sum = {sum} (positive means more profit-takes than stop-outs)");

    // Binary conversion with min_return threshold.
    let binary: Vec<i32> = events
        .iter()
        .map(|e| e.to_binary(config.min_return))
        .collect();
    let n_positive = binary.iter().filter(|&&x| x == 1).count();
    let n_negative = binary.iter().filter(|&&x| x == 0).count();
    println!("Binary: {n_positive} positive, {n_negative} negative");

    // Cross-check via the Labeler trait with FixedHorizonLabeler.
    let labeler = FixedHorizonLabeler::new(10, 0.001);
    let fh_events = labeler
        .label(&closes, &entry_indices)
        .expect("fixed-horizon label");
    assert_eq!(fh_events.len(), events.len());
    println!(
        "\nFixedHorizonLabeler (horizon=10) labeled {} events (same count)",
        fh_events.len()
    );
}
