//! Example 13: Purged K-Fold Cross-Validation with Embargo (AFML Ch.7)
//!
//! Level: Advanced
//!
//! Standard k-fold CV leaks information when samples overlap in time.
//! The purged k-fold method (López de Prado, AFML Ch.7) removes training
//! samples whose `[entry, exit]` interval overlaps the test period, and
//! additionally applies an embargo after each test fold to handle
//! autocorrelation.
//!
//! This example labels PETR4 with the triple barrier, then generates
//! purged k-fold splits and verifies there is no leakage between
//! train and test sets.
//!
//! Uses `quant-backtest` (purged_kfold_splits, PurgedKFoldConfig,
//! event_overlaps, PurgedSplit).
//!
//! Run:
//! ```bash
//! cargo run -p quant-lib --example 13_purged_cv
//! ```

#[path = "common/mod.rs"]
mod common;

use quant_lib::backtest::event_overlaps;
use quant_lib::prelude::*;

fn main() {
    let path = common::b3_json_path(env!("CARGO_MANIFEST_DIR"), "PETR4");
    let bars = common::load_json_ohlcv(&path);
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();

    println!("=== Example 13: Purged K-Fold Cross-Validation ===");
    println!("PETR4: {} bars", bars.len());

    // Label events with the triple barrier.
    let entries: Vec<usize> = (0..closes.len()).step_by(5).collect();
    let config = TripleBarrierConfig {
        upper_barrier: 0.02,
        lower_barrier: -0.02,
        time_barrier: 5,
        min_return: 0.001,
    };
    let events = triple_barrier_label(&closes, &entries, &config).expect("label");
    println!("Labeled {} events (entry every 5 bars)", events.len());

    // 5-fold purged CV with 3-bar embargo.
    let cv_config = PurgedKFoldConfig {
        n_folds: 5,
        embargo: 3,
    };
    let splits = purged_kfold_splits(&events, bars.len(), &cv_config);
    println!(
        "\nPurged {}-fold CV with {}-bar embargo:",
        cv_config.n_folds, cv_config.embargo
    );
    let total_purged: usize = splits.iter().map(|s| s.purged_count).sum();
    let total_embargoed: usize = splits.iter().map(|s| s.embargoed_count).sum();
    println!("  total purged (overlap) events   = {total_purged}");
    println!("  total embargoed events          = {total_embargoed}");

    for (k, split) in splits.iter().enumerate() {
        println!(
            "  fold {k}: train={}, test={}, purged={}, embargoed={}",
            split.train_indices.len(),
            split.test_indices.len(),
            split.purged_count,
            split.embargoed_count
        );
    }

    // Verify no leakage: no training event overlaps any test event.
    for split in &splits {
        for &ti in &split.test_indices {
            let test_ev = &events[ti];
            for &tr in &split.train_indices {
                let train_ev = &events[tr];
                assert!(
                    train_ev.exit_index <= test_ev.entry_index
                        || train_ev.entry_index >= test_ev.exit_index,
                    "leakage: train event {tr:?} overlaps test event {ti:?}"
                );
            }
        }
    }
    println!("\nNo train/test leakage detected across all folds.");

    // Demonstrate the event_overlaps helper.
    let ev = &events[0];
    let overlap_a = event_overlaps(ev, 0, ev.entry_index + 1);
    let overlap_b = event_overlaps(ev, ev.exit_index, ev.exit_index + 100);
    println!(
        "\nevent_overlaps demo (event [{}, {}]):",
        ev.entry_index, ev.exit_index
    );
    println!("  overlaps [0, {}): {overlap_a}", ev.entry_index + 1);
    println!(
        "  overlaps [{}, {}): {overlap_b}",
        ev.exit_index,
        ev.exit_index + 100
    );
}
