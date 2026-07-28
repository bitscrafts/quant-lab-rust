//! Purged k-fold cross-validation demo.
//!
//! Generates 200 labeled events over 1000 bars, splits them into 5
//! purged folds with an embargo of 10 bars, and prints the per-fold
//! train/test counts, purged count, and embargoed count. Demonstrates
//! how purging + embargo prevent leakage from overlapping events.

use quant_backtest::{
    purged_kfold_splits, sample_weights, triple_barrier_label, LabeledEvent, PurgedKFoldConfig,
    TripleBarrierConfig, TripleBarrierLabel,
};

fn main() {
    let n_bars = 1000usize;
    // Synthetic events: one entry every 5 bars, time barrier of 8 bars,
    // so events overlap heavily.
    let entries: Vec<usize> = (0..n_bars).step_by(5).collect();
    let config = TripleBarrierConfig {
        upper_barrier: 0.02,
        lower_barrier: -0.02,
        time_barrier: 8,
        min_return: 0.0,
    };
    // Use a flat price series so all events hit the time barrier.
    let prices: Vec<f64> = vec![100.0; n_bars];
    let events: Vec<LabeledEvent> = triple_barrier_label(&prices, &entries, &config).unwrap();
    assert!(events.iter().all(|e| e.label == TripleBarrierLabel::Time));

    let weights = sample_weights(&events, n_bars);
    let weight_sum: f64 = weights.iter().sum();
    let weight_min = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let weight_max = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("=== Sample Weights (Uniqueness) ===");
    println!("Events: {}", events.len());
    println!("Weight sum (normalised): {weight_sum:.6}");
    println!("Weight range: [{weight_min:.6}, {weight_max:.6}]");
    println!();

    let cv_config = PurgedKFoldConfig {
        n_folds: 5,
        embargo: 10,
    };
    let splits = purged_kfold_splits(&events, n_bars, &cv_config);

    println!("=== Purged K-Fold (n_folds=5, embargo=10) ===");
    println!(
        "{:<8} {:<14} {:<12} {:<12} {:<12}",
        "Fold", "TestRange", "Train", "Test", "Purged"
    );
    let fold_size = n_bars / cv_config.n_folds;
    for (k, split) in splits.iter().enumerate() {
        let t_start = k * fold_size;
        let t_end = if k == cv_config.n_folds - 1 {
            n_bars
        } else {
            (k + 1) * fold_size
        };
        println!(
            "fold {:<3} [{:>4},{:>4})    {:<12} {:<12} {:<12} embargoed={}",
            k,
            t_start,
            t_end,
            split.train_indices.len(),
            split.test_indices.len(),
            split.purged_count,
            split.embargoed_count
        );
    }

    // Verify no leakage: no training event overlaps any test event.
    for split in &splits {
        for &ti in &split.test_indices {
            let ev_t = &events[ti];
            for &tr in &split.train_indices {
                let ev_tr = &events[tr];
                assert!(
                    ev_tr.exit_index <= ev_t.entry_index
                        || ev_tr.entry_index >= ev_t.exit_index,
                    "leak: train event {tr:?} overlaps test event {ti:?}"
                );
            }
        }
    }
    println!();
    println!("No leakage: verified (no training event overlaps any test event).");
}