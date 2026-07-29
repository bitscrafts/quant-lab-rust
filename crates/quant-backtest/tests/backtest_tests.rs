//! Integration tests for quant-backtest (Phase 14).
//!
//! 15-test TDD contract covering: triple-barrier labeling (upper, lower,
//! time, immediate exit, event count), sample uniqueness (concurrent
//! events, weights, uniqueness bounds), purged k-fold CV (no leakage,
//! embargo), Kelly criterion (basic, zero-edge, returns-based, half
//! Kelly), and the end-to-end AFML backtest pipeline.

#![allow(clippy::useless_vec)]

use quant_backtest::{
    AfmlBacktestConfig, BetSizing, LabeledEvent, PurgedKFoldConfig, TripleBarrierConfig,
    TripleBarrierLabel, afml_backtest, average_uniqueness, compute_position_size,
    concurrent_events, fractional_kelly, kelly_fraction, kelly_from_returns, purged_kfold_splits,
    sample_weights, triple_barrier_label,
};

fn cfg(upper: f64, lower: f64, time: usize) -> TripleBarrierConfig {
    TripleBarrierConfig {
        upper_barrier: upper,
        lower_barrier: lower,
        time_barrier: time,
        min_return: 0.0,
    }
}

// =========================================================================
// R14.2: Triple-Barrier Labeling
// =========================================================================

#[test]
fn t01_triple_barrier_upper_hit() {
    let prices = vec![100.0, 101.0, 103.0, 102.5];
    let events = triple_barrier_label(&prices, &[0], &cfg(0.02, -0.02, 5)).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].label, TripleBarrierLabel::Upper);
    assert!((events[0].return_pct - 0.03).abs() < 1e-9);
    assert_eq!(events[0].exit_index, 2);
    assert_eq!(events[0].holding_period, 2);
}

#[test]
fn t02_triple_barrier_lower_hit() {
    let prices = vec![100.0, 99.0, 97.0, 98.0];
    let events = triple_barrier_label(&prices, &[0], &cfg(0.05, -0.02, 5)).unwrap();
    assert_eq!(events[0].label, TripleBarrierLabel::Lower);
    assert!((events[0].return_pct - (-0.03)).abs() < 1e-9);
}

#[test]
fn t03_triple_barrier_time_hit() {
    let prices = vec![100.0, 100.5, 100.2, 100.1, 100.3];
    let events = triple_barrier_label(&prices, &[0], &cfg(0.02, -0.02, 5)).unwrap();
    assert_eq!(events[0].label, TripleBarrierLabel::Time);
    assert_eq!(events[0].holding_period, 4);
}

#[test]
fn t04_triple_barrier_immediate_exit() {
    let prices = vec![100.0, 105.0, 110.0];
    let events = triple_barrier_label(&prices, &[0], &cfg(0.02, -0.02, 5)).unwrap();
    assert_eq!(events[0].exit_index, 1);
    assert_eq!(events[0].holding_period, 1);
}

#[test]
fn t05_labeled_events_count() {
    let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01)).collect();
    let entries: Vec<usize> = (0..100).step_by(10).collect();
    let events = triple_barrier_label(&prices, &entries, &cfg(0.02, -0.02, 5)).unwrap();
    assert_eq!(events.len(), 10);
    for ev in &events {
        assert!(ev.holding_period <= 5);
    }
}

// =========================================================================
// R14.3: Sample Weights (Uniqueness)
// =========================================================================

fn ev(entry: usize, exit: usize) -> LabeledEvent {
    LabeledEvent {
        entry_index: entry,
        exit_index: exit,
        label: TripleBarrierLabel::Upper,
        return_pct: 0.02,
        holding_period: exit - entry,
    }
}

#[test]
fn t06_concurrent_events_overlap() {
    let events = vec![ev(0, 5), ev(3, 8), ev(7, 10)];
    let counts = concurrent_events(&events, 12);
    assert_eq!(counts[0], 1);
    assert_eq!(counts[3], 2);
    assert_eq!(counts[7], 2);
    assert_eq!(counts[11], 0);
    // The maximum concurrency must be > 1 somewhere (events overlap).
    let max_c = *counts.iter().max().unwrap();
    assert!(max_c > 1, "expected overlap, max concurrency = {max_c}");
}

#[test]
fn t07_sample_weights_sum() {
    let events = vec![ev(0, 5), ev(3, 8), ev(7, 10)];
    let w = sample_weights(&events, 12);
    let sum: f64 = w.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "weights must sum to 1, got {sum}");
    for &wi in &w {
        assert!(wi > 0.0 && wi <= 1.0);
    }
}

#[test]
fn t08_average_uniqueness_bounds() {
    let events = vec![ev(0, 5), ev(3, 8), ev(7, 10)];
    let u = average_uniqueness(&events, 12);
    assert_eq!(u.len(), 3);
    for &v in &u {
        assert!(v > 0.0 && v <= 1.0, "uniqueness out of bounds: {v}");
    }
    // An event with no overlap has uniqueness 1.0.
    let single = vec![ev(100, 105)];
    let u2 = average_uniqueness(&single, 200);
    assert!((u2[0] - 1.0).abs() < 1e-9);
}

// =========================================================================
// R14.4: Purged K-Fold Cross-Validation
// =========================================================================

#[test]
fn t09_purged_kfold_no_leakage() {
    let events: Vec<LabeledEvent> = (0..10).map(|i| ev(i * 10, i * 10 + 3)).collect();
    let splits = purged_kfold_splits(
        &events,
        100,
        &PurgedKFoldConfig {
            n_folds: 5,
            embargo: 0,
        },
    );
    assert_eq!(splits.len(), 5);
    for split in &splits {
        for &ti in &split.test_indices {
            let ev_t = &events[ti];
            for &tr in &split.train_indices {
                let ev_tr = &events[tr];
                assert!(
                    ev_tr.exit_index <= ev_t.entry_index || ev_tr.entry_index >= ev_t.exit_index,
                    "train event {tr:?} overlaps test event {ti:?}"
                );
            }
        }
    }
}

#[test]
fn t10_purged_kfold_embargo() {
    // 20 events of width 2 bars, starting every 5 bars across 100 bars.
    let events: Vec<LabeledEvent> = (0..20).map(|i| ev(i * 5, i * 5 + 2)).collect();
    let splits = purged_kfold_splits(
        &events,
        100,
        &PurgedKFoldConfig {
            n_folds: 4,
            embargo: 5,
        },
    );
    // Folds 0..3 have test [0,25), [25,50), [50,75), [75,100) and embargo
    // periods [25,30), [50,55), [75,80), [100,100) (empty). Events with
    // entry 25, 50, 75 fall in the first three embargo periods, so total
    // embargoed_count across folds = 3.
    let total_embargoed: usize = splits.iter().map(|s| s.embargoed_count).sum();
    assert_eq!(
        total_embargoed, 3,
        "expected 3 embargoed events total, got {total_embargoed}"
    );
    // The last fold has an empty embargo (clamped at n_bars).
    assert_eq!(splits[3].embargoed_count, 0);
}

// =========================================================================
// R14.5: Bet Sizing (Kelly Criterion)
// =========================================================================

#[test]
fn t11_kelly_criterion_basic() {
    // p = 0.6, q = 0.4, b = 1.0 -> f* = 0.6 - 0.4/1.0 = 0.2
    let f = kelly_fraction(0.6, 1.0);
    assert!((f - 0.2).abs() < 1e-9, "expected f*=0.2, got {f}");
}

#[test]
fn t12_kelly_zero_edge() {
    // p = 0.5, q = 0.5, b = 1.0 -> f* = 0.5 - 0.5/1.0 = 0.0
    let f = kelly_fraction(0.5, 1.0);
    assert!(f.abs() < 1e-9, "expected f*=0, got {f}");
}

#[test]
fn t13_kelly_from_returns() {
    // 10 trades: 6 wins of +1.0, 4 losses of -1.0 -> p=0.6, b=1.0 -> f*=0.2
    let returns: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
    let f = kelly_from_returns(&returns);
    assert!((f - 0.2).abs() < 1e-6, "expected f*~0.2, got {f}");
}

#[test]
fn t14_position_size_half_kelly() {
    // Half Kelly = fraction * full Kelly. With p=0.6, b=1.0 -> full=0.2, half=0.1.
    let half = fractional_kelly(0.6, 1.0, 0.5);
    assert!(
        (half - 0.1).abs() < 1e-9,
        "expected half-Kelly=0.1, got {half}"
    );
    let returns: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
    let pos = compute_position_size(&returns);
    assert!((pos.kelly_full - 0.2).abs() < 1e-6);
    assert!((pos.kelly_half - 0.1).abs() < 1e-6);
}

// =========================================================================
// R14.6: AFML Backtest Pipeline
// =========================================================================

#[test]
fn t15_afml_backtest_smoke() {
    // 100-bar synthetic price series with a gentle uptrend and weekly noise.
    let prices: Vec<f64> = (0..100)
        .map(|i| 100.0 + (i as f64 * 0.01) + (i as f64 % 7.0 - 3.0) * 0.05)
        .collect();
    let config = AfmlBacktestConfig {
        barrier_config: TripleBarrierConfig {
            upper_barrier: 0.02,
            lower_barrier: -0.02,
            time_barrier: 5,
            min_return: 0.0,
        },
        cv_config: PurgedKFoldConfig {
            n_folds: 3,
            embargo: 2,
        },
        use_weights: true,
        bet_sizing: BetSizing::Equal,
        entry_step: 3,
    };
    let result = afml_backtest(&prices, &config).unwrap();
    assert!(!result.events.is_empty());
    assert_eq!(result.events.len(), result.weights.len());
    assert_eq!(result.cv_splits.len(), 3);
    assert!(result.in_sample_sharpe.is_finite());
    assert!(result.out_of_sample_sharpe.is_finite());
    assert!(result.total_return.is_finite());
    assert!(result.max_drawdown <= 0.0 || result.max_drawdown.is_finite());
}
