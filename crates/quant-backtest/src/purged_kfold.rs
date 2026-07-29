//! Purged k-fold cross-validation (AFML Ch. 7).
//!
//! Standard k-fold CV leaks information when samples overlap in time:
//! a test fold containing bars `[t1, t2]` may share information with
//! training samples whose events span the test boundary. Purging
//! removes training samples whose `[entry, exit]` interval overlaps
//! the test period. Embargo additionally removes training samples
//! whose entry falls within `[t2, t2 + embargo]` to handle
//! autocorrelation.

use crate::triple_barrier::LabeledEvent;

/// Purged k-fold split configuration.
#[derive(Debug, Clone)]
pub struct PurgedKFoldConfig {
    /// Number of folds.
    pub n_folds: usize,
    /// Embargo period after test set (in bars).
    pub embargo: usize,
}

/// A single train/test split with purging.
#[derive(Debug, Clone)]
pub struct PurgedSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub purged_count: usize,
    pub embargoed_count: usize,
}

/// Check whether an event overlaps with a half-open time range
/// `[start, end)`. An event overlaps if `entry < end` and `exit > start`.
pub fn event_overlaps(event: &LabeledEvent, start: usize, end: usize) -> bool {
    event.entry_index < end && event.exit_index > start
}

/// Generate purged k-fold splits for labeled events.
///
/// The bars `[0, n_bars)` are split into `n_folds` contiguous test
/// periods. For each test fold `[t_start, t_end)`:
///   - Training indices exclude events that overlap the test period.
///   - Training indices also exclude events whose entry falls within
///     `[t_end, t_end + embargo]`.
pub fn purged_kfold_splits(
    events: &[LabeledEvent],
    n_bars: usize,
    config: &PurgedKFoldConfig,
) -> Vec<PurgedSplit> {
    let fold_size = n_bars / config.n_folds;
    let mut splits = Vec::with_capacity(config.n_folds);
    for k in 0..config.n_folds {
        let t_start = k * fold_size;
        let t_end = if k == config.n_folds - 1 {
            n_bars
        } else {
            (k + 1) * fold_size
        };
        let embargo_end = (t_end + config.embargo).min(n_bars);

        let mut train = Vec::new();
        let mut test = Vec::new();
        let mut purged = 0usize;
        let mut embargoed = 0usize;
        for (i, ev) in events.iter().enumerate() {
            // Test set: events whose entry falls in [t_start, t_end).
            if ev.entry_index >= t_start && ev.entry_index < t_end {
                test.push(i);
                continue;
            }
            // Purge: events overlapping the test period.
            if event_overlaps(ev, t_start, t_end) {
                purged += 1;
                continue;
            }
            // Embargo: events whose entry falls in [t_end, embargo_end).
            if ev.entry_index >= t_end && ev.entry_index < embargo_end {
                embargoed += 1;
                continue;
            }
            train.push(i);
        }
        splits.push(PurgedSplit {
            train_indices: train,
            test_indices: test,
            purged_count: purged,
            embargoed_count: embargoed,
        });
    }
    splits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::triple_barrier::{LabeledEvent, TripleBarrierLabel};

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
    fn test_event_overlaps() {
        let e = ev(2, 6);
        assert!(event_overlaps(&e, 0, 5));
        assert!(event_overlaps(&e, 5, 10));
        assert!(!event_overlaps(&e, 6, 10));
        assert!(!event_overlaps(&e, 0, 2));
    }

    #[test]
    fn test_purged_kfold_no_leakage() {
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
                    // No overlap: training event exit <= test entry OR
                    // training event entry >= test exit.
                    assert!(
                        ev_tr.exit_index <= ev_t.entry_index
                            || ev_tr.entry_index >= ev_t.exit_index,
                        "train event {tr:?} overlaps test event {ti:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_purged_kfold_embargo() {
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
        // Folds 0..3 have test [0,25), [25,50), [50,75), [75,100) and
        // embargo periods [25,30), [50,55), [75,80), [100,100) (empty).
        // Events with entry 25, 50, 75 fall in the first three embargo
        // periods, so total embargoed_count across folds = 3.
        let total_embargoed: usize = splits.iter().map(|s| s.embargoed_count).sum();
        assert_eq!(
            total_embargoed, 3,
            "expected 3 embargoed events total, got {total_embargoed}"
        );
        // The last fold has an empty embargo (clamped at n_bars).
        assert_eq!(splits[3].embargoed_count, 0);
    }
}
